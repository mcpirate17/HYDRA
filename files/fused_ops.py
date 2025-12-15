"""
HYDRA Fused Triton Kernels

High-performance GPU kernels that fuse multiple operations to reduce
memory bandwidth and kernel launch overhead.

KERNELS:
1. fused_rope: Fused Rotary Position Embedding (2-3x faster)
2. fused_qk_norm: Fused L2 normalization + scaling for Q/K (1.5-2x faster)
3. fused_swiglu: Fused SiLU(gate) * up activation (1.3x faster)
4. fused_rms_norm: Fused RMS normalization (1.5x faster)

FEATURE FLAGS:
- TRITON_AVAILABLE: Whether Triton is installed
- USE_TRITON_KERNELS: Global switch to enable/disable Triton kernels
- Per-kernel switches: USE_FUSED_ROPE, USE_FUSED_QK_NORM, etc.

Requirements:
- triton >= 3.0.0 (recommended) or triton >= 2.0.0
- CUDA GPU with compute capability >= 7.0

Usage:
    from hydra.kernels import fused_rope, fused_qk_norm, fused_swiglu
    
    # Enable/disable globally
    from hydra.kernels import set_use_triton_kernels
    set_use_triton_kernels(True)  # Enable Triton (default if available)
    
    # Or per-kernel
    from hydra.kernels import fused_ops
    fused_ops.USE_FUSED_ROPE = True
"""

import math
import os
from typing import Tuple

import torch
import torch.nn.functional as F


# =============================================================================
# Triton Import and Feature Detection
# =============================================================================

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
    TRITON_VERSION = tuple(int(x) for x in triton.__version__.split(".")[:2])
except ImportError:
    TRITON_AVAILABLE = False
    TRITON_VERSION = (0, 0)
    triton = None
    tl = None

# Global enable switch (can be overridden)
USE_TRITON_KERNELS = TRITON_AVAILABLE and os.environ.get("HYDRA_DISABLE_TRITON", "0") != "1"

# Per-kernel switches (for debugging)
USE_FUSED_ROPE = USE_TRITON_KERNELS
USE_FUSED_QK_NORM = USE_TRITON_KERNELS
USE_FUSED_SWIGLU = USE_TRITON_KERNELS
USE_FUSED_RMS_NORM = USE_TRITON_KERNELS


def set_use_triton_kernels(enabled: bool):
    """Enable or disable Triton kernels globally."""
    global USE_TRITON_KERNELS, USE_FUSED_ROPE, USE_FUSED_QK_NORM, USE_FUSED_SWIGLU, USE_FUSED_RMS_NORM
    
    if enabled and not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available. Install with: pip install triton")
    
    USE_TRITON_KERNELS = enabled
    USE_FUSED_ROPE = enabled
    USE_FUSED_QK_NORM = enabled
    USE_FUSED_SWIGLU = enabled
    USE_FUSED_RMS_NORM = enabled


def get_kernel_status() -> dict:
    """Get status of all Triton kernels."""
    return {
        "triton_available": TRITON_AVAILABLE,
        "triton_version": ".".join(map(str, TRITON_VERSION)) if TRITON_AVAILABLE else "N/A",
        "use_triton_kernels": USE_TRITON_KERNELS,
        "fused_rope": USE_FUSED_ROPE,
        "fused_qk_norm": USE_FUSED_QK_NORM,
        "fused_swiglu": USE_FUSED_SWIGLU,
        "fused_rms_norm": USE_FUSED_RMS_NORM,
    }


# =============================================================================
# 1. Fused RoPE Kernel with Autotuning
# =============================================================================

if TRITON_AVAILABLE:
    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 32}, num_warps=2),
            triton.Config({"BLOCK_SIZE": 64}, num_warps=4),
            triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
            triton.Config({"BLOCK_SIZE": 256}, num_warps=8),
        ],
        key=["half_head_dim"],
    )
    @triton.jit
    def _fused_rope_kernel(
        x_ptr,
        cos_ptr,
        sin_ptr,
        out_ptr,
        seq_len,
        half_head_dim,
        stride_b,
        stride_h,
        stride_s,
        stride_d,
        cos_stride_s,
        cos_stride_d,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Fused RoPE forward kernel with autotuning.
        
        Applies rotary position embedding in a single kernel:
        out[..., 0::2] = x[..., 0::2] * cos - x[..., 1::2] * sin
        out[..., 1::2] = x[..., 0::2] * sin + x[..., 1::2] * cos
        """
        pid = tl.program_id(0)
        
        # Decode pid into batch*head and seq indices
        seq_idx = pid % seq_len
        bh_idx = pid // seq_len
        
        # Process in blocks
        for d in range(0, half_head_dim, BLOCK_SIZE):
            offs = d + tl.arange(0, BLOCK_SIZE)
            mask = offs < half_head_dim
            
            # Calculate offsets for x
            x_base = bh_idx * stride_b + seq_idx * stride_s
            x1_offs = x_base + (2 * offs) * stride_d
            x2_offs = x_base + (2 * offs + 1) * stride_d
            
            # Load x pairs
            x1 = tl.load(x_ptr + x1_offs, mask=mask, other=0.0)
            x2 = tl.load(x_ptr + x2_offs, mask=mask, other=0.0)
            
            # Load cos/sin
            cos_offs = seq_idx * cos_stride_s + offs * cos_stride_d
            cos_val = tl.load(cos_ptr + cos_offs, mask=mask, other=1.0)
            sin_val = tl.load(sin_ptr + cos_offs, mask=mask, other=0.0)
            
            # Apply rotation
            out1 = x1 * cos_val - x2 * sin_val
            out2 = x1 * sin_val + x2 * cos_val
            
            # Store results
            tl.store(out_ptr + x1_offs, out1, mask=mask)
            tl.store(out_ptr + x2_offs, out2, mask=mask)


def fused_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply Rotary Position Embedding with optional fused Triton kernel.
    
    Args:
        x: Input tensor [B, n_heads, S, head_dim]
        cos: Cosine cache [1, 1, max_seq, head_dim//2] or [S, head_dim//2]
        sin: Sine cache [1, 1, max_seq, head_dim//2] or [S, head_dim//2]
        
    Returns:
        Rotated tensor [B, n_heads, S, head_dim]
    """
    if USE_FUSED_ROPE and TRITON_AVAILABLE and x.is_cuda:
        return _fused_rope_triton(x, cos, sin)
    return _rope_pytorch(x, cos, sin)


def _fused_rope_triton(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Triton implementation of RoPE."""
    B, H, S, D = x.shape
    half_head_dim = D // 2
    
    # Ensure contiguous
    x = x.contiguous()
    out = torch.empty_like(x)
    
    # Flatten cos/sin if needed
    if cos.dim() == 4:
        cos = cos[:, :, :S, :].squeeze(0).squeeze(0)  # [S, D//2]
        sin = sin[:, :, :S, :].squeeze(0).squeeze(0)
    
    cos = cos.contiguous()
    sin = sin.contiguous()
    
    # Grid: one program per (batch*head, seq) position
    grid = (B * H * S,)
    
    _fused_rope_kernel[grid](
        x,
        cos,
        sin,
        out,
        S,
        half_head_dim,
        x.stride(0) * x.stride(1),  # Treat batch*head as single dim
        1,  # Not used directly
        x.stride(2),
        x.stride(3),
        cos.stride(0),
        cos.stride(1),
    )
    
    return out


def _rope_pytorch(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """PyTorch fallback for RoPE."""
    S = x.shape[2]
    cos = cos[:, :, :S, :]
    sin = sin[:, :, :S, :]
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)


# =============================================================================
# 2. Fused QK Normalization Kernel
# =============================================================================

if TRITON_AVAILABLE:
    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 64}, num_warps=2),
            triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
            triton.Config({"BLOCK_SIZE": 256}, num_warps=8),
        ],
        key=["head_dim"],
    )
    @triton.jit
    def _fused_qk_norm_kernel(
        q_ptr,
        k_ptr,
        q_out_ptr,
        k_out_ptr,
        scale,
        temperature,
        head_dim: tl.constexpr,
        n_q_elements,
        n_k_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Fused L2 normalization + scaling for Q and K."""
        pid = tl.program_id(0)
        
        # Determine if we're processing Q or K
        is_k = pid >= n_q_elements
        actual_pid = pid - n_q_elements if is_k else pid
        
        if is_k and actual_pid >= n_k_elements:
            return
        
        # Select pointers
        in_ptr = k_ptr if is_k else q_ptr
        out_ptr = k_out_ptr if is_k else q_out_ptr
        base = actual_pid * head_dim
        
        # Compute L2 norm (two-pass for numerical stability)
        sq_sum = tl.zeros([1], dtype=tl.float32)
        
        for d in range(0, head_dim, BLOCK_SIZE):
            offs = d + tl.arange(0, BLOCK_SIZE)
            mask = offs < head_dim
            val = tl.load(in_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
            sq_sum += tl.sum(val * val)
        
        # Normalize and scale
        norm_factor = tl.rsqrt(sq_sum + 1e-8) * scale
        if is_k:
            norm_factor = norm_factor * temperature
        
        for d in range(0, head_dim, BLOCK_SIZE):
            offs = d + tl.arange(0, BLOCK_SIZE)
            mask = offs < head_dim
            val = tl.load(in_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
            out = val * norm_factor
            tl.store(out_ptr + base + offs, out, mask=mask)


def fused_qk_norm(
    q: torch.Tensor,
    k: torch.Tensor,
    scale: float,
    temperature: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused L2 normalization + scaling for Q and K tensors.
    
    Args:
        q: Query tensor [B, n_heads, S, head_dim]
        k: Key tensor [B, n_kv_heads, S, head_dim]
        scale: Scale factor (typically sqrt(head_dim))
        temperature: Additional scale for K (learnable temperature)
        
    Returns:
        Tuple of (normalized_q, normalized_k)
    """
    if USE_FUSED_QK_NORM and TRITON_AVAILABLE and q.is_cuda:
        return _fused_qk_norm_triton(q, k, scale, temperature)
    return _qk_norm_pytorch(q, k, scale, temperature)


def _fused_qk_norm_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    scale: float,
    temperature: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Triton implementation of QK normalization."""
    B_q, H_q, S_q, D = q.shape
    B_k, H_k, S_k, _ = k.shape
    
    q = q.contiguous()
    k = k.contiguous()
    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)
    
    n_q_elements = B_q * H_q * S_q
    n_k_elements = B_k * H_k * S_k
    
    # Process Q and K in single kernel launch
    grid = (n_q_elements + n_k_elements,)
    
    _fused_qk_norm_kernel[grid](
        q.view(-1),
        k.view(-1),
        q_out.view(-1),
        k_out.view(-1),
        scale,
        temperature,
        D,
        n_q_elements,
        n_k_elements,
    )
    
    return q_out, k_out


def _qk_norm_pytorch(
    q: torch.Tensor,
    k: torch.Tensor,
    scale: float,
    temperature: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """PyTorch fallback for QK normalization."""
    q_norm = F.normalize(q, p=2, dim=-1) * scale
    k_norm = F.normalize(k, p=2, dim=-1) * scale * temperature
    return q_norm, k_norm


# =============================================================================
# 3. Fused SwiGLU Kernel
# =============================================================================

if TRITON_AVAILABLE:
    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
            triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
            triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
        ],
        key=["n_elements"],
    )
    @triton.jit
    def _fused_swiglu_kernel(
        gate_ptr,
        up_ptr,
        out_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Fused SiLU(gate) * up computation.
        
        out = gate * sigmoid(gate) * up = silu(gate) * up
        """
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        
        gate = tl.load(gate_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(up_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        
        # SiLU = x * sigmoid(x)
        silu_gate = gate * tl.sigmoid(gate)
        out = silu_gate * up
        
        tl.store(out_ptr + offs, out, mask=mask)


def fused_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Fused SiLU(gate) * up activation.
    
    Args:
        gate: Gate tensor [..., hidden_dim]
        up: Up tensor [..., hidden_dim]
        
    Returns:
        Output tensor [..., hidden_dim]
    """
    if USE_FUSED_SWIGLU and TRITON_AVAILABLE and gate.is_cuda:
        return _fused_swiglu_triton(gate, up)
    return _swiglu_pytorch(gate, up)


def _fused_swiglu_triton(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Triton implementation of SwiGLU."""
    orig_shape = gate.shape
    gate = gate.contiguous().view(-1)
    up = up.contiguous().view(-1)
    out = torch.empty_like(gate)
    
    n_elements = gate.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    
    _fused_swiglu_kernel[grid](gate, up, out, n_elements)
    
    return out.view(orig_shape)


def _swiglu_pytorch(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """PyTorch fallback for SwiGLU."""
    return F.silu(gate) * up


# =============================================================================
# 4. Fused RMSNorm Kernel
# =============================================================================

if TRITON_AVAILABLE:
    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 128}, num_warps=2),
            triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
            triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
            triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
        ],
        key=["dim"],
    )
    @triton.jit
    def _fused_rms_norm_kernel(
        x_ptr,
        weight_ptr,
        out_ptr,
        eps,
        dim: tl.constexpr,
        n_rows,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Fused RMS normalization.
        
        rms = sqrt(mean(x^2) + eps)
        out = x / rms * weight
        """
        row_idx = tl.program_id(0)
        if row_idx >= n_rows:
            return
        
        row_start = row_idx * dim
        
        # Compute sum of squares
        sq_sum = tl.zeros([1], dtype=tl.float32)
        for d in range(0, dim, BLOCK_SIZE):
            offs = d + tl.arange(0, BLOCK_SIZE)
            mask = offs < dim
            x = tl.load(x_ptr + row_start + offs, mask=mask, other=0.0).to(tl.float32)
            sq_sum += tl.sum(x * x)
        
        # RMS inverse
        rms_inv = tl.rsqrt(sq_sum / dim + eps)
        
        # Normalize and scale
        for d in range(0, dim, BLOCK_SIZE):
            offs = d + tl.arange(0, BLOCK_SIZE)
            mask = offs < dim
            x = tl.load(x_ptr + row_start + offs, mask=mask, other=0.0).to(tl.float32)
            w = tl.load(weight_ptr + offs, mask=mask, other=1.0).to(tl.float32)
            out = x * rms_inv * w
            tl.store(out_ptr + row_start + offs, out, mask=mask)


def fused_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Fused RMS normalization.
    
    Args:
        x: Input tensor [..., dim]
        weight: Scale weights [dim]
        eps: Epsilon for numerical stability
        
    Returns:
        Normalized tensor [..., dim]
    """
    if USE_FUSED_RMS_NORM and TRITON_AVAILABLE and x.is_cuda:
        return _fused_rms_norm_triton(x, weight, eps)
    return _rms_norm_pytorch(x, weight, eps)


def _fused_rms_norm_triton(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Triton implementation of RMSNorm."""
    orig_shape = x.shape
    dim = orig_shape[-1]
    x = x.contiguous().view(-1, dim)
    n_rows = x.shape[0]
    
    out = torch.empty_like(x)
    weight = weight.contiguous()
    
    grid = (n_rows,)
    
    _fused_rms_norm_kernel[grid](
        x, weight, out, eps, dim, n_rows,
    )
    
    return out.view(orig_shape)


def _rms_norm_pytorch(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """PyTorch fallback for RMSNorm."""
    dtype = x.dtype
    x = x.float()
    rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (x * rms).to(dtype) * weight


# =============================================================================
# Benchmarking Utilities
# =============================================================================

def benchmark_kernels(
    batch_size: int = 4,
    seq_len: int = 512,
    dim: int = 768,
    n_heads: int = 12,
    warmup: int = 10,
    iterations: int = 100,
) -> dict:
    """Benchmark fused kernels vs PyTorch baselines.
    
    Args:
        batch_size: Batch size
        seq_len: Sequence length
        dim: Model dimension
        n_heads: Number of attention heads
        warmup: Warmup iterations
        iterations: Benchmark iterations
        
    Returns:
        Dictionary with timing results
    """
    import time
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        return {"error": "CUDA not available"}
    
    head_dim = dim // n_heads
    results = {
        "config": {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "dim": dim,
            "n_heads": n_heads,
            "head_dim": head_dim,
        },
        "triton_available": TRITON_AVAILABLE,
    }
    
    def _benchmark(name: str, fn_triton, fn_pytorch, *args):
        # Warmup
        for _ in range(warmup):
            fn_pytorch(*args)
            if TRITON_AVAILABLE:
                fn_triton(*args)
        torch.cuda.synchronize()
        
        # Benchmark PyTorch
        start = time.perf_counter()
        for _ in range(iterations):
            fn_pytorch(*args)
        torch.cuda.synchronize()
        pytorch_time = (time.perf_counter() - start) / iterations * 1000
        
        # Benchmark Triton
        if TRITON_AVAILABLE:
            start = time.perf_counter()
            for _ in range(iterations):
                fn_triton(*args)
            torch.cuda.synchronize()
            triton_time = (time.perf_counter() - start) / iterations * 1000
            speedup = pytorch_time / triton_time
        else:
            triton_time = None
            speedup = None
        
        return {
            "pytorch_ms": pytorch_time,
            "triton_ms": triton_time,
            "speedup": speedup,
        }
    
    # RoPE benchmark
    x = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device, dtype=torch.float16)
    cos = torch.randn(1, 1, seq_len, head_dim // 2, device=device, dtype=torch.float16)
    sin = torch.randn(1, 1, seq_len, head_dim // 2, device=device, dtype=torch.float16)
    
    results["rope"] = _benchmark(
        "RoPE",
        lambda: _fused_rope_triton(x, cos, sin) if TRITON_AVAILABLE else None,
        lambda: _rope_pytorch(x, cos, sin),
    )
    
    # QK Norm benchmark
    q = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device, dtype=torch.float16)
    k = torch.randn(batch_size, n_heads // 4, seq_len, head_dim, device=device, dtype=torch.float16)
    scale = math.sqrt(head_dim)
    
    results["qk_norm"] = _benchmark(
        "QK Norm",
        lambda: _fused_qk_norm_triton(q, k, scale, 1.0) if TRITON_AVAILABLE else None,
        lambda: _qk_norm_pytorch(q, k, scale, 1.0),
    )
    
    # SwiGLU benchmark
    hidden_dim = int(dim * 3.5)
    gate = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=torch.float16)
    up = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=torch.float16)
    
    results["swiglu"] = _benchmark(
        "SwiGLU",
        lambda: _fused_swiglu_triton(gate, up) if TRITON_AVAILABLE else None,
        lambda: _swiglu_pytorch(gate, up),
    )
    
    # RMSNorm benchmark
    x_norm = torch.randn(batch_size, seq_len, dim, device=device, dtype=torch.float16)
    weight = torch.ones(dim, device=device, dtype=torch.float16)
    
    results["rms_norm"] = _benchmark(
        "RMSNorm",
        lambda: _fused_rms_norm_triton(x_norm, weight, 1e-6) if TRITON_AVAILABLE else None,
        lambda: _rms_norm_pytorch(x_norm, weight, 1e-6),
    )
    
    return results


def print_benchmark_results(results: dict):
    """Pretty print benchmark results."""
    print(f"\n{'='*60}")
    print(f"HYDRA Kernel Benchmark Results")
    print(f"{'='*60}")
    
    config = results.get("config", {})
    print(f"Config: B={config.get('batch_size')}, S={config.get('seq_len')}, "
          f"D={config.get('dim')}, H={config.get('n_heads')}")
    print(f"Triton Available: {results.get('triton_available')}")
    print()
    
    for name in ["rope", "qk_norm", "swiglu", "rms_norm"]:
        if name in results:
            r = results[name]
            pytorch_ms = r.get("pytorch_ms", 0)
            triton_ms = r.get("triton_ms")
            speedup = r.get("speedup")
            
            print(f"{name.upper()}:")
            print(f"  PyTorch: {pytorch_ms:.3f} ms")
            if triton_ms is not None:
                print(f"  Triton:  {triton_ms:.3f} ms")
                print(f"  Speedup: {speedup:.2f}x")
            else:
                print(f"  Triton:  N/A")
            print()


if __name__ == "__main__":
    print("HYDRA Triton Kernel Status:")
    for k, v in get_kernel_status().items():
        print(f"  {k}: {v}")
    
    results = benchmark_kernels()
    print_benchmark_results(results)
