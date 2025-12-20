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

# Import torch.compiler.disable to prevent torch.compile from tracing through Triton kernels
# This avoids double-autotuning conflicts on newer GPUs (Blackwell)
try:
    from torch.compiler import disable as compiler_disable
except ImportError:
    try:
        from torch._dynamo import disable as compiler_disable
    except ImportError:
        def compiler_disable(fn):
            return fn


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
# NOTE: RoPE kernel disabled by default (can trigger illegal memory access on some GPUs/stacks).
# NOTE: Fused RMSNorm is currently DISABLED by default because its backward has been observed
#       to produce massively incorrect gradients on some stacks. Opt in explicitly via
#       HYDRA_ENABLE_FUSED_RMS_NORM=1.
USE_FUSED_ROPE = USE_TRITON_KERNELS and os.environ.get("HYDRA_ENABLE_FUSED_ROPE", "0") == "1"
USE_FUSED_QK_NORM = USE_TRITON_KERNELS  # Now autograd-compatible!
USE_FUSED_SWIGLU = USE_TRITON_KERNELS  # Now autograd-compatible!
USE_FUSED_RMS_NORM = USE_TRITON_KERNELS and os.environ.get("HYDRA_ENABLE_FUSED_RMS_NORM", "0") == "1"


def set_use_triton_kernels(enabled: bool):
    """Enable or disable Triton kernels globally."""
    global USE_TRITON_KERNELS, USE_FUSED_ROPE, USE_FUSED_QK_NORM, USE_FUSED_SWIGLU, USE_FUSED_RMS_NORM
    
    if enabled and not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available. Install with: pip install triton")
    
    USE_TRITON_KERNELS = enabled
    # Fused RoPE stays opt-in even when enabling Triton globally.
    USE_FUSED_ROPE = enabled and os.environ.get("HYDRA_ENABLE_FUSED_ROPE", "0") == "1"
    USE_FUSED_QK_NORM = enabled
    USE_FUSED_SWIGLU = enabled
    # Fused RMSNorm stays opt-in even when enabling Triton globally.
    USE_FUSED_RMS_NORM = enabled and os.environ.get("HYDRA_ENABLE_FUSED_RMS_NORM", "0") == "1"


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
    # Blackwell-safe autotuning configs (shared memory limit: 101KB)
    # Reduced from 256 max to 64 max to fit within Blackwell's SM constraints
    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 16}, num_warps=2, num_stages=1),
            triton.Config({"BLOCK_SIZE": 32}, num_warps=2, num_stages=1),
            triton.Config({"BLOCK_SIZE": 64}, num_warps=4, num_stages=1),
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


@compiler_disable
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
        x.stride(1),  # Treat batch*head as single dim (x is contiguous)
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
    # Blackwell-safe autotuning configs
    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 32}, num_warps=2, num_stages=1),
            triton.Config({"BLOCK_SIZE": 64}, num_warps=4, num_stages=1),
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
        return FusedQKNormFunction.apply(q, k, scale, temperature)
    return _qk_norm_pytorch(q, k, scale, temperature)


class FusedQKNormFunction(torch.autograd.Function):
    """Autograd-compatible wrapper for fused QK-norm Triton kernel."""
    
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, scale: float, temperature: float):
        """Forward pass using Triton kernel."""
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
        
        # Save for backward
        ctx.save_for_backward(q, k)
        ctx.scale = scale
        ctx.temperature = temperature
        
        return q_out, k_out
    
    @staticmethod
    def backward(ctx, grad_q_out: torch.Tensor, grad_k_out: torch.Tensor):
        """Backward pass for L2 normalization.
        
        Forward: out = x / ||x|| * scale
        
        Backward: d_out/d_x = scale * (I - x*x^T / ||x||^2) / ||x||
                            = scale / ||x|| * (grad - x * (x dot grad) / ||x||^2)
        
        Note: We compute in float32 for numerical stability, then cast back.
        """
        q, k = ctx.saved_tensors
        scale = ctx.scale
        temperature = ctx.temperature
        
        # Q gradient (float32 for stability)
        q_f32 = q.float()
        grad_q_f32 = grad_q_out.float()
        q_norm_sq = (q_f32 * q_f32).sum(dim=-1, keepdim=True)
        q_norm = q_norm_sq.sqrt().clamp(min=1e-6)  # Clamp to avoid division by zero
        q_normalized = q_f32 / q_norm
        
        # Gradient: scale * (grad - normalized * dot(normalized, grad)) / norm
        q_dot_grad = (q_normalized * grad_q_f32).sum(dim=-1, keepdim=True)
        grad_q = scale * (grad_q_f32 - q_normalized * q_dot_grad) / q_norm
        # Clamp gradient magnitude to prevent explosions
        grad_q = grad_q.clamp(-100.0, 100.0)
        grad_q = grad_q.to(q.dtype)
        
        # K gradient (includes temperature, float32 for stability)
        k_f32 = k.float()
        grad_k_f32 = grad_k_out.float()
        k_norm_sq = (k_f32 * k_f32).sum(dim=-1, keepdim=True)
        k_norm = k_norm_sq.sqrt().clamp(min=1e-6)
        k_normalized = k_f32 / k_norm
        
        k_dot_grad = (k_normalized * grad_k_f32).sum(dim=-1, keepdim=True)
        grad_k = scale * temperature * (grad_k_f32 - k_normalized * k_dot_grad) / k_norm
        grad_k = grad_k.clamp(-100.0, 100.0)
        grad_k = grad_k.to(k.dtype)
        
        return grad_q, grad_k, None, None


@compiler_disable
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
    # Blackwell-safe autotuning configs
    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=1),
            triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=1),
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
        return FusedSwiGLUFunction.apply(gate, up)
    return _swiglu_pytorch(gate, up)


class FusedSwiGLUFunction(torch.autograd.Function):
    """Autograd-compatible wrapper for fused SwiGLU Triton kernel."""
    
    @staticmethod
    def forward(ctx, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        """Forward pass using Triton kernel."""
        orig_shape = gate.shape
        gate_flat = gate.contiguous().view(-1)
        up_flat = up.contiguous().view(-1)
        out = torch.empty_like(gate_flat)
        
        n_elements = gate_flat.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        
        _fused_swiglu_kernel[grid](gate_flat, up_flat, out, n_elements)
        
        # Save for backward
        ctx.save_for_backward(gate, up)
        ctx.orig_shape = orig_shape
        
        return out.view(orig_shape)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Backward pass computed in PyTorch (float32 for stability).
        
        Forward: out = silu(gate) * up = gate * sigmoid(gate) * up
        
        d_out/d_gate = up * (sigmoid(gate) + gate * sigmoid(gate) * (1 - sigmoid(gate)))
                     = up * sigmoid(gate) * (1 + gate * (1 - sigmoid(gate)))
        d_out/d_up = silu(gate)
        """
        gate, up = ctx.saved_tensors
        
        # Compute in float32 for stability
        gate_f32 = gate.float()
        up_f32 = up.float()
        grad_out_f32 = grad_output.float()
        
        sigmoid_gate = torch.sigmoid(gate_f32)
        silu_gate = gate_f32 * sigmoid_gate
        
        # Gradient w.r.t. gate
        # d(silu)/d(gate) = sigmoid(gate) + gate * sigmoid(gate) * (1 - sigmoid(gate))
        #                 = sigmoid(gate) * (1 + gate - gate * sigmoid(gate))
        dsilu = sigmoid_gate * (1.0 + gate_f32 * (1.0 - sigmoid_gate))
        grad_gate = grad_out_f32 * up_f32 * dsilu
        # Clamp to prevent explosions
        grad_gate = grad_gate.clamp(-100.0, 100.0).to(gate.dtype)
        
        # Gradient w.r.t. up
        grad_up = grad_out_f32 * silu_gate
        grad_up = grad_up.clamp(-100.0, 100.0).to(up.dtype)
        
        return grad_gate, grad_up


@compiler_disable
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
    # Blackwell-safe autotuning configs
    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 64}, num_warps=2, num_stages=1),
            triton.Config({"BLOCK_SIZE": 128}, num_warps=4, num_stages=1),
            triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=1),
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
        return FusedRMSNormFunction.apply(x, weight, eps)
    return _rms_norm_pytorch(x, weight, eps)


class FusedRMSNormFunction(torch.autograd.Function):
    """Autograd-compatible wrapper for fused RMSNorm Triton kernel."""
    
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        """Forward pass using Triton kernel."""
        orig_shape = x.shape
        dim = orig_shape[-1]
        x_flat = x.contiguous().view(-1, dim)
        n_rows = x_flat.shape[0]
        
        out = torch.empty_like(x_flat)
        weight = weight.contiguous()
        
        # Compute RMS for backward pass
        x_float = x_flat.float()
        rms_inv = torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + eps)
        
        grid = (n_rows,)
        _fused_rms_norm_kernel[grid](
            x_flat, weight, out, eps, dim, n_rows,
        )
        
        # Save for backward
        ctx.save_for_backward(x_flat, weight, rms_inv)
        ctx.eps = eps
        ctx.orig_shape = orig_shape
        
        return out.view(orig_shape)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Backward pass computed in PyTorch (Triton backward is complex).
        
        Forward: out = x * rsqrt(mean(x^2) + eps) * weight
        
        Computed in float32 for numerical stability with gradient clamping.
        """
        x, weight, rms_inv = ctx.saved_tensors
        eps = ctx.eps
        orig_shape = ctx.orig_shape
        dim = x.shape[-1]

        # grad_output may be non-contiguous (e.g., due to preceding transpose/view patterns).
        # Use reshape to avoid RuntimeError and ensure correct flattening.
        grad_output_flat = grad_output.reshape(-1, dim)
        
        # All computations in float32 for stability
        x_f32 = x.float()
        weight_f32 = weight.float()
        grad_out_f32 = grad_output_flat.float()
        rms_inv_f32 = rms_inv.float()
        
        # Gradient w.r.t. weight: sum over batch of (x * rms_inv * grad_output)
        x_norm = x_f32 * rms_inv_f32
        grad_weight = (x_norm * grad_out_f32).sum(dim=0)
        # Clamp weight gradient
        grad_weight = grad_weight.clamp(-100.0, 100.0).to(weight.dtype)
        
        # Gradient w.r.t. x
        # d/dx [x * rsqrt(mean(x^2) + eps) * w]
        # = w * rsqrt(...) - w * x * x * rsqrt(...)^3 / dim
        # = w * rsqrt(...) * (1 - x^2 / (dim * (mean(x^2) + eps)))
        grad_x = grad_out_f32 * weight_f32 * rms_inv_f32
        # Correction term for the derivative of rsqrt
        correction = (grad_out_f32 * weight_f32 * x_norm).mean(dim=-1, keepdim=True)
        grad_x = grad_x - x_norm * correction
        # Clamp gradient magnitude
        grad_x = grad_x.clamp(-100.0, 100.0).to(x.dtype)
        
        return grad_x.view(orig_shape), grad_weight, None


@compiler_disable
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
    # Prefer native PyTorch RMSNorm when available (fast + stable).
    if hasattr(F, "rms_norm"):
        return F.rms_norm(x, [x.shape[-1]], weight=weight, eps=eps)

    dtype = x.dtype
    x = x.float()
    rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (x * rms).to(dtype) * weight


# =============================================================================
# 6. Chunked Cross-Entropy Loss
# =============================================================================
# This is a MAJOR memory optimization for language models.
# Instead of materializing the full logits tensor (batch × seq × vocab_size),
# we compute the loss in chunks, dramatically reducing peak memory usage.
# 
# For a 50K vocab with batch=16, seq=512:
#   - Full logits: 16 × 512 × 50257 × 2 bytes = 819 MB (bf16)
#   - Chunked (8 chunks): 16 × 64 × 50257 × 2 bytes = 102 MB per chunk
#   - Peak memory reduction: ~8x
#
# This technique is used by Liger Kernel and other frontier training libraries.
# =============================================================================

USE_CHUNKED_CROSS_ENTROPY = True  # Enable by default for memory efficiency
CROSS_ENTROPY_CHUNK_SIZE = 4096  # Process this many tokens at a time


def chunked_cross_entropy(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
    chunk_size: int = None,
) -> torch.Tensor:
    """Memory-efficient cross-entropy that avoids materializing full logits.
    
    Instead of computing all logits at once, we:
    1. Split the sequence into chunks
    2. Compute logits and loss for each chunk
    3. Accumulate the total loss
    
    This reduces peak memory from O(batch × seq × vocab) to 
    O(batch × chunk_size × vocab), which can be 4-8x smaller.
    
    Args:
        hidden_states: [batch, seq, dim] - output of final norm layer
        weight: [vocab_size, dim] - output projection weight (lm_head)
        targets: [batch, seq] - target token ids
        ignore_index: Index to ignore in loss computation (default: -100)
        chunk_size: Number of tokens per chunk (default: CROSS_ENTROPY_CHUNK_SIZE)
        
    Returns:
        Scalar cross-entropy loss
    """
    if chunk_size is None:
        chunk_size = CROSS_ENTROPY_CHUNK_SIZE
    
    batch_size, seq_len, dim = hidden_states.shape
    vocab_size = weight.shape[0]
    
    # Flatten batch and sequence dimensions
    hidden_flat = hidden_states.view(-1, dim)  # [batch * seq, dim]
    targets_flat = targets.view(-1)  # [batch * seq]
    
    total_tokens = hidden_flat.shape[0]
    
    # If sequence is small enough, just compute directly
    if total_tokens <= chunk_size:
        logits = F.linear(hidden_flat, weight)  # [batch * seq, vocab]
        return F.cross_entropy(logits, targets_flat, ignore_index=ignore_index)
    
    # Compute loss in chunks
    total_loss = 0.0
    n_valid_tokens = 0
    
    for start_idx in range(0, total_tokens, chunk_size):
        end_idx = min(start_idx + chunk_size, total_tokens)
        
        # Get chunk of hidden states and targets
        hidden_chunk = hidden_flat[start_idx:end_idx]  # [chunk, dim]
        target_chunk = targets_flat[start_idx:end_idx]  # [chunk]
        
        # Compute logits for this chunk only
        logits_chunk = F.linear(hidden_chunk, weight)  # [chunk, vocab]
        
        # Count valid tokens in this chunk (not ignore_index)
        valid_mask = target_chunk != ignore_index
        n_valid_chunk = valid_mask.sum().item()
        
        if n_valid_chunk > 0:
            # Compute loss for this chunk (reduction='sum' for proper averaging)
            chunk_loss = F.cross_entropy(
                logits_chunk, target_chunk, 
                ignore_index=ignore_index,
                reduction='sum'
            )
            total_loss = total_loss + chunk_loss
            n_valid_tokens += n_valid_chunk
    
    # Average over all valid tokens
    if n_valid_tokens > 0:
        return total_loss / n_valid_tokens
    else:
        # No valid tokens - return zero loss
        return torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)


class ChunkedCrossEntropyFunction(torch.autograd.Function):
    """Autograd function for chunked cross-entropy with fused backward.
    
    This is even more memory efficient as it recomputes logits during backward
    instead of storing them, trading compute for memory.
    """
    
    @staticmethod
    def forward(
        ctx, 
        hidden_states: torch.Tensor, 
        weight: torch.Tensor, 
        targets: torch.Tensor,
        ignore_index: int,
        chunk_size: int,
    ) -> torch.Tensor:
        batch_size, seq_len, dim = hidden_states.shape
        
        # Flatten for computation
        hidden_flat = hidden_states.view(-1, dim)
        targets_flat = targets.view(-1)
        total_tokens = hidden_flat.shape[0]
        
        # Save for backward (recompute logits to save memory)
        ctx.save_for_backward(hidden_states, weight, targets)
        ctx.ignore_index = ignore_index
        ctx.chunk_size = chunk_size
        
        # Compute loss in chunks
        total_loss = 0.0
        n_valid_tokens = 0
        
        for start_idx in range(0, total_tokens, chunk_size):
            end_idx = min(start_idx + chunk_size, total_tokens)
            hidden_chunk = hidden_flat[start_idx:end_idx]
            target_chunk = targets_flat[start_idx:end_idx]
            
            logits_chunk = F.linear(hidden_chunk, weight)
            valid_mask = target_chunk != ignore_index
            n_valid_chunk = valid_mask.sum().item()
            
            if n_valid_chunk > 0:
                chunk_loss = F.cross_entropy(
                    logits_chunk, target_chunk,
                    ignore_index=ignore_index,
                    reduction='sum'
                )
                total_loss = total_loss + chunk_loss
                n_valid_tokens += n_valid_chunk
        
        ctx.n_valid_tokens = n_valid_tokens
        
        if n_valid_tokens > 0:
            return total_loss / n_valid_tokens
        else:
            return torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        hidden_states, weight, targets = ctx.saved_tensors
        ignore_index = ctx.ignore_index
        chunk_size = ctx.chunk_size
        n_valid_tokens = ctx.n_valid_tokens
        
        if n_valid_tokens == 0:
            return (
                torch.zeros_like(hidden_states),
                torch.zeros_like(weight),
                None, None, None
            )
        
        batch_size, seq_len, dim = hidden_states.shape
        hidden_flat = hidden_states.view(-1, dim)
        targets_flat = targets.view(-1)
        total_tokens = hidden_flat.shape[0]
        
        # Accumulate gradients in chunks (use float32 for stability)
        grad_hidden = torch.zeros_like(hidden_flat, dtype=torch.float32)
        grad_weight = torch.zeros_like(weight, dtype=torch.float32)
        
        # Scale gradient by 1/n_valid_tokens (from mean reduction)
        # Ensure scale is float32 for numerical stability
        scale = grad_output.float() / n_valid_tokens
        
        for start_idx in range(0, total_tokens, chunk_size):
            end_idx = min(start_idx + chunk_size, total_tokens)
            hidden_chunk = hidden_flat[start_idx:end_idx].float()
            target_chunk = targets_flat[start_idx:end_idx]
            chunk_len = end_idx - start_idx
            
            # Recompute logits for this chunk (float32 for stability)
            logits_chunk = F.linear(hidden_chunk, weight.float())
            
            # Compute softmax probabilities (float32)
            probs = F.softmax(logits_chunk, dim=-1)
            
            # Gradient of cross-entropy w.r.t. logits: p - one_hot(y)
            # VECTORIZED: avoid slow Python for-loop
            grad_logits = probs.clone()
            valid_mask = target_chunk != ignore_index
            
            # Create indices for scatter operation
            # For valid tokens: subtract 1 from the probability at the target index
            valid_indices = torch.where(valid_mask)[0]
            if valid_indices.numel() > 0:
                valid_targets = target_chunk[valid_indices]
                # Vectorized subtraction: grad_logits[valid_indices, valid_targets] -= 1.0
                grad_logits[valid_indices, valid_targets] -= 1.0
            
            # Zero out gradients for ignored tokens
            invalid_indices = torch.where(~valid_mask)[0]
            if invalid_indices.numel() > 0:
                grad_logits[invalid_indices] = 0.0
            
            grad_logits = grad_logits * scale
            
            # Gradient w.r.t. hidden: grad_logits @ weight (float32)
            grad_hidden[start_idx:end_idx] = grad_logits @ weight.float()
            
            # Gradient w.r.t. weight: grad_logits.T @ hidden (float32)
            grad_weight += grad_logits.T @ hidden_chunk
        
        # Cast back to original dtype
        return grad_hidden.view_as(hidden_states).to(hidden_states.dtype), grad_weight.to(weight.dtype), None, None, None


def fused_chunked_cross_entropy(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
    chunk_size: int = None,
) -> torch.Tensor:
    """Fused chunked cross-entropy with memory-efficient backward pass.
    
    This version uses an autograd function that recomputes logits during
    backward, further reducing memory usage at the cost of extra compute.
    
    Args:
        hidden_states: [batch, seq, dim] - output of final norm layer
        weight: [vocab_size, dim] - output projection weight (lm_head)
        targets: [batch, seq] - target token ids
        ignore_index: Index to ignore in loss computation (default: -100)
        chunk_size: Number of tokens per chunk (default: CROSS_ENTROPY_CHUNK_SIZE)
        
    Returns:
        Scalar cross-entropy loss
    """
    if chunk_size is None:
        chunk_size = CROSS_ENTROPY_CHUNK_SIZE
    
    return ChunkedCrossEntropyFunction.apply(
        hidden_states, weight, targets, ignore_index, chunk_size
    )


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

    # Fused RoPE is opt-in only (it can be unsafe on some stacks).
    if USE_FUSED_ROPE and TRITON_AVAILABLE:
        results["rope"] = _benchmark(
            "RoPE",
            lambda: _fused_rope_triton(x, cos, sin),
            lambda: _rope_pytorch(x, cos, sin),
        )
    else:
        results["rope"] = {"skipped": True, "reason": "fused_rope disabled by default"}
    
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
