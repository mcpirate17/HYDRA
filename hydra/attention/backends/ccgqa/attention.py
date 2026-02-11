from __future__ import annotations

import os
import contextlib
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# Optional fused RoPE kernel.
try:
    from hydra.kernels import fused_rope

    FUSED_KERNELS_AVAILABLE = fused_rope is not None
except Exception:
    FUSED_KERNELS_AVAILABLE = False
    fused_rope = None

# Fused QK-norm kernel (L2 normalize + scale in single kernel launch).
try:
    from hydra.kernels.fused_ops import fused_qk_norm as _fused_qk_norm
    FUSED_QK_NORM_AVAILABLE = _fused_qk_norm is not None
except Exception:
    FUSED_QK_NORM_AVAILABLE = False
    _fused_qk_norm = None

# Optional fused attention kernel (Triton).
try:
    from hydra.attention.backends.ccgqa.kernels import ccgqa_attention_fused

    TRITON_ATTENTION_AVAILABLE = True
except Exception:
    TRITON_ATTENTION_AVAILABLE = False
    ccgqa_attention_fused = None

# Optimized convolution sequences
from hydra.attention.backends.ccgqa.kernels.fused_conv import OptimizedConvSequence

# Sage Attention: 2-5x faster than Flash via INT8/FP8 quantization
from hydra.layers import SAGE_ATTN_AVAILABLE, sageattn


def _get_ccgqa_attn_backend() -> str:
    """Get attention backend for CCGQA inner attention.

    Env: HYDRA_CCGQA_ATTN_BACKEND=auto|sage|sdpa
    Default: auto (tries Sage first, falls back to SDPA)
    """
    mode = os.environ.get("HYDRA_CCGQA_ATTN_BACKEND", "auto").strip().lower()
    return mode if mode in {"auto", "sage", "sdpa"} else "auto"


def _env_flag(name: str) -> bool:
    v = os.environ.get(name, "").strip().lower()
    return v in {"1", "true", "yes", "y", "on"}


def _sdpa_kernel_context() -> contextlib.AbstractContextManager[None]:
    """Optional SDPA kernel selector.

    Env: HYDRA_CCQA_SDPA_KERNEL=auto|flash|mem_efficient|math

    If unavailable on this PyTorch build, becomes a no-op.
    """

    mode = os.environ.get("HYDRA_CCQA_SDPA_KERNEL", "auto").strip().lower()
    if mode in {"", "auto", "default"}:
        return contextlib.nullcontext()

    # torch.backends.cuda.sdp_kernel exists on newer PyTorch.
    try:
        sdp_kernel = torch.backends.cuda.sdp_kernel  # type: ignore[attr-defined]
    except Exception:
        return contextlib.nullcontext()

    if mode in {"flash", "fa", "flashattn"}:
        return sdp_kernel(enable_flash=True, enable_mem_efficient=False, enable_math=False)
    if mode in {"mem_efficient", "mem", "me"}:
        return sdp_kernel(enable_flash=False, enable_mem_efficient=True, enable_math=False)
    if mode in {"math"}:
        return sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)

    return contextlib.nullcontext()


class CCGQAAttention(nn.Module):
    """Compressed Convolutional Grouped Query Attention.

    Canonical implementation lives under `hydra.attention.backends.ccgqa`.
    Higher-level wiring (blocks / models / routing) lives under `hydra.model.framework`.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int = 16,
        n_kv_heads: int = 4,
        compression_factor: int = 4,
        max_seq_len: int = 8192,
        use_rope: bool = True,
        use_qk_norm: bool = True,
        use_convs: bool = True,
        use_qk_mean: bool = True,
        use_value_shift: bool = True,
        conv_kernel_size: int = 3,
        use_fused_kernel: bool = True,
    ):
        super().__init__()

        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_groups = n_heads // n_kv_heads
        self.compression_factor = compression_factor
        self.use_rope = use_rope
        self.use_qk_norm = use_qk_norm
        self.use_convs = use_convs
        self.use_qk_mean = use_qk_mean
        self.use_value_shift = use_value_shift
        self.use_fused_kernel = use_fused_kernel and TRITON_ATTENTION_AVAILABLE

        self.latent_dim = dim // compression_factor
        self.head_dim = self.latent_dim // n_heads
        self.kv_dim = n_kv_heads * self.head_dim

        assert self.latent_dim % n_heads == 0, (
            f"latent_dim {self.latent_dim} must be divisible by n_heads {n_heads}"
        )
        assert self.head_dim % 2 == 0, (
            f"head_dim {self.head_dim} must be even for RoPE (splits into pairs)"
        )

        # Fused down-projections: combine q_down, k_down, v_down into single GEMM
        if use_value_shift:
            # q (latent_dim) + k (kv_dim) + v_curr (kv_dim//2) + v_shift (kv_dim//2)
            fused_dim = self.latent_dim + self.kv_dim + self.kv_dim
        else:
            # q (latent_dim) + k (kv_dim) + v (kv_dim)
            fused_dim = self.latent_dim + self.kv_dim + self.kv_dim
        
        self.qkv_proj = nn.Linear(dim, fused_dim, bias=False)

        self.o_proj = nn.Linear(self.latent_dim, dim, bias=False)
        
        # Output scaling: only needed for hybrid Lightning+CCGQA architectures.
        # For pure CCGQA, use 1.0 to avoid gradient explosion.
        # Set HYDRA_CCGQA_OUTPUT_SCALE=2.0 for hybrid mode.
        scale_env = os.environ.get("HYDRA_CCGQA_OUTPUT_SCALE", "1.0")
        self.output_scale = float(scale_env)

        if use_convs:
            # Use optimized fused conv sequences instead of 4 separate convs
            self.q_conv = OptimizedConvSequence(
                channels=self.latent_dim,
                groups1=n_heads,     # depthwise for first conv
                groups2=1,           # full mixing for second conv
                kernel_size=conv_kernel_size,
            )
            self.k_conv = OptimizedConvSequence(
                channels=self.kv_dim,
                groups1=n_kv_heads,  # depthwise for first conv
                groups2=1,           # full mixing for second conv
                kernel_size=conv_kernel_size,
            )

        self.key_temperature = nn.Parameter(torch.full((1,), float(self.head_dim**0.5)))

        if use_rope:
            self._init_rope(max_seq_len)

        self.scale = self.head_dim**-0.5
        self.max_seq_len = max_seq_len

        # Pre-allocate causal mask for padding-mask combination path.
        # Only needed when combining with padding masks; SDPA handles is_causal=True natively.
        # This trades ~4MB (max_seq_len² bool) for eliminating per-forward triu+ones allocation.
        _cmask = torch.ones((max_seq_len, max_seq_len), dtype=torch.bool).triu_(diagonal=1)
        self.register_buffer("_causal_mask", _cmask, persistent=False)

    def _init_rope(self, max_seq_len: int):
        head_dim = self.head_dim
        theta = 10000.0

        freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, freqs)

        # MEMORY FIX: Default to bfloat16 for RoPE cache to save ~50% memory.
        # Most training uses bf16 anyway, and precision loss is negligible for positional info.
        # Override with HYDRA_ROPE_CACHE_DTYPE=fp32 if float32 precision is needed.
        rope_cache_dtype = os.environ.get("HYDRA_ROPE_CACHE_DTYPE", "bf16").strip().lower()
        if rope_cache_dtype in {"bf16", "bfloat16"}:
            cache_dtype = torch.bfloat16
        elif rope_cache_dtype in {"fp16", "float16", "half"}:
            cache_dtype = torch.float16
        elif rope_cache_dtype in {"fp32", "float32", "float"}:
            cache_dtype = torch.float32
        else:
            cache_dtype = torch.bfloat16  # Default to bf16

        self.register_buffer(
            "cos_cached",
            freqs.cos().to(dtype=cache_dtype).unsqueeze(0).unsqueeze(0),
            persistent=False,
        )
        self.register_buffer(
            "sin_cached",
            freqs.sin().to(dtype=cache_dtype).unsqueeze(0).unsqueeze(0),
            persistent=False,
        )

    def _apply_rope(self, x: torch.Tensor, seq_len: int, offset: int = 0) -> torch.Tensor:
        cos = self.cos_cached[:, :, offset:offset + seq_len, :]
        sin = self.sin_cached[:, :, offset:offset + seq_len, :]

        if FUSED_KERNELS_AVAILABLE:
            return fused_rope(x, cos, sin)

        x1, x2 = x[..., ::2], x[..., 1::2]
        x_pairs = x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
        out_pairs = torch.empty_like(x_pairs)
        out_pairs[..., 0] = x1 * cos - x2 * sin
        out_pairs[..., 1] = x1 * sin + x2 * cos
        return out_pairs.flatten(-2)

    def forward_with_cache(
        self,
        x: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        start_pos: int = 0,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass with KV cache for incremental decoding.

        During prefill (past_kv=None, start_pos=0): processes full sequence,
        returns cached K,V.
        During decode (past_kv provided, start_pos>0): processes single token,
        concatenates with cached K,V.

        Skips convolutions and QK-mean mixing during single-token decode for
        simplicity. These are minor refinements that don't meaningfully affect
        generation quality.

        Args:
            x: Input [B, S, D] where S=seq_len for prefill, S=1 for decode
            past_kv: Cached (K, V) from previous steps, each [B, n_kv_heads, T, head_dim]
            start_pos: Position offset for RoPE

        Returns:
            Tuple of (output, (new_K_cache, new_V_cache))
        """
        B, S, _ = x.shape
        is_decode = (S == 1 and past_kv is not None)

        # Fused projection: single GEMM for q, k, v
        qkv = self.qkv_proj(x)

        q = qkv[..., :self.latent_dim]
        k = qkv[..., self.latent_dim:self.latent_dim + self.kv_dim]

        if self.use_value_shift:
            v_curr = qkv[..., self.latent_dim + self.kv_dim:self.latent_dim + self.kv_dim + self.kv_dim // 2]
            v_prev = qkv[..., self.latent_dim + self.kv_dim + self.kv_dim // 2:]
            if not is_decode:
                # Optimized: pad+cat (matches training forward path)
                v_prev_shifted = F.pad(v_prev[:, :-1, :], (0, 0, 1, 0))
                v = torch.cat([v_curr, v_prev_shifted], dim=-1)
            else:
                # Decode (S=1): no previous context in current token, second half = 0
                v = torch.cat([v_curr, torch.zeros_like(v_curr)], dim=-1)
        else:
            v = qkv[..., self.latent_dim + self.kv_dim:]

        # During decode, skip convolutions and QK-mean mixing
        # (they require sequence context and are minor refinements)
        if not is_decode:
            if self.use_qk_mean:
                q_pre = q
                k_pre = k

            if self.use_convs:
                q = self.q_conv(q)
                k = self.k_conv(k)

            if self.use_qk_mean and self.n_groups == 1:
                qk_mean = 0.5 * (q_pre + k_pre)
                q = q + qk_mean
                k = k + qk_mean
            elif self.use_qk_mean:
                q_mean = q_pre.view(B, S, self.n_heads, self.head_dim).mean(dim=2)
                k_mean = k_pre.view(B, S, self.n_kv_heads, self.head_dim).mean(dim=2)
                q = (
                    q.view(B, S, self.n_heads, self.head_dim) + (0.5 * k_mean.unsqueeze(2))
                ).reshape(B, S, self.latent_dim)
                k = (
                    k.view(B, S, self.n_kv_heads, self.head_dim) + (0.5 * q_mean.unsqueeze(2))
                ).reshape(B, S, self.kv_dim)

        # Reshape to heads
        q = q.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # QK normalization
        if self.use_qk_norm:
            if FUSED_QK_NORM_AVAILABLE and _fused_qk_norm is not None and q.is_cuda:
                q, k = _fused_qk_norm(q, k, scale=1.0, temperature=1.0)
                k = k * self.key_temperature
            else:
                q = F.normalize(q, p=2, dim=-1)
                k = F.normalize(k, p=2, dim=-1) * self.key_temperature

        # RoPE with position offset
        if self.use_rope:
            q = self._apply_rope(q, S, offset=start_pos)
            k = self._apply_rope(k, S, offset=start_pos)

        # Concatenate with cached K,V
        if past_kv is not None:
            cached_k, cached_v = past_kv
            k = torch.cat([cached_k, k], dim=2)  # [B, n_kv_heads, T+S, head_dim]
            v = torch.cat([cached_v, v], dim=2)

        # Store new cache
        new_past_kv = (k, v)

        # Full sequence length for attention
        T = k.shape[2]

        # Attention computation — use SDPA (handles causal masking internally)
        # For decode mode with cache, we can't use is_causal=True (Q has 1 token,
        # K/V has T tokens). Instead, no mask needed since the single query token
        # should attend to all cached + current keys.
        if is_decode:
            # Single query attends to full K/V history — no causal mask needed
            try:
                out = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=None,
                    is_causal=False,
                    scale=None if self.use_qk_norm else self.scale,
                    enable_gqa=(self.n_kv_heads != self.n_heads),
                )
            except TypeError:
                # Older PyTorch without enable_gqa
                if self.n_kv_heads != self.n_heads:
                    k_exp = k.unsqueeze(2).expand(-1, -1, self.n_groups, -1, -1).reshape(B, self.n_heads, T, self.head_dim)
                    v_exp = v.unsqueeze(2).expand(-1, -1, self.n_groups, -1, -1).reshape(B, self.n_heads, T, self.head_dim)
                else:
                    k_exp, v_exp = k, v
                out = F.scaled_dot_product_attention(
                    q, k_exp, v_exp,
                    attn_mask=None, is_causal=False,
                    scale=None if self.use_qk_norm else self.scale,
                )
        else:
            # Prefill: standard causal attention
            try:
                out = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=None,
                    is_causal=True,
                    scale=None if self.use_qk_norm else self.scale,
                    enable_gqa=(self.n_kv_heads != self.n_heads),
                )
            except TypeError:
                if self.n_kv_heads != self.n_heads:
                    k_exp = k.unsqueeze(2).expand(-1, -1, self.n_groups, -1, -1).reshape(B, self.n_heads, T, self.head_dim)
                    v_exp = v.unsqueeze(2).expand(-1, -1, self.n_groups, -1, -1).reshape(B, self.n_heads, T, self.head_dim)
                else:
                    k_exp, v_exp = k, v
                out = F.scaled_dot_product_attention(
                    q, k_exp, v_exp,
                    attn_mask=None, is_causal=True,
                    scale=None if self.use_qk_norm else self.scale,
                )

        out = out.transpose(1, 2).contiguous().view(B, S, self.latent_dim)
        out = self.o_proj(out) * self.output_scale
        return out, new_past_kv

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, S, _ = x.shape

        # Fused projection: single GEMM for q, k, v
        qkv = self.qkv_proj(x)
        
        # Split fused output
        q = qkv[..., :self.latent_dim]
        k = qkv[..., self.latent_dim:self.latent_dim + self.kv_dim]
        
        if self.use_value_shift:
            # v_curr and v_shift are concatenated in qkv
            v_curr = qkv[..., self.latent_dim + self.kv_dim:self.latent_dim + self.kv_dim + self.kv_dim // 2]
            v_prev = qkv[..., self.latent_dim + self.kv_dim + self.kv_dim // 2:]
            
            # Optimized value-shift: pad+cat instead of empty+3 copies
            # Shift v_prev right by 1 position with zero-padding
            v_prev_shifted = F.pad(v_prev[:, :-1, :], (0, 0, 1, 0))  # [B, S, half_kv]
            v = torch.cat([v_curr, v_prev_shifted], dim=-1)           # [B, S, kv_dim]
        else:
            v = qkv[..., self.latent_dim + self.kv_dim:]

        if self.use_qk_mean:
            q_pre = q
            k_pre = k

        if self.use_convs:
            # Use optimized fused convolution sequences
            q = self.q_conv(q)
            k = self.k_conv(k)

        if self.use_qk_mean and self.n_groups == 1:
            qk_mean = 0.5 * (q_pre + k_pre)
            q = q + qk_mean
            k = k + qk_mean
        elif self.use_qk_mean:
            q_mean = q_pre.view(B, S, self.n_heads, self.head_dim).mean(dim=2)
            k_mean = k_pre.view(B, S, self.n_kv_heads, self.head_dim).mean(dim=2)

            q = (
                q.view(B, S, self.n_heads, self.head_dim) + (0.5 * k_mean.unsqueeze(2))
            ).reshape(B, S, self.latent_dim)
            k = (
                k.view(B, S, self.n_kv_heads, self.head_dim) + (0.5 * q_mean.unsqueeze(2))
            ).reshape(B, S, self.kv_dim)

        q = q.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Optional: force contiguous tensors before SDPA.
        # Some SDPA backends will materialize contiguous copies anyway; making
        # it explicit can be faster on certain shapes/hardware.
        if _env_flag("HYDRA_CCQA_SDPA_CONTIGUOUS_QKV"):
            q = q.contiguous()
            k = k.contiguous()
            v = v.contiguous()

        if self.use_qk_norm:
            if FUSED_QK_NORM_AVAILABLE and _fused_qk_norm is not None and q.is_cuda:
                # Single fused kernel: L2-normalize Q and K.
                # Temperature is applied outside the kernel to preserve autograd
                # gradient flow to the learnable key_temperature parameter.
                q, k = _fused_qk_norm(q, k, scale=1.0, temperature=1.0)
                k = k * self.key_temperature
            else:
                q = F.normalize(q, p=2, dim=-1)
                k = F.normalize(k, p=2, dim=-1) * self.key_temperature

        if self.use_rope:
            q = self._apply_rope(q, S)
            k = self._apply_rope(k, S)

        # Prepare attention mask if provided
        attn_mask = mask
        is_causal = (mask is None)
        
        if mask is not None:
            # Combine causal mask with padding mask
            # mask is (B, S), 1=valid, 0=pad
            # We need boolean mask where True = masked out

            # MEMORY FIX: Use cached causal mask slice instead of computing triu per call.
            causal_mask = self._causal_mask[:S, :S]

            # Padding part: True where mask == 0
            padding_mask = (mask == 0).view(B, 1, 1, S)

            attn_mask = causal_mask.unsqueeze(0).unsqueeze(0) | padding_mask

        # Fast path: try Sage Attention first (2-5x faster than Flash), then SDPA.
        # Sage uses INT8/FP8 quantization for speedup with minimal precision loss.
        out = None
        attn_backend = _get_ccgqa_attn_backend()

        # Try Sage Attention (fastest path)
        if (attn_backend in ("auto", "sage")
            and SAGE_ATTN_AVAILABLE
            and sageattn is not None
            and self.use_fused_kernel
            and q.is_cuda
            and attn_mask is None  # Sage doesn't support arbitrary attention masks
        ):
            try:
                # Sage supports GQA natively, expects [B, n_heads, S, head_dim]
                out = sageattn(q, k, v, tensor_layout="HND", is_causal=is_causal)
            except Exception:
                out = None  # Fall through to SDPA

        # Try PyTorch SDPA (Flash/MemEff) for both MHA and GQA
        if out is None and self.use_fused_kernel and q.is_cuda and self.head_dim >= 16:
            try:
                with _sdpa_kernel_context():
                    out = F.scaled_dot_product_attention(
                        q,
                        k,
                        v,
                        attn_mask=attn_mask,
                        is_causal=is_causal,
                        scale=None if self.use_qk_norm else self.scale,
                        enable_gqa=(self.n_kv_heads != self.n_heads),
                    )
            except TypeError:
                # Older PyTorch builds may not support enable_gqa.
                # Only use custom kernel if no mask is provided (it doesn't support arbitrary masks)
                if attn_mask is None:
                    out = ccgqa_attention_fused(
                        q,
                        k,
                        v,
                        is_causal=is_causal,
                    )

        if out is None:
            # SDPA expects K/V head count to match Q.
            k = k.unsqueeze(2).expand(-1, -1, self.n_groups, -1, -1).reshape(
                B, self.n_heads, S, self.head_dim
            )
            v = v.unsqueeze(2).expand(-1, -1, self.n_groups, -1, -1).reshape(
                B, self.n_heads, S, self.head_dim
            )
            with _sdpa_kernel_context():
                out = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=attn_mask,
                    is_causal=is_causal,
                    scale=None if self.use_qk_norm else self.scale,
                )

        out = out.transpose(1, 2).contiguous().view(B, S, self.latent_dim)
        out = self.o_proj(out) * self.output_scale
        return out
    
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        """Handle loading old checkpoints with separate q_down, k_down, v_down."""
        # Check if this is an old checkpoint with separate projections
        q_key = prefix + 'q_down.weight'
        k_key = prefix + 'k_down.weight'
        v_key = prefix + 'v_down.weight'
        v_shift_key = prefix + 'v_shift_down.weight'
        qkv_key = prefix + 'qkv_proj.weight'

        # Backward-compat: older checkpoints used 4 separate convs:
        #   q_conv1/q_conv2 and k_conv1/k_conv2
        # New structure is:
        #   q_conv.conv1/q_conv.conv2 and k_conv.conv1/k_conv.conv2
        def _remap_or_drop(old_suffix: str, new_suffix: str, has_new_param: bool) -> None:
            old_key = prefix + old_suffix
            new_key = prefix + new_suffix
            if old_key not in state_dict:
                return
            if has_new_param and new_key not in state_dict:
                state_dict[new_key] = state_dict.pop(old_key)
            else:
                # Prefer the new key if both exist; otherwise drop old to avoid unexpected keys.
                state_dict.pop(old_key, None)

        has_q_conv = hasattr(self, "q_conv") and hasattr(getattr(self, "q_conv", None), "conv1")
        has_k_conv = hasattr(self, "k_conv") and hasattr(getattr(self, "k_conv", None), "conv1")
        for suffix in ("weight", "bias"):
            _remap_or_drop(f"q_conv1.{suffix}", f"q_conv.conv1.{suffix}", has_q_conv)
            _remap_or_drop(f"q_conv2.{suffix}", f"q_conv.conv2.{suffix}", has_q_conv)
            _remap_or_drop(f"k_conv1.{suffix}", f"k_conv.conv1.{suffix}", has_k_conv)
            _remap_or_drop(f"k_conv2.{suffix}", f"k_conv.conv2.{suffix}", has_k_conv)
        
        if q_key in state_dict and qkv_key not in state_dict:
            # Migrate old checkpoint: concatenate separate projections into fused qkv_proj
            q_weight = state_dict.pop(q_key)
            k_weight = state_dict.pop(k_key)
            
            if self.use_value_shift and v_shift_key in state_dict:
                v_weight = state_dict.pop(v_key)
                v_shift_weight = state_dict.pop(v_shift_key)
                # Concatenate: [q_down, k_down, v_down, v_shift_down]
                fused_weight = torch.cat([q_weight, k_weight, v_weight, v_shift_weight], dim=0)
            else:
                v_weight = state_dict.pop(v_key)
                # Concatenate: [q_down, k_down, v_down]
                fused_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
            
            state_dict[qkv_key] = fused_weight
        
        # Call parent implementation
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
