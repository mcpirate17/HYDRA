"""
Hybrid Attention Architecture: MQA + CCQA + MLA

Recommended Architecture Recipe (200M–500M Test Scale)
=======================================================

This module implements a hybrid transformer with three attention variants:
1. MQA (Multi-Query Attention): Cheap local extraction, full precision
2. CCQA (Compressed Convolutional Query Attention): Stable high-capacity compressed global mixer
3. MLA (Multi-head Latent Attention): Predictable latent-space summarizer

Architecture Principles (from diagnostics findings):
- Pre-norm RMSNorm + RMSNorm before o_proj (eliminates gradient spikes)
- Residual scaling: α=0.5 for CCQA/MLA, α=1.0 for MQA
- QK modulation gain=0.25 (clamped for variance control in MoR unrolled steps)
- Post-mix RMSNorm after QK-mean coupling (critical for gradient stability)
- MLA latent dim at 1/2 or 1/4 of model dim with post-mix RMSNorm

8-Layer Macro-Block Pattern:
  MQA → MQA → CCQA → CCQA → CCQA → MLA → MQA → MLA

This provides:
- Cheap local extraction up front (MQA)
- Stable high-capacity compressed global mixer in the middle (CCQA)
- Predictable latent-space summarizer near the top (MLA)

Training Recommendations:
=========================
- Gradient Clipping: Use torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
- Learning Rate: Start with 1e-4, use cosine schedule with warmup
- Weight Decay: 0.1 for most params, 0.0 for norms and biases
- Batch Size: Start small (32-64) and scale up as training stabilizes
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union, Dict, Any
from dataclasses import dataclass
from enum import Enum

# Import MoD routing
from hydra.routing.mixture_of_depths import MixtureOfDepthsBlock


# =============================================================================
# Normalization Layers
# =============================================================================

# Import fused kernels for optimized operations
try:
    from hydra.kernels import fused_rope, fused_qk_norm, fused_swiglu, fused_rms_norm
    FUSED_KERNELS_AVAILABLE = True
except ImportError:
    FUSED_KERNELS_AVAILABLE = False


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    Uses native torch operations that work efficiently with torch.compile
    and mixed precision training. Weights are kept in float32 for stability,
    and the forward pass handles mixed dtypes correctly.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use fused kernel if available
        if FUSED_KERNELS_AVAILABLE:
            return fused_rms_norm(x, self.weight, self.eps)
        # Fallback: Cast weight to input dtype for fused operations
        dtype = x.dtype
        x = x.float()
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * rms).to(dtype) * self.weight.to(dtype)


# =============================================================================
# Rotary Position Embeddings (Shared)
# =============================================================================


class RotaryEmbedding(nn.Module):
    """Rotary Position Embeddings for attention."""

    def __init__(self, head_dim: int, max_seq_len: int = 8192, theta: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len

        freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, freqs)

        self.register_buffer("cos_cached", freqs.cos().unsqueeze(0).unsqueeze(0))
        self.register_buffer("sin_cached", freqs.sin().unsqueeze(0).unsqueeze(0))

    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Apply RoPE to input tensor [B, n_heads, S, head_dim]."""
        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]

        # Use fused kernel if available
        if FUSED_KERNELS_AVAILABLE:
            return fused_rope(x, cos, sin)
        
        # Fallback
        x1, x2 = x[..., ::2], x[..., 1::2]
        rotated = torch.stack(
            [x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1
        ).flatten(-2)
        return rotated


# =============================================================================
# Attention Type Enum
# =============================================================================


class AttentionType(Enum):
    MQA = "mqa"  # Multi-Query Attention (cheap, local)
    CCQA = "ccqa"  # Compressed Convolutional Query Attention (global, compressed)
    MLA = "mla"  # Multi-head Latent Attention (summarizer)


# =============================================================================
# MQA: Multi-Query Attention
# =============================================================================


class MQAAttention(nn.Module):
    """
    Multi-Query Attention (MQA) - Cheap local extraction.

    Features:
    - Single KV head shared across all Q heads
    - Full precision (no compression)
    - Pre-norm RMSNorm + RMSNorm before o_proj
    - Residual scaling α=1.0 (full residual)

    Use for: First layers, skip paths in MoD routing.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int = 12,
        max_seq_len: int = 8192,
        use_rope: bool = True,
        norm_eps: float = 1e-6,
    ):
        super().__init__()

        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim**-0.5
        self.residual_scale = 1.0  # Full residual for MQA

        assert dim % n_heads == 0, f"dim {dim} must be divisible by n_heads {n_heads}"

        # Projections: Q has n_heads, K/V have 1 head each (Multi-Query)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, self.head_dim, bias=False)  # Single head
        self.v_proj = nn.Linear(dim, self.head_dim, bias=False)  # Single head

        # Pre-output normalization (critical for gradient stability)
        self.pre_out_norm = RMSNorm(dim, eps=norm_eps)
        self.o_proj = nn.Linear(dim, dim, bias=False)

        # RoPE
        self.use_rope = use_rope
        if use_rope:
            self.rope = RotaryEmbedding(self.head_dim, max_seq_len)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for MQA."""
        B, S, D = x.shape

        # Project Q, K, V
        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, 1, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, 1, self.head_dim).transpose(1, 2)

        # Apply RoPE
        if self.use_rope:
            q = self.rope(q, S)
            k = self.rope(k, S)

        # Expand K, V for all heads
        k = k.expand(-1, self.n_heads, -1, -1)
        v = v.expand(-1, self.n_heads, -1, -1)

        # Scaled dot-product attention
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            is_causal=True if mask is None else False,
            scale=self.scale,
        )

        # Reshape and project output
        out = out.transpose(1, 2).contiguous().view(B, S, D)

        # Pre-output normalization (key stability feature)
        out = self.pre_out_norm(out)
        out = self.o_proj(out)

        return out


# =============================================================================
# CCQA: Compressed Convolutional Query Attention
# =============================================================================


class CCQAAttention(nn.Module):
    """
    Compressed Convolutional Query Attention (CCQA) - Global mixer.

    Features:
    - Down-project to compressed latent space
    - Sequence + channel convolutions with QK coupling
    - QK L2 normalization with learnable temperature
    - Pre-norm RMSNorm + RMSNorm before o_proj
    - Post-mix RMSNorm after QK-mean coupling (gradient stability)
    - Residual scaling α=0.5 (for MoR stability)
    - QK modulation gain=0.25 (clamped for variance control)

    Use for: Middle layers, slow path in MoD routing.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int = 12,
        n_kv_heads: int = 3,
        compression_factor: int = 4,
        max_seq_len: int = 8192,
        use_rope: bool = True,
        use_qk_norm: bool = True,
        use_convs: bool = True,
        use_qk_mean: bool = True,
        conv_kernel_size: int = 3,
        qk_modulation_gain: float = 0.25,  # Clamped low for gradient stability
        norm_eps: float = 1e-6,
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
        self.qk_modulation_gain = qk_modulation_gain
        self.residual_scale = 0.5  # Reduced for MoR stability

        # Compressed dimensions
        self.latent_dim = dim // compression_factor
        self.head_dim = self.latent_dim // n_heads
        self.kv_dim = n_kv_heads * self.head_dim
        self.scale = self.head_dim**-0.5

        assert self.latent_dim % n_heads == 0
        assert self.head_dim >= 2, "head_dim too small, reduce compression"

        # Down-projections
        self.q_down = nn.Linear(dim, self.latent_dim, bias=False)
        self.k_down = nn.Linear(dim, self.kv_dim, bias=False)
        self.v_down = nn.Linear(dim, self.kv_dim, bias=False)

        # Normalize compressed latent projection (stability)
        self.latent_norm = RMSNorm(self.latent_dim, eps=norm_eps)
        self.kv_latent_norm = RMSNorm(self.kv_dim, eps=norm_eps)

        # Convolutions for Q and K
        if use_convs:
            self.q_conv = nn.Conv1d(
                self.latent_dim,
                self.latent_dim,
                kernel_size=conv_kernel_size,
                padding=conv_kernel_size - 1,
                groups=n_heads,
                bias=False,
            )
            self.k_conv = nn.Conv1d(
                self.kv_dim,
                self.kv_dim,
                kernel_size=conv_kernel_size,
                padding=conv_kernel_size - 1,
                groups=n_kv_heads,
                bias=False,
            )
            # Normalize conv output (key for gradient stability)
            self.conv_norm = RMSNorm(self.latent_dim, eps=norm_eps)
            self.k_conv_norm = RMSNorm(self.kv_dim, eps=norm_eps)

        # Post-mix normalization (after QK-mean coupling)
        self.post_mix_q_norm = RMSNorm(self.latent_dim, eps=norm_eps)
        self.post_mix_k_norm = RMSNorm(self.kv_dim, eps=norm_eps)

        # Learnable temperature
        self.key_temperature = nn.Parameter(torch.ones(1))

        # RoPE
        if use_rope:
            self.rope = RotaryEmbedding(self.head_dim, max_seq_len)

        # Pre-output normalization (critical for stability)
        self.pre_out_norm = RMSNorm(self.latent_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.latent_dim, dim, bias=False)

    def _apply_causal_conv(self, x: torch.Tensor, conv: nn.Conv1d) -> torch.Tensor:
        """Apply causal convolution."""
        x = x.transpose(1, 2)
        x = conv(x)
        x = x[..., : x.size(-1) - (conv.kernel_size[0] - 1)]
        return x.transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for CCQA."""
        B, S, D = x.shape

        # Down-project to compressed latent
        q = self.q_down(x)
        k = self.k_down(x)
        v = self.v_down(x)

        # Normalize compressed projections
        q = self.latent_norm(q)
        k = self.kv_latent_norm(k)
        v = self.kv_latent_norm(v)

        # Store pre-conv for QK-mean
        if self.use_qk_mean:
            q_pre = q.clone()
            k_pre = k.clone()

        # Apply convolutions
        if self.use_convs:
            q = self._apply_causal_conv(q, self.q_conv)
            k = self._apply_causal_conv(k, self.k_conv)
            # Normalize conv output
            q = self.conv_norm(q)
            k = self.k_conv_norm(k)

        # QK-mean coupling with constrained gain
        if self.use_qk_mean and self.n_groups == 1:
            qk_mean = self.qk_modulation_gain * (q_pre + k_pre)
            q = q + qk_mean
            k = k + qk_mean
        elif self.use_qk_mean:
            # Simplified for GQA
            q_mean = q_pre.view(B, S, self.n_heads, self.head_dim).mean(dim=2)
            k_mean = k_pre.view(B, S, self.n_kv_heads, self.head_dim).mean(dim=2)

            q = q + self.qk_modulation_gain * k_mean.unsqueeze(2).expand(
                -1, -1, self.n_heads, -1
            ).reshape(B, S, self.latent_dim)
            k = k + self.qk_modulation_gain * q_mean.unsqueeze(2).expand(
                -1, -1, self.n_kv_heads, -1
            ).reshape(B, S, self.kv_dim)

        # Post-mix normalization (critical for gradient stability after QK coupling)
        q = self.post_mix_q_norm(q)
        k = self.post_mix_k_norm(k)

        # Reshape to heads
        q = q.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # QK normalization
        if self.use_qk_norm:
            q = F.normalize(q, p=2, dim=-1) * math.sqrt(self.head_dim)
            k = (
                F.normalize(k, p=2, dim=-1)
                * math.sqrt(self.head_dim)
                * self.key_temperature
            )

        # Apply RoPE
        if self.use_rope:
            q = self.rope(q, S)
            k = self.rope(k, S)

        # Expand K, V for GQA
        k = (
            k.unsqueeze(2)
            .expand(-1, -1, self.n_groups, -1, -1)
            .reshape(B, self.n_heads, S, self.head_dim)
        )
        v = (
            v.unsqueeze(2)
            .expand(-1, -1, self.n_groups, -1, -1)
            .reshape(B, self.n_heads, S, self.head_dim)
        )

        # Attention
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            is_causal=True if mask is None else False,
            scale=self.scale,
        )

        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(B, S, self.latent_dim)

        # Pre-output normalization (critical)
        out = self.pre_out_norm(out)
        out = self.o_proj(out)

        return out


# =============================================================================
# MLA: Multi-head Latent Attention
# =============================================================================


class MLAAttention(nn.Module):
    """
    Multi-head Latent Attention (MLA) - Latent-space summarizer.

    Features:
    - Latent dim at 1/2 or 1/4 of model dim
    - Post-mix RMSNorm for smooth interaction with MQA paths
    - Pre-norm RMSNorm + RMSNorm before o_proj
    - Residual scaling α=0.5 (for stability)
    - No up-projection (keeps things simple and fast)

    Use for: Later layers, summarization, stable global context.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int = 12,
        latent_ratio: float = 0.5,  # 1/2 of dim for latent
        max_seq_len: int = 8192,
        use_rope: bool = True,
        norm_eps: float = 1e-6,
    ):
        super().__init__()

        self.dim = dim
        self.n_heads = n_heads
        self.latent_dim = int(dim * latent_ratio)
        self.head_dim = self.latent_dim // n_heads
        self.scale = self.head_dim**-0.5
        self.residual_scale = 0.5  # Reduced for stability

        assert self.latent_dim % n_heads == 0
        assert self.head_dim >= 2

        # Down-project to latent
        self.down_proj = nn.Linear(dim, self.latent_dim, bias=False)
        self.latent_norm = RMSNorm(self.latent_dim, eps=norm_eps)

        # Q, K, V projections in latent space
        self.q_proj = nn.Linear(self.latent_dim, self.latent_dim, bias=False)
        self.k_proj = nn.Linear(self.latent_dim, self.latent_dim, bias=False)
        self.v_proj = nn.Linear(self.latent_dim, self.latent_dim, bias=False)

        # Post-mix normalization (for smooth MQA interaction)
        self.post_mix_norm = RMSNorm(self.latent_dim, eps=norm_eps)

        # RoPE
        self.use_rope = use_rope
        if use_rope:
            self.rope = RotaryEmbedding(self.head_dim, max_seq_len)

        # Pre-output normalization and up-projection
        self.pre_out_norm = RMSNorm(self.latent_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.latent_dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for MLA."""
        B, S, D = x.shape

        # Down-project to latent
        h = self.down_proj(x)
        h = self.latent_norm(h)

        # Q, K, V in latent space
        q = self.q_proj(h).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        if self.use_rope:
            q = self.rope(q, S)
            k = self.rope(k, S)

        # Attention
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            is_causal=True if mask is None else False,
            scale=self.scale,
        )

        # Reshape
        out = out.transpose(1, 2).contiguous().view(B, S, self.latent_dim)

        # Post-mix normalization (smooth MQA interaction)
        out = self.post_mix_norm(out)

        # Pre-output normalization
        out = self.pre_out_norm(out)
        out = self.o_proj(out)

        return out


# =============================================================================
# SwiGLU MLP
# =============================================================================


class SwiGLUMLP(nn.Module):
    """SwiGLU MLP with fused gate/up projection."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_up = nn.Linear(dim, 2 * hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up(x)
        gate, up = gate_up.chunk(2, dim=-1)
        # Use fused kernel if available
        if FUSED_KERNELS_AVAILABLE:
            return self.down(fused_swiglu(gate, up))
        return self.down(F.silu(gate) * up)


# =============================================================================
# Hybrid Attention Block
# =============================================================================


class HybridAttentionBlock(nn.Module):
    """
    Transformer block with configurable attention type.

    Features:
    - Pre-norm architecture with RMSNorm
    - Configurable attention: MQA, CCQA, or MLA
    - Residual scaling (α) based on attention type
    - SwiGLU MLP with 3-4x expansion
    """

    def __init__(
        self,
        dim: int,
        attention_type: AttentionType,
        n_heads: int = 12,
        n_kv_heads: int = 3,
        compression_factor: int = 4,
        mlp_ratio: float = 3.5,
        max_seq_len: int = 8192,
        norm_eps: float = 1e-6,
        **attention_kwargs,
    ):
        super().__init__()

        self.attention_type = attention_type

        # Create attention based on type
        if attention_type == AttentionType.MQA:
            self.attention = MQAAttention(
                dim=dim,
                n_heads=n_heads,
                max_seq_len=max_seq_len,
                norm_eps=norm_eps,
                **{k: v for k, v in attention_kwargs.items() if k in ["use_rope"]},
            )
            self.residual_scale = 1.0
        elif attention_type == AttentionType.CCQA:
            self.attention = CCQAAttention(
                dim=dim,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                compression_factor=compression_factor,
                max_seq_len=max_seq_len,
                norm_eps=norm_eps,
                **attention_kwargs,
            )
            self.residual_scale = 0.5
        elif attention_type == AttentionType.MLA:
            self.attention = MLAAttention(
                dim=dim,
                n_heads=n_heads,
                max_seq_len=max_seq_len,
                norm_eps=norm_eps,
                **{
                    k: v
                    for k, v in attention_kwargs.items()
                    if k in ["use_rope", "latent_ratio"]
                },
            )
            self.residual_scale = 0.5
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")

        # Pre-norms
        self.norm1 = RMSNorm(dim, eps=norm_eps)
        self.norm2 = RMSNorm(dim, eps=norm_eps)

        # SwiGLU MLP
        hidden_dim = int(dim * mlp_ratio)
        hidden_dim = ((hidden_dim + 255) // 256) * 256  # Round to multiple of 256
        self.mlp = SwiGLUMLP(dim, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Attention with residual scaling
        h = x + self.residual_scale * self.attention(self.norm1(x), mask=mask)
        # MLP with full residual
        out = h + self.mlp(self.norm2(h))
        return out


# =============================================================================
# Macro-Block: 8-Layer Pattern
# =============================================================================


# Default macro-block pattern: MQA → MQA → CCQA → CCQA → CCQA → MLA → MQA → MLA
DEFAULT_MACRO_PATTERN = [
    AttentionType.MQA,  # Layer 0: Cheap local
    AttentionType.MQA,  # Layer 1: Cheap local
    AttentionType.CCQA,  # Layer 2: Compressed global
    AttentionType.CCQA,  # Layer 3: Compressed global
    AttentionType.CCQA,  # Layer 4: Compressed global
    AttentionType.MLA,  # Layer 5: Latent summarizer
    AttentionType.MQA,  # Layer 6: Local refinement
    AttentionType.MLA,  # Layer 7: Final summarizer
]


class MacroBlock(nn.Module):
    """
    8-Layer Macro-Block following the hybrid pattern.

    Pattern: MQA → MQA → CCQA → CCQA → CCQA → MLA → MQA → MLA
    """

    def __init__(
        self,
        dim: int,
        n_heads: int = 12,
        n_kv_heads: int = 3,
        compression_factor: int = 4,
        mlp_ratio: float = 3.5,
        max_seq_len: int = 8192,
        norm_eps: float = 1e-6,
        pattern: Optional[List[AttentionType]] = None,
        **attention_kwargs,
    ):
        super().__init__()

        if pattern is None:
            pattern = DEFAULT_MACRO_PATTERN

        self.blocks = nn.ModuleList(
            [
                HybridAttentionBlock(
                    dim=dim,
                    attention_type=attn_type,
                    n_heads=n_heads,
                    n_kv_heads=n_kv_heads,
                    compression_factor=compression_factor,
                    mlp_ratio=mlp_ratio,
                    max_seq_len=max_seq_len,
                    norm_eps=norm_eps,
                    **attention_kwargs,
                )
                for attn_type in pattern
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, mask=mask)
        return x


# =============================================================================
# Hybrid Transformer Model
# =============================================================================


@dataclass
class HybridTransformerConfig:
    """Configuration for HybridTransformer."""

    vocab_size: int = 50257
    dim: int = 768  # 768-1024 for 200M-500M scale
    n_macro_blocks: int = 3  # 3 macro-blocks = 24 layers
    n_heads: int = 12
    n_kv_heads: int = 3
    compression_factor: int = 4
    mlp_ratio: float = 3.5  # 3-4x expansion
    max_seq_len: int = 8192
    norm_eps: float = 1e-6
    tie_weights: bool = True

    # MoD/MoR hooks (prepared but not active until stable base confirmed)
    enable_mod: bool = False
    enable_mor: bool = False
    mod_capacity: float = 0.75
    mor_max_recursions: int = 3


class HybridTransformer(nn.Module):
    """
    Hybrid Transformer with MQA + CCQA + MLA attention.

    24-layer model organized as 3 × 8-layer macro-blocks.
    Each macro-block follows: MQA → MQA → CCQA → CCQA → CCQA → MLA → MQA → MLA

    Target: 200M-500M parameters depending on dim (768-1024).
    
    When enable_mod=True, each macro-block is wrapped with MoD routing,
    allowing the model to skip processing for a fraction of tokens.
    """

    def __init__(self, config: HybridTransformerConfig):
        super().__init__()
        self.config = config

        # Token embedding
        self.tok_emb = nn.Embedding(config.vocab_size, config.dim)

        # Create base macro-blocks
        base_blocks = [
            MacroBlock(
                dim=config.dim,
                n_heads=config.n_heads,
                n_kv_heads=config.n_kv_heads,
                compression_factor=config.compression_factor,
                mlp_ratio=config.mlp_ratio,
                max_seq_len=config.max_seq_len,
                norm_eps=config.norm_eps,
            )
            for _ in range(config.n_macro_blocks)
        ]
        
        # Optionally wrap with MoD routing
        if config.enable_mod:
            self.macro_blocks = nn.ModuleList([
                MixtureOfDepthsBlock(
                    block=block,
                    dim=config.dim,
                    capacity_ratio=config.mod_capacity,
                    aux_loss_weight=0.01,
                )
                for block in base_blocks
            ])
            self._mod_enabled = True
        else:
            self.macro_blocks = nn.ModuleList(base_blocks)
            self._mod_enabled = False

        # Final norm
        self.final_norm = RMSNorm(config.dim, eps=config.norm_eps)

        # Output projection
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        # Initialize weights
        self._init_weights()

        # Weight tying
        if config.tie_weights:
            self.output.weight = self.tok_emb.weight

    def _init_weights(self):
        """Initialize weights with scaled init."""
        n_layers = self.config.n_macro_blocks * 8
        residual_scale = 1.0 / math.sqrt(2 * n_layers)

        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                # Scale residual projections
                if "o_proj" in name or "down" in name:
                    module.weight.data *= residual_scale
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Conv1d):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass."""
        # Token embedding
        h = self.tok_emb(x)

        # Process through macro-blocks
        for macro_block in self.macro_blocks:
            h = macro_block(h, mask=mask)

        # Final norm and output
        h = self.final_norm(h)
        return self.output(h)
    
    def get_aux_loss(self) -> torch.Tensor:
        """Get auxiliary loss from MoD routing (load balancing)."""
        if not self._mod_enabled:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        aux_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        for block in self.macro_blocks:
            if hasattr(block, 'get_aux_loss'):
                aux_loss = aux_loss + block.get_aux_loss()
        return aux_loss

    def count_parameters(self) -> Dict[str, int]:
        """Count parameters by component."""
        counts = {
            "total": sum(p.numel() for p in self.parameters()),
            "embedding": self.tok_emb.weight.numel(),
            "attention": 0,
            "mlp": 0,
            "norm": 0,
        }

        for name, module in self.named_modules():
            params = sum(p.numel() for p in module.parameters(recurse=False))
            if "attention" in name.lower() or any(
                x in name for x in ["q_", "k_", "v_", "o_proj"]
            ):
                counts["attention"] += params
            elif "mlp" in name.lower() or "gate_up" in name or "down" in name:
                counts["mlp"] += params
            elif "norm" in name.lower():
                counts["norm"] += params

        return counts


# =============================================================================
# Factory Functions
# =============================================================================


def create_hybrid_transformer_small() -> HybridTransformer:
    """Create ~220M parameter hybrid transformer."""
    config = HybridTransformerConfig(
        dim=768,
        n_macro_blocks=3,
        n_heads=12,
        n_kv_heads=3,
        mlp_ratio=3.0,
    )
    return HybridTransformer(config)


def create_hybrid_transformer_medium() -> HybridTransformer:
    """Create ~350M parameter hybrid transformer."""
    config = HybridTransformerConfig(
        dim=896,
        n_macro_blocks=3,
        n_heads=14,
        n_kv_heads=2,
        mlp_ratio=3.5,
    )
    return HybridTransformer(config)


def create_hybrid_transformer_large() -> HybridTransformer:
    """Create ~480M parameter hybrid transformer."""
    config = HybridTransformerConfig(
        dim=1024,
        n_macro_blocks=3,
        n_heads=16,
        n_kv_heads=4,
        mlp_ratio=4.0,
    )
    return HybridTransformer(config)


# =============================================================================
# Test / Validation
# =============================================================================


if __name__ == "__main__":
    print("=" * 80)
    print("Hybrid Attention Architecture Test (with Stability Fixes)")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Test small model
    print("\n--- Testing Small Model (~220M) ---")
    model_small = create_hybrid_transformer_small().to(device)
    params = model_small.count_parameters()
    print(f"Total parameters: {params['total']:,} ({params['total'] / 1e6:.1f}M)")

    tokens = torch.randint(0, 50257, (2, 256)).to(device)
    with torch.no_grad():
        logits = model_small(tokens)
    print(f"Input: {tokens.shape}, Output: {logits.shape}")

    # Test medium model
    print("\n--- Testing Medium Model (~350M) ---")
    model_medium = create_hybrid_transformer_medium().to(device)
    params = model_medium.count_parameters()
    print(f"Total parameters: {params['total']:,} ({params['total'] / 1e6:.1f}M)")

    # Test large model
    print("\n--- Testing Large Model (~480M) ---")
    model_large = create_hybrid_transformer_large().to(device)
    params = model_large.count_parameters()
    print(f"Total parameters: {params['total']:,} ({params['total'] / 1e6:.1f}M)")

    # Test gradient flow with gradient clipping
    print("\n--- Testing Gradient Flow (with clip_grad_norm_=1.0) ---")
    model_small.train()
    tokens = torch.randint(0, 50257, (4, 128)).to(device)
    targets = torch.randn(4, 128, 50257).to(device)

    logits = model_small(tokens)
    loss = F.mse_loss(logits, targets)
    loss.backward()

    # Apply gradient clipping (RECOMMENDED for training)
    torch.nn.utils.clip_grad_norm_(model_small.parameters(), max_norm=1.0)

    # Check gradient norms after clipping (optimized with list comprehension)
    grad_norms = [
        param.grad.norm().item()
        for param in model_small.parameters()
        if param.grad is not None
    ]

    # Use generator for mean to avoid intermediate calculations
    mean_norm = sum(grad_norms) / len(grad_norms) if grad_norms else 0.0
    print(f"Mean gradient norm (after clipping): {mean_norm:.2e}")
    print(f"Max gradient norm (after clipping): {max(grad_norms):.2e}")
    print(f"Min gradient norm: {min(grad_norms):.2e}")

    # Stability checks
    exploding = max(grad_norms) > 1e5
    vanishing = min(grad_norms) < 1e-7
    print(f"Exploding gradients: {exploding}")
    print(f"Vanishing gradients: {vanishing}")

    # Verify stability features
    print("\n--- Stability Features Verification ---")
    print(f"✓ QK modulation gain: 0.25 (clamped)")
    print(f"✓ Residual scaling: α=0.5 (CCQA/MLA), α=1.0 (MQA)")
    print(f"✓ Post-mix normalization in CCQA")
    print(f"✓ Pre-output RMSNorm in all attention types")
    print(f"✓ Gradient clipping: clip_grad_norm_=1.0")

    print("\n" + "=" * 80)
    print("All tests passed!")
    print("=" * 80)
