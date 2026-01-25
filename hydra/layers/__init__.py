"""
HYDRA Shared Layers

Deduplicated common components used across attention variants.
Import from here instead of duplicating in each module.

Usage:
    from hydra.layers import RMSNorm, SwiGLUMLP, RotaryEmbedding
    from hydra.layers import flexible_attention, GradientCheckpointMixin
"""

from .common import (
    # Core layers
    RMSNorm,
    AdaRMSNorm,
    SwiGLUMLP,
    SwiGLUMLPFused,
    RotaryEmbedding,
    get_activation,
)
from .manifold_connections import ManifoldConstrainedHyperConnection

# Re-export common imports
from .common import (
    # Attention
    flexible_attention,
    set_attention_backend,
    get_attention_backend,
    # Gradient checkpointing
    GradientCheckpointMixin,
    checkpoint_sequential,
    # Weight initialization
    init_weights_normal,
    scale_residual_weights,
    # Utilities
    get_alibi_slopes,
    compute_memory_footprint,
    # Feature flags
    FUSED_KERNELS_AVAILABLE,
    FLASH_ATTN_AVAILABLE,
    FLASH_ATTN_VERSION,
    XFORMERS_AVAILABLE,
    SAGE_ATTN_AVAILABLE,
    SAGE_ATTN_VERSION,
    sageattn,
)

__all__ = [
    # Core layers
    "RMSNorm",
    "AdaRMSNorm",
    "SwiGLUMLP",
    "SwiGLUMLPFused",
    "RotaryEmbedding",
    "get_activation",
    # Manifold connections
    "ManifoldConstrainedHyperConnection",
    # Attention
    "flexible_attention",
    "set_attention_backend",
    "get_attention_backend",
    # Gradient checkpointing
    "GradientCheckpointMixin",
    "checkpoint_sequential",
    # Weight initialization
    "init_weights_normal",
    "scale_residual_weights",
    # Utilities
    "get_alibi_slopes",
    "compute_memory_footprint",
    # Feature flags
    "FUSED_KERNELS_AVAILABLE",
    "FLASH_ATTN_AVAILABLE",
    "FLASH_ATTN_VERSION",
    "XFORMERS_AVAILABLE",
    "SAGE_ATTN_AVAILABLE",
    "SAGE_ATTN_VERSION",
    "sageattn",
]
