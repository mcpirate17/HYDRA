"""
HYDRA Shared Layers

Deduplicated common components used across attention variants.
Import from here instead of duplicating in each module.
"""

from .common import (
    RMSNorm,
    SwiGLUMLP,
    RotaryEmbedding,
    get_activation,
    flexible_attention,
    set_attention_backend,
    get_attention_backend,
    GradientCheckpointMixin,
    init_weights_normal,
    scale_residual_weights,
    compute_memory_footprint,
    FUSED_KERNELS_AVAILABLE,
    FLASH_ATTN_AVAILABLE,
    XFORMERS_AVAILABLE,
)

__all__ = [
    "RMSNorm",
    "SwiGLUMLP", 
    "RotaryEmbedding",
    "get_activation",
    "flexible_attention",
    "set_attention_backend",
    "get_attention_backend",
    "GradientCheckpointMixin",
    "init_weights_normal",
    "scale_residual_weights",
    "compute_memory_footprint",
    "FUSED_KERNELS_AVAILABLE",
    "FLASH_ATTN_AVAILABLE",
    "XFORMERS_AVAILABLE",
]
