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
    FUSED_KERNELS_AVAILABLE,
    FLASH_ATTN_AVAILABLE,
    XFORMERS_AVAILABLE,
)

__all__ = [
    "RMSNorm",
    "SwiGLUMLP", 
    "RotaryEmbedding",
    "get_activation",
    "FUSED_KERNELS_AVAILABLE",
    "FLASH_ATTN_AVAILABLE",
    "XFORMERS_AVAILABLE",
]
