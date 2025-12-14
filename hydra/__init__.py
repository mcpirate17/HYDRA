"""
HYDRA: Hybrid Dynamic Routing Architecture

A scalable transformer architecture combining:
- CCGQA: Compressed Convolutional Grouped Query Attention
- MoD: Mixture-of-Depths (token-level routing)
- MoR: Mixture-of-Recursions (layer-level depth adaptation)

Paper references:
- CCGQA: arXiv:2510.04476
- MoD: arXiv:2404.02258
- MoR: arXiv:2507.10524
"""

__version__ = "0.2.0"  # Updated with optimizations

from .model.ccgqa import (
    CCGQAAttention,
    CCGQABlock,
    CCGQAMoRBlock,
    CCGQAMoDBlockWrapper,
    CCGQAMoDMoRModel,
    create_ccgqa_mod_mor_model,
)

# Shared layers module (optimized, deduplicated)
from .layers import (
    RMSNorm,
    SwiGLUMLP,
    RotaryEmbedding,
    get_activation,
    FUSED_KERNELS_AVAILABLE,
    FLASH_ATTN_AVAILABLE,
    XFORMERS_AVAILABLE,
)

# Kernel control functions
from .kernels import (
    TRITON_AVAILABLE,
    get_kernel_status,
    set_use_triton_kernels,
)

# Optimization module (optional import - requires optuna)
try:
    from .optimization import (
        optimize_mlp_hyperparams,
        optimize_attention_hyperparams,
        optimize_norm_hyperparams,
        optimize_all_components,
    )

    _HAS_OPTIMIZATION = True
except ImportError:
    _HAS_OPTIMIZATION = False

__all__ = [
    # Model components
    "CCGQAAttention",
    "CCGQABlock",
    "CCGQAMoRBlock",
    "CCGQAMoDBlockWrapper",
    "CCGQAMoDMoRModel",
    "create_ccgqa_mod_mor_model",
    # Shared layers
    "RMSNorm",
    "SwiGLUMLP",
    "RotaryEmbedding",
    "get_activation",
    # Feature flags
    "FUSED_KERNELS_AVAILABLE",
    "FLASH_ATTN_AVAILABLE",
    "XFORMERS_AVAILABLE",
    "TRITON_AVAILABLE",
    # Kernel control
    "get_kernel_status",
    "set_use_triton_kernels",
]

if _HAS_OPTIMIZATION:
    __all__.extend(
        [
            "optimize_mlp_hyperparams",
            "optimize_attention_hyperparams",
            "optimize_norm_hyperparams",
            "optimize_all_components",
        ]
    )
