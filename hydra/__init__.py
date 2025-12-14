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

__version__ = "0.1.0"

from .model.ccgqa import (
    CCGQAAttention,
    CCGQABlock,
    CCGQAMoRBlock,
    CCGQAMoDBlockWrapper,
    CCGQAMoDMoRModel,
    create_ccgqa_mod_mor_model,
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
    "CCGQAAttention",
    "CCGQABlock",
    "CCGQAMoRBlock",
    "CCGQAMoDBlockWrapper",
    "CCGQAMoDMoRModel",
    "create_ccgqa_mod_mor_model",
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
