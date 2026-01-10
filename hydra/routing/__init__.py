"""HYDRA routing components (MoD, MoR, MoE).

This package provides:
- Mixture of Depths (MoD): Dynamic layer skipping via top-k token selection
- Mixture of Recursions (MoR): Adaptive depth routing per token
- Mixture of Experts (MoE): Sparse FFN routing to expert networks
- Loss Tracker: Moving average baseline for loss-driven routing

Usage:
    from hydra.routing import (
        # MoD
        MoDConfig, MoDRouter, MixtureOfDepthsBlock,
        # MoR  
        MoRConfig, MoRRouter, MoRExecutor,
        # MoE
        MoEConfig, MoERouter, MoEFFNBlock,
        # Loss tracking
        MovingAverageBaseline,
    )
"""

# Mixture of Depths
from .mixture_of_depths import (
    MoDConfig,
    MoDRouter,
    MixtureOfDepthsBlock,
    MoDTransformerLayer,
    MoDAttention,
)

# Mixture of Recursions
from .mixture_of_recursions import (
    MoRConfig,
    MoRRouter,
    MoRExecutor,
    MoROutput,
    dim_to_depth_scale,
    compute_layer_target_prob,
)

# Mixture of Experts
from .mixture_of_experts import (
    MoEConfig,
    MoERouter,
    MoEDispatcher,
    get_moe_scaling,
    compute_moe_placement,
)

from .moe_block import (
    MoEFFNBlock,
    MoEExpertMLP,
    MoELayerWrapper,
)

# Loss tracking for loss-driven routing
from .loss_tracker import (
    MovingAverageBaseline,
    AdvantageScaledSTE,
    apply_advantage_ste,
)

# Shared operations (advanced usage)
from .ops import (
    soft_clamp_logits,
    ste_round,
    compute_exit_masks,
    gather_by_mask,
    scatter_by_indices,
    compute_ste_weights,
)

__all__ = [
    # MoD
    "MoDConfig",
    "MoDRouter",
    "MixtureOfDepthsBlock",
    "MoDTransformerLayer",
    "MoDAttention",
    # MoR
    "MoRConfig",
    "MoRRouter",
    "MoRExecutor",
    "MoROutput",
    "dim_to_depth_scale",
    "compute_layer_target_prob",
    # MoE
    "MoEConfig",
    "MoERouter",
    "MoEDispatcher",
    "MoEFFNBlock",
    "MoEExpertMLP",
    "MoELayerWrapper",
    "get_moe_scaling",
    "compute_moe_placement",
    # Loss tracking
    "MovingAverageBaseline",
    "AdvantageScaledSTE",
    "apply_advantage_ste",
    # Ops
    "soft_clamp_logits",
    "ste_round",
    "compute_exit_masks",
    "gather_by_mask",
    "scatter_by_indices",
    "compute_ste_weights",
]
