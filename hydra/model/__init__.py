"""
HYDRA Model Module

Contains the core transformer architectures:
- HydraModel: Full efficiency stack (LA3 + MoD + MoR)
- HydraBaseModel: Base model without routing

HYDRA uses Lightning Attention 3 (LA3) - an O(n) linear attention mechanism.
"""

from .framework import (
    # Canonical exports (use these)
    HydraModel,
    HydraBaseModel,
    HydraBlock,
    HydraBlockWithMoDMLP,
    HydraMoRBlock,
    HydraMoDBlockWrapper,
    MoDMLPWrapper,
    create_hydra_model,
    create_base_model,
    # Attention backend (still called CCGQA)
    CCGQAAttention,
    # Backward compat aliases
    CCGQABlock,
    CCGQABlockWithMoDMLP,
    CCGQAModel,
    CCGQAMoRBlock,
    CCGQAMoDMoRModel,
    CCGQAMoDBlockWrapper,
    create_ccgqa_model,
    create_ccgqa_mod_mor_model,
)
from hydra.utils import save_model_architecture

__all__ = [
    # Canonical exports (use these)
    "HydraModel",
    "HydraBaseModel",
    "HydraBlock",
    "HydraBlockWithMoDMLP",
    "HydraMoRBlock",
    "HydraMoDBlockWrapper",
    "MoDMLPWrapper",
    "create_hydra_model",
    "create_base_model",
    "save_model_architecture",
    # Attention backend (still called CCGQA)
    "CCGQAAttention",
    # Backward compat aliases
    "CCGQABlock",
    "CCGQABlockWithMoDMLP",
    "CCGQAModel",
    "CCGQAMoRBlock",
    "CCGQAMoDMoRModel",
    "CCGQAMoDBlockWrapper",
    "create_ccgqa_model",
    "create_ccgqa_mod_mor_model",
]
