"""
HYDRA Model Module

Contains the core transformer architectures:
- CCGQAModel: Base CCQA model
- CCGQAMoDMoRModel: Full efficiency stack (CCQA + MoD + MoR)
"""

from .ccgqa import (
    CCGQAAttention,
    CCGQABlock,
    CCGQABlockWithMoDMLP,
    CCGQAModel,
    CCGQAMoRBlock,
    CCGQAMoDMoRModel,
    CCGQAMoDBlockWrapper,
    MoDMLPWrapper,
    create_ccgqa_model,
    create_ccgqa_mod_mor_model,
)
from hydra.utils import save_model_architecture

__all__ = [
    # CCGQA
    "CCGQAAttention",
    "CCGQABlock",
    "CCGQABlockWithMoDMLP",
    "CCGQAModel",
    "CCGQAMoRBlock",
    "CCGQAMoDMoRModel",
    "CCGQAMoDBlockWrapper",
    "MoDMLPWrapper",
    "create_ccgqa_model",
    "create_ccgqa_mod_mor_model",
    "save_model_architecture",
]
