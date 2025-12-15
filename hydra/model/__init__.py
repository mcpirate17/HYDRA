"""
HYDRA Model Module

Contains the core transformer architectures:
- CCGQAModel: Base CCGQA model
- CCGQAMoDMoRModel: Full efficiency stack (CCGQA + MoD + MoR)
- Hybrid attention variants (MQA, MLA, CCQA)
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
    save_model_architecture,
)

from .hybrid_attention import (
    AttentionType,
    MQAAttention,
    MLAAttention,
    CCQAAttention,
    HybridAttentionBlock,
)

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
    # Hybrid attention
    "AttentionType",
    "MQAAttention",
    "MLAAttention", 
    "CCQAAttention",
    "HybridAttentionBlock",
]
