"""Model framework package.

This package holds the HYDRA model wiring (blocks/models/factories) and routing
composition (MoD/MoR). The prior path `hydra.model.ccgqa` is deprecated; use
`hydra.model.framework` imports instead.

HYDRA uses Lightning Attention 3 (LA3) - an O(n) linear attention mechanism.
"""

from __future__ import annotations

# Attention implementation lives in hydra.attention.backends.ccgqa
from hydra.attention.backends.ccgqa.attention import CCGQAAttention

from .blocks import (
    HydraBlock,
    HydraBlockWithMoDMLP,
    HydraMoDBlockWrapper,
    HydraMoRBlock,
    MoDMLPWrapper,
    # Backward compat aliases
    CCGQABlock,
    CCGQABlockWithMoDMLP,
    CCGQAMoDBlockWrapper,
    CCGQAMoRBlock,
)
from .factory import (
    create_hydra_model,
    create_base_model,
    # Backward compat aliases
    create_ccgqa_model,
    create_ccgqa_mod_mor_model,
)
from .model import (
    HydraModel,
    HydraBaseModel,
    # Backward compat aliases
    CCGQAModel,
    CCGQAMoDMoRModel,
)

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
