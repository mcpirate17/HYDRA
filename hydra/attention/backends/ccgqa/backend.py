from __future__ import annotations

from typing import Any

from hydra.attention.backends.ccgqa.attention import CCGQAAttention


def build_ccqa_attention(**kwargs: Any) -> CCGQAAttention:
    """Standard constructor for the CCQA backend.

    Notes:
    - Runtime selection is handled by `hydra.attention.registry`.
    - `hydra.attention.factory.build_hybrid_attention_module(...)` remains the
      primary entrypoint for model wiring.
    """

    return CCGQAAttention(**kwargs)
