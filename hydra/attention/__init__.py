"""HYDRA attention backends.

This package is the future home for a consistent backend interface that covers
HYDRA's "hybrid attention" choices (e.g., CCQA vs LLA3).

Phase 3.1 (scaffold): provides a small registry-style API that can be adopted
incrementally by the model code without changing behavior.
"""

from .registry import (
    HybridAttentionBackend,
    available_hybrid_attention_backends,
    is_hybrid_attention_backend_available,
    resolve_hybrid_attention_backend,
)
from .backends.ccgqa.attention import CCGQAAttention
from .factory import build_hybrid_attention_module

__all__ = [
    "HybridAttentionBackend",
    "available_hybrid_attention_backends",
    "is_hybrid_attention_backend_available",
    "resolve_hybrid_attention_backend",
    "CCGQAAttention",
    "build_hybrid_attention_module",
]
