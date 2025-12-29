from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True, slots=True)
class HybridAttentionBackend:
    """Resolved hybrid-attention backend.

    Only CCGQA is supported. LA3 was removed due to gradient spike issues.
    """

    name: str


def is_hybrid_attention_backend_available(name: str) -> bool:
    """Check if an attention backend is available. Only CCGQA is supported."""
    n = str(name).strip().lower()
    return n in ("ccqa", "ccgqa")


def available_hybrid_attention_backends() -> tuple[str, ...]:
    """Return available attention backends. Only CCGQA is supported."""
    return ("ccqa",)


def resolve_hybrid_attention_backend(
    requested: Optional[str],
    *,
    default: str = "ccqa",
) -> HybridAttentionBackend:
    """Resolve a requested backend into a supported backend name.

    Only CCGQA is supported. LA3 was removed due to gradient spike issues.
    """
    if requested is None:
        return HybridAttentionBackend("ccqa")

    req = str(requested).strip().lower()

    # Back-compat aliases
    if req in ("ccqa", "ccgqa"):
        return HybridAttentionBackend("ccqa")

    # No silent fallback - error on unknown backend
    raise ValueError(
        f"Unknown attention backend '{requested}'. Only 'ccgqa' is supported."
    )
