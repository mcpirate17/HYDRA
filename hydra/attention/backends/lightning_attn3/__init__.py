"""Lightning Attention 3 implementation (taxonomy + code).

This folder is the single canonical home for LightningAttn3 inside HYDRA.

- Low-level kernels and helpers live in subpackages (`ops/`, `utils/`).
- `is_available()` / `get_lightning_attn_func()` provide a stable accessor API.
"""

from __future__ import annotations

from typing import Any, Callable

import torch


def is_available() -> bool:
	if not torch.cuda.is_available():
		return False
	try:
		from .ops import lightning_attn_func

		return lightning_attn_func is not None
	except Exception:
		return False


def get_lightning_attn_func() -> Callable[..., Any]:
	if not torch.cuda.is_available():
		raise RuntimeError("LightningAttn3 requires CUDA")

	try:
		from .ops import lightning_attn_func
	except Exception as e:  # pragma: no cover
		raise RuntimeError("Failed to import hydra.attention.backends.lightning_attn3.ops") from e

	if lightning_attn_func is None:  # pragma: no cover
		raise RuntimeError("lightning_attn_func is unavailable")

	return lightning_attn_func


# Explicit imports for public API (replaces wildcard imports)
from .ops import (
    lightning_attn_func,
    validate_lightning_attn3,
    validate_at_init,
    LightningAttn3ValidationError,
)
from .utils import get_memory

__all__ = [
    "is_available",
    "get_lightning_attn_func",
    "lightning_attn_func",
    "validate_lightning_attn3",
    "validate_at_init",
    "LightningAttn3ValidationError",
    "get_memory",
]

