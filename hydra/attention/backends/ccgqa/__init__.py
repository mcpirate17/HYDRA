"""CCQA attention backend component bundle.

This package is intentionally import-light to avoid circular imports with
`hydra.attention.ccqa`.
"""

from __future__ import annotations

from typing import Any


def build_ccqa_attention(**kwargs: Any):
	from .backend import build_ccqa_attention as _build

	return _build(**kwargs)


__all__ = ["build_ccqa_attention"]
