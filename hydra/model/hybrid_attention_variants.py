"""hydra.model.hybrid_attention_variants

Minimal attention variants for HYDRA.

This project is locked to two attention layers:
- CCQA (HYDRA native, implemented in `hydra.attention.ccqa.CCGQAAttention`)
- LLA3 (Lightning Attention-3, in `hydra.attention.backends.lightning_attn3`)

This module only provides the LLA3 adapter used by MoR blocks.
"""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import nullcontext
from typing import Optional, Union

import torch
import torch.nn as nn

from hydra.layers import RotaryEmbedding

from hydra.attention import is_hybrid_attention_backend_available

def _get_lightning_attn3_func():
    # NOTE: On CPU-only environments, importing Triton-backed kernels can raise
    # during driver initialization. Keep this module importable for unit tests.
    if not is_hybrid_attention_backend_available("lla3"):
        return None
    try:
        from hydra.attention.backends.lightning_attn3 import get_lightning_attn_func

        return get_lightning_attn_func()
    except Exception:
        return None

@dataclass(frozen=True)
class HybridAttentionChoice:
    """Resolved attention choice used by MoR blocks."""

    name: str


def resolve_hybrid_attention_choice(
    choice: Union[str, object, None],
    *,
    default: str,
) -> HybridAttentionChoice:
    if choice is None:
        return HybridAttentionChoice(default)

    if isinstance(choice, str):
        return HybridAttentionChoice(choice.strip().lower())

    # Allow Enum-like objects with a string `value`.
    value = getattr(choice, "value", None)
    if isinstance(value, str):
        return HybridAttentionChoice(value.strip().lower())

    return HybridAttentionChoice(default)


class LightningAttn3Attention(nn.Module):
    """Lightning Attention-3 adapter.

    This wraps the local `lightning_attn3` kernels so they can be used as a
    drop-in attention module inside HYDRA MoR blocks.

    Constraints:
    - Causal-only (HYDRA uses causal LM training). `mask` is not supported.
    - Expects per-head tensor shapes [B, H, S, D].
    - Supports GQA by projecting K/V with `n_kv_heads` and expanding.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        *,
        n_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        dropout: float = 0.0,
        use_rope: bool = True,
        max_seq_len: int = 8192,
        variant: str = "chunk_loop",
        te_fp8_projections: bool = False,
    ):
        super().__init__()

        self._lightning_attn3_func = _get_lightning_attn3_func()
        if self._lightning_attn3_func is None:
            raise ImportError(
                "lightning_attn3 not found. Check hydra/kernels/lightning_attn3/ exists."
            )

        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.head_dim = head_dim or (dim // n_heads)
        self.groups = n_heads // self.n_kv_heads
        self.dropout = nn.Dropout(float(dropout))
        self.use_rope = bool(use_rope)
        self.variant = str(variant)

        self._te_fp8_projections_enabled = False
        self._fp8_autocast = nullcontext

        LinearImpl: type[nn.Module] = nn.Linear
        if te_fp8_projections:
            try:
                from hydra.kernels.te_integration import (
                    FP8_AVAILABLE,
                    TE_AVAILABLE,
                    TELinear,
                    fp8_autocast,
                )

                if TE_AVAILABLE and FP8_AVAILABLE:
                    LinearImpl = TELinear
                    self._te_fp8_projections_enabled = True
                    self._fp8_autocast = fp8_autocast
            except Exception:
                # Best-effort: keep standard projections if TE is not usable.
                LinearImpl = nn.Linear

        assert dim % n_heads == 0
        assert n_heads % self.n_kv_heads == 0

        self.q_proj = LinearImpl(dim, n_heads * self.head_dim, bias=False)
        self.k_proj = LinearImpl(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = LinearImpl(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = LinearImpl(n_heads * self.head_dim, dim, bias=False)

        if self.use_rope:
            self.rope = RotaryEmbedding(self.head_dim, max_seq_len)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is not None:
            raise ValueError("LightningAttn3Attention does not support attn masks (causal-only).")

        b, s, _ = x.shape

        if self._te_fp8_projections_enabled:
            with self._fp8_autocast(enabled=True):
                q = self.q_proj(x).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
                k = self.k_proj(x).view(b, s, self.n_kv_heads, self.head_dim).transpose(1, 2)
                v = self.v_proj(x).view(b, s, self.n_kv_heads, self.head_dim).transpose(1, 2)
        else:
            q = self.q_proj(x).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(x).view(b, s, self.n_kv_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(b, s, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.use_rope:
            q = self.rope(q, s)
            k = self.rope(k, s)

        # Expand KV heads for GQA if needed (lightning_attn expects H to match).
        if self.n_kv_heads != self.n_heads:
            k = k.unsqueeze(2).expand(-1, -1, self.groups, -1, -1).reshape(
                b, self.n_heads, s, self.head_dim
            )
            v = v.unsqueeze(2).expand(-1, -1, self.groups, -1, -1).reshape(
                b, self.n_heads, s, self.head_dim
            )

        try:
            out = self._lightning_attn3_func(q, k, v, s=None, variant=self.variant)
        except AssertionError as e:
            # The kernel asserts on supported head/value dims.
            raise RuntimeError(
                f"lightning_attn3 rejected shapes: q={tuple(q.shape)} v={tuple(v.shape)}. "
                f"Try head_dim/value_dim divisible by 16. Original error: {e}"
            )

        out = out.transpose(1, 2).contiguous().view(b, s, self.n_heads * self.head_dim)
        out = self.dropout(out)

        if self._te_fp8_projections_enabled:
            with self._fp8_autocast(enabled=True):
                return self.o_proj(out)

        return self.o_proj(out)


# Backwards-compat alias: older code/tests used LightningAttn2Attention.
LightningAttn2Attention = LightningAttn3Attention


def build_hybrid_attention_module(
    choice: HybridAttentionChoice,
    *,
    dim: int,
    n_heads: int,
    n_kv_heads: int,
    max_seq_len: int,
    attention_kwargs: dict,
) -> nn.Module:
    """Factory for MoR block attention variants.

    Returns a module with signature `forward(x, mask=None) -> Tensor`.
    """

    name = choice.name

    # Default rope behavior: keep parity with the rest of HYDRA.
    use_rope = bool(attention_kwargs.get("use_rope", True))

    if name in ("lla2", "lla3"):  # Accept both for backwards compatibility
        return LightningAttn3Attention(
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            dropout=float(attention_kwargs.get("attn_dropout", 0.0)),
            use_rope=use_rope,
            max_seq_len=max_seq_len,
            variant=str(attention_kwargs.get("lla3_variant", "chunk_loop")),
            te_fp8_projections=bool(attention_kwargs.get("te_fp8_projections", False)),
        )

    raise ValueError(
        f"Unknown HYDRA attention variant '{name}'. Expected: lla3 (or lla2 for compat)."
    )
