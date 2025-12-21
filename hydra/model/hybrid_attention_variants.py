"""hydra.model.hybrid_attention_variants

Minimal attention variants for HYDRA.

This project is locked to two attention layers:
- CCQA (HYDRA native, implemented in `hydra.model.ccgqa.CCGQAAttention`)
- LLA3 (Lightning Attention-3, local fork in `hydra.kernels.lightning_attn3`)

This module only provides the LLA3 adapter used by MoR blocks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn

from hydra.layers import RotaryEmbedding

# Local Lightning Attention-3 with Blackwell compatibility and chunked backward.
# See hydra/kernels/lightning_attn3/README.md for benchmarks.
from hydra.kernels.lightning_attn3.ops import lightning_attn_func as _lightning_attn3_func

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
    ):
        super().__init__()

        if _lightning_attn3_func is None:
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

        assert dim % n_heads == 0
        assert n_heads % self.n_kv_heads == 0

        self.q_proj = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, dim, bias=False)

        if self.use_rope:
            self.rope = RotaryEmbedding(self.head_dim, max_seq_len)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is not None:
            raise ValueError("LightningAttn3Attention does not support attn masks (causal-only).")

        b, s, _ = x.shape

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
            out = _lightning_attn3_func(q, k, v, s=None, variant=self.variant)
        except AssertionError as e:
            # The kernel asserts on supported head/value dims.
            raise RuntimeError(
                f"lightning_attn3 rejected shapes: q={tuple(q.shape)} v={tuple(v.shape)}. "
                f"Try head_dim/value_dim divisible by 16. Original error: {e}"
            )

        out = out.transpose(1, 2).contiguous().view(b, s, self.n_heads * self.head_dim)
        out = self.dropout(out)
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
        )

    raise ValueError(
        f"Unknown HYDRA attention variant '{name}'. Expected: lla3 (or lla2 for compat)."
    )
