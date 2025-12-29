from __future__ import annotations

import os
from typing import Any

import torch.nn as nn


def build_hybrid_attention_module(
    requested: str,
    *,
    dim: int,
    n_heads: int,
    n_kv_heads: int,
    max_seq_len: int,
    compression_factor: int,
    attention_kwargs: dict[str, Any],
) -> nn.Module:
    """Instantiate CCGQA attention module.

    Only CCGQA is supported. LA3 was removed due to gradient spike issues.
    """
    # Normalize requested backend name
    req = str(requested).strip().lower()
    if req not in ("ccqa", "ccgqa"):
        raise ValueError(
            f"Unknown attention backend '{requested}'. Only 'ccgqa' is supported."
        )

    from hydra.attention.backends.ccgqa.attention import CCGQAAttention

    # Only forward kwargs recognized by CCGQAAttention
    allowed_ccqa_keys = {
        "use_rope",
        "use_qk_norm",
        "use_convs",
        "use_qk_mean",
        "use_value_shift",
        "conv_kernel_size",
        "use_fused_kernel",
    }
    ccqa_kwargs = {k: v for k, v in attention_kwargs.items() if k in allowed_ccqa_keys}

    env_ccqa_fused = os.environ.get("HYDRA_CCQA_USE_FUSED_KERNEL", "").strip().lower()
    if env_ccqa_fused in {"1", "true", "yes", "y"}:
        ccqa_kwargs["use_fused_kernel"] = True
    elif env_ccqa_fused in {"0", "false", "no", "n"}:
        ccqa_kwargs["use_fused_kernel"] = False

    return CCGQAAttention(
        dim=dim,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        compression_factor=compression_factor,
        max_seq_len=max_seq_len,
        **ccqa_kwargs,
    )
