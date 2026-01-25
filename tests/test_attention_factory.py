import pytest
import torch

from hydra.attention.factory import build_hybrid_attention_module
from hydra.attention.backends.ccgqa.attention import CCGQAAttention


def test_factory_builds_ccqa_by_name():
    """Test that factory correctly builds CCGQAAttention for recognized names."""
    mod = build_hybrid_attention_module(
        "ccgqa",
        dim=64,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=128,
        compression_factor=2,
        attention_kwargs={},
    )
    assert isinstance(mod, CCGQAAttention)


def test_factory_rejects_unknown_backend():
    """Test that factory raises ValueError for unknown backend names."""
    with pytest.raises(ValueError, match="Unknown attention backend"):
        build_hybrid_attention_module(
            "unknown_backend",
            dim=64,
            n_heads=4,
            n_kv_heads=2,
            max_seq_len=128,
            compression_factor=2,
            attention_kwargs={},
        )
