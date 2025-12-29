import torch

from hydra.attention import (
    available_hybrid_attention_backends,
    is_hybrid_attention_backend_available,
    resolve_hybrid_attention_backend,
)


def test_registry_always_has_ccqa():
    backends = available_hybrid_attention_backends()
    assert "ccqa" in backends


def test_registry_lla3_availability_matches_cuda():
    # On CPU-only machines, lla3 should not be considered available.
    if not torch.cuda.is_available():
        assert is_hybrid_attention_backend_available("lla3") is False


def test_resolve_falls_back_to_ccqa_on_cpu():
    if not torch.cuda.is_available():
        resolved = resolve_hybrid_attention_backend("lla3", default="ccqa")
        assert resolved.name == "ccqa"


def test_resolve_aliases():
    resolved = resolve_hybrid_attention_backend("ccgqa", default="ccqa")
    assert resolved.name == "ccqa"
