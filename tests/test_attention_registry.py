import pytest

from hydra.attention import (
    available_hybrid_attention_backends,
    is_hybrid_attention_backend_available,
    resolve_hybrid_attention_backend,
)


def test_registry_always_has_ccqa():
    """Test that CCQA is always available as a backend."""
    backends = available_hybrid_attention_backends()
    assert "ccqa" in backends


def test_registry_only_supports_ccqa():
    """Test that only CCGQA/CCQA variants are supported."""
    assert is_hybrid_attention_backend_available("ccqa") is True
    assert is_hybrid_attention_backend_available("ccgqa") is True
    assert is_hybrid_attention_backend_available("unknown") is False


def test_resolve_returns_ccqa():
    """Test that resolve returns CCQA for valid aliases."""
    resolved = resolve_hybrid_attention_backend("ccgqa", default="ccqa")
    assert resolved.name == "ccqa"

    resolved = resolve_hybrid_attention_backend("ccqa", default="ccqa")
    assert resolved.name == "ccqa"


def test_resolve_rejects_unknown():
    """Test that resolve raises ValueError for unknown backends."""
    with pytest.raises(ValueError, match="Unknown attention backend"):
        resolve_hybrid_attention_backend("unknown_backend", default="ccqa")
