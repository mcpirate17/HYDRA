import importlib


def test_ccqa_backend_component_bundle_imports():
    mod = importlib.import_module("hydra.attention.backends.ccgqa")
    assert hasattr(mod, "build_ccqa_attention")


def test_ccqa_backend_standard_builder_returns_module():
    from hydra.attention.backends.ccgqa import build_ccqa_attention

    attn = build_ccqa_attention(
        dim=64,
        n_heads=4,
        n_kv_heads=2,
        compression_factor=2,
        max_seq_len=128,
    )

    # Smoke check forward on CPU
    import torch

    x = torch.randn(2, 16, 64)
    y = attn(x)
    assert y.shape == x.shape
