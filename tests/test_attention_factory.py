import torch

from hydra.attention.factory import build_hybrid_attention_module
from hydra.attention.backends.ccgqa.attention import CCGQAAttention


def test_factory_falls_back_to_ccqa_on_cpu():
    if torch.cuda.is_available():
        return

    mod = build_hybrid_attention_module(
        "lla3",
        dim=64,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=128,
        compression_factor=2,
        attention_kwargs={"lla3_variant": "chunk_loop"},
    )
    assert isinstance(mod, CCGQAAttention)
