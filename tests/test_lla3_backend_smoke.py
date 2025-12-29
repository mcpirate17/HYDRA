import pytest
import torch

from hydra.attention.registry import is_hybrid_attention_backend_available
from hydra.attention.factory import build_hybrid_attention_module


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_lla3_backend_forward_smoke_cuda():
    if not is_hybrid_attention_backend_available("lla3"):
        pytest.skip("LLA3 backend not available (CUDA kernels missing?)")

    device = torch.device("cuda")

    batch = 2
    seq = 128
    dim = 256
    n_heads = 8
    n_kv_heads = 8

    x = torch.randn(batch, seq, dim, device=device, dtype=torch.float16)

    attn = build_hybrid_attention_module(
        "lla3",
        dim=dim,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        max_seq_len=seq,
        compression_factor=1,
        attention_kwargs={
            "use_rope": True,
            "attn_dropout": 0.0,
            "lla3_variant": "chunk_loop",
        },
    ).to(device=device, dtype=torch.float16)

    y = attn(x)
    assert y.shape == x.shape
    assert y.dtype == x.dtype
