"""CPU microbenchmark for the CCQA attention backend.

This is intentionally lightweight and CPU-only: it checks basic forward execution
and reports a rough tokens/sec estimate.

Run:
  source /home/tim/venvs/llm/bin/activate && python -m hydra.attention.backends.ccgqa.benchmarks.benchmark_ccqa_cpu
"""

from __future__ import annotations

import time

import torch

from hydra.attention.backends.ccgqa.attention import CCGQAAttention


def main() -> None:
    torch.set_num_threads(max(1, torch.get_num_threads()))

    device = torch.device("cpu")
    dtype = torch.float32

    # Small-ish CPU-friendly config
    batch = 2
    seq = 128
    dim = 256
    n_heads = 8
    n_kv_heads = 2
    compression_factor = 2

    attn = CCGQAAttention(
        dim=dim,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        compression_factor=compression_factor,
        max_seq_len=seq,
    ).to(device=device, dtype=dtype)

    x = torch.randn(batch, seq, dim, device=device, dtype=dtype)

    # Warmup
    for _ in range(5):
        y = attn(x)
        assert y.shape == x.shape

    iters = 25
    t0 = time.perf_counter()
    for _ in range(iters):
        y = attn(x)
    t1 = time.perf_counter()

    elapsed = max(1e-9, t1 - t0)
    tokens = batch * seq * iters
    tps = tokens / elapsed

    print("CCQA CPU microbench")
    print(f"  shape: B={batch}, N={seq}, D={dim}")
    print(f"  iters: {iters}")
    print(f"  time:  {elapsed:.4f}s")
    print(f"  tok/s: {tps:,.0f}")


if __name__ == "__main__":
    main()
