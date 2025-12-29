"""CPU-only micro-benchmark for CCGQAAttention.

Goal: quick regression signal for allocation-heavy refactors.
This is NOT meant to represent GPU throughput.

Usage:
  source /home/tim/venvs/llm/bin/activate && \
    CUDA_VISIBLE_DEVICES= python diagnostics/benchmark_ccgqa_cpu.py
"""

from __future__ import annotations

import os
import sys
import time

import torch

# Allow running as a script: `python diagnostics/benchmark_ccgqa_cpu.py`
_ROOT = os.path.dirname(os.path.dirname(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from hydra.attention import CCGQAAttention


def bench_once(*, batch: int, seq: int, dim: int, n_heads: int, n_kv_heads: int, compression_factor: int, iters: int) -> None:
    attn = CCGQAAttention(
        dim=dim,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        compression_factor=compression_factor,
        max_seq_len=max(2 * seq, 128),
        use_rope=True,
        use_qk_norm=True,
        use_convs=True,
        use_qk_mean=True,
        use_value_shift=True,
    ).cpu()
    attn.eval()

    x = torch.randn(batch, seq, dim, dtype=torch.float32)

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = attn(x)

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(iters):
            _ = attn(x)
    t1 = time.perf_counter()

    dt = (t1 - t0) / iters
    toks_per_s = (batch * seq) / dt
    print(
        f"B={batch} S={seq} dim={dim} heads={n_heads}/{n_kv_heads} C={compression_factor} | "
        f"{dt*1e3:.3f} ms/iter | {toks_per_s:,.0f} tokens/s"
    )


def main() -> None:
    torch.manual_seed(0)

    print(f"torch={torch.__version__}")
    print(f"num_threads={torch.get_num_threads()}")

    # Keep shapes modest to finish quickly on CPU.
    bench_once(batch=2, seq=128, dim=512, n_heads=8, n_kv_heads=2, compression_factor=4, iters=50)
    bench_once(batch=2, seq=256, dim=512, n_heads=8, n_kv_heads=2, compression_factor=4, iters=25)


if __name__ == "__main__":
    main()
