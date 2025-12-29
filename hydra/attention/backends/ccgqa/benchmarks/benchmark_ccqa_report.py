"""CCQA benchmark report generator.

Purpose:
- Kernel-centric benchmark entrypoint for CCQA (mirrors the tooling story of
    `hydra.attention.backends.lightning_attn3`, but for the pure-torch CCQA backend).
- Produces a small JSON report suitable for tracking perf over time.

Run (CPU):
    source /home/tim/venvs/llm/bin/activate && python -m hydra.attention.backends.ccgqa.benchmarks.benchmark_ccqa_report --device cpu

Run (CUDA, if available):
    source /home/tim/venvs/llm/bin/activate && python -m hydra.attention.backends.ccgqa.benchmarks.benchmark_ccqa_report --device cuda

Write JSON:
  ... --out reports/ccqa_bench.json
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from hydra.attention.backends.ccgqa.attention import CCGQAAttention


@dataclass(frozen=True)
class Case:
    batch: int
    seq: int
    dim: int
    n_heads: int
    n_kv_heads: int
    compression_factor: int
    dtype: str


def _resolve_dtype(name: str) -> torch.dtype:
    name = name.strip().lower()
    if name in {"fp32", "float32"}:
        return torch.float32
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    raise ValueError(f"Unsupported dtype '{name}'")


def _bench_one(
    case: Case,
    *,
    device: torch.device,
    warmup: int,
    iters: int,
) -> Dict[str, Any]:
    dtype = _resolve_dtype(case.dtype)

    attn = CCGQAAttention(
        dim=case.dim,
        n_heads=case.n_heads,
        n_kv_heads=case.n_kv_heads,
        compression_factor=case.compression_factor,
        max_seq_len=case.seq,
    ).to(device=device, dtype=dtype)

    x = torch.randn(case.batch, case.seq, case.dim, device=device, dtype=dtype)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Warmup
    for _ in range(warmup):
        y = attn(x)
        if y.shape != x.shape:
            raise RuntimeError(f"bad output shape: got {tuple(y.shape)}, expected {tuple(x.shape)}")

    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        y = attn(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    elapsed = max(1e-9, t1 - t0)
    tokens = case.batch * case.seq * iters
    tok_s = tokens / elapsed

    result: Dict[str, Any] = {
        "case": asdict(case),
        "device": str(device),
        "elapsed_s": elapsed,
        "iters": iters,
        "tokens": tokens,
        "tok_s": tok_s,
    }

    if device.type == "cuda":
        # Best-effort memory snapshot.
        result["cuda_max_mem_bytes"] = int(torch.cuda.max_memory_allocated())

    return result


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=25)
    p.add_argument("--out", type=str, default="")
    args = p.parse_args(argv)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but torch.cuda.is_available() is False")

    if device.type == "cpu":
        torch.set_num_threads(max(1, torch.get_num_threads()))
    else:
        torch.cuda.reset_peak_memory_stats()

    # Keep this small and stable: enough to track regressions.
    cases = [
        Case(batch=2, seq=128, dim=256, n_heads=8, n_kv_heads=2, compression_factor=2, dtype="fp32"),
        Case(batch=2, seq=256, dim=512, n_heads=8, n_kv_heads=2, compression_factor=4, dtype="fp32"),
    ]

    if device.type == "cuda":
        # Include a mixed-precision-ish case when CUDA is available.
        cases.append(
            Case(batch=4, seq=512, dim=1024, n_heads=16, n_kv_heads=4, compression_factor=4, dtype="bf16")
        )

    results: List[Dict[str, Any]] = []
    for case in cases:
        results.append(_bench_one(case, device=device, warmup=args.warmup, iters=args.iters))

    report: Dict[str, Any] = {
        "name": "ccqa",
        "device": str(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda if hasattr(torch.version, "cuda") else None,
        "cases": results,
    }

    print(json.dumps(report, indent=2))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

    
