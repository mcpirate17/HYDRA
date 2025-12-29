"""Wrapper benchmark entry for the CCGQA CPU benchmark.

The canonical CPU benchmark lives in `diagnostics/benchmark_ccgqa_cpu.py`.

Run with:
    python -m hydra.model.framework.benchmarks.benchmark_ccgqa_cpu
"""

from __future__ import annotations

from diagnostics.benchmark_ccgqa_cpu import main


if __name__ == "__main__":
    main()
