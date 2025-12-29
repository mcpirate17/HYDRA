"""Wrapper benchmark entry for the CCGQA suite.

The canonical benchmarks live in `diagnostics/benchmark_ccgqa.py`.
This wrapper exists to keep benchmarks discoverable next to the CCGQA
implementation.

Run with:
    python -m hydra.model.framework.benchmarks.benchmark_ccgqa
"""

from __future__ import annotations

from diagnostics.benchmark_ccgqa import main


if __name__ == "__main__":
    main()
