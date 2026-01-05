"""CCGQA Attention Benchmarks

Available benchmarks:
- benchmark_attention: Comprehensive GPU/CPU benchmark comparing CCGQA vs GQA/MHA/Flash/SDPA
- benchmark_ccqa_cpu: Quick CPU-only micro-benchmark for regression testing  
- benchmark_ccqa_report: JSON report generator for CI/tracking

Usage:
    # Full attention benchmark with plots
    python -m hydra.attention.backends.ccgqa.benchmarks.benchmark_attention --save --plot
    
    # Quick CPU regression test
    python -m hydra.attention.backends.ccgqa.benchmarks.benchmark_ccqa_cpu
    
    # Generate JSON report
    python -m hydra.attention.backends.ccgqa.benchmarks.benchmark_ccqa_report --device cuda --out reports/ccqa.json

For full model benchmarks (HydraModel, HydraBaseModel, memory usage), see:
    python -m diagnostics.benchmark_hydra_models
"""

from .benchmark_attention import run_benchmark, print_summary_table, generate_plots
from .benchmark_ccqa_report import main as run_report

__all__ = ["run_benchmark", "print_summary_table", "generate_plots", "run_report"]
