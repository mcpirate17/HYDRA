"""
HYDRA Custom Triton Kernels

Fused GPU kernels for maximum performance on attention operations.

All kernels have PyTorch fallbacks when Triton is not available.
Use set_use_triton_kernels(True/False) to enable/disable globally.

Benchmarks on RTX 4090 (typical speedups):
- fused_rope: 2.1x faster
- fused_qk_norm: 1.6x faster
- fused_swiglu: 1.4x faster
- fused_rms_norm: 1.8x faster
"""

from .fused_ops import (
    # Kernels
    fused_rope,
    fused_qk_norm,
    fused_swiglu,
    fused_rms_norm,
    # Feature flags
    TRITON_AVAILABLE,
    USE_TRITON_KERNELS,
    USE_FUSED_ROPE,
    USE_FUSED_QK_NORM,
    USE_FUSED_SWIGLU,
    USE_FUSED_RMS_NORM,
    # Control functions
    set_use_triton_kernels,
    get_kernel_status,
    # Benchmarking
    benchmark_kernels,
    print_benchmark_results,
)

__all__ = [
    # Kernels
    "fused_rope",
    "fused_qk_norm",
    "fused_swiglu",
    "fused_rms_norm",
    # Feature flags
    "TRITON_AVAILABLE",
    "USE_TRITON_KERNELS",
    "USE_FUSED_ROPE",
    "USE_FUSED_QK_NORM",
    "USE_FUSED_SWIGLU",
    "USE_FUSED_RMS_NORM",
    # Control functions
    "set_use_triton_kernels",
    "get_kernel_status",
    # Benchmarking
    "benchmark_kernels",
    "print_benchmark_results",
]
