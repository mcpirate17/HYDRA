"""
HYDRA Custom Triton Kernels

Fused GPU kernels for maximum performance on attention operations.
"""

from .fused_ops import (
    # Kernel functions
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
    "fused_rope",
    "fused_qk_norm", 
    "fused_swiglu",
    "fused_rms_norm",
    "TRITON_AVAILABLE",
    "USE_TRITON_KERNELS",
    "USE_FUSED_ROPE",
    "USE_FUSED_QK_NORM",
    "USE_FUSED_SWIGLU",
    "USE_FUSED_RMS_NORM",
    "set_use_triton_kernels",
    "get_kernel_status",
    "benchmark_kernels",
    "print_benchmark_results",
]
