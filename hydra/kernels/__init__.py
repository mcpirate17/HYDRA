"""
HYDRA Custom Triton Kernels

Fused GPU kernels for maximum performance on attention operations.

Performance Hierarchy (best to worst):
1. Transformer Engine FP8 (Hopper+ only) - Maximum TFLOPS
2. Liger Kernels - Best memory savings, great speed
3. Flash Attention 3 - Best for long sequences
4. HYDRA Triton Kernels - Good baseline with custom ops
"""

from .fused_ops import (
    # Kernel functions
    fused_rope,
    fused_qk_norm,
    fused_swiglu,
    fused_rms_norm,
    # Chunked cross-entropy (major memory optimization)
    chunked_cross_entropy,
    fused_chunked_cross_entropy,
    USE_CHUNKED_CROSS_ENTROPY,
    CROSS_ENTROPY_CHUNK_SIZE,
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

# Liger Kernel integration (LinkedIn's fused BF16 kernels)
from .liger_integration import (
    LIGER_AVAILABLE,
    LIGER_ENABLED,
    get_liger_status,
    liger_rms_norm,
    liger_swiglu_forward,
    liger_cross_entropy_loss,
    liger_fused_linear_cross_entropy,
    liger_rope,
    patch_hydra_with_liger,
    apply_liger_kernel_to_model,
    LigerCrossEntropyLoss,
    LigerFusedLinearCrossEntropyLoss,
)

# Transformer Engine integration (NVIDIA FP8)
from .te_integration import (
    TE_AVAILABLE,
    FP8_AVAILABLE,
    get_te_status,
    fp8_autocast,
    TELinear,
    TELayerNorm,
    TERMSNorm,
    patch_model_with_te,
    print_te_status,
)

# Unified performance configuration
from .perf_config import (
    PerfConfig,
    detect_hardware,
    get_optimal_config,
    configure_max_performance,
    print_perf_status,
    get_compile_config,
)

__all__ = [
    # HYDRA Triton kernels
    "fused_rope",
    "fused_qk_norm", 
    "fused_swiglu",
    "fused_rms_norm",
    # Chunked cross-entropy
    "chunked_cross_entropy",
    "fused_chunked_cross_entropy",
    "USE_CHUNKED_CROSS_ENTROPY",
    "CROSS_ENTROPY_CHUNK_SIZE",
    # Feature flags
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
    # Liger integration
    "LIGER_AVAILABLE",
    "LIGER_ENABLED",
    "get_liger_status",
    "liger_rms_norm",
    "liger_swiglu_forward",
    "liger_cross_entropy_loss",
    "liger_fused_linear_cross_entropy",
    "liger_rope",
    "patch_hydra_with_liger",
    # Transformer Engine integration
    "TE_AVAILABLE",
    "FP8_AVAILABLE",
    "get_te_status",
    "fp8_autocast",
    "TELinear",
    "TELayerNorm",
    "TERMSNorm",
    "patch_model_with_te",
    "print_te_status",
    # Unified performance configuration
    "PerfConfig",
    "detect_hardware",
    "get_optimal_config",
    "configure_max_performance",
    "print_perf_status",
    "get_compile_config",
]
