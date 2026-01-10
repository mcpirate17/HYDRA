"""
HYDRA Custom Triton Kernels

Fused GPU kernels for maximum performance on attention operations.

Performance Hierarchy (best to worst):
1. Transformer Engine FP8 (Hopper+ only) - Maximum TFLOPS
2. Liger Kernels - Best memory savings, great speed
3. Flash Attention 3 - Best for long sequences
4. HYDRA Triton Kernels - Good baseline with custom ops
"""

try:
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
except Exception:
    # CPU-only environments (or missing GPU drivers) can raise during Triton
    # driver initialization at import time. Keep the package importable so
    # unit tests and CPU workflows run without requiring a Triton backend.
    fused_rope = None
    fused_qk_norm = None
    fused_swiglu = None
    fused_rms_norm = None

    chunked_cross_entropy = None
    fused_chunked_cross_entropy = None
    USE_CHUNKED_CROSS_ENTROPY = False
    CROSS_ENTROPY_CHUNK_SIZE = 0

    TRITON_AVAILABLE = False
    USE_TRITON_KERNELS = False
    USE_FUSED_ROPE = False
    USE_FUSED_QK_NORM = False
    USE_FUSED_SWIGLU = False
    USE_FUSED_RMS_NORM = False

    def set_use_triton_kernels(_: bool) -> None:
        return None

    def get_kernel_status() -> dict:
        return {
            "triton_available": False,
            "use_triton_kernels": False,
            "use_fused_rope": False,
            "use_fused_qk_norm": False,
            "use_fused_swiglu": False,
            "use_fused_rms_norm": False,
            "use_chunked_cross_entropy": False,
            "cross_entropy_chunk_size": 0,
        }

    def benchmark_kernels(*args, **kwargs):
        raise RuntimeError("Triton kernels unavailable in this environment")

    def print_benchmark_results(*args, **kwargs) -> None:
        return None

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
try:
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
except Exception:
    TE_AVAILABLE = False
    FP8_AVAILABLE = False

    def get_te_status() -> dict:
        return {"te_available": False, "fp8_available": False}

    def fp8_autocast(*args, **kwargs):
        raise RuntimeError("Transformer Engine unavailable in this environment")

    TELinear = None
    TELayerNorm = None
    TERMSNorm = None

    def patch_model_with_te(*args, **kwargs):
        return None

    def print_te_status(*args, **kwargs) -> None:
        return None

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
    "apply_liger_kernel_to_model",
    "LigerCrossEntropyLoss",
    "LigerFusedLinearCrossEntropyLoss",
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
