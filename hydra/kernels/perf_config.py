"""
HYDRA Performance Configuration

Unified configuration for all performance optimizations.
Automatically detects available backends and configures the optimal settings.

Usage:
    from hydra.kernels.perf_config import configure_max_performance, print_perf_status
    
    # Auto-configure for best available performance
    configure_max_performance()
    
    # Check what's enabled
    print_perf_status()

Environment Variables:
    HYDRA_PERF_PROFILE: "max", "balanced", "memory_saver", "debug"
    HYDRA_USE_LIGER: "1" to enable Liger kernels
    HYDRA_USE_TE: "1" to enable Transformer Engine
    HYDRA_COMPILE_MODE: "max-autotune", "reduce-overhead", "default"
"""

import os
from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch


@dataclass
class PerfConfig:
    """Performance configuration container."""
    
    # Compile settings
    compile_enabled: bool = True
    compile_mode: str = "max-autotune-no-cudagraphs"
    compile_fullgraph: bool = False  # True breaks with dynamic shapes
    compile_dynamic: bool = True
    
    # AMP settings
    amp_enabled: bool = True
    amp_dtype: torch.dtype = torch.bfloat16
    
    # Kernel selection (priority order)
    use_te_fp8: bool = False  # Transformer Engine FP8 (Hopper+)
    use_liger: bool = False   # Liger fused kernels
    use_flash_attn: bool = True  # Flash Attention
    use_triton: bool = True   # HYDRA Triton kernels
    
    # Memory optimizations
    use_chunked_ce: bool = True
    ce_chunk_size: int = 4096
    gradient_checkpointing: bool = True
    checkpoint_every_n: int = 2
    
    # Backend settings
    tf32_enabled: bool = True
    cudnn_benchmark: bool = True
    
    # Dynamo settings
    dynamo_cache_size: int = 64
    dynamo_suppress_errors: bool = False
    
    def apply(self):
        """Apply this configuration to PyTorch."""
        # TF32 settings (Ampere+)
        torch.backends.cuda.matmul.allow_tf32 = self.tf32_enabled
        torch.backends.cudnn.allow_tf32 = self.tf32_enabled
        torch.backends.cudnn.benchmark = self.cudnn_benchmark
        
        if self.tf32_enabled:
            torch.set_float32_matmul_precision('high')
        
        # Dynamo settings
        torch._dynamo.config.cache_size_limit = self.dynamo_cache_size
        torch._dynamo.config.suppress_errors = self.dynamo_suppress_errors
        
        # Enable kernel selection
        if self.use_triton:
            from hydra.kernels import set_use_triton_kernels
            set_use_triton_kernels(True)


def detect_hardware() -> Dict[str, Any]:
    """Detect hardware capabilities for optimization decisions."""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": None,
        "gpu_name": None,
        "compute_capability": None,
        "gpu_memory_gb": None,
        "supports_bf16": False,
        "supports_fp8": False,
        "supports_flash_attn": False,
    }
    
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_name"] = torch.cuda.get_device_name()
        cc = torch.cuda.get_device_capability()
        info["compute_capability"] = cc
        info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        # BF16: Ampere+ (sm_80)
        info["supports_bf16"] = cc[0] >= 8
        
        # FP8: Hopper+ (sm_90)
        info["supports_fp8"] = cc[0] >= 9
        
        # Flash Attention: Ampere+ with good perf
        info["supports_flash_attn"] = cc[0] >= 8
    
    return info


def get_optimal_config(profile: str = "auto") -> PerfConfig:
    """Get optimal performance configuration for the current hardware.
    
    Profiles:
        "auto": Automatically select best settings
        "max": Maximum performance (may use more memory)
        "balanced": Good performance with reasonable memory usage
        "memory_saver": Minimize memory usage
        "debug": Disable optimizations for debugging
    """
    hw = detect_hardware()
    config = PerfConfig()
    
    if profile == "debug":
        config.compile_enabled = False
        config.amp_enabled = False
        config.use_te_fp8 = False
        config.use_liger = False
        config.use_triton = False
        config.gradient_checkpointing = False
        return config
    
    # Memory saver profile
    if profile == "memory_saver":
        config.compile_mode = "reduce-overhead"
        config.use_chunked_ce = True
        config.ce_chunk_size = 2048  # Smaller chunks
        config.gradient_checkpointing = True
        config.checkpoint_every_n = 1  # Every layer
        config.use_liger = True  # Liger is great for memory
        return config
    
    # Auto or max profile - detect best settings
    if hw["supports_fp8"] and profile == "max":
        # Hopper+: Use FP8 for max throughput
        config.use_te_fp8 = True
        config.use_liger = True  # Liger for non-FP8 ops
    elif hw["supports_bf16"]:
        # Ampere+: Use Liger BF16 kernels
        config.use_liger = True
        config.use_te_fp8 = False
    
    # Flash Attention for all Ampere+ GPUs
    config.use_flash_attn = hw["supports_flash_attn"]
    
    # Compile settings based on profile
    if profile == "max":
        config.compile_mode = "max-autotune-no-cudagraphs"
        config.gradient_checkpointing = False  # Trade memory for speed
    else:  # balanced or auto
        config.compile_mode = "max-autotune-no-cudagraphs"
        config.gradient_checkpointing = True
    
    return config


def configure_max_performance(profile: str = "auto") -> PerfConfig:
    """Configure PyTorch for maximum performance.
    
    Call this before creating your model!
    
    Args:
        profile: "auto", "max", "balanced", "memory_saver", "debug"
    
    Returns:
        The applied configuration
    """
    # Check environment override
    env_profile = os.environ.get("HYDRA_PERF_PROFILE")
    if env_profile:
        profile = env_profile
    
    config = get_optimal_config(profile)
    config.apply()
    
    # Apply Liger patches if enabled
    if config.use_liger:
        from hydra.kernels.liger_integration import patch_hydra_with_liger
        patch_hydra_with_liger()
    
    return config


def print_perf_status():
    """Print comprehensive performance status."""
    from hydra.kernels import (
        TRITON_AVAILABLE, USE_TRITON_KERNELS,
        get_kernel_status,
    )
    from hydra.kernels.liger_integration import get_liger_status, LIGER_AVAILABLE
    from hydra.kernels.te_integration import get_te_status, TE_AVAILABLE
    from hydra.layers import FLASH_ATTN_AVAILABLE
    
    hw = detect_hardware()
    
    print("\n" + "=" * 70)
    print("🚀 HYDRA PERFORMANCE STATUS")
    print("=" * 70)
    
    # Hardware
    print("\n📟 HARDWARE:")
    print(f"   GPU: {hw.get('gpu_name', 'N/A')}")
    print(f"   Compute Capability: {hw.get('compute_capability', 'N/A')}")
    print(f"   Memory: {hw.get('gpu_memory_gb', 0):.1f} GB")
    print(f"   BF16 Support: {'✅' if hw.get('supports_bf16') else '❌'}")
    print(f"   FP8 Support: {'✅' if hw.get('supports_fp8') else '❌'}")
    
    # Kernel Status
    print("\n⚡ KERNEL BACKENDS:")
    
    # Transformer Engine
    te_status = get_te_status() if TE_AVAILABLE else {}
    print(f"   Transformer Engine: {'✅ v' + te_status.get('te_version', '?') if TE_AVAILABLE else '❌ Not installed'}")
    if TE_AVAILABLE:
        print(f"      FP8 Enabled: {'✅' if te_status.get('fp8_available') else '❌ (need Hopper+)'}")
    
    # Liger
    liger_status = get_liger_status()
    print(f"   Liger Kernels: {'✅' if liger_status.get('liger_available') else '❌ Not installed'}")
    if liger_status.get('liger_available'):
        print(f"      Enabled: {'✅' if liger_status.get('liger_enabled') else '❌'}")
    
    # Flash Attention
    print(f"   Flash Attention: {'✅' if FLASH_ATTN_AVAILABLE else '❌ Not installed'}")
    
    # Triton
    triton_status = get_kernel_status()
    print(f"   Triton Kernels: {'✅ v' + triton_status.get('triton_version', '?') if triton_status.get('triton_available') else '❌'}")
    if triton_status.get('triton_available'):
        print(f"      Fused SwiGLU: {'✅' if triton_status.get('fused_swiglu') else '❌'}")
        print(f"      Fused QK-Norm: {'✅' if triton_status.get('fused_qk_norm') else '❌'}")
        print(f"      Fused RoPE: {'✅' if triton_status.get('fused_rope') else '⚠️ Opt-in'}")
        print(f"      Fused RMSNorm: {'✅' if triton_status.get('fused_rms_norm') else '⚠️ Opt-in'}")
    
    # PyTorch settings
    print("\n🔧 PYTORCH SETTINGS:")
    print(f"   TF32 Matmul: {'✅' if torch.backends.cuda.matmul.allow_tf32 else '❌'}")
    print(f"   cuDNN Benchmark: {'✅' if torch.backends.cudnn.benchmark else '❌'}")
    print(f"   Float32 Precision: {torch.get_float32_matmul_precision()}")
    print(f"   Dynamo Cache: {torch._dynamo.config.cache_size_limit}")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    if not LIGER_AVAILABLE:
        print("   → pip install liger-kernel (50-60% memory savings)")
    if not FLASH_ATTN_AVAILABLE:
        print("   → pip install flash-attn --no-build-isolation (faster attention)")
    if hw.get('supports_fp8') and not TE_AVAILABLE:
        print("   → pip install transformer-engine[pytorch] (FP8 for 2x TFLOPS)")
    if not any([LIGER_AVAILABLE, FLASH_ATTN_AVAILABLE, TE_AVAILABLE]):
        print("   → All optional optimizations missing - install above packages!")
    elif LIGER_AVAILABLE and FLASH_ATTN_AVAILABLE:
        print("   ✅ Great setup! Consider Transformer Engine for Hopper+ GPUs.")
    
    print("=" * 70 + "\n")


def get_compile_config() -> Dict[str, Any]:
    """Get recommended torch.compile configuration."""
    return {
        "mode": "max-autotune-no-cudagraphs",
        "fullgraph": False,  # Allow graph breaks for flexibility
        "dynamic": True,     # Handle dynamic shapes
    }


# Convenience: auto-configure on import if env var set
if os.environ.get("HYDRA_AUTO_CONFIG", "0") == "1":
    configure_max_performance()
