#!/usr/bin/env python3
"""
HYDRA Optimization Verification Script

Run this after applying the optimizations to verify everything works:
    python verify_optimizations.py
"""

import sys
import warnings
warnings.filterwarnings("ignore")

def check_imports():
    """Verify all new imports work."""
    print("=" * 60)
    print("HYDRA Optimization Verification")
    print("=" * 60)
    print()
    
    errors = []
    
    # Check kernels
    print("1. Checking hydra.kernels...")
    try:
        from hydra.kernels import (
            fused_rope, fused_swiglu, fused_rms_norm,
            TRITON_AVAILABLE, get_kernel_status
        )
        status = get_kernel_status()
        print(f"   ✓ Triton available: {status['triton_available']}")
        print(f"   ✓ Kernels enabled: {status['use_triton_kernels']}")
    except Exception as e:
        errors.append(f"hydra.kernels: {e}")
        print(f"   ✗ Error: {e}")
    
    # Check layers
    print("\n2. Checking hydra.layers...")
    try:
        from hydra.layers import (
            RMSNorm, SwiGLUMLP, RotaryEmbedding,
            flexible_attention, GradientCheckpointMixin
        )
        print("   ✓ All layer classes imported")
    except Exception as e:
        errors.append(f"hydra.layers: {e}")
        print(f"   ✗ Error: {e}")
    
    # Check torch
    print("\n3. Checking PyTorch...")
    try:
        import torch
        print(f"   ✓ PyTorch version: {torch.__version__}")
        print(f"   ✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   ✓ GPU: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        errors.append(f"torch: {e}")
        print(f"   ✗ Error: {e}")
    
    # Check Flash Attention
    print("\n4. Checking Flash Attention 2...")
    try:
        from flash_attn import flash_attn_func
        print("   ✓ Flash Attention 2 available")
    except ImportError:
        print("   - Flash Attention 2 not installed (optional)")
        print("     Install with: pip install flash-attn --no-build-isolation")
    
    # Check xFormers
    print("\n5. Checking xFormers...")
    try:
        import xformers.ops
        print("   ✓ xFormers available")
    except ImportError:
        print("   - xFormers not installed (optional)")
        print("     Install with: pip install xformers")
    
    # Run basic functionality test
    print("\n6. Testing basic functionality...")
    try:
        import torch
        from hydra.layers import RMSNorm, SwiGLUMLP, RotaryEmbedding
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Test RMSNorm
        norm = RMSNorm(256).to(device)
        x = torch.randn(2, 64, 256, device=device)
        out = norm(x)
        assert out.shape == x.shape
        print("   ✓ RMSNorm forward pass")
        
        # Test SwiGLUMLP
        mlp = SwiGLUMLP(256, 512).to(device)
        out = mlp(x)
        assert out.shape == x.shape
        print("   ✓ SwiGLUMLP forward pass")
        
        # Test RoPE
        rope = RotaryEmbedding(32, max_seq_len=256).to(device)
        q = torch.randn(2, 8, 64, 32, device=device)
        out = rope(q, 64)
        assert out.shape == q.shape
        print("   ✓ RotaryEmbedding forward pass")
        
    except Exception as e:
        errors.append(f"functionality test: {e}")
        print(f"   ✗ Error: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    if errors:
        print("VERIFICATION FAILED")
        print("=" * 60)
        for err in errors:
            print(f"  - {err}")
        return 1
    else:
        print("VERIFICATION PASSED")
        print("=" * 60)
        print("All optimizations are working correctly!")
        return 0


def run_benchmark():
    """Run kernel benchmarks."""
    print("\n" + "=" * 60)
    print("Running Kernel Benchmarks")
    print("=" * 60)
    
    try:
        import torch
        if not torch.cuda.is_available():
            print("CUDA not available, skipping benchmark")
            return
        
        from hydra.kernels import benchmark_kernels, print_benchmark_results
        
        results = benchmark_kernels(
            batch_size=4,
            seq_len=512,
            dim=768,
            n_heads=12,
            warmup=5,
            iterations=50,
        )
        print_benchmark_results(results)
        
    except Exception as e:
        print(f"Benchmark error: {e}")


if __name__ == "__main__":
    exit_code = check_imports()
    
    if exit_code == 0 and "--benchmark" in sys.argv:
        run_benchmark()
    
    sys.exit(exit_code)
