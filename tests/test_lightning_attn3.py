"""
Tests for lightning_attn3 - Local fork with Blackwell (SM 12.x) compatibility.

Tests cover:
1. Import verification
2. Forward pass correctness
3. Backward pass (gradient computation)
4. Blackwell tile selection
5. Various tensor shapes and dtypes
6. Head-splitting for large D
7. Value padding for non-power-of-2 E
"""

import pytest
import torch
import torch.nn.functional as F

# Skip all tests if CUDA not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for lightning_attn3 tests"
)


class TestImports:
    """Verify all expected imports work."""
    
    def test_import_lightning_attn_func(self):
        from hydra.kernels.lightning_attn3.ops import lightning_attn_func
        assert callable(lightning_attn_func)
    
    def test_import_no_decay_kernel(self):
        from hydra.kernels.lightning_attn3.ops.triton import lightning_attn3_no_decay
        assert lightning_attn3_no_decay is not None
    
    def test_import_decay_kernel(self):
        from hydra.kernels.lightning_attn3.ops.triton import lightning_attn3
        assert lightning_attn3 is not None
    
    def test_import_parallel_kernel(self):
        from hydra.kernels.lightning_attn3.ops.triton import lightning_attn3_parallel
        assert lightning_attn3_parallel is not None


class TestBlackwellTileSelection:
    """Test the Blackwell-specific kernel selection logic."""
    
    def test_kernel_cache_exists_no_decay(self):
        """Verify _BWD_KERNEL_CACHE exists in no_decay module."""
        import importlib
        no_decay_module = importlib.import_module(
            'hydra.kernels.lightning_attn3.ops.triton.lightning_attn3_no_decay'
        )
        assert hasattr(no_decay_module, '_BWD_KERNEL_CACHE')
    
    def test_kernel_cache_exists_decay(self):
        """Verify _BWD_TILE_CACHE exists in decay module."""
        import importlib
        decay_module = importlib.import_module(
            'hydra.kernels.lightning_attn3.ops.triton.lightning_attn3'
        )
        assert hasattr(decay_module, '_BWD_TILE_CACHE')
    
    def test_kernel_cache_populated_after_backward(self):
        """Verify _BWD_KERNEL_CACHE is populated after a backward pass."""
        import importlib
        from hydra.kernels.lightning_attn3.ops import lightning_attn_func
        
        no_decay_module = importlib.import_module(
            'hydra.kernels.lightning_attn3.ops.triton.lightning_attn3_no_decay'
        )
        
        # Run a backward pass to populate the cache
        B, H, N, D = 2, 4, 128, 64
        q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16, requires_grad=True)
        k = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16, requires_grad=True)
        v = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16, requires_grad=True)
        
        out = lightning_attn_func(q, k, v)
        out.sum().backward()
        
        # Cache should now have an entry for current device
        device_idx = torch.cuda.current_device()
        assert device_idx in no_decay_module._BWD_KERNEL_CACHE
        
        kernel_type, block_or_cblock, cblock_or_zero = no_decay_module._BWD_KERNEL_CACHE[device_idx]
        assert kernel_type in ("original", "chunked")
        if kernel_type == "original":
            assert block_or_cblock in (32, 64)  # BLOCK
            assert cblock_or_zero in (16, 32)   # CBLOCK
        else:
            assert block_or_cblock in (16, 32)  # CBLOCK for chunked
    
    def test_blackwell_kernel_selection(self):
        """Test that Blackwell GPUs get chunked kernel after backward pass."""
        import importlib
        from hydra.kernels.lightning_attn3.ops import lightning_attn_func
        
        no_decay_module = importlib.import_module(
            'hydra.kernels.lightning_attn3.ops.triton.lightning_attn3_no_decay'
        )
        
        # Run backward to populate cache
        B, H, N, D = 2, 4, 128, 64
        q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16, requires_grad=True)
        k = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16, requires_grad=True)
        v = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16, requires_grad=True)
        
        out = lightning_attn_func(q, k, v)
        out.sum().backward()
        
        device_idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_idx)
        kernel_type, block_or_cblock, cblock_or_zero = no_decay_module._BWD_KERNEL_CACHE[device_idx]
        
        if props.major >= 12:  # Blackwell
            assert kernel_type == "chunked", f"Expected 'chunked' for Blackwell, got {kernel_type}"
            assert block_or_cblock == 16, f"Expected CBLOCK=16 for Blackwell chunked, got {block_or_cblock}"
        else:
            assert kernel_type == "original", f"Expected 'original' for non-Blackwell, got {kernel_type}"
            assert block_or_cblock == 64, f"Expected BLOCK=64 for non-Blackwell, got {block_or_cblock}"
            assert cblock_or_zero == 32, f"Expected CBLOCK=32 for non-Blackwell, got {cblock_or_zero}"


class TestForwardPass:
    """Test forward pass functionality."""
    
    @pytest.fixture
    def qkv_tensors(self):
        """Create standard Q, K, V tensors."""
        B, H, N, D = 2, 4, 256, 64
        q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
        k = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
        v = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
        return q, k, v
    
    def test_forward_basic(self, qkv_tensors):
        """Basic forward pass with standard shapes."""
        from hydra.kernels.lightning_attn3.ops import lightning_attn_func
        
        q, k, v = qkv_tensors
        out = lightning_attn_func(q, k, v)
        
        assert out.shape == v.shape
        assert out.dtype == v.dtype
        assert not torch.isnan(out).any(), "Output contains NaN"
        assert not torch.isinf(out).any(), "Output contains Inf"
    
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_forward_dtypes(self, dtype):
        """Test forward pass with different dtypes."""
        from hydra.kernels.lightning_attn3.ops import lightning_attn_func
        
        B, H, N, D = 2, 4, 128, 64
        q = torch.randn(B, H, N, D, device='cuda', dtype=dtype)
        k = torch.randn(B, H, N, D, device='cuda', dtype=dtype)
        v = torch.randn(B, H, N, D, device='cuda', dtype=dtype)
        
        out = lightning_attn_func(q, k, v)
        assert out.dtype == dtype
    
    @pytest.mark.parametrize("N", [64, 128, 256, 512, 1024])
    def test_forward_sequence_lengths(self, N):
        """Test forward pass with various sequence lengths."""
        from hydra.kernels.lightning_attn3.ops import lightning_attn_func
        
        B, H, D = 2, 4, 64
        q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
        k = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
        v = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
        
        out = lightning_attn_func(q, k, v)
        assert out.shape == (B, H, N, D)
    
    @pytest.mark.parametrize("D", [32, 64, 128])
    def test_forward_head_dims(self, D):
        """Test forward pass with various head dimensions."""
        from hydra.kernels.lightning_attn3.ops import lightning_attn_func
        
        B, H, N = 2, 4, 256
        q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
        k = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
        v = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
        
        out = lightning_attn_func(q, k, v)
        assert out.shape == (B, H, N, D)
    
    def test_forward_large_head_dim_splitting(self):
        """Test head-splitting for D > 128."""
        from hydra.kernels.lightning_attn3.ops import lightning_attn_func
        
        B, H, N, D = 2, 4, 128, 192  # D > 128 triggers splitting
        q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
        k = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
        v = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
        
        out = lightning_attn_func(q, k, v)
        assert out.shape == (B, H, N, D)
    
    def test_forward_non_power_of_2_value_dim(self):
        """Test value padding for non-power-of-2 E."""
        from hydra.kernels.lightning_attn3.ops import lightning_attn_func
        
        B, H, N, D, E = 2, 4, 128, 64, 48  # E=48 not power of 2
        q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
        k = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
        v = torch.randn(B, H, N, E, device='cuda', dtype=torch.float16)
        
        out = lightning_attn_func(q, k, v)
        assert out.shape == (B, H, N, E)  # Should unpad back to original E


class TestBackwardPass:
    """Test backward pass (gradient computation)."""
    
    def test_backward_basic(self):
        """Basic backward pass with gradient computation."""
        from hydra.kernels.lightning_attn3.ops import lightning_attn_func
        
        B, H, N, D = 2, 4, 128, 64
        q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16, requires_grad=True)
        k = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16, requires_grad=True)
        v = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16, requires_grad=True)
        
        out = lightning_attn_func(q, k, v)
        loss = out.sum()
        loss.backward()
        
        assert q.grad is not None, "q.grad is None"
        assert k.grad is not None, "k.grad is None"
        assert v.grad is not None, "v.grad is None"
        
        assert q.grad.shape == q.shape
        assert k.grad.shape == k.shape
        assert v.grad.shape == v.shape
        
        assert not torch.isnan(q.grad).any(), "q.grad contains NaN"
        assert not torch.isnan(k.grad).any(), "k.grad contains NaN"
        assert not torch.isnan(v.grad).any(), "v.grad contains NaN"
    
    def test_backward_gradient_flow(self):
        """Verify gradients actually flow (non-zero)."""
        from hydra.kernels.lightning_attn3.ops import lightning_attn_func
        
        B, H, N, D = 2, 4, 128, 64
        q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16, requires_grad=True)
        k = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16, requires_grad=True)
        v = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16, requires_grad=True)
        
        out = lightning_attn_func(q, k, v)
        loss = out.sum()
        loss.backward()
        
        # Gradients should be non-zero (statistically almost certain with random inputs)
        assert q.grad.abs().sum() > 0, "q.grad is all zeros"
        assert k.grad.abs().sum() > 0, "k.grad is all zeros"
        assert v.grad.abs().sum() > 0, "v.grad is all zeros"
    
    @pytest.mark.parametrize("N", [128, 256, 512])
    def test_backward_sequence_lengths(self, N):
        """Test backward pass with various sequence lengths."""
        from hydra.kernels.lightning_attn3.ops import lightning_attn_func
        
        B, H, D = 2, 4, 64
        q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16, requires_grad=True)
        k = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16, requires_grad=True)
        v = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16, requires_grad=True)
        
        out = lightning_attn_func(q, k, v)
        loss = out.sum()
        loss.backward()
        
        assert q.grad is not None and q.grad.shape == q.shape
        assert k.grad is not None and k.grad.shape == k.shape
        assert v.grad is not None and v.grad.shape == v.shape
    
    @pytest.mark.slow
    def test_backward_large_sequence(self):
        """Test backward pass with large sequence (stress test)."""
        from hydra.kernels.lightning_attn3.ops import lightning_attn_func
        
        B, H, N, D = 1, 8, 4096, 64
        q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16, requires_grad=True)
        k = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16, requires_grad=True)
        v = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16, requires_grad=True)
        
        out = lightning_attn_func(q, k, v)
        loss = out.sum()
        loss.backward()
        
        assert q.grad is not None
        assert not torch.isnan(q.grad).any()


class TestDecayVariant:
    """Test the decay variant with slope tensor."""
    
    def test_decay_forward(self):
        """Test forward pass with decay."""
        from hydra.kernels.lightning_attn3.ops import lightning_attn_func
        
        B, H, N, D = 2, 4, 64, 64  # Shorter sequence for stability
        q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16) * 0.1
        k = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16) * 0.1
        v = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16) * 0.1
        # Use small positive slopes for stability (typical ALiBi-style)
        s = torch.linspace(0.01, 0.1, H, device='cuda', dtype=torch.float32)
        
        out = lightning_attn_func(q, k, v, s=s)
        
        assert out.shape == v.shape
        assert not torch.isnan(out).any()
    
    def test_decay_backward(self):
        """Test backward pass with decay."""
        from hydra.kernels.lightning_attn3.ops import lightning_attn_func
        
        B, H, N, D = 2, 4, 64, 64
        q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16, requires_grad=True)
        k = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16, requires_grad=True)
        v = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16, requires_grad=True)
        # Scale in-place to keep as leaf tensors
        q.data.mul_(0.1)
        k.data.mul_(0.1)
        v.data.mul_(0.1)
        s = torch.linspace(0.01, 0.1, H, device='cuda', dtype=torch.float32)
        
        out = lightning_attn_func(q, k, v, s=s)
        loss = out.sum()
        loss.backward()
        
        assert q.grad is not None
        assert k.grad is not None
        assert v.grad is not None
    
    def test_parallel_variant(self):
        """Test parallel variant with decay."""
        from hydra.kernels.lightning_attn3.ops import lightning_attn_func
        
        B, H, N, D = 2, 4, 64, 64
        q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16) * 0.1
        k = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16) * 0.1
        v = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16) * 0.1
        s = torch.linspace(0.01, 0.1, H, device='cuda', dtype=torch.float32)
        
        out = lightning_attn_func(q, k, v, s=s, variant="parallel")
        
        assert out.shape == v.shape


class TestNumericalStability:
    """Test numerical stability under various conditions."""
    
    def test_small_values(self):
        """Test with small input values."""
        from hydra.kernels.lightning_attn3.ops import lightning_attn_func
        
        B, H, N, D = 2, 4, 128, 64
        scale = 1e-3
        q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16) * scale
        k = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16) * scale
        v = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16) * scale
        
        out = lightning_attn_func(q, k, v)
        
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()
    
    def test_large_values(self):
        """Test with moderately larger input values (within fp16 safe range)."""
        from hydra.kernels.lightning_attn3.ops import lightning_attn_func
        
        B, H, N, D = 2, 4, 128, 64
        scale = 2.0  # Moderate scale - 10.0 causes fp16 overflow with cumulative ops
        q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16) * scale
        k = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16) * scale
        v = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16) * scale
        
        out = lightning_attn_func(q, k, v)
        
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()
    
    def test_determinism(self):
        """Test that same inputs produce same outputs."""
        from hydra.kernels.lightning_attn3.ops import lightning_attn_func
        
        B, H, N, D = 2, 4, 128, 64
        
        torch.manual_seed(42)
        q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
        k = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
        v = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
        
        out1 = lightning_attn_func(q.clone(), k.clone(), v.clone())
        out2 = lightning_attn_func(q.clone(), k.clone(), v.clone())
        
        torch.testing.assert_close(out1, out2)


class TestIntegration:
    """Integration tests with HYDRA model components."""
    
    def test_hydra_imports(self):
        """Verify HYDRA model files can import lightning_attn3."""
        from hydra.model.hybrid_attention_variants import LightningAttn2Attention
        from hydra.model.ccgqa import CCGQAMoDMoRModel
        
        assert LightningAttn2Attention is not None
        assert CCGQAMoDMoRModel is not None
    
    def test_lla2_attention_layer(self):
        """Test LightningAttn2Attention layer forward."""
        from hydra.model.hybrid_attention_variants import LightningAttn2Attention
        
        dim = 256
        n_heads = 4
        layer = LightningAttn2Attention(dim=dim, n_heads=n_heads).cuda().half()
        
        B, N = 2, 128
        x = torch.randn(B, N, dim, device='cuda', dtype=torch.float16)
        
        out = layer(x)
        
        assert out.shape == x.shape
        assert not torch.isnan(out).any()


# Benchmark utilities (not run by default)
class TestBenchmarks:
    """Performance benchmarks (marked slow)."""
    
    @pytest.mark.slow
    def test_forward_throughput(self):
        """Measure forward pass throughput."""
        from hydra.kernels.lightning_attn3.ops import lightning_attn_func
        import time
        
        B, H, N, D = 4, 8, 2048, 64
        q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
        k = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
        v = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
        
        # Warmup
        for _ in range(10):
            _ = lightning_attn_func(q, k, v)
        torch.cuda.synchronize()
        
        # Benchmark
        n_iters = 100
        start = time.perf_counter()
        for _ in range(n_iters):
            _ = lightning_attn_func(q, k, v)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        throughput = n_iters / elapsed
        print(f"\nForward throughput: {throughput:.1f} iter/s")
        print(f"Latency: {elapsed/n_iters*1000:.2f} ms/iter")
    
    @pytest.mark.slow
    def test_backward_throughput(self):
        """Measure forward+backward pass throughput."""
        from hydra.kernels.lightning_attn3.ops import lightning_attn_func
        import time
        
        B, H, N, D = 4, 8, 2048, 64
        
        # Warmup
        for _ in range(10):
            q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16, requires_grad=True)
            k = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16, requires_grad=True)
            v = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16, requires_grad=True)
            out = lightning_attn_func(q, k, v)
            out.sum().backward()
        torch.cuda.synchronize()
        
        # Benchmark
        n_iters = 50
        start = time.perf_counter()
        for _ in range(n_iters):
            q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16, requires_grad=True)
            k = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16, requires_grad=True)
            v = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16, requires_grad=True)
            out = lightning_attn_func(q, k, v)
            out.sum().backward()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        throughput = n_iters / elapsed
        print(f"\nFwd+Bwd throughput: {throughput:.1f} iter/s")
        print(f"Latency: {elapsed/n_iters*1000:.2f} ms/iter")


class TestChunkedBackward:
    """Test the recompute-heavy chunked backward kernel for Blackwell."""
    
    def test_import_chunked_kernel(self):
        """Verify chunked kernel import."""
        from hydra.kernels.lightning_attn3.ops.triton import lightning_attn3_no_decay_chunked
        assert callable(lightning_attn3_no_decay_chunked)
    
    def test_validate_config(self):
        """Test SRAM budget validation."""
        from hydra.kernels.lightning_attn3.ops.triton.lightning_attn3_no_decay_chunked import (
            validate_config, SRAM_BUDGET
        )
        
        # Safe configs
        is_valid, sram = validate_config(16, 64)
        assert is_valid, f"Config (CBLOCK=16, d=64) should be valid: {sram} bytes"
        
        is_valid, sram = validate_config(16, 128)
        assert is_valid, f"Config (CBLOCK=16, d=128) should be valid: {sram} bytes"
        
        # Both should be well under limit
        assert sram < SRAM_BUDGET
    
    def test_chunked_forward_basic(self):
        """Test basic forward pass through chunked kernel."""
        from hydra.kernels.lightning_attn3.ops.triton import lightning_attn3_no_decay_chunked
        
        B, H, N, D = 2, 4, 128, 64
        q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
        k = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
        v = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
        
        out = lightning_attn3_no_decay_chunked(q, k, v)
        
        assert out.shape == (B, H, N, D)
        assert out.dtype == torch.float16
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()
    
    def test_chunked_backward_basic(self):
        """Test basic backward pass."""
        from hydra.kernels.lightning_attn3.ops.triton import lightning_attn3_no_decay_chunked
        
        B, H, N, D = 2, 4, 128, 64
        q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16, requires_grad=True)
        k = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16, requires_grad=True)
        v = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16, requires_grad=True)
        
        out = lightning_attn3_no_decay_chunked(q, k, v)
        loss = out.sum()
        loss.backward()
        
        # Check gradients exist and have correct shape
        assert q.grad is not None
        assert k.grad is not None
        assert v.grad is not None
        assert q.grad.shape == q.shape
        assert k.grad.shape == k.shape
        assert v.grad.shape == v.shape
        
        # Check no NaN or Inf
        assert not torch.isnan(q.grad).any()
        assert not torch.isnan(k.grad).any()
        assert not torch.isnan(v.grad).any()
    
    def test_chunked_gradient_flow(self):
        """Ensure gradients flow through all inputs."""
        from hydra.kernels.lightning_attn3.ops.triton import lightning_attn3_no_decay_chunked
        
        B, H, N, D = 2, 4, 256, 64
        q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16, requires_grad=True)
        k = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16, requires_grad=True)
        v = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16, requires_grad=True)
        
        out = lightning_attn3_no_decay_chunked(q, k, v)
        loss = out.sum()
        loss.backward()
        
        # Gradients should be non-zero
        assert q.grad.abs().sum() > 0, "dQ is all zeros"
        assert k.grad.abs().sum() > 0, "dK is all zeros"
        assert v.grad.abs().sum() > 0, "dV is all zeros"
    
    @pytest.mark.parametrize("N", [64, 128, 256, 512])
    def test_chunked_sequence_lengths(self, N):
        """Test chunked kernel with various sequence lengths."""
        from hydra.kernels.lightning_attn3.ops.triton import lightning_attn3_no_decay_chunked
        
        B, H, D = 2, 4, 64
        q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16, requires_grad=True)
        k = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16, requires_grad=True)
        v = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16, requires_grad=True)
        
        out = lightning_attn3_no_decay_chunked(q, k, v)
        loss = out.sum()
        loss.backward()
        
        assert out.shape == (B, H, N, D)
        assert not torch.isnan(out).any()
        assert not torch.isnan(q.grad).any()
    
    @pytest.mark.parametrize("D", [32, 64, 128])
    def test_chunked_head_dims(self, D):
        """Test chunked kernel with various head dimensions."""
        from hydra.kernels.lightning_attn3.ops.triton import lightning_attn3_no_decay_chunked
        
        B, H, N = 2, 4, 128
        q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16, requires_grad=True)
        k = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16, requires_grad=True)
        v = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16, requires_grad=True)
        
        out = lightning_attn3_no_decay_chunked(q, k, v)
        loss = out.sum()
        loss.backward()
        
        assert out.shape == (B, H, N, D)
        assert not torch.isnan(out).any()
        assert not torch.isnan(q.grad).any()
    
    def test_chunked_determinism(self):
        """Test that chunked kernel produces deterministic results."""
        from hydra.kernels.lightning_attn3.ops.triton import lightning_attn3_no_decay_chunked
        
        torch.manual_seed(42)
        B, H, N, D = 2, 4, 128, 64
        q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16, requires_grad=True)
        k = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16, requires_grad=True)
        v = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16, requires_grad=True)
        
        # First run
        out1 = lightning_attn3_no_decay_chunked(q, k, v)
        loss1 = out1.sum()
        loss1.backward()
        dq1, dk1, dv1 = q.grad.clone(), k.grad.clone(), v.grad.clone()
        
        # Reset grads
        q.grad, k.grad, v.grad = None, None, None
        
        # Second run
        out2 = lightning_attn3_no_decay_chunked(q, k, v)
        loss2 = out2.sum()
        loss2.backward()
        
        # Compare
        assert torch.allclose(out1, out2), "Forward pass not deterministic"
        assert torch.allclose(dq1, q.grad), "dQ not deterministic"
        assert torch.allclose(dk1, k.grad), "dK not deterministic"
        assert torch.allclose(dv1, v.grad), "dV not deterministic"
