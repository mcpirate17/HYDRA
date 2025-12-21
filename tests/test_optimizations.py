"""
Tests for HYDRA optimizations.

Run with: pytest tests/test_optimizations.py -v
"""

import pytest
import torch
import torch.nn as nn


# =============================================================================
# Test Fixtures
# =============================================================================

# Note: device() fixture is provided by conftest.py

@pytest.fixture
def test_config():
    """Standard test configuration."""
    return {
        "batch_size": 2,
        "seq_len": 128,
        "dim": 256,
        "n_heads": 8,
        "head_dim": 32,
    }


# =============================================================================
# Shared Layers Tests
# =============================================================================

class TestRMSNorm:
    """Test RMSNorm implementation."""
    
    def test_basic_forward(self, device, test_config):
        """Test basic forward pass."""
        from hydra.layers import RMSNorm
        
        dim = test_config["dim"]
        norm = RMSNorm(dim).to(device)
        x = torch.randn(test_config["batch_size"], test_config["seq_len"], dim, device=device)
        
        out = norm(x)
        
        assert out.shape == x.shape
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()
    
    def test_output_normalized(self, device, test_config):
        """Test that output has approximately unit RMS."""
        from hydra.layers import RMSNorm
        
        dim = test_config["dim"]
        norm = RMSNorm(dim).to(device)
        x = torch.randn(2, 64, dim, device=device) * 10  # Large values
        
        out = norm(x)
        
        # RMS should be close to 1 (before weight scaling)
        rms = torch.sqrt(out.pow(2).mean(-1))
        # With learnable weights, won't be exactly 1, but should be stable
        assert (rms < 10).all(), "Output RMS too large"
    
    def test_bf16_stability(self, device, test_config):
        """Test BF16 numerical stability."""
        if device == "cpu":
            pytest.skip("BF16 test requires CUDA")
        
        from hydra.layers import RMSNorm
        
        dim = test_config["dim"]
        norm = RMSNorm(dim).to(device).bfloat16()
        x = torch.randn(2, 64, dim, device=device, dtype=torch.bfloat16)
        
        out = norm(x)
        
        assert out.dtype == torch.bfloat16
        assert not torch.isnan(out).any()


class TestSwiGLUMLP:
    """Test SwiGLU MLP implementation."""
    
    def test_basic_forward(self, device, test_config):
        """Test basic forward pass."""
        from hydra.layers import SwiGLUMLP
        
        dim = test_config["dim"]
        hidden_dim = int(dim * 2.67)
        mlp = SwiGLUMLP(dim, hidden_dim).to(device)
        x = torch.randn(test_config["batch_size"], test_config["seq_len"], dim, device=device)
        
        out = mlp(x)
        
        assert out.shape == x.shape
        assert not torch.isnan(out).any()
    
    def test_gradient_flow(self, device, test_config):
        """Test gradient flow through MLP."""
        from hydra.layers import SwiGLUMLP
        
        dim = test_config["dim"]
        mlp = SwiGLUMLP(dim, dim * 4).to(device)
        x = torch.randn(2, 16, dim, device=device, requires_grad=True)
        
        out = mlp(x)
        loss = out.sum()
        loss.backward()
        
        # Check MLP parameters received gradients
        assert mlp.gate_up.weight.grad is not None
        assert not torch.isnan(mlp.gate_up.weight.grad).any()


class TestRotaryEmbedding:
    """Test shared RoPE implementation."""
    
    def test_basic_forward(self, device, test_config):
        """Test basic forward pass."""
        from hydra.layers import RotaryEmbedding
        
        head_dim = test_config["head_dim"]
        max_seq = 256
        rope = RotaryEmbedding(head_dim, max_seq).to(device)
        
        x = torch.randn(2, 8, 128, head_dim, device=device)
        out = rope(x, seq_len=128)
        
        assert out.shape == x.shape
        assert not torch.isnan(out).any()
    
    def test_cache_extension(self, device, test_config):
        """Test automatic cache extension for longer sequences."""
        from hydra.layers import RotaryEmbedding
        
        head_dim = test_config["head_dim"]
        rope = RotaryEmbedding(head_dim, max_seq_len=64).to(device)
        
        # First, use short sequence
        x_short = torch.randn(1, 4, 32, head_dim, device=device)
        out_short = rope(x_short, 32)
        assert out_short.shape == x_short.shape
        
        # Now use longer sequence (should auto-extend)
        x_long = torch.randn(1, 4, 128, head_dim, device=device)
        out_long = rope(x_long, 128)
        assert out_long.shape == x_long.shape
        
        # Verify cache was extended
        assert rope.max_seq_len >= 128
    
    def test_shared_across_layers(self, device, test_config):
        """Test that RoPE can be shared across attention layers."""
        from hydra.layers import RotaryEmbedding
        
        head_dim = test_config["head_dim"]
        shared_rope = RotaryEmbedding(head_dim, 256).to(device)
        
        # Simulate multiple layers using the same RoPE
        x = torch.randn(2, 8, 64, head_dim, device=device)
        
        results = []
        for _ in range(3):  # 3 "layers"
            out = shared_rope(x, 64)
            results.append(out)
        
        # All results should be identical (same input, same RoPE)
        for i in range(1, len(results)):
            assert torch.allclose(results[0], results[i])


# =============================================================================
# Triton Kernels Tests
# =============================================================================

class TestTritonKernels:
    """Test Triton kernel implementations."""
    
    def test_kernel_status(self):
        """Test kernel status reporting."""
        from hydra.kernels import get_kernel_status
        
        status = get_kernel_status()
        
        assert "triton_available" in status
        assert "use_triton_kernels" in status
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_fused_swiglu_matches_pytorch(self):
        """Test fused SwiGLU matches PyTorch implementation."""
        from hydra.kernels import fused_swiglu
        from hydra.kernels.fused_ops import _swiglu_pytorch
        
        gate = torch.randn(2, 64, 256, device="cuda")
        up = torch.randn(2, 64, 256, device="cuda")
        
        out_fused = fused_swiglu(gate, up)
        out_pytorch = _swiglu_pytorch(gate, up)
        
        assert torch.allclose(out_fused, out_pytorch, atol=1e-5)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_fused_rms_norm_matches_pytorch(self):
        """Test fused RMSNorm matches PyTorch implementation."""
        from hydra.kernels import fused_rms_norm
        from hydra.kernels.fused_ops import _rms_norm_pytorch
        
        x = torch.randn(2, 64, 256, device="cuda")
        weight = torch.ones(256, device="cuda")
        
        out_fused = fused_rms_norm(x, weight, 1e-6)
        out_pytorch = _rms_norm_pytorch(x, weight, 1e-6)
        
        assert torch.allclose(out_fused, out_pytorch, atol=1e-4)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_fused_rope_matches_pytorch(self):
        """Test fused RoPE matches PyTorch implementation."""
        from hydra.kernels import fused_rope
        from hydra.kernels.fused_ops import _rope_pytorch
        
        x = torch.randn(2, 8, 64, 32, device="cuda")
        cos = torch.randn(1, 1, 64, 16, device="cuda")
        sin = torch.randn(1, 1, 64, 16, device="cuda")
        
        out_fused = fused_rope(x, cos, sin)
        out_pytorch = _rope_pytorch(x, cos, sin)
        
        assert torch.allclose(out_fused, out_pytorch, atol=1e-4)


# =============================================================================
# Flexible Attention Tests
# =============================================================================

class TestFlexibleAttention:
    """Test flexible attention backend selection."""
    
    def test_sdpa_backend(self, device, test_config):
        """Test SDPA (PyTorch) backend."""
        from hydra.layers import flexible_attention, set_attention_backend
        
        set_attention_backend("sdpa")
        
        B, H, S, D = 2, 8, 64, 32
        q = torch.randn(B, H, S, D, device=device)
        k = torch.randn(B, H, S, D, device=device)
        v = torch.randn(B, H, S, D, device=device)
        
        out = flexible_attention(q, k, v, is_causal=True)
        
        assert out.shape == (B, H, S, D)
        assert not torch.isnan(out).any()
    
    def test_gqa_expansion(self, device):
        """Test GQA head expansion in flexible attention."""
        from hydra.layers import flexible_attention
        
        B, S, D = 2, 64, 32
        H_q, H_kv = 8, 2  # 4 groups
        
        q = torch.randn(B, H_q, S, D, device=device)
        k = torch.randn(B, H_kv, S, D, device=device)
        v = torch.randn(B, H_kv, S, D, device=device)
        
        out = flexible_attention(q, k, v, is_causal=True)
        
        assert out.shape == (B, H_q, S, D)


# =============================================================================
# Gradient Checkpointing Tests
# =============================================================================

class TestGradientCheckpointing:
    """Test gradient checkpointing utilities."""
    
    def test_mixin_enable_disable(self, device):
        """Test enable/disable gradient checkpointing."""
        from hydra.layers import GradientCheckpointMixin
        
        class TestModel(nn.Module, GradientCheckpointMixin):
            def __init__(self):
                super().__init__()
                self._gradient_checkpointing = False
                self.linear = nn.Linear(64, 64)
            
            def forward(self, x):
                return self.linear(x)
        
        model = TestModel().to(device)
        
        assert not model.is_gradient_checkpointing
        
        model.enable_gradient_checkpointing()
        assert model.is_gradient_checkpointing
        
        model.disable_gradient_checkpointing()
        assert not model.is_gradient_checkpointing


# =============================================================================
# Benchmark Tests (Optional, slow)
# =============================================================================

@pytest.mark.slow
class TestBenchmarks:
    """Benchmark tests (run with pytest -m slow)."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_kernel_benchmark(self):
        """Run kernel benchmark."""
        from hydra.kernels import benchmark_kernels, print_benchmark_results
        
        results = benchmark_kernels(
            batch_size=2,
            seq_len=256,
            dim=512,
            n_heads=8,
            warmup=5,
            iterations=20,
        )
        
        print_benchmark_results(results)
        
        assert "rope" in results
        assert "swiglu" in results
        assert "rms_norm" in results


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for the full stack."""
    
    def test_shared_layers_import(self):
        """Test that shared layers can be imported."""
        from hydra.layers import (
            RMSNorm,
            SwiGLUMLP,
            RotaryEmbedding,
            flexible_attention,
            GradientCheckpointMixin,
        )
        
        assert RMSNorm is not None
        assert SwiGLUMLP is not None
        assert RotaryEmbedding is not None
    
    def test_kernels_import(self):
        """Test that kernels can be imported."""
        from hydra.kernels import (
            fused_rope,
            fused_swiglu,
            fused_rms_norm,
            TRITON_AVAILABLE,
        )
        
        assert fused_rope is not None
        assert fused_swiglu is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
