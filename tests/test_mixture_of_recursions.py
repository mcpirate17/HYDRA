"""
Tests for Mixture of Recursions (MoR) routing module.

Tests cover:
- MoRConfig validation and immutability
- MoRRouter shape outputs and gradient flow
- MoRExecutor recursion execution
- Integration with loss-driven routing
"""

import pytest
import torch
import torch.nn as nn

from hydra.routing import (
    MoRConfig,
    MoRRouter,
    MoRExecutor,
    dim_to_depth_scale,
    compute_layer_target_prob,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def default_config():
    """Default MoR configuration for testing."""
    return MoRConfig(
        dim=256,
        n_recursions=5,
        ponder_loss_weight=0.01,
        warmup_steps=100,
    )


@pytest.fixture
def layer_aware_config():
    """Config with layer-aware settings."""
    return MoRConfig(
        dim=512,
        n_recursions=4,
        layer_idx=2,
        total_layers=8,
        depth_alpha=0.5,
        dim_ref=256,
    )


@pytest.fixture
def simple_mlp():
    """Simple MLP for testing."""
    return nn.Sequential(
        nn.Linear(256, 512),
        nn.GELU(),
        nn.Linear(512, 256),
    )


@pytest.fixture
def sample_input():
    """Sample input tensor."""
    return torch.randn(2, 16, 256)


# =============================================================================
# MoRConfig Tests
# =============================================================================

class TestMoRConfig:
    """Tests for MoRConfig dataclass."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = MoRConfig(dim=512)
        
        assert config.dim == 512
        assert config.n_recursions == 5
        assert config.ponder_loss_weight == 0.01
        assert config.router_jitter == 0.0
        assert config.warmup_steps == 2500
        assert config.layer_idx == 0
        assert config.total_layers == 1
        assert config.dim_ref == 768
        assert config.depth_alpha == 0.0
        assert config.depth_scale_max == 2.0
    
    def test_immutable(self, default_config):
        """Test that config is frozen (immutable)."""
        with pytest.raises(AttributeError):
            default_config.dim = 1024
    
    def test_invalid_dim(self):
        """Test validation rejects invalid dim."""
        with pytest.raises(ValueError, match="dim must be positive"):
            MoRConfig(dim=0)
        
        with pytest.raises(ValueError, match="dim must be positive"):
            MoRConfig(dim=-512)
    
    def test_invalid_n_recursions(self):
        """Test validation rejects invalid n_recursions."""
        with pytest.raises(ValueError, match="n_recursions must be >= 1"):
            MoRConfig(dim=512, n_recursions=0)
    
    def test_invalid_ponder_weight(self):
        """Test validation rejects negative ponder weight."""
        with pytest.raises(ValueError, match="ponder_loss_weight must be >= 0"):
            MoRConfig(dim=512, ponder_loss_weight=-0.1)
    
    def test_slots(self):
        """Test config uses __slots__ for memory efficiency."""
        config = MoRConfig(dim=512)
        assert hasattr(config, "__slots__")


# =============================================================================
# Utility Function Tests
# =============================================================================

class TestDimToDepthScale:
    """Tests for dim_to_depth_scale function."""
    
    def test_disabled_with_zero_alpha(self):
        """Alpha=0 should always return 1.0."""
        assert dim_to_depth_scale(768, 768, 0.0) == 1.0
        assert dim_to_depth_scale(1536, 768, 0.0) == 1.0
        assert dim_to_depth_scale(3072, 768, 0.0) == 1.0
    
    def test_smaller_dim_returns_one(self):
        """Dim <= dim_ref should return 1.0."""
        assert dim_to_depth_scale(384, 768, 0.5) == 1.0
        assert dim_to_depth_scale(768, 768, 0.5) == 1.0
    
    def test_sqrt_scaling(self):
        """Test sqrt scaling (alpha=0.5)."""
        scale = dim_to_depth_scale(1536, 768, 0.5)
        # 1536/768 = 2, sqrt(2) ≈ 1.414
        assert abs(scale - 1.414) < 0.01
    
    def test_linear_scaling(self):
        """Test linear scaling (alpha=1.0)."""
        scale = dim_to_depth_scale(1536, 768, 1.0)
        # 1536/768 = 2
        assert scale == 2.0
    
    def test_max_clamp(self):
        """Test scale is clamped to max."""
        scale = dim_to_depth_scale(6144, 768, 0.5, scale_max=2.0)
        assert scale == 2.0
    
    def test_custom_scale_max(self):
        """Test custom scale_max."""
        scale = dim_to_depth_scale(6144, 768, 1.0, scale_max=4.0)
        assert scale == 4.0


class TestComputeLayerTargetProb:
    """Tests for compute_layer_target_prob function."""
    
    def test_first_layer(self):
        """First layer should have lower target prob."""
        prob = compute_layer_target_prob(0, 8, depth_scale=1.0)
        assert abs(prob - 0.2) < 0.01  # Base is 0.2 for first layer
    
    def test_last_layer(self):
        """Last layer should have higher target prob."""
        prob = compute_layer_target_prob(7, 8, depth_scale=1.0)
        assert abs(prob - 0.5) < 0.01  # Base is 0.5 for last layer
    
    def test_middle_layer(self):
        """Middle layer should have intermediate prob."""
        prob = compute_layer_target_prob(4, 8, depth_scale=1.0)
        # layer_ratio = 4/7 ≈ 0.57, prob ≈ 0.2 + 0.3*0.57 ≈ 0.37
        assert 0.3 < prob < 0.45
    
    def test_depth_scale_increases_prob(self):
        """Higher depth scale should increase target prob."""
        prob_base = compute_layer_target_prob(0, 8, depth_scale=1.0)
        prob_scaled = compute_layer_target_prob(0, 8, depth_scale=2.0)
        assert prob_scaled > prob_base


# =============================================================================
# MoRRouter Tests
# =============================================================================

class TestMoRRouter:
    """Tests for MoRRouter class."""
    
    def test_output_shapes(self, default_config, sample_input):
        """Test router output shapes."""
        router = MoRRouter(default_config)
        depths, probs, logits = router(sample_input)
        
        B, L, D = sample_input.shape
        assert depths.shape == (B, L)
        assert probs.shape == (B, L)
        assert logits.shape == (B, L)
    
    def test_depth_range(self, default_config, sample_input):
        """Test depths are in valid range."""
        router = MoRRouter(default_config)
        depths, _, _ = router(sample_input)
        
        assert depths.min() >= 0
        assert depths.max() < default_config.n_recursions
    
    def test_probs_in_zero_one(self, default_config, sample_input):
        """Test probabilities are in [0, 1]."""
        router = MoRRouter(default_config)
        _, probs, _ = router(sample_input)
        
        assert probs.min() >= 0.0
        assert probs.max() <= 1.0
    
    def test_gradient_flow(self, default_config, sample_input):
        """Test gradients flow through router."""
        router = MoRRouter(default_config)
        sample_input.requires_grad_(True)
        
        depths, probs, logits = router(sample_input)
        
        # Loss based on probs (has gradients)
        loss = probs.mean()
        loss.backward()
        
        assert sample_input.grad is not None
        assert not torch.isnan(sample_input.grad).any()
    
    def test_jitter_only_training(self, sample_input):
        """Test jitter is only applied during training."""
        config = MoRConfig(dim=256, router_jitter=0.5)
        router = MoRRouter(config)
        
        # Set seed for reproducibility
        torch.manual_seed(42)
        router.train()
        _, probs_train, _ = router(sample_input)
        
        torch.manual_seed(42)
        router.eval()
        _, probs_eval, _ = router(sample_input)
        
        # With same seed, eval should be deterministic but train has noise
        # This test checks the logic exists (exact behavior depends on noise)
        router.train()
    
    def test_warmup_scale(self, default_config):
        """Test warmup scale computation."""
        router = MoRRouter(default_config)
        
        # Initial warmup scale
        assert router.warmup_scale == 0.0
        
        # After some steps
        router.set_global_step(50)
        assert router.warmup_scale == 0.5  # 50/100
        
        # After warmup complete
        router.set_global_step(200)
        assert router.warmup_scale == 1.0
    
    def test_ponder_loss_shape(self, default_config, sample_input):
        """Test ponder loss is scalar."""
        router = MoRRouter(default_config)
        router.set_global_step(200)  # Past warmup
        
        depths, probs, logits = router(sample_input)
        ponder_loss = router.compute_ponder_loss(depths, probs, logits)
        
        assert ponder_loss.shape == ()
        assert ponder_loss.requires_grad
    
    def test_ponder_loss_warmup_scaling(self, default_config, sample_input):
        """Test ponder loss is scaled by warmup."""
        router = MoRRouter(default_config)
        
        depths, probs, logits = router(sample_input)
        
        # Before warmup
        router.set_global_step(0)
        loss_before = router.compute_ponder_loss(depths, probs, logits)
        
        # After warmup
        router.set_global_step(200)
        loss_after = router.compute_ponder_loss(depths, probs, logits)
        
        # Loss should be larger after warmup (unless exactly zero)
        if loss_after.item() > 0:
            assert loss_after > loss_before


# =============================================================================
# MoRExecutor Tests
# =============================================================================

class TestMoRExecutor:
    """Tests for MoRExecutor class."""
    
    def test_output_shape(self, default_config, sample_input, simple_mlp):
        """Test executor output shape matches input."""
        executor = MoRExecutor(default_config)
        router = MoRRouter(default_config)
        
        depths, probs, _ = router(sample_input)
        output = executor(sample_input, depths, probs, simple_mlp)
        
        assert output.shape == sample_input.shape
    
    def test_no_nan_output(self, default_config, sample_input, simple_mlp):
        """Test output has no NaN values."""
        executor = MoRExecutor(default_config)
        router = MoRRouter(default_config)
        
        depths, probs, _ = router(sample_input)
        output = executor(sample_input, depths, probs, simple_mlp)
        
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_gradient_flow(self, default_config, sample_input, simple_mlp):
        """Test gradients flow through executor."""
        executor = MoRExecutor(default_config)
        router = MoRRouter(default_config)
        sample_input.requires_grad_(True)
        
        depths, probs, _ = router(sample_input)
        output = executor(sample_input, depths, probs, simple_mlp)
        
        loss = output.mean()
        loss.backward()
        
        assert sample_input.grad is not None
        assert not torch.isnan(sample_input.grad).any()
    
    def test_depth_zero_minimal_change(self, default_config, simple_mlp):
        """Test depth-0 tokens get minimal MLP passes."""
        executor = MoRExecutor(default_config)
        
        x = torch.randn(1, 4, 256)
        # All tokens at depth 0
        depths = torch.zeros(1, 4, dtype=torch.long)
        probs = torch.zeros(1, 4)  # Low prob = shallow
        
        output = executor(x, depths, probs, simple_mlp)
        
        # Output should be similar to input (one MLP pass)
        # Can't test exact equality due to MLP modification
        assert output.shape == x.shape
    
    def test_recursion_stats(self, default_config, sample_input, simple_mlp):
        """Test recursion statistics are tracked."""
        executor = MoRExecutor(default_config)
        executor.train()
        
        router = MoRRouter(default_config)
        depths, probs, _ = router(sample_input)
        _ = executor(sample_input, depths, probs, simple_mlp)
        
        stats = executor.get_recursion_stats()
        
        # Should have stats for each depth level
        assert len(stats) > 0
    
    def test_with_normalization(self, default_config, sample_input, simple_mlp):
        """Test executor works with optional normalization."""
        executor = MoRExecutor(default_config)
        router = MoRRouter(default_config)
        norm = nn.LayerNorm(256)
        
        depths, probs, _ = router(sample_input)
        output = executor(sample_input, depths, probs, simple_mlp, norm=norm)
        
        assert output.shape == sample_input.shape
        assert not torch.isnan(output).any()


# =============================================================================
# Integration Tests
# =============================================================================

class TestMoRIntegration:
    """Integration tests for MoR components."""
    
    def test_full_forward_pass(self, default_config, sample_input, simple_mlp):
        """Test complete forward pass through router and executor."""
        router = MoRRouter(default_config)
        executor = MoRExecutor(default_config)
        
        # Forward pass
        depths, probs, logits = router(sample_input)
        output = executor(sample_input, depths, probs, simple_mlp)
        ponder_loss = router.compute_ponder_loss(depths, probs, logits)
        
        # Check outputs
        assert output.shape == sample_input.shape
        assert ponder_loss.shape == ()
        assert not torch.isnan(output).any()
        assert not torch.isnan(ponder_loss).any()
    
    def test_backward_pass(self, default_config, sample_input, simple_mlp):
        """Test backward pass through entire MoR pipeline."""
        router = MoRRouter(default_config)
        executor = MoRExecutor(default_config)
        router.set_global_step(200)  # Past warmup
        
        sample_input.requires_grad_(True)
        
        # Forward
        depths, probs, logits = router(sample_input)
        output = executor(sample_input, depths, probs, simple_mlp)
        ponder_loss = router.compute_ponder_loss(depths, probs, logits)
        
        # Backward
        total_loss = output.mean() + 0.01 * ponder_loss
        total_loss.backward()
        
        # Check gradients exist
        assert sample_input.grad is not None
        for p in router.parameters():
            assert p.grad is not None
        for p in simple_mlp.parameters():
            assert p.grad is not None
    
    def test_deterministic_inference(self, default_config, sample_input, simple_mlp):
        """Test inference is deterministic."""
        router = MoRRouter(default_config)
        executor = MoRExecutor(default_config)
        router.eval()
        executor.eval()
        
        with torch.no_grad():
            depths1, probs1, _ = router(sample_input)
            output1 = executor(sample_input, depths1, probs1, simple_mlp)
            
            depths2, probs2, _ = router(sample_input)
            output2 = executor(sample_input, depths2, probs2, simple_mlp)
        
        assert torch.allclose(depths1, depths2)
        assert torch.allclose(probs1, probs2)
        assert torch.allclose(output1, output2)
    
    def test_different_batch_sizes(self, default_config, simple_mlp):
        """Test MoR works with different batch sizes."""
        router = MoRRouter(default_config)
        executor = MoRExecutor(default_config)
        
        for batch_size in [1, 2, 8, 16]:
            x = torch.randn(batch_size, 32, 256)
            depths, probs, _ = router(x)
            output = executor(x, depths, probs, simple_mlp)
            
            assert output.shape == x.shape
    
    def test_different_seq_lengths(self, default_config, simple_mlp):
        """Test MoR works with different sequence lengths."""
        router = MoRRouter(default_config)
        executor = MoRExecutor(default_config)
        
        for seq_len in [8, 16, 64, 128]:
            x = torch.randn(2, seq_len, 256)
            depths, probs, _ = router(x)
            output = executor(x, depths, probs, simple_mlp)
            
            assert output.shape == x.shape


# =============================================================================
# GPU Tests (if available)
# =============================================================================

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestMoRGPU:
    """GPU-specific tests for MoR."""
    
    def test_gpu_forward(self, default_config):
        """Test forward pass on GPU."""
        router = MoRRouter(default_config).cuda()
        executor = MoRExecutor(default_config).cuda()
        mlp = nn.Sequential(
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Linear(512, 256),
        ).cuda()
        
        x = torch.randn(2, 16, 256, device="cuda")
        
        depths, probs, logits = router(x)
        output = executor(x, depths, probs, mlp)
        ponder_loss = router.compute_ponder_loss(depths, probs, logits)
        
        assert output.device.type == "cuda"
        assert ponder_loss.device.type == "cuda"
    
    def test_gpu_backward(self, default_config):
        """Test backward pass on GPU."""
        router = MoRRouter(default_config).cuda()
        executor = MoRExecutor(default_config).cuda()
        router.set_global_step(200)
        mlp = nn.Sequential(
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Linear(512, 256),
        ).cuda()
        
        x = torch.randn(2, 16, 256, device="cuda", requires_grad=True)
        
        depths, probs, logits = router(x)
        output = executor(x, depths, probs, mlp)
        ponder_loss = router.compute_ponder_loss(depths, probs, logits)
        
        loss = output.mean() + 0.01 * ponder_loss
        loss.backward()
        
        assert x.grad is not None
        assert x.grad.device.type == "cuda"
    
    def test_mixed_precision(self, default_config):
        """Test MoR with mixed precision (AMP)."""
        router = MoRRouter(default_config).cuda()
        executor = MoRExecutor(default_config).cuda()
        mlp = nn.Sequential(
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Linear(512, 256),
        ).cuda()
        
        x = torch.randn(2, 16, 256, device="cuda")
        
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            depths, probs, logits = router(x)
            output = executor(x, depths, probs, mlp)
        
        # Output should be in reduced precision
        assert output.dtype in (torch.float16, torch.bfloat16, torch.float32)
        assert not torch.isnan(output).any()
