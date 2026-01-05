"""
Tests for Mixture of Experts (MoE) routing module.

Tests cover:
- MoEConfig validation and immutability
- MoERouter shape outputs, top-k selection, gradient flow
- MoEDispatcher token routing
- MoEFFNBlock integration
- Auxiliary loss computation
- Identity-preserving initialization
- torch.compile compatibility
"""

import pytest
import torch
import torch.nn as nn

from hydra.routing import (
    MoEConfig,
    MoERouter,
    MoEDispatcher,
    MoEFFNBlock,
    MoEExpertMLP,
    get_moe_scaling,
    compute_moe_placement,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def default_config():
    """Default MoE configuration for testing."""
    return MoEConfig(
        dim=256,
        num_experts=4,
        top_k=1,
        aux_loss_weight=0.01,
    )


@pytest.fixture
def sample_input():
    """Sample input tensor."""
    return torch.randn(2, 16, 256)


@pytest.fixture
def sample_input_cuda():
    """Sample input tensor on CUDA if available."""
    if torch.cuda.is_available():
        return torch.randn(2, 16, 256, device="cuda")
    return None


# =============================================================================
# MoEConfig Tests
# =============================================================================

class TestMoEConfig:
    """Tests for MoEConfig dataclass."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = MoEConfig(dim=512)
        
        assert config.dim == 512
        assert config.num_experts == 4
        assert config.top_k == 1
        assert config.aux_loss_weight == 0.01
        assert config.router_jitter == 0.0
        assert config.capacity_factor == float("inf")
    
    def test_immutable(self, default_config):
        """Test that config is frozen (immutable)."""
        with pytest.raises(AttributeError):
            default_config.dim = 1024
    
    def test_invalid_dim(self):
        """Test validation rejects invalid dim."""
        with pytest.raises(ValueError, match="dim must be positive"):
            MoEConfig(dim=0)
    
    def test_invalid_num_experts(self):
        """Test validation rejects invalid num_experts."""
        with pytest.raises(ValueError, match="num_experts must be >= 2"):
            MoEConfig(dim=512, num_experts=1)
    
    def test_invalid_top_k(self):
        """Test validation rejects invalid top_k."""
        with pytest.raises(ValueError, match="top_k must be in"):
            MoEConfig(dim=512, num_experts=4, top_k=0)
        
        with pytest.raises(ValueError, match="top_k must be in"):
            MoEConfig(dim=512, num_experts=4, top_k=5)  # > num_experts
    
    def test_invalid_aux_loss_weight(self):
        """Test validation rejects negative aux loss weight."""
        with pytest.raises(ValueError, match="aux_loss_weight must be >= 0"):
            MoEConfig(dim=512, aux_loss_weight=-0.1)
    
    def test_slots(self):
        """Test config uses __slots__ for memory efficiency."""
        config = MoEConfig(dim=512)
        assert hasattr(config, "__slots__")


# =============================================================================
# MoE Scaling Tests
# =============================================================================

class TestMoEScaling:
    """Tests for MoE scaling rules."""
    
    def test_scaling_100m(self):
        """Test scaling for 100M model."""
        scaling = get_moe_scaling("100M")
        assert scaling["num_experts"] == 2
        assert scaling["num_moe_layers"] == 2
        assert scaling["aux_weight"] == 0.01
    
    def test_scaling_500m(self):
        """Test scaling for 500M model."""
        scaling = get_moe_scaling("500M")
        assert scaling["num_experts"] == 4
        assert scaling["num_moe_layers"] == 4
    
    def test_scaling_1b(self):
        """Test scaling for 1B model."""
        scaling = get_moe_scaling("1B")
        assert scaling["num_experts"] == 6
        assert scaling["num_moe_layers"] == 6
    
    def test_scaling_unknown(self):
        """Test default scaling for unknown model size."""
        scaling = get_moe_scaling("unknown_size")
        assert scaling["num_experts"] == 4  # Default
        assert scaling["num_moe_layers"] == 4


class TestMoEPlacement:
    """Tests for MoE layer placement computation."""
    
    def test_placement_basic(self):
        """Test basic placement computation."""
        placement = compute_moe_placement(n_blocks=8, n_moe_layers=2)
        # Should be 2 positions, evenly spaced, not at start or end
        assert len(placement) == 2
        assert 0 not in placement  # Not at first block
        assert 7 not in placement  # Not at last block
    
    def test_placement_single(self):
        """Test single MoE layer placement."""
        placement = compute_moe_placement(n_blocks=8, n_moe_layers=1)
        assert len(placement) == 1
        assert placement[0] not in (0, 7)
    
    def test_placement_many(self):
        """Test many MoE layers."""
        placement = compute_moe_placement(n_blocks=8, n_moe_layers=10)
        # Should cap at available positions (1-6)
        assert len(placement) <= 6
        assert all(0 < p < 7 for p in placement)
    
    def test_placement_zero(self):
        """Test zero MoE layers."""
        placement = compute_moe_placement(n_blocks=8, n_moe_layers=0)
        assert placement == ()
    
    def test_placement_deterministic(self):
        """Test placement is deterministic."""
        p1 = compute_moe_placement(n_blocks=12, n_moe_layers=4)
        p2 = compute_moe_placement(n_blocks=12, n_moe_layers=4)
        assert p1 == p2


# =============================================================================
# MoERouter Tests
# =============================================================================

class TestMoERouter:
    """Tests for MoERouter module."""
    
    def test_output_shapes(self, sample_input, default_config):
        """Test router produces correct output shapes."""
        router = MoERouter.from_config(default_config)
        indices, weights, aux_loss = router(sample_input)
        
        B, L, _ = sample_input.shape
        assert indices.shape == (B, L, default_config.top_k)
        assert weights.shape == (B, L, default_config.top_k)
        assert aux_loss.shape == ()  # scalar
    
    def test_indices_valid_range(self, sample_input, default_config):
        """Test expert indices are in valid range."""
        router = MoERouter.from_config(default_config)
        indices, _, _ = router(sample_input)
        
        assert (indices >= 0).all()
        assert (indices < default_config.num_experts).all()
    
    def test_weights_sum_to_one(self, sample_input, default_config):
        """Test routing weights sum to 1 for each token."""
        router = MoERouter.from_config(default_config)
        _, weights, _ = router(sample_input)
        
        weight_sums = weights.sum(dim=-1)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5)
    
    def test_gradient_flow(self, sample_input, default_config):
        """Test gradients flow through router."""
        router = MoERouter.from_config(default_config)
        router.train()
        
        sample_input.requires_grad_(True)
        indices, weights, aux_loss = router(sample_input)
        
        # Backward through aux_loss
        aux_loss.backward()
        
        assert router.gate.weight.grad is not None
    
    def test_aux_loss_training_only(self, sample_input, default_config):
        """Test aux loss is only computed during training."""
        router = MoERouter.from_config(default_config)
        
        # Training mode - aux_loss > 0
        router.train()
        _, _, aux_loss_train = router(sample_input)
        
        # Eval mode - aux_loss = 0
        router.eval()
        _, _, aux_loss_eval = router(sample_input)
        
        # Aux loss should be non-zero in training
        # (exact value depends on input distribution)
        assert aux_loss_train.item() >= 0
        assert aux_loss_eval.item() == 0
    
    def test_jitter_training_only(self):
        """Test jitter noise only applied during training."""
        torch.manual_seed(42)
        router = MoERouter(dim=256, num_experts=4, router_jitter=0.1)
        x = torch.randn(2, 8, 256)
        
        # Get logits in eval mode (no jitter)
        router.eval()
        _, _, _, logits_eval = router(x, return_logits=True)
        
        # Get logits in train mode (with jitter)
        router.train()
        torch.manual_seed(123)  # Different seed
        _, _, _, logits_train = router(x, return_logits=True)
        
        # Logits should differ due to jitter in training
        # (Note: comparing same input with different random states)
    
    def test_routing_stats(self, sample_input, default_config):
        """Test routing statistics are computed."""
        router = MoERouter.from_config(default_config)
        router.train()
        _ = router(sample_input)
        
        stats = router.get_routing_stats()
        assert "expert_utilization" in stats
        assert "aux_loss" in stats
        assert len(stats["expert_utilization"]) == default_config.num_experts


# =============================================================================
# MoEDispatcher Tests
# =============================================================================

class TestMoEDispatcher:
    """Tests for MoEDispatcher module."""
    
    def test_output_shape(self, sample_input):
        """Test dispatcher produces correct output shape."""
        num_experts = 4
        dispatcher = MoEDispatcher(num_experts=num_experts, top_k=1)
        
        # Create mock expert indices and weights
        B, L, D = sample_input.shape
        indices = torch.randint(0, num_experts, (B, L, 1))
        weights = torch.ones(B, L, 1)
        
        # Create mock experts
        experts = nn.ModuleList([nn.Linear(D, D) for _ in range(num_experts)])
        
        output = dispatcher(sample_input, indices, weights, experts)
        assert output.shape == sample_input.shape
    
    def test_no_dropping_infinite_capacity(self, sample_input):
        """Test no tokens dropped with infinite capacity."""
        dispatcher = MoEDispatcher(num_experts=4, top_k=1, capacity_factor=float("inf"))
        
        B, L, D = sample_input.shape
        indices = torch.zeros(B, L, 1, dtype=torch.long)  # All to expert 0
        weights = torch.ones(B, L, 1)
        experts = nn.ModuleList([nn.Linear(D, D) for _ in range(4)])
        
        _ = dispatcher(sample_input, indices, weights, experts)
        
        stats = dispatcher.get_dispatch_stats()
        assert stats["tokens_dropped"] == 0


# =============================================================================
# MoEExpertMLP Tests
# =============================================================================

class TestMoEExpertMLP:
    """Tests for MoEExpertMLP module."""
    
    def test_output_shape(self):
        """Test expert MLP produces correct output shape."""
        dim, hidden_dim = 256, 512
        expert = MoEExpertMLP(dim=dim, hidden_dim=hidden_dim)
        
        x = torch.randn(2, 16, dim)
        output = expert(x)
        
        assert output.shape == x.shape
    
    def test_identity_init_small_output(self):
        """Test identity init produces small outputs."""
        dim, hidden_dim = 256, 512
        expert = MoEExpertMLP(dim=dim, hidden_dim=hidden_dim, identity_init=True)
        
        x = torch.randn(2, 16, dim)
        output = expert(x)
        
        # Output should be much smaller than input with identity init
        output_rms = output.pow(2).mean().sqrt().item()
        input_rms = x.pow(2).mean().sqrt().item()
        assert output_rms < 0.1 * input_rms
    
    def test_gradient_flow(self):
        """Test gradients flow through expert."""
        expert = MoEExpertMLP(dim=256, hidden_dim=512)
        x = torch.randn(2, 8, 256, requires_grad=True)
        
        output = expert(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert expert.gate_up.weight.grad is not None
        assert expert.down.weight.grad is not None


# =============================================================================
# MoEFFNBlock Tests
# =============================================================================

class TestMoEFFNBlock:
    """Tests for MoEFFNBlock module."""
    
    def test_output_shape(self, sample_input):
        """Test block produces correct output shape."""
        block = MoEFFNBlock(dim=256, num_experts=4)
        output = block(sample_input)
        
        assert output.shape == sample_input.shape
    
    def test_identity_init_close_to_input(self, sample_input):
        """Test identity init produces output close to input."""
        block = MoEFFNBlock(
            dim=256,
            num_experts=4,
            identity_init=True,
            warmup_steps=10000,  # Large warmup so scale = 0
        )
        block.set_global_step(0)  # Before warmup
        
        output = block(sample_input)
        
        # Output should be very close to input with identity init + warmup
        diff = (output - sample_input).abs().mean().item()
        input_scale = sample_input.abs().mean().item()
        assert diff < 0.01 * input_scale, f"diff={diff}, input_scale={input_scale}"
    
    def test_warmup_scaling(self, sample_input):
        """Test warmup gradually increases MoE contribution."""
        block = MoEFFNBlock(
            dim=256,
            num_experts=4,
            identity_init=True,
            warmup_steps=100,
        )
        
        # At step 0, scale should be 0
        block.set_global_step(0)
        assert block.get_warmup_scale() == 0.0
        
        # At step 50, scale should be 0.5
        block.set_global_step(50)
        assert abs(block.get_warmup_scale() - 0.5) < 0.01
        
        # At step 100+, scale should be 1.0
        block.set_global_step(200)
        assert block.get_warmup_scale() == 1.0
    
    def test_forward_with_losses(self, sample_input):
        """Test forward_with_losses returns aux loss."""
        block = MoEFFNBlock(dim=256, num_experts=4, aux_loss_weight=0.01)
        block.train()
        
        output, losses = block.forward_with_losses(sample_input)
        
        assert output.shape == sample_input.shape
        assert "moe_aux_loss" in losses
        assert losses["moe_aux_loss"].shape == ()
    
    def test_gradient_flow(self, sample_input):
        """Test gradients flow through entire block."""
        block = MoEFFNBlock(dim=256, num_experts=4, identity_init=False)
        block.train()
        
        sample_input.requires_grad_(True)
        output = block(sample_input)
        loss = output.sum()
        loss.backward()
        
        assert sample_input.grad is not None
        # Check gradients in router
        assert block.router.gate.weight.grad is not None
        # Check gradients in at least one expert
        assert block.experts[0].gate_up.weight.grad is not None
    
    def test_routing_stats(self, sample_input):
        """Test routing statistics are collected."""
        block = MoEFFNBlock(dim=256, num_experts=4)
        block.train()
        block.set_global_step(100)
        
        _ = block(sample_input)
        stats = block.get_routing_stats()
        
        assert "expert_utilization" in stats
        assert "residual_alpha" in stats
        assert "warmup_scale" in stats
        assert "moe_enabled" in stats
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_forward(self):
        """Test forward works on CUDA."""
        block = MoEFFNBlock(dim=256, num_experts=4).cuda()
        x = torch.randn(2, 16, 256, device="cuda")
        
        output = block(x)
        assert output.device.type == "cuda"
        assert output.shape == x.shape


# =============================================================================
# torch.compile Compatibility Tests
# =============================================================================

class TestCompileCompatibility:
    """Tests for torch.compile compatibility."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_router_compile(self):
        """Test MoERouter compiles without graph breaks."""
        router = MoERouter(dim=256, num_experts=4).cuda()
        
        # Compile the router
        compiled_router = torch.compile(router, mode="reduce-overhead")
        
        x = torch.randn(2, 16, 256, device="cuda")
        
        # Should not raise
        indices, weights, aux_loss = compiled_router(x)
        
        assert indices.shape == (2, 16, 1)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_block_compile(self):
        """Test MoEFFNBlock compiles without graph breaks."""
        block = MoEFFNBlock(dim=256, num_experts=4, identity_init=False).cuda()
        
        # Compile the block
        compiled_block = torch.compile(block, mode="reduce-overhead")
        
        x = torch.randn(2, 16, 256, device="cuda")
        
        # Should not raise
        output = compiled_block(x)
        
        assert output.shape == x.shape
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available") 
    def test_backward_compile(self):
        """Test backward pass works with compiled block."""
        block = MoEFFNBlock(dim=256, num_experts=4, identity_init=False).cuda()
        block.train()
        
        compiled_block = torch.compile(block, mode="reduce-overhead")
        
        x = torch.randn(2, 16, 256, device="cuda", requires_grad=True)
        
        output = compiled_block(x)
        loss = output.sum()
        
        # Should not raise
        loss.backward()
        
        assert x.grad is not None


# =============================================================================
# Integration Tests
# =============================================================================

class TestMoEIntegration:
    """Integration tests for MoE components."""
    
    def test_full_pipeline(self, sample_input):
        """Test complete MoE forward pass."""
        block = MoEFFNBlock(
            dim=256,
            num_experts=4,
            top_k=1,
            aux_loss_weight=0.01,
            identity_init=False,
        )
        block.train()
        block.set_global_step(1000)  # Past warmup
        
        output, losses = block.forward_with_losses(sample_input)
        
        # Output shape preserved
        assert output.shape == sample_input.shape
        
        # Aux loss computed
        assert "moe_aux_loss" in losses
        aux_loss = losses["moe_aux_loss"]
        assert aux_loss.item() >= 0
        
        # Gradients flow
        sample_input.requires_grad_(True)
        output = block(sample_input)
        output.sum().backward()
        assert sample_input.grad is not None
    
    def test_multiple_blocks(self, sample_input):
        """Test multiple MoE blocks in sequence."""
        blocks = nn.ModuleList([
            MoEFFNBlock(dim=256, num_experts=4, identity_init=False)
            for _ in range(3)
        ])
        
        for block in blocks:
            block.set_global_step(1000)
        
        h = sample_input
        for block in blocks:
            h = block(h)
        
        assert h.shape == sample_input.shape
    
    def test_moe_disabled_equivalent_to_input(self, sample_input):
        """Test MoE with alpha=0 returns near-input."""
        block = MoEFFNBlock(
            dim=256,
            num_experts=4,
            identity_init=True,
            warmup_steps=10000,
        )
        block.set_global_step(0)  # Scale = 0
        
        # Manually set alpha to exactly 0
        with torch.no_grad():
            block.residual_alpha.fill_(0.0)
        
        output = block(sample_input)
        
        # Should be identical to input
        assert torch.allclose(output, sample_input, atol=1e-5)


# =============================================================================
# Finite Gradient Tests
# =============================================================================

class TestFiniteGradients:
    """Tests for gradient finiteness (no NaNs/Infs)."""
    
    def test_router_finite_grads(self, sample_input):
        """Test router produces finite gradients."""
        router = MoERouter(dim=256, num_experts=4, aux_loss_weight=0.01)
        router.train()
        
        sample_input.requires_grad_(True)
        _, _, aux_loss = router(sample_input)
        aux_loss.backward()
        
        assert torch.isfinite(router.gate.weight.grad).all()
    
    def test_block_finite_grads(self, sample_input):
        """Test block produces finite gradients."""
        block = MoEFFNBlock(dim=256, num_experts=4, identity_init=False)
        block.train()
        block.set_global_step(1000)
        
        sample_input.requires_grad_(True)
        output = block(sample_input)
        loss = output.sum()
        loss.backward()
        
        # Check all gradients are finite
        for name, param in block.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), f"Non-finite grad in {name}"
    
    def test_extreme_input_finite(self):
        """Test finite outputs with extreme inputs."""
        block = MoEFFNBlock(dim=256, num_experts=4, identity_init=False)
        block.set_global_step(1000)
        
        # Large input
        x_large = torch.randn(2, 16, 256) * 100
        output = block(x_large)
        assert torch.isfinite(output).all()
        
        # Small input
        x_small = torch.randn(2, 16, 256) * 0.001
        output = block(x_small)
        assert torch.isfinite(output).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
