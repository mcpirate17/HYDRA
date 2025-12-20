"""
Tests for Mixture of Depths (MoD) routing module.

Tests cover:
- MoDConfig validation and immutability
- MoDRouter shape outputs, top-k selection, gradient flow
- MixtureOfDepthsBlock integration
- Auxiliary loss computation
"""

import pytest
import torch
import torch.nn as nn

from hydra.routing import (
    MoDConfig,
    MoDRouter,
    MixtureOfDepthsBlock,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def default_config():
    """Default MoD configuration for testing."""
    return MoDConfig(
        dim=256,
        capacity_ratio=0.5,
        aux_loss_weight=0.01,
        max_seq_len=512,
    )


@pytest.fixture
def sample_input():
    """Sample input tensor."""
    return torch.randn(2, 16, 256)


@pytest.fixture
def simple_block():
    """Simple transformer block for testing."""
    return nn.Sequential(
        nn.LayerNorm(256),
        nn.Linear(256, 512),
        nn.GELU(),
        nn.Linear(512, 256),
    )


# =============================================================================
# MoDConfig Tests
# =============================================================================

class TestMoDConfig:
    """Tests for MoDConfig dataclass."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = MoDConfig(dim=512)
        
        assert config.dim == 512
        assert config.capacity_ratio == 0.5
        assert config.aux_loss_weight == 0.01
        assert config.jitter_noise == 0.0
        assert config.max_seq_len == 2048
        assert config.warmup_steps == 100
    
    def test_immutable(self, default_config):
        """Test that config is frozen (immutable)."""
        with pytest.raises(AttributeError):
            default_config.dim = 1024
    
    def test_invalid_dim(self):
        """Test validation rejects invalid dim."""
        with pytest.raises(ValueError, match="dim must be positive"):
            MoDConfig(dim=0)
    
    def test_invalid_capacity_ratio(self):
        """Test validation rejects invalid capacity ratio."""
        with pytest.raises(ValueError, match="capacity_ratio must be in"):
            MoDConfig(dim=512, capacity_ratio=0.0)
        
        with pytest.raises(ValueError, match="capacity_ratio must be in"):
            MoDConfig(dim=512, capacity_ratio=1.5)
    
    def test_invalid_aux_loss_weight(self):
        """Test validation rejects negative aux loss weight."""
        with pytest.raises(ValueError, match="aux_loss_weight must be >= 0"):
            MoDConfig(dim=512, aux_loss_weight=-0.1)
    
    def test_slots(self):
        """Test config uses __slots__ for memory efficiency."""
        config = MoDConfig(dim=512)
        assert hasattr(config, "__slots__")


# =============================================================================
# MoDRouter Tests
# =============================================================================

class TestMoDRouter:
    """Tests for MoDRouter class."""
    
    def test_output_shapes(self, sample_input):
        """Test router output shapes."""
        router = MoDRouter(dim=256, capacity_ratio=0.5)
        mask, indices, scores = router(sample_input, return_scores=True)
        
        B, L, D = sample_input.shape
        k = int(L * 0.5)
        
        assert mask.shape == (B, L)
        assert indices.shape == (B, k)
        assert scores.shape == (B, L)
    
    def test_mask_sum_equals_k(self, sample_input):
        """Test mask has exactly k ones per batch."""
        router = MoDRouter(dim=256, capacity_ratio=0.5)
        mask, indices, _ = router(sample_input)
        
        B, L, D = sample_input.shape
        expected_k = int(L * 0.5)
        
        # Each batch should have exactly k ones
        for b in range(B):
            assert mask[b].sum().item() == expected_k
    
    def test_indices_correspond_to_mask(self, sample_input):
        """Test indices match the mask positions."""
        router = MoDRouter(dim=256, capacity_ratio=0.5)
        mask, indices, _ = router(sample_input)
        
        B, L, D = sample_input.shape
        
        # Indices should point to positions where mask is 1
        for b in range(B):
            for idx in indices[b]:
                assert mask[b, idx] == 1.0
    
    def test_from_config(self, default_config, sample_input):
        """Test router creation from config."""
        router = MoDRouter.from_config(default_config)
        mask, indices, _ = router(sample_input)
        
        assert mask.shape == (2, 16)
    
    def test_capacity_ratio_respected(self):
        """Test different capacity ratios work correctly."""
        for ratio in [0.25, 0.5, 0.75]:
            router = MoDRouter(dim=256, capacity_ratio=ratio)
            x = torch.randn(2, 32, 256)
            mask, indices, _ = router(x)
            
            expected_k = int(32 * ratio)
            assert indices.shape[1] == expected_k
    
    def test_minimum_k_one(self):
        """Test k is at least 1 even with very low capacity."""
        router = MoDRouter(dim=256, capacity_ratio=0.01)  # Very low
        x = torch.randn(2, 10, 256)  # Small sequence
        mask, indices, _ = router(x)
        
        assert indices.shape[1] >= 1
    
    def test_gradient_flow(self, sample_input):
        """Test gradients flow through router (via STE)."""
        router = MoDRouter(dim=256, capacity_ratio=0.5)
        sample_input.requires_grad_(True)
        
        mask, indices, scores = router(sample_input, return_scores=True)
        
        # Scores should have gradients (soft path)
        loss = scores.mean()
        loss.backward()
        
        assert sample_input.grad is not None
        assert not torch.isnan(sample_input.grad).any()
    
    def test_aux_loss_computed(self, sample_input):
        """Test auxiliary loss is computed during training."""
        router = MoDRouter(dim=256, capacity_ratio=0.5, aux_loss_weight=0.01)
        router.train()
        
        _ = router(sample_input)
        aux_loss = router.get_aux_loss()
        
        assert aux_loss is not None
        assert aux_loss.shape == ()
    
    def test_jitter_only_training(self, sample_input):
        """Test jitter is only applied during training."""
        router = MoDRouter(dim=256, capacity_ratio=0.5, jitter_noise=0.1)
        
        # Eval mode should be deterministic
        router.eval()
        torch.manual_seed(42)
        _, indices1, _ = router(sample_input)
        torch.manual_seed(42)
        _, indices2, _ = router(sample_input)
        
        assert torch.equal(indices1, indices2)
    
    def test_deterministic_inference(self, sample_input):
        """Test inference is deterministic."""
        router = MoDRouter(dim=256, capacity_ratio=0.5)
        router.eval()
        
        with torch.no_grad():
            _, indices1, _ = router(sample_input)
            _, indices2, _ = router(sample_input)
        
        assert torch.equal(indices1, indices2)


# =============================================================================
# MixtureOfDepthsBlock Tests
# =============================================================================

class TestMixtureOfDepthsBlock:
    """Tests for MixtureOfDepthsBlock class."""
    
    def test_output_shape(self, simple_block, sample_input):
        """Test block output shape matches input."""
        mod_block = MixtureOfDepthsBlock(
            block=simple_block,
            dim=256,
            capacity_ratio=0.5,
        )
        
        output = mod_block(sample_input)
        assert output.shape == sample_input.shape
    
    def test_residual_connection(self, sample_input):
        """Test unselected tokens use residual (identity)."""
        # Create a block that doubles the input
        class DoublingBlock(nn.Module):
            def forward(self, x):
                return x * 2.0
        
        mod_block = MixtureOfDepthsBlock(
            block=DoublingBlock(),
            dim=256,
            capacity_ratio=0.5,
        )
        
        output = mod_block(sample_input)
        
        # Output should be different from input (some tokens processed)
        # but not all doubled (some skip via residual)
        assert not torch.allclose(output, sample_input)
        assert not torch.allclose(output, sample_input * 2.0)
    
    def test_gradient_flow(self, simple_block, sample_input):
        """Test gradients flow through MoD block."""
        mod_block = MixtureOfDepthsBlock(
            block=simple_block,
            dim=256,
            capacity_ratio=0.5,
        )
        sample_input.requires_grad_(True)
        
        output = mod_block(sample_input)
        loss = output.mean()
        loss.backward()
        
        assert sample_input.grad is not None
        assert not torch.isnan(sample_input.grad).any()
    
    def test_aux_loss_accessible(self, simple_block, sample_input):
        """Test aux loss is accessible from block."""
        mod_block = MixtureOfDepthsBlock(
            block=simple_block,
            dim=256,
            capacity_ratio=0.5,
            aux_loss_weight=0.01,
        )
        mod_block.train()
        
        _ = mod_block(sample_input)
        aux_loss = mod_block.router.get_aux_loss()
        
        assert aux_loss is not None
    
    def test_no_nan_output(self, simple_block, sample_input):
        """Test output has no NaN values."""
        mod_block = MixtureOfDepthsBlock(
            block=simple_block,
            dim=256,
            capacity_ratio=0.5,
        )
        
        output = mod_block(sample_input)
        
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


# =============================================================================
# Integration Tests
# =============================================================================

class TestMoDIntegration:
    """Integration tests for MoD components."""
    
    def test_different_batch_sizes(self, simple_block):
        """Test MoD works with different batch sizes."""
        mod_block = MixtureOfDepthsBlock(
            block=simple_block,
            dim=256,
            capacity_ratio=0.5,
        )
        
        for batch_size in [1, 2, 8, 16]:
            x = torch.randn(batch_size, 32, 256)
            output = mod_block(x)
            assert output.shape == x.shape
    
    def test_different_seq_lengths(self, simple_block):
        """Test MoD works with different sequence lengths."""
        mod_block = MixtureOfDepthsBlock(
            block=simple_block,
            dim=256,
            capacity_ratio=0.5,
        )
        
        for seq_len in [8, 16, 64, 128]:
            x = torch.randn(2, seq_len, 256)
            output = mod_block(x)
            assert output.shape == x.shape
    
    def test_stacked_mod_blocks(self, simple_block):
        """Test multiple MoD blocks can be stacked."""
        blocks = nn.Sequential(
            MixtureOfDepthsBlock(simple_block, dim=256, capacity_ratio=0.5),
            MixtureOfDepthsBlock(simple_block, dim=256, capacity_ratio=0.5),
            MixtureOfDepthsBlock(simple_block, dim=256, capacity_ratio=0.5),
        )
        
        x = torch.randn(2, 16, 256)
        output = blocks(x)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()
    
    def test_backward_through_stacked(self):
        """Test backward pass through stacked MoD blocks."""
        block1 = nn.Linear(256, 256)
        block2 = nn.Linear(256, 256)
        
        blocks = nn.Sequential(
            MixtureOfDepthsBlock(block1, dim=256, capacity_ratio=0.5),
            MixtureOfDepthsBlock(block2, dim=256, capacity_ratio=0.5),
        )
        
        x = torch.randn(2, 16, 256, requires_grad=True)
        output = blocks(x)
        loss = output.mean()
        loss.backward()
        
        assert x.grad is not None
        assert block1.weight.grad is not None
        assert block2.weight.grad is not None


# =============================================================================
# Compute Savings Tests
# =============================================================================

class TestMoDComputeSavings:
    """Tests to verify MoD actually saves compute."""
    
    def test_forward_count_calls(self, sample_input):
        """Test that inner block is called fewer times than full sequence."""
        call_count = [0]
        tokens_seen = [0]
        
        class CountingBlock(nn.Module):
            def forward(self, x):
                call_count[0] += 1
                tokens_seen[0] += x.shape[0] * x.shape[1]
                return x
        
        mod_block = MixtureOfDepthsBlock(
            block=CountingBlock(),
            dim=256,
            capacity_ratio=0.5,
        )
        
        B, L, D = sample_input.shape
        total_tokens = B * L
        
        _ = mod_block(sample_input)
        
        # Should process approximately 50% of tokens
        expected_processed = total_tokens * 0.5
        # Allow some tolerance
        assert tokens_seen[0] <= total_tokens
        assert abs(tokens_seen[0] - expected_processed) < total_tokens * 0.1


# =============================================================================
# GPU Tests
# =============================================================================

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestMoDGPU:
    """GPU-specific tests for MoD."""
    
    def test_router_gpu(self):
        """Test router works on GPU."""
        router = MoDRouter(dim=256, capacity_ratio=0.5).cuda()
        x = torch.randn(2, 16, 256, device="cuda")
        
        mask, indices, scores = router(x, return_scores=True)
        
        assert mask.device.type == "cuda"
        assert indices.device.type == "cuda"
        assert scores.device.type == "cuda"
    
    def test_block_gpu(self):
        """Test MoD block works on GPU."""
        block = nn.Linear(256, 256).cuda()
        mod_block = MixtureOfDepthsBlock(
            block=block,
            dim=256,
            capacity_ratio=0.5,
        ).cuda()
        
        x = torch.randn(2, 16, 256, device="cuda")
        output = mod_block(x)
        
        assert output.device.type == "cuda"
    
    def test_backward_gpu(self):
        """Test backward pass on GPU."""
        block = nn.Linear(256, 256).cuda()
        mod_block = MixtureOfDepthsBlock(
            block=block,
            dim=256,
            capacity_ratio=0.5,
        ).cuda()
        
        x = torch.randn(2, 16, 256, device="cuda", requires_grad=True)
        output = mod_block(x)
        loss = output.mean()
        loss.backward()
        
        assert x.grad is not None
        assert x.grad.device.type == "cuda"
    
    def test_mixed_precision(self):
        """Test MoD with mixed precision (AMP)."""
        block = nn.Linear(256, 256).cuda()
        mod_block = MixtureOfDepthsBlock(
            block=block,
            dim=256,
            capacity_ratio=0.5,
        ).cuda()
        
        x = torch.randn(2, 16, 256, device="cuda")
        
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output = mod_block(x)
        
        assert not torch.isnan(output).any()
