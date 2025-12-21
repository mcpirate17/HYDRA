"""
Tests for CCGQA model architecture validation.

These tests validate the full CCGQA model including:
- CCGQAModel (base model)
- CCGQAMoDMoRModel (full efficiency stack)
- Shape invariants
- Cross-batch independence
- Gradient flow
- Numerical stability in BF16
"""

import pytest
import torch
import torch.nn.functional as F

from hydra.model.ccgqa import (
    CCGQAModel,
    CCGQAMoDMoRModel,
    create_ccgqa_model,
    create_ccgqa_mod_mor_model,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture
def small_ccgqa_model(device):
    """Small CCGQA model for testing."""
    model = CCGQAModel(
        vocab_size=1000,
        dim=128,
        n_layers=4,
        n_heads=4,
        n_kv_heads=2,
        compression_factor=2,
        max_seq_len=512,
    )
    return model.to(device)


@pytest.fixture
def small_mod_mor_model(device):
    """Small MoD+MoR model for testing."""
    model = CCGQAMoDMoRModel(
        vocab_size=1000,
        dim=128,
        n_mor_blocks=4,
        recursions_per_block=2,
        n_heads=4,
        n_kv_heads=2,
        mod_capacity=0.5,
        max_seq_len=512,
    )
    model.train()
    model.set_global_step(200)  # Enable hard routing
    return model.to(device)


# =============================================================================
# CCGQAModel Tests
# =============================================================================

class TestCCGQAModel:
    """Tests for base CCGQA model."""
    
    def test_model_creates(self, small_ccgqa_model):
        """Test model instantiation."""
        assert small_ccgqa_model.dim == 128
        assert small_ccgqa_model.n_layers == 4
        assert len(small_ccgqa_model.layers) == 4
    
    def test_forward_shape(self, small_ccgqa_model):
        """Test forward pass shape."""
        device = small_ccgqa_model.tok_emb.weight.device
        x = torch.randint(0, 1000, (2, 32), device=device)
        out = small_ccgqa_model(x)
        
        assert out.shape == (2, 32, 1000)
    
    def test_parameter_count(self, small_ccgqa_model):
        """Test parameter count is reasonable."""
        total = sum(p.numel() for p in small_ccgqa_model.parameters())
        
        # Should be less than 10M params for this small config
        assert total < 10_000_000
        assert total > 100_000  # But not trivially small
    
    def test_factory_function(self):
        """Test create_ccgqa_model factory."""
        class MockSpec:
            vocab_size = 1000
            dim = 128
            n_layers = 4
            n_heads = 4
            n_kv_heads = 2
            compression_factor = 2
            mlp_ratio = 2.67
            max_seq_len = 512
            tie_weights = True
        
        model = create_ccgqa_model(MockSpec())
        assert model.dim == 128
        assert model.n_layers == 4


# =============================================================================
# CCGQAMoDMoRModel Tests
# =============================================================================

class TestCCGQAMoDMoRModel:
    """Tests for full MoD+MoR model."""
    
    def test_model_creates(self, small_mod_mor_model):
        """Test model instantiation."""
        model = small_mod_mor_model
        assert model.dim == 128
        assert model.n_mor_blocks == 4
        assert model.recursions_per_block == 2
        assert model.effective_layers == 8
    
    def test_shape_invariants(self, small_mod_mor_model):
        """Test output shapes for various batch/seq combinations."""
        model = small_mod_mor_model
        device = model.tok_emb.weight.device
        
        test_shapes = [(1, 16), (2, 32), (4, 64), (1, 128)]
        for batch, seq in test_shapes:
            x = torch.randint(0, 1000, (batch, seq), device=device)
            out, losses = model(x, return_losses=True)
            
            expected_shape = (batch, seq, 1000)
            assert out.shape == expected_shape, \
                f"Shape mismatch: {out.shape} != {expected_shape}"
            assert "aux_loss" in losses
            assert "ponder_loss" in losses
            assert losses["aux_loss"].shape == ()
            assert losses["ponder_loss"].shape == ()
    
    def test_no_cross_batch_mixing(self, small_mod_mor_model):
        """Test that batching doesn't cause cross-sequence contamination."""
        model = small_mod_mor_model
        device = model.tok_emb.weight.device
        model.eval()
        
        torch.manual_seed(42)
        seq_a = torch.randint(0, 1000, (1, 32), device=device)
        seq_b = torch.randint(0, 1000, (1, 32), device=device)
        
        # Run separately
        with torch.no_grad():
            out_a_solo = model(seq_a)
        
        # Run batched
        with torch.no_grad():
            batched = torch.cat([seq_a, seq_b], dim=0)
            out_batched = model(batched)
        
        # First element should match (within tolerance)
        diff = (out_a_solo - out_batched[0:1]).abs().max()
        assert diff < 1e-4, f"Cross-batch contamination detected: diff={diff}"
    
    def test_routing_modes(self, small_mod_mor_model):
        """Test soft vs hard routing modes."""
        model = small_mod_mor_model
        device = model.tok_emb.weight.device
        model.train()
        
        x = torch.randint(0, 1000, (2, 32), device=device)
        
        # Soft routing (warmup)
        model.set_global_step(50)
        out_soft, losses_soft = model(x, return_losses=True)
        
        # Hard routing (post-warmup)
        model.set_global_step(200)
        out_hard, losses_hard = model(x, return_losses=True)
        
        # Both should produce valid outputs
        assert out_soft.shape == out_hard.shape
        assert not torch.isnan(out_soft).any()
        assert not torch.isnan(out_hard).any()
    
    def test_gradient_flow(self, small_mod_mor_model):
        """Test gradients flow through all components."""
        model = small_mod_mor_model
        device = model.tok_emb.weight.device
        model.train()
        model.zero_grad()
        
        x = torch.randint(0, 1000, (2, 32), device=device)
        targets = torch.randint(0, 1000, (2, 32), device=device)
        
        out, losses = model(x, return_losses=True)
        total_loss = (
            F.cross_entropy(out.view(-1, 1000), targets.view(-1)) +
            losses["aux_loss"] +
            losses["ponder_loss"]
        )
        total_loss.backward()
        
        # Count parameters with gradients
        grad_params = sum(1 for p in model.parameters() if p.grad is not None)
        total_params = sum(1 for _ in model.parameters())
        
        # Most params should have gradients
        assert grad_params > total_params * 0.8, \
            f"Too few params have gradients: {grad_params}/{total_params}"
        
        # Critical components should have gradients
        assert model.tok_emb.weight.grad is not None
        assert model.output.weight.grad is not None
    
    def test_bf16_stability(self, small_mod_mor_model):
        """Test numerical stability in BF16."""
        model = small_mod_mor_model.bfloat16()
        device = model.tok_emb.weight.device
        model.train()
        model.zero_grad()
        
        x = torch.randint(0, 1000, (2, 32), device=device)
        targets = torch.randint(0, 1000, (2, 32), device=device)
        
        out, losses = model(x, return_losses=True)
        
        # Check for NaN/Inf
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()
        assert not torch.isnan(losses["aux_loss"])
        assert not torch.isnan(losses["ponder_loss"])
        
        # Backward should also work
        loss = (
            F.cross_entropy(out.view(-1, 1000), targets.view(-1)) +
            losses["aux_loss"] +
            losses["ponder_loss"]
        )
        loss.backward()
        
        # Check gradients for NaN
        nan_grads = sum(
            1 for p in model.parameters()
            if p.grad is not None and torch.isnan(p.grad).any()
        )
        assert nan_grads == 0, f"Found {nan_grads} parameters with NaN gradients"
    
    def test_loss_baseline_update(self, small_mod_mor_model):
        """Test loss baseline tracking."""
        model = small_mod_mor_model
        device = model.tok_emb.weight.device
        model.train()
        
        x = torch.randint(0, 1000, (2, 32), device=device)
        targets = torch.randint(0, 1000, (2, 32), device=device)
        
        out, _ = model(x, return_losses=True)
        
        # Update baseline
        advantage_loss = model.update_loss_baseline(out, targets)
        
        # Should return a scalar
        assert advantage_loss.shape == ()
        assert not torch.isnan(advantage_loss)
    
    def test_mor_curriculum(self, small_mod_mor_model):
        """Test MoR curriculum scheduling."""
        model = small_mod_mor_model
        
        # Set curriculum
        model.set_mor_curriculum(enable_step=1000, rampup_steps=500)
        
        # Before enable
        model.set_global_step(500)
        assert not model.is_mor_adaptive_enabled()
        
        # After enable but during rampup
        model.set_global_step(1250)
        assert model.is_mor_adaptive_enabled()
        
        # After rampup
        model.set_global_step(1600)
        assert model.is_mor_adaptive_enabled()
    
    def test_factory_function(self):
        """Test create_ccgqa_mod_mor_model factory."""
        model = create_ccgqa_mod_mor_model(
            vocab_size=1000,
            dim=128,
            n_mor_blocks=4,
            recursions_per_block=2,
            n_heads=4,
            n_kv_heads=2,
        )
        
        assert model.dim == 128
        assert model.n_mor_blocks == 4
        assert model.effective_layers == 8


# =============================================================================
# Integration Tests
# =============================================================================

class TestModelIntegration:
    """Integration tests for full model stack."""
    
    def test_efficient_inference(self, small_mod_mor_model):
        """Test inference mode (no losses)."""
        model = small_mod_mor_model
        device = model.tok_emb.weight.device
        model.eval()
        
        x = torch.randint(0, 1000, (2, 32), device=device)
        
        with torch.no_grad():
            out = model(x, return_losses=False)
        
        assert out.shape == (2, 32, 1000)
        assert not torch.isnan(out).any()
    
    def test_training_mode(self, small_mod_mor_model):
        """Test training mode (with losses)."""
        model = small_mod_mor_model
        device = model.tok_emb.weight.device
        model.train()
        
        x = torch.randint(0, 1000, (2, 32), device=device)
        out, losses = model(x, return_losses=True)
        
        assert "aux_loss" in losses
        assert "ponder_loss" in losses
        
        # Losses should be reasonable magnitudes
        assert 0 <= losses["aux_loss"] < 1.0
        assert 0 <= losses["ponder_loss"] < 10.0
    
    def test_gradient_checkpointing(self, small_mod_mor_model):
        """Test gradient checkpointing doesn't break forward/backward."""
        model = small_mod_mor_model
        device = model.tok_emb.weight.device
        model.enable_gradient_checkpointing()
        model.train()
        
        x = torch.randint(0, 1000, (2, 32), device=device)
        targets = torch.randint(0, 1000, (2, 32), device=device)
        
        out, losses = model(x, return_losses=True)
        loss = F.cross_entropy(out.view(-1, 1000), targets.view(-1))
        loss.backward()
        
        # Should have gradients
        assert model.tok_emb.weight.grad is not None
        
        model.disable_gradient_checkpointing()
    
    def test_rope_cache_resize(self, small_mod_mor_model):
        """Test RoPE cache resizing."""
        model = small_mod_mor_model
        device = model.tok_emb.weight.device
        
        # Resize to larger
        model.resize_rope_cache(1024)
        assert model.max_seq_len == 1024
        
        # Should still work
        x = torch.randint(0, 1000, (2, 128), device=device)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 128, 1000)
