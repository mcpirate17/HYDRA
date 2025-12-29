"""
Tests for Loss Tracker module.

Tests cover:
- MovingAverageBaseline EMA updates
- Advantage computation and normalization
- Warmup behavior
- AdvantageScaledSTE gradient flow
"""

import pytest
import torch
import torch.nn as nn

from hydra.routing import (
    MovingAverageBaseline,
    AdvantageScaledSTE,
    apply_advantage_ste,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def baseline():
    """Default baseline for testing."""
    return MovingAverageBaseline(decay=0.99, warmup_steps=100)


@pytest.fixture
def fast_baseline():
    """Baseline with faster adaptation."""
    return MovingAverageBaseline(decay=0.9, warmup_steps=10)


@pytest.fixture
def sample_losses():
    """Sample per-token losses."""
    return torch.rand(2, 16) * 5.0 + 1.0  # Losses in [1, 6]


# =============================================================================
# MovingAverageBaseline Tests
# =============================================================================

class TestMovingAverageBaseline:
    """Tests for MovingAverageBaseline class."""
    
    def test_initial_state(self, baseline):
        """Test initial baseline state."""
        assert baseline.step == 0
        assert baseline.baseline == 0.0
        assert not baseline.is_active
    
    def test_first_update_initializes(self, baseline, sample_losses):
        """Test first update initializes EMA."""
        baseline.update(sample_losses)
        
        expected_mean = sample_losses.mean().item()
        assert abs(baseline.baseline - expected_mean) < 0.01
        assert baseline.step == 1
    
    def test_ema_decay(self, fast_baseline):
        """Test EMA decay behavior."""
        # First update
        losses1 = torch.ones(2, 16) * 2.0
        fast_baseline.update(losses1)
        baseline1 = fast_baseline.baseline
        
        # Second update with different loss
        losses2 = torch.ones(2, 16) * 4.0
        fast_baseline.update(losses2)
        baseline2 = fast_baseline.baseline
        
        # Baseline should move toward new loss
        assert baseline2 > baseline1
        # But not all the way (EMA)
        assert baseline2 < 4.0
    
    def test_warmup_behavior(self, baseline, sample_losses):
        """Test warmup disables advantage computation."""
        # During warmup
        baseline.update(sample_losses)
        advantage = baseline.compute_advantage(sample_losses)
        
        # Should be zeros during warmup
        assert torch.all(advantage == 0.0)
        assert not baseline.is_active
    
    def test_after_warmup_active(self, baseline, sample_losses):
        """Test baseline becomes active after warmup."""
        # Run through warmup
        for _ in range(100):
            baseline.update(sample_losses)
        
        assert baseline.is_active
    
    def test_advantage_sign(self, fast_baseline):
        """Test advantage sign is correct."""
        # Build up baseline with medium losses
        for _ in range(20):
            medium_losses = torch.ones(2, 16) * 3.0
            fast_baseline.update(medium_losses)
        
        # High losses should have positive advantage
        high_losses = torch.ones(2, 16) * 5.0
        advantage_high = fast_baseline.compute_advantage(high_losses)
        
        # Low losses should have negative advantage
        low_losses = torch.ones(2, 16) * 1.0
        advantage_low = fast_baseline.compute_advantage(low_losses)
        
        assert advantage_high.mean() > 0
        assert advantage_low.mean() < 0
    
    def test_advantage_normalization(self, fast_baseline):
        """Test advantage is normalized by std."""
        # Build baseline
        for _ in range(20):
            fast_baseline.update(torch.randn(2, 16).abs() + 2.0)
        
        # Compute advantage
        test_losses = torch.randn(2, 16).abs() + 2.0
        advantage = fast_baseline.compute_advantage(test_losses)
        
        # Advantage should be in reasonable range due to tanh
        assert advantage.min() >= -1.0
        assert advantage.max() <= 1.0
    
    def test_advantage_tanh_clamp(self, fast_baseline):
        """Test advantage is soft-clamped with tanh."""
        # Build baseline
        for _ in range(20):
            fast_baseline.update(torch.ones(2, 16) * 3.0)
        
        # Extreme losses
        extreme_high = torch.ones(2, 16) * 100.0
        extreme_low = torch.ones(2, 16) * 0.01
        
        adv_high = fast_baseline.compute_advantage(extreme_high)
        adv_low = fast_baseline.compute_advantage(extreme_low)
        
        # Should be clamped to [-1, 1] by tanh
        assert adv_high.max() <= 1.0
        assert adv_low.min() >= -1.0
    
    def test_state_dict(self, baseline, sample_losses):
        """Test state dict for checkpointing."""
        baseline.update(sample_losses)
        baseline.update(sample_losses * 2)
        
        state = baseline.state_dict_extra()
        
        assert "ema" in state
        assert "step" in state
        assert state["step"] == 2
    
    def test_load_state_dict(self, baseline):
        """Test loading state from checkpoint."""
        state = {
            "ema": 3.5,
            "ema_var": 1.0,
            "step": 150,
            "initialized": True,
        }
        
        baseline.load_state_dict_extra(state)
        
        assert baseline.baseline == 3.5
        assert baseline.step == 150
        assert baseline.is_active  # Past warmup
    
    def test_std_floor(self, fast_baseline):
        """Test std has a minimum floor to prevent division by zero."""
        # All same losses = zero variance
        for _ in range(20):
            fast_baseline.update(torch.ones(2, 16) * 3.0)
        
        # std should be floored at 0.01
        assert fast_baseline.baseline_std >= 0.01


# =============================================================================
# AdvantageScaledSTE Tests
# =============================================================================

class TestAdvantageScaledSTE:
    """Tests for AdvantageScaledSTE autograd function."""
    
    def test_forward_returns_depths(self):
        """Test forward returns depths unchanged."""
        router_logits = torch.randn(2, 16, requires_grad=True)
        depths = torch.randint(0, 5, (2, 16))
        advantages = torch.randn(2, 16)
        
        output = AdvantageScaledSTE.apply(router_logits, depths, advantages)
        
        assert torch.allclose(output, depths.float())
    
    def test_backward_scales_gradient(self):
        """Test backward scales gradient by advantage."""
        router_logits = torch.randn(2, 16, requires_grad=True)
        depths = torch.randint(0, 5, (2, 16))
        advantages = torch.ones(2, 16) * 0.5  # Uniform positive advantage
        
        output = AdvantageScaledSTE.apply(router_logits, depths, advantages)
        
        # Backward pass
        grad_output = torch.ones_like(output)
        output.sum().backward()
        
        # Gradient should be scaled by (1 + advantage) = 1.5
        expected_grad = torch.ones_like(router_logits) * 1.5
        assert torch.allclose(router_logits.grad, expected_grad)
    
    def test_positive_advantage_increases_gradient(self):
        """Test positive advantage increases gradient magnitude."""
        router_logits1 = torch.randn(2, 16, requires_grad=True)
        router_logits2 = router_logits1.clone().detach().requires_grad_(True)
        depths = torch.randint(0, 5, (2, 16))
        
        # Zero advantage
        adv_zero = torch.zeros(2, 16)
        out1 = AdvantageScaledSTE.apply(router_logits1, depths, adv_zero)
        out1.sum().backward()
        
        # Positive advantage
        adv_pos = torch.ones(2, 16) * 0.5
        out2 = AdvantageScaledSTE.apply(router_logits2, depths, adv_pos)
        out2.sum().backward()
        
        # Gradient with positive advantage should be larger
        assert router_logits2.grad.abs().mean() > router_logits1.grad.abs().mean()
    
    def test_negative_advantage_decreases_gradient(self):
        """Test negative advantage decreases gradient magnitude."""
        router_logits1 = torch.randn(2, 16, requires_grad=True)
        router_logits2 = router_logits1.clone().detach().requires_grad_(True)
        depths = torch.randint(0, 5, (2, 16))
        
        # Zero advantage
        adv_zero = torch.zeros(2, 16)
        out1 = AdvantageScaledSTE.apply(router_logits1, depths, adv_zero)
        out1.sum().backward()
        
        # Negative advantage
        adv_neg = torch.ones(2, 16) * -0.5
        out2 = AdvantageScaledSTE.apply(router_logits2, depths, adv_neg)
        out2.sum().backward()
        
        # Gradient with negative advantage should be smaller
        assert router_logits2.grad.abs().mean() < router_logits1.grad.abs().mean()


class TestApplyAdvantageSTE:
    """Tests for apply_advantage_ste convenience function."""
    
    def test_same_as_class(self):
        """Test convenience function matches class directly."""
        router_logits = torch.randn(2, 16, requires_grad=True)
        depths = torch.randint(0, 5, (2, 16))
        advantages = torch.randn(2, 16)
        
        output = apply_advantage_ste(router_logits, depths, advantages)
        
        assert torch.allclose(output, depths.float())


# =============================================================================
# Integration Tests
# =============================================================================

class TestLossTrackerIntegration:
    """Integration tests for loss-driven routing."""
    
    def test_full_training_loop_simulation(self):
        """Simulate a training loop with loss-driven routing."""
        baseline = MovingAverageBaseline(decay=0.99, warmup_steps=10)
        
        # Simulated "model" - just a linear layer
        model = nn.Linear(32, 10)
        optimizer = torch.optim.Adam(model.parameters())
        
        for step in range(50):
            # Forward pass
            x = torch.randn(4, 32)
            logits = model(x)
            targets = torch.randint(0, 10, (4,))
            
            # Per-token loss (in real case, this would be per-position)
            ce_loss = nn.functional.cross_entropy(logits, targets, reduction='none')
            
            # Update baseline
            baseline.update(ce_loss)
            
            # Compute advantage (would be used for router)
            advantage = baseline.compute_advantage(ce_loss)
            
            # Backward
            loss = ce_loss.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # After training, baseline should be active
        assert baseline.is_active
        # Baseline should have adapted to loss level
        assert baseline.baseline > 0
    
    def test_advantage_differentiates_difficulty(self):
        """Test advantage correctly differentiates easy vs hard tokens."""
        baseline = MovingAverageBaseline(decay=0.9, warmup_steps=5)
        
        # Train on medium difficulty
        for _ in range(10):
            medium = torch.ones(8) * 2.5
            baseline.update(medium)
        
        # Create batch with easy and hard tokens (narrower range to avoid saturation)
        mixed_losses = torch.tensor([2.0, 2.2, 2.4, 2.5, 2.6, 2.8, 3.0, 3.2])
        advantage = baseline.compute_advantage(mixed_losses)
        
        # Easy tokens (low loss relative to baseline 2.5) should have negative advantage
        assert advantage[0] < 0  # loss=2.0, easiest
        assert advantage[1] < 0  # loss=2.2, easy
        
        # Hard tokens (high loss relative to baseline 2.5) should have positive advantage
        assert advantage[6] > 0  # loss=3.0, hard
        assert advantage[7] > 0  # loss=3.2, hardest
        
        # Monotonicity: higher loss = higher advantage (before tanh saturation)
        for i in range(len(advantage) - 1):
            assert advantage[i] <= advantage[i + 1]


# =============================================================================
# GPU Tests
# =============================================================================

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestLossTrackerGPU:
    """GPU-specific tests for loss tracker."""
    
    def test_baseline_gpu(self):
        """Test baseline works on GPU."""
        baseline = MovingAverageBaseline(decay=0.99, warmup_steps=10)
        baseline.cuda()
        
        losses = torch.randn(2, 16, device="cuda").abs() + 1.0
        baseline.update(losses)
        
        advantage = baseline.compute_advantage(losses)
        assert advantage.device.type == "cuda"
    
    def test_ste_gpu(self):
        """Test STE works on GPU."""
        router_logits = torch.randn(2, 16, device="cuda", requires_grad=True)
        depths = torch.randint(0, 5, (2, 16), device="cuda")
        advantages = torch.randn(2, 16, device="cuda")
        
        output = apply_advantage_ste(router_logits, depths, advantages)
        output.sum().backward()
        
        assert router_logits.grad.device.type == "cuda"
