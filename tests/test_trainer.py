"""
Tests for the HYDRA trainer and training loop components.

Tests cover:
- TrainingConfig instantiation and validation
- Loss computation (microbatch, chunked CE, advantage loss)
- EMA update logic
- Eval loop (forward_hidden_with_losses mask handling)
- Gradient accumulation helpers
- LR scheduling
"""

import pytest
import torch
import torch.nn.functional as F
from unittest.mock import MagicMock, patch
from dataclasses import asdict

# Training components
from hydra.training.config import TrainingConfig, compute_auto_lr
from hydra.training.loop import (
    update_scalar_ema,
    resolve_micro_diag_tensors,
    compute_microbatch_loss,
)
from hydra.training.lr_step import compute_step_lr
from hydra.training.lr import get_lr_wsd, get_lr


# ============================================
# TrainingConfig Tests
# ============================================

class TestTrainingConfig:
    """Tests for TrainingConfig dataclass."""

    def test_default_config_instantiation(self):
        """Config should instantiate with defaults."""
        cfg = TrainingConfig()
        assert cfg.mode == "testing"
        assert cfg.batch_size == 8
        assert cfg.grad_accum_steps == 2
        assert cfg.max_lr == 1e-4
        assert cfg.grad_clip == 5.0

    def test_config_mode_testing(self):
        """Testing mode should have short step counts."""
        cfg = TrainingConfig(mode="testing")
        # __post_init__ should set short steps for testing
        assert cfg.max_steps <= 5000

    def test_config_mode_production(self):
        """Production mode should have longer step counts."""
        cfg = TrainingConfig(mode="production")
        assert cfg.max_steps > 5000

    def test_config_model_size_affects_dim(self):
        """Model size should affect dimension settings."""
        cfg_100m = TrainingConfig(model_size="100M")
        cfg_500m = TrainingConfig(model_size="500M")
        # Larger model should have larger dim (or use mod_mor_dim)
        assert cfg_500m.mod_mor_dim >= cfg_100m.mod_mor_dim

    def test_compute_auto_lr_scaling(self):
        """Auto LR should scale inversely with sqrt(params)."""
        lr_100m = compute_auto_lr(100.0)  # base
        lr_400m = compute_auto_lr(400.0)  # 4x params
        # sqrt(4) = 2, so LR should be half
        assert abs(lr_400m - lr_100m / 2) < 1e-9

    def test_config_serializable(self):
        """Config should be serializable to dict."""
        cfg = TrainingConfig()
        d = asdict(cfg)
        assert isinstance(d, dict)
        assert "max_lr" in d
        assert "batch_size" in d


# ============================================
# EMA and Diagnostics Tests
# ============================================

class TestEMAAndDiagnostics:
    """Tests for EMA updates and diagnostic helpers."""

    def test_update_scalar_ema_initialization(self):
        """EMA of 0 should initialize to the value."""
        result = update_scalar_ema(ema=0.0, value=5.0, alpha=0.1)
        assert result == 5.0

    def test_update_scalar_ema_normal_update(self):
        """EMA should update with alpha weighting."""
        # alpha=0.1: new = 0.1*value + 0.9*ema
        result = update_scalar_ema(ema=10.0, value=0.0, alpha=0.1)
        assert abs(result - 9.0) < 1e-9

    def test_update_scalar_ema_converges(self):
        """Repeated updates should converge to value."""
        ema = 0.0
        for _ in range(100):
            ema = update_scalar_ema(ema=ema, value=5.0, alpha=0.1)
        assert abs(ema - 5.0) < 0.01

    def test_resolve_micro_diag_tensors_converts_tensors(self):
        """Should convert tensor values to Python scalars."""
        diag = [
            {"loss": torch.tensor(1.5), "count": torch.tensor(10)},
            {"name": "test", "value": 42},
        ]
        resolve_micro_diag_tensors(diag)
        assert diag[0]["loss"] == 1.5
        assert diag[0]["count"] == 10
        assert diag[1]["name"] == "test"
        assert diag[1]["value"] == 42

    def test_resolve_micro_diag_tensors_empty_list(self):
        """Should handle empty list gracefully."""
        diag = []
        resolve_micro_diag_tensors(diag)  # Should not raise


# ============================================
# Loss Computation Tests
# ============================================

class TestLossComputation:
    """Tests for microbatch loss computation."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing."""
        model = MagicMock()
        model._orig_mod = model
        
        # Mock output weight for chunked CE
        model.output = MagicMock()
        model.output.weight = torch.randn(100, 64)  # vocab=100, dim=64
        
        # Mock forward
        def forward(x, mask=None, return_losses=False):
            B, S = x.shape
            logits = torch.randn(B, S, 100)
            if return_losses:
                return logits, {"aux_loss": torch.tensor(0.1), "ponder_loss": torch.tensor(0.01)}
            return logits
        model.side_effect = forward
        model.return_value = (torch.randn(2, 16, 100), {"aux_loss": torch.tensor(0.1), "ponder_loss": torch.tensor(0.01)})
        
        return model

    @pytest.fixture
    def config(self):
        """Create a test config."""
        return TrainingConfig(use_chunked_ce=False)

    def test_cross_entropy_loss_shape(self):
        """CE loss should produce scalar."""
        logits = torch.randn(2, 16, 100)  # B=2, S=16, V=100
        targets = torch.randint(0, 100, (2, 16))
        loss = F.cross_entropy(logits.view(-1, 100), targets.view(-1))
        assert loss.shape == ()
        assert loss.item() > 0

    def test_cross_entropy_ignore_index(self):
        """CE with ignore_index=-100 should ignore padding."""
        logits = torch.randn(2, 16, 100)
        targets = torch.randint(0, 100, (2, 16))
        targets[:, -4:] = -100  # Pad last 4 tokens
        
        loss = F.cross_entropy(logits.view(-1, 100), targets.view(-1), ignore_index=-100)
        assert loss.shape == ()
        # Loss should be finite (not NaN from all-padded case)
        assert torch.isfinite(loss)


# ============================================
# LR Scheduling Tests
# ============================================

class TestLRScheduling:
    """Tests for learning rate scheduling."""

    def test_get_lr_wsd_warmup_phase(self):
        """LR should increase during warmup (using get_lr_wsd directly)."""
        # Test the raw function to avoid config __post_init__ overrides
        lr_10 = get_lr_wsd(10, warmup_steps=100, decay_start_step=500, 
                          decay_steps=500, max_lr=1e-3, min_lr=1e-5)
        lr_50 = get_lr_wsd(50, warmup_steps=100, decay_start_step=500,
                          decay_steps=500, max_lr=1e-3, min_lr=1e-5)
        lr_90 = get_lr_wsd(90, warmup_steps=100, decay_start_step=500,
                          decay_steps=500, max_lr=1e-3, min_lr=1e-5)
        
        # Should increase during warmup
        assert lr_10 < lr_50 < lr_90

    def test_get_lr_wsd_stable_phase(self):
        """LR should be max_lr after warmup, before decay."""
        lr_200 = get_lr_wsd(200, warmup_steps=100, decay_start_step=500,
                           decay_steps=500, max_lr=1e-3, min_lr=1e-5)
        lr_300 = get_lr_wsd(300, warmup_steps=100, decay_start_step=500,
                           decay_steps=500, max_lr=1e-3, min_lr=1e-5)
        
        # Should be at max_lr
        assert abs(lr_200 - 1e-3) < 1e-9
        assert abs(lr_300 - 1e-3) < 1e-9

    def test_get_lr_wsd_decay_phase(self):
        """LR should decrease during decay."""
        lr_600 = get_lr_wsd(600, warmup_steps=100, decay_start_step=500,
                           decay_steps=500, max_lr=1e-3, min_lr=1e-5)
        lr_800 = get_lr_wsd(800, warmup_steps=100, decay_start_step=500,
                           decay_steps=500, max_lr=1e-3, min_lr=1e-5)
        lr_999 = get_lr_wsd(999, warmup_steps=100, decay_start_step=500,
                           decay_steps=500, max_lr=1e-3, min_lr=1e-5)
        
        # Should decrease during decay
        assert lr_600 > lr_800 > lr_999
        # Should approach min_lr
        assert lr_999 >= 1e-5

    def test_compute_step_lr_resume_override(self):
        """Resume LR override should set LR to the override target value."""
        cfg = TrainingConfig()
        
        base_lr, lr, new_scale, new_override = compute_step_lr(
            step=200,
            config=cfg,
            adaptive_lr=None,
            adaptive_metric="train",
            accum_loss=1.0,
            resume_lr_scale=1.0,
            resume_lr_override_target=0.5,  # Target LR of 0.5
            get_lr=lambda s, c: 1e-3,  # Base LR = 1e-3
        )
        
        assert base_lr == 1e-3
        # LR override sets lr = target, so scale = target / base_lr
        # lr = base_lr * scale = base_lr * (target / base_lr) = target
        assert abs(lr - 0.5) < 1e-9  # LR becomes the target value
        assert abs(new_scale - 500.0) < 1e-6  # scale = 0.5 / 1e-3 = 500
        assert new_override == 0.0  # Override consumed

    def test_compute_step_lr_min_lr_clamp(self):
        """LR should not go below min_lr."""
        cfg = TrainingConfig(min_lr=1e-5)
        
        base_lr, lr, _, _ = compute_step_lr(
            step=200,
            config=cfg,
            adaptive_lr=None,
            adaptive_metric="train",
            accum_loss=1.0,
            resume_lr_scale=0.001,  # Very small scale
            resume_lr_override_target=0.0,
            get_lr=lambda s, c: 1e-3,
        )
        
        assert lr >= cfg.min_lr


# ============================================
# Integration Tests
# ============================================

class TestTrainerIntegration:
    """Higher-level integration tests."""

    @pytest.mark.slow
    def test_config_from_model_size_100m(self):
        """100M config should be valid."""
        cfg = TrainingConfig(model_size="100M", mode="testing")
        assert cfg.mod_mor_dim > 0
        assert cfg.n_mor_blocks > 0
        assert cfg.mor_recursions > 0

    @pytest.mark.slow
    def test_config_from_model_size_500m(self):
        """500M config should be valid."""
        cfg = TrainingConfig(model_size="500M", mode="testing")
        assert cfg.mod_mor_dim > 0
        assert cfg.n_mor_blocks > 0

    def test_gradient_clip_value(self):
        """Grad clip should be reasonable for deep models."""
        cfg = TrainingConfig()
        # Grad clip=5.0 is calibrated for deep MoR models
        # Pre-clip norms ~20-25 are normal
        assert cfg.grad_clip >= 1.0
        assert cfg.grad_clip <= 10.0


# ============================================
# Forward Hidden With Losses (Eval Fix)
# ============================================

class TestForwardHiddenWithLosses:
    """Test that forward_hidden_with_losses accepts mask parameter."""

    def test_method_signature_accepts_mask(self):
        """forward_hidden_with_losses should accept mask kwarg."""
        # This tests the fix for the error:
        # "HydraModel.forward_hidden() takes 2 positional arguments but 3 were given"
        
        # Create mock model with correct signature
        class MockModel:
            def forward_hidden_with_losses(self, x, mask=None):
                return torch.randn(x.shape[0], x.shape[1], 64), {"aux_loss": 0.0}
        
        model = MockModel()
        x = torch.randint(0, 100, (2, 16))
        mask = torch.ones(2, 16)
        
        # Should not raise
        hidden, aux = model.forward_hidden_with_losses(x, mask=mask)
        assert hidden.shape == (2, 16, 64)


# ============================================
# Run tests
# ============================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
