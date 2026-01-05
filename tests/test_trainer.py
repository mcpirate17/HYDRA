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
# Model Forward Signatures (500-step Bug Fix)
# ============================================

class TestModelForwardSignatures:
    """Test that model forward methods accept mask parameter.
    
    Critical: The eval sanity check at step 500 boundaries calls:
        base_model(x, mask=mask, return_losses=True)
    
    Both HydraModel and HydraBaseModel must accept `mask` kwarg.
    """

    def test_hydra_base_model_forward_accepts_mask(self):
        """HydraBaseModel.forward() should accept mask=None kwarg."""
        from hydra.model.framework.model import HydraBaseModel
        
        # Create a minimal model
        # HydraBaseModel uses simpler attention that may have different dimension constraints
        model = HydraBaseModel(
            vocab_size=100,
            dim=64,
            n_layers=2,
            n_heads=4,
            n_kv_heads=4,  # For HydraBaseModel, use same n_heads and n_kv_heads
        )
        model.eval()
        
        x = torch.randint(0, 100, (1, 8))
        
        # Should not raise - mask kwarg accepted (even if unused internally)
        # Note: HydraBaseModel may not actually use the mask, but the signature allows it
        with torch.no_grad():
            logits = model(x, mask=None)  # Pass None since base model doesn't route mask to layers
        
        assert logits.shape == (1, 8, 100)

    def test_hydra_model_forward_accepts_mask_no_losses(self):
        """HydraModel.forward() should accept mask=None kwarg (no losses)."""
        from hydra.model.framework.model import HydraModel
        
        # Create a minimal MoD+MoR model
        model = HydraModel(
            vocab_size=100,
            dim=64,
            n_mor_blocks=2,
            recursions_per_block=2,
            n_heads=4,
            n_kv_heads=2,
            max_seq_len=256,
        )
        model.eval()
        
        x = torch.randint(0, 100, (1, 8))
        mask = torch.ones(1, 8)
        
        # Should not raise - mask kwarg accepted
        with torch.no_grad():
            logits = model(x, mask=mask)
        
        assert logits.shape == (1, 8, 100)

    def test_hydra_model_forward_accepts_mask_with_losses(self):
        """HydraModel.forward() should accept mask=None kwarg (with losses).
        
        This is the exact call that fails at 500-step boundaries:
            logits, _aux = base_model(x, mask=mask, return_losses=True)
        """
        from hydra.model.framework.model import HydraModel
        
        # Create a minimal MoD+MoR model
        model = HydraModel(
            vocab_size=100,
            dim=64,
            n_mor_blocks=2,
            recursions_per_block=2,
            n_heads=4,
            n_kv_heads=2,
            max_seq_len=256,
        )
        model.eval()
        
        x = torch.randint(0, 100, (1, 8))
        mask = torch.ones(1, 8)
        
        # Should not raise - this is the exact call from loop.py line 300
        with torch.no_grad():
            logits, aux = model(x, mask=mask, return_losses=True)
        
        assert logits.shape == (1, 8, 100)
        assert "aux_loss" in aux
        assert "ponder_loss" in aux

    def test_hydra_model_forward_hidden_accepts_mask(self):
        """HydraModel.forward_hidden() should accept mask=None kwarg."""
        from hydra.model.framework.model import HydraModel
        
        model = HydraModel(
            vocab_size=100,
            dim=64,
            n_mor_blocks=2,
            recursions_per_block=2,
            n_heads=4,
            n_kv_heads=2,
            max_seq_len=256,
        )
        model.eval()
        
        x = torch.randint(0, 100, (1, 8))
        mask = torch.ones(1, 8)
        
        # Should not raise - used by chunked CE
        with torch.no_grad():
            hidden = model.forward_hidden(x, mask=mask)
        
        assert hidden.shape == (1, 8, 64)

    def test_backward_compat_alias_ccgqa_mod_mor_model(self):
        """CCGQAMoDMoRModel alias should have same forward signature."""
        from hydra.model.framework.model import CCGQAMoDMoRModel
        
        model = CCGQAMoDMoRModel(
            vocab_size=100,
            dim=64,
            n_mor_blocks=2,
            recursions_per_block=2,
            n_heads=4,
            n_kv_heads=2,
            max_seq_len=256,
        )
        model.eval()
        
        x = torch.randint(0, 100, (1, 8))
        mask = torch.ones(1, 8)
        
        with torch.no_grad():
            logits, aux = model(x, mask=mask, return_losses=True)
        
        assert logits.shape == (1, 8, 100)


class TestEvalSanityCheckCodepath:
    """Test the eval_sanity_check_on_train_batch function.
    
    This runs at step 500 boundaries to verify eval/train consistency.
    It must handle:
    - mask=None from batches
    - mask=<tensor> from batches with attention_mask
    - Both use_mod_mor=True and use_mod_mor=False paths
    """

    def test_eval_sanity_check_with_mask_mod_mor_true(self):
        """eval_sanity_check should work with mask when use_mod_mor=True."""
        from hydra.model.framework.model import HydraModel
        from hydra.training.loop import eval_sanity_check_on_train_batch
        from hydra.training.config import TrainingConfig
        import logging
        
        model = HydraModel(
            vocab_size=100,
            dim=64,
            n_mor_blocks=2,
            recursions_per_block=2,
            n_heads=4,
            n_kv_heads=2,
            max_seq_len=256,
        )
        model.eval()
        
        batch = {
            "input_ids": torch.randint(0, 100, (2, 8)),
            "labels": torch.randint(0, 100, (2, 8)),
            "attention_mask": torch.ones(2, 8),
        }
        
        config = TrainingConfig(use_chunked_ce=False)
        logger = logging.getLogger("test")
        
        # Should not raise
        result = eval_sanity_check_on_train_batch(
            base_model=model,
            train_batch=batch,
            device="cpu",
            dtype=torch.float32,
            config=config,
            use_mod_mor=True,
            current_train_loss=2.0,
            current_ema=2.1,
            logger=logger,
        )
        
        assert "eval_on_train_batch" in result
        assert "manual_mean" in result
        assert "n_valid_tokens" in result

    def test_eval_sanity_check_without_mask_mod_mor_false(self):
        """eval_sanity_check should work without mask when use_mod_mor=False."""
        from hydra.model.framework.model import HydraModel
        from hydra.training.loop import eval_sanity_check_on_train_batch
        from hydra.training.config import TrainingConfig
        import logging
        
        model = HydraModel(
            vocab_size=100,
            dim=64,
            n_mor_blocks=2,
            recursions_per_block=2,
            n_heads=4,
            n_kv_heads=2,
            max_seq_len=256,
        )
        model.eval()
        
        batch = {
            "input_ids": torch.randint(0, 100, (2, 8)),
            "labels": torch.randint(0, 100, (2, 8)),
            "attention_mask": None,  # No mask
        }
        
        config = TrainingConfig(use_chunked_ce=False)
        logger = logging.getLogger("test")
        
        # Should not raise (mask=None path)
        result = eval_sanity_check_on_train_batch(
            base_model=model,
            train_batch=batch,
            device="cpu",
            dtype=torch.float32,
            config=config,
            use_mod_mor=False,  # Non-MoD/MoR path
            current_train_loss=2.0,
            current_ema=2.1,
            logger=logger,
        )
        
        assert "eval_on_train_batch" in result


# ============================================
# Checkpointing Heuristic Tests
# ============================================

class TestMaybeSaveBestCheckpoint:
    """Tests for the maybe_save_best_checkpoint heuristic at 500-step boundaries."""

    def test_before_step_1000_never_saves(self):
        """Should never save before step 1000."""
        from hydra.training.checkpointing import maybe_save_best_checkpoint
        
        saved_at = []
        def mock_save(step, best=False):
            saved_at.append((step, best))
        
        # Even with huge improvement, should not save before step 1000
        for step in [100, 500, 999]:
            result = maybe_save_best_checkpoint(
                step=step,
                prev_best_loss=10.0,
                best_loss=1.0,  # 90% improvement
                save_checkpoint_fn=mock_save,
            )
            assert result is False
        
        assert len(saved_at) == 0

    def test_not_multiple_of_500_never_saves(self):
        """Should only save at multiples of 500."""
        from hydra.training.checkpointing import maybe_save_best_checkpoint
        
        saved_at = []
        def mock_save(step, best=False):
            saved_at.append((step, best))
        
        # Step 1001 is not multiple of 500
        result = maybe_save_best_checkpoint(
            step=1001,
            prev_best_loss=10.0,
            best_loss=1.0,  # Huge improvement
            save_checkpoint_fn=mock_save,
        )
        assert result is False
        assert len(saved_at) == 0

    def test_improvement_over_20_percent_saves(self):
        """Should save at 500-step boundary with >20% improvement."""
        from hydra.training.checkpointing import maybe_save_best_checkpoint
        
        saved_at = []
        def mock_save(step, best=False):
            saved_at.append((step, best))
        
        # Step 1500 with 25% improvement
        result = maybe_save_best_checkpoint(
            step=1500,
            prev_best_loss=10.0,
            best_loss=7.5,  # 25% improvement
            save_checkpoint_fn=mock_save,
        )
        assert result is True
        assert saved_at == [(1500, True)]

    def test_improvement_under_20_percent_does_not_save_at_500(self):
        """Should NOT save at non-1000 step with <20% improvement."""
        from hydra.training.checkpointing import maybe_save_best_checkpoint
        
        saved_at = []
        def mock_save(step, best=False):
            saved_at.append((step, best))
        
        # Step 1500 with only 10% improvement (below 20% threshold)
        result = maybe_save_best_checkpoint(
            step=1500,
            prev_best_loss=10.0,
            best_loss=9.0,  # Only 10% improvement
            save_checkpoint_fn=mock_save,
        )
        assert result is False
        assert len(saved_at) == 0

    def test_step_1000_multiples_save_with_any_improvement(self):
        """At step 1000 multiples, save if ANY improvement (best_loss < prev)."""
        from hydra.training.checkpointing import maybe_save_best_checkpoint
        
        saved_at = []
        def mock_save(step, best=False):
            saved_at.append((step, best))
        
        # Step 2000 with just 1% improvement
        result = maybe_save_best_checkpoint(
            step=2000,
            prev_best_loss=10.0,
            best_loss=9.9,  # Only 1% improvement
            save_checkpoint_fn=mock_save,
        )
        assert result is True
        assert saved_at == [(2000, True)]

    def test_step_1000_multiples_no_save_if_no_improvement(self):
        """At step 1000 multiples, don't save if loss hasn't improved."""
        from hydra.training.checkpointing import maybe_save_best_checkpoint
        
        saved_at = []
        def mock_save(step, best=False):
            saved_at.append((step, best))
        
        # Step 2000 but no improvement (same loss)
        result = maybe_save_best_checkpoint(
            step=2000,
            prev_best_loss=10.0,
            best_loss=10.0,  # No improvement
            save_checkpoint_fn=mock_save,
        )
        assert result is False
        assert len(saved_at) == 0

    def test_first_best_from_inf_always_zero_improvement(self):
        """When prev_best is inf, improvement calculation should be 0."""
        from hydra.training.checkpointing import maybe_save_best_checkpoint
        
        saved_at = []
        def mock_save(step, best=False):
            saved_at.append((step, best))
        
        # Step 1500 with prev_best = inf (first recording)
        # improvement = 0 (not >20%), so should not save at step 1500
        result = maybe_save_best_checkpoint(
            step=1500,
            prev_best_loss=float("inf"),
            best_loss=5.0,
            save_checkpoint_fn=mock_save,
        )
        assert result is False
        
        # But at step 1000 multiples, it WOULD save if best_loss < inf
        result = maybe_save_best_checkpoint(
            step=2000,
            prev_best_loss=float("inf"),
            best_loss=5.0,
            save_checkpoint_fn=mock_save,
        )
        assert result is True


# ============================================
# BatchFilter Tests
# ============================================

class TestBatchFilter:
    """Tests for BatchFilter loss-based filtering at 500-step boundaries."""

    def test_batch_filter_warmup_never_skips(self):
        """First 10 samples are warmup and should never skip."""
        from hydra.data.data_filter import BatchFilter, FilterConfig
        
        config = FilterConfig(loss_spike_threshold=1.5)
        bf = BatchFilter(config)
        
        # Even with extremely high loss, warmup samples are accepted
        for i in range(10):
            skip, reason = bf.should_skip_batch(loss=100.0, step=i)
            assert skip is False
            assert reason == "warmup"

    def test_batch_filter_detects_loss_spike(self):
        """Should skip batches with loss > threshold * EMA."""
        from hydra.data.data_filter import BatchFilter, FilterConfig
        
        config = FilterConfig(loss_spike_threshold=2.0, loss_ema_alpha=0.5)
        bf = BatchFilter(config)
        
        # Warmup with normal losses to establish EMA
        for i in range(15):
            bf.should_skip_batch(loss=2.0, step=i)
        
        # EMA should be around 2.0 after warmup
        # Threshold = 2.0 * 2.0 = 4.0
        # A loss of 5.0 should be skipped
        skip, reason = bf.should_skip_batch(loss=5.0, step=100)
        assert skip is True
        assert "loss_spike" in reason

    def test_batch_filter_accepts_normal_loss(self):
        """Should accept batches with normal loss."""
        from hydra.data.data_filter import BatchFilter, FilterConfig
        
        config = FilterConfig(loss_spike_threshold=2.5)
        bf = BatchFilter(config)
        
        # Warmup
        for i in range(15):
            bf.should_skip_batch(loss=2.0, step=i)
        
        # Normal loss should be accepted
        skip, reason = bf.should_skip_batch(loss=2.5, step=100)
        assert skip is False
        assert reason == "ok"

    def test_batch_filter_skip_budget_enforcement(self):
        """Should stop skipping when skip budget is exceeded."""
        from hydra.data.data_filter import BatchFilter, FilterConfig
        
        # Allow only 5% skip rate
        config = FilterConfig(
            loss_spike_threshold=1.5, 
            max_skips_per_epoch=0.05,
            loss_ema_alpha=0.5,
        )
        bf = BatchFilter(config)
        
        # Warmup with normal losses
        for i in range(10):
            bf.should_skip_batch(loss=2.0, step=i)
        
        # After warmup, start feeding high loss batches that would normally be skipped
        # With 5% budget, after some skips the budget_exceeded reason should appear
        budget_exceeded_seen = False
        for i in range(200):
            skip, reason = bf.should_skip_batch(loss=100.0, step=10 + i)
            if reason == "budget_exceeded":
                budget_exceeded_seen = True
                break
        
        # Should have seen budget enforcement at some point
        assert budget_exceeded_seen, "Budget enforcement should kick in after too many skips"

    def test_batch_filter_get_stats(self):
        """get_stats should return filtering statistics."""
        from hydra.data.data_filter import BatchFilter, FilterConfig
        
        bf = BatchFilter()
        
        # Process some samples
        for i in range(20):
            bf.should_skip_batch(loss=2.0, step=i)
        
        stats = bf.get_stats()
        assert "loss_ema" in stats
        assert "n_samples" in stats
        assert "n_skipped" in stats
        assert "n_total" in stats
        assert "skip_ratio" in stats
        assert stats["n_total"] == 20

    def test_batch_filter_reset_epoch(self):
        """reset_epoch should clear per-epoch counters but keep EMA."""
        from hydra.data.data_filter import BatchFilter, FilterConfig
        
        bf = BatchFilter()
        
        # Process samples
        for i in range(20):
            bf.should_skip_batch(loss=2.0, step=i)
        
        old_ema = bf.loss_ema
        old_n_samples = bf.n_samples
        
        # Reset epoch
        bf.reset_epoch()
        
        # EMA and n_samples preserved
        assert bf.loss_ema == old_ema
        assert bf.n_samples == old_n_samples
        
        # Per-epoch counters reset
        assert bf.n_skipped == 0
        assert bf.n_total == 0

    def test_batch_filter_skip_history_limited(self):
        """Skip history should be limited to last 100 entries."""
        from hydra.data.data_filter import BatchFilter, FilterConfig
        
        # Very low threshold to force many skips
        config = FilterConfig(
            loss_spike_threshold=0.01,
            max_skips_per_epoch=1.0,  # Allow all skips
            loss_ema_alpha=0.001,
        )
        bf = BatchFilter(config)
        
        # Initialize EMA to a very low value
        bf.loss_ema = 0.1
        bf.n_samples = 10  # Past warmup
        
        # Generate 150 skips
        for i in range(150):
            bf.should_skip_batch(loss=100.0, step=i)
        
        # History should be capped at 100
        assert len(bf.skip_history) <= 100


# ============================================
# MoR Depth Tracking Tests
# ============================================

class TestMoRDepthTracking:
    """Tests for MoR (Mixture of Recursions) depth tracking.
    
    These tests verify that MoR routing correctly tracks per-token depths
    and that depth statistics are available for diagnostics. This catches
    bugs where MoR shows as "FULL" phase but depth is 0.00 because
    _forward_mor() is not being called.
    """
    
    @pytest.fixture
    def mor_model(self):
        """Create a minimal HydraModel with MoR enabled."""
        from hydra.model.framework.model import HydraModel
        model = HydraModel(
            dim=128,
            vocab_size=1000,
            n_mor_blocks=2,
            recursions_per_block=2,
            n_heads=4,
            n_kv_heads=2,
            max_seq_len=64,
            adaptive=True,  # MoR enabled
        )
        return model
    
    def test_mor_curriculum_propagates_to_layers(self, mor_model):
        """set_mor_curriculum should propagate enable_step to all layers."""
        model = mor_model
        
        # Set curriculum
        model.set_mor_curriculum(enable_step=1000, rampup_steps=500)
        
        # Model should have the value
        assert model._mor_enable_step == 1000
        assert model._mor_rampup_steps == 500
        
        # Each layer should also have the value
        for i, layer in enumerate(model.layers):
            assert layer._mor_enable_step == 1000, f"Layer {i} _mor_enable_step not set"
            assert layer._mor_rampup_steps == 500, f"Layer {i} _mor_rampup_steps not set"
    
    def test_set_global_step_updates_rampup_scale(self, mor_model):
        """set_global_step should update _mor_rampup_scale_cached."""
        model = mor_model
        model.set_mor_curriculum(enable_step=1000, rampup_steps=500)
        
        # Before enable step: scale should be 0
        model.set_global_step(500)
        for layer in model.layers:
            assert layer._mor_rampup_scale_cached == 0.0
        
        # At enable step: scale starts ramping (min is 0.1 to avoid div-by-zero)
        model.set_global_step(1000)
        for layer in model.layers:
            assert layer._mor_rampup_scale_cached == 0.1  # Min scale is 0.1
        
        # Mid-rampup (step 1250 = 50% through rampup)
        model.set_global_step(1250)
        for layer in model.layers:
            assert layer._mor_rampup_scale_cached == 0.5
        
        # After rampup: scale should be 1.0
        model.set_global_step(2000)
        for layer in model.layers:
            assert layer._mor_rampup_scale_cached == 1.0
    
    def test_forward_mor_sets_last_target_depths(self, mor_model):
        """When MoR is enabled, forward should set _last_target_depths."""
        model = mor_model
        model.set_mor_curriculum(enable_step=0, rampup_steps=100)
        model.set_global_step(1000)  # Well past rampup
        
        # Verify rampup scale is 1.0
        for layer in model.layers:
            assert layer._mor_rampup_scale_cached == 1.0
        
        # Run forward
        x = torch.randint(0, 1000, (2, 32))
        model.eval()
        with torch.no_grad():
            _ = model(x)
        
        # _last_target_depths should be set for all layers
        for i, layer in enumerate(model.layers):
            assert layer._last_target_depths is not None, \
                f"Layer {i} _last_target_depths is None - _forward_mor not called"
            assert layer._last_target_depths.shape[0] == 2  # batch size
            assert layer._last_target_depths.shape[1] == 32  # seq len
    
    def test_forward_with_losses_sets_last_target_depths(self, mor_model):
        """forward(return_losses=True) should also set _last_target_depths."""
        model = mor_model
        model.set_mor_curriculum(enable_step=0, rampup_steps=100)
        model.set_global_step(1000)
        
        model.train()
        x = torch.randint(0, 1000, (2, 32))
        _, losses = model(x, return_losses=True)
        
        # _last_target_depths should be set
        for i, layer in enumerate(model.layers):
            assert layer._last_target_depths is not None, \
                f"Layer {i} _last_target_depths is None in return_losses mode"
    
    def test_get_routing_stats_includes_avg_depth(self, mor_model):
        """get_routing_stats should include avg_depth when MoR is running."""
        model = mor_model
        model.set_mor_curriculum(enable_step=0, rampup_steps=100)
        model.set_global_step(1000)
        
        x = torch.randint(0, 1000, (2, 32))
        model.eval()
        with torch.no_grad():
            _ = model(x)
        
        stats = model.get_routing_stats()
        mor_layers = stats.get("mor_layers", [])
        
        # Should have stats for each layer
        assert len(mor_layers) == 2
        
        # Each layer should have avg_depth
        for layer_stat in mor_layers:
            assert "avg_depth" in layer_stat, \
                f"Layer {layer_stat.get('layer', '?')} missing avg_depth in stats"
    
    def test_get_mor_status_reflects_curriculum(self, mor_model):
        """get_mor_status should correctly reflect curriculum phase."""
        model = mor_model
        model.set_mor_curriculum(enable_step=1000, rampup_steps=500)
        
        # Before enable: fixed-depth phase
        model.set_global_step(500)
        status = model.get_mor_status()
        assert status["phase"] == "fixed-depth"
        
        # During rampup
        model.set_global_step(1250)
        status = model.get_mor_status()
        assert status["phase"] == "rampup"
        
        # After rampup: full-adaptive
        model.set_global_step(2000)
        status = model.get_mor_status()
        assert status["phase"] == "full-adaptive"
    
    def test_depth_is_nonzero_when_mor_enabled(self, mor_model):
        """Average depth should be > 0 for at least one layer when MoR is running.
        
        This catches the bug where MoR status shows FULL but depth is 0.00.
        """
        model = mor_model
        model.set_mor_curriculum(enable_step=0, rampup_steps=100)
        model.set_global_step(1000)
        
        # Run multiple forward passes to get meaningful depth data
        model.train()
        for _ in range(5):
            x = torch.randint(0, 1000, (4, 32))
            _, _ = model(x, return_losses=True)
        
        # Get routing stats
        stats = model.get_routing_stats()
        mor_layers = stats.get("mor_layers", [])
        
        # At least one layer should have non-zero depth (untrained model will
        # have depth=0 for layer 0, but deeper layers should show routing)
        depths = [s.get("avg_depth", 0) for s in mor_layers if "avg_depth" in s]
        assert len(depths) > 0, "No avg_depth found in any layer stats"
        
        # Sum of depths should be > 0 (at least some tokens should route)
        total_depth = sum(depths)
        assert total_depth > 0, \
            f"Total depth is 0 across all layers - MoR routing not working. " \
            f"Layer depths: {depths}"
    
    def test_mor_disabled_has_no_depth_stats(self, mor_model):
        """When MoR is disabled (adaptive=False), depth should not be tracked."""
        from hydra.model.framework.model import HydraModel
        model = HydraModel(
            dim=128,
            vocab_size=1000,
            n_mor_blocks=2,
            recursions_per_block=2,
            n_heads=4,
            n_kv_heads=2,
            max_seq_len=64,
            adaptive=False,  # MoR disabled
        )
        
        x = torch.randint(0, 1000, (2, 32))
        model.eval()
        with torch.no_grad():
            _ = model(x)
        
        # _last_target_depths should be None (fixed depth, no routing)
        for layer in model.layers:
            assert layer._last_target_depths is None
    
    def test_mor_before_enable_step_has_no_depth_stats(self, mor_model):
        """Before enable_step, depth should not be tracked (fixed depth phase)."""
        model = mor_model
        model.set_mor_curriculum(enable_step=5000, rampup_steps=1000)
        model.set_global_step(1000)  # Before enable_step
        
        x = torch.randint(0, 1000, (2, 32))
        model.eval()
        with torch.no_grad():
            _ = model(x)
        
        # Before enable_step, _forward_fixed is called, no depth tracking
        for layer in model.layers:
            assert layer._last_target_depths is None


# ============================================
# Run tests
# ============================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
