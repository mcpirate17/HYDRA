"""Unit tests for hydra/training/gradients.py - gradient clipping and scaling functions."""

import math
import time
from unittest.mock import MagicMock, patch

import pytest
import torch

from hydra.training.gradients import (
    log_gradient_pathology_diagnostic,
    skip_update_for_nonfinite_gradients,
    reset_optimizer_moments_for_gradient_spike,
    maybe_prepare_halt_on_spike,
)


# ---------------------------------------------------------------------------
# Tests for skip_update_for_nonfinite_gradients
# ---------------------------------------------------------------------------

class TestSkipUpdateForNonfiniteGradients:
    """Tests for skip_update_for_nonfinite_gradients function."""

    def test_returns_false_when_grads_are_finite(self):
        """Should return False and not call zero_grad when gradients are finite."""
        optimizer = MagicMock()
        scaler = MagicMock()

        result = skip_update_for_nonfinite_gradients(
            nonfinite_grads=False,
            optimizer=optimizer,
            use_scaler=True,
            scaler=scaler,
        )

        assert result is False
        optimizer.zero_grad.assert_not_called()
        scaler.update.assert_not_called()

    def test_returns_true_and_zeros_grad_when_nonfinite(self):
        """Should zero gradients and return True when nonfinite detected."""
        optimizer = MagicMock()
        scaler = MagicMock()

        result = skip_update_for_nonfinite_gradients(
            nonfinite_grads=True,
            optimizer=optimizer,
            use_scaler=False,
            scaler=scaler,
        )

        assert result is True
        optimizer.zero_grad.assert_called_once_with(set_to_none=True)
        scaler.update.assert_not_called()

    def test_updates_scaler_when_use_scaler_is_true(self):
        """Should call scaler.update() when use_scaler is True and grads are nonfinite."""
        optimizer = MagicMock()
        scaler = MagicMock()

        result = skip_update_for_nonfinite_gradients(
            nonfinite_grads=True,
            optimizer=optimizer,
            use_scaler=True,
            scaler=scaler,
        )

        assert result is True
        optimizer.zero_grad.assert_called_once_with(set_to_none=True)
        scaler.update.assert_called_once()


# ---------------------------------------------------------------------------
# Tests for reset_optimizer_moments_for_gradient_spike
# ---------------------------------------------------------------------------

class TestResetOptimizerMomentsForGradientSpike:
    """Tests for reset_optimizer_moments_for_gradient_spike function."""

    def test_noop_when_spike_not_detected(self):
        """Should do nothing when spike_detected is False."""
        optimizer = MagicMock()
        base_model = MagicMock()
        logger = MagicMock()

        reset_optimizer_moments_for_gradient_spike(
            spike_detected=False,
            grad_spike_reset_moments=True,
            grad_spike_topk=3,
            grad_info_pre_clip=[("layer.weight", 100.0, 50.0, False, False)],
            base_model=base_model,
            optimizer=optimizer,
            logger=logger,
        )

        logger.warning.assert_not_called()

    def test_noop_when_reset_moments_disabled(self):
        """Should do nothing when grad_spike_reset_moments is False."""
        optimizer = MagicMock()
        base_model = MagicMock()
        logger = MagicMock()

        reset_optimizer_moments_for_gradient_spike(
            spike_detected=True,
            grad_spike_reset_moments=False,
            grad_spike_topk=3,
            grad_info_pre_clip=[("layer.weight", 100.0, 50.0, False, False)],
            base_model=base_model,
            optimizer=optimizer,
            logger=logger,
        )

        logger.warning.assert_not_called()

    def test_resets_moments_for_top_offenders(self):
        """Should reset Adam moments for top-k gradient offenders."""
        # Create mock parameters with gradients
        param1 = torch.nn.Parameter(torch.randn(4, 4))
        param1.grad = torch.randn(4, 4)
        param2 = torch.nn.Parameter(torch.randn(4, 4))
        param2.grad = torch.randn(4, 4)
        param3 = torch.nn.Parameter(torch.randn(4, 4))
        param3.grad = torch.randn(4, 4)

        # Create optimizer state with moments
        exp_avg1 = torch.randn(4, 4)
        exp_avg_sq1 = torch.randn(4, 4).abs()
        exp_avg2 = torch.randn(4, 4)
        exp_avg_sq2 = torch.randn(4, 4).abs()

        optimizer = MagicMock()
        optimizer.state = {
            param1: {"exp_avg": exp_avg1, "exp_avg_sq": exp_avg_sq1},
            param2: {"exp_avg": exp_avg2, "exp_avg_sq": exp_avg_sq2},
            param3: {},  # No state for param3
        }

        base_model = MagicMock()
        base_model.named_parameters.return_value = [
            ("layer1.weight", param1),
            ("layer2.weight", param2),
            ("layer3.weight", param3),
        ]

        logger = MagicMock()

        # param1 (layer1.weight) has highest grad norm, param2 second
        grad_info_pre_clip = [
            ("layer1.weight", 100.0, 50.0, False, False),  # highest
            ("layer2.weight", 50.0, 25.0, False, False),   # second
            ("layer3.weight", 10.0, 5.0, False, False),    # lowest
        ]

        reset_optimizer_moments_for_gradient_spike(
            spike_detected=True,
            grad_spike_reset_moments=True,
            grad_spike_topk=2,
            grad_info_pre_clip=grad_info_pre_clip,
            base_model=base_model,
            optimizer=optimizer,
            logger=logger,
        )

        # Check that moments for top 2 offenders are zeroed
        assert torch.all(exp_avg1 == 0)
        assert torch.all(exp_avg_sq1 == 0)
        assert torch.all(exp_avg2 == 0)
        assert torch.all(exp_avg_sq2 == 0)

        logger.warning.assert_called_once()
        assert "reset Adam moments" in str(logger.warning.call_args)

    def test_prioritizes_nan_inf_params(self):
        """Should prioritize params with NaN/Inf over high norms."""
        param1 = torch.nn.Parameter(torch.randn(2, 2))
        param1.grad = torch.randn(2, 2)
        param2 = torch.nn.Parameter(torch.randn(2, 2))
        param2.grad = torch.randn(2, 2)

        exp_avg1 = torch.randn(2, 2)
        exp_avg2 = torch.randn(2, 2)

        optimizer = MagicMock()
        optimizer.state = {
            param1: {"exp_avg": exp_avg1},
            param2: {"exp_avg": exp_avg2},
        }

        base_model = MagicMock()
        base_model.named_parameters.return_value = [
            ("layer1.weight", param1),
            ("layer2.weight", param2),
        ]

        # param2 has NaN (should be prioritized), param1 has high norm
        grad_info_pre_clip = [
            ("layer1.weight", 1000.0, 500.0, False, False),  # high norm
            ("layer2.weight", 1.0, 0.5, True, False),        # has NaN
        ]

        reset_optimizer_moments_for_gradient_spike(
            spike_detected=True,
            grad_spike_reset_moments=True,
            grad_spike_topk=1,
            grad_info_pre_clip=grad_info_pre_clip,
            base_model=base_model,
            optimizer=optimizer,
            logger=None,  # Silent mode
        )

        # NaN param should be reset first (layer2), not high-norm param (layer1)
        assert torch.all(exp_avg2 == 0)
        # layer1 should NOT be reset since topk=1 and layer2 was prioritized
        assert not torch.all(exp_avg1 == 0)

    def test_handles_max_exp_avg_sq(self):
        """Should also reset max_exp_avg_sq if present (for AMSGrad)."""
        param = torch.nn.Parameter(torch.randn(2, 2))
        param.grad = torch.randn(2, 2)

        exp_avg = torch.randn(2, 2)
        exp_avg_sq = torch.randn(2, 2).abs()
        max_exp_avg_sq = torch.randn(2, 2).abs()

        optimizer = MagicMock()
        optimizer.state = {
            param: {
                "exp_avg": exp_avg,
                "exp_avg_sq": exp_avg_sq,
                "max_exp_avg_sq": max_exp_avg_sq,
            }
        }

        base_model = MagicMock()
        base_model.named_parameters.return_value = [("layer.weight", param)]

        grad_info_pre_clip = [("layer.weight", 100.0, 50.0, False, False)]

        reset_optimizer_moments_for_gradient_spike(
            spike_detected=True,
            grad_spike_reset_moments=True,
            grad_spike_topk=1,
            grad_info_pre_clip=grad_info_pre_clip,
            base_model=base_model,
            optimizer=optimizer,
            logger=None,
        )

        assert torch.all(exp_avg == 0)
        assert torch.all(exp_avg_sq == 0)
        assert torch.all(max_exp_avg_sq == 0)

    def test_silent_mode_without_logger(self):
        """Should operate silently when logger is None."""
        param = torch.nn.Parameter(torch.randn(2, 2))
        param.grad = torch.randn(2, 2)

        optimizer = MagicMock()
        optimizer.state = {param: {"exp_avg": torch.randn(2, 2)}}

        base_model = MagicMock()
        base_model.named_parameters.return_value = [("layer.weight", param)]

        grad_info_pre_clip = [("layer.weight", 100.0, 50.0, False, False)]

        # Should not raise
        reset_optimizer_moments_for_gradient_spike(
            spike_detected=True,
            grad_spike_reset_moments=True,
            grad_spike_topk=1,
            grad_info_pre_clip=grad_info_pre_clip,
            base_model=base_model,
            optimizer=optimizer,
            logger=None,
        )

    def test_handles_exception_gracefully(self):
        """Should handle exceptions and log warning."""
        base_model = MagicMock()
        base_model.named_parameters.side_effect = RuntimeError("Test error")

        logger = MagicMock()

        reset_optimizer_moments_for_gradient_spike(
            spike_detected=True,
            grad_spike_reset_moments=True,
            grad_spike_topk=1,
            grad_info_pre_clip=[("layer.weight", 100.0, 50.0, False, False)],
            base_model=base_model,
            optimizer=MagicMock(),
            logger=logger,
        )

        logger.warning.assert_called_once()
        assert "failed" in str(logger.warning.call_args)

    def test_skips_params_without_grad(self):
        """Should skip parameters that don't have gradients."""
        param1 = torch.nn.Parameter(torch.randn(2, 2))
        param1.grad = None  # No gradient
        param2 = torch.nn.Parameter(torch.randn(2, 2))
        param2.grad = torch.randn(2, 2)

        exp_avg1 = torch.randn(2, 2)
        exp_avg2 = torch.randn(2, 2)

        optimizer = MagicMock()
        optimizer.state = {
            param1: {"exp_avg": exp_avg1},
            param2: {"exp_avg": exp_avg2},
        }

        base_model = MagicMock()
        base_model.named_parameters.return_value = [
            ("layer1.weight", param1),
            ("layer2.weight", param2),
        ]

        grad_info_pre_clip = [
            ("layer1.weight", 100.0, 50.0, False, False),
            ("layer2.weight", 50.0, 25.0, False, False),
        ]

        reset_optimizer_moments_for_gradient_spike(
            spike_detected=True,
            grad_spike_reset_moments=True,
            grad_spike_topk=2,
            grad_info_pre_clip=grad_info_pre_clip,
            base_model=base_model,
            optimizer=optimizer,
            logger=None,
        )

        # param1 should NOT be reset (no grad)
        assert not torch.all(exp_avg1 == 0)
        # param2 should be reset
        assert torch.all(exp_avg2 == 0)


# ---------------------------------------------------------------------------
# Tests for maybe_prepare_halt_on_spike
# ---------------------------------------------------------------------------

class TestMaybePrepareHaltOnSpike:
    """Tests for maybe_prepare_halt_on_spike function."""

    def test_returns_false_when_no_spike(self):
        """Should return False when spike_detected is False."""
        result = maybe_prepare_halt_on_spike(
            spike_detected=False,
            halt_on_spike=True,
            step=100,
            accum_loss=1.0,
            lr_effective=1e-4,
            grad_norm=1.0,
            tokens_per_step=1024,
            step_start=time.time(),
            metrics=MagicMock(),
            save_checkpoint=MagicMock(),
            logger=MagicMock(),
        )
        assert result is False

    def test_returns_false_when_halt_disabled(self):
        """Should return False when halt_on_spike is False."""
        result = maybe_prepare_halt_on_spike(
            spike_detected=True,
            halt_on_spike=False,
            step=100,
            accum_loss=1.0,
            lr_effective=1e-4,
            grad_norm=1.0,
            tokens_per_step=1024,
            step_start=time.time(),
            metrics=MagicMock(),
            save_checkpoint=MagicMock(),
            logger=MagicMock(),
        )
        assert result is False

    def test_updates_metrics_and_saves_checkpoint_on_halt(self):
        """Should update metrics, save checkpoint, and return True on halt."""
        metrics = MagicMock()
        metrics.total_tokens = 0
        save_checkpoint = MagicMock()
        logger = MagicMock()

        step_start = time.time() - 0.1  # 100ms ago

        result = maybe_prepare_halt_on_spike(
            spike_detected=True,
            halt_on_spike=True,
            step=100,
            accum_loss=2.5,
            lr_effective=1e-4,
            grad_norm=50.0,
            tokens_per_step=1024,
            step_start=step_start,
            metrics=metrics,
            save_checkpoint=save_checkpoint,
            logger=logger,
        )

        assert result is True
        metrics.update.assert_called_once()
        call_args = metrics.update.call_args[0]
        assert call_args[0] == 100  # step
        assert call_args[1] == 2.5  # accum_loss
        assert call_args[2] == 1e-4  # lr_effective
        assert call_args[3] == 50.0  # grad_norm

        assert metrics.total_tokens == 1024
        assert metrics.final_loss == 2.5
        save_checkpoint.assert_called_once_with(100)

    def test_handles_exception_gracefully(self):
        """Should handle exceptions and log warning."""
        metrics = MagicMock()
        metrics.update.side_effect = RuntimeError("Test error")
        logger = MagicMock()

        result = maybe_prepare_halt_on_spike(
            spike_detected=True,
            halt_on_spike=True,
            step=100,
            accum_loss=1.0,
            lr_effective=1e-4,
            grad_norm=1.0,
            tokens_per_step=1024,
            step_start=time.time(),
            metrics=metrics,
            save_checkpoint=MagicMock(),
            logger=logger,
        )

        assert result is True
        logger.warning.assert_called_once()
        assert "failed" in str(logger.warning.call_args)

    def test_calculates_tokens_per_second_correctly(self):
        """Should calculate tokens per second based on step time."""
        metrics = MagicMock()
        metrics.total_tokens = 0

        # Simulate 1 second step time
        step_start = time.time() - 1.0

        result = maybe_prepare_halt_on_spike(
            spike_detected=True,
            halt_on_spike=True,
            step=100,
            accum_loss=1.0,
            lr_effective=1e-4,
            grad_norm=1.0,
            tokens_per_step=2000,
            step_start=step_start,
            metrics=metrics,
            save_checkpoint=MagicMock(),
            logger=MagicMock(),
        )

        assert result is True
        call_args = metrics.update.call_args[0]
        # tps should be approximately 2000 tokens / 1 second = 2000
        tps = call_args[4]
        assert 1500 < tps < 2500  # Allow some tolerance for timing


# ---------------------------------------------------------------------------
# Tests for log_gradient_pathology_diagnostic
# ---------------------------------------------------------------------------

class TestLogGradientPathologyDiagnostic:
    """Tests for log_gradient_pathology_diagnostic function."""

    def _make_mock_model(self, with_layers=True, num_layers=2):
        """Create a mock base model for testing."""
        model = MagicMock()
        if with_layers:
            layers = []
            for i in range(num_layers):
                layer = MagicMock()
                # Set up default attributes
                layer._last_attn_out_rms_t = None
                layer._last_attn_out_dtype = None
                layer._last_attn_out_shape = None
                layer._last_attn_out_finite_frac_t = None
                layer._last_attn_out_nan_ct_t = None
                layer._last_attn_out_inf_ct_t = None
                layers.append(layer)
            model.layers = layers
        else:
            model.layers = None
        return model

    def test_logs_warning_with_spike_info(self):
        """Should log a warning with gradient spike information."""
        logger = MagicMock()
        base_model = self._make_mock_model()

        log_gradient_pathology_diagnostic(
            logger=logger,
            step=100,
            pre_clip_norm=500.0,
            grad_norm=10.0,
            grad_clip=1.0,
            clip_coef=0.02,
            clip_scale=1.0,
            scaler_scale=1024.0,
            lr=1e-4,
            lr_effective=5e-5,
            spike_detected=True,
            grad_spike_lr_factor=0.5,
            accum_loss=2.5,
            last_ce_loss=2.3,
            last_aux_loss=0.1,
            last_ponder_loss=0.1,
            micro_diag=[],
            grad_info_pre_clip=[
                ("layers.0.attention.o_proj.weight", 100.0, 50.0, False, False),
                ("layers.1.attention.o_proj.weight", 80.0, 40.0, False, False),
            ],
            base_model=base_model,
        )

        logger.warning.assert_called_once()
        call_str = str(logger.warning.call_args)
        assert "Step 100" in call_str
        assert "500" in call_str or "5.00e+02" in call_str
        assert "loss=2.5" in call_str

    def test_extracts_layer_o_proj_grad_norms(self):
        """Should extract layer 0 and 1 o_proj gradient norms."""
        logger = MagicMock()
        base_model = self._make_mock_model()

        log_gradient_pathology_diagnostic(
            logger=logger,
            step=100,
            pre_clip_norm=100.0,
            grad_norm=10.0,
            grad_clip=1.0,
            clip_coef=0.1,
            clip_scale=1.0,
            scaler_scale=None,
            lr=1e-4,
            lr_effective=1e-4,
            spike_detected=True,
            grad_spike_lr_factor=1.0,
            accum_loss=1.0,
            last_ce_loss=1.0,
            last_aux_loss=0.0,
            last_ponder_loss=0.0,
            micro_diag=[],
            grad_info_pre_clip=[
                ("layers.0.attention.o_proj.weight", 123.45, 50.0, False, False),
                ("layers.1.attention.o_proj.weight", 67.89, 30.0, False, False),
                ("other.layer", 10.0, 5.0, False, False),
            ],
            base_model=base_model,
        )

        call_str = str(logger.warning.call_args)
        assert "1.23e+02" in call_str or "123" in call_str  # layer 0 norm
        assert "6.79e+01" in call_str or "67" in call_str   # layer 1 norm

    def test_handles_none_scaler_scale(self):
        """Should handle None scaler_scale gracefully."""
        logger = MagicMock()
        base_model = self._make_mock_model()

        log_gradient_pathology_diagnostic(
            logger=logger,
            step=100,
            pre_clip_norm=100.0,
            grad_norm=10.0,
            grad_clip=1.0,
            clip_coef=0.1,
            clip_scale=1.0,
            scaler_scale=None,
            lr=1e-4,
            lr_effective=1e-4,
            spike_detected=True,
            grad_spike_lr_factor=1.0,
            accum_loss=1.0,
            last_ce_loss=1.0,
            last_aux_loss=0.0,
            last_ponder_loss=0.0,
            micro_diag=[],
            grad_info_pre_clip=[],
            base_model=base_model,
        )

        call_str = str(logger.warning.call_args)
        assert "amp_scale=n/a" in call_str

    def test_handles_nonfinite_scaler_scale(self):
        """Should handle non-finite scaler_scale."""
        logger = MagicMock()
        base_model = self._make_mock_model()

        log_gradient_pathology_diagnostic(
            logger=logger,
            step=100,
            pre_clip_norm=100.0,
            grad_norm=10.0,
            grad_clip=1.0,
            clip_coef=0.1,
            clip_scale=1.0,
            scaler_scale=float('inf'),
            lr=1e-4,
            lr_effective=1e-4,
            spike_detected=True,
            grad_spike_lr_factor=1.0,
            accum_loss=1.0,
            last_ce_loss=1.0,
            last_aux_loss=0.0,
            last_ponder_loss=0.0,
            micro_diag=[],
            grad_info_pre_clip=[],
            base_model=base_model,
        )

        call_str = str(logger.warning.call_args)
        assert "amp_scale=n/a" in call_str

    def test_shows_top_grad_offenders(self):
        """Should show top 3 gradient offenders."""
        logger = MagicMock()
        base_model = self._make_mock_model()

        log_gradient_pathology_diagnostic(
            logger=logger,
            step=100,
            pre_clip_norm=100.0,
            grad_norm=10.0,
            grad_clip=1.0,
            clip_coef=0.1,
            clip_scale=1.0,
            scaler_scale=1024.0,
            lr=1e-4,
            lr_effective=1e-4,
            spike_detected=True,
            grad_spike_lr_factor=1.0,
            accum_loss=1.0,
            last_ce_loss=1.0,
            last_aux_loss=0.0,
            last_ponder_loss=0.0,
            micro_diag=[],
            grad_info_pre_clip=[
                ("layer1.weight", 1000.0, 500.0, False, False),
                ("layer2.weight", 500.0, 250.0, False, False),
                ("layer3.weight", 100.0, 50.0, False, False),
                ("layer4.weight", 10.0, 5.0, False, False),
            ],
            base_model=base_model,
        )

        call_str = str(logger.warning.call_args)
        # Should include top 3 offenders
        assert "layer1.weight" in call_str
        assert "layer2.weight" in call_str
        assert "layer3.weight" in call_str
        # layer4 is not in top 3
        assert "layer4.weight" not in call_str

    def test_marks_nan_inf_params_with_warning(self):
        """Should mark params with NaN/Inf with (!). marking."""
        logger = MagicMock()
        base_model = self._make_mock_model()

        log_gradient_pathology_diagnostic(
            logger=logger,
            step=100,
            pre_clip_norm=100.0,
            grad_norm=10.0,
            grad_clip=1.0,
            clip_coef=0.1,
            clip_scale=1.0,
            scaler_scale=1024.0,
            lr=1e-4,
            lr_effective=1e-4,
            spike_detected=True,
            grad_spike_lr_factor=1.0,
            accum_loss=1.0,
            last_ce_loss=1.0,
            last_aux_loss=0.0,
            last_ponder_loss=0.0,
            micro_diag=[],
            grad_info_pre_clip=[
                ("nan_layer.weight", 10.0, 5.0, True, False),  # has NaN
                ("inf_layer.weight", 10.0, 5.0, False, True),  # has Inf
                ("ok_layer.weight", 10.0, 5.0, False, False),  # normal
            ],
            base_model=base_model,
        )

        call_str = str(logger.warning.call_args)
        # Should have (!) markers for NaN/Inf params
        assert "(!)" in call_str

    def test_extracts_attention_output_stats(self):
        """Should extract attention output stats from model layers."""
        logger = MagicMock()
        base_model = self._make_mock_model()

        # Set up layer 0 with attention stats
        base_model.layers[0]._last_attn_out_rms_t = torch.tensor([0.5])
        base_model.layers[0]._last_attn_out_dtype = "bfloat16"
        base_model.layers[0]._last_attn_out_shape = (2, 128, 768)
        base_model.layers[0]._last_attn_out_finite_frac_t = torch.tensor([0.99])
        base_model.layers[0]._last_attn_out_nan_ct_t = torch.tensor([5])
        base_model.layers[0]._last_attn_out_inf_ct_t = torch.tensor([3])

        log_gradient_pathology_diagnostic(
            logger=logger,
            step=100,
            pre_clip_norm=100.0,
            grad_norm=10.0,
            grad_clip=1.0,
            clip_coef=0.1,
            clip_scale=1.0,
            scaler_scale=1024.0,
            lr=1e-4,
            lr_effective=1e-4,
            spike_detected=True,
            grad_spike_lr_factor=1.0,
            accum_loss=1.0,
            last_ce_loss=1.0,
            last_aux_loss=0.0,
            last_ponder_loss=0.0,
            micro_diag=[],
            grad_info_pre_clip=[],
            base_model=base_model,
        )

        call_str = str(logger.warning.call_args)
        assert "L0_attn" in call_str
        # Should contain RMS value
        assert "rms=" in call_str
        # Should contain finite fraction
        assert "ff=" in call_str

    def test_handles_model_without_layers(self):
        """Should handle model without layers attribute."""
        logger = MagicMock()
        base_model = self._make_mock_model(with_layers=False)

        # Should not raise
        log_gradient_pathology_diagnostic(
            logger=logger,
            step=100,
            pre_clip_norm=100.0,
            grad_norm=10.0,
            grad_clip=1.0,
            clip_coef=0.1,
            clip_scale=1.0,
            scaler_scale=1024.0,
            lr=1e-4,
            lr_effective=1e-4,
            spike_detected=True,
            grad_spike_lr_factor=1.0,
            accum_loss=1.0,
            last_ce_loss=1.0,
            last_aux_loss=0.0,
            last_ponder_loss=0.0,
            micro_diag=[],
            grad_info_pre_clip=[],
            base_model=base_model,
        )

        logger.warning.assert_called_once()

    def test_handles_empty_grad_info(self):
        """Should handle empty grad_info_pre_clip list."""
        logger = MagicMock()
        base_model = self._make_mock_model()

        log_gradient_pathology_diagnostic(
            logger=logger,
            step=100,
            pre_clip_norm=100.0,
            grad_norm=10.0,
            grad_clip=1.0,
            clip_coef=0.1,
            clip_scale=1.0,
            scaler_scale=1024.0,
            lr=1e-4,
            lr_effective=1e-4,
            spike_detected=True,
            grad_spike_lr_factor=1.0,
            accum_loss=1.0,
            last_ce_loss=1.0,
            last_aux_loss=0.0,
            last_ponder_loss=0.0,
            micro_diag=[],
            grad_info_pre_clip=[],
            base_model=base_model,
        )

        logger.warning.assert_called_once()
        call_str = str(logger.warning.call_args)
        assert "top_grad=[]" in call_str

    def test_verbose_mode(self):
        """Should work with verbose=True (currently no-op but shouldn't fail)."""
        logger = MagicMock()
        base_model = self._make_mock_model()

        log_gradient_pathology_diagnostic(
            logger=logger,
            step=100,
            pre_clip_norm=100.0,
            grad_norm=10.0,
            grad_clip=1.0,
            clip_coef=0.1,
            clip_scale=1.0,
            scaler_scale=1024.0,
            lr=1e-4,
            lr_effective=1e-4,
            spike_detected=True,
            grad_spike_lr_factor=1.0,
            accum_loss=1.0,
            last_ce_loss=1.0,
            last_aux_loss=0.0,
            last_ponder_loss=0.0,
            micro_diag=[],
            grad_info_pre_clip=[],
            base_model=base_model,
            verbose=True,
        )

        logger.warning.assert_called_once()
