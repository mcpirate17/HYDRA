"""Unit tests for hydra/training/gradients.py."""

import math
import time
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

from hydra.training.gradients import (
    log_gradient_pathology_diagnostic,
    maybe_prepare_halt_on_spike,
    reset_optimizer_moments_for_gradient_spike,
    skip_update_for_nonfinite_gradients,
)


class TestSkipUpdateForNonfiniteGradients:
    """Tests for skip_update_for_nonfinite_gradients function."""

    def test_returns_false_when_grads_finite(self):
        """Should return False and not modify anything when gradients are finite."""
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

    def test_zeros_grads_and_updates_scaler_when_nonfinite(self):
        """Should zero grads and update scaler when nonfinite gradients detected."""
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

    def test_zeros_grads_without_scaler_update(self):
        """Should zero grads but not update scaler when use_scaler=False."""
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


class TestResetOptimizerMomentsForGradientSpike:
    """Tests for reset_optimizer_moments_for_gradient_spike function."""

    def test_does_nothing_when_no_spike(self):
        """Should return immediately when spike_detected=False."""
        optimizer = MagicMock()
        logger = MagicMock()

        reset_optimizer_moments_for_gradient_spike(
            spike_detected=False,
            grad_spike_reset_moments=True,
            grad_spike_topk=3,
            grad_info_pre_clip=[("param1", 10.0, 5.0, False, False)],
            base_model=MagicMock(),
            optimizer=optimizer,
            logger=logger,
        )

        logger.warning.assert_not_called()

    def test_does_nothing_when_reset_moments_disabled(self):
        """Should return immediately when grad_spike_reset_moments=False."""
        optimizer = MagicMock()
        logger = MagicMock()

        reset_optimizer_moments_for_gradient_spike(
            spike_detected=True,
            grad_spike_reset_moments=False,
            grad_spike_topk=3,
            grad_info_pre_clip=[("param1", 10.0, 5.0, False, False)],
            base_model=MagicMock(),
            optimizer=optimizer,
            logger=logger,
        )

        logger.warning.assert_not_called()

    def test_resets_moments_for_top_offenders(self):
        """Should reset exp_avg and exp_avg_sq for top-k gradient offenders."""
        # Create mock model with named parameters
        param1 = torch.nn.Parameter(torch.randn(10))
        param1.grad = torch.randn(10)
        param2 = torch.nn.Parameter(torch.randn(10))
        param2.grad = torch.randn(10)
        param3 = torch.nn.Parameter(torch.randn(10))
        param3.grad = torch.randn(10)

        base_model = MagicMock()
        base_model.named_parameters.return_value = [
            ("layer1.weight", param1),
            ("layer2.weight", param2),
            ("layer3.weight", param3),
        ]

        # Create optimizer state with moments
        exp_avg1 = torch.randn(10)
        exp_avg_sq1 = torch.randn(10).abs()
        exp_avg2 = torch.randn(10)
        exp_avg_sq2 = torch.randn(10).abs()

        optimizer = MagicMock()
        optimizer.state = {
            param1: {"exp_avg": exp_avg1, "exp_avg_sq": exp_avg_sq1},
            param2: {"exp_avg": exp_avg2, "exp_avg_sq": exp_avg_sq2},
            param3: {},  # No state for param3
        }

        logger = MagicMock()

        # grad_info_pre_clip: (name, grad_norm, grad_max, has_nan, has_inf)
        # layer1 has highest norm (100.0), layer2 has medium (50.0)
        grad_info_pre_clip = [
            ("layer1.weight", 100.0, 20.0, False, False),
            ("layer2.weight", 50.0, 10.0, False, False),
            ("layer3.weight", 10.0, 5.0, False, False),
        ]

        reset_optimizer_moments_for_gradient_spike(
            spike_detected=True,
            grad_spike_reset_moments=True,
            grad_spike_topk=2,  # Only top 2 offenders
            grad_info_pre_clip=grad_info_pre_clip,
            base_model=base_model,
            optimizer=optimizer,
            logger=logger,
        )

        # Check that moments for layer1 and layer2 are zeroed
        assert torch.all(exp_avg1 == 0)
        assert torch.all(exp_avg_sq1 == 0)
        assert torch.all(exp_avg2 == 0)
        assert torch.all(exp_avg_sq2 == 0)

        logger.warning.assert_called_once()
        assert "reset Adam moments for top 2 params" in str(logger.warning.call_args)

    def test_prioritizes_nan_inf_gradients(self):
        """Should treat NaN/Inf gradients as highest priority offenders."""
        param1 = torch.nn.Parameter(torch.randn(10))
        param1.grad = torch.randn(10)
        param2 = torch.nn.Parameter(torch.randn(10))
        param2.grad = torch.randn(10)

        base_model = MagicMock()
        base_model.named_parameters.return_value = [
            ("layer1.weight", param1),
            ("layer2.weight", param2),
        ]

        exp_avg1 = torch.randn(10)
        exp_avg_sq1 = torch.randn(10).abs()

        optimizer = MagicMock()
        optimizer.state = {
            param1: {"exp_avg": exp_avg1, "exp_avg_sq": exp_avg_sq1},
            param2: {},
        }

        logger = MagicMock()

        # layer1 has NaN grad (has_nan=True), lower norm but should be top offender
        grad_info_pre_clip = [
            ("layer1.weight", 10.0, 5.0, True, False),  # NaN
            ("layer2.weight", 100.0, 20.0, False, False),  # Higher norm but finite
        ]

        reset_optimizer_moments_for_gradient_spike(
            spike_detected=True,
            grad_spike_reset_moments=True,
            grad_spike_topk=1,  # Only top 1
            grad_info_pre_clip=grad_info_pre_clip,
            base_model=base_model,
            optimizer=optimizer,
            logger=logger,
        )

        # layer1 should be reset (NaN takes priority)
        assert torch.all(exp_avg1 == 0)
        assert torch.all(exp_avg_sq1 == 0)

    def test_resets_max_exp_avg_sq_if_present(self):
        """Should also reset max_exp_avg_sq (AMSGrad) if present."""
        param = torch.nn.Parameter(torch.randn(10))
        param.grad = torch.randn(10)

        base_model = MagicMock()
        base_model.named_parameters.return_value = [("layer.weight", param)]

        max_exp_avg_sq = torch.randn(10).abs() + 1.0  # All positive

        optimizer = MagicMock()
        optimizer.state = {
            param: {
                "exp_avg": torch.randn(10),
                "exp_avg_sq": torch.randn(10).abs(),
                "max_exp_avg_sq": max_exp_avg_sq,
            }
        }

        grad_info_pre_clip = [("layer.weight", 100.0, 20.0, False, False)]

        reset_optimizer_moments_for_gradient_spike(
            spike_detected=True,
            grad_spike_reset_moments=True,
            grad_spike_topk=1,
            grad_info_pre_clip=grad_info_pre_clip,
            base_model=base_model,
            optimizer=optimizer,
            logger=None,  # Silent mode
        )

        assert torch.all(max_exp_avg_sq == 0)

    def test_handles_exception_gracefully(self):
        """Should catch exceptions and log warning."""
        base_model = MagicMock()
        base_model.named_parameters.side_effect = RuntimeError("Test error")

        logger = MagicMock()

        reset_optimizer_moments_for_gradient_spike(
            spike_detected=True,
            grad_spike_reset_moments=True,
            grad_spike_topk=3,
            grad_info_pre_clip=[],
            base_model=base_model,
            optimizer=MagicMock(),
            logger=logger,
        )

        logger.warning.assert_called_once()
        assert "moment reset failed" in str(logger.warning.call_args)

    def test_skips_params_without_grad(self):
        """Should skip parameters that have no gradient."""
        param_with_grad = torch.nn.Parameter(torch.randn(10))
        param_with_grad.grad = torch.randn(10)
        param_without_grad = torch.nn.Parameter(torch.randn(10))
        param_without_grad.grad = None

        base_model = MagicMock()
        base_model.named_parameters.return_value = [
            ("layer1.weight", param_with_grad),
            ("layer2.weight", param_without_grad),
        ]

        exp_avg1 = torch.randn(10)
        exp_avg2 = torch.randn(10)  # Should not be zeroed

        optimizer = MagicMock()
        optimizer.state = {
            param_with_grad: {"exp_avg": exp_avg1, "exp_avg_sq": torch.randn(10).abs()},
            param_without_grad: {"exp_avg": exp_avg2, "exp_avg_sq": torch.randn(10).abs()},
        }

        grad_info_pre_clip = [
            ("layer1.weight", 100.0, 20.0, False, False),
            ("layer2.weight", 50.0, 10.0, False, False),
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

        # param_with_grad moments should be zeroed
        assert torch.all(exp_avg1 == 0)
        # param_without_grad moments should NOT be zeroed (grad is None)
        assert not torch.all(exp_avg2 == 0)


class TestMaybePrepareHaltOnSpike:
    """Tests for maybe_prepare_halt_on_spike function."""

    def test_returns_false_when_no_spike(self):
        """Should return False when spike_detected=False."""
        result = maybe_prepare_halt_on_spike(
            spike_detected=False,
            halt_on_spike=True,
            step=100,
            accum_loss=1.0,
            lr_effective=0.001,
            grad_norm=10.0,
            tokens_per_step=1024,
            step_start=time.time() - 1.0,
            metrics=MagicMock(),
            save_checkpoint=MagicMock(),
            logger=MagicMock(),
        )

        assert result is False

    def test_returns_false_when_halt_disabled(self):
        """Should return False when halt_on_spike=False."""
        result = maybe_prepare_halt_on_spike(
            spike_detected=True,
            halt_on_spike=False,
            step=100,
            accum_loss=1.0,
            lr_effective=0.001,
            grad_norm=10.0,
            tokens_per_step=1024,
            step_start=time.time() - 1.0,
            metrics=MagicMock(),
            save_checkpoint=MagicMock(),
            logger=MagicMock(),
        )

        assert result is False

    def test_updates_metrics_and_saves_checkpoint(self):
        """Should update metrics and save checkpoint when spike + halt enabled."""
        metrics = MagicMock()
        metrics.total_tokens = 0
        save_checkpoint = MagicMock()
        step_start = time.time() - 1.0  # 1 second ago

        result = maybe_prepare_halt_on_spike(
            spike_detected=True,
            halt_on_spike=True,
            step=100,
            accum_loss=2.5,
            lr_effective=0.001,
            grad_norm=50.0,
            tokens_per_step=2048,
            step_start=step_start,
            metrics=metrics,
            save_checkpoint=save_checkpoint,
            logger=MagicMock(),
        )

        assert result is True
        metrics.update.assert_called_once()
        call_args = metrics.update.call_args[0]
        assert call_args[0] == 100  # step
        assert call_args[1] == 2.5  # accum_loss
        assert call_args[2] == 0.001  # lr_effective
        assert call_args[3] == 50.0  # grad_norm

        assert metrics.total_tokens == 2048
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
            accum_loss=2.5,
            lr_effective=0.001,
            grad_norm=50.0,
            tokens_per_step=2048,
            step_start=time.time() - 1.0,
            metrics=metrics,
            save_checkpoint=MagicMock(),
            logger=logger,
        )

        assert result is True  # Still returns True even on exception
        logger.warning.assert_called_once()
        assert "failed to record/save state" in str(logger.warning.call_args)

    def test_calculates_correct_tokens_per_second(self):
        """Should calculate TPS correctly based on step time."""
        metrics = MagicMock()
        metrics.total_tokens = 0
        step_start = time.time() - 2.0  # 2 seconds ago

        maybe_prepare_halt_on_spike(
            spike_detected=True,
            halt_on_spike=True,
            step=100,
            accum_loss=1.0,
            lr_effective=0.001,
            grad_norm=10.0,
            tokens_per_step=4000,  # 4000 tokens in 2 seconds = 2000 TPS
            step_start=step_start,
            metrics=metrics,
            save_checkpoint=MagicMock(),
            logger=MagicMock(),
        )

        # Check TPS calculation (4th positional arg to metrics.update)
        call_args = metrics.update.call_args[0]
        tps = call_args[4]
        # Allow some tolerance since time.time() is involved
        assert 1500 < tps < 2500


class TestLogGradientPathologyDiagnostic:
    """Tests for log_gradient_pathology_diagnostic function."""

    def test_logs_spike_warning(self):
        """Should log a warning with gradient spike info."""
        logger = MagicMock()
        base_model = MagicMock()
        base_model.layers = []

        log_gradient_pathology_diagnostic(
            logger=logger,
            step=100,
            pre_clip_norm=100.0,
            grad_norm=10.0,
            grad_clip=1.0,
            clip_coef=0.1,
            clip_scale=0.5,
            scaler_scale=65536.0,
            lr=0.001,
            lr_effective=0.0005,
            spike_detected=True,
            grad_spike_lr_factor=0.5,
            accum_loss=2.5,
            last_ce_loss=2.3,
            last_aux_loss=0.1,
            last_ponder_loss=0.1,
            micro_diag=[],
            grad_info_pre_clip=[
                ("layer.weight", 100.0, 20.0, False, False),
            ],
            base_model=base_model,
        )

        logger.warning.assert_called_once()
        warning_msg = str(logger.warning.call_args)
        assert "Step 100" in warning_msg
        assert "grad=" in warning_msg
        assert "amp_scale=" in warning_msg

    def test_extracts_layer0_layer1_o_proj_grad_norms(self):
        """Should extract o_proj grad norms for layers 0 and 1."""
        logger = MagicMock()
        base_model = MagicMock()
        base_model.layers = []

        grad_info_pre_clip = [
            ("layers.0.attention.o_proj.weight", 50.0, 10.0, False, False),
            ("layers.1.attention.o_proj.weight", 30.0, 8.0, False, False),
            ("layers.2.attention.o_proj.weight", 20.0, 5.0, False, False),
        ]

        log_gradient_pathology_diagnostic(
            logger=logger,
            step=100,
            pre_clip_norm=100.0,
            grad_norm=10.0,
            grad_clip=1.0,
            clip_coef=0.1,
            clip_scale=0.5,
            scaler_scale=None,
            lr=0.001,
            lr_effective=0.001,
            spike_detected=True,
            grad_spike_lr_factor=1.0,
            accum_loss=2.5,
            last_ce_loss=2.5,
            last_aux_loss=0.0,
            last_ponder_loss=0.0,
            micro_diag=[],
            grad_info_pre_clip=grad_info_pre_clip,
            base_model=base_model,
        )

        warning_msg = str(logger.warning.call_args)
        # L0 and L1 o_proj norms should appear in output
        assert "L0/L1_o_proj_grad=" in warning_msg

    def test_handles_nan_inf_gradients_in_top_offenders(self):
        """Should mark NaN/Inf gradients with (!) in top offenders."""
        logger = MagicMock()
        base_model = MagicMock()
        base_model.layers = []

        grad_info_pre_clip = [
            ("layer1.weight", 10.0, 5.0, True, False),  # NaN
            ("layer2.weight", 20.0, 8.0, False, True),  # Inf
            ("layer3.weight", 50.0, 10.0, False, False),  # Normal
        ]

        log_gradient_pathology_diagnostic(
            logger=logger,
            step=100,
            pre_clip_norm=100.0,
            grad_norm=10.0,
            grad_clip=1.0,
            clip_coef=0.1,
            clip_scale=0.5,
            scaler_scale=65536.0,
            lr=0.001,
            lr_effective=0.001,
            spike_detected=True,
            grad_spike_lr_factor=1.0,
            accum_loss=2.5,
            last_ce_loss=2.5,
            last_aux_loss=0.0,
            last_ponder_loss=0.0,
            micro_diag=[],
            grad_info_pre_clip=grad_info_pre_clip,
            base_model=base_model,
        )

        warning_msg = str(logger.warning.call_args)
        # NaN/Inf params should be marked with (!)
        assert "(!)" in warning_msg

    def test_handles_scaler_scale_none(self):
        """Should handle None scaler_scale gracefully."""
        logger = MagicMock()
        base_model = MagicMock()
        base_model.layers = []

        log_gradient_pathology_diagnostic(
            logger=logger,
            step=100,
            pre_clip_norm=100.0,
            grad_norm=10.0,
            grad_clip=1.0,
            clip_coef=0.1,
            clip_scale=0.5,
            scaler_scale=None,
            lr=0.001,
            lr_effective=0.001,
            spike_detected=True,
            grad_spike_lr_factor=1.0,
            accum_loss=2.5,
            last_ce_loss=2.5,
            last_aux_loss=0.0,
            last_ponder_loss=0.0,
            micro_diag=[],
            grad_info_pre_clip=[],
            base_model=base_model,
        )

        warning_msg = str(logger.warning.call_args)
        assert "amp_scale=n/a" in warning_msg

    def test_handles_model_with_layers_and_attn_stats(self):
        """Should extract attention output stats from model layers."""
        logger = MagicMock()

        # Create mock layer with attention output stats
        layer0 = MagicMock()
        layer0._last_attn_out_rms_t = torch.tensor([1.5])
        layer0._last_attn_out_dtype = "float16"
        layer0._last_attn_out_shape = (2, 128, 768)
        layer0._last_attn_out_finite_frac_t = torch.tensor([0.999])
        layer0._last_attn_out_nan_ct_t = torch.tensor([1])
        layer0._last_attn_out_inf_ct_t = torch.tensor([0])

        layer1 = MagicMock()
        layer1._last_attn_out_rms_t = torch.tensor([2.0])
        layer1._last_attn_out_dtype = "float16"
        layer1._last_attn_out_shape = (2, 128, 768)
        layer1._last_attn_out_finite_frac_t = torch.tensor([1.0])
        layer1._last_attn_out_nan_ct_t = torch.tensor([0])
        layer1._last_attn_out_inf_ct_t = torch.tensor([0])

        base_model = MagicMock()
        base_model.layers = [layer0, layer1]

        log_gradient_pathology_diagnostic(
            logger=logger,
            step=100,
            pre_clip_norm=100.0,
            grad_norm=10.0,
            grad_clip=1.0,
            clip_coef=0.1,
            clip_scale=0.5,
            scaler_scale=65536.0,
            lr=0.001,
            lr_effective=0.001,
            spike_detected=True,
            grad_spike_lr_factor=1.0,
            accum_loss=2.5,
            last_ce_loss=2.5,
            last_aux_loss=0.0,
            last_ponder_loss=0.0,
            micro_diag=[],
            grad_info_pre_clip=[],
            base_model=base_model,
        )

        warning_msg = str(logger.warning.call_args)
        assert "L0_attn=" in warning_msg
        assert "L1_attn=" in warning_msg

    def test_handles_nonfinite_attn_rms(self):
        """Should handle non-finite attention RMS values."""
        logger = MagicMock()

        layer0 = MagicMock()
        layer0._last_attn_out_rms_t = torch.tensor([float("inf")])
        layer0._last_attn_out_dtype = None
        layer0._last_attn_out_shape = None
        layer0._last_attn_out_finite_frac_t = None
        layer0._last_attn_out_nan_ct_t = None
        layer0._last_attn_out_inf_ct_t = None

        base_model = MagicMock()
        base_model.layers = [layer0]

        log_gradient_pathology_diagnostic(
            logger=logger,
            step=100,
            pre_clip_norm=100.0,
            grad_norm=10.0,
            grad_clip=1.0,
            clip_coef=0.1,
            clip_scale=0.5,
            scaler_scale=65536.0,
            lr=0.001,
            lr_effective=0.001,
            spike_detected=True,
            grad_spike_lr_factor=1.0,
            accum_loss=2.5,
            last_ce_loss=2.5,
            last_aux_loss=0.0,
            last_ponder_loss=0.0,
            micro_diag=[],
            grad_info_pre_clip=[],
            base_model=base_model,
        )

        warning_msg = str(logger.warning.call_args)
        assert "rms=n/a" in warning_msg

    def test_handles_exception_in_layer_access(self):
        """Should handle exceptions when accessing layer attributes."""
        logger = MagicMock()

        base_model = MagicMock()
        # Make layers raise an exception
        type(base_model).layers = property(lambda self: (_ for _ in ()).throw(RuntimeError("Access denied")))

        # Should not raise, just skip layer stats
        log_gradient_pathology_diagnostic(
            logger=logger,
            step=100,
            pre_clip_norm=100.0,
            grad_norm=10.0,
            grad_clip=1.0,
            clip_coef=0.1,
            clip_scale=0.5,
            scaler_scale=65536.0,
            lr=0.001,
            lr_effective=0.001,
            spike_detected=True,
            grad_spike_lr_factor=1.0,
            accum_loss=2.5,
            last_ce_loss=2.5,
            last_aux_loss=0.0,
            last_ponder_loss=0.0,
            micro_diag=[],
            grad_info_pre_clip=[],
            base_model=base_model,
        )

        # Should still log something
        logger.warning.assert_called_once()

    def test_verbose_parameter_accepted(self):
        """Should accept verbose parameter without error."""
        logger = MagicMock()
        base_model = MagicMock()
        base_model.layers = []

        # Should not raise with verbose=True
        log_gradient_pathology_diagnostic(
            logger=logger,
            step=100,
            pre_clip_norm=100.0,
            grad_norm=10.0,
            grad_clip=1.0,
            clip_coef=0.1,
            clip_scale=0.5,
            scaler_scale=65536.0,
            lr=0.001,
            lr_effective=0.001,
            spike_detected=True,
            grad_spike_lr_factor=1.0,
            accum_loss=2.5,
            last_ce_loss=2.5,
            last_aux_loss=0.0,
            last_ponder_loss=0.0,
            micro_diag=[],
            grad_info_pre_clip=[],
            base_model=base_model,
            verbose=True,
        )

        logger.warning.assert_called_once()

    def test_empty_grad_info_pre_clip(self):
        """Should handle empty grad_info_pre_clip list."""
        logger = MagicMock()
        base_model = MagicMock()
        base_model.layers = []

        log_gradient_pathology_diagnostic(
            logger=logger,
            step=100,
            pre_clip_norm=100.0,
            grad_norm=10.0,
            grad_clip=1.0,
            clip_coef=0.1,
            clip_scale=0.5,
            scaler_scale=65536.0,
            lr=0.001,
            lr_effective=0.001,
            spike_detected=True,
            grad_spike_lr_factor=1.0,
            accum_loss=2.5,
            last_ce_loss=2.5,
            last_aux_loss=0.0,
            last_ponder_loss=0.0,
            micro_diag=[],
            grad_info_pre_clip=[],
            base_model=base_model,
        )

        # Should still log even with empty grad info
        logger.warning.assert_called_once()
        warning_msg = str(logger.warning.call_args)
        assert "top_grad=[]" in warning_msg
