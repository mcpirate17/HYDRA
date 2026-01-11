"""Tests for hydra/training/gradients.py - gradient clipping and scaling functions."""

import math
import time
from unittest.mock import MagicMock, patch

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

    def test_returns_false_when_grads_are_finite(self):
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

    def test_calls_scaler_update_when_use_scaler_true(self):
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


class TestResetOptimizerMomentsForGradientSpike:
    """Tests for reset_optimizer_moments_for_gradient_spike function."""

    def test_does_nothing_when_no_spike(self):
        optimizer = MagicMock()
        base_model = MagicMock()

        reset_optimizer_moments_for_gradient_spike(
            spike_detected=False,
            grad_spike_reset_moments=True,
            grad_spike_topk=3,
            grad_info_pre_clip=[("param1", 1.0, 0.5, False, False)],
            base_model=base_model,
            optimizer=optimizer,
            logger=None,
        )

        # No interactions expected
        base_model.named_parameters.assert_not_called()

    def test_does_nothing_when_reset_moments_disabled(self):
        optimizer = MagicMock()
        base_model = MagicMock()

        reset_optimizer_moments_for_gradient_spike(
            spike_detected=True,
            grad_spike_reset_moments=False,
            grad_spike_topk=3,
            grad_info_pre_clip=[("param1", 1.0, 0.5, False, False)],
            base_model=base_model,
            optimizer=optimizer,
            logger=None,
        )

        base_model.named_parameters.assert_not_called()

    def test_resets_moments_for_top_offenders(self):
        # Create mock model parameters
        param1 = torch.nn.Parameter(torch.randn(10))
        param1.grad = torch.randn(10)
        param2 = torch.nn.Parameter(torch.randn(10))
        param2.grad = torch.randn(10)
        param3 = torch.nn.Parameter(torch.randn(10))
        param3.grad = None  # No gradient

        base_model = MagicMock()
        base_model.named_parameters.return_value = [
            ("param1", param1),
            ("param2", param2),
            ("param3", param3),
        ]

        # Create optimizer state with moments
        exp_avg_1 = torch.ones(10)
        exp_avg_sq_1 = torch.ones(10) * 2
        exp_avg_2 = torch.ones(10) * 3
        exp_avg_sq_2 = torch.ones(10) * 4

        optimizer = MagicMock()
        optimizer.state = {
            param1: {"exp_avg": exp_avg_1, "exp_avg_sq": exp_avg_sq_1},
            param2: {"exp_avg": exp_avg_2, "exp_avg_sq": exp_avg_sq_2},
        }

        # grad_info_pre_clip: (name, grad_norm, grad_max, has_nan, has_inf)
        # param1 has highest norm, param2 second
        grad_info = [
            ("param1", 10.0, 5.0, False, False),
            ("param2", 5.0, 2.5, False, False),
            ("param3", 1.0, 0.5, False, False),
        ]

        reset_optimizer_moments_for_gradient_spike(
            spike_detected=True,
            grad_spike_reset_moments=True,
            grad_spike_topk=2,  # Only top 2
            grad_info_pre_clip=grad_info,
            base_model=base_model,
            optimizer=optimizer,
            logger=None,
        )

        # Check param1 moments were zeroed
        assert torch.all(exp_avg_1 == 0).item()
        assert torch.all(exp_avg_sq_1 == 0).item()
        # Check param2 moments were zeroed
        assert torch.all(exp_avg_2 == 0).item()
        assert torch.all(exp_avg_sq_2 == 0).item()

    def test_prioritizes_nan_and_inf_parameters(self):
        param1 = torch.nn.Parameter(torch.randn(10))
        param1.grad = torch.randn(10)
        param_nan = torch.nn.Parameter(torch.randn(10))
        param_nan.grad = torch.randn(10)

        base_model = MagicMock()
        base_model.named_parameters.return_value = [
            ("param1", param1),
            ("param_nan", param_nan),
        ]

        exp_avg_nan = torch.ones(10)
        exp_avg_sq_nan = torch.ones(10)

        optimizer = MagicMock()
        optimizer.state = {
            param1: {"exp_avg": torch.ones(10), "exp_avg_sq": torch.ones(10)},
            param_nan: {"exp_avg": exp_avg_nan, "exp_avg_sq": exp_avg_sq_nan},
        }

        # param_nan has NaN gradients (lower norm but should be prioritized)
        grad_info = [
            ("param1", 100.0, 50.0, False, False),
            ("param_nan", 1.0, 0.5, True, False),  # has_nan=True
        ]

        reset_optimizer_moments_for_gradient_spike(
            spike_detected=True,
            grad_spike_reset_moments=True,
            grad_spike_topk=1,  # Only top 1
            grad_info_pre_clip=grad_info,
            base_model=base_model,
            optimizer=optimizer,
            logger=None,
        )

        # param_nan should be reset (NaN takes priority over high norm)
        assert torch.all(exp_avg_nan == 0).item()
        assert torch.all(exp_avg_sq_nan == 0).item()

    def test_resets_max_exp_avg_sq_if_present(self):
        param = torch.nn.Parameter(torch.randn(10))
        param.grad = torch.randn(10)

        base_model = MagicMock()
        base_model.named_parameters.return_value = [("param", param)]

        max_exp_avg_sq = torch.ones(10) * 5

        optimizer = MagicMock()
        optimizer.state = {
            param: {
                "exp_avg": torch.ones(10),
                "exp_avg_sq": torch.ones(10),
                "max_exp_avg_sq": max_exp_avg_sq,  # AMSGrad variant
            },
        }

        grad_info = [("param", 10.0, 5.0, False, False)]

        reset_optimizer_moments_for_gradient_spike(
            spike_detected=True,
            grad_spike_reset_moments=True,
            grad_spike_topk=1,
            grad_info_pre_clip=grad_info,
            base_model=base_model,
            optimizer=optimizer,
            logger=None,
        )

        assert torch.all(max_exp_avg_sq == 0).item()

    def test_logs_warning_when_logger_provided(self):
        param = torch.nn.Parameter(torch.randn(10))
        param.grad = torch.randn(10)

        base_model = MagicMock()
        base_model.named_parameters.return_value = [("param", param)]

        optimizer = MagicMock()
        optimizer.state = {param: {"exp_avg": torch.ones(10), "exp_avg_sq": torch.ones(10)}}

        logger = MagicMock()
        grad_info = [("param", 10.0, 5.0, False, False)]

        reset_optimizer_moments_for_gradient_spike(
            spike_detected=True,
            grad_spike_reset_moments=True,
            grad_spike_topk=1,
            grad_info_pre_clip=grad_info,
            base_model=base_model,
            optimizer=optimizer,
            logger=logger,
        )

        logger.warning.assert_called_once()
        assert "reset Adam moments" in logger.warning.call_args[0][0]

    def test_handles_exception_gracefully(self):
        base_model = MagicMock()
        base_model.named_parameters.side_effect = RuntimeError("Test error")

        optimizer = MagicMock()
        logger = MagicMock()

        # Should not raise
        reset_optimizer_moments_for_gradient_spike(
            spike_detected=True,
            grad_spike_reset_moments=True,
            grad_spike_topk=1,
            grad_info_pre_clip=[("param", 10.0, 5.0, False, False)],
            base_model=base_model,
            optimizer=optimizer,
            logger=logger,
        )

        logger.warning.assert_called_once()
        assert "failed" in logger.warning.call_args[0][0]


class TestMaybePrepareHaltOnSpike:
    """Tests for maybe_prepare_halt_on_spike function."""

    def test_returns_false_when_no_spike(self):
        result = maybe_prepare_halt_on_spike(
            spike_detected=False,
            halt_on_spike=True,
            step=100,
            accum_loss=1.0,
            lr_effective=1e-4,
            grad_norm=10.0,
            tokens_per_step=1024,
            step_start=time.time(),
            metrics=MagicMock(),
            save_checkpoint=MagicMock(),
            logger=MagicMock(),
        )

        assert result is False

    def test_returns_false_when_halt_disabled(self):
        result = maybe_prepare_halt_on_spike(
            spike_detected=True,
            halt_on_spike=False,
            step=100,
            accum_loss=1.0,
            lr_effective=1e-4,
            grad_norm=10.0,
            tokens_per_step=1024,
            step_start=time.time(),
            metrics=MagicMock(),
            save_checkpoint=MagicMock(),
            logger=MagicMock(),
        )

        assert result is False

    def test_returns_true_and_saves_checkpoint_on_spike(self):
        # Use a simple class instead of MagicMock for attributes modified with +=
        class MockMetrics:
            def __init__(self):
                self.total_tokens = 0
                self.final_loss = 0.0

            def update(self, step, loss, lr, grad_norm, tps, step_time):
                pass

        metrics = MockMetrics()
        save_checkpoint = MagicMock()
        step_start = time.time() - 1.0  # 1 second ago

        result = maybe_prepare_halt_on_spike(
            spike_detected=True,
            halt_on_spike=True,
            step=100,
            accum_loss=2.5,
            lr_effective=1e-4,
            grad_norm=100.0,
            tokens_per_step=2048,
            step_start=step_start,
            metrics=metrics,
            save_checkpoint=save_checkpoint,
            logger=MagicMock(),
        )

        assert result is True
        save_checkpoint.assert_called_once_with(100)
        assert metrics.total_tokens == 2048
        assert metrics.final_loss == 2.5

    def test_handles_exception_in_checkpoint_save(self):
        metrics = MagicMock()
        metrics.update.side_effect = RuntimeError("Save failed")
        logger = MagicMock()

        result = maybe_prepare_halt_on_spike(
            spike_detected=True,
            halt_on_spike=True,
            step=100,
            accum_loss=2.5,
            lr_effective=1e-4,
            grad_norm=100.0,
            tokens_per_step=2048,
            step_start=time.time(),
            metrics=metrics,
            save_checkpoint=MagicMock(),
            logger=logger,
        )

        assert result is True  # Still returns True
        logger.warning.assert_called_once()
        assert "failed" in logger.warning.call_args[0][0]


class TestLogGradientPathologyDiagnostic:
    """Tests for log_gradient_pathology_diagnostic function."""

    def test_logs_warning_with_basic_info(self):
        logger = MagicMock()
        base_model = MagicMock()
        base_model.layers = []  # Empty layers

        log_gradient_pathology_diagnostic(
            logger=logger,
            step=100,
            pre_clip_norm=50.0,
            grad_norm=1.0,
            grad_clip=1.0,
            clip_coef=0.02,
            clip_scale=1.0,
            scaler_scale=65536.0,
            lr=1e-4,
            lr_effective=5e-5,
            spike_detected=True,
            grad_spike_lr_factor=0.5,
            accum_loss=2.5,
            last_ce_loss=2.4,
            last_aux_loss=0.05,
            last_ponder_loss=0.05,
            micro_diag=[],
            grad_info_pre_clip=[("param1", 25.0, 10.0, False, False)],
            base_model=base_model,
        )

        logger.warning.assert_called_once()
        call_args = logger.warning.call_args[0][0]
        assert "Step 100" in call_args
        assert "50.0" in call_args or "5.00e+01" in call_args
        assert "loss=2.5" in call_args

    def test_extracts_layer_0_and_1_o_proj_grad_norms(self):
        logger = MagicMock()
        base_model = MagicMock()
        base_model.layers = []

        grad_info = [
            ("layers.0.attention.o_proj.weight", 15.0, 5.0, False, False),
            ("layers.1.attention.o_proj.weight", 10.0, 3.0, False, False),
            ("layers.2.attention.o_proj.weight", 5.0, 2.0, False, False),
        ]

        log_gradient_pathology_diagnostic(
            logger=logger,
            step=50,
            pre_clip_norm=30.0,
            grad_norm=1.0,
            grad_clip=1.0,
            clip_coef=0.033,
            clip_scale=1.0,
            scaler_scale=None,
            lr=1e-4,
            lr_effective=1e-4,
            spike_detected=True,
            grad_spike_lr_factor=1.0,
            accum_loss=1.5,
            last_ce_loss=1.4,
            last_aux_loss=0.05,
            last_ponder_loss=0.05,
            micro_diag=[],
            grad_info_pre_clip=grad_info,
            base_model=base_model,
        )

        call_args = logger.warning.call_args[0][0]
        # Should contain L0/L1 o_proj grad norms
        assert "L0/L1_o_proj_grad" in call_args
        assert "1.50e+01" in call_args  # 15.0
        assert "1.00e+01" in call_args  # 10.0

    def test_handles_scaler_scale_none(self):
        logger = MagicMock()
        base_model = MagicMock()
        base_model.layers = []

        log_gradient_pathology_diagnostic(
            logger=logger,
            step=100,
            pre_clip_norm=50.0,
            grad_norm=1.0,
            grad_clip=1.0,
            clip_coef=0.02,
            clip_scale=1.0,
            scaler_scale=None,  # No AMP scaler
            lr=1e-4,
            lr_effective=1e-4,
            spike_detected=True,
            grad_spike_lr_factor=1.0,
            accum_loss=2.0,
            last_ce_loss=2.0,
            last_aux_loss=0.0,
            last_ponder_loss=0.0,
            micro_diag=[],
            grad_info_pre_clip=[],
            base_model=base_model,
        )

        call_args = logger.warning.call_args[0][0]
        assert "amp_scale=n/a" in call_args

    def test_identifies_top_gradient_offenders(self):
        logger = MagicMock()
        base_model = MagicMock()
        base_model.layers = []

        grad_info = [
            ("small_param", 1.0, 0.5, False, False),
            ("medium_param", 10.0, 5.0, False, False),
            ("large_param", 100.0, 50.0, False, False),
            ("nan_param", 5.0, 2.0, True, False),  # Has NaN
        ]

        log_gradient_pathology_diagnostic(
            logger=logger,
            step=100,
            pre_clip_norm=100.0,
            grad_norm=1.0,
            grad_clip=1.0,
            clip_coef=0.01,
            clip_scale=1.0,
            scaler_scale=65536.0,
            lr=1e-4,
            lr_effective=1e-4,
            spike_detected=True,
            grad_spike_lr_factor=1.0,
            accum_loss=2.0,
            last_ce_loss=2.0,
            last_aux_loss=0.0,
            last_ponder_loss=0.0,
            micro_diag=[],
            grad_info_pre_clip=grad_info,
            base_model=base_model,
        )

        call_args = logger.warning.call_args[0][0]
        # NaN param should appear in top with (!) marker
        assert "nan_param" in call_args
        assert "(!)" in call_args
        # Large param should also be in top
        assert "large_param" in call_args

    def test_extracts_attention_output_stats_from_layers(self):
        logger = MagicMock()

        # Create mock layers with attention output stats
        layer0 = MagicMock()
        layer0._last_attn_out_rms_t = torch.tensor([0.5])
        layer0._last_attn_out_dtype = "bfloat16"
        layer0._last_attn_out_shape = (4, 128, 768)
        layer0._last_attn_out_finite_frac_t = torch.tensor([0.999])
        layer0._last_attn_out_nan_ct_t = torch.tensor([2])
        layer0._last_attn_out_inf_ct_t = torch.tensor([0])

        layer1 = MagicMock()
        layer1._last_attn_out_rms_t = torch.tensor([0.6])
        layer1._last_attn_out_dtype = "bfloat16"
        layer1._last_attn_out_shape = (4, 128, 768)
        layer1._last_attn_out_finite_frac_t = torch.tensor([1.0])
        layer1._last_attn_out_nan_ct_t = torch.tensor([0])
        layer1._last_attn_out_inf_ct_t = torch.tensor([0])

        base_model = MagicMock()
        base_model.layers = [layer0, layer1]

        log_gradient_pathology_diagnostic(
            logger=logger,
            step=100,
            pre_clip_norm=50.0,
            grad_norm=1.0,
            grad_clip=1.0,
            clip_coef=0.02,
            clip_scale=1.0,
            scaler_scale=65536.0,
            lr=1e-4,
            lr_effective=1e-4,
            spike_detected=True,
            grad_spike_lr_factor=1.0,
            accum_loss=2.0,
            last_ce_loss=2.0,
            last_aux_loss=0.0,
            last_ponder_loss=0.0,
            micro_diag=[],
            grad_info_pre_clip=[],
            base_model=base_model,
        )

        call_args = logger.warning.call_args[0][0]
        # Should contain attention stats
        assert "L0_attn" in call_args
        assert "L1_attn" in call_args
        assert "bfloat16" in call_args

    def test_handles_missing_layer_attributes_gracefully(self):
        logger = MagicMock()

        # Layer without attention stats
        layer0 = MagicMock(spec=[])  # No attributes

        base_model = MagicMock()
        base_model.layers = [layer0]

        # Should not raise
        log_gradient_pathology_diagnostic(
            logger=logger,
            step=100,
            pre_clip_norm=50.0,
            grad_norm=1.0,
            grad_clip=1.0,
            clip_coef=0.02,
            clip_scale=1.0,
            scaler_scale=65536.0,
            lr=1e-4,
            lr_effective=1e-4,
            spike_detected=True,
            grad_spike_lr_factor=1.0,
            accum_loss=2.0,
            last_ce_loss=2.0,
            last_aux_loss=0.0,
            last_ponder_loss=0.0,
            micro_diag=[],
            grad_info_pre_clip=[],
            base_model=base_model,
        )

        logger.warning.assert_called_once()

    def test_handles_infinite_scaler_scale(self):
        logger = MagicMock()
        base_model = MagicMock()
        base_model.layers = []

        log_gradient_pathology_diagnostic(
            logger=logger,
            step=100,
            pre_clip_norm=50.0,
            grad_norm=1.0,
            grad_clip=1.0,
            clip_coef=0.02,
            clip_scale=1.0,
            scaler_scale=float("inf"),
            lr=1e-4,
            lr_effective=1e-4,
            spike_detected=True,
            grad_spike_lr_factor=1.0,
            accum_loss=2.0,
            last_ce_loss=2.0,
            last_aux_loss=0.0,
            last_ponder_loss=0.0,
            micro_diag=[],
            grad_info_pre_clip=[],
            base_model=base_model,
        )

        call_args = logger.warning.call_args[0][0]
        assert "amp_scale=n/a" in call_args

    def test_handles_nan_rms_in_attention_stats(self):
        logger = MagicMock()

        layer0 = MagicMock()
        layer0._last_attn_out_rms_t = torch.tensor([float("nan")])
        layer0._last_attn_out_dtype = "bfloat16"
        layer0._last_attn_out_shape = (4, 128, 768)
        layer0._last_attn_out_finite_frac_t = torch.tensor([0.5])
        layer0._last_attn_out_nan_ct_t = torch.tensor([100])
        layer0._last_attn_out_inf_ct_t = torch.tensor([50])

        base_model = MagicMock()
        base_model.layers = [layer0]

        # Should not raise
        log_gradient_pathology_diagnostic(
            logger=logger,
            step=100,
            pre_clip_norm=50.0,
            grad_norm=1.0,
            grad_clip=1.0,
            clip_coef=0.02,
            clip_scale=1.0,
            scaler_scale=65536.0,
            lr=1e-4,
            lr_effective=1e-4,
            spike_detected=True,
            grad_spike_lr_factor=1.0,
            accum_loss=2.0,
            last_ce_loss=2.0,
            last_aux_loss=0.0,
            last_ponder_loss=0.0,
            micro_diag=[],
            grad_info_pre_clip=[],
            base_model=base_model,
        )

        call_args = logger.warning.call_args[0][0]
        # RMS should show n/a for NaN
        assert "rms=n/a" in call_args
