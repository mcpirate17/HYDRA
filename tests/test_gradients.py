"""Tests for hydra.training.gradients module.

Tests gradient clipping, scaling, and spike handling utilities.
"""

from __future__ import annotations

import math
import time
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from hydra.training.gradients import (
    log_gradient_pathology_diagnostic,
    skip_update_for_nonfinite_gradients,
    reset_optimizer_moments_for_gradient_spike,
    maybe_prepare_halt_on_spike,
)


class TestSkipUpdateForNonfiniteGradients:
    """Tests for skip_update_for_nonfinite_gradients."""

    def test_returns_false_when_grads_finite(self):
        """When gradients are finite, should return False and do nothing."""
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

    def test_zeros_grad_when_nonfinite(self):
        """When gradients are non-finite, should zero gradients."""
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

    def test_updates_scaler_when_nonfinite_and_use_scaler(self):
        """When nonfinite and using scaler, should update scaler."""
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
    """Tests for reset_optimizer_moments_for_gradient_spike."""

    def test_returns_early_when_no_spike(self):
        """Should do nothing when spike_detected is False."""
        optimizer = MagicMock()
        base_model = MagicMock()

        reset_optimizer_moments_for_gradient_spike(
            spike_detected=False,
            grad_spike_reset_moments=True,
            grad_spike_topk=3,
            grad_info_pre_clip=[],
            base_model=base_model,
            optimizer=optimizer,
            logger=None,
        )

        base_model.named_parameters.assert_not_called()

    def test_returns_early_when_reset_disabled(self):
        """Should do nothing when grad_spike_reset_moments is False."""
        optimizer = MagicMock()
        base_model = MagicMock()

        reset_optimizer_moments_for_gradient_spike(
            spike_detected=True,
            grad_spike_reset_moments=False,
            grad_spike_topk=3,
            grad_info_pre_clip=[],
            base_model=base_model,
            optimizer=optimizer,
            logger=None,
        )

        base_model.named_parameters.assert_not_called()

    def test_resets_moments_for_top_offenders(self):
        """Should reset Adam moments for parameters with highest grad norms."""
        # Create a simple model with real parameters
        model = nn.Linear(4, 4)
        param_name = "weight"

        # Create gradient info with the param as top offender
        grad_info = [
            (param_name, 100.0, 50.0, False, False),  # High norm
            ("bias", 1.0, 0.5, False, False),  # Low norm
        ]

        # Create optimizer with state
        optimizer = torch.optim.Adam(model.parameters())
        # Run a step to initialize state
        model.weight.grad = torch.ones_like(model.weight)
        model.bias.grad = torch.ones_like(model.bias)
        optimizer.step()

        # Verify state exists
        weight_state = optimizer.state[model.weight]
        assert "exp_avg" in weight_state
        weight_state["exp_avg"].fill_(5.0)
        weight_state["exp_avg_sq"].fill_(10.0)

        reset_optimizer_moments_for_gradient_spike(
            spike_detected=True,
            grad_spike_reset_moments=True,
            grad_spike_topk=1,
            grad_info_pre_clip=grad_info,
            base_model=model,
            optimizer=optimizer,
            logger=None,
        )

        # Weight moments should be zeroed (top offender)
        assert torch.all(optimizer.state[model.weight]["exp_avg"] == 0)
        assert torch.all(optimizer.state[model.weight]["exp_avg_sq"] == 0)
        # Bias moments should NOT be zeroed (not in top-k)
        assert not torch.all(optimizer.state[model.bias]["exp_avg"] == 0)

    def test_handles_nan_inf_as_infinite_norm(self):
        """Parameters with NaN/Inf grads should be treated as infinite norm."""
        model = nn.Linear(4, 4)

        # NaN grad should be prioritized over high finite norm
        grad_info = [
            ("weight", 1.0, 0.5, True, False),  # has NaN
            ("bias", 1000.0, 500.0, False, False),  # high norm but finite
        ]

        optimizer = torch.optim.Adam(model.parameters())
        model.weight.grad = torch.ones_like(model.weight)
        model.bias.grad = torch.ones_like(model.bias)
        optimizer.step()

        optimizer.state[model.weight]["exp_avg"].fill_(5.0)
        optimizer.state[model.bias]["exp_avg"].fill_(5.0)

        reset_optimizer_moments_for_gradient_spike(
            spike_detected=True,
            grad_spike_reset_moments=True,
            grad_spike_topk=1,  # Only top 1
            grad_info_pre_clip=grad_info,
            base_model=model,
            optimizer=optimizer,
            logger=None,
        )

        # Weight should be zeroed (has NaN = infinite priority)
        assert torch.all(optimizer.state[model.weight]["exp_avg"] == 0)
        # Bias should NOT be zeroed (not in top-1 due to NaN priority)
        assert not torch.all(optimizer.state[model.bias]["exp_avg"] == 0)

    def test_logs_warning_on_success_when_logger_provided(self):
        """Should log a warning message when logger is provided."""
        model = nn.Linear(4, 4)
        grad_info = [("weight", 100.0, 50.0, False, False)]

        optimizer = torch.optim.Adam(model.parameters())
        model.weight.grad = torch.ones_like(model.weight)
        optimizer.step()

        logger = MagicMock()

        reset_optimizer_moments_for_gradient_spike(
            spike_detected=True,
            grad_spike_reset_moments=True,
            grad_spike_topk=1,
            grad_info_pre_clip=grad_info,
            base_model=model,
            optimizer=optimizer,
            logger=logger,
        )

        logger.warning.assert_called_once()
        assert "reset Adam moments" in logger.warning.call_args[0][0]

    def test_handles_missing_optimizer_state(self):
        """Should handle parameters with no optimizer state gracefully."""
        model = nn.Linear(4, 4)
        grad_info = [("weight", 100.0, 50.0, False, False)]

        # Create optimizer but don't initialize state
        optimizer = torch.optim.Adam(model.parameters())
        model.weight.grad = torch.ones_like(model.weight)
        # Don't call optimizer.step() - state won't be initialized

        # Should not raise
        reset_optimizer_moments_for_gradient_spike(
            spike_detected=True,
            grad_spike_reset_moments=True,
            grad_spike_topk=1,
            grad_info_pre_clip=grad_info,
            base_model=model,
            optimizer=optimizer,
            logger=None,
        )

    def test_handles_amsgrad_max_exp_avg_sq(self):
        """Should also reset max_exp_avg_sq for AMSGrad optimizers."""
        model = nn.Linear(4, 4)
        grad_info = [("weight", 100.0, 50.0, False, False)]

        optimizer = torch.optim.Adam(model.parameters(), amsgrad=True)
        model.weight.grad = torch.ones_like(model.weight)
        optimizer.step()

        optimizer.state[model.weight]["max_exp_avg_sq"].fill_(5.0)

        reset_optimizer_moments_for_gradient_spike(
            spike_detected=True,
            grad_spike_reset_moments=True,
            grad_spike_topk=1,
            grad_info_pre_clip=grad_info,
            base_model=model,
            optimizer=optimizer,
            logger=None,
        )

        assert torch.all(optimizer.state[model.weight]["max_exp_avg_sq"] == 0)


class TestMaybePrepareHaltOnSpike:
    """Tests for maybe_prepare_halt_on_spike."""

    def test_returns_false_when_no_spike(self):
        """Should return False and do nothing when spike_detected is False."""
        metrics = MagicMock()
        save_checkpoint = MagicMock()
        logger = MagicMock()

        result = maybe_prepare_halt_on_spike(
            spike_detected=False,
            halt_on_spike=True,
            step=100,
            accum_loss=1.0,
            lr_effective=1e-4,
            grad_norm=1.0,
            tokens_per_step=1024,
            step_start=time.time(),
            metrics=metrics,
            save_checkpoint=save_checkpoint,
            logger=logger,
        )

        assert result is False
        metrics.update.assert_not_called()
        save_checkpoint.assert_not_called()

    def test_returns_false_when_halt_disabled(self):
        """Should return False when halt_on_spike is False."""
        metrics = MagicMock()
        save_checkpoint = MagicMock()
        logger = MagicMock()

        result = maybe_prepare_halt_on_spike(
            spike_detected=True,
            halt_on_spike=False,
            step=100,
            accum_loss=1.0,
            lr_effective=1e-4,
            grad_norm=1.0,
            tokens_per_step=1024,
            step_start=time.time(),
            metrics=metrics,
            save_checkpoint=save_checkpoint,
            logger=logger,
        )

        assert result is False
        metrics.update.assert_not_called()
        save_checkpoint.assert_not_called()

    def test_saves_checkpoint_on_halt(self):
        """Should update metrics and save checkpoint when halting."""
        metrics = MagicMock()
        metrics.total_tokens = 0
        save_checkpoint = MagicMock()
        logger = MagicMock()

        step_start = time.time() - 1.0  # 1 second ago

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
        save_checkpoint.assert_called_once_with(100)
        assert metrics.final_loss == 2.5
        assert metrics.total_tokens == 1024

    def test_handles_exception_gracefully(self):
        """Should catch exceptions and log warning."""
        metrics = MagicMock()
        metrics.update.side_effect = RuntimeError("Test error")
        save_checkpoint = MagicMock()
        logger = MagicMock()

        result = maybe_prepare_halt_on_spike(
            spike_detected=True,
            halt_on_spike=True,
            step=100,
            accum_loss=2.5,
            lr_effective=1e-4,
            grad_norm=50.0,
            tokens_per_step=1024,
            step_start=time.time(),
            metrics=metrics,
            save_checkpoint=save_checkpoint,
            logger=logger,
        )

        assert result is True
        logger.warning.assert_called_once()
        assert "failed to record/save state" in logger.warning.call_args[0][0]


class TestLogGradientPathologyDiagnostic:
    """Tests for log_gradient_pathology_diagnostic."""

    def test_logs_basic_spike_info(self):
        """Should log basic spike information."""
        logger = MagicMock()
        base_model = MagicMock()
        base_model.layers = []

        log_gradient_pathology_diagnostic(
            logger=logger,
            step=100,
            pre_clip_norm=100.0,
            grad_norm=1.0,
            grad_clip=1.0,
            clip_coef=0.01,
            clip_scale=1.0,
            scaler_scale=1024.0,
            lr=1e-4,
            lr_effective=1e-5,
            spike_detected=True,
            grad_spike_lr_factor=0.1,
            accum_loss=2.5,
            last_ce_loss=2.0,
            last_aux_loss=0.3,
            last_ponder_loss=0.2,
            micro_diag=[],
            grad_info_pre_clip=[],
            base_model=base_model,
        )

        logger.warning.assert_called_once()
        log_msg = logger.warning.call_args[0][0]
        assert "Step 100" in log_msg
        assert "1.00e+02" in log_msg  # pre_clip_norm
        assert "1.00e+00" in log_msg  # grad_norm

    def test_extracts_layer_o_proj_grad_norms(self):
        """Should extract o_proj grad norms for layers 0 and 1."""
        logger = MagicMock()
        base_model = MagicMock()
        base_model.layers = []

        grad_info = [
            ("layers.0.attention.o_proj.weight", 10.0, 5.0, False, False),
            ("layers.1.attention.o_proj.weight", 20.0, 10.0, False, False),
            ("layers.2.attention.o_proj.weight", 30.0, 15.0, False, False),
        ]

        log_gradient_pathology_diagnostic(
            logger=logger,
            step=100,
            pre_clip_norm=100.0,
            grad_norm=1.0,
            grad_clip=1.0,
            clip_coef=0.01,
            clip_scale=1.0,
            scaler_scale=1024.0,
            lr=1e-4,
            lr_effective=1e-5,
            spike_detected=True,
            grad_spike_lr_factor=0.1,
            accum_loss=2.5,
            last_ce_loss=2.0,
            last_aux_loss=0.3,
            last_ponder_loss=0.2,
            micro_diag=[],
            grad_info_pre_clip=grad_info,
            base_model=base_model,
        )

        log_msg = logger.warning.call_args[0][0]
        # L0 and L1 o_proj grad norms should be in output
        assert "L0/L1_o_proj_grad" in log_msg
        assert "1.00e+01" in log_msg  # Layer 0: 10.0
        assert "2.00e+01" in log_msg  # Layer 1: 20.0

    def test_shows_top_offenders(self):
        """Should show top 3 gradient offenders."""
        logger = MagicMock()
        base_model = MagicMock()
        base_model.layers = []

        grad_info = [
            ("param_a", 5.0, 2.0, False, False),
            ("param_b", 100.0, 50.0, False, False),  # Top offender
            ("param_c", 50.0, 25.0, False, False),   # 2nd
            ("param_d", 75.0, 35.0, False, False),   # 3rd
        ]

        log_gradient_pathology_diagnostic(
            logger=logger,
            step=100,
            pre_clip_norm=100.0,
            grad_norm=1.0,
            grad_clip=1.0,
            clip_coef=0.01,
            clip_scale=1.0,
            scaler_scale=None,
            lr=1e-4,
            lr_effective=1e-5,
            spike_detected=True,
            grad_spike_lr_factor=0.1,
            accum_loss=2.5,
            last_ce_loss=2.0,
            last_aux_loss=0.3,
            last_ponder_loss=0.2,
            micro_diag=[],
            grad_info_pre_clip=grad_info,
            base_model=base_model,
        )

        log_msg = logger.warning.call_args[0][0]
        assert "top_grad=" in log_msg
        assert "param_b" in log_msg  # Top offender should be included

    def test_marks_nan_inf_offenders(self):
        """Should mark NaN/Inf parameters with (!) indicator."""
        logger = MagicMock()
        base_model = MagicMock()
        base_model.layers = []

        grad_info = [
            ("param_with_nan", 1.0, 0.5, True, False),  # has NaN
            ("param_with_inf", 1.0, 0.5, False, True),  # has Inf
            ("param_normal", 10.0, 5.0, False, False),
        ]

        log_gradient_pathology_diagnostic(
            logger=logger,
            step=100,
            pre_clip_norm=100.0,
            grad_norm=1.0,
            grad_clip=1.0,
            clip_coef=0.01,
            clip_scale=1.0,
            scaler_scale=1024.0,
            lr=1e-4,
            lr_effective=1e-5,
            spike_detected=True,
            grad_spike_lr_factor=0.1,
            accum_loss=2.5,
            last_ce_loss=2.0,
            last_aux_loss=0.3,
            last_ponder_loss=0.2,
            micro_diag=[],
            grad_info_pre_clip=grad_info,
            base_model=base_model,
        )

        log_msg = logger.warning.call_args[0][0]
        # NaN/Inf params should have (!) indicator
        assert "(!)" in log_msg

    def test_handles_scaler_scale_none(self):
        """Should handle None scaler_scale gracefully."""
        logger = MagicMock()
        base_model = MagicMock()
        base_model.layers = []

        log_gradient_pathology_diagnostic(
            logger=logger,
            step=100,
            pre_clip_norm=100.0,
            grad_norm=1.0,
            grad_clip=1.0,
            clip_coef=0.01,
            clip_scale=1.0,
            scaler_scale=None,
            lr=1e-4,
            lr_effective=1e-5,
            spike_detected=True,
            grad_spike_lr_factor=0.1,
            accum_loss=2.5,
            last_ce_loss=2.0,
            last_aux_loss=0.3,
            last_ponder_loss=0.2,
            micro_diag=[],
            grad_info_pre_clip=[],
            base_model=base_model,
        )

        log_msg = logger.warning.call_args[0][0]
        assert "amp_scale=n/a" in log_msg

    def test_extracts_attention_output_stats(self):
        """Should extract attention output RMS from model layers."""
        logger = MagicMock()

        # Create mock layers with attention stats
        layer0 = MagicMock()
        layer0._last_attn_out_rms_t = torch.tensor([1.5])
        layer0._last_attn_out_finite_frac_t = torch.tensor([0.999])
        layer0._last_attn_out_nan_ct_t = torch.tensor([0])
        layer0._last_attn_out_inf_ct_t = torch.tensor([0])
        layer0._last_attn_out_dtype = "float16"
        layer0._last_attn_out_shape = (2, 128, 768)

        layer1 = MagicMock()
        layer1._last_attn_out_rms_t = torch.tensor([2.5])
        layer1._last_attn_out_finite_frac_t = torch.tensor([0.998])
        layer1._last_attn_out_nan_ct_t = torch.tensor([1])
        layer1._last_attn_out_inf_ct_t = torch.tensor([2])
        layer1._last_attn_out_dtype = "float16"
        layer1._last_attn_out_shape = (2, 128, 768)

        base_model = MagicMock()
        base_model.layers = [layer0, layer1]

        log_gradient_pathology_diagnostic(
            logger=logger,
            step=100,
            pre_clip_norm=100.0,
            grad_norm=1.0,
            grad_clip=1.0,
            clip_coef=0.01,
            clip_scale=1.0,
            scaler_scale=1024.0,
            lr=1e-4,
            lr_effective=1e-5,
            spike_detected=True,
            grad_spike_lr_factor=0.1,
            accum_loss=2.5,
            last_ce_loss=2.0,
            last_aux_loss=0.3,
            last_ponder_loss=0.2,
            micro_diag=[],
            grad_info_pre_clip=[],
            base_model=base_model,
        )

        log_msg = logger.warning.call_args[0][0]
        assert "L0_attn=" in log_msg
        assert "L1_attn=" in log_msg
        assert "rms=" in log_msg

    def test_handles_model_without_layers(self):
        """Should handle models without layers attribute gracefully."""
        logger = MagicMock()
        base_model = MagicMock(spec=[])  # No layers attribute

        log_gradient_pathology_diagnostic(
            logger=logger,
            step=100,
            pre_clip_norm=100.0,
            grad_norm=1.0,
            grad_clip=1.0,
            clip_coef=0.01,
            clip_scale=1.0,
            scaler_scale=1024.0,
            lr=1e-4,
            lr_effective=1e-5,
            spike_detected=True,
            grad_spike_lr_factor=0.1,
            accum_loss=2.5,
            last_ce_loss=2.0,
            last_aux_loss=0.3,
            last_ponder_loss=0.2,
            micro_diag=[],
            grad_info_pre_clip=[],
            base_model=base_model,
        )

        # Should not raise, should still log
        logger.warning.assert_called_once()

    def test_handles_nonfinite_scaler_scale(self):
        """Should show n/a for non-finite scaler_scale."""
        logger = MagicMock()
        base_model = MagicMock()
        base_model.layers = []

        log_gradient_pathology_diagnostic(
            logger=logger,
            step=100,
            pre_clip_norm=100.0,
            grad_norm=1.0,
            grad_clip=1.0,
            clip_coef=0.01,
            clip_scale=1.0,
            scaler_scale=float("inf"),
            lr=1e-4,
            lr_effective=1e-5,
            spike_detected=True,
            grad_spike_lr_factor=0.1,
            accum_loss=2.5,
            last_ce_loss=2.0,
            last_aux_loss=0.3,
            last_ponder_loss=0.2,
            micro_diag=[],
            grad_info_pre_clip=[],
            base_model=base_model,
        )

        log_msg = logger.warning.call_args[0][0]
        assert "amp_scale=n/a" in log_msg

    def test_handles_empty_grad_info(self):
        """Should handle empty grad_info_pre_clip gracefully."""
        logger = MagicMock()
        base_model = MagicMock()
        base_model.layers = []

        log_gradient_pathology_diagnostic(
            logger=logger,
            step=100,
            pre_clip_norm=100.0,
            grad_norm=1.0,
            grad_clip=1.0,
            clip_coef=0.01,
            clip_scale=1.0,
            scaler_scale=1024.0,
            lr=1e-4,
            lr_effective=1e-5,
            spike_detected=True,
            grad_spike_lr_factor=0.1,
            accum_loss=2.5,
            last_ce_loss=2.0,
            last_aux_loss=0.3,
            last_ponder_loss=0.2,
            micro_diag=[],
            grad_info_pre_clip=[],
            base_model=base_model,
        )

        logger.warning.assert_called_once()
        log_msg = logger.warning.call_args[0][0]
        assert "top_grad=[]" in log_msg
