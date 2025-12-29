import torch

from hydra.training.config import TrainingConfig
from hydra.training.loop import resolve_micro_diag_tensors, update_scalar_ema
from hydra.training.lr_step import compute_step_lr


def test_update_scalar_ema_initializes_and_updates():
    assert update_scalar_ema(ema=0.0, value=10.0, alpha=0.05) == 10.0
    out = update_scalar_ema(ema=10.0, value=0.0, alpha=0.1)
    assert abs(out - 9.0) < 1e-9


def test_resolve_micro_diag_tensors_converts_tensor_values():
    micro_diag = [
        {"x_min": torch.tensor(1), "x_max": torch.tensor(2.5), "keep": "ok"},
        {"y_oob": torch.tensor(0), "loss": 1.234},
    ]
    resolve_micro_diag_tensors(micro_diag)
    assert micro_diag[0]["x_min"] == 1
    assert micro_diag[0]["x_max"] == 2.5
    assert micro_diag[0]["keep"] == "ok"
    assert micro_diag[1]["y_oob"] == 0
    assert micro_diag[1]["loss"] == 1.234


def test_compute_step_lr_applies_resume_override_and_clamps_min_lr():
    cfg = TrainingConfig()

    base_lr, lr, new_scale, new_override = compute_step_lr(
        step=123,
        config=cfg,
        adaptive_lr=None,
        adaptive_metric="train",
        accum_loss=1.0,
        resume_lr_scale=1.0,
        resume_lr_override_target=0.5,
        get_lr=lambda _step, _cfg: 1.0,
    )

    assert base_lr == 1.0
    assert lr == 0.5
    assert new_scale == 0.5
    assert new_override == 0.0

    # Clamp check: scale to below min_lr should still clamp
    cfg.min_lr = 0.2
    base_lr, lr, *_ = compute_step_lr(
        step=123,
        config=cfg,
        adaptive_lr=None,
        adaptive_metric="train",
        accum_loss=1.0,
        resume_lr_scale=0.01,
        resume_lr_override_target=0.0,
        get_lr=lambda _step, _cfg: 1.0,
    )
    assert base_lr == 1.0
    assert lr == 0.2
