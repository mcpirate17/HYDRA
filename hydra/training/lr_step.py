from __future__ import annotations

from typing import Optional, Protocol, Tuple

from .config import TrainingConfig


class _AdaptiveLRLike(Protocol):
    def update(self, step: int, loss: float) -> None: ...

    def get_lr(self, step: int) -> float: ...


def compute_step_lr(
    *,
    step: int,
    config: TrainingConfig,
    adaptive_lr: Optional[_AdaptiveLRLike],
    adaptive_metric: str,
    accum_loss: float,
    resume_lr_scale: float,
    resume_lr_override_target: float,
    resume_rewarmup_start_step: int = 0,
    resume_rewarmup_steps: int = 0,
    get_lr,
) -> Tuple[float, float, float, float]:
    """Compute base LR and effective LR for this step.

    Returns:
      base_lr, lr, new_resume_lr_scale, new_resume_lr_override_target

    Notes:
    - If `adaptive_lr` is present and `adaptive_metric=='train'`, it is updated every 100 steps.
    - If `resume_lr_override_target>0`, it is converted into a scale factor against `base_lr`
      and then cleared (returned as 0).
    - `lr` is clamped to `config.min_lr`.
    """

    if adaptive_lr is not None:
        if adaptive_metric == "train" and step % 100 == 0:
            adaptive_lr.update(step, float(accum_loss))
        base_lr = float(adaptive_lr.get_lr(step))
    else:
        base_lr = float(get_lr(step, config))

    new_resume_lr_scale = float(resume_lr_scale)
    new_resume_lr_override_target = float(resume_lr_override_target)

    if new_resume_lr_override_target > 0.0:
        denom = max(1e-12, float(base_lr))
        new_resume_lr_scale = float(new_resume_lr_override_target) / denom
        new_resume_lr_override_target = 0.0

    lr = float(base_lr) * float(new_resume_lr_scale)

    # Optional: re-warmup LR after resume when optimizer/scaler/RNG state could not be restored.
    # This intentionally allows LR to go below min_lr during the rewarm window.
    if int(resume_rewarmup_steps) > 0 and int(step) >= int(resume_rewarmup_start_step):
        rel = int(step) - int(resume_rewarmup_start_step)
        if rel < int(resume_rewarmup_steps):
            denom = max(1, int(resume_rewarmup_steps))
            frac = float(rel + 1) / float(denom)
            frac = max(0.0, min(1.0, frac))
            lr = float(lr) * float(frac)
            return base_lr, lr, new_resume_lr_scale, new_resume_lr_override_target

    lr = max(float(config.min_lr), float(lr))

    return base_lr, lr, new_resume_lr_scale, new_resume_lr_override_target
