from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from .config import TrainingConfig


def get_lr_cosine(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (max_lr - min_lr) * cosine_decay


def get_lr_wsd(step: int, warmup_steps: int, decay_start_step: int, decay_steps: int, max_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step < decay_start_step:
        return max_lr
    decay_progress = (step - decay_start_step) / max(1, decay_steps)
    decay_progress = min(1.0, decay_progress)
    # Smooth warmdown: cosine easing avoids a sharp kink at decay_start_step.
    # This keeps LR continuous with zero slope at the start/end of decay.
    warmdown = 0.5 * (1.0 - math.cos(math.pi * decay_progress))
    return max_lr - (max_lr - min_lr) * warmdown


class ProgressAwareLRManager:
    def __init__(self, config: TrainingConfig, start_step: int = 0):
        self.config = config
        self.start_step = start_step
        self.is_resume = start_step > 0

        self.max_lr = config.max_lr
        self.min_lr = config.min_lr

        self._recalculate_schedule(start_step, config.max_steps)

        self.cooldown_triggered = False
        self.adaptive_decay_start = None
        self.adaptive_decay_steps = None

        self.loss_history: List[float] = []
        self.loss_ema_short = 0.0
        self.loss_ema_long = 0.0
        self.patience_counter = 0

        self.swa_active = False
        self.swa_n = 0
        self.swa_state: Optional[Dict[str, torch.Tensor]] = None

    def _recalculate_schedule(self, current_step: int, max_steps: int) -> None:
        remaining_steps = max_steps - current_step

        if self.is_resume:
            self.warmup_steps = 0
            self.warmup_end_step = current_step
            stable_steps = int(remaining_steps * 0.85)
            self.decay_start_step = current_step + stable_steps
            self.decay_steps = remaining_steps - stable_steps
        else:
            self.warmup_steps = max(100, int(max_steps * 0.01))
            self.warmup_end_step = self.warmup_steps
            post_warmup_steps = max_steps - self.warmup_steps
            stable_steps = int(post_warmup_steps * 0.85)
            self.decay_start_step = self.warmup_steps + stable_steps
            self.decay_steps = post_warmup_steps - stable_steps

        self.swa_start_step = int(max_steps * self.config.swa_start_pct)
        self._print_schedule(current_step, max_steps)

    def _print_schedule(self, current_step: int, max_steps: int) -> None:
        print(f"\n{'='*60}")
        print(f"📈 LR SCHEDULE {'(RESUMED)' if self.is_resume else '(FRESH START)'}")
        print(f"{'='*60}")
        print(f"   Current step:  {current_step:,}")
        print(f"   Target steps:  {max_steps:,}")
        print(f"   Remaining:     {max_steps - current_step:,} steps")
        print(f"   ---")
        if not self.is_resume:
            print(f"   Warmup:        steps 0 - {self.warmup_end_step:,} (LR 0 → {self.max_lr})")
        print(f"   Stable:        steps {self.warmup_end_step:,} - {self.decay_start_step:,} (LR = {self.max_lr})")
        print(f"   Decay:         steps {self.decay_start_step:,} - {max_steps:,} (LR {self.max_lr} → {self.min_lr})")
        if self.config.use_swa:
            print(f"   SWA:           starts at step {self.swa_start_step:,}")
        print(f"{'='*60}\n")

    def update_max_steps(self, new_max_steps: int, current_step: int) -> None:
        print(f"\n⚠️  MAX_STEPS CHANGED: {self.config.max_steps:,} → {new_max_steps:,}")
        self.config.max_steps = new_max_steps
        self._recalculate_schedule(current_step, new_max_steps)

    def update(self, step: int, loss: float) -> None:
        if self.loss_ema_short == 0:
            self.loss_ema_short = loss
            self.loss_ema_long = loss
        else:
            self.loss_ema_short = 0.1 * loss + 0.9 * self.loss_ema_short
            self.loss_ema_long = 0.02 * loss + 0.98 * self.loss_ema_long

        self.loss_history.append(loss)
        if len(self.loss_history) > 100:
            self.loss_history.pop(0)

        if self.config.adaptive_lr and not self.cooldown_triggered:
            self._check_cooldown_trigger(step, loss)

    def _check_cooldown_trigger(self, step: int, loss: float) -> None:
        # Guardrail: don't allow adaptive cooldown too early in the run.
        # Early training (and curriculum ramps) can produce transient spikes;
        # without this, a single early trigger can drive LR to min_lr long
        # before the planned decay window.
        min_pct = float(getattr(self.config, "adaptive_min_trigger_pct", 0.0) or 0.0)
        if min_pct > 0:
            min_step = int(self.config.max_steps * min_pct)
            if step < min_step:
                return
        if len(self.loss_history) < 10 or self.loss_ema_long == 0:
            return
        relative_increase = (self.loss_ema_short - self.loss_ema_long) / self.loss_ema_long
        if relative_increase > self.config.adaptive_threshold:
            self.patience_counter += 1
            if self.patience_counter >= self.config.adaptive_patience:
                self._trigger_cooldown(step)
        else:
            self.patience_counter = max(0, self.patience_counter - 1)

    def _trigger_cooldown(self, step: int) -> None:
        self.cooldown_triggered = True
        self.adaptive_decay_start = step
        remaining = self.config.max_steps - step
        self.adaptive_decay_steps = int(remaining * self.config.adaptive_cooldown_factor)
        print(f"\n{'='*60}")
        print(f"⚡ ADAPTIVE COOLDOWN TRIGGERED at step {step:,}")
        print(f"   Short-term EMA: {self.loss_ema_short:.4f}")
        print(f"   Long-term EMA:  {self.loss_ema_long:.4f}")
        print(f"   Relative increase: {(self.loss_ema_short - self.loss_ema_long) / self.loss_ema_long:.1%}")
        print(f"   Starting decay over {self.adaptive_decay_steps:,} steps")
        print(f"   LR will decay: {self.max_lr} → {self.min_lr}")
        print(f"{'='*60}\n")

    def get_lr(self, step: int) -> float:
        if self.cooldown_triggered:
            decay_start = self.adaptive_decay_start
            decay_steps = self.adaptive_decay_steps
        else:
            decay_start = self.decay_start_step
            decay_steps = self.decay_steps

        if not self.is_resume and step < self.warmup_steps:
            lr = self.max_lr * (step + 1) / self.warmup_steps
        elif step < decay_start:
            lr = self.max_lr
        else:
            progress = (step - decay_start) / max(1, decay_steps)
            progress = min(1.0, progress)
            warmdown = 0.5 * (1.0 - math.cos(math.pi * progress))
            lr = self.max_lr - (self.max_lr - self.min_lr) * warmdown

        if self.config.use_swa and step >= self.swa_start_step:
            lr = lr * self.config.swa_lr_factor
            if not self.swa_active:
                self.swa_active = True
                print(f"\n⚡ SWA activated at step {step:,}")

        return lr

    def update_swa(self, model: nn.Module) -> None:
        if not self.config.use_swa or not self.swa_active:
            return

        self.swa_n += 1
        with torch.no_grad():
            if self.swa_state is None:
                self.swa_state = {n: p.data.clone() for n, p in model.named_parameters()}
            else:
                for n, p in model.named_parameters():
                    self.swa_state[n].add_((p.data - self.swa_state[n]) / self.swa_n)

    def apply_swa(self, model: nn.Module) -> None:
        if self.swa_state is None:
            return
        print(f"\nApplying SWA weights (averaged over {self.swa_n} updates)...")
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in self.swa_state:
                    p.data.copy_(self.swa_state[n])

    def get_state(self) -> Dict[str, Any]:
        return {
            "start_step": self.start_step,
            "is_resume": self.is_resume,
            "warmup_steps": self.warmup_steps,
            "warmup_end_step": self.warmup_end_step,
            "decay_start_step": self.decay_start_step,
            "decay_steps": self.decay_steps,
            "cooldown_triggered": self.cooldown_triggered,
            "adaptive_decay_start": self.adaptive_decay_start,
            "adaptive_decay_steps": self.adaptive_decay_steps,
            "loss_ema_short": self.loss_ema_short,
            "loss_ema_long": self.loss_ema_long,
            "loss_history": list(self.loss_history),
            "patience_counter": self.patience_counter,
            "swa_active": self.swa_active,
            "swa_n": self.swa_n,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        self.cooldown_triggered = state.get("cooldown_triggered", False)
        self.adaptive_decay_start = state.get("adaptive_decay_start")
        self.adaptive_decay_steps = state.get("adaptive_decay_steps")
        self.loss_ema_short = state.get("loss_ema_short", state.get("loss_ema", 0.0))
        self.loss_ema_long = state.get("loss_ema_long", state.get("loss_ema", 0.0))
        self.patience_counter = state.get("patience_counter", 0)
        self.swa_active = state.get("swa_active", False)
        self.swa_n = state.get("swa_n", 0)
        saved_history = state.get("loss_history", [])
        if saved_history:
            self.loss_history = saved_history.copy()
            print(f"  Restored {len(self.loss_history)} loss history entries")
            print(f"  EMA short (10-step): {self.loss_ema_short:.4f}, EMA long (100-step): {self.loss_ema_long:.4f}")


class AdaptiveLRManager:
    """
    DEPRECATED: Use ProgressAwareLRManager instead.
    
    This class is kept for backward compatibility but is not used internally.
    It will be removed in a future release.
    """
    def __init__(self, config: TrainingConfig):
        import warnings
        warnings.warn(
            "AdaptiveLRManager is deprecated and will be removed. "
            "Use ProgressAwareLRManager instead.",
            DeprecationWarning,
            stacklevel=2
        )
        self.config = config
        self.base_max_lr = config.max_lr
        self.current_max_lr = config.max_lr

        self.cooldown_triggered = False
        self.cooldown_start_step = config.decay_start_step
        self.cooldown_steps = config.decay_steps

        self.loss_history: List[float] = []
        self.loss_ema = 0.0
        self.loss_ema_alpha = 0.1
        self.best_loss_window = float("inf")
        self.patience_counter = 0

        self.swa_active = False
        self.swa_start_step = int(config.max_steps * config.swa_start_pct)
        self.swa_n = 0
        self.swa_state: Optional[Dict[str, torch.Tensor]] = None

        self.plateau_counter = 0
        self.plateau_threshold = 10

    def update(self, step: int, loss: float) -> None:
        if self.loss_ema == 0:
            self.loss_ema = loss
        else:
            self.loss_ema = self.loss_ema_alpha * loss + (1 - self.loss_ema_alpha) * self.loss_ema

        self.loss_history.append(loss)
        if len(self.loss_history) > 100:
            self.loss_history.pop(0)

        if self.config.adaptive_lr and not self.cooldown_triggered:
            self._check_cooldown_trigger(step, loss)

        if loss < self.best_loss_window:
            self.best_loss_window = loss
            self.plateau_counter = 0
        else:
            self.plateau_counter += 1

    def _check_cooldown_trigger(self, step: int, loss: float) -> None:
        if len(self.loss_history) < 10:
            return
        recent_avg = sum(self.loss_history[-5:]) / 5
        older_avg = sum(self.loss_history[-15:-5]) / 10 if len(self.loss_history) >= 15 else recent_avg
        if older_avg > 0 and (recent_avg - older_avg) / older_avg > self.config.adaptive_threshold:
            self.patience_counter += 1
            if self.patience_counter >= self.config.adaptive_patience:
                self._trigger_cooldown(step)
        else:
            self.patience_counter = max(0, self.patience_counter - 1)

    def _trigger_cooldown(self, step: int) -> None:
        self.cooldown_triggered = True
        self.cooldown_start_step = step
        remaining_steps = self.config.max_steps - step
        self.cooldown_steps = int(remaining_steps * self.config.adaptive_cooldown_factor)
        self.current_max_lr = self.base_max_lr * self.config.adaptive_lr_reduction
        print(f"\n" + "=" * 60)
        print(f"⚡ ADAPTIVE LR: Cooldown triggered at step {step}")
        print(f"   Loss trend: {sum(self.loss_history[-15:-5])/10:.4f} -> {sum(self.loss_history[-5:])/5:.4f}")
        print(f"   New decay phase: {self.cooldown_steps} steps")
        print(f"   Max LR reduced: {self.base_max_lr} -> {self.current_max_lr}")
        print("=" * 60 + "\n")

    def get_lr(self, step: int) -> float:
        config = self.config
        if self.cooldown_triggered:
            decay_start = self.cooldown_start_step
            decay_steps = self.cooldown_steps
            max_lr = self.current_max_lr
        else:
            decay_start = config.decay_start_step
            decay_steps = config.decay_steps
            max_lr = config.max_lr

        lr = get_lr_wsd(step, config.warmup_steps, decay_start, decay_steps, max_lr, config.min_lr)

        if config.use_swa and step >= self.swa_start_step:
            lr = lr * config.swa_lr_factor
            if not self.swa_active:
                self.swa_active = True
                print(f"\n⚡ SWA activated at step {step}, LR factor: {config.swa_lr_factor}")

        return lr

    def update_swa(self, model: nn.Module) -> None:
        if not self.config.use_swa or not self.swa_active:
            return
        self.swa_n += 1
        with torch.no_grad():
            if self.swa_state is None:
                self.swa_state = {}
                for name, param in model.named_parameters():
                    self.swa_state[name] = param.data.clone()
            else:
                for name, param in model.named_parameters():
                    self.swa_state[name].add_((param.data - self.swa_state[name]) / self.swa_n)

    def apply_swa(self, model: nn.Module) -> None:
        if self.swa_state is None:
            return
        print(f"\nApplying SWA weights (averaged over {self.swa_n} updates)...")
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.swa_state:
                    param.data.copy_(self.swa_state[name])

    def get_state(self) -> Dict[str, Any]:
        return {
            "cooldown_triggered": self.cooldown_triggered,
            "cooldown_start_step": self.cooldown_start_step,
            "cooldown_steps": self.cooldown_steps,
            "current_max_lr": self.current_max_lr,
            "loss_ema": self.loss_ema,
            "best_loss_window": self.best_loss_window,
            "patience_counter": self.patience_counter,
            "swa_active": self.swa_active,
            "swa_n": self.swa_n,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        self.cooldown_triggered = state.get("cooldown_triggered", False)
        self.cooldown_start_step = state.get("cooldown_start_step", self.config.decay_start_step)
        self.cooldown_steps = state.get("cooldown_steps", self.config.decay_steps)
        self.current_max_lr = state.get("current_max_lr", self.config.max_lr)
        self.loss_ema = state.get("loss_ema", 0.0)
        self.best_loss_window = state.get("best_loss_window", float("inf"))
        self.patience_counter = state.get("patience_counter", 0)
        self.swa_active = state.get("swa_active", False)
        self.swa_n = state.get("swa_n", 0)


def get_lr(step: int, config: TrainingConfig) -> float:
    if config.lr_schedule in ("wsd", "wsd_adaptive"):
        return get_lr_wsd(step, config.warmup_steps, config.decay_start_step, config.decay_steps, config.max_lr, config.min_lr)
    return get_lr_cosine(step, config.warmup_steps, config.max_steps, config.max_lr, config.min_lr)
