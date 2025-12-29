from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class TrainingMetrics:
    """Track training metrics with preallocated storage."""

    losses: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    grad_norms: List[float] = field(default_factory=list)
    tokens_per_sec: List[float] = field(default_factory=list)
    step_times: List[float] = field(default_factory=list)

    start_time: float = 0.0
    end_time: float = 0.0
    total_tokens: int = 0
    best_loss: float = float("inf")
    best_loss_step: int = 0
    initial_loss: float = 0.0
    final_loss: float = 0.0

    ema_loss: float = 0.0
    ema_alpha: float = 0.05

    def update(self, step: int, loss: float, lr: float, grad_norm: float, tps: float, step_time: float) -> None:
        self.losses.append(loss)
        self.learning_rates.append(lr)
        self.grad_norms.append(grad_norm)
        self.tokens_per_sec.append(tps)
        self.step_times.append(step_time)

        if self.ema_loss == 0.0:
            self.ema_loss = loss
        else:
            self.ema_loss = self.ema_alpha * loss + (1 - self.ema_alpha) * self.ema_loss

        if loss < self.best_loss:
            self.best_loss = loss
            self.best_loss_step = step

    def to_dict(self) -> Dict[str, Any]:
        return {
            "losses": self.losses,
            "learning_rates": self.learning_rates,
            "grad_norms": self.grad_norms,
            "tokens_per_sec": self.tokens_per_sec,
            "step_times": self.step_times,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_tokens": self.total_tokens,
            "best_loss": self.best_loss,
            "best_loss_step": self.best_loss_step,
            "initial_loss": self.initial_loss,
            "final_loss": self.final_loss,
            "training_time_seconds": self.end_time - self.start_time,
        }
