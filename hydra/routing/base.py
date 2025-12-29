from __future__ import annotations

import math

import torch
import torch.nn as nn


class BaseRouter(nn.Module):
    def __init__(self, dim: int, jitter_noise: float = 0.0) -> None:
        super().__init__()
        self.dim = dim
        self.jitter_noise = jitter_noise
        self.router = nn.Linear(dim, 1, bias=True)

    def _init_bias(self, prob: float) -> None:
        with torch.no_grad():
            p = float(min(max(prob, 1e-4), 1.0 - 1e-4))
            self.router.bias.fill_(math.log(p / (1.0 - p)))

    def _apply_jitter(self, logits: torch.Tensor) -> torch.Tensor:
        if self.training and self.jitter_noise > 0:
            return logits + torch.randn_like(logits) * self.jitter_noise
        return logits

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.router(x).squeeze(-1)
        return self._apply_jitter(logits)
