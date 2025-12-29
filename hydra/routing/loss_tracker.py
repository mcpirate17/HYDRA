"""
Loss Tracker - Moving Average Baseline for Loss-Driven Routing

Implements adaptive computation routing based on token difficulty,
as determined by cross-entropy loss relative to a moving baseline.

Key Concept:
    Instead of forcing a target depth distribution, let the model learn
    which tokens need more compute from the task signal (CE loss).
    
    - Token loss > baseline → harder than average → reward deeper routing
    - Token loss < baseline → easier than average → reward shallow routing

Usage:
    baseline = MovingAverageBaseline(decay=0.99, warmup_steps=1000)
    
    # In training loop:
    token_losses = compute_per_token_loss(logits, targets)
    advantage = baseline.compute_advantage(token_losses)
    baseline.update(token_losses)
    
    # Use advantage to scale router gradients
    ponder_loss = router.compute_ponder_loss(..., token_losses, baseline)
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = [
    "MovingAverageBaseline",
    "AdvantageScaledSTE",
]


class MovingAverageBaseline(nn.Module):
    """Exponential moving average of per-token losses.
    
    Provides a self-calibrating baseline for advantage computation.
    The baseline adapts as the model improves, maintaining a relative
    difficulty signal throughout training.
    
    Attributes:
        decay: EMA decay rate (0.99 = slow adaptation, 0.9 = fast)
        warmup_steps: Steps before advantage computation activates
    """
    
    __slots__ = ('decay', 'warmup_steps', '_step')
    
    def __init__(
        self,
        decay: float = 0.99,
        warmup_steps: int = 1000,
    ) -> None:
        super().__init__()
        self.decay = decay
        self.warmup_steps = warmup_steps
        
        # EMA state (not a learnable parameter)
        self.register_buffer("_ema", torch.tensor(0.0), persistent=True)
        self.register_buffer("_ema_var", torch.tensor(1.0), persistent=True)
        self.register_buffer("_step", torch.tensor(0, dtype=torch.long), persistent=True)
        self.register_buffer("_initialized", torch.tensor(False), persistent=True)
    
    @property
    def baseline(self) -> float:
        """Current baseline value (EMA of loss)."""
        return self._ema.item()
    
    @property
    def baseline_std(self) -> float:
        """Standard deviation estimate for normalization."""
        return max(0.01, self._ema_var.sqrt().item())
    
    @property
    def step(self) -> int:
        """Current step count."""
        return self._step.item()
    
    @property
    def is_active(self) -> bool:
        """Whether advantage computation is active (past warmup)."""
        return self._step.item() >= self.warmup_steps
    
    @torch.no_grad()
    def update(self, token_losses: torch.Tensor) -> None:
        """Update EMA with new batch of losses.
        
        Args:
            token_losses: Per-token losses [batch, seq] or flattened [N]
        """
        # Flatten and compute batch statistics
        flat = token_losses.detach().flatten()
        batch_mean = flat.mean()
        batch_var = flat.var()
        
        # Initialize on first update
        if not self._initialized:
            self._ema.copy_(batch_mean)
            self._ema_var.copy_(batch_var)
            self._initialized.fill_(True)
        else:
            # EMA update
            self._ema.mul_(self.decay).add_(batch_mean * (1 - self.decay))
            self._ema_var.mul_(self.decay).add_(batch_var * (1 - self.decay))
        
        # Increment step
        self._step.add_(1)
    
    def compute_advantage(self, token_losses: torch.Tensor) -> torch.Tensor:
        """Compute advantage signal for router training.
        
        Advantage = (token_loss - baseline) / std
        
        - Positive: token is harder than average → benefit from more depth
        - Negative: token is easier than average → can use less depth
        - Zero: during warmup (no advantage signal)
        
        Args:
            token_losses: Per-token losses [batch, seq]
        
        Returns:
            Advantage tensor [batch, seq], normalized, zero during warmup.
        """
        if not self.is_active:
            return torch.zeros_like(token_losses)
        
        # Normalize advantage by standard deviation for stable gradients
        std = max(0.01, self._ema_var.sqrt().item())
        advantage = (token_losses - self._ema) / std
        
        # Soft clamp to prevent extreme values
        return torch.tanh(advantage)
    
    def state_dict_extra(self) -> dict:
        """Get extra state for checkpointing."""
        return {
            "ema": self._ema.item(),
            "ema_var": self._ema_var.item(),
            "step": self._step.item(),
            "initialized": self._initialized.item(),
        }
    
    def load_state_dict_extra(self, state: dict) -> None:
        """Load extra state from checkpoint."""
        if "ema" in state:
            self._ema.fill_(state["ema"])
        if "ema_var" in state:
            self._ema_var.fill_(state["ema_var"])
        if "step" in state:
            self._step.fill_(state["step"])
        if "initialized" in state:
            self._initialized.fill_(state["initialized"])


class AdvantageScaledSTE(torch.autograd.Function):
    """Straight-through estimator with advantage-scaled gradients.
    
    Forward: passes discrete depth decisions unchanged
    Backward: scales gradients by advantage (difficulty signal)
    
    This allows the router to learn from the task loss:
    - Hard tokens (high advantage) get stronger depth gradients
    - Easy tokens (low advantage) get weaker depth gradients
    """
    
    @staticmethod
    def forward(
        ctx,
        router_logits: torch.Tensor,
        depths: torch.Tensor,
        advantages: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass: return depths unchanged.
        
        Args:
            router_logits: Raw router outputs [batch, seq]
            depths: Discrete depth assignments [batch, seq]
            advantages: Per-token advantage signal [batch, seq]
        
        Returns:
            depths tensor (unchanged)
        """
        ctx.save_for_backward(router_logits, advantages)
        return depths.float()
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Backward pass: scale gradients by advantage.
        
        Args:
            grad_output: Gradient from downstream [batch, seq]
        
        Returns:
            Tuple of gradients (router_logits_grad, None, None)
        """
        router_logits, advantages = ctx.saved_tensors
        
        # Scale gradient by advantage
        # High advantage (hard token) → larger gradient → router learns to go deeper
        # Low advantage (easy token) → smaller gradient → router learns to stay shallow
        scaled_grad = grad_output * (1.0 + advantages)
        
        return scaled_grad, None, None


def apply_advantage_ste(
    router_logits: torch.Tensor,
    depths: torch.Tensor,
    advantages: torch.Tensor,
) -> torch.Tensor:
    """Apply advantage-scaled STE to depth assignments.
    
    Convenience wrapper for AdvantageScaledSTE.apply().
    
    Args:
        router_logits: Raw router outputs [batch, seq]
        depths: Discrete depth assignments [batch, seq]
        advantages: Per-token advantage signal [batch, seq]
    
    Returns:
        depths as float tensor with STE gradient path.
    """
    return AdvantageScaledSTE.apply(router_logits, depths, advantages)
