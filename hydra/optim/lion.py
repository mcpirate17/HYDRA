"""LION and C-LION optimizers.

LION (Evolved Sign Momentum): Google Brain's discovered optimizer that uses
sign-based updates for memory efficiency and speed.
Paper: https://arxiv.org/abs/2302.06675

C-LION (Cautious LION): A variant with cautious updates that only applies
updates when the sign of gradient and momentum agree, reducing noise.
Paper: https://arxiv.org/abs/2411.16085
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch
from torch.optim.optimizer import Optimizer


class Lion(Optimizer):
    """LION optimizer - Evolved Sign Momentum.
    
    Key properties:
    - Uses sign(interpolate(momentum, grad)) for updates
    - 2x memory savings vs AdamW (no second moment)
    - Faster than AdamW per step
    - Requires ~3-10x lower LR than AdamW
    
    Recommended LR: 1e-4 to 3e-4 (vs AdamW's 1e-3 to 3e-3)
    Recommended weight_decay: 0.1 to 0.3 (higher than AdamW)
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.1,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            wd = group["weight_decay"]
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]
                
                # Initialize momentum buffer
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                
                exp_avg = state["exp_avg"]
                
                # Weight decay (decoupled)
                if wd != 0:
                    p.mul_(1 - lr * wd)
                
                # LION update: sign(beta1 * m + (1 - beta1) * g)
                update = exp_avg.mul(beta1).add(grad, alpha=1 - beta1)
                p.add_(update.sign(), alpha=-lr)
                
                # Update momentum: m = beta2 * m + (1 - beta2) * g
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)
        
        return loss


class CautiousLion(Optimizer):
    """Cautious LION (C-LION) optimizer.
    
    Applies updates only when gradient and momentum agree in sign.
    This reduces noise and can improve stability.
    
    Key difference from LION:
    - Creates a mask where sign(grad) == sign(momentum)
    - Only applies updates where mask is True
    - Can lead to more stable training, especially early on
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.1,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            wd = group["weight_decay"]
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]
                
                # Initialize momentum buffer
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                
                exp_avg = state["exp_avg"]
                
                # Weight decay (decoupled)
                if wd != 0:
                    p.mul_(1 - lr * wd)
                
                # Compute update direction
                update = exp_avg.mul(beta1).add(grad, alpha=1 - beta1)
                
                # C-LION: Only update where grad and momentum agree
                # mask = (sign(grad) == sign(exp_avg))
                mask = (grad * exp_avg) > 0
                
                # Apply masked sign update
                signed_update = update.sign()
                signed_update = signed_update * mask.float()
                p.add_(signed_update, alpha=-lr)
                
                # Update momentum (always, not masked)
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)
        
        return loss


class ScheduleFreeLion(Optimizer):
    """Schedule-Free LION variant.
    
    Combines LION's sign-based updates with schedule-free training.
    Maintains both x (evaluation point) and z (optimization point).
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.1,
        warmup_steps: int = 0,
    ):
        defaults = dict(
            lr=lr, 
            betas=betas, 
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
        )
        super().__init__(params, defaults)
        self._step = 0
    
    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        self._step += 1
        
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            wd = group["weight_decay"]
            warmup = group["warmup_steps"]
            
            # Warmup LR
            if warmup > 0 and self._step < warmup:
                lr = lr * self._step / warmup
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]
                
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                    state["z"] = p.clone()  # Optimization point
                
                exp_avg = state["exp_avg"]
                z = state["z"]
                
                # Weight decay on z
                if wd != 0:
                    z.mul_(1 - lr * wd)
                
                # LION update on z
                update = exp_avg.mul(beta1).add(grad, alpha=1 - beta1)
                z.add_(update.sign(), alpha=-lr)
                
                # Update momentum
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)
                
                # Schedule-free: interpolate x toward z
                # x = (1 - 1/t) * x + (1/t) * z
                t = self._step
                p.mul_(1 - 1/t).add_(z, alpha=1/t)
        
        return loss
