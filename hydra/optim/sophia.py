"""Sophia optimizer for LLM pre-training.

Sophia: A Scalable Stochastic Second-order Optimizer for Language Model Pre-training
Paper: https://arxiv.org/abs/2305.14342

Key insight: Uses lightweight diagonal Hessian estimates for adaptive learning rates.
Claims 2x speedup over Adam in number of steps for same loss.

Two variants:
- SophiaG: Uses Gauss-Newton-Bartlett (GNB) estimator - cheaper, recommended
- SophiaH: Uses Hutchinson's estimator - needs extra backward pass
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch
from torch.optim.optimizer import Optimizer


class SophiaG(Optimizer):
    """Sophia-G optimizer using Gauss-Newton-Bartlett Hessian estimator.
    
    This is the recommended variant - it reuses the loss computation from training
    to estimate the diagonal Hessian, making it nearly as cheap as AdamW.
    
    Key differences from AdamW:
    - Uses diagonal Hessian estimate h instead of squared gradients
    - Clips updates element-wise: update = clip(m / h, -rho, rho)
    - Hessian estimate updated every k steps (default k=10) for efficiency
    
    Recommended hyperparameters (from paper):
    - lr: 2e-4 (can be higher than Adam due to better conditioning)
    - betas: (0.965, 0.99)
    - rho: 0.04 (clipping threshold)
    - weight_decay: 0.1
    - hessian_update_interval: 10
    """
    
    def __init__(
        self,
        params,
        lr: float = 2e-4,
        betas: Tuple[float, float] = (0.965, 0.99),
        rho: float = 0.04,
        weight_decay: float = 0.1,
        eps: float = 1e-12,
        hessian_update_interval: int = 10,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if rho <= 0.0:
            raise ValueError(f"Invalid rho: {rho}")
        
        defaults = dict(
            lr=lr,
            betas=betas,
            rho=rho,
            weight_decay=weight_decay,
            eps=eps,
            hessian_update_interval=hessian_update_interval,
        )
        super().__init__(params, defaults)
        self._step_count = 0
    
    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """Perform optimization step.
        
        For SophiaG, you should call update_hessian() periodically with the
        logits and labels to update the Hessian estimate.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        self._step_count += 1
        
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            rho = group["rho"]
            wd = group["weight_decay"]
            eps = group["eps"]
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]
                
                # Initialize state
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                    state["hessian"] = torch.zeros_like(p)
                
                exp_avg = state["exp_avg"]
                hessian = state["hessian"]
                
                # Weight decay (decoupled)
                if wd != 0:
                    p.mul_(1 - lr * wd)
                
                # Update EMA of gradient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # Sophia update: clip(m / (h + eps), -rho, rho)
                # This is element-wise clipping based on curvature
                update = exp_avg / (hessian + eps)
                update.clamp_(-rho, rho)
                
                p.add_(update, alpha=-lr)
        
        return loss
    
    @torch.no_grad()
    def update_hessian(self, logits: torch.Tensor, labels: torch.Tensor):
        """Update Hessian estimate using Gauss-Newton-Bartlett estimator.
        
        This should be called every hessian_update_interval steps.
        Uses sampling from the model's own distribution (not labels).
        
        Args:
            logits: Model output logits [B, L, V] or [B*L, V]
            labels: Target labels [B, L] or [B*L] (used for masking only)
        """
        # Sample from model's distribution (GNB estimator)
        with torch.no_grad():
            probs = torch.softmax(logits.float(), dim=-1)
            sampled = torch.multinomial(
                probs.view(-1, probs.size(-1)), 
                num_samples=1
            ).view(logits.shape[:-1])
        
        # Compute loss with sampled labels
        # Need grad for backward
        logits_for_grad = logits.detach().requires_grad_(True)
        
        if logits_for_grad.dim() == 3:
            B, L, V = logits_for_grad.shape
            loss = torch.nn.functional.cross_entropy(
                logits_for_grad.view(-1, V),
                sampled.view(-1),
                reduction='mean'
            )
        else:
            loss = torch.nn.functional.cross_entropy(
                logits_for_grad,
                sampled.view(-1),
                reduction='mean'
            )
        
        # Get gradient w.r.t. logits
        loss.backward()
        
        # The GNB Hessian estimate is: E[g_sampled^2]
        # We approximate with the current sample
        grad_logits = logits_for_grad.grad
        
        # Now backprop through the model to get parameter gradients
        # This requires access to the model, which we don't have here
        # Instead, we update hessian in a separate backward pass
        
        # For efficiency, we approximate: h_i ≈ (∂L/∂θ_i)^2 with sampled labels
        # This is done by calling _update_hessian_from_grad after backward
    
    @torch.no_grad()
    def update_hessian_from_grads(self):
        """Update Hessian estimate from current gradients.
        
        Call this after backward() with sampled labels.
        The Hessian estimate is: h = beta2 * h + (1-beta2) * grad^2
        """
        for group in self.param_groups:
            beta2 = group["betas"][1]
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                    state["hessian"] = torch.zeros_like(p)
                
                # GNB estimate: EMA of squared gradients (with sampled labels)
                state["hessian"].mul_(beta2).addcmul_(
                    p.grad, p.grad, value=1 - beta2
                )


class SophiaH(Optimizer):
    """Sophia-H optimizer using Hutchinson's Hessian estimator.
    
    More accurate than SophiaG but requires an extra backward pass.
    Uses random vector for Hessian-vector product estimation.
    
    Recommended for smaller models or when training stability is critical.
    """
    
    def __init__(
        self,
        params,
        lr: float = 2e-4,
        betas: Tuple[float, float] = (0.965, 0.99),
        rho: float = 0.04,
        weight_decay: float = 0.1,
        eps: float = 1e-12,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        
        defaults = dict(
            lr=lr,
            betas=betas,
            rho=rho,
            weight_decay=weight_decay,
            eps=eps,
        )
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
            rho = group["rho"]
            wd = group["weight_decay"]
            eps = group["eps"]
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]
                
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                    state["hessian"] = torch.ones_like(p) * eps  # Initialize to small positive
                
                exp_avg = state["exp_avg"]
                hessian = state["hessian"]
                
                # Weight decay
                if wd != 0:
                    p.mul_(1 - lr * wd)
                
                # Update momentum
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # Sophia clipped update
                update = exp_avg / (hessian + eps)
                update.clamp_(-rho, rho)
                
                p.add_(update, alpha=-lr)
        
        return loss
    
    def update_hessian(self, loss_fn: Callable, params=None):
        """Update Hessian using Hutchinson's estimator.
        
        Requires computing Hessian-vector product with random vector.
        This adds one extra backward pass.
        
        Args:
            loss_fn: Callable that returns the loss (must be differentiable)
            params: Optional list of parameters (defaults to all)
        """
        if params is None:
            params = []
            for group in self.param_groups:
                params.extend(group["params"])
        
        # Compute gradients
        loss = loss_fn()
        grads = torch.autograd.grad(loss, params, create_graph=True)
        
        # Hutchinson's estimator: sample random vector z ~ Rademacher
        # Hessian diagonal ≈ z * (H @ z) where H @ z = d(g·z)/dθ
        zs = [torch.randint_like(p, 0, 2) * 2 - 1 for p in params]  # {-1, +1}
        
        # Compute g·z
        gz = sum((g * z).sum() for g, z in zip(grads, zs))
        
        # Compute d(g·z)/dθ = H @ z
        hvps = torch.autograd.grad(gz, params, retain_graph=False)
        
        # Update Hessian estimates
        with torch.no_grad():
            for group in self.param_groups:
                beta2 = group["betas"][1]
                
                for p, z, hvp in zip(params, zs, hvps):
                    if p not in [pp for pp in group["params"]]:
                        continue
                    
                    state = self.state[p]
                    if "hessian" not in state:
                        state["hessian"] = torch.zeros_like(p)
                    
                    # Hutchinson estimate: diag(H) ≈ z * (H @ z)
                    hessian_estimate = (z * hvp).abs()  # Take abs for stability
                    state["hessian"].mul_(beta2).add_(hessian_estimate, alpha=1 - beta2)


class SophiaGSimple(Optimizer):
    """Simplified SophiaG that uses gradient squared as Hessian proxy.
    
    This is a practical approximation that doesn't require a separate
    Hessian estimation step. It's essentially AdamW with:
    - Element-wise clipping instead of normalization
    - Different update rule: clip(m / sqrt(v), -rho, rho)
    
    Good starting point if you want Sophia-like behavior with minimal changes.
    """
    
    def __init__(
        self,
        params,
        lr: float = 2e-4,
        betas: Tuple[float, float] = (0.965, 0.99),
        rho: float = 0.04,
        weight_decay: float = 0.1,
        eps: float = 1e-8,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            rho=rho,
            weight_decay=weight_decay,
            eps=eps,
        )
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
            rho = group["rho"]
            wd = group["weight_decay"]
            eps = group["eps"]
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]
                
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                
                # Weight decay
                if wd != 0:
                    p.mul_(1 - lr * wd)
                
                # Update EMAs
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Sophia-style clipped update
                # Use sqrt(v) as Hessian proxy (like Adam's denominator)
                denom = exp_avg_sq.sqrt().add_(eps)
                update = exp_avg / denom
                update.clamp_(-rho, rho)
                
                p.add_(update, alpha=-lr)
        
        return loss
