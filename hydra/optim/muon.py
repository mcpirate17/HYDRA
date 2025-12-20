from __future__ import annotations

import math
from typing import Callable, Optional

import torch


def _newton_schulz_orthogonalize(
    grad_matrix: torch.Tensor,
    *,
    steps: int = 5,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Approximate $G (G^T G)^{-1/2}$ via a Newton–Schulz quintic iteration.

    This is an algorithmic implementation (no code copied) matching the common
    Muon-style orthogonalization step described in public references.

    Returns a tensor with the same shape/dtype as `grad_matrix`.
    """
    if grad_matrix.ndim != 2:
        raise ValueError(f"Expected 2D tensor, got shape={tuple(grad_matrix.shape)}")

    original_dtype = grad_matrix.dtype

    # Prefer bf16 on CUDA where supported; otherwise float32.
    if grad_matrix.is_cuda and torch.cuda.is_bf16_supported():
        work = grad_matrix.to(torch.bfloat16)
    else:
        work = grad_matrix.to(torch.float32)

    # Normalize spectral norm-ish to keep iteration stable.
    work = work / (work.norm() + eps)

    transposed = False
    if work.shape[0] > work.shape[1]:
        work = work.mT
        transposed = True

    # Quintic iteration coefficients (commonly cited for Muon-like NS updates).
    a, b, c = 3.4445, -4.7750, 2.0315

    for _ in range(int(steps)):
        a_mat = work @ work.mT
        ax = a_mat @ work
        work = a * work + b * ax + c * (a_mat @ ax)

    if transposed:
        work = work.mT

    return work.to(original_dtype)


class Muon2D(torch.optim.Optimizer):
    """Muon-style optimizer for 2D parameters only.

    Update rule (high level): momentum on gradients, then orthogonalize the
    momentum buffer via Newton–Schulz, scale it, and apply weight decay + update.

    Notes:
    - Only parameters with `p.ndim == 2` are updated.
    - Weight decay here is applied as an L2 term in the update (Muon-style).
    """

    def __init__(
        self,
        params,
        *,
        lr: float = 1e-3,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        ns_steps: int = 5,
        eps: float = 1e-7,
        rms_scale: float = 0.2,
    ):
        if lr <= 0:
            raise ValueError("lr must be positive")
        if not (0.0 <= momentum <= 1.0):
            raise ValueError("momentum must be in [0, 1]")
        if weight_decay < 0:
            raise ValueError("weight_decay must be >= 0")
        if ns_steps <= 0:
            raise ValueError("ns_steps must be > 0")
        if eps <= 0:
            raise ValueError("eps must be > 0")
        if rms_scale <= 0:
            raise ValueError("rms_scale must be > 0")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
            eps=eps,
            rms_scale=rms_scale,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr: float = group["lr"]
            momentum: float = group["momentum"]
            weight_decay: float = group["weight_decay"]
            ns_steps: int = group["ns_steps"]
            eps: float = group["eps"]
            rms_scale: float = group["rms_scale"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.ndim != 2:
                    continue

                grad = p.grad
                state = self.state[p]
                if "momentum" not in state:
                    state["momentum"] = torch.zeros_like(p)

                buf = state["momentum"]
                buf.mul_(momentum).add_(grad)

                ortho = _newton_schulz_orthogonalize(buf, steps=ns_steps, eps=eps)

                # RMS matching factor
                n, m = p.shape
                ortho = ortho * math.sqrt(max(n, m) * rms_scale)

                # Muon-style update with L2 term
                p.add_(ortho + weight_decay * p, alpha=-lr)

        return loss


class MuonAdamWHybrid:
    """Hybrid optimizer: Muon2D for 2D params + AdamW/AdamW8bit for the rest.

    This is designed to mirror the *behavior* described in the referenced blog:
    - 2D params (e.g., Linear weights) -> Muon-style
    - non-2D params (e.g., norms/biases) -> AdamW (optionally 8-bit via bitsandbytes)

    For checkpoint compatibility with this repo, this wrapper exposes:
    - `param_groups` (concatenated, as references)
    - `state_dict()` / `load_state_dict()`
    - `step()` / `zero_grad()`

    It is intentionally lightweight and intended for testing.
    """

    def __init__(
        self,
        *,
        muon: torch.optim.Optimizer,
        adamw: torch.optim.Optimizer,
    ):
        self._muon = muon
        self._adamw = adamw

        # Expose param groups as references so LR schedulers that mutate dicts
        # will affect the inner optimizers.
        self.param_groups = list(self._muon.param_groups) + list(self._adamw.param_groups)

        # Expose a "state" mapping for compatibility with spike-handling logic.
        # (AdamW side is the only one with exp_avg/exp_avg_sq.)
        self.state = getattr(self._adamw, "state", {})

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = closure() if closure is not None else None
        self._adamw.step()
        self._muon.step()
        return loss

    def zero_grad(self, set_to_none: bool = True):
        self._adamw.zero_grad(set_to_none=set_to_none)
        self._muon.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {
            "muon": self._muon.state_dict(),
            "adamw": self._adamw.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self._muon.load_state_dict(state_dict["muon"])
        self._adamw.load_state_dict(state_dict["adamw"])

        # Refresh references (some optimizers rebuild param_groups on load)
        self.param_groups = list(self._muon.param_groups) + list(self._adamw.param_groups)
        self.state = getattr(self._adamw, "state", {})

    def add_param_group(self, param_group):
        raise NotImplementedError("Add param groups to the inner optimizers directly.")
