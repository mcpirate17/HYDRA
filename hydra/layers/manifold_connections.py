"""Manifold-Constrained Hyper Connections.

Implements geometric constraints on residual stream updates to bound
gradient magnitudes and prevent gradient explosions.

Key idea: By projecting updates onto a manifold (sphere or Poincare ball),
we bound the Jacobian spectral norm, preventing cascading gradient explosions.

References:
- Sphere projection: ||grad|| <= ||input|| (unit norm constraint)
- Poincare ball: Hyperbolic geometry for hierarchical representations
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ManifoldConstrainedHyperConnection(nn.Module):
    """Manifold-constrained hyper connection for gradient stabilization.

    Architecture:
    1. Hypernetwork: x -> mixture weights over basis vectors [B, L, n_components]
    2. Basis combination: weighted sum of learnable manifold-constrained basis
    3. Manifold projection: projects combined output onto sphere or Poincare ball
    4. Residual: output = x + alpha * warmup_scale * projected_combination * x_scale

    Manifold Types:
    - sphere: L2 normalize to unit sphere. Bounded gradients: ||Jacobian|| <= 1
    - hyperbolic: Poincare ball projection. Captures hierarchical structure.

    The manifold constraint bounds output magnitude, preventing gradient explosions
    from cascading through the residual stream.

    Args:
        dim: Model dimension
        n_components: Number of learnable basis vectors
        manifold_type: "sphere" or "hyperbolic"
        curvature: Poincare ball curvature (hyperbolic only, default 1.0)
        warmup_steps: Steps before full contribution (0 = no warmup)
        init_std: Standard deviation for weight initialization
    """

    def __init__(
        self,
        dim: int,
        n_components: int = 8,
        manifold_type: str = "sphere",
        curvature: float = 1.0,
        warmup_steps: int = 1000,
        init_std: float = 0.02,
    ):
        super().__init__()
        self.dim = dim
        self.n_components = n_components
        self.manifold_type = manifold_type
        self.curvature = curvature
        self.warmup_steps = warmup_steps

        if manifold_type not in ("sphere", "hyperbolic"):
            raise ValueError(f"manifold_type must be 'sphere' or 'hyperbolic', got {manifold_type}")

        # Hypernetwork: generates per-token mixture weights
        # dim -> dim//2 -> n_components
        hidden_dim = max(dim // 2, n_components * 2)
        self.hyper_net = nn.Sequential(
            nn.Linear(dim, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_components, bias=False),
        )

        # Learnable basis vectors (n_components x dim)
        # These are projected onto the manifold during forward
        self.basis = nn.Parameter(torch.empty(n_components, dim))

        # Learnable residual scale - starts small but non-zero for gradient flow
        # Using 0.01 allows gradients to flow from step 0 while being conservative
        self.residual_alpha = nn.Parameter(torch.tensor(0.01))

        # Step tracking for warmup (tensor buffer to avoid torch.compile recompilation)
        self.register_buffer("_global_step", torch.zeros((), dtype=torch.int64), persistent=False)
        # Start step for relative warmup calculation (set when manifold is added to trained model)
        self.register_buffer("_warmup_start_step", torch.zeros((), dtype=torch.int64), persistent=False)
        # Persistent zero scalar to avoid graph breaks and allocations
        self.register_buffer("_zero_scalar", torch.tensor(0.0), persistent=False)
        
        # Gradient scale for stability - manifold gradients can be large
        self._grad_scale = 0.1  # Dampen manifold gradients
        
        # Track whether grads are currently frozen (for checkpoint resume scenarios)
        self._grads_frozen = False

        self._init_weights(init_std)

    def _init_weights(self, std: float = 0.02) -> None:
        """Initialize weights following HYDRA conventions."""
        # Hypernetwork weights: standard normal init
        for module in self.hyper_net.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)

        # Basis vectors: normal init, then project to manifold
        nn.init.normal_(self.basis, mean=0.0, std=std)
        with torch.no_grad():
            self.basis.data = self._project_to_manifold(self.basis.data)

    def _project_to_manifold(self, x: torch.Tensor) -> torch.Tensor:
        """Project tensor onto the specified manifold.

        Args:
            x: Tensor of shape [..., dim]

        Returns:
            Projected tensor on manifold, same shape
        """
        if self.manifold_type == "sphere":
            # L2 normalize: x / ||x||
            # Clamp norm for numerical stability
            norm = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            return x / norm
        else:
            # Poincare ball: x / (1 + sqrt(1 + c * ||x||^2))
            # This maps R^d -> B^d (open unit ball)
            c = self.curvature
            norm_sq = x.pow(2).sum(dim=-1, keepdim=True).clamp(min=1e-12)
            sqrt_term = torch.sqrt(1.0 + c * norm_sq)
            denom = (1.0 + sqrt_term).clamp(min=1e-8)
            return x / denom

    def _get_warmup_scale(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Compute warmup scale factor [0, 1] for gradual contribution.
        
        Uses cosine schedule for smoother gradient transition.
        Warmup is relative to _warmup_start_step (for resume scenarios).
        """
        if self.warmup_steps <= 0:
            return torch.ones((), device=device, dtype=dtype)

        step_f = self._global_step.to(device=device, dtype=dtype)
        start_f = self._warmup_start_step.to(device=device, dtype=dtype)
        # Relative progress since warmup started
        relative_step = step_f - start_f
        progress = (relative_step / float(self.warmup_steps)).clamp(0.0, 1.0)
        # Cosine schedule: starts slow, accelerates in middle, slows at end
        # This gives smoother gradient transition than linear warmup
        scale = 0.5 * (1.0 - torch.cos(progress * math.pi))
        return scale

    def set_global_step(self, step: int) -> None:
        """Update global step for warmup schedule."""
        self._global_step.fill_(step)
    
    def set_warmup_start_step(self, step: int) -> None:
        """Set the step at which warmup should start.
        
        Call this when adding manifold to a trained checkpoint to ensure
        warmup is relative to when manifold was added, not absolute step 0.
        """
        self._warmup_start_step.fill_(step)
    
    def freeze_grads(self) -> None:
        """Freeze all manifold parameters to prevent gradient updates.
        
        Use this when resuming from a checkpoint that doesn't have manifold params
        to prevent catastrophic forgetting while the model adapts.
        """
        if self._grads_frozen:
            return
        for param in self.parameters():
            param.requires_grad = False
        self._grads_frozen = True
    
    def unfreeze_grads(self) -> None:
        """Unfreeze manifold parameters to resume learning."""
        if not self._grads_frozen:
            return
        for param in self.parameters():
            param.requires_grad = True
        self._grads_frozen = False
    
    @property
    def grads_frozen(self) -> bool:
        """Return whether gradients are frozen."""
        return self._grads_frozen

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with manifold-constrained hyper connection.

        Args:
            x: Input tensor [batch, seq_len, dim]

        Returns:
            Output tensor [batch, seq_len, dim] with manifold-constrained residual
        """
        B, L, D = x.shape
        
        # Compute warmup scale (cosine schedule for smoother gradient transition)
        warmup_scale = self._get_warmup_scale(x.device, x.dtype)
        
        # During warmup, return identity to avoid destabilizing a trained model
        # Manifold parameters will train but contribution is gated
        if warmup_scale < 0.1:
            return x
        
        # After warmup, contribution ramps up from 0.1 to 1.0
        effective_scale = warmup_scale
        
        # Effective alpha with warmup - very conservative scaling
        alpha_raw = self.residual_alpha.to(dtype=x.dtype, device=x.device)
        # Clamp raw alpha to small range
        alpha_raw = alpha_raw.clamp(-0.05, 0.05)
        # Scale by warmup - final contribution is alpha_raw * scale
        # For 500M model (D=1792), we want contribution ~0.1% of input
        alpha = alpha_raw * effective_scale * 0.01
        
        # Skip if contribution would be negligible
        if alpha.abs() < 1e-10:
            return x

        # 1. Generate per-token mixture weights via hypernetwork
        weights = self.hyper_net(x)  # [B, L, n_components]
        weights = F.softmax(weights, dim=-1)  # Normalize to convex combination

        # 2. Project basis onto manifold (differentiable)
        basis_projected = self._project_to_manifold(self.basis)  # [n_components, dim]

        # 3. Compute weighted combination of basis vectors
        # [B, L, n_components] @ [n_components, dim] -> [B, L, dim]
        combined = torch.einsum("blc,cd->bld", weights, basis_projected)

        # 4. Project the combined output onto manifold
        combined_projected = self._project_to_manifold(combined)

        # 5. Scale contribution to be proportional to input magnitude
        x_scale = x.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        contribution = combined_projected * x_scale
        
        # 6. Apply residual with very small alpha
        return x + alpha * contribution

    @torch.compiler.disable
    def get_stats(self) -> dict:
        """Get diagnostic statistics (outside torch.compile graph)."""
        return {
            "residual_alpha": float(self.residual_alpha.item()),
            "warmup_scale": float(
                self._get_warmup_scale(self.residual_alpha.device, self.residual_alpha.dtype).item()
            ),
            "global_step": int(self._global_step.item()),
            "basis_norm_mean": float(self.basis.norm(dim=-1).mean().item()),
            "basis_norm_std": float(self.basis.norm(dim=-1).std().item()),
        }

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, n_components={self.n_components}, "
            f"manifold_type={self.manifold_type}, warmup_steps={self.warmup_steps}"
        )
