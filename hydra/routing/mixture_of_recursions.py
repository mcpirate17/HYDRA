"""
Mixture of Recursions (MoR) - Adaptive Depth Routing

Paper: "Mixture of Recursions: Learning Dynamic Depth for Transformers"
       (arXiv:2507.10524)

Key Insight: Not all tokens need the same amount of processing depth.
MoR learns to route "easy" tokens through fewer MLP recursions while
giving "hard" tokens more compute.

How it works:
1. A router predicts per-token depth (how many MLP passes)
2. Easy tokens exit early (depth 0-1), hard tokens go deeper
3. Straight-through estimator enables gradient flow through discrete depth
4. Loss-driven routing: CE loss backprop teaches router token difficulty

Architecture (Option A - attention-safe):
- Attention runs ONCE on full sequence (dense)
- MLP recursions apply with adaptive halting (sparse, per-token)
- Early exit only skips MLP recursions, not attention

Benefits:
- Adaptive compute allocation per token
- Parameter efficient (MLP weights shared across recursions)
- Can be combined with MoD for additional savings

Usage:
    config = MoRConfig(dim=768, n_recursions=5)
    router = MoRRouter(config)
    executor = MoRExecutor(config)
    
    # Route tokens
    depths, probs, logits = router(hidden_states)
    
    # Execute with routing
    output = executor(hidden_states, depths, mlp_module)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseRouter

if TYPE_CHECKING:
    from .loss_tracker import MovingAverageBaseline

__all__ = [
    "MoRConfig",
    "MoRRouter",
    "MoRExecutor",
    "MoROutput",
]


# =============================================================================
# Configuration
# =============================================================================

@dataclass(frozen=True, slots=True)
class MoRConfig:
    """Immutable configuration for Mixture of Recursions.
    
    Attributes:
        dim: Model hidden dimension
        n_recursions: Maximum number of MLP recursions (effective depth multiplier)
        ponder_loss_weight: Weight for ponder cost in loss (encourages efficiency)
        router_jitter: Noise added to router during training for exploration
        warmup_steps: Steps before ponder loss reaches full strength
        layer_idx: Index of this block in the model (for layer-aware routing)
        total_layers: Total MoR blocks in model (for layer-aware routing)
        dim_ref: Reference dimension for depth scaling (default 768)
        depth_alpha: Power-law exponent for dim-aware scaling (0=disabled)
        depth_scale_max: Maximum depth scaling factor
    """
    dim: int
    n_recursions: int = 5
    ponder_loss_weight: float = 0.01
    router_jitter: float = 0.0
    warmup_steps: int = 2500
    layer_idx: int = 0
    total_layers: int = 1
    dim_ref: int = 768
    depth_alpha: float = 0.0
    depth_scale_max: float = 2.0
    advantage_loss_scale: float = 0.1
    
    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.dim <= 0:
            raise ValueError(f"dim must be positive, got {self.dim}")
        if self.n_recursions < 1:
            raise ValueError(f"n_recursions must be >= 1, got {self.n_recursions}")
        if self.ponder_loss_weight < 0:
            raise ValueError(f"ponder_loss_weight must be >= 0, got {self.ponder_loss_weight}")


@dataclass(slots=True)
class MoROutput:
    """Output from MoR forward pass.
    
    Attributes:
        hidden_states: Output tensor [batch, seq, dim]
        ponder_loss: Differentiable ponder cost for training
        depths: Discrete depth assignments [batch, seq]
        router_probs: Router probabilities [batch, seq]
        router_logits: Raw router logits [batch, seq]
    """
    hidden_states: torch.Tensor
    ponder_loss: torch.Tensor
    depths: torch.Tensor
    router_probs: torch.Tensor
    router_logits: torch.Tensor


# =============================================================================
# Utility Functions
# =============================================================================

def dim_to_depth_scale(
    dim: int,
    dim_ref: int = 768,
    depth_alpha: float = 0.0,
    scale_max: float = 2.0,
) -> float:
    """Compute depth scale factor based on model dimension.
    
    Larger models naturally benefit from deeper MoR recursions. This function
    computes a scale factor that increases the expected average depth for
    larger dimensions.
    
    Args:
        dim: Current model dimension
        dim_ref: Reference dimension where scale=1.0 (default: 768)
        depth_alpha: Scaling exponent (0.0 = no scaling for backward compat)
        scale_max: Maximum scale factor
    
    Returns:
        Scale factor in [1.0, scale_max]. When depth_alpha=0, always returns 1.0.
    """
    if depth_alpha <= 0.0 or dim <= dim_ref:
        return 1.0
    raw_scale = (dim / dim_ref) ** depth_alpha
    return min(raw_scale, scale_max)


def compute_layer_target_prob(
    layer_idx: int,
    total_layers: int,
    depth_scale: float = 1.0,
) -> float:
    """Compute target depth probability for a layer.
    
    Early layers target shallower depths, late layers target deeper.
    This provides a natural curriculum: early = refine, late = complex reasoning.
    
    Args:
        layer_idx: Index of this layer (0-based)
        total_layers: Total number of MoR layers
        depth_scale: Dimension-aware scaling factor (1.0 = no scaling)
    
    Returns:
        Target probability in (0, 1) for router bias initialization.
    """
    # Layer ratio: 0.0 for first layer, 1.0 for last
    layer_ratio = layer_idx / max(1, total_layers - 1)
    
    # Base target: 0.2 for early layers, 0.5 for late layers
    base_target_prob = 0.2 + 0.3 * layer_ratio
    
    # Apply dimension-aware scaling (shifts toward deeper)
    if depth_scale > 1.0:
        target_prob = 1.0 - (1.0 - base_target_prob) / depth_scale
    else:
        target_prob = base_target_prob
    
    return float(target_prob)


# =============================================================================
# Router
# =============================================================================

class MoRRouter(BaseRouter):
    """Mixture-of-Recursions Router.
    
    Predicts per-token recursion depth using a learned linear projection.
    Uses sigmoid activation to map to [0, 1], then scales to depth range.
    
    The router learns from cross-entropy loss backpropagation through the
    straight-through estimator. No forced distribution - the model learns
    which tokens need more compute from the task signal.
    """
    
    __slots__ = (
        'config', '_depth_scale', '_target_prob', '_warmup_steps',
        '_global_step', '_cached_global_step', '_zero_scalar',
    )
    
    def __init__(self, config: MoRConfig) -> None:
        super().__init__(config.dim, config.router_jitter)
        self.config = config
        
        # Compute depth scale once (compile-safe constant)
        self._depth_scale = dim_to_depth_scale(
            config.dim, config.dim_ref, config.depth_alpha, config.depth_scale_max
        )
        
        # Compute layer-aware target probability
        self._target_prob = compute_layer_target_prob(
            config.layer_idx, config.total_layers, self._depth_scale
        )
        
        # Initialize bias to target probability (this sets weights to zero)
        self._init_bias(self._target_prob)
        
        # Re-initialize weights to normal as per original implementation
        nn.init.normal_(self.router.weight, mean=0.0, std=0.02)
        
        # Warmup tracking
        self._warmup_steps = config.warmup_steps
        self.register_buffer(
            "_global_step", 
            torch.zeros((), dtype=torch.int64), 
            persistent=False
        )
        # OPTIMIZATION: Persistent zero scalar to avoid graph breaks and allocations
        self.register_buffer("_zero_scalar", torch.tensor(0.0), persistent=False)
        self._cached_global_step: int = 0
    
    def set_global_step(self, step: int) -> None:
        """Update global step for warmup scheduling."""
        self._global_step.fill_(step)
        self._cached_global_step = step
    
    @property
    def warmup_scale(self) -> float:
        """Current warmup scale factor [0, 1]."""
        return min(1.0, self._cached_global_step / max(1, self._warmup_steps))
    
    def forward(
        self,
        x: torch.Tensor,
        return_continuous: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Route tokens to recursion depths.
        
        Args:
            x: Input hidden states [batch, seq, dim]
            return_continuous: If True, also return continuous depth for STE
        
        Returns:
            depths: Discrete depth assignments [batch, seq] in [0, n_recursions-1]
            probs: Router probabilities [batch, seq] in [0, 1]
            logits: Raw router logits [batch, seq]
        """
        # Get router scores
        logits = self.forward_logits(x)  # [B, L]
        
        # Soft clamp with tanh (preserves gradients at boundaries)
        logits = torch.tanh(logits / 2.0) * 3.0
        
        # Convert to probabilities
        probs = torch.sigmoid(logits)  # [B, L] in [0, 1]
        
        # Map to continuous depth
        n_rec = self.config.n_recursions
        depth_continuous = probs * (n_rec - 1)  # [B, L] in [0, n_rec-1]
        
        # Discretize with rounding
        depths = torch.round(depth_continuous).long()
        depths = torch.clamp(depths, 0, n_rec - 1)
        
        return depths, probs, logits
    
    def compute_ponder_loss(
        self,
        depths: torch.Tensor,
        probs: torch.Tensor,
        logits: torch.Tensor,
        token_losses: Optional[torch.Tensor] = None,
        baseline: Optional["MovingAverageBaseline"] = None,
    ) -> torch.Tensor:
        """Compute ponder loss for router training.
        
        If token_losses and baseline are provided, uses loss-driven routing.
        Otherwise, uses light regularization only.
        
        Args:
            depths: Discrete depth assignments [batch, seq]
            probs: Router probabilities [batch, seq]
            logits: Router logits [batch, seq]
            token_losses: Per-token CE losses [batch, seq] (optional)
            baseline: Moving average baseline for advantage computation (optional)
        
        Returns:
            Scalar ponder loss tensor with gradients.
        """
        device = probs.device
        dtype = probs.dtype
        n_rec = self.config.n_recursions
        
        # Base ponder cost: weak efficiency pressure
        depth_continuous = probs * (n_rec - 1)
        avg_depth = depth_continuous.mean()
        depth_divisor = float(max(1, n_rec - 1))
        ponder_cost = avg_depth / depth_divisor
        
        # Collapse prevention: want router to differentiate tokens
        router_variance = probs.var()
        collapse_loss = torch.exp(-router_variance * 10.0)
        
        # Logit variance: prevent clustering near 0
        logit_var_loss = F.relu(0.5 - logits.var())
        
        # Loss-driven component (your approach)
        if token_losses is not None and baseline is not None:
            advantage = baseline.compute_advantage(token_losses)
            # Scale router gradients by advantage
            # Positive advantage = harder than average -> reward depth
            # Negative advantage = easier than average -> penalize depth
            advantage_loss = -(advantage * depth_continuous).mean() * 0.1
        else:
            advantage_loss = self._zero_scalar
        
        # Combine losses (light touch - CE loss dominates)
        ponder_loss = (
            0.0005 * ponder_cost +      # Efficiency pressure (very light)
            0.01 * collapse_loss +       # Prevent all-same-depth
            0.1 * logit_var_loss +       # Prevent logit collapse
            advantage_loss               # Loss-driven (if provided)
        )
        
        # Apply warmup
        ponder_loss = self.warmup_scale * ponder_loss
        
        # Safety clamp
        return torch.clamp(ponder_loss, max=5.0)


# =============================================================================
# Executor
# =============================================================================

class MoRExecutor(nn.Module):
    """Mixture-of-Recursions Executor.
    
    Executes MLP recursions based on per-token depth assignments.
    Uses vectorized computation (all tokens through all depths, masked output)
    which is faster on GPU than sparse token packing for moderate sequences.
    
    The executor handles:
    - Adding recursion-specific embeddings/biases
    - Masked accumulation of outputs
    - STE gradient path for discrete depth decisions
    """
    
    def __init__(self, config: MoRConfig) -> None:
        super().__init__()
        self.config = config
        
        # Recursion-specific parameters (cheap differentiation between passes)
        self.recursion_bias = nn.Parameter(
            torch.zeros(config.n_recursions, 1, 1, config.dim) * 0.02
        )
        self.recursion_embed = nn.Embedding(config.n_recursions, config.dim)
        nn.init.normal_(self.recursion_embed.weight, std=0.02)
        
        # Pre-compute indices for efficiency
        self.register_buffer(
            "_recursion_indices",
            torch.arange(config.n_recursions, dtype=torch.long),
            persistent=False,
        )
        
        # Diagnostics (not persistent)
        self._recursion_tokens_processed: list = []
    
    def forward(
        self,
        x: torch.Tensor,
        depths: torch.Tensor,
        probs: torch.Tensor,
        mlp: nn.Module,
        norm: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """Execute MLP recursions with per-token depth routing.
        
        Args:
            x: Input hidden states [batch, seq, dim] (post-attention)
            depths: Discrete depth per token [batch, seq] in [0, n_rec-1]
            probs: Router probabilities [batch, seq] for STE gradient
            mlp: MLP module to apply recursively
            norm: Optional pre-MLP normalization
        
        Returns:
            Output tensor [batch, seq, dim] with depth-routed MLP outputs.
        """
        B, L, D = x.shape
        device = x.device
        dtype = x.dtype
        n_rec = self.config.n_recursions
        
        # Pre-compute masks for all depths
        depth_indices = torch.arange(n_rec, device=device)
        
        # exit_at_depth[i, b, l] = 1 if token (b,l) exits at depth i
        exit_at_depth = (depths.unsqueeze(0) == depth_indices.view(-1, 1, 1))  # [R, B, L]
        
        # active_at_depth[i, b, l] = 1 if token (b,l) is active at depth i
        active_at_depth = (depths.unsqueeze(0) >= depth_indices.view(-1, 1, 1))  # [R, B, L]
        
        # STE weights: Gaussian centered on continuous depth
        depth_continuous = probs * (n_rec - 1)
        depth_indices_f = depth_indices.to(dtype)
        ste_weights = torch.exp(
            -((depth_continuous.unsqueeze(0) - depth_indices_f.view(-1, 1, 1)) ** 2)
        )  # [R, B, L]
        
        # Accumulation
        output = torch.zeros_like(x)
        current = x
        
        # Track for diagnostics
        if self.training:
            self._recursion_tokens_processed = []
        
        for i in range(n_rec):
            # Masks for this depth
            active_mask = active_at_depth[i].unsqueeze(-1).to(dtype)  # [B, L, 1]
            exit_mask = exit_at_depth[i].unsqueeze(-1).to(dtype)  # [B, L, 1]
            ste_weight_i = ste_weights[i].unsqueeze(-1)  # [B, L, 1]
            
            # Diagnostics
            if self.training:
                self._recursion_tokens_processed.append(active_at_depth[i].sum().detach())
            
            # Add recursion embeddings
            rec_bias = self.recursion_bias[i].squeeze()  # [D]
            rec_embed = self.recursion_embed(self._recursion_indices[i:i+1]).squeeze()  # [D]
            h_with_rec = current + rec_bias + rec_embed
            
            # Apply normalization if provided
            if norm is not None:
                h_with_rec = norm(h_with_rec)
            
            # MLP pass
            mlp_delta = mlp(h_with_rec)
            
            # Update current state (only active tokens evolve)
            current = current + mlp_delta * active_mask
            
            # Accumulate output for exiting tokens with STE
            # STE: forward uses 1.0, backward flows through ste_weight
            ste_grad = ste_weight_i - ste_weight_i.detach()
            weighted_exit = (1.0 + ste_grad) * current * exit_mask
            output = output + weighted_exit
        
        return output
    
    def get_recursion_stats(self) -> dict:
        """Get statistics about recursion token distribution."""
        if not self._recursion_tokens_processed:
            return {}
        
        total = sum(t.item() for t in self._recursion_tokens_processed)
        return {
            f"depth_{i}_tokens": t.item() 
            for i, t in enumerate(self._recursion_tokens_processed)
        }
