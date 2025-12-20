"""
Shared Operations for Routing Modules

Optimized utility functions used by MoD and MoR.
Designed for:
- torch.compile compatibility
- In-place operations where safe
- Future Cython/C replacement hook

These are pure functions operating on tensors - no module state.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = [
    "soft_clamp_logits",
    "ste_round",
    "compute_exit_masks",
    "gather_by_mask",
    "scatter_by_indices",
    "compute_ste_weights",
]


# =============================================================================
# Logit Processing
# =============================================================================

def soft_clamp_logits(
    logits: torch.Tensor,
    scale: float = 2.0,
    clamp_range: float = 3.0,
) -> torch.Tensor:
    """Soft clamp logits using tanh scaling.
    
    Preserves gradients at boundaries unlike hard clamp.
    tanh(x/scale) * range maps (-inf, inf) -> (-range, range)
    
    Args:
        logits: Input logits tensor
        scale: Division factor before tanh (larger = softer saturation)
        clamp_range: Output range after tanh multiplication
    
    Returns:
        Soft-clamped logits in (-clamp_range, clamp_range)
    """
    return torch.tanh(logits / scale) * clamp_range


def temperature_scaled_sigmoid(
    logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Sigmoid with temperature scaling.
    
    Lower temperature = sharper decisions (closer to hard routing)
    Higher temperature = softer decisions (more exploration)
    
    Args:
        logits: Input logits tensor
        temperature: Temperature for scaling (default 1.0)
    
    Returns:
        Sigmoid probabilities
    """
    return torch.sigmoid(logits / max(temperature, 1e-6))


# =============================================================================
# Straight-Through Estimators
# =============================================================================

class STERound(torch.autograd.Function):
    """Straight-through estimator for rounding.
    
    Forward: round(x) - discrete values
    Backward: gradient of x - continuous gradient flow
    """
    
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return torch.round(x)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output  # Pass gradient through unchanged


def ste_round(x: torch.Tensor) -> torch.Tensor:
    """Round with straight-through gradient estimator.
    
    Args:
        x: Continuous tensor to round
    
    Returns:
        Rounded tensor with STE gradient
    """
    return STERound.apply(x)


class STETopK(torch.autograd.Function):
    """Straight-through estimator for top-k selection.
    
    Forward: binary mask from top-k
    Backward: gradient scaled by original scores
    """
    
    @staticmethod
    def forward(ctx, scores: torch.Tensor, k: int) -> torch.Tensor:
        B, L = scores.shape
        _, indices = torch.topk(scores, k, dim=-1)
        mask = torch.zeros_like(scores)
        mask.scatter_(1, indices, 1.0)
        ctx.save_for_backward(scores, mask)
        return mask
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        scores, mask = ctx.saved_tensors
        # Gradient flows through selected positions, scaled by score
        probs = torch.sigmoid(scores)
        return grad_output * probs * mask, None


def ste_topk(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Top-k selection with straight-through gradient.
    
    Args:
        scores: Score tensor [batch, seq]
        k: Number of elements to select
    
    Returns:
        Binary mask [batch, seq] with k ones per batch
    """
    return STETopK.apply(scores, k)


# =============================================================================
# Mask Computation
# =============================================================================

def compute_exit_masks(
    depths: torch.Tensor,
    n_depths: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-compute exit and active masks for all depths.
    
    Args:
        depths: Per-token depth assignments [batch, seq]
        n_depths: Maximum number of depth levels
        device: Target device
    
    Returns:
        exit_at_depth: [n_depths, batch, seq] - True where token exits
        active_at_depth: [n_depths, batch, seq] - True where token is active
    """
    depth_indices = torch.arange(n_depths, device=device)
    
    # exit_at_depth[i] = (depths == i)
    exit_at_depth = (depths.unsqueeze(0) == depth_indices.view(-1, 1, 1))
    
    # active_at_depth[i] = (depths >= i)
    active_at_depth = (depths.unsqueeze(0) >= depth_indices.view(-1, 1, 1))
    
    return exit_at_depth, active_at_depth


def compute_ste_weights(
    probs: torch.Tensor,
    n_depths: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Compute STE weights for smooth gradient flow.
    
    Gaussian-shaped weights centered on continuous depth target.
    Tokens get strongest gradient signal at their target depth.
    
    Args:
        probs: Router probabilities [batch, seq] in [0, 1]
        n_depths: Number of depth levels
        device: Target device
        dtype: Target dtype
    
    Returns:
        STE weights [n_depths, batch, seq]
    """
    depth_indices = torch.arange(n_depths, device=device, dtype=dtype)
    depth_continuous = probs * (n_depths - 1)
    
    # Gaussian weights: exp(-((continuous - index)^2))
    return torch.exp(
        -((depth_continuous.unsqueeze(0) - depth_indices.view(-1, 1, 1)) ** 2)
    )


# =============================================================================
# Gather/Scatter Operations
# =============================================================================

def gather_by_mask(
    x: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather tokens where mask is True.
    
    Args:
        x: Input tensor [batch, seq, dim]
        mask: Boolean mask [batch, seq]
    
    Returns:
        selected: Gathered tokens [total_selected, dim]
        indices: Original indices for scatter back [total_selected, 2]
    """
    B, L, D = x.shape
    
    # Get indices where mask is True
    batch_idx, seq_idx = torch.where(mask)
    indices = torch.stack([batch_idx, seq_idx], dim=-1)
    
    # Gather selected tokens
    selected = x[batch_idx, seq_idx]  # [N, D]
    
    return selected, indices


def scatter_by_indices(
    values: torch.Tensor,
    indices: torch.Tensor,
    output_shape: tuple[int, int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Scatter values back to original positions.
    
    Args:
        values: Values to scatter [N, dim]
        indices: Original positions [N, 2] (batch_idx, seq_idx)
        output_shape: Target shape (batch, seq, dim)
        device: Target device
        dtype: Target dtype
    
    Returns:
        Output tensor [batch, seq, dim] with values at indices
    """
    output = torch.zeros(output_shape, device=device, dtype=dtype)
    batch_idx = indices[:, 0]
    seq_idx = indices[:, 1]
    output[batch_idx, seq_idx] = values
    return output


def gather_by_indices(
    x: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """Gather tokens by sorted indices (for MoD-style routing).
    
    Args:
        x: Input tensor [batch, seq, dim]
        indices: Token indices to gather [batch, k]
    
    Returns:
        Gathered tokens [batch, k, dim]
    """
    D = x.shape[-1]
    indices_exp = indices.unsqueeze(-1).expand(-1, -1, D)
    return torch.gather(x, 1, indices_exp)


def scatter_add_by_indices(
    output: torch.Tensor,
    values: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """Scatter-add values to output at indices.
    
    Args:
        output: Output tensor [batch, seq, dim] (modified in-place)
        values: Values to scatter [batch, k, dim]
        indices: Target indices [batch, k]
    
    Returns:
        Modified output tensor
    """
    D = output.shape[-1]
    indices_exp = indices.unsqueeze(-1).expand(-1, -1, D)
    output.scatter_add_(1, indices_exp, values)
    return output


# =============================================================================
# Capacity Utilities
# =============================================================================

def compute_capacity_k(
    seq_len: int,
    capacity_ratio: float,
) -> int:
    """Compute k (number of tokens to process) from capacity ratio.
    
    Args:
        seq_len: Sequence length
        capacity_ratio: Fraction of tokens to process (0, 1]
    
    Returns:
        k: Number of tokens to process, at least 1
    """
    k = int(seq_len * capacity_ratio)
    return max(1, min(seq_len, k))


def build_capacity_schedule(
    n_depths: int,
    decay_rate: float = 0.15,
    min_capacity: float = 0.25,
) -> list[float]:
    """Build hierarchical capacity schedule for MoR.
    
    Deeper recursions process fewer tokens (hierarchical filtering).
    
    Args:
        n_depths: Number of recursion levels
        decay_rate: How much capacity decreases per level
        min_capacity: Minimum capacity at deepest level
    
    Returns:
        List of capacity ratios for each depth level
    """
    return [max(min_capacity, 1.0 - decay_rate * i) for i in range(n_depths)]
