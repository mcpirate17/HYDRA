"""
Mixture of Experts (MoE) - Sparse FFN Routing

Implements sparse routing to multiple FFN experts, enabling higher model
capacity with constant compute cost per token.

Key design decisions for HYDRA integration:
1. Top-1 routing (simplest, most stable) - top-2 as future extension
2. No token dropping (capacity factor = infinity, all tokens processed)
3. Auxiliary load-balancing loss (Switch-style)
4. torch.compile compatible (no graph breaks in forward)
5. Identity-safe initialization for checkpoint cloning

Architecture:
- Each MoE block is an FFN-only block inserted between existing transformer blocks
- Structure: x -> LN -> MoE(select expert) -> scaled residual add -> x
- Additive design: when MoE is OFF, forward is identity + base model unchanged

Usage:
    # Create router
    router = MoERouter(dim=512, num_experts=4)
    
    # Get routing decisions
    indices, weights, aux_loss = router(x)
    
    # Or use config-based initialization
    config = MoEConfig(dim=512, num_experts=4)
    router = MoERouter.from_config(config)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "MoEConfig",
    "MoERouter",
    "MoEDispatcher",
    "get_moe_scaling",
    "compute_moe_placement",
]


# =============================================================================
# Configuration
# =============================================================================

@dataclass(frozen=True, slots=True)
class MoEConfig:
    """Immutable configuration for Mixture of Experts.
    
    Attributes:
        dim: Model hidden dimension
        num_experts: Total number of expert FFN networks
        top_k: Number of experts to route each token to (1 = top-1 routing)
        aux_loss_weight: Weight for load-balancing auxiliary loss
        router_jitter: Noise added to router logits during training
        capacity_factor: Multiplier for expert capacity (>= 1.0, inf = no dropping)
        forced_routing_steps: Steps to use position-based forced routing for diversification
    """
    dim: int
    num_experts: int = 4
    top_k: int = 1  # top-1 routing initially
    aux_loss_weight: float = 0.01
    router_jitter: float = 0.0
    capacity_factor: float = float("inf")  # No dropping by default
    forced_routing_steps: int = 0  # 0 = disabled, >0 = enable forced routing schedule
    teacher_until_step: int = 0  # 0 = disabled, >0 = apply teacher loss until this global step
    
    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.dim <= 0:
            raise ValueError(f"dim must be positive, got {self.dim}")
        if self.num_experts < 2:
            raise ValueError(f"num_experts must be >= 2, got {self.num_experts}")
        if self.top_k < 1 or self.top_k > self.num_experts:
            raise ValueError(f"top_k must be in [1, num_experts], got {self.top_k}")
        if self.aux_loss_weight < 0:
            raise ValueError(f"aux_loss_weight must be >= 0, got {self.aux_loss_weight}")
        if self.capacity_factor < 1.0:
            raise ValueError(f"capacity_factor must be >= 1.0, got {self.capacity_factor}")


# =============================================================================
# MoE Scaling Rules
# =============================================================================

def get_moe_scaling(model_size: str) -> Dict[str, Any]:
    """Get MoE scaling parameters based on model size.
    
    Returns dict with:
        num_experts: Number of expert networks
        num_moe_layers: Number of MoE layers to insert
        aux_weight: Recommended auxiliary loss weight
        
    Args:
        model_size: One of "100M", "250M", "500M", "750M", "1B"
    
    Scaling rationale:
    - Smaller models: fewer experts to avoid underfitting
    - Larger models: more experts for capacity, more MoE layers
    - Keep active params (1 expert) similar across sizes
    """
    scaling_rules = {
        # size: (num_experts, num_moe_layers, aux_weight)
        "debug": (2, 1, 0.01),
        "50M": (2, 1, 0.01),
        "100M": (2, 2, 0.01),
        "250M": (3, 2, 0.01),
        "300M": (3, 3, 0.01),
        "500M": (4, 4, 0.01),
        "750M": (4, 4, 0.01),
        "1B": (6, 6, 0.01),  # 6 experts, 6 MoE layers
        "1.5B": (8, 6, 0.01),
    }
    
    num_experts, num_moe_layers, aux_weight = scaling_rules.get(
        model_size, (4, 4, 0.01)  # Default for unknown sizes
    )
    
    return {
        "num_experts": num_experts,
        "num_moe_layers": num_moe_layers,
        "aux_weight": aux_weight,
    }


def compute_moe_placement(n_blocks: int, n_moe_layers: int) -> Tuple[int, ...]:
    """Compute deterministic MoE layer placement indices.
    
    Places MoE layers at evenly-spaced intervals through the model,
    avoiding the first and last blocks for stability.
    
    Args:
        n_blocks: Total number of transformer blocks in the model
        n_moe_layers: Number of MoE layers to insert
        
    Returns:
        Tuple of block indices after which to insert MoE layers
        
    Example:
        n_blocks=8, n_moe_layers=2 -> (2, 5)  # After blocks 2 and 5
    """
    if n_moe_layers <= 0:
        return ()
    
    if n_moe_layers >= n_blocks - 1:
        # More MoE layers requested than available positions
        # Skip first block (idx 0) and last block (idx n_blocks-1)
        return tuple(range(1, n_blocks - 1))
    
    # Evenly space MoE layers, avoiding first and last blocks
    # Available positions: blocks 1 through n_blocks-2
    available_start = 1
    available_end = n_blocks - 2
    n_available = available_end - available_start + 1
    
    if n_available <= 0:
        return ()
    
    if n_moe_layers >= n_available:
        return tuple(range(available_start, available_end + 1))
    
    # Compute evenly-spaced positions
    step = n_available / (n_moe_layers + 1)
    positions = []
    for i in range(1, n_moe_layers + 1):
        pos = available_start + int(i * step) - 1
        pos = max(available_start, min(available_end, pos))
        if pos not in positions:
            positions.append(pos)
    
    # Ensure we have exactly n_moe_layers positions
    while len(positions) < n_moe_layers and len(positions) < n_available:
        for p in range(available_start, available_end + 1):
            if p not in positions:
                positions.append(p)
                break
    
    return tuple(sorted(positions)[:n_moe_layers])


# =============================================================================
# MoE Router
# =============================================================================

class MoERouter(nn.Module):
    """
    Mixture-of-Experts Router with top-k gating.
    
    Produces routing weights and expert indices for each token.
    Computes load-balancing auxiliary loss to encourage uniform expert usage.
    
    torch.compile compatible: no graph breaks, uses tensor ops only.
    """
    
    def __init__(
        self,
        dim: int,
        num_experts: int = 4,
        top_k: int = 1,
        aux_loss_weight: float = 0.01,
        router_jitter: float = 0.0,
        forced_routing_steps: int = 0,
        teacher_until_step: int = 0,
    ):
        """
        Args:
            dim: Model dimension
            num_experts: Number of expert networks
            top_k: Number of experts per token (1 for top-1 routing)
            aux_loss_weight: Weight for load-balancing loss
            router_jitter: Noise added during training for exploration
            forced_routing_steps: Steps to force position-based routing (for diversification)
        """
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.router_jitter = router_jitter
        self.forced_routing_steps = forced_routing_steps
        self.teacher_until_step = teacher_until_step
        
        # Router projection: maps hidden state to expert logits
        self.gate = nn.Linear(dim, num_experts, bias=False)
        
        # Initialize with small std to prevent early collapse
        nn.init.normal_(self.gate.weight, mean=0.0, std=0.02)
        
        # Buffers for loss tracking (avoid graph breaks)
        self.register_buffer("_aux_loss", torch.tensor(0.0))
        self.register_buffer("_expert_counts", torch.zeros(num_experts))
        self.register_buffer("_expert_counts_accum", torch.zeros(num_experts))  # Accumulated over logging interval
        self.register_buffer("_expert_counts_n", torch.zeros((), dtype=torch.int64))  # Number of batches accumulated
        self.register_buffer("_global_step", torch.zeros((), dtype=torch.int64))

        # Optional domain-forced routing + teacher routing loss.
        # These are updated by the Trainer *outside* compiled regions.
        self.register_buffer("_forced_expert_id", torch.full((), -1, dtype=torch.int64))
        self.register_buffer("_teacher_target_expert", torch.full((), -1, dtype=torch.int64))
        self.register_buffer("_teacher_loss", torch.tensor(0.0))
        
        # Pre-generated jitter noise buffer (updated outside CUDA graph capture)
        # Shape will be set on first use; stores noise for [B, L, num_experts]
        self.register_buffer("_jitter_noise", torch.empty(0), persistent=False)
    
    def set_global_step(self, step: int) -> None:
        """Update global step for forced routing schedule."""
        self._global_step.fill_(step)

    def set_forced_expert(self, expert_id: int) -> None:
        """Force all tokens in the current batch to a specific expert (domain forcing)."""
        self._forced_expert_id.fill_(int(expert_id))

    def set_teacher_target(self, expert_id: int) -> None:
        """Set the teacher target expert for the current batch."""
        self._teacher_target_expert.fill_(int(expert_id))

    def get_teacher_loss(self) -> torch.Tensor:
        return self._teacher_loss
    
    @torch.compiler.disable
    def refresh_jitter_noise(self, shape: tuple) -> None:
        """Pre-generate jitter noise outside CUDA graph capture.
        
        Call this before each forward pass when using CUDA graphs.
        The noise is stored in a buffer and reused during forward.
        
        Args:
            shape: Expected logits shape (B, L, num_experts)
        """
        if self.router_jitter > 0 and self.training:
            if self._jitter_noise.shape != shape:
                self._jitter_noise = torch.empty(shape, device=self.gate.weight.device, 
                                                  dtype=self.gate.weight.dtype)
            self._jitter_noise.normal_()
        
    @classmethod
    def from_config(cls, config: MoEConfig) -> "MoERouter":
        """Create MoERouter from config object."""
        return cls(
            dim=config.dim,
            num_experts=config.num_experts,
            top_k=config.top_k,
            aux_loss_weight=config.aux_loss_weight,
            router_jitter=config.router_jitter,
            forced_routing_steps=getattr(config, 'forced_routing_steps', 0),
            teacher_until_step=getattr(config, 'teacher_until_step', 0),
        )
    
    def _apply_jitter(self, logits: torch.Tensor) -> torch.Tensor:
        """Add pre-generated noise to logits during training for exploration.
        
        Uses pre-generated noise from _jitter_noise buffer to avoid
        random number generation during CUDA graph capture.
        """
        if self.training and self.router_jitter > 0:
            # Use pre-generated noise if available and shape matches
            if self._jitter_noise.numel() > 0 and self._jitter_noise.shape == logits.shape:
                return logits + self._jitter_noise * self.router_jitter
            # Fallback: no jitter (safe for CUDA graphs)
            return logits
        return logits
    
    def forward(
        self,
        x: torch.Tensor,
        return_logits: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route tokens to experts.
        
        Args:
            x: Input tensor [batch, seq_len, dim]
            return_logits: If True, also return raw router logits
            
        Returns:
            expert_indices: Selected expert indices [batch, seq_len, top_k]
            expert_weights: Routing weights (softmax) [batch, seq_len, top_k]
            aux_loss: Load-balancing auxiliary loss (scalar tensor)
            logits: (optional) Raw router logits [batch, seq_len, num_experts]
        """
        B, L, D = x.shape
        
        # Compute router logits
        logits = self.gate(x)  # [B, L, num_experts]
        logits = self._apply_jitter(logits)
        
        # Compute routing probabilities (for load balancing loss)
        probs = F.softmax(logits, dim=-1)  # [B, L, num_experts]

        # Normal routing: Top-k selection
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)  # [B, L, top_k]
        top_k_weights = F.softmax(top_k_logits, dim=-1)  # [B, L, top_k]

        # Forced routing schedule is scalar-tensor gated (compile-safe).
        if self.forced_routing_steps > 0:
            forced_until = torch.tensor(self.forced_routing_steps, device=x.device, dtype=self._global_step.dtype)
            forced_active = (self._global_step < forced_until)

            # Domain-forced expert: if set >= 0, route all tokens to that expert.
            forced_id_valid = forced_active & (self._forced_expert_id >= 0)
            forced_expert = self._forced_expert_id.clamp(min=0, max=self.num_experts - 1)  # scalar
            domain_forced = forced_expert.view(1, 1, 1).expand(B, L, 1)

            if self.top_k > 1:
                indices_list = [domain_forced]
                for k in range(1, self.top_k):
                    indices_list.append(((domain_forced + k) % self.num_experts))
                domain_forced_indices = torch.cat(indices_list, dim=-1)
            else:
                domain_forced_indices = domain_forced

            # Position-forced fallback: split seq positions evenly across experts.
            positions = torch.arange(L, device=x.device)
            chunk_size = max(1, L // self.num_experts)
            pos_forced_experts = (positions // chunk_size).clamp(max=self.num_experts - 1)  # [L]
            pos_forced = pos_forced_experts.view(1, L, 1).expand(B, L, 1)
            if self.top_k > 1:
                indices_list = [pos_forced]
                for k in range(1, self.top_k):
                    indices_list.append(((pos_forced + k) % self.num_experts))
                pos_forced_indices = torch.cat(indices_list, dim=-1)
            else:
                pos_forced_indices = pos_forced

            forced_indices = torch.where(forced_id_valid.view(1, 1, 1), domain_forced_indices, pos_forced_indices)
            forced_weights = torch.ones(B, L, self.top_k, device=x.device, dtype=x.dtype) / self.top_k

            top_k_indices = torch.where(forced_active.view(1, 1, 1), forced_indices, top_k_indices)
            top_k_weights = torch.where(forced_active.view(1, 1, 1), forced_weights, top_k_weights)

        # Teacher loss: encourages router logits to match a target expert for this batch.
        # This is returned unscaled; the trainer scales by alpha.
        if self.training:
            target = self._teacher_target_expert
            if self.teacher_until_step and self.teacher_until_step > 0:
                teacher_until = torch.tensor(self.teacher_until_step, device=x.device, dtype=self._global_step.dtype)
                teacher_active = (self._global_step < teacher_until) & (target >= 0)
            else:
                teacher_active = (target >= 0)

            # Compute even if inactive; mask it to keep compile-friendly control flow.
            flat_logits = logits.reshape(-1, self.num_experts)
            flat_targets = target.clamp(min=0, max=self.num_experts - 1).expand(B, L).reshape(-1)
            ce = F.cross_entropy(flat_logits, flat_targets, reduction="mean")
            self._teacher_loss = ce * teacher_active.to(dtype=ce.dtype)
        else:
            self._teacher_loss = torch.zeros((), device=x.device, dtype=x.dtype)
        
        # Compute load-balancing auxiliary loss (Switch Transformer style)
        # Goal: encourage uniform expert usage
        if self.training and self.aux_loss_weight > 0:
            # Expert utilization: fraction of tokens routed to each expert
            # Use hard assignment for counting (top-1 only for simplicity)
            expert_mask = F.one_hot(top_k_indices[..., 0], self.num_experts).float()  # [B, L, E]
            expert_fraction = expert_mask.mean(dim=(0, 1))  # [E]
            
            # Average routing probability per expert
            prob_fraction = probs.mean(dim=(0, 1))  # [E]
            
            # Switch-style aux loss: N * sum(f_i * P_i)
            # Minimized when both f and P are uniform (1/N each)
            aux_loss = self.num_experts * (expert_fraction * prob_fraction).sum()
            self._aux_loss = aux_loss * self.aux_loss_weight
            
            # Track expert counts for diagnostics (last batch)
            self._expert_counts = expert_fraction.detach()
            # Also accumulate for interval reporting
            self._expert_counts_accum = self._expert_counts_accum + expert_fraction.detach()
            self._expert_counts_n = self._expert_counts_n + 1
        else:
            self._aux_loss = torch.zeros((), device=x.device, dtype=x.dtype)
        
        if return_logits:
            return top_k_indices, top_k_weights, self._aux_loss, logits
        return top_k_indices, top_k_weights, self._aux_loss
    
    def get_aux_loss(self) -> torch.Tensor:
        """Get the auxiliary load-balancing loss."""
        return self._aux_loss
    
    def reset_accumulated_counts(self):
        """Reset accumulated expert counts (call after logging)."""
        self._expert_counts_accum.zero_()
        self._expert_counts_n.zero_()
    
    @torch.compiler.disable
    def get_routing_stats(self, use_accumulated: bool = True) -> Dict[str, Any]:
        """Get routing statistics for diagnostics (non-compiled).
        
        Args:
            use_accumulated: If True, use accumulated counts over interval (better signal).
                           If False, use last batch only (legacy behavior).
        """
        import numpy as np
        
        if use_accumulated and int(self._expert_counts_n.item()) > 0:
            # Use accumulated counts averaged over batches
            counts = (self._expert_counts_accum / self._expert_counts_n).detach().cpu().numpy()
            n_batches = int(self._expert_counts_n.item())
        else:
            # Fall back to last batch
            counts = self._expert_counts.detach().cpu().numpy()
            n_batches = 1
        
        # Compute entropy using numpy
        if counts.sum() > 0:
            probs = counts / (counts.sum() + 1e-8)
            entropy = float(-np.sum(probs * np.log(probs + 1e-8)))
        else:
            entropy = 0.0
        return {
            "expert_utilization": counts.tolist(),
            "expert_balance_entropy": entropy,
            "max_expert_fraction": float(counts.max()),
            "min_expert_fraction": float(counts.min()),
            "aux_loss": float(self._aux_loss.item()) if self._aux_loss.numel() > 0 else 0.0,
            "n_batches_accumulated": n_batches,
        }


# =============================================================================
# MoE Dispatcher
# =============================================================================

class MoEDispatcher(nn.Module):
    """
    Dispatches tokens to experts and combines outputs.
    
    Handles the gather/scatter operations to route tokens through
    appropriate experts efficiently. Uses tensor operations only
    for torch.compile compatibility.
    
    For top-1 routing, this is straightforward indexing.
    For top-k, we process each expert selection and weight-combine.
    """
    
    def __init__(
        self,
        num_experts: int,
        top_k: int = 1,
        capacity_factor: float = float("inf"),
    ):
        """
        Args:
            num_experts: Number of expert networks
            top_k: Number of experts per token
            capacity_factor: Max tokens per expert as multiple of (total_tokens / num_experts)
                            Set to inf for no dropping (default).
        """
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        
        # Track tokens dropped for diagnostics
        self.register_buffer("_tokens_dropped", torch.tensor(0))
        self.register_buffer("_tokens_total", torch.tensor(0))
    
    def forward(
        self,
        x: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor,
        experts: nn.ModuleList,
    ) -> torch.Tensor:
        """
        Dispatch tokens to experts and combine outputs.
        
        Args:
            x: Input tensor [batch, seq_len, dim]
            expert_indices: Expert assignments [batch, seq_len, top_k]
            expert_weights: Routing weights [batch, seq_len, top_k]
            experts: List of expert modules
            
        Returns:
            Combined expert outputs [batch, seq_len, dim]
        """
        B, L, D = x.shape
        device = x.device
        dtype = x.dtype
        
        # For compile compatibility, we use a loop over experts
        # but with tensor masking (no Python-level token loops)
        
        # Initialize output
        output = torch.zeros(B, L, D, device=device, dtype=dtype)
        
        # Track total tokens processed
        self._tokens_total = torch.tensor(B * L * self.top_k, device=device)
        self._tokens_dropped = torch.tensor(0, device=device)
        
        # Process each expert
        for expert_idx in range(self.num_experts):
            expert = experts[expert_idx]
            
            # Find tokens assigned to this expert (across all top-k slots)
            # expert_indices: [B, L, top_k]
            # mask: [B, L, top_k] - True where this expert is selected
            mask = expert_indices == expert_idx  # [B, L, top_k]
            
            if not mask.any():
                continue
            
            # Sum of weights for this expert across all slots where it's selected
            # This handles the case where same expert is selected in multiple slots
            weights_for_expert = expert_weights * mask.float()  # [B, L, top_k]
            token_weights = weights_for_expert.sum(dim=-1)  # [B, L]
            
            # Mask of tokens that have non-zero weight for this expert
            token_mask = token_weights > 0  # [B, L]
            
            if not token_mask.any():
                continue
            
            # Capacity checking (if not infinite)
            if self.capacity_factor < float("inf"):
                expected_tokens = (B * L * self.top_k) / self.num_experts
                capacity = int(self.capacity_factor * expected_tokens)
                n_selected = token_mask.sum().item()
                if n_selected > capacity:
                    # Drop excess tokens (keep first `capacity` in flat order)
                    flat_mask = token_mask.view(-1)
                    selected_positions = flat_mask.nonzero(as_tuple=True)[0]
                    drop_positions = selected_positions[capacity:]
                    flat_mask[drop_positions] = False
                    token_mask = flat_mask.view(B, L)
                    self._tokens_dropped = self._tokens_dropped + (n_selected - capacity)
            
            # Gather tokens for this expert
            # We process all tokens but mask the output
            expert_input = x  # [B, L, D]
            expert_output = expert(expert_input)  # [B, L, D]
            
            # Weighted addition to output (only for selected tokens)
            output = output + expert_output * token_weights.unsqueeze(-1)
        
        return output
    
    @torch.compiler.disable  
    def get_dispatch_stats(self) -> Dict[str, Any]:
        """Get dispatch statistics for diagnostics."""
        total = self._tokens_total.item() if self._tokens_total.numel() > 0 else 1
        dropped = self._tokens_dropped.item() if self._tokens_dropped.numel() > 0 else 0
        return {
            "tokens_total": total,
            "tokens_dropped": dropped,
            "drop_rate": dropped / max(1, total),
        }
