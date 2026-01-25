"""
MoE FFN Block - Additive Sparse Expert Layer

Implements MoE as an FFN-only residual block that can be inserted between
existing transformer blocks without modifying attention or other components.

Design:
- Structure: x -> LN -> MoE(FFN experts) -> scaled residual add -> x
- Identity-safe: with proper initialization, output ≈ input for checkpoint cloning
- No token dropping by default (capacity_factor = inf)
- Compatible with torch.compile

Initialization strategy for checkpoint cloning:
1. All expert FFNs start with identity-like weights (small scale)
2. Residual scale alpha starts at 0.0 (pure bypass)
3. Gradually ramp alpha during training to blend in MoE

Usage:
    # Create MoE FFN block
    moe_block = MoEFFNBlock(
        dim=512,
        hidden_dim=1024,
        num_experts=4,
    )
    
    # Forward (can be inserted between any blocks)
    y = moe_block(x)
    
    # Get aux loss for training
    aux_loss = moe_block.get_aux_loss()
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from hydra.layers import RMSNorm, SwiGLUMLP
from .mixture_of_experts import MoEConfig, MoERouter, MoEDispatcher

__all__ = [
    "MoEFFNBlock",
    "MoEExpertMLP",
]


class MoEExpertMLP(nn.Module):
    """
    Single expert MLP for MoE.
    
    Uses the same SwiGLU architecture as the base model for consistency.
    Supports identity-preserving initialization for checkpoint cloning.
    """
    
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        identity_init: bool = False,
        init_scale: float = 1.0,
    ):
        """
        Args:
            dim: Input/output dimension
            hidden_dim: Hidden dimension (typically 2.67x dim)
            identity_init: If True, initialize weights for near-identity output
            init_scale: Scale factor for weight initialization
        """
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        
        # Use fused gate/up projection (matches base model SwiGLUMLPFused)
        self.gate_up = nn.Linear(dim, 2 * hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, dim, bias=False)
        
        self._init_weights(identity_init, init_scale)
    
    def _init_weights(self, identity_init: bool, init_scale: float):
        """Initialize weights."""
        if identity_init:
            # For identity-like initialization:
            # - gate_up: small random so SiLU(gate) * up is small
            # - down: small random
            # Combined effect: output is near-zero, so residual dominates
            nn.init.normal_(self.gate_up.weight, mean=0.0, std=0.001 * init_scale)
            nn.init.normal_(self.down.weight, mean=0.0, std=0.001 * init_scale)
        else:
            # Standard initialization
            nn.init.normal_(self.gate_up.weight, mean=0.0, std=0.02 * init_scale)
            nn.init.normal_(self.down.weight, mean=0.0, std=0.02 * init_scale)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through expert MLP."""
        # Fused gate/up projection
        gate_up = self.gate_up(x)
        gate, up = gate_up.chunk(2, dim=-1)
        
        # SwiGLU activation
        hidden = F.silu(gate) * up
        
        # Down projection
        return self.down(hidden)


class MoEFFNBlock(nn.Module):
    """
    Mixture-of-Experts FFN block for insertion between transformer blocks.
    
    Structure: x -> LN -> MoE(select expert) -> scaled residual -> x
    
    Key features:
    - Additive: when alpha=0, output = input (identity)
    - Identity-safe init: gradual ramp-in for checkpoint cloning
    - No attention: pure FFN routing
    - torch.compile compatible
    """
    
    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        num_experts: int = 4,
        top_k: int = 1,
        aux_loss_weight: float = 0.01,
        router_jitter: float = 0.0,
        capacity_factor: float = float("inf"),
        residual_scale: float = 1.0,
        identity_init: bool = True,
        expert_diversity_noise: float = 0.0,
        warmup_steps: int = 1000,
        norm_eps: float = 1e-6,
        mlp_ratio: float = 2.67,
        forced_routing_steps: int = 0,
        teacher_until_step: int = 0,
    ):
        """
        Args:
            dim: Model dimension
            hidden_dim: Expert hidden dimension (default: dim * mlp_ratio, aligned to 256)
            num_experts: Number of expert FFN networks
            top_k: Number of experts per token (1 for top-1)
            aux_loss_weight: Weight for load-balancing loss
            router_jitter: Noise for router exploration
            capacity_factor: Expert capacity multiplier (inf = no dropping)
            residual_scale: Initial scale for residual connection (0 = pure bypass)
            identity_init: Use identity-preserving initialization
            expert_diversity_noise: Additive noise std to break expert symmetry (try 0.01-0.05)
            warmup_steps: Steps before full MoE contribution
            norm_eps: LayerNorm epsilon
            mlp_ratio: Hidden dim multiplier if hidden_dim not specified
            forced_routing_steps: Steps to use position-based forced routing for diversification
        """
        super().__init__()
        
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.warmup_steps = warmup_steps
        self.identity_init = identity_init
        self.forced_routing_steps = forced_routing_steps
        
        # Compute hidden dim (aligned to 256 for efficiency)
        if hidden_dim is None:
            hidden_dim = int(dim * mlp_ratio)
            hidden_dim = ((hidden_dim + 255) // 256) * 256
        self.hidden_dim = hidden_dim
        
        # Pre-MoE normalization
        self.norm = RMSNorm(dim, eps=norm_eps)
        
        # Router
        self.router = MoERouter(
            dim=dim,
            num_experts=num_experts,
            top_k=top_k,
            aux_loss_weight=aux_loss_weight,
            router_jitter=router_jitter,
            forced_routing_steps=forced_routing_steps,
            teacher_until_step=teacher_until_step,
        )
        
        # Expert FFNs
        self.experts = nn.ModuleList([
            MoEExpertMLP(
                dim=dim,
                hidden_dim=hidden_dim,
                identity_init=identity_init,
                init_scale=1.0 / math.sqrt(num_experts),  # Scale down for expert averaging
            )
            for _ in range(num_experts)
        ])
        
        # Apply diversity noise to break expert symmetry (critical for specialization)
        if expert_diversity_noise > 0:
            with torch.no_grad():
                for i, expert in enumerate(self.experts):
                    # Use expert index as seed for reproducibility
                    gen = torch.Generator().manual_seed(42 + i * 1337)
                    for param in expert.parameters():
                        noise = torch.randn_like(param, generator=gen) * expert_diversity_noise
                        param.add_(noise)
        
        # Dispatcher
        self.dispatcher = MoEDispatcher(
            num_experts=num_experts,
            top_k=top_k,
            capacity_factor=capacity_factor,
        )
        
        # Learnable residual scale (starts at 0 for identity-preserving init)
        # Use a parameter so it can be learned during training
        initial_alpha = 0.0 if identity_init else residual_scale
        self.residual_alpha = nn.Parameter(torch.tensor(initial_alpha))
        
        # Step tracking for warmup
        self.register_buffer("_global_step", torch.zeros((), dtype=torch.int64), persistent=False)
        self.register_buffer("_zero_scalar", torch.tensor(0.0), persistent=False)
        
        # Diagnostic buffers
        self._aux_loss: torch.Tensor = self._zero_scalar
        self._moe_enabled: bool = not identity_init  # Start disabled if identity init
        
    def set_global_step(self, step: int) -> None:
        """Update global step for warmup schedule."""
        self._global_step.fill_(step)
        # Enable MoE after warmup (ramp handled in forward)
        self._moe_enabled = step >= self.warmup_steps or not self.identity_init
        # Also update router for forced routing schedule
        self.router.set_global_step(step)

    def set_forced_expert(self, expert_id: int) -> None:
        self.router.set_forced_expert(expert_id)

    def set_teacher_target(self, expert_id: int) -> None:
        self.router.set_teacher_target(expert_id)
    
    def _get_warmup_scale_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """Compile-safe warmup scale factor [0, 1] for gradual MoE ramp-in."""
        if not self.identity_init:
            return torch.ones((), device=x.device, dtype=x.dtype)
        denom = max(1, int(self.warmup_steps))
        step_f = self._global_step.to(device=x.device, dtype=x.dtype)
        scale = (step_f / float(denom)).clamp(0.0, 1.0)
        return scale

    def get_warmup_scale(self) -> float:
        """Non-compiled warmup scale for diagnostics."""
        if not self.identity_init:
            return 1.0
        step = int(self._global_step.detach().cpu().item())
        if step >= self.warmup_steps:
            return 1.0
        return step / max(1, self.warmup_steps)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MoE FFN block.
        
        Args:
            x: Input tensor [batch, seq_len, dim]
            
        Returns:
            Output tensor [batch, seq_len, dim]
        """
        # Pre-norm
        normed = self.norm(x)
        
        # Router: get expert assignments and weights
        expert_indices, expert_weights, aux_loss = self.router(normed)
        self._aux_loss = aux_loss
        
        # Dispatch to experts
        moe_output = self.dispatcher(normed, expert_indices, expert_weights, self.experts)
        
        # Residual connection with learnable scale
        # During warmup, scale down MoE contribution
        warmup_scale = self._get_warmup_scale_tensor(x)
        alpha = self.residual_alpha.to(dtype=x.dtype, device=x.device) * warmup_scale
        
        # Clamp alpha for stability (allow slight negative for fine-tuning flexibility)
        alpha = alpha.clamp(-0.5, 2.0)
        
        return x + alpha * moe_output
    
    def forward_with_losses(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass returning auxiliary losses."""
        output = self.forward(x)
        losses = {
            "moe_aux_loss": self._aux_loss,
            "moe_teacher_loss": self.router.get_teacher_loss(),
        }
        return output, losses
    
    def get_aux_loss(self) -> torch.Tensor:
        """Get MoE auxiliary load-balancing loss."""
        return self._aux_loss
    
    @torch.compiler.disable
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics for diagnostics."""
        router_stats = self.router.get_routing_stats()
        dispatch_stats = self.dispatcher.get_dispatch_stats()
        
        return {
            **router_stats,
            **dispatch_stats,
            "residual_alpha": float(self.residual_alpha.item()),
            "warmup_scale": self.get_warmup_scale(),
            "moe_enabled": self._moe_enabled,
            "global_step": int(self._global_step.item()),
            "warmup_steps": self.warmup_steps,
        }
    
    @torch.compiler.disable
    def get_expert_stats(self) -> Dict[str, Any]:
        """Get per-expert weight statistics for debugging."""
        stats = {}
        for i, expert in enumerate(self.experts):
            gate_up_norm = expert.gate_up.weight.norm().item()
            down_norm = expert.down.weight.norm().item()
            stats[f"expert_{i}_gate_up_norm"] = gate_up_norm
            stats[f"expert_{i}_down_norm"] = down_norm
        return stats


class MoELayerWrapper(nn.Module):
    """
    Wrapper that adds MoE FFN after an existing transformer block.
    
    This allows inserting MoE at specific positions without modifying
    the base model structure.
    
    Structure: block_output -> MoE FFN Block -> final_output
    """
    
    def __init__(
        self,
        block: nn.Module,
        moe_block: MoEFFNBlock,
    ):
        """
        Args:
            block: Existing transformer block
            moe_block: MoE FFN block to add after
        """
        super().__init__()
        self.block = block
        self.moe_block = moe_block
    
    def set_global_step(self, step: int) -> None:
        """Update global step for both block and MoE."""
        if hasattr(self.block, "set_global_step"):
            self.block.set_global_step(step)
        self.moe_block.set_global_step(step)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward through block then MoE."""
        # Run original block
        h = self.block(x, mask=mask) if mask is not None else self.block(x)
        # Run MoE FFN
        return self.moe_block(h)
    
    def forward_with_losses(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward with loss collection."""
        # Run original block with losses if supported
        if hasattr(self.block, "forward_with_losses"):
            h, block_losses = self.block.forward_with_losses(x, mask=mask)
        else:
            h = self.block(x, mask=mask) if mask is not None else self.block(x)
            block_losses = {}
        
        # Run MoE FFN with losses
        output, moe_losses = self.moe_block.forward_with_losses(h)
        
        # Combine losses
        all_losses = {**block_losses, **moe_losses}
        return output, all_losses
    
    def get_aux_loss(self) -> torch.Tensor:
        """Get combined auxiliary loss."""
        block_aux = self.block.get_aux_loss() if hasattr(self.block, "get_aux_loss") else 0.0
        moe_aux = self.moe_block.get_aux_loss()
        return block_aux + moe_aux
    
    @torch.compiler.disable
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing stats from both block and MoE."""
        stats = {}
        if hasattr(self.block, "get_routing_stats"):
            block_stats = self.block.get_routing_stats()
            stats["block"] = block_stats
        stats["moe"] = self.moe_block.get_routing_stats()
        return stats
