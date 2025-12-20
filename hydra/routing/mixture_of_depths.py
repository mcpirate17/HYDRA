"""
Mixture of Depths (MoD) - Dynamic Layer Skipping

Paper: "Mixture-of-Depths: Dynamically allocating compute in transformer-based
language models" (Raposo et al., Google 2024)

Key Insight: Not all tokens need to pass through all layers. MoD learns to
route "easy" tokens around layers, saving compute while maintaining quality.

How it works:
1. A lightweight router scores each token at each layer
2. Top-k tokens (by score) go through the full layer
3. Other tokens skip via residual connection (identity)
4. Capacity ratio controls what fraction of tokens are processed

Benefits:
- 50%+ FLOPs reduction with minimal quality loss
- Learns which tokens need deep processing
- Easy to integrate into existing transformers
- Works with any attention/MLP combination

Usage:
    # Wrap any transformer block with MoD
    mod_block = MixtureOfDepthsBlock(
        block=TransformerBlock(dim=512, ...),
        dim=512,
        capacity_ratio=0.5,  # Process 50% of tokens
    )

    # Or use the router standalone
    router = MoDRouter(dim=512, capacity_ratio=0.5)
    mask, indices = router(x)  # Get which tokens to process
    
    # Or use config-based initialization
    config = MoDConfig(dim=512, capacity_ratio=0.5)
    router = MoDRouter.from_config(config)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

__all__ = [
    "MoDConfig",
    "MoDRouter",
    "MixtureOfDepthsBlock",
    "MoDMLPBlock",
    "MoDAttentionMLPBlock",
    "MoDConditionalBlock",
]


# =============================================================================
# Configuration
# =============================================================================

@dataclass(frozen=True, slots=True)
class MoDConfig:
    """Immutable configuration for Mixture of Depths.
    
    Attributes:
        dim: Model hidden dimension
        capacity_ratio: Fraction of tokens to process (0.0-1.0)
        aux_loss_weight: Weight for load balancing auxiliary loss
        jitter_noise: Noise added during training for exploration
        max_seq_len: Maximum sequence length (for static k computation)
        warmup_steps: Steps before switching to hard routing
    """
    dim: int
    capacity_ratio: float = 0.5
    aux_loss_weight: float = 0.01
    jitter_noise: float = 0.0
    max_seq_len: int = 2048
    warmup_steps: int = 100
    
    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.dim <= 0:
            raise ValueError(f"dim must be positive, got {self.dim}")
        if not 0.0 < self.capacity_ratio <= 1.0:
            raise ValueError(f"capacity_ratio must be in (0, 1], got {self.capacity_ratio}")
        if self.aux_loss_weight < 0:
            raise ValueError(f"aux_loss_weight must be >= 0, got {self.aux_loss_weight}")


class MoDRouter(nn.Module):
    """
    Mixture-of-Depths Router.

    Scores each token and selects top-k for processing.
    Uses straight-through estimator for gradient flow.
    
        NOTE: k is computed from the *actual* sequence length L so that capacity
        behaves as expected during stepped-sequence training (e.g., 512/1024).
        For fixed L (typical training), torch.compile will still specialize once
        per sequence length.
    """

    def __init__(
        self,
        dim: int,
        capacity_ratio: float = 0.5,
        aux_loss_weight: float = 0.01,
        jitter_noise: float = 0.0,
        max_seq_len: int = 2048,
    ):
        """
        Args:
            dim: Model dimension
            capacity_ratio: Fraction of tokens to process (0.0-1.0)
            aux_loss_weight: Weight for load balancing loss
            jitter_noise: Noise for exploration during training
            max_seq_len: Maximum sequence length (used to compute static k)
        """
        super().__init__()
        self.dim = dim
        self.capacity_ratio = capacity_ratio
        self.aux_loss_weight = aux_loss_weight
        self.jitter_noise = jitter_noise
        # Keep max_seq_len as a config field for backward compatibility / diagnostics.
        self.max_seq_len = max_seq_len
        # Historical: used to compute a static k. We keep the attribute for compatibility,
        # but routing now uses k derived from the runtime sequence length.
        self.static_k = max(1, int(max_seq_len * capacity_ratio))

        # Simple linear router (cheap)
        self.router = nn.Linear(dim, 1, bias=True)

        # Initialize so sigmoid(scores) starts near the target capacity.
        # This makes early diagnostics stable and reduces router collapse.
        with torch.no_grad():
            nn.init.zeros_(self.router.weight)
            cap = float(min(max(self.capacity_ratio, 1e-4), 1.0 - 1e-4))
            self.router.bias.fill_(math.log(cap / (1.0 - cap)))

        # Track load for auxiliary loss
        self.register_buffer("_aux_loss", torch.tensor(0.0))
    
    @classmethod
    def from_config(cls, config: MoDConfig) -> "MoDRouter":
        """Create MoDRouter from config object.
        
        Args:
            config: MoDConfig with router parameters
        
        Returns:
            Initialized MoDRouter
        """
        return cls(
            dim=config.dim,
            capacity_ratio=config.capacity_ratio,
            aux_loss_weight=config.aux_loss_weight,
            jitter_noise=config.jitter_noise,
            max_seq_len=config.max_seq_len,
        )

    def forward(
        self,
        x: torch.Tensor,
        return_scores: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Route tokens for Mixture-of-Depths.

        Args:
            x: Input tensor [batch, seq_len, dim]
            return_scores: Whether to return router scores

        Returns:
            mask: Binary mask [batch, seq_len] - 1 for processed tokens
            indices: Indices of selected tokens [batch, k]
            scores: Router scores if return_scores=True
        """
        B, L, D = x.shape
        # Compute k from the runtime sequence length so capacity_ratio is respected
        # for stepped-sequence training (e.g., L=512/1024).
        cap = float(min(max(self.capacity_ratio, 1e-4), 1.0))
        k = int(L * cap)
        k = max(1, min(L, k))

        # Get router scores
        scores = self.router(x).squeeze(-1)  # [B, L]

        # Add noise during training for exploration
        if self.training and self.jitter_noise > 0:
            noise = torch.randn_like(scores) * self.jitter_noise
            scores = scores + noise

        # Select top-k tokens
        top_scores, top_indices = torch.topk(scores, k, dim=-1)  # [B, k]

        # Create binary mask
        mask = torch.zeros(B, L, device=x.device, dtype=x.dtype)
        mask.scatter_(1, top_indices, 1.0)

        # Compute auxiliary loss for load balancing
        if self.training and self.aux_loss_weight > 0:
            # Encourage uniform selection across positions
            probs = torch.sigmoid(scores.clamp(-10.0, 10.0))
            mean_prob = probs.mean()
            target_prob = self.capacity_ratio
            self._aux_loss = self.aux_loss_weight * (mean_prob - target_prob).pow(2)

        if return_scores:
            return mask, top_indices, scores
        return mask, top_indices, None

    def get_aux_loss(self) -> torch.Tensor:
        """Get the auxiliary load-balancing loss."""
        return self._aux_loss


class MixtureOfDepthsBlock(nn.Module):
    """
    Wraps a transformer block with Mixture-of-Depths routing.

    Only selected tokens pass through the wrapped block;
    others use identity (residual) skip.
    """

    def __init__(
        self,
        block: nn.Module,
        dim: int,
        capacity_ratio: float = 0.5,
        aux_loss_weight: float = 0.01,
    ):
        """
        Args:
            block: The transformer block to wrap (attention + MLP)
            dim: Model dimension
            capacity_ratio: Fraction of tokens to process
            aux_loss_weight: Weight for load balancing loss
        """
        super().__init__()
        self.block = block
        self.router = MoDRouter(
            dim=dim,
            capacity_ratio=capacity_ratio,
            aux_loss_weight=aux_loss_weight,
        )

    def forward(
        self,
        x: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward with Mixture-of-Depths routing.

        Args:
            x: Input tensor [batch, seq_len, dim]
            **kwargs: Additional arguments for the wrapped block

        Returns:
            Output tensor [batch, seq_len, dim]
        """
        B, L, D = x.shape

        # Get routing decision
        mask, indices, _ = self.router(x)  # mask: [B, L], indices: [B, k]

        k = indices.shape[1]

        if k == L:
            # All tokens selected - just run the block normally
            return self.block(x, **kwargs)

        if k == 0:
            # No tokens selected - pure skip
            return x

        # Gather selected tokens
        # indices: [B, k] -> [B, k, D]
        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, D)
        x_selected = torch.gather(x, 1, indices_expanded)  # [B, k, D]

        # Process selected tokens through the block
        # Note: The block should handle variable sequence lengths
        out_selected = self.block(x_selected, **kwargs)  # [B, k, D]

        # Scatter back to full sequence (with identity for skipped)
        # Ensure output dtype matches out_selected (for mixed precision)
        output = x.clone().to(out_selected.dtype)  # Start with identity (residual)
        output.scatter_(1, indices_expanded, out_selected)

        return output

    def get_aux_loss(self) -> torch.Tensor:
        """Get the auxiliary load-balancing loss."""
        return self.router.get_aux_loss()


class MoDTransformerLayer(nn.Module):
    """
    Complete transformer layer with built-in MoD routing.

    Applies MoD to both attention and MLP independently,
    allowing different tokens to skip each component.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        mlp_ratio: float = 4.0,
        attention_capacity: float = 0.5,
        mlp_capacity: float = 0.5,
        dropout: float = 0.0,
        aux_loss_weight: float = 0.01,
    ):
        """
        Args:
            dim: Model dimension
            n_heads: Number of attention heads
            mlp_ratio: MLP hidden dim = dim * mlp_ratio
            attention_capacity: Fraction of tokens for attention
            mlp_capacity: Fraction of tokens for MLP
            dropout: Dropout rate
            aux_loss_weight: Weight for load balancing loss
        """
        super().__init__()
        self.dim = dim

        # Attention components
        self.attn_router = MoDRouter(dim, attention_capacity, aux_loss_weight)
        self.norm1 = nn.RMSNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, n_heads, dropout=dropout, batch_first=True
        )

        # MLP components
        self.mlp_router = MoDRouter(dim, mlp_capacity, aux_loss_weight)
        self.norm2 = nn.RMSNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward with separate MoD routing for attention and MLP.
        """
        B, L, D = x.shape

        # === Attention with MoD ===
        attn_mask_tokens, attn_indices, _ = self.attn_router(x)
        k_attn = attn_indices.shape[1]

        if k_attn > 0:
            indices_exp = attn_indices.unsqueeze(-1).expand(-1, -1, D)
            x_attn = torch.gather(x, 1, indices_exp)
            x_attn_norm = self.norm1(x_attn)

            # Self-attention on selected tokens
            attn_out, _ = self.attn(x_attn_norm, x_attn_norm, x_attn_norm)

            # Add residual and scatter back
            x_attn = x_attn + attn_out
            x = x.clone()
            x.scatter_(1, indices_exp, x_attn)

        # === MLP with MoD ===
        mlp_mask, mlp_indices, _ = self.mlp_router(x)
        k_mlp = mlp_indices.shape[1]

        if k_mlp > 0:
            indices_exp = mlp_indices.unsqueeze(-1).expand(-1, -1, D)
            x_mlp = torch.gather(x, 1, indices_exp)
            x_mlp_norm = self.norm2(x_mlp)

            mlp_out = self.mlp(x_mlp_norm)

            x_mlp = x_mlp + mlp_out
            x = x.clone()
            x.scatter_(1, indices_exp, x_mlp)

        return x

    def get_aux_loss(self) -> torch.Tensor:
        """Get combined auxiliary loss from both routers."""
        return self.attn_router.get_aux_loss() + self.mlp_router.get_aux_loss()


class MoDAttention(nn.Module):
    """
    Mixture-of-Depths Attention with standard interface for benchmarking.

    Applies MoD routing to a simple attention mechanism.
    Only selected tokens (top-k by router score) go through attention.

    Standard interface: __init__(dim, n_heads, head_dim, n_kv_heads, dropout)
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        head_dim: int = None,
        n_kv_heads: int = None,
        dropout: float = 0.0,
        capacity_ratio: float = 0.5,
        **kwargs,
    ):
        super().__init__()
        head_dim = head_dim or dim // n_heads
        n_kv_heads = n_kv_heads or n_heads

        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.groups = n_heads // n_kv_heads
        self.capacity_ratio = capacity_ratio

        # MoD Router
        self.router = MoDRouter(
            dim=dim,
            capacity_ratio=capacity_ratio,
            aux_loss_weight=0.01,
        )

        # Attention components
        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.out_proj = nn.Linear(n_heads * head_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward with MoD routing.

        Only top-k tokens (by router score) go through attention.
        Others are passed through via identity.
        """
        B, L, D = x.shape

        # Get routing mask
        mask, indices, _ = self.router(x)  # indices: [B, k]
        k = indices.shape[1]

        if k == 0:
            return x  # All tokens skipped

        # Gather selected tokens
        indices_exp = indices.unsqueeze(-1).expand(-1, -1, D)
        x_selected = torch.gather(x, 1, indices_exp)  # [B, k, D]

        # Standard attention on selected tokens
        q = (
            self.q_proj(x_selected)
            .view(B, k, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )
        k_proj = (
            self.k_proj(x_selected)
            .view(B, k, self.n_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(x_selected)
            .view(B, k, self.n_kv_heads, self.head_dim)
            .transpose(1, 2)
        )

        # GQA expansion
        if self.groups > 1:
            k_proj = k_proj.unsqueeze(2).expand(-1, -1, self.groups, -1, -1)
            v = v.unsqueeze(2).expand(-1, -1, self.groups, -1, -1)
            k_proj = k_proj.reshape(B, self.n_heads, k, self.head_dim)
            v = v.reshape(B, self.n_heads, k, self.head_dim)

        # SDPA on selected tokens (causal within selected set)
        out = F.scaled_dot_product_attention(q, k_proj, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, k, -1)
        out = self.out_proj(out)
        out = self.dropout(out)

        # Scatter back to full sequence
        output = x.clone()
        output.scatter_(1, indices_exp, out)

        return output


# =============================================================================
# Utility Functions
# =============================================================================


def compute_mod_efficiency(capacity_ratio: float, n_layers: int) -> dict:
    """
    Compute theoretical efficiency gains from MoD.

    Args:
        capacity_ratio: Fraction of tokens processed per layer
        n_layers: Number of transformer layers

    Returns:
        Dictionary with efficiency metrics
    """
    # FLOPs savings (attention is O(n²), MLP is O(n))
    # With MoD, both scale with capacity_ratio
    attn_savings = 1 - capacity_ratio**2  # Quadratic savings
    mlp_savings = 1 - capacity_ratio  # Linear savings

    # Weighted average (attention ~40%, MLP ~60% of compute typically)
    total_savings = 0.4 * attn_savings + 0.6 * mlp_savings

    return {
        "capacity_ratio": capacity_ratio,
        "attention_flops_saved": f"{attn_savings:.1%}",
        "mlp_flops_saved": f"{mlp_savings:.1%}",
        "total_flops_saved": f"{total_savings:.1%}",
        "effective_depth": n_layers * capacity_ratio,
    }
