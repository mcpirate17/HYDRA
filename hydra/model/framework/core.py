"""Compatibility module for the (former) CCGQA model package.

Historically, `hydra.model.ccgqa` was a single large module. During the
package split, and later the rename to `hydra.model.framework`, some code may
still import the old module paths.

Canonical implementations now live in:
- `hydra.model.framework.blocks`
- `hydra.model.framework.model`
- `hydra.model.framework.factory`
"""

from __future__ import annotations

from .blocks import (
    CCGQABlock,
    CCGQABlockWithMoDMLP,
    CCGQAMoDBlockWrapper,
    CCGQAMoRBlock,
    MoDMLPWrapper,
)
from .factory import create_ccgqa_model, create_ccgqa_mod_mor_model
from .model import CCGQAModel, CCGQAMoDMoRModel

__all__ = [
    "CCGQABlock",
    "CCGQABlockWithMoDMLP",
    "CCGQAModel",
    "CCGQAMoRBlock",
    "CCGQAMoDMoRModel",
    "CCGQAMoDBlockWrapper",
    "MoDMLPWrapper",
    "create_ccgqa_model",
    "create_ccgqa_mod_mor_model",
]
"""
Compressed Convolutional Grouped Query Attention (CCGQA)

From: "Compressed Convolutional Attention: Efficient Attention in a Compressed Latent Space"
Paper: https://arxiv.org/abs/2510.04476

Key innovations:
1. Down-project Q, K, V into compressed latent space (C factor compression)
2. Sequence + channel convolutions on Q and K for enhanced expressivity
3. QK-mean coupling: share information between Q and K pre/post convolution
4. Value-shift: half heads see previous token's value (temporal inductive bias)
5. QK L2 normalization with learnable temperature
6. GQA-style head sharing in the compressed space
7. Attention performed entirely in compressed space (faster prefill + training)

Benefits vs MLA:
- No up-projection matrices needed (2x fewer params)
- RoPE works seamlessly in compressed space
- Reduces attention FLOPs by compression factor C
- 1.7x faster prefill, 1.3x faster backward on H100

Benefits vs GQA:
- Better quality at same KV-cache compression
- Can achieve 8x compression with no quality loss
"""

import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from typing import Optional, Tuple, Union
from torch.utils.checkpoint import checkpoint as gradient_checkpoint

_log = logging.getLogger(__name__)

# =============================================================================
# VALIDATION: Routing modules imported from hydra.routing
# MoD: gather/scatter compute-skipping for token selection
# MoR: Mixture-of-Recursions for adaptive depth per token
# =============================================================================
from hydra.routing.mixture_of_depths import MoDRouter
from hydra.routing.mixture_of_recursions import (
    MoRConfig,
    MoRRouter,
    MoRExecutor,
    dim_to_depth_scale,
)
from hydra.routing.loss_tracker import MovingAverageBaseline

# =============================================================================
# Optional attention variants (hybrid backends).
# =============================================================================
from hydra.attention.backends.ccgqa.attention import CCGQAAttention
from hydra.attention.factory import build_hybrid_attention_module

# Import shared layers from hydra.layers (canonical implementations)
from hydra.layers import RMSNorm, SwiGLUMLPFused as SwiGLUMLP

# Validate MoDRouter import at module load time
assert hasattr(MoDRouter, 'forward'), "MoDRouter must have forward method"
assert hasattr(MoDRouter, 'get_aux_loss'), "MoDRouter must have get_aux_loss method"


class CCGQABlock(nn.Module):
    """
    Transformer block with CCGQA attention.

    Pre-norm architecture with SwiGLU MLP.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int = 16,
        n_kv_heads: int = 4,
        compression_factor: int = 4,
        mlp_ratio: float = 2.67,  # SwiGLU optimal ratio
        max_seq_len: int = 8192,
        norm_eps: float = 1e-6,
        **attention_kwargs,
    ):
        super().__init__()

        # CCGQA attention
        self.attention = CCGQAAttention(
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            compression_factor=compression_factor,
            max_seq_len=max_seq_len,
            **attention_kwargs,
        )

        # RMSNorm
        self.norm1 = RMSNorm(dim, eps=norm_eps)
        self.norm2 = RMSNorm(dim, eps=norm_eps)

        # SwiGLU MLP
        hidden_dim = int(dim * mlp_ratio)
        # Make hidden_dim multiple of 256 for efficiency
        hidden_dim = ((hidden_dim + 255) // 256) * 256

        self.mlp = SwiGLUMLP(dim, hidden_dim)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Pre-norm attention with residual
        h = x + self.attention(self.norm1(x), mask=mask)
        # Pre-norm MLP with residual
        out = h + self.mlp(self.norm2(h))
        return out


# NOTE: RMSNorm and SwiGLUMLP are now imported from hydra.layers
# See hydra/layers/common.py for canonical implementations


class MoDMLPWrapper(nn.Module):
    """Mixture-of-Depths wrapper for MLP sublayer ONLY.
    
    KEY DESIGN PRINCIPLE:
    - MoD applies ONLY to MLP, NOT to attention
    - Attention must ALWAYS be dense (all tokens attend to all tokens)
    - MLP is position-independent, so we can skip tokens for MLP only
    
    TWO-PHASE ROUTING CURRICULUM (controlled by global step):
    
    Phase 1 (step < warmup_steps): SOFT ROUTING
    - All tokens computed through MLP
    - Output weighted by router probabilities (soft gate)
    - Purpose: stable router learning before hard decisions
    - NO compute savings
    
    Phase 2 (step >= warmup_steps): HARD ROUTING  
    - Top-k tokens selected by router
    - Gather only selected tokens, process through MLP
    - Scatter results back, skipped tokens get zero MLP delta
    - REAL compute savings (only k tokens pass through MLP)
    
    Args:
        mlp: The MLP module to wrap (e.g., SwiGLUMLP)
        dim: Model dimension
        capacity_ratio: Fraction of tokens to process (0.5 = 50%)
        aux_loss_weight: Weight for load balancing auxiliary loss
        warmup_steps: Steps before switching to hard routing
    """

    def __init__(
        self,
        mlp: nn.Module,
        dim: int,
        capacity_ratio: float = 0.5,
        aux_loss_weight: float = 0.01,
        warmup_steps: int = 100,
        force_enable_step: Optional[int] = None,
        max_seq_len: int = 2048,
        enable_loss_threshold: Optional[float] = None,
        loss_aware_weight: float = 0.0,
    ):
        super().__init__()
        self.mlp = mlp
        self.capacity_ratio = capacity_ratio
        self.aux_loss_weight = aux_loss_weight
        self.warmup_steps = warmup_steps
        self.force_enable_step = (
            int(force_enable_step)
            if (force_enable_step is not None and int(force_enable_step) > 0)
            else None
        )
        self.max_seq_len = max_seq_len
        self.loss_aware_weight = float(loss_aware_weight)

        self.enable_loss_threshold = (
            float(enable_loss_threshold)
            if (enable_loss_threshold is not None and enable_loss_threshold > 0)
            else None
        )
        self._loss_unlocked: bool = self.enable_loss_threshold is None
        
        # Use tensor buffer for global_step to avoid torch.compile recompilation
        # (Dynamo treats Python int attributes as static, causing recompile storm)
        self.register_buffer("_global_step", torch.zeros((), dtype=torch.int64), persistent=False)
        # OPTIMIZATION: Persistent zero scalar to avoid graph breaks and allocations
        self.register_buffer("_zero_scalar", torch.tensor(0.0), persistent=False)
        
        # MoDRouter for token selection (gather/scatter pattern)
        # max_seq_len enables static k computation for torch.compile compatibility
        self.mod_router = MoDRouter(
            dim=dim,
            capacity_ratio=capacity_ratio,
            aux_loss_weight=aux_loss_weight,
            max_seq_len=max_seq_len,
        )
        assert isinstance(self.mod_router, MoDRouter), \
            f"mod_router must be MoDRouter, got {type(self.mod_router)}"
        
        # Diagnostics
        self._aux_loss: torch.Tensor = self._zero_scalar
        self._last_probs_mean_t: torch.Tensor = self._zero_scalar
        self._last_probs_std_t: torch.Tensor = self._zero_scalar
        self._routing_mode: str = "soft"
        self._tokens_processed: int = 0
        self._tokens_total: int = 0
        self._last_scores: Optional[torch.Tensor] = None

    def set_global_step(self, step: int):
        """Set global training step for curriculum scheduling."""
        self._global_step.fill_(step)
        # Curriculum:
        # - Before warmup_steps: MoD DISABLED (dense MLP; no gating / no aux loss)
        # - After warmup_steps: MoD ENABLED (hard top-k routing)
        # - If force_enable_step is set: MoD MUST be enabled no later than that step
        # Cache enable decision to avoid .item() in forward (prevents graph break)
        if self.force_enable_step is not None and step >= self.force_enable_step:
            self._loss_unlocked = True
        self._mod_enabled = (step >= self.warmup_steps) and self._loss_unlocked

    @torch.compiler.disable
    def update_loss_ema(self, loss_ema: float) -> None:
        """Unlock MoD once the EMA CE (cross-entropy) is below threshold.

        This is called from the trainer (outside torch.compile graphs).
        """
        if self._loss_unlocked:
            return
        if self.enable_loss_threshold is None:
            self._loss_unlocked = True
            return
        try:
            loss_val = float(loss_ema)
        except Exception:
            return
        if loss_val < self.enable_loss_threshold:
            self._loss_unlocked = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with MoD routing on MLP only.
        
        NOTE: x is the NORMALIZED hidden state (post-norm2).
        The caller (CCGQABlockWithMoDMLP) handles the residual connection.
        
        Returns the MLP output delta (to be added to residual by caller).
        """
        if not getattr(self, "_mod_enabled", False):
            # Keep semantics dense, but still compute router aux loss and cache
            # router scores so the router doesn't cold-start when MoD flips on.
            self._routing_mode = "disabled"
            B, L, _ = x.shape
            self._tokens_total = L
            self._tokens_processed = L
            scores = self.mod_router.forward_logits(x)  # [B, L]
            self._last_scores = scores
            probs = torch.sigmoid(scores.clamp(-10.0, 10.0))
            self._last_probs_mean_t = probs.mean().detach()
            self._last_probs_std_t = probs.std().detach()

            if self.training and self.aux_loss_weight > 0:
                mean_prob = probs.mean()
                target_prob = self.capacity_ratio
                capacity_loss = (mean_prob - target_prob).pow(2)

                prob_variance = probs.var()
                expected_var = target_prob * (1 - target_prob) * 0.5
                collapse_loss = torch.exp(-prob_variance / max(expected_var, 0.01) * 5.0)
                self._aux_loss = self.aux_loss_weight * (capacity_loss + 0.5 * collapse_loss)
            else:
                self._aux_loss = self._zero_scalar
            return self.mlp(x)

        if not self.training:
            # Inference: hard routing for maximum speed
            return self._forward_hard(x)

        # Training: hard routing with STE for gradient flow
        return self._forward_hard_with_ste(x)

    def _forward_hard(self, x: torch.Tensor) -> torch.Tensor:
        """Hard routing: gather top-k, MLP, scatter back.
        
        NOTE: No data-dependent branches (if k >= L, if k == 0) to ensure
        torch.compile generates a single static graph.
        """
        B, L, D = x.shape

        self._tokens_total = L
        self._routing_mode = "hard"

        mask, indices, scores = self.mod_router(x, return_scores=True)  # indices: [B, k]
        self._last_scores = scores
        k = indices.shape[1]

        # Diagnostics (safe: store detached tensors; no .item() here)
        probs = torch.sigmoid(scores.clamp(-10.0, 10.0))  # [B, L]
        self._last_probs_mean_t = probs.mean().detach()
        self._last_probs_std_t = probs.std().detach()
        self._tokens_processed = k

        if self.aux_loss_weight > 0:
            self._aux_loss = self.mod_router.get_aux_loss()
        
        # Sort indices ascending to maintain position monotonicity
        indices, _ = torch.sort(indices, dim=1)
        
        # GATHER: Extract only top-k tokens
        indices_exp = indices.unsqueeze(-1).expand(-1, -1, D)  # [B, k, D]
        x_selected = torch.gather(x, 1, indices_exp)  # [B, k, D]
        
        # MLP on selected tokens only (real compute savings!)
        mlp_out_selected = self.mlp(x_selected)  # [B, k, D]
        
        # SCATTER: Place MLP outputs back, zeros for skipped tokens
        # Ensure dtype consistency for scatter_ (handles autocast edge cases)
        output = torch.zeros_like(x)
        output.scatter_(1, indices_exp, mlp_out_selected.to(output.dtype))
        
        return output

    def _forward_hard_with_ste(self, x: torch.Tensor) -> torch.Tensor:
        """Hard routing with STE for gradient flow during training.
        
        NOTE: No data-dependent branches (if k >= L, if k == 0) to ensure
        torch.compile generates a single static graph.
        """
        B, L, D = x.shape
        self._tokens_total = L
        self._routing_mode = "hard"
        
        mask, indices, scores = self.mod_router(x, return_scores=True)
        self._last_scores = scores
        k = indices.shape[1]
        
        # Soft probs for STE gradient path
        probs = torch.sigmoid(scores.clamp(-10.0, 10.0))  # [B, L]
        # Store detached tensors for diagnostics (avoids .item() graph break)
        self._last_probs_mean_t = probs.mean().detach()
        self._last_probs_std_t = probs.std().detach()
        self._tokens_processed = k
        
        # Sort indices ascending to maintain position monotonicity
        indices, sort_order = torch.sort(indices, dim=1)
        
        # GATHER
        indices_exp = indices.unsqueeze(-1).expand(-1, -1, D)
        x_selected = torch.gather(x, 1, indices_exp)  # [B, k, D]
        
        # Validate gather
        assert x_selected.shape[1] == k, \
            f"gather failed: expected {k} tokens, got {x_selected.shape[1]}"
        
        # MLP on selected tokens only
        mlp_out_selected = self.mlp(x_selected)  # [B, k, D]
        self._tokens_processed = k
        
        # SCATTER - match output dtype to MLP output for AMP compatibility
        output = torch.zeros(B, L, D, dtype=mlp_out_selected.dtype, device=x.device)
        output.scatter_(1, indices_exp, mlp_out_selected)
        
        # STE gradient path: connect router gradients
        selected_probs = torch.gather(probs, 1, indices)  # [B, k]
        ste_grad_path = (selected_probs.sum() - selected_probs.sum().detach()) * 0.0
        output = output + ste_grad_path
        
        # Store aux_loss
        if self.aux_loss_weight > 0:
            self._aux_loss = self.mod_router.get_aux_loss()
        
        return output

    def _forward_soft(self, x: torch.Tensor) -> torch.Tensor:
        """Soft routing: all tokens through MLP, weighted output."""
        B, L, D = x.shape
        self._tokens_total = L
        self._tokens_processed = L  # All tokens in soft mode
        self._routing_mode = "soft"
        
        _, _, scores = self.mod_router(x, return_scores=True)
        self._last_scores = scores
        probs = torch.sigmoid(scores.clamp(-10.0, 10.0))  # [B, L]
        # Store detached tensors for diagnostics (avoids .item() graph break)
        self._last_probs_mean_t = probs.mean().detach()
        self._last_probs_std_t = probs.std().detach()
        
        # Process ALL tokens (no compute savings during warmup)
        mlp_out = self.mlp(x)
        
        # Soft weighted output: high prob -> MLP output, low prob -> zero
        gate = probs.unsqueeze(-1)  # [B, L, 1]
        output = gate * mlp_out
        
        # Store aux_loss
        if self.aux_loss_weight > 0:
            self._aux_loss = self.mod_router.get_aux_loss()
        
        return output

    def get_aux_loss(self) -> torch.Tensor:
        """Returns MoD auxiliary loss.

        Note: Loss-aware supervision (based on per-token CE loss) is added via
        the model's `compute_advantage_loss()` path.
        """
        return self._aux_loss

    def compute_loss_aware_loss(self, token_losses: torch.Tensor) -> torch.Tensor:
        """Supervise router scores to prioritize hard tokens.

        Uses top-k by per-token CE loss as a teacher (stop-grad) and trains the
        router logits to match that selection.
        """
        if self.loss_aware_weight <= 0:
            return self._zero_scalar
        scores = getattr(self, "_last_scores", None)
        if scores is None:
            return self._zero_scalar

        B, L = token_losses.shape
        k = int(max(1, min(L, int(L * float(self.capacity_ratio)))))

        # Mask out ignored positions (caller uses -inf for ignore_index).
        valid_mask = torch.isfinite(token_losses)
        if not valid_mask.any():
            return self._zero_scalar

        with torch.no_grad():
            _, hard_idx = torch.topk(token_losses, k, dim=1)
            teacher = torch.zeros((B, L), device=token_losses.device, dtype=torch.float32)
            teacher.scatter_(1, hard_idx, 1.0)

            pos_weight_val = float(max(1.0, (L - k) / max(1, k)))
            pos_weight = torch.tensor(pos_weight_val, device=token_losses.device, dtype=torch.float32)

        # Prevent huge early losses from unbounded logits.
        scores_f = scores.float().clamp(-10.0, 10.0)
        per_tok = F.binary_cross_entropy_with_logits(
            scores_f,
            teacher,
            pos_weight=pos_weight,
            reduction="none",
        )
        loss = (per_tok * valid_mask.float()).sum() / valid_mask.float().sum().clamp_min(1.0)
        return loss * self.loss_aware_weight

    @torch.compiler.disable
    def get_routing_stats(self) -> dict:
        """Get MoD routing statistics for debugging (compile-disabled)."""
        tokens_processed = getattr(self, "_tokens_processed", 0)
        tokens_total = getattr(self, "_tokens_total", 1)
        compute_ratio = tokens_processed / max(1, tokens_total)
        
        # Extract .item() from stored tensors (safe here, outside compile)
        mean_t = getattr(self, "_last_probs_mean_t", None)
        std_t = getattr(self, "_last_probs_std_t", None)
        
        return {
            "probs_mean": float(mean_t.item()) if mean_t is not None else 0.0,
            "probs_std": float(std_t.item()) if std_t is not None else 0.0,
            "target_capacity": self.capacity_ratio,
            "tokens_processed": tokens_processed,
            "tokens_total": tokens_total,
            "compute_ratio": compute_ratio,
            "compute_savings_pct": (1.0 - compute_ratio) * 100.0,
            "routing_mode": getattr(self, "_routing_mode", "unknown"),
            "global_step": int(self._global_step),
            "warmup_steps": self.warmup_steps,
            "force_enable_step": self.force_enable_step,
            "enable_loss_threshold": self.enable_loss_threshold,
            "loss_unlocked": bool(getattr(self, "_loss_unlocked", True)),
        }


class CCGQABlockWithMoDMLP(nn.Module):
    """CCGQA Block with MoD applied ONLY to MLP sublayer.
    
    Architecture:
    - Attention: ALWAYS dense (all tokens attend to all tokens)
    - MLP: MoD-routed (only top-k tokens pass through MLP)
    
    This is the correct MoD design because:
    1. Attention NEEDS full context - token i must attend to all j <= i
    2. MLP is position-independent - each token's MLP is independent
    
    Forward pass:
        h = x + attention(norm1(x), mask)      # Dense attention (full seq)
        out = h + mod_mlp(norm2(h))            # MoD-routed MLP (sparse)
    """

    def __init__(
        self,
        dim: int,
        n_heads: int = 16,
        n_kv_heads: int = 4,
        compression_factor: int = 4,
        mlp_ratio: float = 2.67,
        max_seq_len: int = 8192,
        norm_eps: float = 1e-6,
        mod_capacity_ratio: float = 0.5,
        mod_aux_loss_weight: float = 0.01,
        mod_warmup_steps: int = 100,
        mod_loss_aware_weight: float = 0.0,
        **attention_kwargs,
    ):
        super().__init__()

        # CCGQA attention (always dense)
        self.attention = CCGQAAttention(
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            compression_factor=compression_factor,
            max_seq_len=max_seq_len,
            **attention_kwargs,
        )

        # Layer norms
        self.norm1 = RMSNorm(dim, eps=norm_eps)
        self.norm2 = RMSNorm(dim, eps=norm_eps)

        # SwiGLU MLP (wrapped with MoD)
        hidden_dim = int(dim * mlp_ratio)
        hidden_dim = ((hidden_dim + 255) // 256) * 256  # Align for efficiency
        
        base_mlp = SwiGLUMLP(dim, hidden_dim)
        self.mod_mlp = MoDMLPWrapper(
            mlp=base_mlp,
            dim=dim,
            capacity_ratio=mod_capacity_ratio,
            aux_loss_weight=mod_aux_loss_weight,
            warmup_steps=mod_warmup_steps,
            max_seq_len=max_seq_len,
            loss_aware_weight=mod_loss_aware_weight,
        )

    def set_global_step(self, step: int):
        """Propagate global step to MoD router."""
        self.mod_mlp.set_global_step(step)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass: dense attention + MoD-routed MLP."""
        # Pre-norm ATTENTION with residual (ALWAYS dense, full sequence)
        h = x + self.attention(self.norm1(x), mask=mask)
        
        # Pre-norm MLP with residual (MoD-routed, sparse computation)
        out = h + self.mod_mlp(self.norm2(h))
        
        return out

    def forward_with_losses(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """Forward with aux_loss for training."""
        output = self.forward(x, mask=mask)
        
        losses = {}
        if self.training:
            losses["aux_loss"] = self.mod_mlp.get_aux_loss()
        
        return output, losses

    def get_aux_loss(self) -> torch.Tensor:
        """Get aux_loss from MoD router."""
        return self.mod_mlp.get_aux_loss()

    @torch.compiler.disable
    def get_routing_stats(self) -> dict:
        """Get MoD routing stats (compile-disabled)."""
        return self.mod_mlp.get_routing_stats()


class CCGQAModel(nn.Module):
    """
    Full transformer model using CCGQA attention.

    Architecture:
    - Token embedding with optional weight tying
    - N x CCGQABlock layers
    - Final RMSNorm + output projection

    Args:
        vocab_size: Vocabulary size
        dim: Model dimension
        n_layers: Number of transformer blocks
        n_heads: Number of query heads
        n_kv_heads: Number of KV heads (GQA grouping)
        compression_factor: Attention compression factor C
        mlp_ratio: MLP hidden dim ratio
        max_seq_len: Maximum sequence length
        tie_weights: Whether to tie input/output embeddings
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        dim: int = 1344,
        n_layers: int = 24,
        n_heads: int = 21,
        n_kv_heads: int = 3,
        compression_factor: int = 4,
        mlp_ratio: float = 2.67,
        max_seq_len: int = 8192,
        tie_weights: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.dim = dim
        self.vocab_size = vocab_size
        self.n_layers = n_layers

        # Token embedding
        self.tok_emb = nn.Embedding(vocab_size, dim)

        # Transformer blocks
        self.layers = nn.ModuleList(
            [
                CCGQABlock(
                    dim=dim,
                    n_heads=n_heads,
                    n_kv_heads=n_kv_heads,
                    compression_factor=compression_factor,
                    mlp_ratio=mlp_ratio,
                    max_seq_len=max_seq_len,
                )
                for _ in range(n_layers)
            ]
        )

        # Final norm
        self.norm = RMSNorm(dim)

        # Output projection
        self.output = nn.Linear(dim, vocab_size, bias=False)

        # Initialize weights BEFORE tying (so Linear init doesn't overwrite Embedding)
        self._init_weights()

        # Weight tying (AFTER init so embedding std=1.0 is preserved)
        if tie_weights:
            self.output.weight = self.tok_emb.weight

    def _init_weights(self):
        """Initialize weights with scaled init for deep networks."""
        residual_scale = 1.0 / math.sqrt(2 * self.n_layers)

        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                # Scale down residual projections
                if "o_proj" in name or "down" in name:
                    module.weight.data *= residual_scale
            elif isinstance(module, nn.Embedding):
                # GPT-2 style: std = 0.02
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Conv1d):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input token IDs [B, S]

        Returns:
            Logits [B, S, vocab_size]
        """
        # Token embedding with sqrt(dim) scaling (LLaMA style)
        h = self.tok_emb(x) * math.sqrt(self.dim)

        # Transformer blocks
        for layer in self.layers:
            h = layer(h)

        # Output
        h = self.norm(h)
        return self.output(h)


def create_ccgqa_model(spec) -> CCGQAModel:
    """
    Factory function to create CCGQA model from spec.

    Args:
        spec: Model specification with dim, n_layers, n_heads, etc.

    Returns:
        CCGQAModel instance
    """
    return CCGQAModel(
        vocab_size=getattr(spec, "vocab_size", 50257),
        dim=getattr(spec, "dim", 1344),
        n_layers=getattr(spec, "n_layers", 24),
        n_heads=getattr(spec, "n_heads", 21),
        n_kv_heads=getattr(spec, "n_kv_heads", 3),
        compression_factor=getattr(spec, "compression_factor", 4),
        mlp_ratio=getattr(spec, "mlp_ratio", 2.67),
        max_seq_len=getattr(spec, "max_seq_len", 8192),
        tie_weights=getattr(spec, "tie_weights", True),
    )


# =============================================================================
# CCGQA + MoD + MoR: Full Efficiency Stack
# =============================================================================

# NOTE: CCGQAMoDBlock (soft gating fake MoD) was REMOVED.
# Use CCGQAMoDBlockWrapper with MoDRouter for REAL compute-skipping MoD.


class CCGQAMoRBlock(nn.Module):
    """
    CCGQA block with Mixture-of-Recursions (MoR) on MLP only.

    OPTION A DESIGN (attention-safe):
    - Attention runs ONCE on full sequence (dense, all tokens)
    - MLP recursions apply with adaptive halting (sparse, token-level early exit)
    - Early exit only skips additional MLP recursions, not attention
    
    This is correct because:
    1. Attention NEEDS full context - token i must attend to all j <= i
    2. MLP is position-independent - each token's MLP is independent
    
    Uses MoRRouter and MoRExecutor from hydra.routing.mixture_of_recursions.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int = 16,
        n_kv_heads: int = 4,
        compression_factor: int = 4,
        mlp_ratio: float = 2.67,
        max_seq_len: int = 8192,
        max_recursions: int = 6,
        adaptive: bool = True,
        halt_threshold: float = 0.9,
        ponder_loss_weight: float = 0.01,
        layer_idx: int = 0,
        total_layers: int = 1,
        attention_type: str = "ccqa",  # CCGQA attention (LA3 removed due to spikes)
        # Dimension-aware depth scaling (optional, backward compatible)
        dim_ref: int = 768,
        depth_alpha: float = 0.0,  # 0.0 = disabled (default), 0.25-0.5 = enabled
        depth_scale_max: float = 2.0,
        **attention_kwargs,
    ):
        super().__init__()

        self.max_recursions = max_recursions
        self.adaptive = adaptive
        self.layer_idx = layer_idx
        self.total_layers = total_layers
        self.dim = dim
        self.attention_type = attention_type
        
        # Compute depth scale once at init (compile-safe: build-time constant)
        self._depth_scale = dim_to_depth_scale(dim, dim_ref, depth_alpha, depth_scale_max)

        # Pop MoD params (they apply to MLP only, handled separately)
        mod_mlp_capacity = attention_kwargs.pop("mod_mlp_capacity", None)
        mod_mlp_aux_weight = attention_kwargs.pop("mod_mlp_aux_weight", 0.01)
        mod_mlp_warmup = attention_kwargs.pop("mod_mlp_warmup", 100)
        mod_force_enable_step = attention_kwargs.pop("mod_force_enable_step", None)
        mod_enable_loss_threshold = attention_kwargs.pop("mod_enable_loss_threshold", None)
        mod_loss_aware_weight = attention_kwargs.pop("mod_loss_aware_weight", 0.0)
        mor_warmup = attention_kwargs.pop("mor_warmup", 1000)  # MoR ponder loss warmup
        self.use_mod_mlp = mod_mlp_capacity is not None and mod_mlp_capacity > 0
        
        # =====================================================================
        # Attention: CCGQA backend (LA3 removed due to gradient spike issues)
        # =====================================================================
        self.attention = build_hybrid_attention_module(
            attention_type,
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            max_seq_len=max_seq_len,
            compression_factor=compression_factor,
            attention_kwargs=attention_kwargs,
        )
        self.residual_scale = 0.5

        # Norm selection
        use_ada_norm = os.environ.get("HYDRA_USE_ADA_RMSNORM", "1") == "1"
        if use_ada_norm:
            from hydra.layers import AdaRMSNorm as _Norm
        else:
            from hydra.layers import RMSNorm as _Norm

        self.norm1 = _Norm(dim)  # Pre-attention norm
        
        # MLP
        hidden_dim = int(dim * mlp_ratio)
        hidden_dim = ((hidden_dim + 255) // 256) * 256
        self.mlp = SwiGLUMLP(dim, hidden_dim)
        self.norm2 = _Norm(dim)  # Pre-MLP norm
        
        # Optional MoD on MLP
        if self.use_mod_mlp:
            self.mod_mlp_wrapper = MoDMLPWrapper(
                mlp=self.mlp,
                dim=dim,
                capacity_ratio=mod_mlp_capacity,
                aux_loss_weight=mod_mlp_aux_weight,
                warmup_steps=mod_mlp_warmup,
                force_enable_step=mod_force_enable_step,
                max_seq_len=max_seq_len,
                enable_loss_threshold=mod_enable_loss_threshold,
                loss_aware_weight=mod_loss_aware_weight,
            )
        else:
            self.mod_mlp_wrapper = None
        
        # Backward compatibility
        self.block = None

        # =====================================================================
        # MoR Router and Executor (from hydra.routing.mixture_of_recursions)
        # =====================================================================
        mor_config = MoRConfig(
            dim=dim,
            n_recursions=max_recursions,
            ponder_loss_weight=ponder_loss_weight,
            warmup_steps=mor_warmup,  # From kwargs, not hardcoded
            layer_idx=layer_idx,
            total_layers=total_layers,
            dim_ref=dim_ref,
            depth_alpha=depth_alpha,
            depth_scale_max=depth_scale_max,
            advantage_loss_scale=attention_kwargs.pop("mor_advantage_loss_scale", 0.1),
        )
        self.mor_router = MoRRouter(mor_config)
        self.mor_executor = MoRExecutor(mor_config)
        
        # Store config for diagnostics
        self._mor_config = mor_config
        self.ponder_loss_weight = ponder_loss_weight

        # NOTE: Legacy halt_predictor removed.
        # MoR routing is handled by MoRRouter/MoRExecutor; keeping an unused
        # predictor bloats params and breaks gradient-flow expectations.

        # Global step for warmup scheduling
        self.register_buffer("_global_step", torch.zeros((), dtype=torch.int64), persistent=False)
        # OPTIMIZATION: Persistent zero scalar to avoid graph breaks and allocations
        self.register_buffer("_zero_scalar", torch.tensor(0.0), persistent=False)
        
        self._warmup_steps = 2500
        self._mor_enable_step = 0
        self._mor_rampup_steps = 0

        self.final_norm = RMSNorm(dim)
        self._ponder_loss: torch.Tensor = self._zero_scalar
        self._avg_ponder_time: torch.Tensor = self._zero_scalar

        # Debug stats
        self._last_target_depths: Optional[torch.Tensor] = None
        self._last_router_probs_mean: float = 0.0
        self._last_router_probs_std: float = 0.0

    def _forward_fixed(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Fixed recursion mode - attention once, then all MLP recursions.
        
        OPTION A: Attention is dense (full sequence), MLP is recursive.
        Uses mor_executor's recursion_bias/embed for consistency.
        """
        # ATTENTION: Runs ONCE on full sequence (dense)
        h = x + self.residual_scale * self.attention(self.norm1(x), mask=mask)
        
        # MLP RECURSIONS: Run max_recursions times (all tokens, all depths)
        for i in range(self.max_recursions):
            rec_bias = self.mor_executor.recursion_bias[i].squeeze()
            rec_embed = self.mor_executor.recursion_embed(
                self.mor_executor._recursion_indices[i:i+1]
            ).squeeze()
            h_with_rec = h + rec_bias + rec_embed
            
            # Apply MLP (with optional MoD)
            if self.mod_mlp_wrapper is not None:
                h = h + self.mod_mlp_wrapper(self.norm2(h_with_rec))
            else:
                h = h + self.mlp(self.norm2(h_with_rec))
        
        return self.final_norm(h)

    def _forward_mor(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """MoR forward pass using MoRRouter and MoRExecutor.
        
        Architecture:
        1. ATTENTION runs ONCE on full sequence (dense, all tokens)
        2. MoRRouter predicts per-token depth
        3. MoRExecutor applies MLP recursions with depth routing
        """
        # =====================================================================
        # STEP 1: ATTENTION (ONCE, DENSE)
        # =====================================================================
        h = x + self.residual_scale * self.attention(self.norm1(x), mask=mask)  # [B, L, D]
        
        # =====================================================================
        # STEP 2: MoR ROUTER - Predict per-token MLP recursion depth
        # =====================================================================
        depths, probs, logits = self.mor_router(h)
        
        # Store for diagnostics and later advantage computation
        self._last_target_depths = depths.detach().float()
        self._last_router_probs_tensor = probs.detach()
        self._last_router_logits = logits  # Keep with grad for advantage loss
        self._last_depths = depths
        self._last_probs = probs
        
        # =====================================================================
        # STEP 3: MoR EXECUTOR - Apply MLP recursions with depth routing
        # =====================================================================
        # Choose MLP (with optional MoD wrapper)
        mlp = self.mod_mlp_wrapper if self.mod_mlp_wrapper is not None else self.mlp
        
        output = self.mor_executor(h, depths, probs, mlp, self.norm2)
        
        # =====================================================================
        # STEP 4: COMPUTE PONDER LOSS (use mor_router's method)
        # =====================================================================
        # Base ponder loss without advantage (advantage added post-CE in model)
        ponder_loss = self.mor_router.compute_ponder_loss(
            depths, probs, logits,
            token_losses=None,  # No CE losses yet
            baseline=None,
        )
        
        # Diagnostics
        if self.training:
            n_rec = self.max_recursions
            depth_continuous = probs * (n_rec - 1)
            self._ponder_loss = ponder_loss.detach()
            self._last_avg_depth = depth_continuous.mean().detach()
        
        return self.final_norm(output), ponder_loss

    def compute_advantage_loss(
        self,
        token_losses: torch.Tensor,
        baseline: "MovingAverageBaseline",
    ) -> torch.Tensor:
        """Compute advantage-scaled loss for loss-driven routing.
        
        Call this AFTER forward pass with per-token CE losses.
        Returns additional loss term that scales router gradients by advantage.
        
        Args:
            token_losses: Per-token CE losses [batch, seq]
            baseline: MovingAverageBaseline from parent model
        
        Returns:
            Advantage loss tensor (scalar) - add to total loss for backprop.
        """
        if not hasattr(self, '_last_probs') or self._last_probs is None:
            return self._zero_scalar
        
        probs = self._last_probs
        n_rec = self.max_recursions
        depth_continuous = probs * (n_rec - 1)
        
        # Compute advantage from baseline
        advantage = baseline.compute_advantage(token_losses)
        
        # Loss-driven routing: reward depth for hard tokens, penalize for easy
        # Positive advantage (hard) * high depth → negative loss → encourages depth
        # Negative advantage (easy) * high depth → positive loss → penalizes depth
        scale = self._mor_config.advantage_loss_scale
        advantage_loss = -(advantage * depth_continuous).mean() * scale
        
        return advantage_loss

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.adaptive:
            # GRAPH-STABLE TRANSITION: Compute both paths during rampup to avoid
            # torch.compile recompilation when transitioning. Router learns from step 0.
            
            # Get rampup scale (0.0 before enable, 0.0-1.0 during rampup, 1.0 after)
            rampup_scale = self.get_mor_rampup_scale()

            # Pre-enable: run fixed-depth only (avoid wasting compute on adaptive path
            # when its contribution is exactly zero).
            if rampup_scale <= 0.0:
                return self._forward_fixed(x, mask=mask)
            
            # Always compute adaptive output (router trains from step 0)
            adaptive_output, _ = self._forward_mor(x, mask=mask)
            
            # OPTIMIZATION: Skip fixed path when rampup is complete (rampup_scale=1.0)
            # This saves ~30% compute after MoR is fully enabled.
            # The graph has already seen scale=1.0 at the final quantized rampup step,
            # so this branch won't cause recompilation.
            if rampup_scale >= 1.0:
                return adaptive_output
            
            # During rampup (scale < 1.0): blend both paths for smooth transition
            fixed_output = self._forward_fixed(x, mask=mask)
            return rampup_scale * adaptive_output + (1.0 - rampup_scale) * fixed_output
        return self._forward_fixed(x, mask=mask)

    def forward_with_losses(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """Forward pass that returns output and losses dict.

        Returns (output, {"ponder_loss": tensor, "aux_loss": tensor})
        - ponder_loss: MoR's compute cost (higher depth = more compute)
        - aux_loss: MoD's load balancing loss (if using MoD on MLP)

        GRAPH-STABLE TRANSITION: Compute both paths during rampup to avoid
        torch.compile recompilation. Router learns from step 0.
        After rampup (scale=1.0), skip fixed path for ~30% compute savings.
        """
        if self.adaptive:
            # Get rampup scale (0.0 before enable, 0.0-1.0 during rampup, 1.0 after)
            rampup_scale = self.get_mor_rampup_scale()

            # Pre-enable: run fixed-depth only and fully gate ponder_loss.
            # This preserves the intended curriculum behavior and avoids doing
            # 2x compute for the first ~mor_enable_pct of training.
            if rampup_scale <= 0.0:
                output = self._forward_fixed(x, mask=mask)
                if self.mod_mlp_wrapper is not None:
                    aux_loss = self.mod_mlp_wrapper.get_aux_loss()
                else:
                    aux_loss = self._zero_scalar
                zero_loss = self._zero_scalar
                if self.training:
                    self._ponder_loss = zero_loss.detach()
                return output, {"ponder_loss": zero_loss, "aux_loss": aux_loss}
            
            # Always compute adaptive output (router trains from step 0)
            adaptive_output, ponder_loss = self._forward_mor(x, mask=mask)

            # Apply MoR curriculum to ponder loss.
            # Before enable (rampup_scale=0), this fully gates MoR regularization so
            # fixed-depth training behaves like fixed-depth training.
            ponder_loss = ponder_loss * rampup_scale
            
            # Collect aux_loss from MoD MLP wrapper if present
            if self.mod_mlp_wrapper is not None:
                aux_loss = self.mod_mlp_wrapper.get_aux_loss()
            else:
                aux_loss = self._zero_scalar
            
            # OPTIMIZATION: Skip fixed path when rampup is complete (rampup_scale=1.0)
            # This saves ~30% compute after MoR is fully enabled.
            if rampup_scale >= 1.0:
                if self.training:
                    self._ponder_loss = ponder_loss.detach()
                return adaptive_output, {"ponder_loss": ponder_loss, "aux_loss": aux_loss}
            
            # During rampup (scale < 1.0): blend both paths for smooth transition
            fixed_output = self._forward_fixed(x, mask=mask)
            output = rampup_scale * adaptive_output + (1.0 - rampup_scale) * fixed_output
            
            # Update cached loss for get_ponder_loss() consistency
            if self.training:
                self._ponder_loss = ponder_loss.detach()
            
            return output, {"ponder_loss": ponder_loss, "aux_loss": aux_loss}
        else:
            output = self._forward_fixed(x)
            
            # Collect aux_loss from MoD MLP wrapper if present (even in fixed mode)
            if self.mod_mlp_wrapper is not None:
                aux_loss = self.mod_mlp_wrapper.get_aux_loss()
            else:
                aux_loss = self._zero_scalar
            
            zero_loss = self._zero_scalar
            # Update cached loss for get_ponder_loss() consistency
            if self.training:
                self._ponder_loss = zero_loss.detach()
            return output, {"ponder_loss": zero_loss, "aux_loss": aux_loss}

    def get_ponder_loss(self) -> torch.Tensor:
        """Get the cached ponder_loss from the last forward pass.
        
        Note: This returns the SCALED value (after rampup scaling is applied).
        Use this for logging/debugging. For training, use forward(return_losses=True).
        """
        return self._ponder_loss

    @torch.compiler.disable
    def get_routing_stats(self) -> dict:
        """Get routing statistics for debugging (compile-disabled for Python dict ops).

        Returns dict with:
            - router_probs_mean: Mean router probability
            - router_probs_std: Std of router probabilities
            - depth_histogram: Counts per recursion level (if available)
            - avg_depth: Average target depth
            - compute_savings: Real compute savings from early exit
        """
        # Extract mean/std from stored tensor (deferred .item() outside compile)
        router_probs_tensor = getattr(self, '_last_router_probs_tensor', None)
        if router_probs_tensor is not None:
            self._last_router_probs_mean = router_probs_tensor.mean().item()
            self._last_router_probs_std = router_probs_tensor.std().item()
        
        stats = {
            "router_probs_mean": getattr(self, '_last_router_probs_mean', 0.0),
            "router_probs_std": getattr(self, '_last_router_probs_std', 0.0),
            "layer_idx": self.layer_idx,
            "total_layers": self.total_layers,
            "target_depth_ratio": getattr(self, "target_depth_ratio", 0.6),
            "expected_avg_depth": getattr(self, "target_depth_ratio", 0.6)
            * (self.max_recursions - 1),
        }

        if self._last_target_depths is not None:
            depths = self._last_target_depths.flatten().float()  # Convert to float for mean/std
            stats["avg_depth"] = depths.mean().item()
            stats["depth_std"] = depths.std().item()
            # Histogram: count tokens at each recursion level (0, 1, 2, ...)
            # Cast to float32 for histc (bfloat16 not supported)
            hist = torch.histc(
                depths,  # Already float from above
                bins=self.max_recursions,
                min=0,
                max=self.max_recursions - 1,
            )
            stats["depth_histogram"] = hist.tolist()
        
        # Compute savings from real gather/scatter routing
        if hasattr(self, "_recursion_tokens_processed"):
            tokens_per_recursion = self._recursion_tokens_processed
            # Convert tensors to int for summation
            total_tokens = sum(int(t) if hasattr(t, 'item') else t for t in tokens_per_recursion)
            max_possible = len(tokens_per_recursion) * (self._last_target_depths.numel() if self._last_target_depths is not None else 0)
            if max_possible > 0:
                stats["compute_ratio"] = total_tokens / max_possible
                stats["compute_savings_pct"] = (1.0 - total_tokens / max_possible) * 100
            stats["tokens_per_recursion"] = [int(t) if hasattr(t, 'item') else t for t in tokens_per_recursion]

        return stats

    def set_global_step(self, step: int):
        """Set global training step for warmup scheduling."""
        self._global_step.fill_(step)
        # Cache as Python int for use in forward path (avoids .item() graph break)
        self._cached_global_step = step
        # Cache MoR adaptive decision to avoid .item() in forward (prevents graph break)
        self._mor_adaptive_cached = step >= self._mor_enable_step
        # Cache rampup scale - QUANTIZED to 10 discrete values to prevent recompilation
        # torch.compile guards on Python values; continuous changes cause recompile storms
        if step < self._mor_enable_step:
            self._mor_rampup_scale_cached = 0.0
        elif self._mor_rampup_steps <= 0:
            self._mor_rampup_scale_cached = 1.0
        else:
            steps_since_enable = step - self._mor_enable_step
            raw_scale = min(1.0, steps_since_enable / self._mor_rampup_steps)
            # Quantize to 10 discrete values: 0.0, 0.1, 0.2, ... 1.0
            # This limits torch.compile to 11 possible graphs instead of thousands
            self._mor_rampup_scale_cached = round(raw_scale * 10) / 10
        # Propagate to MoD MLP wrapper if present
        if self.mod_mlp_wrapper is not None:
            self.mod_mlp_wrapper.set_global_step(step)
    
    def set_mor_enable_step(self, enable_step: int, rampup_steps: int = 1000):
        """Set the step at which MoR adaptive routing enables.
        
        Args:
            enable_step: Step at which to enable adaptive routing.
                        Before this, fixed-depth mode is used (all recursions).
                        Set to 0 to always use adaptive routing (legacy).
            rampup_steps: Number of steps to ramp up routing after enable_step.
                         During rampup, routing gradually transitions from fixed to adaptive.
        """
        self._mor_enable_step = enable_step
        self._mor_rampup_steps = rampup_steps
    
    def is_mor_adaptive_enabled(self) -> bool:
        """Check if MoR adaptive routing is currently enabled."""
        # Use cached value set by set_global_step to avoid .item() graph break
        return getattr(self, '_mor_adaptive_cached', False)
    
    def get_mor_rampup_scale(self) -> float:
        """Get the current rampup scale (0 to 1) for MoR routing.
        
        Returns:
            0.0 if before enable_step
            0.0 to 1.0 during rampup period
            1.0 after rampup completes
        """
        # Use cached value from set_global_step (no step-dependent branching)
        return getattr(self, '_mor_rampup_scale_cached', 1.0)


class CCGQAMoDMoRModel(nn.Module):
    """
    Full CCGQA + MoD + MoR Model.

    Architecture:
    - Token embedding
    - N x CCGQAMoRBlock with MoD routing (each block is recursive)
    - Each MoR block wraps MoD routing (some tokens skip, others recursively process)
    - Final norm + output projection

    This combines:
    1. CCGQA: Compressed attention in latent space (memory efficient)
    2. MoD: Dynamic token skipping (compute efficient)
    3. MoR: Recursive weight sharing (parameter efficient)

    Note:
        This model is designed for CAUSAL autoregressive language modeling only.
        Padding masks are NOT supported - all sequences are assumed to be unpadded.
        The attention uses is_causal=True when no mask is provided.

    Args:
        vocab_size: Vocabulary size
        dim: Model dimension
        n_mor_blocks: Number of MoR blocks
        recursions_per_block: Recursions per MoR block (effective_layers = n_mor_blocks * recursions)
        n_heads: Number of query heads
        n_kv_heads: Number of KV heads
        compression_factor: CCGQA compression factor
        mlp_ratio: MLP hidden dim ratio
        max_seq_len: Maximum sequence length
        mod_capacity: MoD capacity ratio (fraction of tokens processed)
        adaptive: Use adaptive halting in MoR
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        dim: int = 1536,
        n_mor_blocks: int = 4,
        recursions_per_block: int = 6,
        n_heads: int = 24,
        n_kv_heads: int = 4,
        compression_factor: int = 4,
        mlp_ratio: float = 2.67,
        max_seq_len: int = 8192,
        mod_capacity: float = 0.5,
        aux_loss_weight: float = None,  # None = auto-scale based on depth
        adaptive: bool = True,
        tie_weights: bool = True,
        hybrid_attention: bool = True,
        mod_loss_aware_weight: float = 0.0,
        # Dimension-aware MoR depth scaling (optional)
        dim_ref: int = 768,  # Reference dimension for scale=1.0
        depth_alpha: float = 0.0,  # Power-law exponent (0=disabled)
        depth_scale_max: float = 2.0,  # Maximum scaling factor
        **kwargs,
    ):
        super().__init__()

        self.dim = dim
        self.vocab_size = vocab_size
        self.n_mor_blocks = n_mor_blocks
        self.recursions_per_block = recursions_per_block
        self.effective_layers = n_mor_blocks * recursions_per_block
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.compression_factor = compression_factor
        self.mlp_ratio = mlp_ratio
        self.max_seq_len = max_seq_len
        self.mod_capacity = mod_capacity
        self.mod_loss_aware_weight = float(mod_loss_aware_weight)
        self.adaptive = adaptive
        self.hybrid_attention = hybrid_attention
        
        # Store dim-aware depth scaling params for introspection/checkpoint
        self.dim_ref = dim_ref
        self.depth_alpha = depth_alpha
        self.depth_scale_max = depth_scale_max

        # Auto-scale aux_loss_weight based on effective depth and dimension
        # Deeper/larger models need stronger regularization to maintain capacity
        # Scale both by depth AND dimension to handle large models
        if aux_loss_weight is None:
            # Base: 0.5 for 32 layers @ dim=768 (increased significantly)
            # Scale by depth ratio AND sqrt of dimension ratio
            # This ensures aux_loss competes with larger CE losses in big models
            # The aux_loss is |probs - target|, so weight needs to be strong
            # enough to prevent router collapse (probs -> 0 or 1)
            # Target: ~5-10% of CE loss when capacity deviates by 0.35
            depth_scale = max(1.0, self.effective_layers / 32)
            dim_scale = max(1.0, (dim / 768) ** 0.5)  # sqrt scaling for dim
            aux_loss_weight = 0.5 * depth_scale * dim_scale
        self.aux_loss_weight = aux_loss_weight

        # Global step for warmup scheduling - use tensor buffer to avoid torch.compile recompilation
        self.register_buffer("_global_step", torch.zeros((), dtype=torch.int64), persistent=False)
        
        # Persistent zero scalar for efficient loss initialization (avoids per-step allocation)
        self.register_buffer("_zero_scalar", torch.tensor(0.0), persistent=False)

        # Token embedding
        self.tok_emb = nn.Embedding(vocab_size, dim)

        # =====================================================================
        # Attention pattern: ALL CCGQA (homogeneous)
        # LA3 was removed due to gradient spike issues. CCGQA with proper
        # output_scale=1.0 provides stable gradients.
        # =====================================================================
        attention_pattern = ["ccqa"] * n_mor_blocks
        self._attention_pattern = attention_pattern  # Store for introspection
        
        # Log attention pattern if rank 0
        if not hasattr(self, '_logged_attention_pattern'):
            import logging
            logger = logging.getLogger("HYDRA")
            logger.info(f"Attention: CCGQA ({n_mor_blocks} blocks)")
            self._logged_attention_pattern = True

        # MoR blocks - MoD applies to MLP sublayer only (attention is always dense)
        # MoD is applied to middle blocks (first and last process all tokens)
        self.layers = nn.ModuleList()
        for i in range(n_mor_blocks):
            # Determine if this block uses MoD on MLP
            use_mod_mlp = (0 < i < n_mor_blocks - 1)
            
            mor_block = CCGQAMoRBlock(
                dim=dim,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                compression_factor=compression_factor,
                mlp_ratio=mlp_ratio,
                max_seq_len=max_seq_len,
                max_recursions=recursions_per_block,
                adaptive=adaptive,
                layer_idx=i,
                total_layers=n_mor_blocks,
                attention_type=attention_pattern[i],
                # MoD on MLP only for middle blocks
                mod_mlp_capacity=mod_capacity if use_mod_mlp else None,
                mod_mlp_aux_weight=self.aux_loss_weight if use_mod_mlp else 0.0,
                mod_mlp_warmup=kwargs.get("mod_mlp_warmup", 100),  # Soft routing warmup steps
                mod_force_enable_step=kwargs.get("mod_force_enable_step", None),
                mod_enable_loss_threshold=kwargs.get("mod_enable_loss_threshold", None),
                mod_loss_aware_weight=self.mod_loss_aware_weight,
                mor_warmup=kwargs.get("mor_warmup", 1000),  # MoR ponder loss warmup
                # Dimension-aware depth scaling
                dim_ref=dim_ref,
                depth_alpha=depth_alpha,
                depth_scale_max=depth_scale_max,
            )
            self.layers.append(mor_block)

        # Final norm
        self.norm = RMSNorm(dim)

        # Output projection
        self.output = nn.Linear(dim, vocab_size, bias=False)

        # Loss-driven routing baseline (Gemini approach)
        # Tracks EMA of per-token CE losses to provide advantage signal
        self.loss_baseline = MovingAverageBaseline(
            decay=0.99,
            warmup_steps=1000,
        )

        # Initialize weights BEFORE tying (so Linear init doesn't overwrite Embedding)
        self._init_weights()

        # Weight tying (AFTER init so embedding std=1.0 is preserved)
        if tie_weights:
            self.output.weight = self.tok_emb.weight

    def _init_weights(self):
        """Initialize weights with scaled init for deep networks.

        NOTE: Router biases in CCGQAMoRBlock and CCGQAMoDBlockWrapper are
        pre-initialized with specific values (logit of target capacity/depth).
        We skip re-initializing these to preserve the intended routing behavior.
        """
        residual_scale = 1.0 / math.sqrt(2 * self.effective_layers)

        for name, module in self.named_modules():
            is_te_linear = (
                module.__class__.__module__.startswith("transformer_engine")
                and hasattr(module, "weight")
                and isinstance(getattr(module, "weight", None), torch.nn.Parameter)
                and getattr(module, "weight").ndim == 2
            )

            if isinstance(module, nn.Linear) or is_te_linear:
                # Skip router layers - they have specially initialized biases
                # for targeting specific capacity ratios and recursion depths
                is_router = "router" in name

                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                bias = getattr(module, "bias", None)
                if bias is not None and not is_router:
                    nn.init.zeros_(bias)
                if "o_proj" in name or "down" in name:
                    module.weight.data *= residual_scale
            elif isinstance(module, nn.Embedding):
                # GPT-2 style: std = 0.02 but NOT scaled down by residual_scale
                # The key is NOT tying weights before init (done above)
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Conv1d):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def resize_rope_cache(self, new_max_seq_len: int) -> None:
        """Resize RoPE cache in all attention modules.
        
        This is needed when resuming from a checkpoint that was saved
        with a different max_seq_len than the current training config.
        Call this after load_state_dict() when using stepped sequence schedules.
        
        Args:
            new_max_seq_len: The new maximum sequence length to support.
        """
        resized_count = 0
        for layer in self.layers:
            if hasattr(layer, "attention"):
                attn = layer.attention
                if hasattr(attn, "_init_rope") and hasattr(attn, "cos_cached"):
                    current_len = attn.cos_cached.shape[2]
                    if current_len < new_max_seq_len:
                        attn._init_rope(new_max_seq_len)
                        # Move buffers to same device as model
                        device = next(attn.parameters()).device
                        attn.cos_cached = attn.cos_cached.to(device)
                        attn.sin_cached = attn.sin_cached.to(device)
                        resized_count += 1
        
        if resized_count > 0:
            _log.debug(f"Resized RoPE cache to {new_max_seq_len} in {resized_count} attention modules")
        self.max_seq_len = new_max_seq_len

    # =========================================================================
    # Gradient Checkpointing - Trade compute for memory
    # =========================================================================
    _gradient_checkpointing: bool = False
    _checkpoint_every_n: int = 1  # Checkpoint every N layers (1 = all, 2 = every other, etc.)
    
    def enable_gradient_checkpointing(self, every_n: int = 1) -> None:
        """Enable gradient checkpointing to reduce memory usage.
        
        When enabled, activations are recomputed during the backward pass
        instead of being stored. This reduces memory but increases compute.
        
        Args:
            every_n: Checkpoint every N layers (default=1, all layers).
                    Use 2 for ~15% less overhead, 3 for ~20% less overhead.
                    Higher values save less memory but reduce recomputation.
        
        Memory/Compute tradeoffs:
            every_n=1: ~50% memory savings, ~30% compute overhead (default)
            every_n=2: ~35% memory savings, ~15% compute overhead
            every_n=3: ~25% memory savings, ~10% compute overhead
        
        Use when:
        - Training with large batch sizes
        - Training with long sequences
        - GPU memory is the bottleneck
        """
        self._gradient_checkpointing = True
        self._checkpoint_every_n = max(1, every_n)
    
    def disable_gradient_checkpointing(self) -> None:
        """Disable gradient checkpointing."""
        self._gradient_checkpointing = False
    
    @property
    def is_gradient_checkpointing(self) -> bool:
        """Check if gradient checkpointing is enabled."""
        return self._gradient_checkpointing

    def forward(
        self, x: torch.Tensor, return_losses: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        """Forward pass.

        Args:
            x: Input token ids [batch, seq_len]
            return_losses: If True, return (logits, losses_dict) for training.
                          Per MoR paper, aux_loss/ponder_loss should be added to
                          training objective with coefficient ~0.001.

        Returns:
            If return_losses=False: logits tensor
            If return_losses=True: (logits, {"aux_loss": ..., "ponder_loss": ...})
        """
        # Token embedding with sqrt(dim) scaling (LLaMA style)
        # This ensures embeddings have std ≈ 1.0 to survive residual connections
        # Without this, embedding std=0.02 is overwhelmed by attention output std≈0.1
        h = self.tok_emb(x) * math.sqrt(self.dim)

        if return_losses:
            # Training path: collect losses AND use gradient checkpointing
            # FIX: We now support checkpointing WITH loss collection.
            # Strategy: Use forward_with_losses but wrap in checkpoint.
            # The losses are live tensors computed in forward - they will be
            # recomputed during backward automatically.
            
            if self._gradient_checkpointing and self.training:
                # Checkpointed forward with loss collection
                layer_results = []
                for i, layer in enumerate(self.layers):
                    if i % self._checkpoint_every_n == 0:
                        # Checkpoint this layer - wrap forward_with_losses
                        # We use a custom wrapper to handle tuple output
                        h, layer_losses = gradient_checkpoint(
                            layer.forward_with_losses,
                            h,
                            use_reentrant=False,
                        )
                    else:
                        h, layer_losses = layer.forward_with_losses(h)
                    layer_results.append(layer_losses)
            else:
                # Non-checkpointed forward with loss collection
                layer_results = []
                for layer in self.layers:
                    h, layer_losses = layer.forward_with_losses(h)
                    layer_results.append(layer_losses)

            # Use list comprehensions for loss extraction
            aux_losses = [
                losses["aux_loss"] for losses in layer_results if "aux_loss" in losses
            ]
            ponder_losses = [
                losses["ponder_loss"]
                for losses in layer_results
                if "ponder_loss" in losses
            ]

            h = self.norm(h)
            logits = self.output(h)

            aux_loss = sum(aux_losses) if aux_losses else self._zero_scalar
            ponder_loss = sum(ponder_losses) if ponder_losses else self._zero_scalar
            return logits, {"aux_loss": aux_loss, "ponder_loss": ponder_loss}
        else:
            # Fast path - no loss computation overhead
            # Use gradient checkpointing if enabled (saves memory, costs compute)
            if self._gradient_checkpointing and self.training:
                for i, layer in enumerate(self.layers):
                    # Selective checkpointing: only checkpoint every N layers
                    # This reduces recomputation overhead while still saving memory
                    if i % self._checkpoint_every_n == 0:
                        # use_reentrant=False is required for torch.compile compatibility
                        h = gradient_checkpoint(layer, h, use_reentrant=False)
                    else:
                        h = layer(h)
            else:
                for layer in self.layers:
                    h = layer(h)

            h = self.norm(h)
            return self.output(h)

    def forward_hidden(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return post-norm hidden states (pre-logits).

        Used for memory-efficient loss functions (e.g., chunked CE) that avoid
        materializing the full logits tensor.

        Args:
            x: Input token ids [batch, seq_len]
            mask: Optional attention mask [batch, seq_len] (1=valid, 0=pad)

        Returns:
            Hidden states after final norm, shape [batch, seq_len, dim]
        """
        # Token embedding with sqrt(dim) scaling (LLaMA style)
        h = self.tok_emb(x) * math.sqrt(self.dim)

        # Forward through layers with gradient checkpointing support
        if self._gradient_checkpointing and self.training:
            for i, layer in enumerate(self.layers):
                if i % self._checkpoint_every_n == 0:
                    h = gradient_checkpoint(layer, h, mask, use_reentrant=False)
                else:
                    h = layer(h, mask)
        else:
            for layer in self.layers:
                h = layer(h, mask)

        return self.norm(h)

    def forward_hidden_with_losses(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, dict]:
        """Return post-norm hidden states + auxiliary losses (no logits).

        Memory-efficient alternative to forward(return_losses=True) that avoids
        materializing the full logits tensor. Use with chunked cross-entropy.

        Args:
            x: Input token ids [batch, seq_len]
            mask: Optional attention mask [batch, seq_len] (1=valid, 0=pad)

        Returns:
            Tuple of:
                - hidden: Hidden states after final norm [batch, seq_len, dim]
                - losses: Dict with 'aux_loss' and 'ponder_loss'
        """
        # Token embedding with sqrt(dim) scaling (LLaMA style)
        h = self.tok_emb(x) * math.sqrt(self.dim)

        # Forward through layers collecting losses
        if self._gradient_checkpointing and self.training:
            # Checkpointed forward WITH loss collection (mirrors forward(return_losses=True)).
            layer_results = []
            for i, layer in enumerate(self.layers):
                if i % self._checkpoint_every_n == 0:
                    h, layer_losses = gradient_checkpoint(
                        layer.forward_with_losses,
                        h,
                        mask,
                        use_reentrant=False,
                    )
                else:
                    h, layer_losses = layer.forward_with_losses(h, mask)
                layer_results.append(layer_losses)
        else:
            layer_results = []
            for layer in self.layers:
                h, layer_losses = layer.forward_with_losses(h, mask)
                layer_results.append(layer_losses)

        # Aggregate losses
        aux_losses = [losses["aux_loss"] for losses in layer_results if "aux_loss" in losses]
        ponder_losses = [losses["ponder_loss"] for losses in layer_results if "ponder_loss" in losses]

        aux_loss = sum(aux_losses) if aux_losses else self._zero_scalar
        ponder_loss = sum(ponder_losses) if ponder_losses else self._zero_scalar

        return self.norm(h), {"aux_loss": aux_loss, "ponder_loss": ponder_loss}

    def get_aux_losses(self) -> dict:
        """Get all auxiliary losses for training.
        
        Returns a dict with:
            - mod_aux_loss: Total MoD load balancing loss (from MoDMLPWrapper)
            - mor_ponder_loss: Total MoR compute cost loss (from CCGQAMoRBlock)
            - total: Combined auxiliary loss (mod_aux_loss + mor_ponder_loss)
        
        Usage in training loop:
            logits, losses = model(x, return_losses=True)
            ce_loss = F.cross_entropy(logits, targets)
            aux = model.get_aux_losses()  # Or use losses dict directly
            total_loss = ce_loss + aux['total']
            total_loss.backward()
        
        Note: When using forward(return_losses=True), the losses dict already
        contains 'aux_loss' and 'ponder_loss' which are the same values.
        This method is provided for cases where you want to query losses
        separately from forward pass.
        """
        mod_aux_loss = self._zero_scalar
        mor_ponder_loss = self._zero_scalar

        for layer in self.layers:
            # MoR ponder_loss from CCGQAMoRBlock
            if hasattr(layer, "get_ponder_loss"):
                mor_ponder_loss = mor_ponder_loss + layer.get_ponder_loss()
            # MoD aux_loss from MoD MLP wrapper
            if hasattr(layer, "mod_mlp_wrapper") and layer.mod_mlp_wrapper is not None:
                mod_aux_loss = mod_aux_loss + layer.mod_mlp_wrapper.get_aux_loss()

        return {
            "mod_aux_loss": mod_aux_loss,
            "mor_ponder_loss": mor_ponder_loss,
            "total": mod_aux_loss + mor_ponder_loss,
        }

    def update_loss_baseline(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        ignore_index: int = -100,
    ) -> torch.Tensor:
        """Update loss baseline with per-token CE losses and return advantage.
        
        This is the core of loss-driven routing (Gemini approach):
        1. Compute per-token CE losses (not reduced)
        2. Update EMA baseline with batch statistics
        3. Return advantage signal for router gradient scaling
        
        Call this in training loop AFTER forward pass but BEFORE backward:
        
            logits, aux_losses = model(x, return_losses=True)
            advantage = model.update_loss_baseline(logits, targets)
            # advantage can be used to scale ponder loss if desired
            ce_loss = F.cross_entropy(logits.view(-1, vocab), targets.view(-1))
            total_loss = ce_loss + aux_losses['aux_loss'] + aux_losses['ponder_loss']
            total_loss.backward()
        
        Args:
            logits: Model output logits [batch, seq, vocab]
            targets: Target token ids [batch, seq]
            ignore_index: Index to ignore in loss computation (default: -100)
        
        Returns:
            Advantage tensor [batch, seq] - positive for hard tokens, negative for easy
        """
        B, L, V = logits.shape
        
        # Compute per-token cross-entropy (no reduction)
        logits_flat = logits.view(-1, V)  # [B*L, V]
        targets_flat = targets.view(-1)   # [B*L]
        
        # Per-token loss without reduction
        with torch.no_grad():
            token_losses = F.cross_entropy(
                logits_flat,
                targets_flat,
                ignore_index=ignore_index,
                reduction='none',
            ).view(B, L)  # [B, L]
            
            # Mask ignored positions
            valid_mask = (targets != ignore_index)
            token_losses = token_losses * valid_mask.float()
        
        # Update baseline with this batch
        self.loss_baseline.update(token_losses[valid_mask])
        
        # Compute advantage (positive = harder than baseline)
        advantage = self.loss_baseline.compute_advantage(token_losses)
        
        # Store for diagnostics
        self._last_advantage = advantage.detach()
        self._last_baseline = self.loss_baseline.baseline
        
        # Return mean advantage as scalar for backward compatibility with tests
        return advantage.mean()

    def compute_advantage_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        ignore_index: int = -100,
    ) -> torch.Tensor:
        """Compute advantage-scaled loss for loss-driven routing.
        
        This is the main entry point for loss-driven MoR training.
        Call after forward pass but with logits that have gradients.
        
        The advantage loss encourages:
        - More depth for hard tokens (high CE loss)
        - Less depth for easy tokens (low CE loss)
        
        Usage in training loop:
            logits, aux_losses = model(x, return_losses=True)
            advantage_loss = model.compute_advantage_loss(logits, targets)
            ce_loss = F.cross_entropy(logits.view(-1, V), targets.view(-1))
            total = ce_loss + aux_losses['ponder_loss'] + advantage_loss
            total.backward()
        
        Args:
            logits: Model output logits [batch, seq, vocab]
            targets: Target token ids [batch, seq]
            ignore_index: Index to ignore in loss computation
        
        Returns:
            Scalar advantage loss to add to total loss.
        """
        B, L, V = logits.shape
        
        # Compute per-token CE losses (no reduction, no grad needed here)
        logits_flat = logits.view(-1, V)
        targets_flat = targets.view(-1)
        
        with torch.no_grad():
            token_losses = F.cross_entropy(
                logits_flat,
                targets_flat,
                ignore_index=ignore_index,
                reduction='none',
            ).view(B, L)
            
            valid_mask = (targets != ignore_index)
            token_losses = token_losses * valid_mask.float()

            # For loss-aware MoD teacher supervision: exclude ignored positions
            token_losses_teacher = token_losses.masked_fill(~valid_mask, float('-inf'))
        
        # Update baseline EMA
        self.loss_baseline.update(token_losses[valid_mask])
        
        # Collect advantage loss from all blocks
        total_advantage_loss = self._zero_scalar
        for layer in self.layers:
            if hasattr(layer, 'compute_advantage_loss'):
                layer_adv = layer.compute_advantage_loss(token_losses, self.loss_baseline)
                total_advantage_loss = total_advantage_loss + layer_adv

            # Optional loss-aware MoD supervision (router logits -> top-k hard tokens)
            mod_wrap = getattr(layer, "mod_mlp_wrapper", None)
            if mod_wrap is not None and hasattr(mod_wrap, "compute_loss_aware_loss"):
                total_advantage_loss = total_advantage_loss + mod_wrap.compute_loss_aware_loss(token_losses_teacher)
        
        # Store diagnostics
        self._last_token_losses = token_losses.detach()
        self._last_baseline_value = self.loss_baseline.baseline
        self._last_advantage_loss = total_advantage_loss.detach()
        
        return total_advantage_loss

    def compute_advantage_loss_from_token_losses(
        self,
        token_losses: torch.Tensor,
        targets: torch.Tensor,
        ignore_index: int = -100,
    ) -> torch.Tensor:
        """Compute routing losses given precomputed per-token CE losses.

        This supports training modes where logits aren't materialized (e.g.
        chunked CE), while still training MoR (advantage) and optional
        loss-aware MoD supervision.
        """
        valid_mask = (targets != ignore_index)
        # Ensure ignored positions don't affect baseline/teacher
        token_losses = token_losses * valid_mask.float()
        token_losses_teacher = token_losses.masked_fill(~valid_mask, float("-inf"))

        # Update baseline EMA
        self.loss_baseline.update(token_losses[valid_mask])

        total_loss = self._zero_scalar
        for layer in self.layers:
            if hasattr(layer, "compute_advantage_loss"):
                total_loss = total_loss + layer.compute_advantage_loss(token_losses, self.loss_baseline)

            mod_wrap = getattr(layer, "mod_mlp_wrapper", None)
            if mod_wrap is not None and hasattr(mod_wrap, "compute_loss_aware_loss"):
                total_loss = total_loss + mod_wrap.compute_loss_aware_loss(token_losses_teacher)

        # Store diagnostics
        self._last_token_losses = token_losses.detach()
        self._last_baseline_value = self.loss_baseline.baseline
        self._last_advantage_loss = total_loss.detach()
        return total_loss

    def get_efficiency_losses(self) -> dict:
        """Get all efficiency-related losses (legacy method).
        
        Deprecated: Use get_aux_losses() instead.
        """
        aux = self.get_aux_losses()
        return {
            "ponder_loss": aux["mor_ponder_loss"],
            "aux_loss": aux["mod_aux_loss"],
        }

    def set_global_step(self, step: int):
        """Set global training step for MoR and MoD warmup scheduling.
        
        Propagates step to all MoR blocks (for depth distribution warmup)
        and all MoD MLP wrappers (for soft->hard routing curriculum).
        """
        self._global_step.fill_(step)
        # Cache as Python int for use in methods (avoids .item() graph break)
        self._cached_global_step = step
        # Cache MoR adaptive enabled flag
        self._mor_adaptive_cached = step >= getattr(self, '_mor_enable_step', 0)
        for layer in self.layers:
            # CCGQAMoRBlock - has its own set_global_step that propagates to MoD MLP
            if hasattr(layer, 'set_global_step'):
                layer.set_global_step(step)

    @torch.compiler.disable
    def update_mod_loss_ema(self, loss_ema: float) -> None:
        """Propagate EMA loss to MoD wrappers to unlock MoD after threshold."""
        for layer in self.layers:
            mod_wrap = getattr(layer, "mod_mlp_wrapper", None)
            if mod_wrap is None:
                continue
            if hasattr(mod_wrap, "update_loss_ema"):
                mod_wrap.update_loss_ema(loss_ema)
    
    def set_mor_curriculum(self, enable_step: int, rampup_steps: int = 1000):
        """Configure MoR depth curriculum.
        
        This enables phased training where:
        1. Steps 0 to enable_step: Fixed-depth MoR (all tokens through all recursions)
           - MoD is still active for token skipping
           - Model learns base representations without routing overhead
        2. Steps enable_step to enable_step+rampup_steps: Gradual MoR rampup
           - Adaptive routing turns on with scaled aux losses
        3. Steps after rampup: Full MoR adaptive routing
        
        Args:
            enable_step: Step at which to enable MoR adaptive routing.
                        Recommended: 25-35% of total training steps.
                        Set to 0 for immediate MoR (legacy behavior).
            rampup_steps: Steps to ramp up MoR after enabling (default 1000).
        
        Example for 90K step run with 30% curriculum:
            model.set_mor_curriculum(enable_step=27000, rampup_steps=1000)
        """
        self._mor_enable_step = enable_step
        self._mor_rampup_steps = rampup_steps
        
        for layer in self.layers:
            # CCGQAMoRBlock has set_mor_enable_step directly
            if hasattr(layer, 'set_mor_enable_step'):
                layer.set_mor_enable_step(enable_step, rampup_steps)
    
    def is_mor_adaptive_enabled(self) -> bool:
        """Check if MoR adaptive routing is currently enabled."""
        # Use cached value from set_global_step (no step-dependent branching)
        return getattr(self, '_mor_adaptive_cached', False)
    
    def get_mor_status(self) -> dict:
        """Get MoR curriculum status for logging."""
        # Use cached step from set_global_step (Python int, no .item() needed)
        global_step = getattr(self, '_cached_global_step', 0)
        enable_step = getattr(self, '_mor_enable_step', 0)
        rampup_steps = getattr(self, '_mor_rampup_steps', 1000)
        
        if global_step < enable_step:
            phase = "fixed-depth"
            progress = global_step / max(1, enable_step)
        elif global_step < enable_step + rampup_steps:
            phase = "rampup"
            progress = (global_step - enable_step) / max(1, rampup_steps)
        else:
            phase = "full-adaptive"
            progress = 1.0
        
        return {
            "phase": phase,
            "global_step": global_step,
            "enable_step": enable_step,
            "rampup_steps": rampup_steps,
            "rampup_progress": progress,
            "mor_enabled": global_step >= enable_step,
        }

    @torch.compiler.disable
    def get_routing_stats(self) -> dict:
        """Get routing statistics from all MoD/MoR layers (compile-disabled).

        Returns:
            dict with:
                - mod_layers: List of MoD probs_mean per MLP wrapper
                - mor_layers: List of MoR stats (avg_depth, depth_histogram) per MoR block
                - summary: Aggregate stats

        Example usage in training loop:
            if step % 500 == 0:
                stats = model.get_routing_stats()
                print(f"MoD probs: {stats['mod_layers']}")
                print(f"MoR depths: {[s['avg_depth'] for s in stats['mor_layers']]}")
        """
        # Collect MoD stats from MoD MLP wrappers inside CCGQAMoRBlocks
        mod_stats = []
        for i, layer in enumerate(self.layers):
            if isinstance(layer, CCGQAMoRBlock) and layer.mod_mlp_wrapper is not None:
                mod_stats.append({"layer": i, **layer.mod_mlp_wrapper.get_routing_stats()})

        # MoR stats from CCGQAMoRBlock directly
        mor_stats = []
        for i, layer in enumerate(self.layers):
            if isinstance(layer, CCGQAMoRBlock):
                mor_stats.append({"layer": i, **layer.get_routing_stats()})

        # Compute summary
        summary = {}
        if mod_stats:
            probs = [s.get("probs_mean", 0) for s in mod_stats]
            summary["mod_probs_mean"] = sum(probs) / len(probs) if probs else 0
        if mor_stats:
            avg_depths = [s.get("avg_depth", 0) for s in mor_stats if "avg_depth" in s]
            if avg_depths:
                summary["mor_avg_depth"] = sum(avg_depths) / len(avg_depths)

        return {
            "mod_layers": mod_stats,
            "mor_layers": mor_stats,
            "summary": summary,
        }


class CCGQAMoDBlockWrapper(nn.Module):
    """Wrapper to add REAL MoD routing around any block.
    
    Uses MoDRouter from mixture_of_depths.py for compute-skipping
    via gather/scatter pattern.
    
    TWO-PHASE ROUTING CURRICULUM (controlled by global step):
    
    Phase 1 (step < warmup_steps): SOFT ROUTING
    - All tokens computed through the block
    - Output weighted by router probabilities (soft gate)
    - Purpose: stable router learning before hard decisions
    - NO compute savings
    
    Phase 2 (step >= warmup_steps): HARD ROUTING  
    - Top-k tokens selected by router
    - Gather only selected tokens, process through block
    - Scatter results back, skipped tokens use identity
    - REAL compute savings (only k tokens processed)
    
    Switch controlled by set_global_step().
    """

    def __init__(
        self,
        block: nn.Module,
        dim: int,
        capacity_ratio: float = 0.5,
        aux_loss_weight: float = 0.01,
        warmup_steps: int = 100,  # Curriculum warmup
    ):
        super().__init__()
        self.block = block
        self.capacity_ratio = capacity_ratio
        self.aux_loss_weight = aux_loss_weight
        self.warmup_steps = warmup_steps
        # Use tensor buffer for global_step to avoid torch.compile recompilation
        self.register_buffer("_global_step", torch.zeros((), dtype=torch.int64), persistent=False)
        # OPTIMIZATION: Persistent zero scalar to avoid graph breaks and allocations
        self.register_buffer("_zero_scalar", torch.tensor(0.0), persistent=False)
        self._last_probs_mean_t: torch.Tensor = self._zero_scalar
        self._last_probs_std_t: torch.Tensor = self._zero_scalar

        # =================================================================
        # VALIDATION: MoDRouter is instantiated (not a custom router)
        # MoDRouter provides: mask, indices, scores for gather/scatter
        # =================================================================
        self.mod_router = MoDRouter(
            dim=dim,
            capacity_ratio=capacity_ratio,
            aux_loss_weight=aux_loss_weight,
        )
        # Assert MoDRouter is correctly instantiated
        assert isinstance(self.mod_router, MoDRouter), \
            f"mod_router must be MoDRouter, got {type(self.mod_router)}"
        
        # Store aux_loss as regular attribute, not buffer (avoids CUDAGraph issues)
        self._aux_loss: torch.Tensor = self._zero_scalar
        self._last_probs_mean: float = 0.0  # For logging
        self._last_probs_std: float = 0.0
        self._routing_mode: str = "soft"  # Track current mode for diagnostics
    
    def set_global_step(self, step: int):
        """Set global training step for curriculum scheduling."""
        self._global_step.fill_(step)
        # Cache routing mode decision to avoid .item() in forward (prevents graph break)
        self._use_hard_routing = step >= self.warmup_steps

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Fast forward without computing losses (for inference)."""
        if self.training:
            output, _ = self.forward_with_losses(x, **kwargs)
            return output
        else:
            # Inference: use MoDRouter for hard routing (maximum speed)
            B, L, D = x.shape
            mask, indices, _ = self.mod_router(x)  # mask: [B, L], indices: [B, k]
            k = indices.shape[1]

            if k == L:
                return self.block(x, **kwargs)
            
            if k == 0:
                return torch.zeros_like(x)

            # Gather selected tokens, process, scatter back
            indices_expanded = indices.unsqueeze(-1).expand(-1, -1, D)
            x_selected = torch.gather(x, 1, indices_expanded)  # [B, k, D]
            out_selected = self.block(x_selected, **kwargs)  # [B, k, D]
            output = torch.zeros_like(x)
            # Ensure dtype consistency for scatter_ (handles autocast edge cases)
            output.scatter_(1, indices_expanded, out_selected.to(output.dtype))
            return output

    def forward_with_losses(
        self, x: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, dict]:
        """Forward pass with two-phase routing curriculum.

        Returns (output, {"aux_loss": tensor, ...})
        
        PHASE 1 (warmup, step < warmup_steps):
        - Soft routing: all tokens computed, output weighted by router probs
        - Purpose: stable router learning before hard decisions
        - NO compute savings (all tokens processed)
        
        PHASE 2 (hard routing, step >= warmup_steps):
        - Hard top-k routing via gather/scatter
        - REAL compute skipping: only k tokens pass through block
        - Skipped tokens use identity residual
        
        Switch controlled by global step counter (set via set_global_step).
        Uses MoDRouter from mixture_of_depths.py.
        """
        B, L, D = x.shape

        # Get routing decision from MoDRouter
        mask, indices, scores = self.mod_router(x, return_scores=True)
        k = indices.shape[1]
        
        # Get soft probs for STE gradient flow and Phase 1
        probs = torch.sigmoid(scores)  # [B, L] in [0, 1]
        
        # Determine phase based on cached routing mode (set by set_global_step)
        use_hard_routing = getattr(self, '_use_hard_routing', False)
        self._routing_mode = "hard" if use_hard_routing else "soft"
        
        # Track compute savings for diagnostics
        self._tokens_total = L
        
        if k >= L:
            # Process all tokens - no routing needed (fallback)
            if hasattr(self.block, "forward_with_losses"):
                block_out, inner_losses = self.block.forward_with_losses(x, **kwargs)
            else:
                block_out = self.block(x, **kwargs)
                inner_losses = {}
            output = block_out
            self._tokens_processed = L
        elif use_hard_routing:
            # =============================================================
            # PHASE 2: HARD ROUTING with REAL compute skipping
            # VALIDATION: This uses gather/scatter - NOT soft gating
            # - torch.gather extracts ONLY k tokens (not all L)
            # - block() receives [B, k, D] NOT [B, L, D]
            # - torch.scatter_ places results back
            # - Skipped tokens (L-k) NEVER enter the block
            # =============================================================
            
            # GATHER: Extract only top-k tokens selected by MoDRouter
            indices_expanded = indices.unsqueeze(-1).expand(-1, -1, D)  # [B, k, D]
            x_selected = torch.gather(x, 1, indices_expanded)  # [B, k, D]
            
            # VALIDATE: Block receives k tokens, NOT L tokens
            assert x_selected.shape[1] == k, \
                f"gather failed: expected {k} tokens, got {x_selected.shape[1]}"
            assert x_selected.shape[1] < L, \
                f"No compute savings: k={k} >= L={L}"
            
            # PROCESS: Block runs on ONLY k tokens (real FLOPs savings!)
            if hasattr(self.block, "forward_with_losses"):
                block_out_selected, inner_losses = self.block.forward_with_losses(
                    x_selected, **kwargs
                )  # [B, k, D]
            else:
                block_out_selected = self.block(x_selected, **kwargs)  # [B, k, D]
                inner_losses = {}
            
            # SCATTER: Place processed tokens back, skipped use identity
            output = torch.zeros_like(x)  # Start with zeros (skipped tokens are zero)
            # Ensure dtype consistency for scatter_ (handles autocast edge cases)
            output.scatter_(1, indices_expanded, block_out_selected.to(output.dtype))
            
            # STE for router gradients (zero in forward, gradient path in backward)
            selected_probs = torch.gather(probs, 1, indices)  # [B, k]
            ste_grad_path = (selected_probs.sum() - selected_probs.sum().detach()) * 0.0
            output = output + ste_grad_path
            
            self._tokens_processed = k  # Only k tokens computed
        else:
            # === PHASE 1: SOFT ROUTING for router warmup ===
            # All tokens computed, output weighted by soft router probabilities
            # This provides smooth gradients for stable router learning
            # NOTE: No compute savings in Phase 1 - this is intentional for warmup
            
            # Process ALL tokens (no compute savings during warmup)
            if hasattr(self.block, "forward_with_losses"):
                block_out, inner_losses = self.block.forward_with_losses(x, **kwargs)
            else:
                block_out = self.block(x, **kwargs)
                inner_losses = {}
            
            # Soft weighted output: high prob -> block output, low prob -> zero
            gate = probs.unsqueeze(-1)  # [B, L, 1]
            output = gate * block_out
            
            self._tokens_processed = L  # All tokens computed in Phase 1

        # Compute aux_loss - use MoDRouter's built-in aux loss
        if self.training and self.aux_loss_weight > 0:
            # Get aux_loss from MoDRouter (already computed during forward)
            aux_loss = self.mod_router.get_aux_loss()
            self._aux_loss = aux_loss.detach()
            self._last_probs_mean_t = probs.mean().detach()
            #self._last_probs_mean = probs.mean().item()
            self._last_probs_std_t  = probs.std().detach()
            #self._last_probs_std = probs.std().item()
        else:
            aux_loss = self._zero_scalar

        inner_losses["aux_loss"] = aux_loss
        return output, inner_losses

    def get_aux_loss(self) -> torch.Tensor:
        """Returns the aux_loss from the MoDRouter."""
        return self.mod_router.get_aux_loss()

    @torch.compiler.disable
    def get_routing_stats(self) -> dict:
        """Get MoD routing statistics for debugging (compile-disabled)."""
        tokens_processed = getattr(self, "_tokens_processed", 0)
        tokens_total = getattr(self, "_tokens_total", 1)
        mean_t = getattr(self, "_last_probs_mean_t", None)
        std_t  = getattr(self, "_last_probs_std_t", None)
        compute_ratio = tokens_processed / max(1, tokens_total)
        return {
            "probs_mean": float(mean_t.item()) if mean_t is not None else 0.0,
            "probs_std":  float(std_t.item())  if std_t  is not None else 0.0,
            "target_capacity": self.capacity_ratio,
            "tokens_processed": tokens_processed,
            "tokens_total": tokens_total,
            "compute_ratio": compute_ratio,
            "compute_savings_pct": (1.0 - compute_ratio) * 100,
            "routing_mode": getattr(self, "_routing_mode", "unknown"),
            "global_step": int(self._global_step),
            "warmup_steps": self.warmup_steps,
        }


def create_ccgqa_mod_mor_model(
    vocab_size: int = 50257,
    dim: int = 2048,
    n_mor_blocks: int = 8,
    recursions_per_block: int = 4,
    n_heads: int = 32,
    n_kv_heads: int = 4,
    compression_factor: int = 4,
    mlp_ratio: float = 2.67,
    max_seq_len: int = 8192,
    mod_capacity: float = 0.5,
    aux_loss_weight: float = None,  # None = auto-scale based on depth
    adaptive: bool = True,
    hybrid_attention: bool = True,
    mod_mlp_warmup: int = 100,
    mor_warmup: int = 1000,
    # Dimension-aware MoR depth scaling (optional)
    dim_ref: int = 768,  # Reference dimension for scale=1.0
    depth_alpha: float = 0.0,  # Power-law exponent (0=disabled)
    depth_scale_max: float = 2.0,  # Maximum scaling factor
) -> CCGQAMoDMoRModel:
    """
    Create CCGQA + MoD + MoR model with specified parameters.

    Default config targets a paper-style baseline with:
    - dim=2048, 32 heads, 4 kv heads
    - 8 MoR blocks x 4 recursions = 32 effective layers
    - 50% MoD capacity (middle blocks)
    - 4x attention compression (CCGQA)
    - mlp_ratio=2.67 (common SwiGLU setting for param parity vs 4x GELU MLP)

    aux_loss_weight: Controls MoD capacity regularization strength.
        None (default): Auto-scales based on effective depth.
        0.01 * (effective_layers / 32) is used as baseline.
        
    dim_ref, depth_alpha, depth_scale_max: Dimension-aware MoR depth scaling.
        Larger models (dim > dim_ref) get scaled up target recursion depths
        when depth_alpha > 0. Formula: scale = (dim/dim_ref)^alpha, clamped to max.
        Default depth_alpha=0.0 disables this (backward compatible).
    """
    return CCGQAMoDMoRModel(
        vocab_size=vocab_size,
        dim=dim,
        n_mor_blocks=n_mor_blocks,
        recursions_per_block=recursions_per_block,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        compression_factor=compression_factor,
        mlp_ratio=mlp_ratio,
        max_seq_len=max_seq_len,
        mod_capacity=mod_capacity,
        aux_loss_weight=aux_loss_weight,
        adaptive=adaptive,
        hybrid_attention=hybrid_attention,
        mod_mlp_warmup=mod_mlp_warmup,
        mor_warmup=mor_warmup,
        dim_ref=dim_ref,
        depth_alpha=depth_alpha,
        depth_scale_max=depth_scale_max,
    )




# Note: Testing and benchmarking code moved to:
# - tests/test_ccgqa_model.py - Model validation tests (pytest)
# - diagnostics/benchmark_ccgqa.py - Performance benchmarks
# - hydra.utils.save_model_architecture() - Architecture export utility
#
# Run tests with: pytest tests/test_ccgqa_model.py -v
# Run benchmarks with: python -m diagnostics.benchmark_ccgqa
