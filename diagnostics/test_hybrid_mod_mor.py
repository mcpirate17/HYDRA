"""
Test Hybrid Attention with MoD and MoR Integration

This module tests the hybrid attention stack (MQA + CCQA + MLA) with:
1. MoD (Mixture-of-Depths): Token-level adaptive compute
2. MoR (Mixture-of-Recursions): Adaptive depth via recursive application

The integration strategy:
- Apply MoD routing at the macro-block level (which tokens skip)
- Apply MoR to CCQA layers (which benefit from recursion)
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from hydra.model.hybrid_attention import (
    RMSNorm,
    MQAAttention,
    CCQAAttention,
    MLAAttention,
    HybridAttentionBlock,
    MacroBlock,
    HybridTransformer,
    HybridTransformerConfig,
    SwiGLUMLP,
    AttentionType,
    DEFAULT_MACRO_PATTERN,
)


# =============================================================================
# MoD Router for Hybrid Blocks
# =============================================================================


class MoDRouter(nn.Module):
    """
    Mixture-of-Depths Router.

    Determines which tokens should be processed vs skipped.
    During training, uses soft gating for differentiability.
    During inference, uses hard top-k selection for efficiency.
    """

    def __init__(
        self,
        dim: int,
        capacity_ratio: float = 0.75,
        aux_loss_weight: float = 0.01,
    ):
        super().__init__()
        self.capacity_ratio = capacity_ratio
        self.aux_loss_weight = aux_loss_weight
        self.router = nn.Linear(dim, 1, bias=False)
        self._aux_loss = torch.tensor(0.0)

    def forward(
        self,
        x: torch.Tensor,
        block_fn,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply MoD routing to a block.

        Args:
            x: Input tensor [B, L, D]
            block_fn: Function that processes the block
            mask: Optional attention mask

        Returns:
            Output tensor [B, L, D]
        """
        B, L, D = x.shape

        # Get router scores
        scores = self.router(x).squeeze(-1)  # [B, L]
        probs = torch.sigmoid(scores)  # [B, L] in [0, 1]

        if self.training:
            # Soft gating for differentiability
            block_out = block_fn(x, mask)
            gate = probs.unsqueeze(-1)  # [B, L, 1]
            output = gate * block_out + (1 - gate) * x

            # Auxiliary loss to maintain capacity
            self._aux_loss = self.aux_loss_weight * (
                probs.mean() - self.capacity_ratio
            ).pow(2)
        else:
            # Hard routing for inference efficiency
            k = max(1, int(L * self.capacity_ratio))
            if k >= L:
                return block_fn(x, mask)

            top_scores, top_indices = torch.topk(scores, k, dim=-1)
            indices_expanded = top_indices.unsqueeze(-1).expand(-1, -1, D)

            # Only process selected tokens
            selected = torch.gather(x, 1, indices_expanded)
            processed = block_fn(selected, None)  # No mask for selected subset

            output = x.clone()
            output.scatter_(1, indices_expanded, processed)

        return output

    def get_aux_loss(self) -> torch.Tensor:
        return self._aux_loss


# =============================================================================
# MoR Wrapper for Hybrid Blocks
# =============================================================================


class MoRWrapper(nn.Module):
    """
    Mixture-of-Recursions Wrapper.

    Applies a block recursively with adaptive halting based on token complexity.
    Uses Gaussian soft routing for differentiable depth selection.
    """

    def __init__(
        self,
        block: nn.Module,
        dim: int,
        max_recursions: int = 3,
        ponder_loss_weight: float = 0.01,
    ):
        super().__init__()
        self.block = block
        self.dim = dim
        self.max_recursions = max_recursions
        self.ponder_loss_weight = ponder_loss_weight

        # Router predicts target depth
        self.router = nn.Linear(dim, 1, bias=True)
        nn.init.normal_(self.router.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.router.bias, 0.5)  # Target middle depth

        # Recursion embeddings
        self.recursion_embed = nn.Embedding(max_recursions, dim)
        nn.init.normal_(self.recursion_embed.weight, std=0.02)

        # Recursion-specific biases
        self.recursion_bias = nn.Parameter(
            torch.zeros(max_recursions, 1, 1, dim) * 0.02
        )

        self.final_norm = RMSNorm(dim)
        self._ponder_loss = torch.tensor(0.0)

        # Pre-compute indices
        self.register_buffer(
            "_recursion_indices",
            torch.arange(max_recursions, dtype=torch.long),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply recursive processing with soft depth routing."""
        B, L, D = x.shape
        device = x.device

        # Compute target depth per token
        router_logits = self.router(x).squeeze(-1)  # [B, L]
        router_logits = torch.clamp(router_logits, min=-3.0, max=3.0)
        router_probs = torch.sigmoid(router_logits)  # [B, L]
        target_depth = router_probs * (self.max_recursions - 1)  # [B, L]

        # Process through all recursions
        recursion_outputs = []
        current = x
        for i in range(self.max_recursions):
            rec_bias = self.recursion_bias[i]
            rec_embed = self.recursion_embed(self._recursion_indices[i : i + 1])
            current = current + rec_bias + rec_embed.unsqueeze(0)
            current = self.block(current, mask)
            recursion_outputs.append(current)

        # Stack outputs: [B, L, max_rec, D]
        outputs_stack = torch.stack(recursion_outputs, dim=2)

        # Gaussian soft weighting
        temperature = 1.0
        levels = torch.arange(
            self.max_recursions, device=device, dtype=target_depth.dtype
        )

        # distances: [B, L, max_rec]
        distances = (target_depth.unsqueeze(-1) - levels.view(1, 1, -1)).pow(2)
        weights = F.softmax(-distances / temperature, dim=-1)  # [B, L, max_rec]

        # Weighted combination: [B, L, D]
        output = (outputs_stack * weights.unsqueeze(-1)).sum(dim=2)
        output = self.final_norm(output)

        # Ponder loss (regularize depth usage)
        if self.training:
            avg_depth = target_depth.mean()
            target_avg = self.max_recursions / 2
            self._ponder_loss = self.ponder_loss_weight * (avg_depth - target_avg).pow(
                2
            )

        return output

    def get_ponder_loss(self) -> torch.Tensor:
        return self._ponder_loss


# =============================================================================
# Hybrid Block with MoD
# =============================================================================


class HybridMoDBlock(nn.Module):
    """
    Hybrid Attention Block with MoD routing.

    Tokens are routed: high-importance tokens go through the block,
    low-importance tokens skip via residual.
    """

    def __init__(
        self,
        dim: int,
        attention_type: AttentionType,
        n_heads: int = 12,
        n_kv_heads: int = 3,
        compression_factor: int = 4,
        mlp_ratio: float = 3.5,
        max_seq_len: int = 8192,
        norm_eps: float = 1e-6,
        mod_capacity: float = 0.75,
        **attention_kwargs,
    ):
        super().__init__()

        # Inner block
        self.inner_block = HybridAttentionBlock(
            dim=dim,
            attention_type=attention_type,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            compression_factor=compression_factor,
            mlp_ratio=mlp_ratio,
            max_seq_len=max_seq_len,
            norm_eps=norm_eps,
            **attention_kwargs,
        )

        # MoD router
        self.mod_router = MoDRouter(
            dim=dim,
            capacity_ratio=mod_capacity,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.mod_router(x, self.inner_block, mask)

    def get_aux_loss(self) -> torch.Tensor:
        return self.mod_router.get_aux_loss()


# =============================================================================
# Hybrid Macro-Block with MoD + MoR
# =============================================================================


class HybridMoDMoRMacroBlock(nn.Module):
    """
    8-Layer Macro-Block with MoD and MoR integration.

    Strategy:
    - MQA layers: Use MoD only (fast path, skip low-importance tokens)
    - CCQA layers: Use MoR (recursive processing for complexity)
    - MLA layers: Plain processing (summarization, needs all tokens)
    """

    def __init__(
        self,
        dim: int,
        n_heads: int = 12,
        n_kv_heads: int = 3,
        compression_factor: int = 4,
        mlp_ratio: float = 3.5,
        max_seq_len: int = 8192,
        norm_eps: float = 1e-6,
        mod_capacity: float = 0.75,
        mor_max_recursions: int = 3,
        pattern: Optional[List[AttentionType]] = None,
        **attention_kwargs,
    ):
        super().__init__()

        if pattern is None:
            pattern = DEFAULT_MACRO_PATTERN

        self.blocks = nn.ModuleList()
        self._block_types = []  # Track types for loss aggregation

        for attn_type in pattern:
            if attn_type == AttentionType.MQA:
                # MQA with MoD routing
                block = HybridMoDBlock(
                    dim=dim,
                    attention_type=attn_type,
                    n_heads=n_heads,
                    n_kv_heads=n_kv_heads,
                    compression_factor=compression_factor,
                    mlp_ratio=mlp_ratio,
                    max_seq_len=max_seq_len,
                    norm_eps=norm_eps,
                    mod_capacity=mod_capacity,
                )
                self._block_types.append("mod")
            elif attn_type == AttentionType.CCQA:
                # CCQA with MoR (recursive processing)
                inner_block = HybridAttentionBlock(
                    dim=dim,
                    attention_type=attn_type,
                    n_heads=n_heads,
                    n_kv_heads=n_kv_heads,
                    compression_factor=compression_factor,
                    mlp_ratio=mlp_ratio,
                    max_seq_len=max_seq_len,
                    norm_eps=norm_eps,
                    **attention_kwargs,
                )
                block = MoRWrapper(
                    block=inner_block,
                    dim=dim,
                    max_recursions=mor_max_recursions,
                )
                self._block_types.append("mor")
            else:  # MLA
                # MLA without routing (needs all tokens for summarization)
                block = HybridAttentionBlock(
                    dim=dim,
                    attention_type=attn_type,
                    n_heads=n_heads,
                    n_kv_heads=n_kv_heads,
                    compression_factor=compression_factor,
                    mlp_ratio=mlp_ratio,
                    max_seq_len=max_seq_len,
                    norm_eps=norm_eps,
                    **attention_kwargs,
                )
                self._block_types.append("plain")

            self.blocks.append(block)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, mask=mask)
        return x

    def get_aux_losses(self) -> Dict[str, torch.Tensor]:
        """Get auxiliary losses from MoD and MoR blocks."""
        losses = {"mod": torch.tensor(0.0), "mor": torch.tensor(0.0)}

        for block, block_type in zip(self.blocks, self._block_types):
            if block_type == "mod" and hasattr(block, "get_aux_loss"):
                losses["mod"] = losses["mod"] + block.get_aux_loss()
            elif block_type == "mor" and hasattr(block, "get_ponder_loss"):
                losses["mor"] = losses["mor"] + block.get_ponder_loss()

        return losses


# =============================================================================
# Full Hybrid Transformer with MoD + MoR
# =============================================================================


@dataclass
class HybridMoDMoRConfig:
    """Configuration for HybridMoDMoRTransformer."""

    vocab_size: int = 50257
    dim: int = 768
    n_macro_blocks: int = 3
    n_heads: int = 12
    n_kv_heads: int = 3
    compression_factor: int = 4
    mlp_ratio: float = 3.5
    max_seq_len: int = 8192
    norm_eps: float = 1e-6
    tie_weights: bool = True

    # MoD settings
    enable_mod: bool = True
    mod_capacity: float = 0.75

    # MoR settings
    enable_mor: bool = True
    mor_max_recursions: int = 3


class HybridMoDMoRTransformer(nn.Module):
    """
    Hybrid Transformer with full MoD + MoR integration.

    Architecture:
    - 24 layers (3 × 8-layer macro-blocks)
    - MQA layers use MoD for token skipping
    - CCQA layers use MoR for adaptive recursion
    - MLA layers process all tokens (no routing)
    """

    def __init__(self, config: HybridMoDMoRConfig):
        super().__init__()
        self.config = config

        # Token embedding
        self.tok_emb = nn.Embedding(config.vocab_size, config.dim)

        # Macro-blocks with MoD/MoR
        self.macro_blocks = nn.ModuleList(
            [
                HybridMoDMoRMacroBlock(
                    dim=config.dim,
                    n_heads=config.n_heads,
                    n_kv_heads=config.n_kv_heads,
                    compression_factor=config.compression_factor,
                    mlp_ratio=config.mlp_ratio,
                    max_seq_len=config.max_seq_len,
                    norm_eps=config.norm_eps,
                    mod_capacity=config.mod_capacity,
                    mor_max_recursions=config.mor_max_recursions,
                )
                for _ in range(config.n_macro_blocks)
            ]
        )

        # Final norm
        self.final_norm = RMSNorm(config.dim, eps=config.norm_eps)

        # Output projection
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        # Initialize weights
        self._init_weights()

        # Weight tying
        if config.tie_weights:
            self.output.weight = self.tok_emb.weight

    def _init_weights(self):
        """Initialize weights with scaled init."""
        import math

        n_layers = self.config.n_macro_blocks * 8
        residual_scale = 1.0 / math.sqrt(2 * n_layers)

        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                if "o_proj" in name or "down" in name:
                    module.weight.data *= residual_scale
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Token embedding
        h = self.tok_emb(x)

        # Process through macro-blocks
        for macro_block in self.macro_blocks:
            h = macro_block(h, mask=mask)

        # Final norm and output
        h = self.final_norm(h)
        return self.output(h)

    def get_aux_losses(self) -> Dict[str, torch.Tensor]:
        """Aggregate auxiliary losses from all macro-blocks."""
        total_mod = torch.tensor(0.0, device=next(self.parameters()).device)
        total_mor = torch.tensor(0.0, device=next(self.parameters()).device)

        for macro_block in self.macro_blocks:
            losses = macro_block.get_aux_losses()
            total_mod = total_mod + losses["mod"]
            total_mor = total_mor + losses["mor"]

        return {"mod": total_mod, "mor": total_mor, "total": total_mod + total_mor}

    def count_parameters(self) -> Dict[str, int]:
        """Count parameters by component."""
        return {
            "total": sum(p.numel() for p in self.parameters()),
            "embedding": self.tok_emb.weight.numel(),
        }


# =============================================================================
# Test Functions
# =============================================================================


def test_mod_router():
    """Test MoD router independently."""
    print("\n=== Testing MoD Router ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dim = 768
    router = MoDRouter(dim=dim, capacity_ratio=0.75).to(device)

    x = torch.randn(4, 128, dim, device=device)

    # Test with a simple block
    block = lambda x, m: x * 2  # Simple doubling

    # Training mode
    router.train()
    out_train = router(x, block)
    print(f"Training output shape: {out_train.shape}")
    print(f"Aux loss: {router.get_aux_loss().item():.6f}")

    # Inference mode
    router.eval()
    with torch.no_grad():
        out_eval = router(x, block)
    print(f"Inference output shape: {out_eval.shape}")

    print("✓ MoD Router test passed")


def test_mor_wrapper():
    """Test MoR wrapper independently."""
    print("\n=== Testing MoR Wrapper ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dim = 768
    inner_block = HybridAttentionBlock(
        dim=dim,
        attention_type=AttentionType.CCQA,
        n_heads=12,
        n_kv_heads=3,
    ).to(device)

    wrapper = MoRWrapper(
        block=inner_block,
        dim=dim,
        max_recursions=3,
    ).to(device)

    x = torch.randn(4, 128, dim, device=device)

    wrapper.train()
    out = wrapper(x)
    print(f"Output shape: {out.shape}")
    print(f"Ponder loss: {wrapper.get_ponder_loss().item():.6f}")

    print("✓ MoR Wrapper test passed")


def test_hybrid_mod_mor_macro_block():
    """Test hybrid macro-block with MoD + MoR."""
    print("\n=== Testing Hybrid MoD+MoR Macro-Block ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    macro = HybridMoDMoRMacroBlock(
        dim=768,
        n_heads=12,
        n_kv_heads=3,
        mod_capacity=0.75,
        mor_max_recursions=3,
    ).to(device)

    x = torch.randn(4, 128, 768, device=device)

    macro.train()
    out = macro(x)
    losses = macro.get_aux_losses()

    print(f"Output shape: {out.shape}")
    print(f"MoD aux loss: {losses['mod'].item():.6f}")
    print(f"MoR ponder loss: {losses['mor'].item():.6f}")

    # Test gradient flow
    loss = out.mean()
    loss.backward()

    grad_count = sum(1 for p in macro.parameters() if p.grad is not None)
    print(f"Parameters with gradients: {grad_count}")

    print("✓ Hybrid MoD+MoR Macro-Block test passed")


def test_full_transformer():
    """Test full hybrid transformer with MoD + MoR."""
    print("\n=== Testing Full Hybrid MoD+MoR Transformer ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = HybridMoDMoRConfig(
        dim=768,
        n_macro_blocks=3,
        n_heads=12,
        n_kv_heads=3,
        mod_capacity=0.75,
        mor_max_recursions=3,
    )

    model = HybridMoDMoRTransformer(config).to(device)
    params = model.count_parameters()
    print(f"Total parameters: {params['total']:,} ({params['total'] / 1e6:.1f}M)")

    # Forward pass
    tokens = torch.randint(0, 50257, (4, 64), device=device)

    model.train()
    logits = model(tokens)
    losses = model.get_aux_losses()

    print(f"Logits shape: {logits.shape}")
    print(f"MoD aux loss: {losses['mod'].item():.6f}")
    print(f"MoR ponder loss: {losses['mor'].item():.6f}")
    print(f"Total aux loss: {losses['total'].item():.6f}")

    # Training step with gradient clipping
    target = torch.randn_like(logits)
    ce_loss = F.mse_loss(logits, target)
    total_loss = ce_loss + losses["total"]

    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    grad_norms = []
    for p in model.parameters():
        if p.grad is not None:
            grad_norms.append(p.grad.norm().item())

    print(f"Max gradient norm (after clipping): {max(grad_norms):.4f}")
    print(f"Mean gradient norm: {sum(grad_norms) / len(grad_norms):.4f}")

    print("✓ Full Hybrid MoD+MoR Transformer test passed")


def test_training_stability():
    """Test training stability over multiple steps."""
    print("\n=== Testing Training Stability ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = HybridMoDMoRConfig(
        dim=768,
        n_macro_blocks=3,
        mod_capacity=0.75,
        mor_max_recursions=3,
    )

    model = HybridMoDMoRTransformer(config).to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    losses = []
    for step in range(30):
        tokens = torch.randint(0, 50257, (4, 64), device=device)
        target = torch.randn(4, 64, 50257, device=device)

        optimizer.zero_grad()

        logits = model(tokens)
        aux_losses = model.get_aux_losses()

        ce_loss = F.mse_loss(logits, target)
        total_loss = ce_loss + aux_losses["total"]

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        losses.append(ce_loss.item())

        if (step + 1) % 10 == 0:
            print(
                f"Step {step + 1}: CE Loss={ce_loss.item():.4f}, "
                f"MoD={aux_losses['mod'].item():.6f}, "
                f"MoR={aux_losses['mor'].item():.6f}"
            )

    improvement = (losses[0] - losses[-1]) / losses[0] * 100
    print(f"\nInitial loss: {losses[0]:.4f}")
    print(f"Final loss: {losses[-1]:.4f}")
    print(f"Improvement: {improvement:.2f}%")

    if improvement > 0:
        print("✓ Training is stable and learning!")
    else:
        print("⚠ Training may be unstable - loss not decreasing")


def main():
    """Run all tests."""
    print("=" * 80)
    print("Hybrid Attention + MoD + MoR Integration Tests")
    print("=" * 80)

    test_mod_router()
    test_mor_wrapper()
    test_hybrid_mod_mor_macro_block()
    test_full_transformer()
    test_training_stability()

    print("\n" + "=" * 80)
    print("All tests completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
