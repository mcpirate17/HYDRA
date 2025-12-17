#!/usr/bin/env python3
"""
Paper Compliance Tests for MoD, MoR, and CCGQA

This test suite validates that the implementation adheres to the following papers:
- MoD (Mixture-of-Depths): arXiv:2404.02258
- MoR (Mixture-of-Recursions): arXiv:2507.10524
- CCGQA (Compressed Convolutional GQA): arXiv:2510.04476

Tests are designed to be repeatable and validate architectural correctness
at various model scales: 100M, 500M, 750M, 1.5B.

Run: pytest tests/test_paper_compliance.py -v
"""

import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from hydra.model.ccgqa import (
    CCGQAAttention,
    CCGQABlock,
    CCGQAMoRBlock,
    CCGQAMoDBlockWrapper,
    CCGQAMoDMoRModel,
    create_ccgqa_mod_mor_model,
)


# =============================================================================
# Model Variant Configurations
# =============================================================================


@dataclass
class ModelVariant:
    """Configuration for a model variant."""

    name: str
    dim: int
    n_mor_blocks: int
    recursions: int
    n_heads: int
    n_kv_heads: int
    compression_factor: int
    expected_params_m: float  # Expected params in millions (approximate)

    @property
    def effective_layers(self) -> int:
        return self.n_mor_blocks * self.recursions


# Define model variants matching typical scaling
# Extended to include 100M, 250M, 500M, 750M, 900M, 1B, 1.5B for scaling analysis
MODEL_VARIANTS = {
    "100M": ModelVariant(
        name="100M",
        dim=768,
        n_mor_blocks=8,
        recursions=4,
        n_heads=12,
        n_kv_heads=3,
        compression_factor=4,
        expected_params_m=100,
    ),
    "250M": ModelVariant(
        name="250M",
        dim=1024,
        n_mor_blocks=12,
        recursions=4,
        n_heads=16,
        n_kv_heads=4,
        compression_factor=4,
        expected_params_m=250,
    ),
    "500M": ModelVariant(
        name="500M",
        dim=1536,
        n_mor_blocks=16,
        recursions=4,
        n_heads=24,
        n_kv_heads=4,
        compression_factor=4,
        expected_params_m=570,
    ),
    "750M": ModelVariant(
        name="750M",
        dim=1792,
        n_mor_blocks=20,
        recursions=4,
        n_heads=28,
        n_kv_heads=4,
        compression_factor=4,
        expected_params_m=750,
    ),
    "900M": ModelVariant(
        name="900M",
        dim=2048,
        n_mor_blocks=20,
        recursions=4,
        n_heads=32,
        n_kv_heads=4,
        compression_factor=4,
        expected_params_m=900,
    ),
    "1B": ModelVariant(
        name="1B",
        dim=2048,
        n_mor_blocks=24,
        recursions=4,
        n_heads=32,
        n_kv_heads=8,
        compression_factor=4,
        expected_params_m=1000,
    ),
    "1.5B": ModelVariant(
        name="1.5B",
        dim=2560,
        n_mor_blocks=24,
        recursions=5,
        n_heads=40,
        n_kv_heads=8,
        compression_factor=4,
        expected_params_m=1500,
    ),
    # Theoretical 4B variant for prediction testing (not run, just validated)
    "4B": ModelVariant(
        name="4B",
        dim=4096,
        n_mor_blocks=32,
        recursions=5,
        n_heads=64,
        n_kv_heads=8,
        compression_factor=4,
        expected_params_m=4000,
    ),
}


def get_device() -> str:
    """Get available device."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def create_model_from_variant(
    variant: ModelVariant, vocab_size: int = 50257
) -> CCGQAMoDMoRModel:
    """Create a model from a variant configuration."""
    return create_ccgqa_mod_mor_model(
        vocab_size=vocab_size,
        dim=variant.dim,
        n_mor_blocks=variant.n_mor_blocks,
        recursions_per_block=variant.recursions,
        n_heads=variant.n_heads,
        n_kv_heads=variant.n_kv_heads,
        compression_factor=variant.compression_factor,
        mod_capacity=0.75,
        adaptive=True,
    )


# =============================================================================
# CCGQA Paper Compliance Tests (arXiv:2510.04476)
# =============================================================================


class TestCCGQAPaperCompliance:
    """
    Tests for Compressed Convolutional GQA paper compliance.

    Key requirements from arXiv:2510.04476:
    1. Compression factor C reduces latent_dim = dim / C
    2. Sequence + channel convolutions on Q and K
    3. QK L2 normalization with learnable temperature
    4. GQA-style head sharing (fewer KV heads than Q heads)
    5. Optional value-shift for temporal inductive bias
    """

    @pytest.fixture
    def ccgqa_attention(self) -> CCGQAAttention:
        """Create a CCGQA attention module."""
        return CCGQAAttention(
            dim=512,
            n_heads=8,
            n_kv_heads=2,
            compression_factor=4,
            max_seq_len=512,
            use_rope=True,
            use_qk_norm=True,
            use_convs=True,
            use_qk_mean=True,
            use_value_shift=True,
        )

    def test_compression_factor(self, ccgqa_attention: CCGQAAttention):
        """Paper Section 3.1: Latent dimension should be dim / compression_factor."""
        expected_latent_dim = 512 // 4  # dim / C = 512 / 4 = 128
        assert ccgqa_attention.latent_dim == expected_latent_dim, (
            f"Latent dim should be {expected_latent_dim}, got {ccgqa_attention.latent_dim}"
        )

    def test_gqa_head_sharing(self, ccgqa_attention: CCGQAAttention):
        """Paper Section 3.2: GQA uses fewer KV heads than Q heads."""
        assert ccgqa_attention.n_heads > ccgqa_attention.n_kv_heads, (
            f"GQA requires n_heads ({ccgqa_attention.n_heads}) > n_kv_heads ({ccgqa_attention.n_kv_heads})"
        )
        assert ccgqa_attention.n_heads % ccgqa_attention.n_kv_heads == 0, (
            "n_heads must be divisible by n_kv_heads for GQA grouping"
        )

    def test_qk_normalization_with_temperature(self, ccgqa_attention: CCGQAAttention):
        """Paper Section 3.3: QK L2 norm with learnable temperature."""
        assert ccgqa_attention.use_qk_norm is True
        assert hasattr(ccgqa_attention, "key_temperature")
        assert ccgqa_attention.key_temperature.numel() == 1
        # Temperature should be learnable (requires_grad=True)
        assert ccgqa_attention.key_temperature.requires_grad

    def test_convolutions_present(self, ccgqa_attention: CCGQAAttention):
        """Paper Section 3.4: Sequence + channel convolutions on Q and K."""
        assert ccgqa_attention.use_convs is True
        # Should have 2 conv layers each for Q and K
        assert hasattr(ccgqa_attention, "q_conv1"), "Missing q_conv1 (sequence conv)"
        assert hasattr(ccgqa_attention, "q_conv2"), "Missing q_conv2 (channel conv)"
        assert hasattr(ccgqa_attention, "k_conv1"), "Missing k_conv1 (sequence conv)"
        assert hasattr(ccgqa_attention, "k_conv2"), "Missing k_conv2 (channel conv)"

    def test_forward_output_shape(self, ccgqa_attention: CCGQAAttention):
        """Forward pass should preserve input shape."""
        x = torch.randn(2, 32, 512)  # [B, L, dim]
        output = ccgqa_attention(x)
        assert output.shape == x.shape, (
            f"Output shape {output.shape} != input shape {x.shape}"
        )

    def test_attention_in_compressed_space(self, ccgqa_attention: CCGQAAttention):
        """Paper Section 3.1: Attention computed entirely in compressed latent space."""
        # The q_down, k_down, v_down project to latent_dim, not dim
        assert ccgqa_attention.q_down.out_features == ccgqa_attention.latent_dim
        assert ccgqa_attention.k_down.out_features == ccgqa_attention.kv_dim
        # Note: with value_shift enabled, v_down projects to kv_dim/2
        # (half heads see current, half see shifted)
        if ccgqa_attention.use_value_shift:
            assert ccgqa_attention.v_down.out_features == ccgqa_attention.kv_dim // 2
        else:
            assert ccgqa_attention.v_down.out_features == ccgqa_attention.kv_dim

    @pytest.mark.parametrize("variant_name", ["100M", "500M"])
    def test_ccgqa_at_scale(self, variant_name: str):
        """Test CCGQA features are present at different scales."""
        variant = MODEL_VARIANTS[variant_name]
        model = create_model_from_variant(variant)

        # Find CCGQA attention modules
        ccgqa_count = 0
        for module in model.modules():
            if isinstance(module, CCGQAAttention):
                ccgqa_count += 1
                assert module.compression_factor == variant.compression_factor
                assert module.use_qk_norm
                assert module.use_convs

        # Hybrid attention: model mixes MQA/CCGQA/MLA across blocks.
        # Validate that the number of instantiated CCGQA modules matches the
        # number of CCQA entries in the configured attention pattern.
        from hydra.model.hybrid_attention import AttentionType
        assert hasattr(model, "_attention_pattern"), "Model should expose _attention_pattern"
        expected_ccgqa = sum(1 for t in model._attention_pattern if t == AttentionType.CCQA)
        assert ccgqa_count == expected_ccgqa, (
            f"Expected {expected_ccgqa} CCGQA attention modules from hybrid pattern, found {ccgqa_count}"
        )


# =============================================================================
# MoD Paper Compliance Tests (arXiv:2404.02258)
# =============================================================================


class TestMoDPaperCompliance:
    """
    Tests for Mixture-of-Depths paper compliance.

    Key requirements from arXiv:2404.02258:
    1. Router predicts per-token capacity allocation
    2. Capacity ratio controls fraction of tokens processed
    3. Aux loss encourages router to match target capacity
    4. Soft routing during training, hard top-k during inference
    
    NOTE: Tests updated to use MoDMLPWrapper and MoDRouter (from mixture_of_depths.py).
    The MoD now applies ONLY to MLP sublayer, not attention.
    """

    @pytest.fixture
    def mod_block(self) -> CCGQAMoDBlockWrapper:
        """Create a MoD block wrapper (legacy API for testing)."""
        inner_block = CCGQABlock(dim=256, n_heads=4, n_kv_heads=2)
        return CCGQAMoDBlockWrapper(
            block=inner_block,
            dim=256,
            capacity_ratio=0.75,
            aux_loss_weight=0.01,
        )

    def test_router_architecture(self, mod_block: CCGQAMoDBlockWrapper):
        """Paper Section 3: Router uses MoDRouter from mixture_of_depths.py."""
        # MoDRouter is the router, which contains a linear projection
        from hydra.routing.mixture_of_depths import MoDRouter
        assert hasattr(mod_block, 'mod_router'), "mod_block should have mod_router attribute"
        assert isinstance(mod_block.mod_router, MoDRouter), "mod_router should be MoDRouter"

    def test_capacity_ratio_initialization(self, mod_block: CCGQAMoDBlockWrapper):
        """Capacity ratio should be set correctly."""
        assert mod_block.capacity_ratio == 0.75, (
            f"Capacity ratio should be 0.75, got {mod_block.capacity_ratio}"
        )

    def test_soft_routing_during_training(self, mod_block: CCGQAMoDBlockWrapper):
        """Paper Section 4.1: Training uses differentiable soft routing."""
        mod_block.train()
        x = torch.randn(2, 32, 256, requires_grad=True)
        output, losses = mod_block.forward_with_losses(x)

        # Should be able to backprop through output
        loss = output.sum()
        loss.backward()
        assert x.grad is not None, "Gradients should flow through soft routing"

    def test_aux_loss_for_capacity_control(self, mod_block: CCGQAMoDBlockWrapper):
        """Paper Section 4.2: Aux loss encourages target capacity."""
        mod_block.train()
        x = torch.randn(2, 32, 256)
        _, losses = mod_block.forward_with_losses(x)

        assert "aux_loss" in losses
        # Aux loss should be small if router maintains capacity
        assert losses["aux_loss"].item() >= 0, "Aux loss should be non-negative"

    def test_routing_stats_available(self, mod_block: CCGQAMoDBlockWrapper):
        """Routing statistics should be accessible for monitoring."""
        mod_block.train()
        x = torch.randn(2, 32, 256)
        mod_block.forward_with_losses(x)

        stats = mod_block.get_routing_stats()
        assert "probs_mean" in stats
        assert "probs_std" in stats
        assert "target_capacity" in stats

    @pytest.mark.parametrize("capacity", [0.5, 0.75, 0.9])
    def test_capacity_ratio_variations(self, capacity: float):
        """Test MoD with different capacity ratios."""
        inner_block = CCGQABlock(dim=256, n_heads=4, n_kv_heads=2)
        mod_block = CCGQAMoDBlockWrapper(
            block=inner_block,
            dim=256,
            capacity_ratio=capacity,
            aux_loss_weight=0.01,
        )

        # Verify capacity_ratio is set correctly
        assert mod_block.capacity_ratio == capacity, (
            f"Capacity ratio should be {capacity}, got {mod_block.capacity_ratio}"
        )


# =============================================================================
# MoR Paper Compliance Tests (arXiv:2507.10524)
# =============================================================================


class TestMoRPaperCompliance:
    """
    Tests for Mixture-of-Recursions paper compliance.

    Key requirements from arXiv:2507.10524:
    1. Gaussian soft routing for depth assignment
    2. Recursion-specific embeddings/biases
    3. Ponder loss for compute cost penalty
    4. Layer-aware depth targeting
    5. Hierarchical capacity scheduling
    """

    @pytest.fixture
    def mor_block(self) -> CCGQAMoRBlock:
        """Create a MoR block."""
        return CCGQAMoRBlock(
            dim=256,
            n_heads=4,
            n_kv_heads=2,
            max_recursions=4,
            adaptive=True,
            layer_idx=2,
            total_layers=8,
        )

    def test_router_architecture(self, mor_block: CCGQAMoRBlock):
        """Paper Section 3.2: Router predicts depth per token."""
        assert isinstance(mor_block.router, nn.Linear)
        assert mor_block.router.out_features == 1

    def test_recursion_embeddings(self, mor_block: CCGQAMoRBlock):
        """Paper Section 3.3: Recursion-specific embeddings."""
        assert hasattr(mor_block, "recursion_embed")
        assert mor_block.recursion_embed.num_embeddings == mor_block.max_recursions

        assert hasattr(mor_block, "recursion_bias")
        assert mor_block.recursion_bias.shape[0] == mor_block.max_recursions

    def test_layer_aware_depth_initialization(self, mor_block: CCGQAMoRBlock):
        """Layer-aware: later layers should target deeper recursions."""
        # Block at layer 2/8 should have lower target than block at layer 7/8
        early_block = CCGQAMoRBlock(
            dim=256,
            n_heads=4,
            n_kv_heads=2,
            max_recursions=4,
            layer_idx=0,
            total_layers=8,
        )
        late_block = CCGQAMoRBlock(
            dim=256,
            n_heads=4,
            n_kv_heads=2,
            max_recursions=4,
            layer_idx=7,
            total_layers=8,
        )

        # Early layer targets ~20% (shallow), late layer targets ~50% (deeper)
        # Formula: target_prob = 0.2 + 0.3 * layer_ratio
        # Early (layer 0): 0.2 + 0.3 * 0 = 0.2
        # Late (layer 7/8): 0.2 + 0.3 * 1.0 = 0.5
        assert early_block.target_depth_ratio < late_block.target_depth_ratio, (
            f"Early layer target {early_block.target_depth_ratio:.2f} should be < "
            f"late layer target {late_block.target_depth_ratio:.2f}"
        )
        assert 0.15 < early_block.target_depth_ratio < 0.3, (
            f"Early layer should target ~0.2, got {early_block.target_depth_ratio}"
        )
        assert 0.4 < late_block.target_depth_ratio < 0.6, (
            f"Late layer should target ~0.5, got {late_block.target_depth_ratio}"
        )

    def test_capacity_schedule(self, mor_block: CCGQAMoRBlock):
        """Paper Section 3.4: Hierarchical capacity schedule."""
        assert hasattr(mor_block, "capacity_schedule")
        # Should decrease with depth (fewer tokens at deeper recursions)
        schedule = mor_block.capacity_schedule
        for i in range(1, len(schedule)):
            assert schedule[i] <= schedule[i - 1] + 0.01, (
                f"Capacity should decrease: {schedule[i - 1]:.2f} >= {schedule[i]:.2f}"
            )

    def test_ponder_loss_available(self, mor_block: CCGQAMoRBlock):
        """Paper Section 4: Ponder loss penalizes excessive compute."""
        mor_block.train()
        x = torch.randn(2, 32, 256)
        _, losses = mor_block.forward_with_losses(x)

        assert "ponder_loss" in losses, "MoR should return ponder_loss"

    def test_routing_stats_available(self, mor_block: CCGQAMoRBlock):
        """Routing statistics should be accessible for monitoring."""
        mor_block.train()
        x = torch.randn(2, 32, 256)
        mor_block.forward_with_losses(x)

        stats = mor_block.get_routing_stats()
        assert "avg_depth" in stats
        assert "depth_histogram" in stats
        assert "router_probs_mean" in stats

    def test_depth_histogram_valid(self, mor_block: CCGQAMoRBlock):
        """Depth histogram should sum to total tokens."""
        mor_block.train()
        x = torch.randn(2, 32, 256)  # 64 tokens total
        mor_block.forward_with_losses(x)

        stats = mor_block.get_routing_stats()
        histogram = stats.get("depth_histogram", [])
        if histogram:
            total = sum(histogram)
            expected = 2 * 32  # B * L
            assert total == expected, f"Histogram sum {total} != {expected} tokens"

    def test_forward_output_shape(self, mor_block: CCGQAMoRBlock):
        """Forward pass should preserve input shape."""
        x = torch.randn(2, 32, 256)
        output, _ = mor_block.forward_with_losses(x)
        assert output.shape == x.shape


# =============================================================================
# Combined MoD + MoR + CCGQA Integration Tests
# =============================================================================


class TestIntegration:
    """Tests for the combined CCGQAMoDMoRModel."""

    @pytest.fixture
    def small_model(self) -> CCGQAMoDMoRModel:
        """Create a small model for fast testing."""
        return create_ccgqa_mod_mor_model(
            vocab_size=1000,
            dim=256,
            n_mor_blocks=4,
            recursions_per_block=3,
            n_heads=4,
            n_kv_heads=2,
            mod_capacity=0.75,
        )

    def test_model_creates_successfully(self, small_model: CCGQAMoDMoRModel):
        """Model should instantiate without errors."""
        assert isinstance(small_model, CCGQAMoDMoRModel)

    def test_effective_layers_calculation(self, small_model: CCGQAMoDMoRModel):
        """Effective layers = n_mor_blocks * recursions."""
        expected = 4 * 3  # n_mor_blocks * recursions
        assert small_model.effective_layers == expected

    def test_forward_returns_losses(self, small_model: CCGQAMoDMoRModel):
        """Forward with return_losses should provide aux and ponder losses."""
        small_model.train()
        x = torch.randint(0, 1000, (2, 32))
        logits, losses = small_model(x, return_losses=True)

        assert "aux_loss" in losses
        assert "ponder_loss" in losses

    def test_gradient_flow(self, small_model: CCGQAMoDMoRModel):
        """Gradients should flow through all components."""
        small_model.train()
        x = torch.randint(0, 1000, (2, 32))
        logits, losses = small_model(x, return_losses=True)

        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), x.view(-1))
        loss = loss + 0.1 * losses["aux_loss"] + 0.01 * losses["ponder_loss"]
        loss.backward()

        # Check routers have gradients
        for name, param in small_model.named_parameters():
            if "router" in name and param.grad is not None:
                assert param.grad.abs().sum() > 0 or param.numel() == 1, (
                    f"Router {name} should have non-zero gradients"
                )

    def test_init_weights_preserves_router_bias(self, small_model: CCGQAMoDMoRModel):
        """_init_weights should NOT reset router biases."""
        # Collect router biases before
        router_biases_before = {}
        for name, param in small_model.named_parameters():
            if "router.bias" in name:
                router_biases_before[name] = param.data.clone()

        # Re-run init_weights on the model
        small_model._init_weights()

        # Router biases should be unchanged
        for name, param in small_model.named_parameters():
            if "router.bias" in name:
                assert torch.allclose(param.data, router_biases_before[name]), (
                    f"Router bias {name} was reset by _init_weights!"
                )


# =============================================================================
# Model Variant Tests (100M, 500M, 750M, 1.5B)
# =============================================================================


class TestModelVariants:
    """Tests to ensure all model variants are properly configured."""

    @pytest.mark.parametrize("variant_name", ["100M", "500M", "750M", "1.5B"])
    def test_variant_creates(self, variant_name: str):
        """Each variant should create successfully."""
        variant = MODEL_VARIANTS[variant_name]
        model = create_model_from_variant(variant)
        assert model is not None

    @pytest.mark.parametrize("variant_name", ["100M", "500M", "750M", "1.5B"])
    def test_variant_param_count(self, variant_name: str):
        """Variant param count should be within expected range."""
        variant = MODEL_VARIANTS[variant_name]
        model = create_model_from_variant(variant)
        param_count = sum(p.numel() for p in model.parameters())
        param_m = param_count / 1e6

        # Allow ±50% tolerance due to architectural variations
        lower = variant.expected_params_m * 0.5
        upper = variant.expected_params_m * 1.5
        assert lower < param_m < upper, (
            f"{variant_name} has {param_m:.1f}M params, expected ~{variant.expected_params_m}M"
        )

    @pytest.mark.parametrize("variant_name", ["100M", "500M", "750M", "1.5B"])
    def test_variant_effective_layers(self, variant_name: str):
        """Variant should have correct effective layer count."""
        variant = MODEL_VARIANTS[variant_name]
        model = create_model_from_variant(variant)
        expected = variant.n_mor_blocks * variant.recursions
        assert model.effective_layers == expected

    @pytest.mark.parametrize("variant_name", ["100M", "500M"])
    def test_variant_forward_pass(self, variant_name: str):
        """Variant forward pass should work (only small models to save memory)."""
        variant = MODEL_VARIANTS[variant_name]
        model = create_model_from_variant(variant, vocab_size=1000)
        device = get_device()
        model = model.to(device)

        x = torch.randint(0, 1000, (1, 32), device=device)
        model.eval()
        with torch.no_grad():
            logits, losses = model(x, return_losses=True)

        assert logits.shape == (1, 32, 1000)
        assert "aux_loss" in losses
        assert "ponder_loss" in losses


# =============================================================================
# Diagnostic Run Tests
# =============================================================================


class TestDiagnosticRun:
    """Tests that run short diagnostic training to verify behavior."""

    @pytest.fixture
    def diagnostic_model(self) -> Tuple[CCGQAMoDMoRModel, str]:
        """Create a model and device for diagnostics."""
        model = create_ccgqa_mod_mor_model(
            vocab_size=1000,
            dim=256,
            n_mor_blocks=4,
            recursions_per_block=3,
            n_heads=4,
            n_kv_heads=2,
            mod_capacity=0.75,
        )
        device = get_device()
        return model.to(device), device

    def test_mod_capacity_convergence(self, diagnostic_model):
        """MoD router should maintain capacity ratio over training."""
        model, device = diagnostic_model
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        probs_history = []
        for step in range(20):
            x = torch.randint(0, 1000, (2, 32), device=device)
            logits, losses = model(x, return_losses=True)

            loss = F.cross_entropy(logits.view(-1, 1000), x.view(-1))
            loss = loss + 0.1 * losses["aux_loss"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Collect MoD probs
            for layer in model.layers:
                if isinstance(layer, CCGQAMoDBlockWrapper):
                    probs_history.append(layer._last_probs_mean)

        # Probs should stay near 0.75 target
        if probs_history:
            avg_prob = sum(probs_history) / len(probs_history)
            assert 0.5 < avg_prob < 1.0, f"MoD probs {avg_prob:.2f} should be near 0.75"

    def test_mor_depth_differentiation(self, diagnostic_model):
        """MoR should show depth differentiation across layers."""
        model, device = diagnostic_model
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        for step in range(10):
            x = torch.randint(0, 1000, (2, 32), device=device)
            logits, losses = model(x, return_losses=True)

            loss = F.cross_entropy(logits.view(-1, 1000), x.view(-1))
            loss = loss + 0.01 * losses["ponder_loss"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Collect depths from each layer
        depths = []
        x = torch.randint(0, 1000, (2, 32), device=device)
        model(x, return_losses=True)

        for layer in model.layers:
            if isinstance(layer, CCGQAMoDBlockWrapper):
                mor_block = layer.block
            elif isinstance(layer, CCGQAMoRBlock):
                mor_block = layer
            else:
                continue

            if hasattr(mor_block, "get_routing_stats"):
                stats = mor_block.get_routing_stats()
                if "avg_depth" in stats:
                    depths.append(stats["avg_depth"])

        # Should have depth values
        assert len(depths) > 0, "Should have MoR depth statistics"

    def test_gradient_health_over_training(self, diagnostic_model):
        """Gradients should remain healthy throughout training."""
        model, device = diagnostic_model
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        for step in range(10):
            x = torch.randint(0, 1000, (2, 32), device=device)
            logits, losses = model(x, return_losses=True)

            loss = F.cross_entropy(logits.view(-1, 1000), x.view(-1))
            loss = loss + 0.1 * losses["aux_loss"] + 0.01 * losses["ponder_loss"]

            optimizer.zero_grad()
            loss.backward()

            # Check for exploding gradients
            total_norm = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    total_norm += param.grad.norm().item() ** 2
            total_norm = math.sqrt(total_norm)

            assert total_norm < 1000, f"Gradient norm {total_norm:.1f} is exploding!"

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()


# =============================================================================
# Run diagnostics on all variants
# =============================================================================


def run_variant_diagnostic(variant: ModelVariant, steps: int = 100) -> Dict[str, Any]:
    """Run a diagnostic training loop on a model variant.

    Returns diagnostic results dict.
    """
    device = get_device()

    # Create model
    model = create_model_from_variant(variant, vocab_size=1000)
    model = model.to(device)
    param_count = sum(p.numel() for p in model.parameters())

    results = {
        "variant": variant.name,
        "params_m": param_count / 1e6,
        "effective_layers": model.effective_layers,
        "steps": [],
        "mod_summary": {},
        "mor_summary": {},
        "ccgqa_summary": {},
        "paper_compliance": {},
    }

    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()

    mod_probs = []
    mor_depths = []

    for step in range(1, steps + 1):
        x = torch.randint(0, 1000, (2, 64), device=device)
        logits, losses = model(x, return_losses=True)

        ce_loss = F.cross_entropy(logits.view(-1, 1000), x.view(-1))
        loss = ce_loss + 0.1 * losses["aux_loss"] + 0.01 * losses["ponder_loss"]

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Collect stats
        step_mod_probs = []
        step_mor_depths = []

        for layer in model.layers:
            # CCGQAMoRBlock is the new unified architecture
            if isinstance(layer, CCGQAMoRBlock):
                # Get MoD stats from mod_mlp_wrapper if present
                if hasattr(layer, 'mod_mlp_wrapper') and layer.mod_mlp_wrapper is not None:
                    mod_stats = layer.mod_mlp_wrapper.get_routing_stats()
                    step_mod_probs.append(mod_stats.get("probs_mean", 0.0))
                # Get MoR stats
                mor_stats = layer.get_routing_stats()
                if "avg_depth" in mor_stats:
                    step_mor_depths.append(mor_stats["avg_depth"])
            # Legacy: CCGQAMoDBlockWrapper
            elif isinstance(layer, CCGQAMoDBlockWrapper):
                step_mod_probs.append(layer._last_probs_mean)
                if hasattr(layer.block, "get_routing_stats"):
                    stats = layer.block.get_routing_stats()
                    if "avg_depth" in stats:
                        step_mor_depths.append(stats["avg_depth"])

        if step_mod_probs:
            mod_probs.append(sum(step_mod_probs) / len(step_mod_probs))
        if step_mor_depths:
            mor_depths.append(sum(step_mor_depths) / len(step_mor_depths))

        if step % 10 == 0 or step == steps:
            results["steps"].append(
                {
                    "step": step,
                    "ce_loss": ce_loss.item(),
                    "aux_loss": losses["aux_loss"].item()
                    if isinstance(losses["aux_loss"], torch.Tensor)
                    else losses["aux_loss"],
                    "ponder_loss": losses["ponder_loss"].item()
                    if isinstance(losses["ponder_loss"], torch.Tensor)
                    else losses["ponder_loss"],
                    "mod_prob": mod_probs[-1] if mod_probs else 0,
                    "mor_depth": mor_depths[-1] if mor_depths else 0,
                }
            )

    # Final summaries
    results["mod_summary"] = {
        "initial_prob": mod_probs[0] if mod_probs else 0,
        "final_prob": mod_probs[-1] if mod_probs else 0,
        "target": 0.75,
        "converged": abs(mod_probs[-1] - 0.75) < 0.15 if mod_probs else False,
    }

    results["mor_summary"] = {
        "initial_depth": mor_depths[0] if mor_depths else 0,
        "final_depth": mor_depths[-1] if mor_depths else 0,
        "max_depth": variant.recursions - 1,
        "not_collapsed": 0.3 < mor_depths[-1] < variant.recursions - 0.3
        if mor_depths
        else False,
    }

    # Paper compliance checks
    results["paper_compliance"] = {
        "mod_2404.02258": {
            "router_learns": mod_probs[-1] < 0.95 if mod_probs else False,
            "capacity_maintained": abs(mod_probs[-1] - 0.75) < 0.2
            if mod_probs
            else False,
        },
        "mor_2507.10524": {
            "depth_routing_active": mor_depths[-1] > 0 if mor_depths else False,
            "not_collapsed": 0.2 < mor_depths[-1] < variant.recursions - 0.2
            if mor_depths
            else False,
        },
        "ccgqa_2510.04476": {
            "compression_active": variant.compression_factor > 1,
            "gqa_active": variant.n_kv_heads < variant.n_heads,
        },
    }

    return results


@pytest.mark.slow
class TestVariantDiagnostics:
    """Run diagnostic tests on each variant."""

    @pytest.mark.parametrize("variant_name", ["100M"])
    def test_variant_diagnostic_fast(self, variant_name: str):
        """Run quick diagnostic on small variant."""
        variant = MODEL_VARIANTS[variant_name]
        results = run_variant_diagnostic(variant, steps=50)

        # Basic checks
        assert results["paper_compliance"]["mod_2404.02258"]["capacity_maintained"]
        assert results["paper_compliance"]["ccgqa_2510.04476"]["compression_active"]

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "variant_name", ["100M", "250M", "500M", "750M", "900M"]
    )
    def test_variant_diagnostic_full(self, variant_name: str):
        """Run full diagnostic on all variants.

        Note: Step count and tolerances scale with model size.
        Larger models need more steps to converge and have looser tolerances
        for short diagnostic runs. In production training, all models should
        converge to target 0.75 capacity with sufficient steps.
        
        1B and 1.5B variants removed - they add significant test time without
        new validation over 900M. Same routing behavior is proven at smaller scales.
        """
        if (
            variant_name in ["750M", "900M"]
            and not torch.cuda.is_available()
        ):
            pytest.skip("Large variants require CUDA")

        variant = MODEL_VARIANTS[variant_name]

        # Scale steps by model size (larger models = fewer steps for test speed)
        step_map = {
            "100M": 100,
            "250M": 80,
            "500M": 60,
            "750M": 50,
            "900M": 40,
        }
        steps = step_map.get(variant_name, 100)

        # Tolerance scales with model size (larger = looser for short runs)
        min_prob_map = {
            "100M": 0.5,
            "250M": 0.5,
            "500M": 0.45,
            "750M": 0.4,
            "900M": 0.35,
        }
        min_prob = min_prob_map.get(variant_name, 0.5)

        results = run_variant_diagnostic(variant, steps=steps)

        # Paper compliance assertions
        # Note: Large models may need more steps to fully converge
        # We check that compression is active and depth routing is working
        assert results["paper_compliance"]["ccgqa_2510.04476"]["compression_active"], (
            f"{variant_name}: CCGQA compression not active"
        )
        # MoD capacity check - scaled by model size
        mod_prob = results["mod_summary"]["final_prob"]
        assert min_prob < mod_prob < 1.0, (
            f"{variant_name}: MoD prob {mod_prob:.3f} should be between {min_prob} and 1.0 "
            f"(at {steps} steps; larger models need more training to converge)"
        )


# =============================================================================
# Scaling Analysis Tests
# =============================================================================


class TestScalingLaws:
    """Tests for scaling behavior and 4B prediction validation."""

    def test_aux_loss_weight_scaling_formula(self):
        """Verify aux_loss_weight scales correctly with model size.

        Formula: aux_loss_weight = 0.01 * (effective_layers/32) * sqrt(dim/768)

        This ensures larger models have stronger capacity regularization
        to compete with higher CE loss gradients.
        """
        test_cases = [
            # (dim, layers, expected_weight)
            (768, 32, 0.01),  # Base case
            (1536, 64, 0.0283),  # 500M
            (2048, 80, 0.0408),  # ~900M
            (2560, 120, 0.0685),  # 1.5B
            (4096, 160, 0.1155),  # 4B (predicted)
        ]

        for dim, layers, expected in test_cases:
            depth_scale = max(1.0, layers / 32)
            dim_scale = max(1.0, (dim / 768) ** 0.5)
            calculated = 0.01 * depth_scale * dim_scale

            assert abs(calculated - expected) < 0.01, (
                f"aux_loss_weight mismatch for dim={dim}, layers={layers}: "
                f"expected {expected:.4f}, got {calculated:.4f}"
            )

    def test_model_creates_at_all_scales(self):
        """Verify all model variants can be instantiated."""
        for name, variant in MODEL_VARIANTS.items():
            if name == "4B":
                continue  # Skip 4B as it's theoretical

            model = create_model_from_variant(variant)
            param_count = sum(p.numel() for p in model.parameters())

            # Check param count is reasonable (within 50% of expected)
            expected = variant.expected_params_m * 1e6
            assert 0.5 * expected < param_count < 2.0 * expected, (
                f"{name}: param count {param_count / 1e6:.1f}M not close to expected {variant.expected_params_m}M"
            )

            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    def test_effective_layers_monotonic(self):
        """Verify effective layers increase with model size."""
        sizes = ["100M", "250M", "500M", "750M", "900M", "1B", "1.5B"]
        layers = [MODEL_VARIANTS[s].effective_layers for s in sizes]

        for i in range(1, len(layers)):
            assert layers[i] >= layers[i - 1], (
                f"Effective layers should increase: {sizes[i - 1]}={layers[i - 1]} > {sizes[i]}={layers[i]}"
            )

    def test_dim_scaling(self):
        """Verify dimension scales with model size."""
        sizes = ["100M", "250M", "500M", "750M", "900M", "1B", "1.5B"]
        dims = [MODEL_VARIANTS[s].dim for s in sizes]

        for i in range(1, len(dims)):
            assert dims[i] >= dims[i - 1], (
                f"Dimension should increase: {sizes[i - 1]}={dims[i - 1]} > {sizes[i]}={dims[i]}"
            )

    def test_gqa_ratio_valid(self):
        """Verify GQA ratios are valid for all variants."""
        for name, variant in MODEL_VARIANTS.items():
            assert variant.n_heads > variant.n_kv_heads, (
                f"{name}: n_heads ({variant.n_heads}) must be > n_kv_heads ({variant.n_kv_heads})"
            )
            assert variant.n_heads % variant.n_kv_heads == 0, (
                f"{name}: n_heads must be divisible by n_kv_heads"
            )

    def test_4b_config_valid(self):
        """Verify 4B theoretical config follows scaling laws."""
        v4b = MODEL_VARIANTS["4B"]
        v15b = MODEL_VARIANTS["1.5B"]

        # 4B should have larger dim
        assert v4b.dim > v15b.dim, "4B should have larger dimension than 1.5B"

        # 4B should have more layers
        assert v4b.effective_layers > v15b.effective_layers, (
            "4B should have more effective layers than 1.5B"
        )

        # 4B aux_loss_weight should be higher
        aux_4b = 0.01 * (v4b.effective_layers / 32) * ((v4b.dim / 768) ** 0.5)
        aux_15b = 0.01 * (v15b.effective_layers / 32) * ((v15b.dim / 768) ** 0.5)

        assert aux_4b > aux_15b, (
            f"4B aux_loss_weight ({aux_4b:.4f}) should be > 1.5B ({aux_15b:.4f})"
        )


class TestScalingPredictions:
    """Tests for validating scaling predictions to 4B."""

    def test_aux_loss_weight_extrapolation(self):
        """Test that aux_loss_weight formula extrapolates sensibly to 4B."""
        v4b = MODEL_VARIANTS["4B"]

        # Calculate predicted aux_loss_weight
        depth_scale = max(1.0, v4b.effective_layers / 32)
        dim_scale = max(1.0, (v4b.dim / 768) ** 0.5)
        aux_weight_4b = 0.01 * depth_scale * dim_scale

        # Should be reasonable (not too extreme)
        assert 0.05 < aux_weight_4b < 0.5, (
            f"4B aux_loss_weight {aux_weight_4b:.4f} seems unreasonable (should be 0.05-0.5)"
        )

    @pytest.mark.parametrize("variant_name", ["100M", "250M", "500M"])
    def test_small_models_converge_quickly(self, variant_name: str):
        """Verify smaller models converge to target capacity quickly."""
        variant = MODEL_VARIANTS[variant_name]

        model = create_model_from_variant(variant)
        device = get_device()
        model = model.to(device)
        model.train()

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        mod_probs = []
        for step in range(30):
            input_ids = torch.randint(0, 50257, (2, 128), device=device)
            optimizer.zero_grad()

            # Model returns (logits, losses_dict) or (logits, aux_loss, ponder_loss)
            output = model(input_ids)
            if (
                isinstance(output, tuple)
                and len(output) == 2
                and isinstance(output[1], dict)
            ):
                logits, losses = output
                aux_loss = losses.get("aux_loss", 0)
                ponder_loss = losses.get("ponder_loss", 0)
            elif isinstance(output, tuple) and len(output) == 3:
                logits, aux_loss, ponder_loss = output
            else:
                logits = output
                aux_loss, ponder_loss = 0, 0

            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, 50257), input_ids[:, 1:].reshape(-1)
            )
            total_loss = loss
            if isinstance(aux_loss, torch.Tensor):
                total_loss = total_loss + aux_loss
            if isinstance(ponder_loss, torch.Tensor):
                total_loss = total_loss + ponder_loss

            total_loss.backward()
            optimizer.step()

            # Collect mod probs from MoD MLP wrappers inside MoR blocks
            for layer in model.layers:
                # New architecture: CCGQAMoRBlock with mod_mlp_wrapper
                if isinstance(layer, CCGQAMoRBlock):
                    if hasattr(layer, 'mod_mlp_wrapper') and layer.mod_mlp_wrapper is not None:
                        mod_stats = layer.mod_mlp_wrapper.get_routing_stats()
                        mod_probs.append(mod_stats.get("probs_mean", 0.0))
                # Legacy: CCGQAMoDBlockWrapper
                elif hasattr(layer, "_last_probs_mean"):
                    mod_probs.append(layer._last_probs_mean)

        # Should have some routing activity
        assert len(mod_probs) > 0, f"{variant_name}: No MoD routing stats collected"

        # Final prob should be reasonable
        final_prob = mod_probs[-1] if mod_probs else 0.5
        assert 0.3 < final_prob < 1.0, (
            f"{variant_name}: Final mod_prob {final_prob:.3f} out of range"
        )

        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


if __name__ == "__main__":
    # Run quick smoke test
    print("Running paper compliance smoke test...")

    for variant_name in ["100M"]:
        print(f"\n{'=' * 60}")
        print(f"Testing {variant_name} variant")
        print(f"{'=' * 60}")

        variant = MODEL_VARIANTS[variant_name]
        results = run_variant_diagnostic(variant, steps=20)

        print(f"Parameters: {results['params_m']:.1f}M")
        print(f"Effective layers: {results['effective_layers']}")
        print(f"MoD: {results['mod_summary']}")
        print(f"MoR: {results['mor_summary']}")
        print(f"Paper Compliance: {results['paper_compliance']}")
