"""
Tests for hydra/routing/ops.py - shared operations for routing modules.

Tests cover:
- soft_clamp_logits
- temperature_scaled_sigmoid
- STERound (straight-through estimator for rounding)
- STETopK (straight-through estimator for top-k)
- compute_exit_masks
- compute_ste_weights
- gather_by_mask / scatter_by_indices
- gather_by_indices / scatter_add_by_indices
- compute_capacity_k
- build_capacity_schedule
"""

import pytest
import torch

from hydra.routing.ops import (
    soft_clamp_logits,
    ste_round,
    compute_exit_masks,
    gather_by_mask,
    scatter_by_indices,
    compute_ste_weights,
    compute_capacity_k,
    build_capacity_schedule,
)

# Also test unexported functions
from hydra.routing.ops import (
    temperature_scaled_sigmoid,
    STERound,
    STETopK,
    ste_topk,
    gather_by_indices,
    scatter_add_by_indices,
)


# =============================================================================
# Logit Processing Tests
# =============================================================================

class TestSoftClampLogits:
    """Tests for soft_clamp_logits function."""

    def test_output_range(self):
        """Test output stays within clamp_range."""
        logits = torch.randn(10, 100) * 10  # Large values
        output = soft_clamp_logits(logits, scale=2.0, clamp_range=3.0)

        # tanh asymptotically approaches but never exceeds the clamp range
        assert (output >= -3.0).all()
        assert (output <= 3.0).all()

    def test_preserves_sign(self):
        """Test that sign is preserved."""
        logits = torch.tensor([-5.0, -1.0, 0.0, 1.0, 5.0])
        output = soft_clamp_logits(logits)

        # Check sign is preserved
        assert (output[logits < 0] < 0).all()
        assert (output[logits > 0] > 0).all()
        assert output[logits == 0].item() == 0.0

    def test_gradient_flow(self):
        """Test gradients flow through soft clamp."""
        logits = torch.randn(4, 8, requires_grad=True)
        output = soft_clamp_logits(logits)
        output.sum().backward()

        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()
        # Gradients should be non-zero
        assert (logits.grad.abs() > 0).any()

    def test_different_scales(self):
        """Test effect of scale parameter."""
        logits = torch.tensor([1.0, 2.0, 3.0])

        # Larger scale = softer saturation
        output_soft = soft_clamp_logits(logits, scale=4.0)
        output_sharp = soft_clamp_logits(logits, scale=1.0)

        # Soft should be more linear (closer to clamp_range * logits/max)
        # Sharp should be more saturated
        assert output_soft.abs().mean() < output_sharp.abs().mean()


class TestTemperatureScaledSigmoid:
    """Tests for temperature_scaled_sigmoid function."""

    def test_output_range(self):
        """Test output is in (0, 1)."""
        logits = torch.randn(10, 100)
        output = temperature_scaled_sigmoid(logits, temperature=1.0)

        assert (output > 0).all()
        assert (output < 1).all()

    def test_temperature_effect(self):
        """Test lower temperature gives sharper decisions."""
        logits = torch.tensor([0.5, 1.0, 2.0])

        # Low temperature - more extreme
        probs_low = temperature_scaled_sigmoid(logits, temperature=0.1)
        # High temperature - softer
        probs_high = temperature_scaled_sigmoid(logits, temperature=10.0)

        # Low temp should be closer to 0 or 1
        assert probs_low.max() > probs_high.max()
        # High temp should be closer to 0.5
        assert (probs_high - 0.5).abs().mean() < (probs_low - 0.5).abs().mean()

    def test_zero_temperature_safe(self):
        """Test near-zero temperature doesn't cause NaN."""
        logits = torch.randn(4, 8)
        output = temperature_scaled_sigmoid(logits, temperature=1e-8)

        assert torch.isfinite(output).all()


# =============================================================================
# Straight-Through Estimator Tests
# =============================================================================

class TestSTERound:
    """Tests for STERound and ste_round."""

    def test_forward_rounds(self):
        """Test forward pass rounds values."""
        x = torch.tensor([0.3, 0.7, 1.4, 2.6])
        output = ste_round(x)

        expected = torch.tensor([0.0, 1.0, 1.0, 3.0])
        assert torch.equal(output, expected)

    def test_backward_passthrough(self):
        """Test backward pass passes gradients through unchanged."""
        x = torch.tensor([0.3, 0.7, 1.4, 2.6], requires_grad=True)
        output = ste_round(x)
        loss = output.sum()
        loss.backward()

        # Gradient should be 1 for all elements (passthrough)
        expected_grad = torch.ones_like(x)
        assert torch.equal(x.grad, expected_grad)

    def test_gradient_shape(self):
        """Test gradient has correct shape."""
        x = torch.randn(4, 8, requires_grad=True)
        output = ste_round(x)
        output.sum().backward()

        assert x.grad.shape == x.shape


class TestSTETopK:
    """Tests for STETopK and ste_topk."""

    def test_mask_shape(self):
        """Test output mask has correct shape."""
        scores = torch.randn(2, 10)
        mask = ste_topk(scores, k=3)

        assert mask.shape == scores.shape

    def test_correct_k_selected(self):
        """Test exactly k elements are selected per batch."""
        scores = torch.randn(4, 16)
        k = 5
        mask = ste_topk(scores, k=k)

        # Each row should have exactly k ones
        assert (mask.sum(dim=1) == k).all()

    def test_top_k_selected(self):
        """Test top-k scores are actually selected."""
        scores = torch.tensor([[1.0, 5.0, 3.0, 2.0, 4.0]])
        mask = ste_topk(scores, k=3)

        # Indices 1, 2, 4 should be selected (values 5, 3, 4)
        expected = torch.tensor([[0.0, 1.0, 1.0, 0.0, 1.0]])
        assert torch.equal(mask, expected)

    def test_gradient_flow(self):
        """Test gradients flow through STE top-k."""
        scores = torch.randn(2, 8, requires_grad=True)
        mask = ste_topk(scores, k=3)
        loss = mask.sum()
        loss.backward()

        assert scores.grad is not None
        # Gradients should exist only at selected positions (times sigmoid)
        assert torch.isfinite(scores.grad).all()


# =============================================================================
# Mask Computation Tests
# =============================================================================

class TestComputeExitMasks:
    """Tests for compute_exit_masks function."""

    def test_output_shapes(self):
        """Test output tensors have correct shapes."""
        depths = torch.tensor([[0, 1, 2, 1], [2, 0, 1, 2]])  # [2, 4]
        n_depths = 3
        exit_at_depth, active_at_depth = compute_exit_masks(depths, n_depths, depths.device)

        assert exit_at_depth.shape == (n_depths, 2, 4)
        assert active_at_depth.shape == (n_depths, 2, 4)

    def test_exit_masks_correct(self):
        """Test exit_at_depth is correct."""
        depths = torch.tensor([[0, 1, 2]])  # Single batch
        n_depths = 3
        exit_at_depth, _ = compute_exit_masks(depths, n_depths, depths.device)

        # Token 0 exits at depth 0
        assert exit_at_depth[0, 0, 0] == True
        assert exit_at_depth[1, 0, 0] == False
        assert exit_at_depth[2, 0, 0] == False

        # Token 1 exits at depth 1
        assert exit_at_depth[0, 0, 1] == False
        assert exit_at_depth[1, 0, 1] == True
        assert exit_at_depth[2, 0, 1] == False

        # Token 2 exits at depth 2
        assert exit_at_depth[0, 0, 2] == False
        assert exit_at_depth[1, 0, 2] == False
        assert exit_at_depth[2, 0, 2] == True

    def test_active_masks_correct(self):
        """Test active_at_depth is correct."""
        depths = torch.tensor([[2]])  # Token exits at depth 2
        n_depths = 3
        _, active_at_depth = compute_exit_masks(depths, n_depths, depths.device)

        # Token with depth 2 is active at depths 0, 1, 2
        assert active_at_depth[0, 0, 0] == True
        assert active_at_depth[1, 0, 0] == True
        assert active_at_depth[2, 0, 0] == True

    def test_mutual_consistency(self):
        """Test exit and active masks are consistent."""
        depths = torch.randint(0, 4, (3, 8))
        n_depths = 4
        exit_at_depth, active_at_depth = compute_exit_masks(depths, n_depths, depths.device)

        # Each token should exit exactly once
        assert (exit_at_depth.sum(dim=0) == 1).all()

        # Active mask should be True up to and including exit depth
        for d in range(n_depths):
            # If exit at d, should be active at d
            assert (exit_at_depth[d] <= active_at_depth[d]).all()


class TestComputeSTEWeights:
    """Tests for compute_ste_weights function."""

    def test_output_shape(self):
        """Test output has correct shape."""
        probs = torch.rand(2, 8)  # [batch, seq]
        n_depths = 4
        weights = compute_ste_weights(probs, n_depths, probs.device, probs.dtype)

        assert weights.shape == (n_depths, 2, 8)

    def test_gaussian_centered_at_target(self):
        """Test Gaussian weights peak at target depth."""
        probs = torch.tensor([[0.0, 0.5, 1.0]])  # Targets depths 0, 1.5, 3
        n_depths = 4
        weights = compute_ste_weights(probs, n_depths, probs.device, probs.dtype)

        # Token 0 (prob=0) should peak at depth 0
        assert weights[:, 0, 0].argmax() == 0

        # Token 2 (prob=1) should peak at depth 3 (n_depths - 1)
        assert weights[:, 0, 2].argmax() == 3

    def test_weights_finite(self):
        """Test weights are always finite."""
        probs = torch.rand(4, 16)
        weights = compute_ste_weights(probs, 5, probs.device, probs.dtype)

        assert torch.isfinite(weights).all()

    def test_weights_positive(self):
        """Test all weights are positive (Gaussian)."""
        probs = torch.rand(2, 8)
        weights = compute_ste_weights(probs, 4, probs.device, probs.dtype)

        assert (weights > 0).all()


# =============================================================================
# Gather/Scatter Tests
# =============================================================================

class TestGatherByMask:
    """Tests for gather_by_mask function."""

    def test_output_shapes(self):
        """Test output tensors have correct shapes."""
        x = torch.randn(2, 4, 8)  # [B, L, D]
        mask = torch.tensor([[True, False, True, False],
                            [False, True, True, True]])

        selected, indices = gather_by_mask(x, mask)

        total_selected = mask.sum().item()
        assert selected.shape == (total_selected, 8)
        assert indices.shape == (total_selected, 2)

    def test_correct_values_gathered(self):
        """Test correct values are gathered."""
        x = torch.arange(24).float().view(2, 4, 3)  # [2, 4, 3]
        mask = torch.tensor([[True, False, True, False],
                            [False, False, False, True]])

        selected, indices = gather_by_mask(x, mask)

        # Should select x[0,0], x[0,2], x[1,3]
        assert torch.equal(selected[0], x[0, 0])
        assert torch.equal(selected[1], x[0, 2])
        assert torch.equal(selected[2], x[1, 3])

    def test_indices_correct(self):
        """Test indices point to correct positions."""
        x = torch.randn(2, 4, 8)
        mask = torch.tensor([[True, False, True, False],
                            [True, True, False, False]])

        selected, indices = gather_by_mask(x, mask)

        # Verify each index points to correct value
        for i in range(selected.shape[0]):
            b, s = indices[i]
            assert torch.equal(selected[i], x[b, s])


class TestScatterByIndices:
    """Tests for scatter_by_indices function."""

    def test_output_shape(self):
        """Test output has correct shape."""
        values = torch.randn(5, 8)
        indices = torch.tensor([[0, 1], [0, 3], [1, 0], [1, 2], [1, 4]])
        output_shape = (2, 6, 8)

        output = scatter_by_indices(values, indices, output_shape, values.device, values.dtype)

        assert output.shape == output_shape

    def test_correct_values_scattered(self):
        """Test values are scattered to correct positions."""
        values = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # [2, 2]
        indices = torch.tensor([[0, 1], [1, 0]])  # positions
        output_shape = (2, 3, 2)

        output = scatter_by_indices(values, indices, output_shape, values.device, values.dtype)

        # Value [1,2] should be at [0,1]
        assert torch.equal(output[0, 1], values[0])
        # Value [3,4] should be at [1,0]
        assert torch.equal(output[1, 0], values[1])
        # Other positions should be zero
        assert output[0, 0].sum() == 0
        assert output[0, 2].sum() == 0

    def test_roundtrip_gather_scatter(self):
        """Test gather then scatter preserves selected values."""
        x = torch.randn(2, 4, 8)
        mask = torch.tensor([[True, False, True, False],
                            [False, True, False, True]])

        selected, indices = gather_by_mask(x, mask)
        reconstructed = scatter_by_indices(selected, indices, x.shape, x.device, x.dtype)

        # Selected positions should match
        assert torch.allclose(reconstructed[mask], x[mask])
        # Non-selected positions should be zero
        assert reconstructed[~mask].sum() == 0


class TestGatherByIndices:
    """Tests for gather_by_indices function."""

    def test_output_shape(self):
        """Test output has correct shape."""
        x = torch.randn(2, 8, 16)  # [B, L, D]
        indices = torch.tensor([[1, 3, 5], [0, 2, 7]])  # [B, k]

        output = gather_by_indices(x, indices)

        assert output.shape == (2, 3, 16)

    def test_correct_values_gathered(self):
        """Test correct values are gathered."""
        x = torch.arange(16).float().view(2, 4, 2)
        indices = torch.tensor([[0, 2], [1, 3]])

        output = gather_by_indices(x, indices)

        # Batch 0: indices 0, 2
        assert torch.equal(output[0, 0], x[0, 0])
        assert torch.equal(output[0, 1], x[0, 2])
        # Batch 1: indices 1, 3
        assert torch.equal(output[1, 0], x[1, 1])
        assert torch.equal(output[1, 1], x[1, 3])


class TestScatterAddByIndices:
    """Tests for scatter_add_by_indices function."""

    def test_output_shape(self):
        """Test output has correct shape (in-place modification)."""
        output = torch.zeros(2, 8, 4)
        values = torch.ones(2, 3, 4)
        indices = torch.tensor([[1, 3, 5], [0, 2, 7]])

        result = scatter_add_by_indices(output, values, indices)

        assert result.shape == (2, 8, 4)
        assert result is output  # In-place

    def test_values_added_correctly(self):
        """Test values are added at correct positions."""
        output = torch.zeros(2, 4, 2)
        values = torch.ones(2, 2, 2)
        indices = torch.tensor([[0, 2], [1, 3]])

        scatter_add_by_indices(output, values, indices)

        # Batch 0: add 1 at indices 0, 2
        assert output[0, 0].sum() == 2  # [1, 1]
        assert output[0, 1].sum() == 0  # Not selected
        assert output[0, 2].sum() == 2  # [1, 1]

    def test_scatter_add_accumulates(self):
        """Test scatter_add accumulates when same index appears."""
        output = torch.zeros(1, 4, 2)
        values = torch.ones(1, 3, 2)
        indices = torch.tensor([[0, 0, 0]])  # Same index repeated

        scatter_add_by_indices(output, values, indices)

        # All 3 values added to index 0
        assert output[0, 0].sum() == 6  # 3 * [1, 1]


# =============================================================================
# Capacity Utilities Tests
# =============================================================================

class TestComputeCapacityK:
    """Tests for compute_capacity_k function."""

    def test_basic_computation(self):
        """Test basic capacity computation."""
        assert compute_capacity_k(100, 0.75) == 75
        assert compute_capacity_k(100, 0.5) == 50
        assert compute_capacity_k(100, 1.0) == 100

    def test_minimum_one(self):
        """Test k is at least 1."""
        assert compute_capacity_k(100, 0.0) == 1
        assert compute_capacity_k(100, 0.001) == 1

    def test_maximum_seq_len(self):
        """Test k doesn't exceed seq_len."""
        assert compute_capacity_k(100, 1.5) == 100
        assert compute_capacity_k(100, 2.0) == 100

    def test_rounding(self):
        """Test integer rounding behavior."""
        # 10 * 0.75 = 7.5 -> 7
        assert compute_capacity_k(10, 0.75) == 7
        # 10 * 0.25 = 2.5 -> 2
        assert compute_capacity_k(10, 0.25) == 2


class TestBuildCapacitySchedule:
    """Tests for build_capacity_schedule function."""

    def test_correct_length(self):
        """Test schedule has correct length."""
        schedule = build_capacity_schedule(n_depths=4)
        assert len(schedule) == 4

    def test_decreasing(self):
        """Test capacities decrease with depth."""
        schedule = build_capacity_schedule(n_depths=5, decay_rate=0.2)

        for i in range(len(schedule) - 1):
            assert schedule[i] >= schedule[i + 1]

    def test_respects_minimum(self):
        """Test minimum capacity is respected."""
        schedule = build_capacity_schedule(n_depths=10, decay_rate=0.2, min_capacity=0.3)

        assert all(cap >= 0.3 for cap in schedule)

    def test_first_is_one(self):
        """Test first depth has capacity 1.0."""
        schedule = build_capacity_schedule(n_depths=5)
        assert schedule[0] == 1.0

    def test_formula(self):
        """Test capacity follows formula: max(min_cap, 1 - decay*i)."""
        n_depths = 4
        decay_rate = 0.15
        min_capacity = 0.25

        schedule = build_capacity_schedule(n_depths, decay_rate, min_capacity)

        for i, cap in enumerate(schedule):
            expected = max(min_capacity, 1.0 - decay_rate * i)
            assert cap == expected


# =============================================================================
# Edge Cases and Numerical Stability
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and numerical stability."""

    def test_empty_mask_gather(self):
        """Test gather with all-False mask."""
        x = torch.randn(2, 4, 8)
        mask = torch.zeros(2, 4, dtype=torch.bool)

        selected, indices = gather_by_mask(x, mask)

        assert selected.shape == (0, 8)
        assert indices.shape == (0, 2)

    def test_full_mask_gather(self):
        """Test gather with all-True mask."""
        x = torch.randn(2, 4, 8)
        mask = torch.ones(2, 4, dtype=torch.bool)

        selected, indices = gather_by_mask(x, mask)

        assert selected.shape == (8, 8)  # 2*4 = 8 tokens

    def test_single_element(self):
        """Test operations with single element tensors."""
        x = torch.randn(1, 1, 4)
        mask = torch.tensor([[True]])

        selected, indices = gather_by_mask(x, mask)
        assert selected.shape == (1, 4)

    def test_large_batch(self):
        """Test with large batch sizes."""
        x = torch.randn(64, 512, 256)
        mask = torch.rand(64, 512) > 0.5

        selected, indices = gather_by_mask(x, mask)
        reconstructed = scatter_by_indices(selected, indices, x.shape, x.device, x.dtype)

        assert torch.allclose(reconstructed[mask], x[mask])

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_operations(self):
        """Test operations on CUDA tensors."""
        x = torch.randn(2, 4, 8, device="cuda")
        mask = torch.tensor([[True, False, True, False],
                            [False, True, True, True]], device="cuda")

        selected, indices = gather_by_mask(x, mask)

        assert selected.device.type == "cuda"
        assert indices.device.type == "cuda"

        reconstructed = scatter_by_indices(selected, indices, x.shape, x.device, x.dtype)
        assert reconstructed.device.type == "cuda"

    def test_float16_stability(self):
        """Test operations are stable in float16."""
        x = torch.randn(2, 4, 8, dtype=torch.float16)
        probs = torch.rand(2, 4, dtype=torch.float16)

        weights = compute_ste_weights(probs, 4, probs.device, probs.dtype)

        assert torch.isfinite(weights).all()
        assert weights.dtype == torch.float16


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
