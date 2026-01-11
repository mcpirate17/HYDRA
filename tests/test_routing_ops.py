"""
Tests for hydra/routing/ops.py - Shared routing operations.

Tests cover:
- Logit processing (soft_clamp_logits, temperature_scaled_sigmoid)
- Straight-through estimators (STERound, STETopK)
- Mask computation (compute_exit_masks, compute_ste_weights)
- Gather/scatter operations
- Capacity utilities
"""

import pytest
import torch


# =============================================================================
# Logit Processing Tests
# =============================================================================


class TestSoftClampLogits:
    """Tests for soft_clamp_logits function."""

    def test_basic_clamping(self):
        """Test that soft clamp produces values in expected range."""
        from hydra.routing.ops import soft_clamp_logits

        logits = torch.tensor([-10.0, -1.0, 0.0, 1.0, 10.0])
        result = soft_clamp_logits(logits, scale=2.0, clamp_range=3.0)

        # Output should be within (-3, 3)
        assert result.max() < 3.0
        assert result.min() > -3.0

    def test_gradient_preservation(self):
        """Test that soft clamp preserves gradients."""
        from hydra.routing.ops import soft_clamp_logits

        logits = torch.tensor([0.5], requires_grad=True)
        result = soft_clamp_logits(logits)
        result.backward()

        # Gradient should exist and be non-zero
        assert logits.grad is not None
        assert logits.grad.abs() > 0

    def test_different_scales(self):
        """Test different scale parameters."""
        from hydra.routing.ops import soft_clamp_logits

        logits = torch.tensor([2.0])

        # Smaller scale = more aggressive clamping (closer to boundaries)
        soft = soft_clamp_logits(logits, scale=1.0)
        softer = soft_clamp_logits(logits, scale=4.0)

        # Smaller scale should push value closer to clamp_range
        assert soft.abs() > softer.abs()

    def test_symmetry(self):
        """Test that clamping is symmetric around zero."""
        from hydra.routing.ops import soft_clamp_logits

        pos = soft_clamp_logits(torch.tensor([5.0]))
        neg = soft_clamp_logits(torch.tensor([-5.0]))

        assert torch.isclose(pos, -neg, atol=1e-6)


class TestTemperatureScaledSigmoid:
    """Tests for temperature_scaled_sigmoid function."""

    def test_basic_output_range(self):
        """Test that output is in (0, 1)."""
        from hydra.routing.ops import temperature_scaled_sigmoid

        logits = torch.tensor([-10.0, 0.0, 10.0])
        result = temperature_scaled_sigmoid(logits)

        assert result.min() > 0
        assert result.max() < 1

    def test_temperature_sharpness(self):
        """Test that lower temperature gives sharper decisions."""
        from hydra.routing.ops import temperature_scaled_sigmoid

        logits = torch.tensor([0.5])

        # Lower temp = closer to 0 or 1
        sharp = temperature_scaled_sigmoid(logits, temperature=0.1)
        soft = temperature_scaled_sigmoid(logits, temperature=2.0)

        # Sharp should be further from 0.5 than soft
        sharp_dist = (sharp - 0.5).abs()
        soft_dist = (soft - 0.5).abs()
        assert sharp_dist > soft_dist

    def test_zero_temperature_handling(self):
        """Test that zero/near-zero temperature doesn't cause NaN."""
        from hydra.routing.ops import temperature_scaled_sigmoid

        logits = torch.tensor([1.0])
        result = temperature_scaled_sigmoid(logits, temperature=0.0)

        assert not torch.isnan(result).any()


# =============================================================================
# Straight-Through Estimator Tests
# =============================================================================


class TestSTERound:
    """Tests for ste_round function."""

    def test_forward_is_round(self):
        """Test that forward pass is standard rounding."""
        from hydra.routing.ops import ste_round

        x = torch.tensor([0.3, 0.7, 1.5, 2.4])
        result = ste_round(x)
        expected = torch.tensor([0.0, 1.0, 2.0, 2.0])

        torch.testing.assert_close(result, expected)

    def test_backward_passes_gradient(self):
        """Test that backward pass preserves gradient."""
        from hydra.routing.ops import ste_round

        x = torch.tensor([1.5], requires_grad=True)
        result = ste_round(x)
        result.backward()

        # STE: gradient should be 1.0 (identity)
        assert x.grad is not None
        torch.testing.assert_close(x.grad, torch.tensor([1.0]))

    def test_negative_values(self):
        """Test rounding of negative values."""
        from hydra.routing.ops import ste_round

        x = torch.tensor([-0.3, -0.7, -1.5])
        result = ste_round(x)
        expected = torch.round(x)

        torch.testing.assert_close(result, expected)


class TestSTETopK:
    """Tests for ste_topk function."""

    def test_forward_selects_topk(self):
        """Test that forward pass selects top-k elements."""
        from hydra.routing.ops import ste_topk

        scores = torch.tensor([[0.1, 0.9, 0.5, 0.3]])
        mask = ste_topk(scores, k=2)

        # Should select positions 1 and 2 (values 0.9 and 0.5)
        expected = torch.tensor([[0.0, 1.0, 1.0, 0.0]])
        torch.testing.assert_close(mask, expected)

    def test_mask_sum_equals_k(self):
        """Test that mask has exactly k ones per batch."""
        from hydra.routing.ops import ste_topk

        scores = torch.randn(4, 16)
        k = 5
        mask = ste_topk(scores, k)

        # Each row should have exactly k ones
        assert mask.sum(dim=-1).allclose(torch.full((4,), float(k)))

    def test_backward_has_gradient(self):
        """Test that backward pass provides gradients."""
        from hydra.routing.ops import ste_topk

        scores = torch.tensor([[0.1, 0.9, 0.5, 0.3]], requires_grad=True)
        mask = ste_topk(scores, k=2)
        loss = mask.sum()
        loss.backward()

        assert scores.grad is not None
        # Gradients should be non-zero for selected positions
        assert scores.grad[0, 1] != 0  # 0.9 was selected
        assert scores.grad[0, 2] != 0  # 0.5 was selected


# =============================================================================
# Mask Computation Tests
# =============================================================================


class TestComputeExitMasks:
    """Tests for compute_exit_masks function."""

    def test_basic_exit_masks(self):
        """Test basic exit mask computation."""
        from hydra.routing.ops import compute_exit_masks

        # Depths: batch=1, seq=4, values [0, 1, 2, 1]
        depths = torch.tensor([[0, 1, 2, 1]])
        n_depths = 3

        exit_at, active_at = compute_exit_masks(depths, n_depths, depths.device)

        # Exit masks: token exits at its depth
        # Depth 0: only first token exits
        assert exit_at[0, 0, 0] == True
        assert exit_at[0, 0, 1] == False
        # Depth 1: tokens 1 and 3 exit
        assert exit_at[1, 0, 1] == True
        assert exit_at[1, 0, 3] == True
        # Depth 2: only token 2 exits
        assert exit_at[2, 0, 2] == True

    def test_active_masks(self):
        """Test active mask computation."""
        from hydra.routing.ops import compute_exit_masks

        depths = torch.tensor([[0, 2, 1]])
        n_depths = 3

        exit_at, active_at = compute_exit_masks(depths, n_depths, depths.device)

        # Active at depth 0: all tokens with depth >= 0
        assert active_at[0].all()
        # Active at depth 1: tokens with depth >= 1 (positions 1, 2)
        assert active_at[1, 0, 0] == False
        assert active_at[1, 0, 1] == True
        assert active_at[1, 0, 2] == True
        # Active at depth 2: only token 1 (depth=2)
        assert active_at[2, 0, 0] == False
        assert active_at[2, 0, 1] == True
        assert active_at[2, 0, 2] == False

    def test_output_shapes(self):
        """Test output tensor shapes."""
        from hydra.routing.ops import compute_exit_masks

        batch, seq = 2, 8
        n_depths = 4
        depths = torch.randint(0, n_depths, (batch, seq))

        exit_at, active_at = compute_exit_masks(depths, n_depths, depths.device)

        assert exit_at.shape == (n_depths, batch, seq)
        assert active_at.shape == (n_depths, batch, seq)


class TestComputeSTEWeights:
    """Tests for compute_ste_weights function."""

    def test_gaussian_shape(self):
        """Test that weights form Gaussian shape around target."""
        from hydra.routing.ops import compute_ste_weights

        # Prob=0.5 means target depth = 1.5 for n_depths=4
        probs = torch.tensor([[0.5]])
        n_depths = 4

        weights = compute_ste_weights(probs, n_depths, probs.device, torch.float32)

        # Weights should peak around depth 1-2 (target 1.5)
        weights_flat = weights[:, 0, 0]
        # Depth 1 and 2 should have higher weights than 0 and 3
        assert weights_flat[1] > weights_flat[0]
        assert weights_flat[2] > weights_flat[3]

    def test_extreme_probs(self):
        """Test weights at extreme probability values."""
        from hydra.routing.ops import compute_ste_weights

        n_depths = 3

        # prob=0 means target depth=0
        weights_low = compute_ste_weights(
            torch.tensor([[0.0]]), n_depths, torch.device('cpu'), torch.float32
        )
        assert weights_low[0, 0, 0] > weights_low[1, 0, 0]  # Depth 0 has highest weight

        # prob=1 means target depth=n_depths-1=2
        weights_high = compute_ste_weights(
            torch.tensor([[1.0]]), n_depths, torch.device('cpu'), torch.float32
        )
        assert weights_high[2, 0, 0] > weights_high[0, 0, 0]  # Depth 2 has highest weight


# =============================================================================
# Gather/Scatter Tests
# =============================================================================


class TestGatherByMask:
    """Tests for gather_by_mask function."""

    def test_basic_gather(self):
        """Test basic masked gathering."""
        from hydra.routing.ops import gather_by_mask

        x = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])  # [1, 3, 2]
        mask = torch.tensor([[True, False, True]])  # Select positions 0 and 2

        selected, indices = gather_by_mask(x, mask)

        # Should select tokens at positions 0 and 2
        assert selected.shape == (2, 2)
        torch.testing.assert_close(selected[0], torch.tensor([1.0, 2.0]))
        torch.testing.assert_close(selected[1], torch.tensor([5.0, 6.0]))

    def test_indices_for_scatter(self):
        """Test that returned indices can be used for scattering."""
        from hydra.routing.ops import gather_by_mask

        x = torch.randn(2, 4, 8)
        mask = torch.tensor([[True, True, False, False], [False, True, True, False]])

        selected, indices = gather_by_mask(x, mask)

        # Indices should have shape [num_selected, 2]
        num_selected = mask.sum()
        assert indices.shape == (num_selected, 2)

        # Indices should be valid
        for idx in indices:
            batch_idx, seq_idx = idx
            assert 0 <= batch_idx < 2
            assert 0 <= seq_idx < 4


class TestScatterByIndices:
    """Tests for scatter_by_indices function."""

    def test_basic_scatter(self):
        """Test basic scattering to output tensor."""
        from hydra.routing.ops import scatter_by_indices

        values = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # 2 tokens, dim=2
        indices = torch.tensor([[0, 0], [0, 2]])  # Scatter to (0,0) and (0,2)

        output = scatter_by_indices(
            values, indices,
            output_shape=(1, 3, 2),
            device=torch.device('cpu'),
            dtype=torch.float32
        )

        torch.testing.assert_close(output[0, 0], torch.tensor([1.0, 2.0]))
        torch.testing.assert_close(output[0, 2], torch.tensor([3.0, 4.0]))
        # Position 1 should be zeros
        torch.testing.assert_close(output[0, 1], torch.tensor([0.0, 0.0]))

    def test_roundtrip_gather_scatter(self):
        """Test that gather followed by scatter recovers masked elements."""
        from hydra.routing.ops import gather_by_mask, scatter_by_indices

        x = torch.randn(2, 4, 8)
        mask = torch.tensor([[True, False, True, False], [False, True, False, True]])

        # Gather masked elements
        selected, indices = gather_by_mask(x, mask)

        # Scatter back
        reconstructed = scatter_by_indices(
            selected, indices, x.shape, x.device, x.dtype
        )

        # Masked positions should match
        for b in range(2):
            for s in range(4):
                if mask[b, s]:
                    torch.testing.assert_close(reconstructed[b, s], x[b, s])


class TestGatherByIndices:
    """Tests for gather_by_indices function."""

    def test_basic_index_gather(self):
        """Test gathering by explicit indices."""
        from hydra.routing.ops import gather_by_indices

        x = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])  # [1, 4, 1]
        indices = torch.tensor([[2, 0, 3]])  # Gather positions 2, 0, 3

        result = gather_by_indices(x, indices)

        expected = torch.tensor([[[3.0], [1.0], [4.0]]])
        torch.testing.assert_close(result, expected)


class TestScatterAddByIndices:
    """Tests for scatter_add_by_indices function."""

    def test_basic_scatter_add(self):
        """Test scatter-add operation."""
        from hydra.routing.ops import scatter_add_by_indices

        output = torch.zeros(1, 3, 2)
        values = torch.tensor([[[1.0, 1.0], [2.0, 2.0]]])  # [1, 2, 2]
        indices = torch.tensor([[0, 1]])

        scatter_add_by_indices(output, values, indices)

        torch.testing.assert_close(output[0, 0], torch.tensor([1.0, 1.0]))
        torch.testing.assert_close(output[0, 1], torch.tensor([2.0, 2.0]))

    def test_accumulation(self):
        """Test that scatter_add accumulates when indices overlap."""
        from hydra.routing.ops import scatter_add_by_indices

        output = torch.zeros(1, 2, 1)
        values = torch.tensor([[[1.0], [2.0]]])
        indices = torch.tensor([[0, 0]])  # Both scatter to position 0

        scatter_add_by_indices(output, values, indices)

        # Should accumulate: 1.0 + 2.0 = 3.0
        torch.testing.assert_close(output[0, 0], torch.tensor([3.0]))


# =============================================================================
# Capacity Utility Tests
# =============================================================================


class TestComputeCapacityK:
    """Tests for compute_capacity_k function."""

    def test_basic_capacity(self):
        """Test basic capacity computation."""
        from hydra.routing.ops import compute_capacity_k

        k = compute_capacity_k(seq_len=100, capacity_ratio=0.5)
        assert k == 50

    def test_minimum_k(self):
        """Test that k is at least 1."""
        from hydra.routing.ops import compute_capacity_k

        k = compute_capacity_k(seq_len=10, capacity_ratio=0.01)
        assert k >= 1

    def test_maximum_k(self):
        """Test that k doesn't exceed seq_len."""
        from hydra.routing.ops import compute_capacity_k

        k = compute_capacity_k(seq_len=10, capacity_ratio=1.5)
        assert k <= 10


class TestBuildCapacitySchedule:
    """Tests for build_capacity_schedule function."""

    def test_decreasing_schedule(self):
        """Test that capacity decreases with depth."""
        from hydra.routing.ops import build_capacity_schedule

        schedule = build_capacity_schedule(n_depths=4, decay_rate=0.15)

        # Should be monotonically decreasing
        for i in range(len(schedule) - 1):
            assert schedule[i] >= schedule[i + 1]

    def test_min_capacity_respected(self):
        """Test that min_capacity is respected."""
        from hydra.routing.ops import build_capacity_schedule

        min_cap = 0.3
        schedule = build_capacity_schedule(n_depths=10, decay_rate=0.2, min_capacity=min_cap)

        for cap in schedule:
            assert cap >= min_cap

    def test_first_level_full_capacity(self):
        """Test that first level starts at full capacity."""
        from hydra.routing.ops import build_capacity_schedule

        schedule = build_capacity_schedule(n_depths=4)

        assert schedule[0] == 1.0
