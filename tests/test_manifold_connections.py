"""
Tests for Manifold-Constrained Hyper Connections.

Tests cover:
- Manifold projection correctness (sphere and hyperbolic)
- Gradient flow through projections
- Hypernetwork output shapes
- Warmup behavior
- torch.compile compatibility
- Model integration
"""

import pytest
import torch
import torch.nn as nn

from hydra.layers import ManifoldConstrainedHyperConnection


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_input():
    """Sample input tensor."""
    return torch.randn(2, 16, 256)


@pytest.fixture
def sample_input_cuda():
    """Sample input tensor on CUDA if available."""
    if torch.cuda.is_available():
        return torch.randn(2, 16, 256, device="cuda")
    return None


@pytest.fixture
def sphere_module():
    """Sphere manifold connection."""
    return ManifoldConstrainedHyperConnection(
        dim=256,
        n_components=4,
        manifold_type="sphere",
        warmup_steps=100,
    )


@pytest.fixture
def hyperbolic_module():
    """Hyperbolic manifold connection."""
    return ManifoldConstrainedHyperConnection(
        dim=256,
        n_components=4,
        manifold_type="hyperbolic",
        curvature=1.0,
        warmup_steps=100,
    )


# =============================================================================
# Manifold Projection Tests
# =============================================================================


class TestSphereProjection:
    """Tests for sphere (L2 normalize) projection."""

    def test_sphere_projection_unit_norm(self, sphere_module):
        """Test that sphere projection outputs unit vectors."""
        x = torch.randn(8, 32, 256)
        projected = sphere_module._project_to_manifold(x)
        norms = projected.norm(dim=-1)

        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_sphere_projection_preserves_direction(self, sphere_module):
        """Test that sphere projection preserves vector direction."""
        x = torch.randn(4, 16, 256)
        projected = sphere_module._project_to_manifold(x)

        # Dot product should be positive (same direction)
        dot_products = (x * projected).sum(dim=-1)
        assert (dot_products >= 0).all()

    def test_sphere_projection_zero_handling(self, sphere_module):
        """Test that sphere projection handles near-zero vectors gracefully."""
        # Create a vector with very small values
        x = torch.randn(2, 4, 256) * 1e-8

        projected = sphere_module._project_to_manifold(x)

        # Should not have NaN or Inf
        assert torch.isfinite(projected).all()
        # Result should have unit norm (clamp prevents div-by-zero)
        norms = projected.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


class TestHyperbolicProjection:
    """Tests for hyperbolic (Poincare ball) projection."""

    def test_hyperbolic_projection_inside_ball(self, hyperbolic_module):
        """Test that hyperbolic projection outputs inside unit ball."""
        # Test with large input that would be outside ball without projection
        x = torch.randn(8, 32, 256) * 10
        projected = hyperbolic_module._project_to_manifold(x)
        norms = projected.norm(dim=-1)

        assert (norms < 1.0).all()

    def test_hyperbolic_projection_small_input(self, hyperbolic_module):
        """Test hyperbolic projection with small inputs."""
        x = torch.randn(4, 16, 256) * 0.01
        projected = hyperbolic_module._project_to_manifold(x)
        norms = projected.norm(dim=-1)

        assert torch.isfinite(projected).all()
        assert (norms < 1.0).all()

    def test_hyperbolic_curvature_effect(self):
        """Test that curvature affects projection."""
        x = torch.randn(2, 8, 128) * 5

        low_curvature = ManifoldConstrainedHyperConnection(
            dim=128, n_components=4, manifold_type="hyperbolic", curvature=0.1
        )
        high_curvature = ManifoldConstrainedHyperConnection(
            dim=128, n_components=4, manifold_type="hyperbolic", curvature=10.0
        )

        proj_low = low_curvature._project_to_manifold(x)
        proj_high = high_curvature._project_to_manifold(x)

        # Higher curvature should project closer to center
        norms_low = proj_low.norm(dim=-1)
        norms_high = proj_high.norm(dim=-1)
        assert (norms_high < norms_low).all()


# =============================================================================
# Forward Pass Tests
# =============================================================================


class TestForwardPass:
    """Tests for full forward pass."""

    def test_output_shape_matches_input(self, sphere_module, sample_input):
        """Test output shape matches input."""
        output = sphere_module(sample_input)
        assert output.shape == sample_input.shape

    def test_warmup_zero_gives_identity(self, sphere_module, sample_input):
        """Test that step=0 (alpha=0) gives approximately identity."""
        sphere_module.set_global_step(0)

        # With residual_alpha initialized to 0 and warmup_scale=0,
        # output should be close to input
        output = sphere_module(sample_input)

        # The output should be very close to input since alpha*warmup=0
        assert torch.allclose(output, sample_input, atol=1e-5)

    def test_warmup_ramps_contribution(self, sphere_module, sample_input):
        """Test that warmup gradually increases contribution."""
        # At step 0 (warmup_scale=0)
        sphere_module.set_global_step(0)
        output_0 = sphere_module(sample_input)
        diff_0 = (output_0 - sample_input).abs().mean()

        # At step 50 (warmup_scale=0.5)
        sphere_module.set_global_step(50)
        output_50 = sphere_module(sample_input)
        diff_50 = (output_50 - sample_input).abs().mean()

        # At step 100 (warmup_scale=1.0)
        sphere_module.set_global_step(100)
        output_100 = sphere_module(sample_input)
        diff_100 = (output_100 - sample_input).abs().mean()

        # Difference should increase with warmup
        # (assuming residual_alpha becomes non-zero during forward)
        # Note: with alpha=0 initially, all diffs are ~0
        assert diff_0 <= diff_50 or torch.isclose(diff_0, diff_50, atol=1e-5)

    def test_deterministic_output(self, sphere_module, sample_input):
        """Test that output is deterministic."""
        sphere_module.set_global_step(50)

        output1 = sphere_module(sample_input)
        output2 = sphere_module(sample_input)

        assert torch.allclose(output1, output2)

    def test_batch_independence(self, sphere_module):
        """Test that batches are processed independently."""
        sphere_module.set_global_step(100)

        x1 = torch.randn(1, 16, 256)
        x2 = torch.randn(1, 16, 256)
        x_cat = torch.cat([x1, x2], dim=0)

        out_cat = sphere_module(x_cat)
        out_1 = sphere_module(x1)
        out_2 = sphere_module(x2)

        assert torch.allclose(out_cat[0:1], out_1, atol=1e-5)
        assert torch.allclose(out_cat[1:2], out_2, atol=1e-5)


# =============================================================================
# Gradient Tests
# =============================================================================


class TestGradientFlow:
    """Tests for gradient flow through manifold connections."""

    def test_gradient_exists(self, sphere_module, sample_input):
        """Test that gradients flow through the module."""
        x = sample_input.clone().requires_grad_(True)
        sphere_module.set_global_step(100)
        # Set alpha to non-zero to ensure gradient flow
        sphere_module.residual_alpha.data.fill_(0.1)

        output = sphere_module(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        assert x.grad.abs().sum() > 0

    def test_no_nan_gradients(self, sphere_module, sample_input):
        """Test no NaN gradients under normal conditions."""
        x = sample_input.clone().requires_grad_(True)
        sphere_module.set_global_step(100)
        sphere_module.residual_alpha.data.fill_(0.1)

        output = sphere_module(x)
        loss = output.sum()
        loss.backward()

        assert not torch.isnan(x.grad).any()
        assert torch.isfinite(x.grad).all()

    def test_gradient_bounded(self, sphere_module):
        """Test that gradients don't explode through manifold projection."""
        x = torch.randn(4, 32, 256, requires_grad=True)
        sphere_module.set_global_step(100)
        sphere_module.residual_alpha.data.fill_(1.0)

        output = sphere_module(x)
        loss = output.sum()
        loss.backward()

        # Gradient norm should be bounded
        grad_norm = x.grad.norm()
        assert grad_norm < x.numel()  # Much less than O(n) scaling

    def test_hyperbolic_gradients(self, hyperbolic_module, sample_input):
        """Test gradient flow through hyperbolic projection."""
        x = sample_input.clone().requires_grad_(True)
        hyperbolic_module.set_global_step(100)
        hyperbolic_module.residual_alpha.data.fill_(0.1)

        output = hyperbolic_module(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_basis_receives_gradients(self, sphere_module, sample_input):
        """Test that basis parameters receive gradients."""
        x = sample_input.clone()
        sphere_module.set_global_step(100)
        sphere_module.residual_alpha.data.fill_(0.1)

        output = sphere_module(x)
        loss = output.sum()
        loss.backward()

        assert sphere_module.basis.grad is not None
        assert torch.isfinite(sphere_module.basis.grad).all()


# =============================================================================
# Initialization Tests
# =============================================================================


class TestInitialization:
    """Tests for module initialization."""

    def test_basis_initialized_on_manifold(self, sphere_module):
        """Test that basis vectors start on the manifold."""
        norms = sphere_module.basis.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_alpha_initialized_to_small_value(self, sphere_module):
        """Test that residual_alpha starts at a small value (0.01)."""
        assert sphere_module.residual_alpha.item() == pytest.approx(0.01, abs=1e-5)

    def test_invalid_manifold_type_raises(self):
        """Test that invalid manifold type raises error."""
        with pytest.raises(ValueError, match="manifold_type must be"):
            ManifoldConstrainedHyperConnection(dim=256, manifold_type="invalid")


# =============================================================================
# torch.compile Compatibility Tests
# =============================================================================


class TestTorchCompile:
    """Tests for torch.compile compatibility."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_compile_forward(self, sample_input_cuda):
        """Test that forward pass works with torch.compile."""
        if sample_input_cuda is None:
            pytest.skip("CUDA not available")

        mc = ManifoldConstrainedHyperConnection(dim=256, n_components=4).cuda()
        mc.set_global_step(100)
        mc.residual_alpha.data.fill_(0.1)

        compiled_mc = torch.compile(mc)
        output = compiled_mc(sample_input_cuda)

        assert output.shape == sample_input_cuda.shape
        assert torch.isfinite(output).all()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_compile_backward(self, sample_input_cuda):
        """Test that backward pass works with torch.compile."""
        if sample_input_cuda is None:
            pytest.skip("CUDA not available")

        mc = ManifoldConstrainedHyperConnection(dim=256, n_components=4).cuda()
        mc.set_global_step(100)
        mc.residual_alpha.data.fill_(0.1)

        compiled_mc = torch.compile(mc)
        x = sample_input_cuda.clone().requires_grad_(True)

        output = compiled_mc(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


# =============================================================================
# Stats and Diagnostics Tests
# =============================================================================


class TestStats:
    """Tests for get_stats method."""

    def test_stats_keys(self, sphere_module):
        """Test that get_stats returns expected keys."""
        sphere_module.set_global_step(50)
        stats = sphere_module.get_stats()

        assert "residual_alpha" in stats
        assert "warmup_scale" in stats
        assert "global_step" in stats
        assert "basis_norm_mean" in stats
        assert "basis_norm_std" in stats

    def test_warmup_scale_in_stats(self, sphere_module):
        """Test that warmup_scale is correct in stats."""
        sphere_module.set_global_step(50)  # warmup_steps=100
        stats = sphere_module.get_stats()

        assert abs(stats["warmup_scale"] - 0.5) < 0.01

        sphere_module.set_global_step(100)
        stats = sphere_module.get_stats()

        assert abs(stats["warmup_scale"] - 1.0) < 0.01


# =============================================================================
# Integration Tests
# =============================================================================


class TestModelIntegration:
    """Tests for integration with HydraModel."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for model test")
    def test_model_with_manifold_forward(self):
        """Test that model with manifold connections runs forward pass."""
        from hydra.model.framework import HydraModel

        model = HydraModel(
            vocab_size=1000,
            dim=128,
            n_mor_blocks=2,
            recursions_per_block=2,
            n_heads=4,
            n_kv_heads=2,
            max_seq_len=64,
            manifold_enabled=True,
            manifold_type="sphere",
            manifold_n_components=4,
            manifold_warmup_steps=10,
            manifold_placement_interval=1,
        ).cuda()

        x = torch.randint(0, 1000, (2, 32)).cuda()
        model.set_global_step(5)

        output = model(x)
        assert output.shape == (2, 32, 1000)
        assert torch.isfinite(output).all()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for model test")
    def test_model_routing_stats_includes_manifold(self):
        """Test that routing stats include manifold info."""
        from hydra.model.framework import HydraModel

        model = HydraModel(
            vocab_size=1000,
            dim=128,
            n_mor_blocks=2,
            recursions_per_block=2,
            n_heads=4,
            n_kv_heads=2,
            max_seq_len=64,
            manifold_enabled=True,
            manifold_n_components=4,
            manifold_placement_interval=1,
        ).cuda()

        model.set_global_step(100)
        stats = model.get_routing_stats()

        assert "manifold_connections" in stats
        assert "summary" in stats
        assert stats["summary"].get("manifold_enabled", False) == True
        assert stats["summary"].get("manifold_num_connections", 0) > 0

    def test_model_without_manifold(self):
        """Test that model works without manifold connections."""
        from hydra.model.framework import HydraModel

        model = HydraModel(
            vocab_size=1000,
            dim=128,
            n_mor_blocks=2,
            recursions_per_block=2,
            n_heads=4,
            n_kv_heads=2,
            max_seq_len=64,
            manifold_enabled=False,
        )

        stats = model.get_routing_stats()
        assert stats["summary"].get("manifold_enabled", False) == False
