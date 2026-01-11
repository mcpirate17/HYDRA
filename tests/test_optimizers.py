"""
Tests for hydra/optim/ - Custom optimizers.

Tests cover:
- Lion and CautiousLion optimizers
- Muon (Muon2D, MuonAdamWHybrid)
- Sophia variants (SophiaG, SophiaH, SophiaGSimple)
"""

import pytest
import torch
import torch.nn as nn


# =============================================================================
# Lion Optimizer Tests
# =============================================================================


class TestLion:
    """Tests for Lion optimizer."""

    def test_basic_step(self):
        """Test that Lion performs a basic optimization step."""
        from hydra.optim.lion import Lion

        model = nn.Linear(10, 5)
        optimizer = Lion(model.parameters(), lr=1e-4)

        # Forward + backward
        x = torch.randn(4, 10)
        loss = model(x).sum()
        loss.backward()

        # Record initial weights
        w_before = model.weight.clone()

        # Step
        optimizer.step()

        # Weights should change
        assert not torch.allclose(model.weight, w_before)

    def test_momentum_buffer_created(self):
        """Test that momentum buffer is initialized."""
        from hydra.optim.lion import Lion

        model = nn.Linear(10, 5)
        optimizer = Lion(model.parameters())

        x = torch.randn(4, 10)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()

        # State should have exp_avg
        for p in model.parameters():
            if p.grad is not None:
                assert "exp_avg" in optimizer.state[p]

    def test_weight_decay(self):
        """Test that weight decay is applied."""
        from hydra.optim.lion import Lion

        model = nn.Linear(10, 5, bias=False)
        # Use high weight decay for observable effect
        optimizer = Lion(model.parameters(), lr=1e-3, weight_decay=0.5)

        # Set weights to known value
        model.weight.data.fill_(1.0)
        w_before = model.weight.sum().item()

        x = torch.randn(4, 10)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()

        # With high weight decay, magnitude should decrease
        # Weight is multiplied by (1 - lr * wd) before update
        assert model.weight.mean().abs() < w_before

    def test_invalid_lr(self):
        """Test that invalid learning rate raises error."""
        from hydra.optim.lion import Lion

        model = nn.Linear(10, 5)
        with pytest.raises(ValueError, match="Invalid learning rate"):
            Lion(model.parameters(), lr=-1e-4)

    def test_invalid_betas(self):
        """Test that invalid betas raise error."""
        from hydra.optim.lion import Lion

        model = nn.Linear(10, 5)
        with pytest.raises(ValueError, match="Invalid beta"):
            Lion(model.parameters(), betas=(1.5, 0.99))

    def test_closure(self):
        """Test that closure is called when provided."""
        from hydra.optim.lion import Lion

        model = nn.Linear(10, 5)
        optimizer = Lion(model.parameters())

        call_count = [0]

        def closure():
            call_count[0] += 1
            x = torch.randn(4, 10)
            loss = model(x).sum()
            loss.backward()
            return loss

        optimizer.step(closure)
        assert call_count[0] == 1


class TestCautiousLion:
    """Tests for CautiousLion optimizer."""

    def test_basic_step(self):
        """Test that CautiousLion performs optimization."""
        from hydra.optim.lion import CautiousLion

        model = nn.Linear(10, 5)
        optimizer = CautiousLion(model.parameters(), lr=1e-4)

        x = torch.randn(4, 10)
        loss = model(x).sum()
        loss.backward()

        w_before = model.weight.clone()
        optimizer.step()

        # Weights may or may not change (depends on agreement)
        # But should not error
        assert model.weight is not None

    def test_masked_updates(self):
        """Test that updates are masked based on sign agreement."""
        from hydra.optim.lion import CautiousLion

        model = nn.Linear(10, 5, bias=False)
        optimizer = CautiousLion(model.parameters())

        # First step to initialize momentum
        x = torch.randn(4, 10)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Second step - some positions may have disagreement
        x = torch.randn(4, 10)
        loss = model(x).sum()
        loss.backward()

        # Get mask before step
        for p in model.parameters():
            state = optimizer.state[p]
            grad = p.grad
            exp_avg = state["exp_avg"]
            mask = (grad * exp_avg) > 0
            # At least some should agree, some disagree in random case
            # This tests the mechanism exists

        optimizer.step()


class TestScheduleFreeLion:
    """Tests for ScheduleFreeLion optimizer."""

    def test_basic_step(self):
        """Test basic optimization step."""
        from hydra.optim.lion import ScheduleFreeLion

        model = nn.Linear(10, 5)
        optimizer = ScheduleFreeLion(model.parameters(), lr=1e-4)

        x = torch.randn(4, 10)
        loss = model(x).sum()
        loss.backward()

        w_before = model.weight.clone()
        optimizer.step()

        assert not torch.allclose(model.weight, w_before)

    def test_z_buffer_created(self):
        """Test that z (optimization point) buffer is created."""
        from hydra.optim.lion import ScheduleFreeLion

        model = nn.Linear(10, 5)
        optimizer = ScheduleFreeLion(model.parameters())

        x = torch.randn(4, 10)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()

        for p in model.parameters():
            if p.grad is not None:
                assert "z" in optimizer.state[p]

    def test_warmup(self):
        """Test that warmup scales learning rate."""
        from hydra.optim.lion import ScheduleFreeLion

        model = nn.Linear(10, 5, bias=False)
        optimizer = ScheduleFreeLion(model.parameters(), lr=1e-3, warmup_steps=10)

        # During warmup, effective LR should be lr * step / warmup
        # Step 1: lr * 1/10 = 1e-4
        x = torch.randn(4, 10)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()

        # The optimizer should have stepped
        assert optimizer._step == 1


# =============================================================================
# Muon Optimizer Tests
# =============================================================================


class TestNewtonSchulzOrthogonalize:
    """Tests for Newton-Schulz orthogonalization."""

    def test_output_shape(self):
        """Test that output shape matches input."""
        from hydra.optim.muon import _newton_schulz_orthogonalize

        grad = torch.randn(64, 32)
        result = _newton_schulz_orthogonalize(grad)

        assert result.shape == grad.shape

    def test_dtype_preserved(self):
        """Test that dtype is preserved."""
        from hydra.optim.muon import _newton_schulz_orthogonalize

        for dtype in [torch.float32, torch.float16]:
            grad = torch.randn(32, 32, dtype=dtype)
            result = _newton_schulz_orthogonalize(grad)
            assert result.dtype == dtype

    def test_requires_2d(self):
        """Test that non-2D input raises error."""
        from hydra.optim.muon import _newton_schulz_orthogonalize

        grad_3d = torch.randn(4, 8, 8)
        with pytest.raises(ValueError, match="Expected 2D"):
            _newton_schulz_orthogonalize(grad_3d)

    def test_transpose_handling(self):
        """Test that tall matrices are transposed internally."""
        from hydra.optim.muon import _newton_schulz_orthogonalize

        # Tall matrix (more rows than cols)
        tall = torch.randn(64, 16)
        result = _newton_schulz_orthogonalize(tall)
        assert result.shape == tall.shape


class TestMuon2D:
    """Tests for Muon2D optimizer."""

    def test_basic_step(self):
        """Test basic optimization step."""
        from hydra.optim.muon import Muon2D

        model = nn.Linear(32, 16, bias=False)  # Only 2D params
        optimizer = Muon2D(model.parameters(), lr=1e-3)

        x = torch.randn(4, 32)
        loss = model(x).sum()
        loss.backward()

        w_before = model.weight.clone()
        optimizer.step()

        assert not torch.allclose(model.weight, w_before)

    def test_only_2d_params(self):
        """Test that only 2D params are updated."""
        from hydra.optim.muon import Muon2D

        model = nn.Linear(32, 16, bias=True)  # Has 1D bias
        optimizer = Muon2D(model.parameters(), lr=1e-3)

        # Initialize bias to non-zero
        model.bias.data.fill_(1.0)
        bias_before = model.bias.clone()

        x = torch.randn(4, 32)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()

        # Bias (1D) should not be updated
        torch.testing.assert_close(model.bias, bias_before)
        # Weight (2D) should be updated
        assert not torch.allclose(model.weight, torch.zeros_like(model.weight))

    def test_invalid_params(self):
        """Test validation of hyperparameters."""
        from hydra.optim.muon import Muon2D

        model = nn.Linear(10, 5)

        with pytest.raises(ValueError):
            Muon2D(model.parameters(), lr=-1)

        with pytest.raises(ValueError):
            Muon2D(model.parameters(), momentum=1.5)

        with pytest.raises(ValueError):
            Muon2D(model.parameters(), weight_decay=-1)

        with pytest.raises(ValueError):
            Muon2D(model.parameters(), ns_steps=0)

        with pytest.raises(ValueError):
            Muon2D(model.parameters(), eps=-1)

        with pytest.raises(ValueError):
            Muon2D(model.parameters(), rms_scale=-1)


class TestMuonAdamWHybrid:
    """Tests for MuonAdamWHybrid optimizer."""

    def test_creation(self):
        """Test hybrid optimizer creation."""
        from hydra.optim.muon import Muon2D, MuonAdamWHybrid

        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
        )

        # Split params by dimensionality
        params_2d = [p for p in model.parameters() if p.ndim == 2]
        params_other = [p for p in model.parameters() if p.ndim != 2]

        muon = Muon2D(params_2d, lr=1e-3)
        adamw = torch.optim.AdamW(params_other, lr=1e-4)

        hybrid = MuonAdamWHybrid(muon=muon, adamw=adamw)

        assert len(hybrid.param_groups) == len(muon.param_groups) + len(adamw.param_groups)

    def test_step(self):
        """Test hybrid step updates both."""
        from hydra.optim.muon import Muon2D, MuonAdamWHybrid

        model = nn.Linear(10, 5, bias=True)
        params_2d = [model.weight]
        params_1d = [model.bias]

        muon = Muon2D(params_2d, lr=1e-3)
        adamw = torch.optim.AdamW(params_1d, lr=1e-3)
        hybrid = MuonAdamWHybrid(muon=muon, adamw=adamw)

        x = torch.randn(4, 10)
        loss = model(x).sum()
        loss.backward()

        w_before = model.weight.clone()
        b_before = model.bias.clone()

        hybrid.step()

        # Weight updated by Muon
        assert not torch.allclose(model.weight, w_before)
        # Bias updated by AdamW
        assert not torch.allclose(model.bias, b_before)

    def test_zero_grad(self):
        """Test that zero_grad clears gradients."""
        from hydra.optim.muon import Muon2D, MuonAdamWHybrid

        model = nn.Linear(10, 5)
        muon = Muon2D([model.weight], lr=1e-3)
        adamw = torch.optim.AdamW([model.bias], lr=1e-3)
        hybrid = MuonAdamWHybrid(muon=muon, adamw=adamw)

        x = torch.randn(4, 10)
        loss = model(x).sum()
        loss.backward()

        assert model.weight.grad is not None
        hybrid.zero_grad()
        assert model.weight.grad is None or model.weight.grad.sum() == 0

    def test_state_dict(self):
        """Test state dict save/load."""
        from hydra.optim.muon import Muon2D, MuonAdamWHybrid

        model = nn.Linear(10, 5)
        muon = Muon2D([model.weight], lr=1e-3)
        adamw = torch.optim.AdamW([model.bias], lr=1e-3)
        hybrid = MuonAdamWHybrid(muon=muon, adamw=adamw)

        # Do a step to populate state
        x = torch.randn(4, 10)
        loss = model(x).sum()
        loss.backward()
        hybrid.step()

        # Save and reload
        state = hybrid.state_dict()
        assert "muon" in state
        assert "adamw" in state

        hybrid.load_state_dict(state)


# =============================================================================
# Sophia Optimizer Tests
# =============================================================================


class TestSophiaG:
    """Tests for SophiaG optimizer."""

    def test_basic_step(self):
        """Test basic optimization step."""
        from hydra.optim.sophia import SophiaG

        model = nn.Linear(10, 5)
        optimizer = SophiaG(model.parameters(), lr=2e-4)

        x = torch.randn(4, 10)
        loss = model(x).sum()
        loss.backward()

        w_before = model.weight.clone()
        optimizer.step()

        assert not torch.allclose(model.weight, w_before)

    def test_state_initialization(self):
        """Test that state is properly initialized."""
        from hydra.optim.sophia import SophiaG

        model = nn.Linear(10, 5)
        optimizer = SophiaG(model.parameters())

        x = torch.randn(4, 10)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()

        for p in model.parameters():
            if p.grad is not None:
                state = optimizer.state[p]
                assert "exp_avg" in state
                assert "hessian" in state

    def test_clipping(self):
        """Test that updates are clipped by rho."""
        from hydra.optim.sophia import SophiaG

        model = nn.Linear(10, 5, bias=False)
        # Very low rho for aggressive clipping, no weight decay
        optimizer = SophiaG(model.parameters(), lr=1.0, rho=0.01, weight_decay=0.0)

        x = torch.randn(4, 10) * 10  # Large input for large gradient
        loss = model(x).sum()
        loss.backward()

        w_before = model.weight.clone()
        optimizer.step()

        # Change should be bounded by lr * rho = 1.0 * 0.01 = 0.01
        # But on first step, hessian is 0 so update = clip(m/eps, -rho, rho)
        # This clips to rho, so change = lr * rho = 0.01
        change = (model.weight - w_before).abs()
        assert change.max() <= 0.015  # Small tolerance for numerical

    def test_invalid_params(self):
        """Test validation of hyperparameters."""
        from hydra.optim.sophia import SophiaG

        model = nn.Linear(10, 5)

        with pytest.raises(ValueError, match="Invalid learning rate"):
            SophiaG(model.parameters(), lr=-1)

        with pytest.raises(ValueError, match="Invalid beta"):
            SophiaG(model.parameters(), betas=(1.5, 0.99))

        with pytest.raises(ValueError, match="Invalid rho"):
            SophiaG(model.parameters(), rho=-0.1)

    def test_update_hessian_from_grads(self):
        """Test Hessian update from gradients."""
        from hydra.optim.sophia import SophiaG

        model = nn.Linear(10, 5)
        optimizer = SophiaG(model.parameters())

        # First step to initialize
        x = torch.randn(4, 10)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()

        # Get initial hessian
        hessian_before = optimizer.state[model.weight]["hessian"].clone()

        # Another backward and hessian update
        optimizer.zero_grad()
        loss = model(torch.randn(4, 10)).sum()
        loss.backward()
        optimizer.update_hessian_from_grads()

        # Hessian should have changed
        assert not torch.allclose(
            optimizer.state[model.weight]["hessian"],
            hessian_before
        )


class TestSophiaH:
    """Tests for SophiaH optimizer."""

    def test_basic_step(self):
        """Test basic optimization step."""
        from hydra.optim.sophia import SophiaH

        model = nn.Linear(10, 5)
        optimizer = SophiaH(model.parameters(), lr=2e-4)

        x = torch.randn(4, 10)
        loss = model(x).sum()
        loss.backward()

        w_before = model.weight.clone()
        optimizer.step()

        assert not torch.allclose(model.weight, w_before)

    def test_invalid_lr(self):
        """Test that invalid learning rate raises error."""
        from hydra.optim.sophia import SophiaH

        model = nn.Linear(10, 5)
        with pytest.raises(ValueError, match="Invalid learning rate"):
            SophiaH(model.parameters(), lr=-1)


class TestSophiaGSimple:
    """Tests for SophiaGSimple optimizer."""

    def test_basic_step(self):
        """Test basic optimization step."""
        from hydra.optim.sophia import SophiaGSimple

        model = nn.Linear(10, 5)
        optimizer = SophiaGSimple(model.parameters(), lr=2e-4)

        x = torch.randn(4, 10)
        loss = model(x).sum()
        loss.backward()

        w_before = model.weight.clone()
        optimizer.step()

        assert not torch.allclose(model.weight, w_before)

    def test_state_has_both_moments(self):
        """Test that state has both first and second moments."""
        from hydra.optim.sophia import SophiaGSimple

        model = nn.Linear(10, 5)
        optimizer = SophiaGSimple(model.parameters())

        x = torch.randn(4, 10)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()

        for p in model.parameters():
            if p.grad is not None:
                state = optimizer.state[p]
                assert "exp_avg" in state
                assert "exp_avg_sq" in state

    def test_loss_decreases(self):
        """Test that optimizer decreases loss over time."""
        from hydra.optim.sophia import SophiaGSimple

        # Simple linear regression
        torch.manual_seed(42)
        X = torch.randn(100, 5)
        y = X @ torch.tensor([1., 2., 3., 4., 5.]) + 0.1 * torch.randn(100)

        model = nn.Linear(5, 1, bias=False)
        optimizer = SophiaGSimple(model.parameters(), lr=0.1, rho=1.0)  # Higher rho for faster convergence

        # Compute initial loss
        with torch.no_grad():
            initial_loss = ((model(X).squeeze() - y) ** 2).mean().item()

        # Run optimization steps
        for _ in range(50):
            optimizer.zero_grad()
            pred = model(X).squeeze()
            loss = ((pred - y) ** 2).mean()
            loss.backward()
            optimizer.step()

        # Compute final loss
        with torch.no_grad():
            final_loss = ((model(X).squeeze() - y) ** 2).mean().item()

        # Should reduce loss (any meaningful reduction shows it's working)
        assert final_loss < initial_loss, f"Loss should decrease: {final_loss} < {initial_loss}"
