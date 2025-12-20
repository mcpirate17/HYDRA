"""
Test suite for lightning-attention library (triton & torch implementations).
Tests forward pass, backward pass, gradient flow, and numerical stability.
"""

import pytest
import torch
import torch.nn.functional as F
from torch.testing import assert_close

# Import lightning_attn components
try:
    from lightning_attn.ops.triton.lightning_attn2_no_decay import LightningAttention2NoDecay
    HAS_LIGHTNING_ATTN = True
except ImportError:
    HAS_LIGHTNING_ATTN = False


@pytest.mark.skipif(not HAS_LIGHTNING_ATTN, reason="lightning-attention not installed")
class TestLightningAttention2NoDecay:
    """Test the lightning_attn2_no_decay implementation."""

    @pytest.fixture
    def attention_inputs(self):
        """Create standard attention inputs for testing."""
        torch.manual_seed(42)
        batch = 2
        heads = 8
        seq_len = 512
        head_dim = 64
        value_dim = 64

        q = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(batch, heads, seq_len, value_dim, dtype=torch.bfloat16, device="cuda")

        return q, k, v

    def test_forward_shape(self, attention_inputs):
        """Test that forward pass produces correct output shape."""
        q, k, v = attention_inputs
        b, h, n, e = v.shape

        output = LightningAttention2NoDecay.apply(q, k, v)

        assert output.shape == (b, h, n, e), f"Expected shape {(b, h, n, e)}, got {output.shape}"
        assert output.dtype == q.dtype, f"Expected dtype {q.dtype}, got {output.dtype}"

    def test_forward_backward(self, attention_inputs):
        """Test that backward pass runs without errors."""
        q, k, v = attention_inputs
        q_req = q.clone().detach().requires_grad_(True)
        k_req = k.clone().detach().requires_grad_(True)
        v_req = v.clone().detach().requires_grad_(True)

        output = LightningAttention2NoDecay.apply(q_req, k_req, v_req)
        loss = output.sum()

        # Backward pass
        loss.backward()

        assert q_req.grad is not None, "q gradient is None"
        assert k_req.grad is not None, "k gradient is None"
        assert v_req.grad is not None, "v gradient is None"

    def test_gradient_magnitude(self, attention_inputs):
        """Test that gradients have reasonable magnitude (not NaN or Inf)."""
        q, k, v = attention_inputs
        q_req = q.clone().detach().requires_grad_(True)
        k_req = k.clone().detach().requires_grad_(True)
        v_req = v.clone().detach().requires_grad_(True)

        output = LightningAttention2NoDecay.apply(q_req, k_req, v_req)
        loss = output.sum()
        loss.backward()

        for grad, name in [(q_req.grad, "q"), (k_req.grad, "k"), (v_req.grad, "v")]:
            assert not torch.isnan(grad).any(), f"{name} gradient contains NaN"
            assert not torch.isinf(grad).any(), f"{name} gradient contains Inf"
            assert grad.abs().max() > 0, f"{name} gradient is all zeros"

    def test_small_input(self):
        """Test with small sequence length to verify kernel compilation."""
        torch.manual_seed(42)
        batch, heads, seq_len, head_dim, value_dim = 1, 4, 64, 64, 64

        q = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(batch, heads, seq_len, value_dim, dtype=torch.bfloat16, device="cuda")

        output = LightningAttention2NoDecay.apply(q, k, v)

        assert output.shape == (batch, heads, seq_len, value_dim)
        assert not torch.isnan(output).any()

    def test_large_input(self):
        """Test with larger sequence length and batch size."""
        torch.manual_seed(42)
        batch, heads, seq_len, head_dim, value_dim = 4, 16, 1024, 64, 64

        q = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(batch, heads, seq_len, value_dim, dtype=torch.bfloat16, device="cuda")

        output = LightningAttention2NoDecay.apply(q, k, v)

        assert output.shape == (batch, heads, seq_len, value_dim)
        assert not torch.isnan(output).any()

    def test_different_value_dims(self):
        """Test with various value embedding dimensions."""
        torch.manual_seed(42)
        batch, heads, seq_len, head_dim = 2, 8, 256, 64

        for value_dim in [32, 64, 128, 256]:
            v = torch.randn(batch, heads, seq_len, value_dim, dtype=torch.bfloat16, device="cuda")
            q = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.bfloat16, device="cuda")
            k = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.bfloat16, device="cuda")

            output = LightningAttention2NoDecay.apply(q, k, v)

            assert output.shape == (batch, heads, seq_len, value_dim), \
                f"Failed for value_dim={value_dim}"
            assert not torch.isnan(output).any(), \
                f"Output contains NaN for value_dim={value_dim}"

    def test_determinism(self, attention_inputs):
        """Test that running the same input twice produces same output."""
        q, k, v = attention_inputs

        output1 = LightningAttention2NoDecay.apply(
            q.clone(), k.clone(), v.clone()
        )
        output2 = LightningAttention2NoDecay.apply(
            q.clone(), k.clone(), v.clone()
        )

        # Note: bfloat16 accumulation may have small differences
        assert_close(output1, output2, rtol=1e-2, atol=1e-4)

    def test_forward_float32(self):
        """Test that float32 tensors are handled correctly."""
        torch.manual_seed(42)
        batch, heads, seq_len, head_dim, value_dim = 2, 8, 256, 64, 64

        q = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float32, device="cuda")
        k = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float32, device="cuda")
        v = torch.randn(batch, heads, seq_len, value_dim, dtype=torch.float32, device="cuda")

        output = LightningAttention2NoDecay.apply(q, k, v)

        assert output.shape == (batch, heads, seq_len, value_dim)
        assert output.dtype == torch.float32

    def test_numerical_stability_large_seq(self):
        """Test numerical stability with very long sequences."""
        torch.manual_seed(42)
        batch, heads, seq_len, head_dim, value_dim = 1, 8, 2048, 64, 64

        q = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.bfloat16, device="cuda") * 0.1
        k = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.bfloat16, device="cuda") * 0.1
        v = torch.randn(batch, heads, seq_len, value_dim, dtype=torch.bfloat16, device="cuda") * 0.1

        output = LightningAttention2NoDecay.apply(q, k, v)

        assert not torch.isnan(output).any(), "Output contains NaN for large sequence"
        assert not torch.isinf(output).any(), "Output contains Inf for large sequence"
        assert output.abs().max() < 1e2, "Output magnitude exploded"

    def test_backward_numerical_stability(self):
        """Test backward pass numerical stability."""
        torch.manual_seed(42)
        batch, heads, seq_len, head_dim, value_dim = 2, 8, 512, 64, 64

        q = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.bfloat16, device="cuda")
        q.requires_grad_(True)
        k = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.bfloat16, device="cuda")
        k.requires_grad_(True)
        v = torch.randn(batch, heads, seq_len, value_dim, dtype=torch.bfloat16, device="cuda")
        v.requires_grad_(True)

        output = LightningAttention2NoDecay.apply(q, k, v)
        loss = (output ** 2).sum()
        loss.backward()

        assert not torch.isnan(q.grad).any(), "q grad contains NaN"
        assert not torch.isinf(q.grad).any(), "q grad contains Inf"
        assert not torch.isnan(k.grad).any(), "k grad contains NaN"
        assert not torch.isinf(k.grad).any(), "k grad contains Inf"
        assert not torch.isnan(v.grad).any(), "v grad contains NaN"
        assert not torch.isinf(v.grad).any(), "v grad contains Inf"


@pytest.mark.skipif(not HAS_LIGHTNING_ATTN, reason="lightning-attention not installed")
class TestLightningAttentionInterface:
    """Test the high-level lightning_attn interface."""

    def test_interface_import(self):
        """Test that we can import the interface."""
        try:
            from lightning_attn.ops.lightning_attn_interface import lightning_attn_func
            assert callable(lightning_attn_func)
        except ImportError as e:
            pytest.skip(f"Could not import interface: {e}")

    def test_interface_forward(self):
        """Test interface forward pass."""
        try:
            from lightning_attn.ops.lightning_attn_interface import lightning_attn_func
        except ImportError as e:
            pytest.skip(f"Could not import interface: {e}")

        torch.manual_seed(42)
        batch, heads, seq_len, head_dim, value_dim = 2, 8, 256, 64, 64

        q = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(batch, heads, seq_len, value_dim, dtype=torch.bfloat16, device="cuda")

        output = lightning_attn_func(q, k, v)

        assert output.shape == (batch, heads, seq_len, value_dim)
        assert not torch.isnan(output).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
