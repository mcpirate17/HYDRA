#!/usr/bin/env python3
"""Tests for fused backward Triton kernels (RMSNorm, QK-Norm).

This validates the Triton backward kernels against the PyTorch reference
implementations and benchmarks the performance improvement.

Usage:
    source /home/tim/venvs/llm/bin/activate && pytest tests/test_fused_backward_kernels.py -v
"""

import pytest
import torch
import time
from typing import Tuple


# Skip all tests if CUDA is not available
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(autouse=True)
def setup_cuda():
    """Ensure CUDA is ready before each test."""
    torch.cuda.synchronize()
    yield
    torch.cuda.synchronize()


# =============================================================================
# RMSNorm Backward Tests
# =============================================================================


class TestFusedRMSNormBackward:
    """Test suite for fused RMSNorm backward kernel."""

    @pytest.fixture
    def sample_tensors(self):
        """Create sample tensors for testing."""
        torch.manual_seed(42)
        batch_size, seq_len, dim = 4, 512, 1792  # 500M model dims
        x = torch.randn(batch_size, seq_len, dim, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(dim, device="cuda", dtype=torch.bfloat16)
        grad_output = torch.randn(batch_size, seq_len, dim, device="cuda", dtype=torch.bfloat16)
        return x, weight, grad_output

    def test_import(self):
        """Test that the module imports correctly."""
        from hydra.kernels.fused_ops import (
            _fused_rms_norm_backward_triton,
            _rms_norm_backward_pytorch,
            USE_FUSED_RMS_NORM_BACKWARD,
            TRITON_AVAILABLE,
        )
        assert TRITON_AVAILABLE, "Triton should be available"
        assert USE_FUSED_RMS_NORM_BACKWARD, "Fused RMSNorm backward should be enabled by default"

    def test_numerical_correctness_bf16(self, sample_tensors):
        """Test Triton backward matches PyTorch reference for BF16."""
        from hydra.kernels.fused_ops import (
            _fused_rms_norm_backward_triton,
            _rms_norm_backward_pytorch,
        )

        x, weight, grad_output = sample_tensors
        eps = 1e-6
        orig_shape = x.shape
        dim = x.shape[-1]
        x_flat = x.contiguous().view(-1, dim)

        # PyTorch reference
        grad_x_ref, grad_weight_ref, _ = _rms_norm_backward_pytorch(
            x_flat, weight, grad_output, eps, orig_shape
        )

        # Triton implementation
        grad_x_triton, grad_weight_triton, _ = _fused_rms_norm_backward_triton(
            x_flat, weight, grad_output, eps, orig_shape
        )

        # Check numerical match (BF16 has lower precision)
        # grad_x: per-element, tight tolerance
        torch.testing.assert_close(
            grad_x_triton.float(), grad_x_ref.float(),
            rtol=1e-2, atol=1e-2,
            msg="grad_x mismatch"
        )
        # grad_weight: accumulated over n_rows in BF16, needs looser tolerance
        torch.testing.assert_close(
            grad_weight_triton.float(), grad_weight_ref.float(),
            rtol=0.05, atol=1.0,  # Larger tolerance for accumulated reduction
            msg="grad_weight mismatch"
        )

    def test_numerical_correctness_fp32(self):
        """Test Triton backward matches PyTorch reference for FP32."""
        from hydra.kernels.fused_ops import (
            _fused_rms_norm_backward_triton,
            _rms_norm_backward_pytorch,
        )

        torch.manual_seed(42)
        batch_size, seq_len, dim = 4, 256, 768
        x = torch.randn(batch_size, seq_len, dim, device="cuda", dtype=torch.float32)
        weight = torch.randn(dim, device="cuda", dtype=torch.float32)
        grad_output = torch.randn(batch_size, seq_len, dim, device="cuda", dtype=torch.float32)
        eps = 1e-6
        orig_shape = x.shape
        x_flat = x.contiguous().view(-1, dim)

        # PyTorch reference
        grad_x_ref, grad_weight_ref, _ = _rms_norm_backward_pytorch(
            x_flat, weight, grad_output, eps, orig_shape
        )

        # Triton implementation
        grad_x_triton, grad_weight_triton, _ = _fused_rms_norm_backward_triton(
            x_flat, weight, grad_output, eps, orig_shape
        )

        # FP32 should have tighter tolerance
        torch.testing.assert_close(
            grad_x_triton, grad_x_ref,
            rtol=1e-3, atol=1e-3,
            msg="grad_x mismatch (FP32)"
        )
        torch.testing.assert_close(
            grad_weight_triton, grad_weight_ref,
            rtol=1e-3, atol=1e-3,
            msg="grad_weight mismatch (FP32)"
        )

    def test_gradient_clamping(self):
        """Test that gradients are properly clamped to [-100, 100]."""
        from hydra.kernels.fused_ops import _fused_rms_norm_backward_triton

        torch.manual_seed(42)
        # Create inputs that would produce large gradients
        x = torch.full((4, 256, 768), 1000.0, device="cuda", dtype=torch.bfloat16)
        x_flat = x.view(-1, 768)
        weight = torch.full((768,), 10.0, device="cuda", dtype=torch.bfloat16)
        grad_output = torch.full_like(x, 100.0)
        eps = 1e-6
        orig_shape = x.shape

        grad_x, grad_weight, _ = _fused_rms_norm_backward_triton(
            x_flat, weight, grad_output, eps, orig_shape
        )

        # Check clamping
        assert grad_x.abs().max() <= 100.0, f"grad_x not clamped: {grad_x.abs().max()}"
        assert grad_weight.abs().max() <= 100.0, f"grad_weight not clamped: {grad_weight.abs().max()}"

    def test_different_shapes(self):
        """Test with various tensor shapes."""
        from hydra.kernels.fused_ops import (
            _fused_rms_norm_backward_triton,
            _rms_norm_backward_pytorch,
        )

        shapes = [
            (1, 1, 64),
            (2, 128, 768),
            (4, 512, 1792),
            (8, 1024, 2048),
        ]

        for shape in shapes:
            torch.manual_seed(42)
            batch_size, seq_len, dim = shape
            x = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
            x_flat = x.contiguous().view(-1, dim)
            weight = torch.randn(dim, device="cuda", dtype=torch.bfloat16)
            grad_output = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
            eps = 1e-6
            orig_shape = x.shape

            grad_x_ref, grad_weight_ref, _ = _rms_norm_backward_pytorch(
                x_flat, weight, grad_output, eps, orig_shape
            )
            grad_x_triton, grad_weight_triton, _ = _fused_rms_norm_backward_triton(
                x_flat, weight, grad_output, eps, orig_shape
            )

            torch.testing.assert_close(
                grad_x_triton.float(), grad_x_ref.float(),
                rtol=1e-2, atol=1e-2,
                msg=f"grad_x mismatch for shape {shape}"
            )

    def test_autograd_integration(self):
        """Test that the backward integrates correctly with autograd."""
        from hydra.kernels.fused_ops import fused_rms_norm, USE_FUSED_RMS_NORM_BACKWARD

        torch.manual_seed(42)
        x = torch.randn(4, 512, 768, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        weight = torch.randn(768, device="cuda", dtype=torch.bfloat16, requires_grad=True)

        # Forward
        out = fused_rms_norm(x, weight, eps=1e-6)
        loss = out.sum()

        # Backward
        loss.backward()

        # Check gradients exist and are reasonable
        assert x.grad is not None, "x.grad should not be None"
        assert weight.grad is not None, "weight.grad should not be None"
        assert not x.grad.isnan().any(), "x.grad contains NaN"
        assert not weight.grad.isnan().any(), "weight.grad contains NaN"
        assert not x.grad.isinf().any(), "x.grad contains Inf"
        assert not weight.grad.isinf().any(), "weight.grad contains Inf"


# =============================================================================
# QK-Norm Backward Tests
# =============================================================================


class TestFusedQKNormBackward:
    """Test suite for fused QK-Norm backward kernel."""

    @pytest.fixture
    def sample_tensors(self):
        """Create sample tensors for testing."""
        torch.manual_seed(42)
        batch_size, n_heads, seq_len, head_dim = 4, 28, 512, 64
        n_kv_heads = 4
        q = torch.randn(batch_size, n_heads, seq_len, head_dim, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(batch_size, n_kv_heads, seq_len, head_dim, device="cuda", dtype=torch.bfloat16)
        grad_q = torch.randn_like(q)
        grad_k = torch.randn_like(k)
        return q, k, grad_q, grad_k

    def test_import(self):
        """Test that the module imports correctly."""
        from hydra.kernels.fused_ops import (
            _fused_qk_norm_backward_triton,
            _qk_norm_backward_pytorch,
            USE_FUSED_QK_NORM_BACKWARD,
            TRITON_AVAILABLE,
        )
        assert TRITON_AVAILABLE, "Triton should be available"
        assert USE_FUSED_QK_NORM_BACKWARD, "Fused QK-Norm backward should be enabled by default"

    def test_numerical_correctness_bf16(self, sample_tensors):
        """Test Triton backward matches PyTorch reference for BF16."""
        from hydra.kernels.fused_ops import (
            _fused_qk_norm_backward_triton,
            _qk_norm_backward_pytorch,
        )
        import math

        q, k, grad_q_out, grad_k_out = sample_tensors
        head_dim = q.shape[-1]
        scale = math.sqrt(head_dim)
        temperature = 1.0

        # PyTorch reference
        grad_q_ref, grad_k_ref, _, _ = _qk_norm_backward_pytorch(
            q, k, grad_q_out, grad_k_out, scale, temperature
        )

        # Triton implementation
        grad_q_triton, grad_k_triton, _, _ = _fused_qk_norm_backward_triton(
            q, k, grad_q_out, grad_k_out, scale, temperature
        )

        # Check numerical match
        torch.testing.assert_close(
            grad_q_triton.float(), grad_q_ref.float(),
            rtol=1e-2, atol=1e-2,
            msg="grad_q mismatch"
        )
        torch.testing.assert_close(
            grad_k_triton.float(), grad_k_ref.float(),
            rtol=1e-2, atol=1e-2,
            msg="grad_k mismatch"
        )

    def test_numerical_correctness_fp32(self):
        """Test Triton backward matches PyTorch reference for FP32."""
        from hydra.kernels.fused_ops import (
            _fused_qk_norm_backward_triton,
            _qk_norm_backward_pytorch,
        )
        import math

        torch.manual_seed(42)
        batch_size, n_heads, seq_len, head_dim = 4, 12, 256, 64
        n_kv_heads = 4
        q = torch.randn(batch_size, n_heads, seq_len, head_dim, device="cuda", dtype=torch.float32)
        k = torch.randn(batch_size, n_kv_heads, seq_len, head_dim, device="cuda", dtype=torch.float32)
        grad_q_out = torch.randn_like(q)
        grad_k_out = torch.randn_like(k)
        scale = math.sqrt(head_dim)
        temperature = 1.0

        # PyTorch reference
        grad_q_ref, grad_k_ref, _, _ = _qk_norm_backward_pytorch(
            q, k, grad_q_out, grad_k_out, scale, temperature
        )

        # Triton implementation
        grad_q_triton, grad_k_triton, _, _ = _fused_qk_norm_backward_triton(
            q, k, grad_q_out, grad_k_out, scale, temperature
        )

        # FP32 should have tighter tolerance
        torch.testing.assert_close(
            grad_q_triton, grad_q_ref,
            rtol=1e-3, atol=1e-3,
            msg="grad_q mismatch (FP32)"
        )
        torch.testing.assert_close(
            grad_k_triton, grad_k_ref,
            rtol=1e-3, atol=1e-3,
            msg="grad_k mismatch (FP32)"
        )

    def test_gradient_clamping(self):
        """Test that gradients are properly clamped to [-100, 100]."""
        from hydra.kernels.fused_ops import _fused_qk_norm_backward_triton
        import math

        torch.manual_seed(42)
        # Create inputs that would produce large gradients
        q = torch.full((4, 12, 256, 64), 0.001, device="cuda", dtype=torch.bfloat16)
        k = torch.full((4, 4, 256, 64), 0.001, device="cuda", dtype=torch.bfloat16)
        grad_q_out = torch.full_like(q, 1000.0)
        grad_k_out = torch.full_like(k, 1000.0)
        scale = math.sqrt(64) * 100  # Large scale
        temperature = 10.0

        grad_q, grad_k, _, _ = _fused_qk_norm_backward_triton(
            q, k, grad_q_out, grad_k_out, scale, temperature
        )

        # Check clamping
        assert grad_q.abs().max() <= 100.0, f"grad_q not clamped: {grad_q.abs().max()}"
        assert grad_k.abs().max() <= 100.0, f"grad_k not clamped: {grad_k.abs().max()}"

    def test_different_shapes(self):
        """Test with various tensor shapes."""
        from hydra.kernels.fused_ops import (
            _fused_qk_norm_backward_triton,
            _qk_norm_backward_pytorch,
        )
        import math

        shapes = [
            # (batch, n_heads, seq, head_dim, n_kv_heads)
            (1, 1, 32, 64, 1),
            (2, 8, 128, 64, 2),
            (4, 28, 512, 64, 4),
            (8, 32, 1024, 128, 8),
        ]

        for batch, n_heads, seq, head_dim, n_kv_heads in shapes:
            torch.manual_seed(42)
            q = torch.randn(batch, n_heads, seq, head_dim, device="cuda", dtype=torch.bfloat16)
            k = torch.randn(batch, n_kv_heads, seq, head_dim, device="cuda", dtype=torch.bfloat16)
            grad_q_out = torch.randn_like(q)
            grad_k_out = torch.randn_like(k)
            scale = math.sqrt(head_dim)
            temperature = 1.0

            grad_q_ref, grad_k_ref, _, _ = _qk_norm_backward_pytorch(
                q, k, grad_q_out, grad_k_out, scale, temperature
            )
            grad_q_triton, grad_k_triton, _, _ = _fused_qk_norm_backward_triton(
                q, k, grad_q_out, grad_k_out, scale, temperature
            )

            torch.testing.assert_close(
                grad_q_triton.float(), grad_q_ref.float(),
                rtol=1e-2, atol=1e-2,
                msg=f"grad_q mismatch for shape ({batch}, {n_heads}, {seq}, {head_dim})"
            )

    def test_autograd_integration(self):
        """Test that the backward integrates correctly with autograd."""
        from hydra.kernels.fused_ops import fused_qk_norm
        import math

        torch.manual_seed(42)
        q = torch.randn(4, 12, 256, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(4, 4, 256, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        scale = math.sqrt(64)

        # Forward
        q_out, k_out = fused_qk_norm(q, k, scale, temperature=1.0)
        loss = q_out.sum() + k_out.sum()

        # Backward
        loss.backward()

        # Check gradients exist and are reasonable
        assert q.grad is not None, "q.grad should not be None"
        assert k.grad is not None, "k.grad should not be None"
        assert not q.grad.isnan().any(), "q.grad contains NaN"
        assert not k.grad.isnan().any(), "k.grad contains NaN"
        assert not q.grad.isinf().any(), "q.grad contains Inf"
        assert not k.grad.isinf().any(), "k.grad contains Inf"


# =============================================================================
# Pretest Functions (for HYDRA pretest system)
# =============================================================================


def run_rms_norm_pretest():
    """Run pretest validation for RMSNorm backward - called by HYDRA's pretest system."""
    from hydra.kernels.fused_ops import (
        _fused_rms_norm_backward_triton,
        _rms_norm_backward_pytorch,
        USE_FUSED_RMS_NORM_BACKWARD,
    )

    if not USE_FUSED_RMS_NORM_BACKWARD:
        return {"status": "skipped", "reason": "Fused RMSNorm backward disabled"}

    # Test on 500M model dimensions
    torch.manual_seed(42)
    x = torch.randn(4, 512, 1792, device="cuda", dtype=torch.bfloat16)
    x_flat = x.contiguous().view(-1, 1792)
    weight = torch.randn(1792, device="cuda", dtype=torch.bfloat16)
    grad_output = torch.randn(4, 512, 1792, device="cuda", dtype=torch.bfloat16)
    eps = 1e-6
    orig_shape = x.shape

    # Reference
    grad_x_ref, grad_weight_ref, _ = _rms_norm_backward_pytorch(
        x_flat, weight, grad_output, eps, orig_shape
    )

    # Triton
    grad_x_triton, grad_weight_triton, _ = _fused_rms_norm_backward_triton(
        x_flat, weight, grad_output, eps, orig_shape
    )

    # Check - grad_x should be tight, grad_weight has accumulated BF16 error
    x_diff = (grad_x_triton.float() - grad_x_ref.float()).abs().max().item()
    w_diff = (grad_weight_triton.float() - grad_weight_ref.float()).abs().max().item()

    # grad_x: tight tolerance (per-element)
    # grad_weight: looser tolerance (accumulated over n_rows in BF16)
    passed = x_diff < 0.1 and w_diff < 1.0

    return {
        "status": "passed" if passed else "failed",
        "grad_x_max_diff": x_diff,
        "grad_weight_max_diff": w_diff,
        "grad_x_tolerance": 0.1,
        "grad_weight_tolerance": 1.0,
    }


def run_qk_norm_pretest():
    """Run pretest validation for QK-Norm backward - called by HYDRA's pretest system."""
    from hydra.kernels.fused_ops import (
        _fused_qk_norm_backward_triton,
        _qk_norm_backward_pytorch,
        USE_FUSED_QK_NORM_BACKWARD,
    )
    import math

    if not USE_FUSED_QK_NORM_BACKWARD:
        return {"status": "skipped", "reason": "Fused QK-Norm backward disabled"}

    # Test on 500M model dimensions
    torch.manual_seed(42)
    q = torch.randn(4, 28, 512, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(4, 4, 512, 64, device="cuda", dtype=torch.bfloat16)
    grad_q_out = torch.randn_like(q)
    grad_k_out = torch.randn_like(k)
    scale = math.sqrt(64)
    temperature = 1.0

    # Reference
    grad_q_ref, grad_k_ref, _, _ = _qk_norm_backward_pytorch(
        q, k, grad_q_out, grad_k_out, scale, temperature
    )

    # Triton
    grad_q_triton, grad_k_triton, _, _ = _fused_qk_norm_backward_triton(
        q, k, grad_q_out, grad_k_out, scale, temperature
    )

    # Check
    q_diff = (grad_q_triton.float() - grad_q_ref.float()).abs().max().item()
    k_diff = (grad_k_triton.float() - grad_k_ref.float()).abs().max().item()

    passed = q_diff < 0.1 and k_diff < 0.1

    return {
        "status": "passed" if passed else "failed",
        "grad_q_max_diff": q_diff,
        "grad_k_max_diff": k_diff,
        "tolerance": 0.1,
    }


if __name__ == "__main__":
    # Quick sanity check when run directly
    print("Running RMSNorm backward pretest...")
    result = run_rms_norm_pretest()
    print(f"Result: {result}")
    if result["status"] == "passed":
        print("RMSNorm pretest passed!")
    else:
        print(f"RMSNorm pretest failed: {result}")
        exit(1)

    print("\nRunning QK-Norm backward pretest...")
    result = run_qk_norm_pretest()
    print(f"Result: {result}")
    if result["status"] == "passed":
        print("QK-Norm pretest passed!")
    else:
        print(f"QK-Norm pretest failed: {result}")
        exit(1)

    print("\nAll pretests passed!")
