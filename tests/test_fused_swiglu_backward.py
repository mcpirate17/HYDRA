#!/usr/bin/env python3
"""Tests for fused SwiGLU backward Triton kernel.

This validates the Triton backward kernel against the PyTorch reference
implementation and benchmarks the performance improvement.

Usage:
    source /home/tim/venvs/llm/bin/activate && pytest tests/test_fused_swiglu_backward.py -v
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


class TestFusedSwiGLUBackward:
    """Test suite for fused SwiGLU backward kernel."""

    @pytest.fixture
    def sample_tensors(self):
        """Create sample tensors for testing."""
        torch.manual_seed(42)
        batch_size, seq_len, hidden_dim = 4, 512, 1792  # 500M model dims
        gate = torch.randn(batch_size, seq_len, hidden_dim, device="cuda", dtype=torch.bfloat16)
        up = torch.randn(batch_size, seq_len, hidden_dim, device="cuda", dtype=torch.bfloat16)
        grad_output = torch.randn(batch_size, seq_len, hidden_dim, device="cuda", dtype=torch.bfloat16)
        return gate, up, grad_output

    def test_import(self):
        """Test that the module imports correctly."""
        from hydra.kernels.fused_ops import (
            _fused_swiglu_backward_triton,
            _swiglu_backward_pytorch,
            TRITON_AVAILABLE,
        )
        assert TRITON_AVAILABLE, "Triton should be available"
        # Note: USE_FUSED_SWIGLU_BACKWARD depends on env vars at import time,
        # which can vary based on test order. We just verify imports work.

    def test_numerical_correctness_bf16(self, sample_tensors):
        """Test Triton backward matches PyTorch reference for BF16."""
        from hydra.kernels.fused_ops import (
            _fused_swiglu_backward_triton,
            _swiglu_backward_pytorch,
        )

        gate, up, grad_output = sample_tensors

        # PyTorch reference
        grad_gate_ref, grad_up_ref = _swiglu_backward_pytorch(gate, up, grad_output)

        # Triton implementation
        grad_gate_triton, grad_up_triton = _fused_swiglu_backward_triton(gate, up, grad_output)

        # Check numerical match (BF16 has lower precision, use appropriate tolerance)
        torch.testing.assert_close(
            grad_gate_triton.float(), grad_gate_ref.float(),
            rtol=1e-2, atol=1e-2,
            msg="grad_gate mismatch"
        )
        torch.testing.assert_close(
            grad_up_triton.float(), grad_up_ref.float(),
            rtol=1e-2, atol=1e-2,
            msg="grad_up mismatch"
        )

    def test_numerical_correctness_fp32(self):
        """Test Triton backward matches PyTorch reference for FP32."""
        from hydra.kernels.fused_ops import (
            _fused_swiglu_backward_triton,
            _swiglu_backward_pytorch,
        )

        torch.manual_seed(42)
        batch_size, seq_len, hidden_dim = 4, 512, 768
        gate = torch.randn(batch_size, seq_len, hidden_dim, device="cuda", dtype=torch.float32)
        up = torch.randn(batch_size, seq_len, hidden_dim, device="cuda", dtype=torch.float32)
        grad_output = torch.randn(batch_size, seq_len, hidden_dim, device="cuda", dtype=torch.float32)

        # PyTorch reference
        grad_gate_ref, grad_up_ref = _swiglu_backward_pytorch(gate, up, grad_output)

        # Triton implementation
        grad_gate_triton, grad_up_triton = _fused_swiglu_backward_triton(gate, up, grad_output)

        # FP32 should have tighter tolerance
        torch.testing.assert_close(
            grad_gate_triton, grad_gate_ref,
            rtol=1e-4, atol=1e-4,
            msg="grad_gate mismatch (FP32)"
        )
        torch.testing.assert_close(
            grad_up_triton, grad_up_ref,
            rtol=1e-4, atol=1e-4,
            msg="grad_up mismatch (FP32)"
        )

    def test_gradient_clamping(self):
        """Test that gradients are properly clamped to [-100, 100]."""
        from hydra.kernels.fused_ops import (
            _fused_swiglu_backward_triton,
            _swiglu_backward_pytorch,
        )

        # Create inputs that would produce large gradients
        torch.manual_seed(42)
        gate = torch.full((4, 512, 768), 50.0, device="cuda", dtype=torch.bfloat16)
        up = torch.full((4, 512, 768), 100.0, device="cuda", dtype=torch.bfloat16)
        grad_output = torch.full((4, 512, 768), 10.0, device="cuda", dtype=torch.bfloat16)

        grad_gate, grad_up = _fused_swiglu_backward_triton(gate, up, grad_output)

        # Check clamping
        assert grad_gate.abs().max() <= 100.0, f"grad_gate not clamped: {grad_gate.abs().max()}"
        assert grad_up.abs().max() <= 100.0, f"grad_up not clamped: {grad_up.abs().max()}"

    def test_different_shapes(self):
        """Test with various tensor shapes."""
        from hydra.kernels.fused_ops import (
            _fused_swiglu_backward_triton,
            _swiglu_backward_pytorch,
        )

        shapes = [
            (1, 1, 64),       # Minimal
            (2, 128, 768),    # Small model
            (4, 512, 1792),   # 500M model
            (8, 1024, 2048),  # Large
            (1, 2048, 4096),  # Long sequence
        ]

        for shape in shapes:
            torch.manual_seed(42)
            gate = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
            up = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
            grad_output = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)

            grad_gate_ref, grad_up_ref = _swiglu_backward_pytorch(gate, up, grad_output)
            grad_gate_triton, grad_up_triton = _fused_swiglu_backward_triton(gate, up, grad_output)

            torch.testing.assert_close(
                grad_gate_triton.float(), grad_gate_ref.float(),
                rtol=1e-2, atol=1e-2,
                msg=f"grad_gate mismatch for shape {shape}"
            )

    def test_autograd_integration(self):
        """Test that the backward integrates correctly with autograd."""
        from hydra.kernels.fused_ops import fused_swiglu, USE_FUSED_SWIGLU_BACKWARD

        torch.manual_seed(42)
        gate = torch.randn(4, 512, 768, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        up = torch.randn(4, 512, 768, device="cuda", dtype=torch.bfloat16, requires_grad=True)

        # Forward
        out = fused_swiglu(gate, up)
        loss = out.sum()

        # Backward
        loss.backward()

        # Check gradients exist and are reasonable
        assert gate.grad is not None, "gate.grad should not be None"
        assert up.grad is not None, "up.grad should not be None"
        assert not gate.grad.isnan().any(), "gate.grad contains NaN"
        assert not up.grad.isnan().any(), "up.grad contains NaN"
        assert not gate.grad.isinf().any(), "gate.grad contains Inf"
        assert not up.grad.isinf().any(), "up.grad contains Inf"


class TestFusedSwiGLUBackwardBenchmark:
    """Benchmark suite for fused SwiGLU backward kernel."""

    @pytest.mark.slow
    def test_benchmark_500m_model(self):
        """Benchmark on 500M model dimensions."""
        from hydra.kernels.fused_ops import (
            _fused_swiglu_backward_triton,
            _swiglu_backward_pytorch,
        )

        # 500M model typical sizes
        batch_size, seq_len, hidden_dim = 4, 1024, 1792 * 4  # MLP hidden = 4x model dim
        warmup_iters = 10
        bench_iters = 100

        torch.manual_seed(42)
        gate = torch.randn(batch_size, seq_len, hidden_dim, device="cuda", dtype=torch.bfloat16)
        up = torch.randn(batch_size, seq_len, hidden_dim, device="cuda", dtype=torch.bfloat16)
        grad_output = torch.randn(batch_size, seq_len, hidden_dim, device="cuda", dtype=torch.bfloat16)

        # Warmup PyTorch
        for _ in range(warmup_iters):
            _swiglu_backward_pytorch(gate, up, grad_output)
        torch.cuda.synchronize()

        # Benchmark PyTorch
        start = time.perf_counter()
        for _ in range(bench_iters):
            _swiglu_backward_pytorch(gate, up, grad_output)
        torch.cuda.synchronize()
        pytorch_time = (time.perf_counter() - start) / bench_iters * 1000

        # Warmup Triton
        for _ in range(warmup_iters):
            _fused_swiglu_backward_triton(gate, up, grad_output)
        torch.cuda.synchronize()

        # Benchmark Triton
        start = time.perf_counter()
        for _ in range(bench_iters):
            _fused_swiglu_backward_triton(gate, up, grad_output)
        torch.cuda.synchronize()
        triton_time = (time.perf_counter() - start) / bench_iters * 1000

        speedup = pytorch_time / triton_time

        print(f"\n{'='*60}")
        print(f"SwiGLU Backward Benchmark (batch={batch_size}, seq={seq_len}, hidden={hidden_dim})")
        print(f"{'='*60}")
        print(f"PyTorch:  {pytorch_time:.3f} ms")
        print(f"Triton:   {triton_time:.3f} ms")
        print(f"Speedup:  {speedup:.2f}x")
        print(f"{'='*60}")

        # Assert meaningful speedup
        assert speedup > 1.2, f"Expected >1.2x speedup, got {speedup:.2f}x"


def run_pretest():
    """Run pretest validation - called by HYDRA's pretest system."""
    from hydra.kernels.fused_ops import (
        _fused_swiglu_backward_triton,
        _swiglu_backward_pytorch,
        USE_FUSED_SWIGLU_BACKWARD,
    )

    if not USE_FUSED_SWIGLU_BACKWARD:
        return {"status": "skipped", "reason": "Fused backward disabled"}

    # Test on 500M model dimensions
    torch.manual_seed(42)
    gate = torch.randn(4, 512, 1792, device="cuda", dtype=torch.bfloat16)
    up = torch.randn(4, 512, 1792, device="cuda", dtype=torch.bfloat16)
    grad_output = torch.randn(4, 512, 1792, device="cuda", dtype=torch.bfloat16)

    # Reference
    grad_gate_ref, grad_up_ref = _swiglu_backward_pytorch(gate, up, grad_output)

    # Triton
    grad_gate_triton, grad_up_triton = _fused_swiglu_backward_triton(gate, up, grad_output)

    # Check
    gate_diff = (grad_gate_triton.float() - grad_gate_ref.float()).abs().max().item()
    up_diff = (grad_up_triton.float() - grad_up_ref.float()).abs().max().item()

    passed = gate_diff < 0.1 and up_diff < 0.1

    return {
        "status": "passed" if passed else "failed",
        "grad_gate_max_diff": gate_diff,
        "grad_up_max_diff": up_diff,
        "tolerance": 0.1,
    }


if __name__ == "__main__":
    # Quick sanity check when run directly
    print("Running SwiGLU backward pretest...")
    result = run_pretest()
    print(f"Result: {result}")

    if result["status"] == "passed":
        print("\n✅ Pretest passed!")
    else:
        print(f"\n❌ Pretest failed: {result}")
        exit(1)
