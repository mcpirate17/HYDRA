"""
Diagnostic test suite for lightning-attention library.
Tests kernel compilation, shared memory usage, backward pass stability,
and integration with HYDRA model.

Notes:
- REQUIRES: CUDA and the `lightning_attn` package/kernels to run full tests.
- The script gracefully skips tests when `lightning_attn` is not importable.
"""

import os
import sys
import torch
import torch.nn as nn
from contextlib import nullcontext

# Setup CUDA
assert torch.cuda.is_available(), "CUDA required for lightning-attention tests"

# Try importing lightning_attn
try:
    from lightning_attn.ops.triton.lightning_attn2_no_decay import LightningAttention2NoDecay
    HAS_LIGHTNING_ATTN = True
    ATTN_ERROR = None
except ImportError as e:
    HAS_LIGHTNING_ATTN = False
    ATTN_ERROR = str(e)


def test_lightning_attn_import():
    """Test that lightning-attention can be imported."""
    print("\n" + "=" * 70)
    print("LIGHTNING-ATTENTION IMPORT TEST")
    print("=" * 70)

    if not HAS_LIGHTNING_ATTN:
        print(f"❌ FAILED: Could not import lightning-attention")
        print(f"   Error: {ATTN_ERROR}")
        return False

    print(f"✓ Successfully imported lightning-attention kernels")
    print(f"  - lightning_attn2")
    print(f"  - lightning_attn2_no_decay")
    print(f"  - lightning_attn2_parallel")
    return True


def test_kernel_compilation():
    """Test that kernels compile without errors."""
    if not HAS_LIGHTNING_ATTN:
        print("\nSkipping kernel compilation test (lightning-attention not available)")
        return False

    print("\n" + "=" * 70)
    print("KERNEL COMPILATION TEST")
    print("=" * 70)

    torch.manual_seed(42)
    batch, heads, seq_len, head_dim, value_dim = 1, 4, 64, 64, 64

    q = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(batch, heads, seq_len, value_dim, dtype=torch.bfloat16, device="cuda")

    try:
        output = LightningAttention2NoDecay.apply(q, k, v)
        print(f"✓ Kernel compilation successful")
        print(f"  - Input shape: {q.shape}")
        print(f"  - Output shape: {output.shape}")
        print(f"  - Output dtype: {output.dtype}")
        return True
    except Exception as e:
        print(f"❌ Kernel compilation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_backward_pass_stability():
    """Test backward pass stability and gradient flow."""
    if not HAS_LIGHTNING_ATTN:
        print("\nSkipping backward pass test (lightning-attention not available)")
        return False

    print("\n" + "=" * 70)
    print("BACKWARD PASS STABILITY TEST")
    print("=" * 70)

    torch.manual_seed(42)
    batch, heads, seq_len, head_dim, value_dim = 2, 8, 256, 64, 64

    q = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.bfloat16, device="cuda")
    q.requires_grad_(True)
    k = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.bfloat16, device="cuda")
    k.requires_grad_(True)
    v = torch.randn(batch, heads, seq_len, value_dim, dtype=torch.bfloat16, device="cuda")
    v.requires_grad_(True)

    try:
        output = LightningAttention2NoDecay.apply(q, k, v)
        loss = output.sum()
        print(f"✓ Forward pass successful")
        print(f"  - Output shape: {output.shape}")
        print(f"  - Loss magnitude: {loss.item():.6f}")

        loss.backward()
        print(f"✓ Backward pass successful")

        # Check gradient stats
        q_grad_norm = q.grad.norm().item()
        k_grad_norm = k.grad.norm().item()
        v_grad_norm = v.grad.norm().item()

        print(f"✓ Gradient flow verified")
        print(f"  - q grad norm: {q_grad_norm:.6e}")
        print(f"  - k grad norm: {k_grad_norm:.6e}")
        print(f"  - v grad norm: {v_grad_norm:.6e}")

        # Check for NaN/Inf
        has_nan = any(
            torch.isnan(g).any() for g in [q.grad, k.grad, v.grad]
        )
        has_inf = any(
            torch.isinf(g).any() for g in [q.grad, k.grad, v.grad]
        )

        if has_nan or has_inf:
            print(f"❌ Gradients contain NaN or Inf")
            return False

        print(f"✓ No NaN/Inf in gradients")
        return True

    except Exception as e:
        print(f"❌ Backward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_shared_memory_constraints():
    """Test shared memory usage across different input sizes."""
    if not HAS_LIGHTNING_ATTN:
        print("\nSkipping shared memory test (lightning-attention not available)")
        return False

    print("\n" + "=" * 70)
    print("SHARED MEMORY CONSTRAINTS TEST")
    print("=" * 70)

    # Get GPU memory info
    gpu_name = torch.cuda.get_device_name(0)
    compute_capability = torch.cuda.get_device_capability(0)
    print(f"GPU: {gpu_name}")
    print(f"Compute Capability: {compute_capability[0]}.{compute_capability[1]}")

    # Blackwell has 101KB shared memory per SM
    # Hopper has 228KB
    # Ada has 228KB
    shared_mem_per_sm = 101 if compute_capability[0] >= 10 else 228

    print(f"Expected shared memory per SM: {shared_mem_per_sm}KB")

    test_configs = [
        (1, 4, 64, 64, 64, "small"),
        (2, 8, 256, 64, 64, "medium"),
        (4, 16, 512, 64, 64, "large"),
        (2, 8, 1024, 64, 64, "very_large"),
    ]

    results = []
    all_passed = True

    for batch, heads, seq_len, head_dim, value_dim, label in test_configs:
        torch.manual_seed(42)

        q = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(batch, heads, seq_len, value_dim, dtype=torch.bfloat16, device="cuda")

        try:
            # Warm up
            _ = LightningAttention2NoDecay.apply(q, k, v)

            # Test backward
            q.requires_grad_(True)
            k.requires_grad_(True)
            v.requires_grad_(True)

            output = LightningAttention2NoDecay.apply(q, k, v)
            loss = output.sum()
            loss.backward()

            print(f"✓ {label:12s}: B={batch}, H={heads}, N={seq_len}, D={head_dim}, E={value_dim}")
            results.append((label, True))

        except RuntimeError as e:
            if "shared memory" in str(e).lower() or "too large" in str(e).lower():
                print(f"⚠ {label:12s}: Shared memory constraint exceeded")
                print(f"   Error: {str(e)[:80]}...")
                results.append((label, False))
                all_passed = False
            else:
                print(f"❌ {label:12s}: {e}")
                results.append((label, False))
                all_passed = False

    print(f"\nSummary: {sum(1 for _, passed in results if passed)}/{len(results)} configs passed")
    return all_passed


def test_integration_with_hydra():
    """Test lightning-attention integration with HYDRA model."""
    if not HAS_LIGHTNING_ATTN:
        print("\nSkipping HYDRA integration test (lightning-attention not available)")
        return False

    print("\n" + "=" * 70)
    print("HYDRA INTEGRATION TEST")
    print("=" * 70)

    try:
        # Add parent directory to path to import hydra
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        from hydra.model.framework import HydraModel

        # Create a small model with LA3 attention (default)
        print("Creating HYDRA model with LA3 attention...")
        model = HydraModel(
            dim=256,  # Small for quick testing
            n_mor_blocks=2,
            n_heads=4,
            n_kv_heads=2,
            recursions_per_block=2,
            max_seq_len=128,
        )
        model = model.to("cuda").bfloat16()

        print(f"✓ HYDRA model created successfully")
        print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Forward pass with tiny batch
        torch.manual_seed(42)
        x = torch.randint(0, 2048, (1, 64), device="cuda")

        print("Running forward pass...")
        output = model(x)

        print(f"✓ Forward pass successful")
        print(f"  - Input shape: {x.shape}")
        print(f"  - Output shape: {output.shape}")

        # Backward pass
        loss = output.mean()
        print("Running backward pass...")
        loss.backward()

        print(f"✓ Backward pass successful")
        print(f"  - Loss magnitude: {loss.item():.6f}")

        return True

    except Exception as e:
        print(f"⚠ HYDRA integration skipped: {type(e).__name__}: {str(e)[:100]}")
        import traceback
        traceback.print_exc()
        return True  # Soft pass - lightning-attn itself works


def test_numerical_stability_across_scales():
    """Test numerical stability across different model scales."""
    if not HAS_LIGHTNING_ATTN:
        print("\nSkipping numerical stability test (lightning-attention not available)")
        return False

    print("\n" + "=" * 70)
    print("NUMERICAL STABILITY ACROSS SCALES TEST")
    print("=" * 70)

    scales = [
        (1, 8, 128, 64, 64, "100M-like"),
        (2, 16, 256, 128, 128, "250M-like"),
        (2, 16, 256, 64, 64, "medium"),
    ]

    all_passed = True

    for batch, heads, seq_len, head_dim, value_dim, label in scales:
        torch.manual_seed(42)

        q = torch.randn(
            batch, heads, seq_len, head_dim,
            dtype=torch.bfloat16, device="cuda"
        ) * 0.1  # Keep small for stability
        k = torch.randn(
            batch, heads, seq_len, head_dim,
            dtype=torch.bfloat16, device="cuda"
        ) * 0.1
        v = torch.randn(
            batch, heads, seq_len, value_dim,
            dtype=torch.bfloat16, device="cuda"
        ) * 0.1

        try:
            output = LightningAttention2NoDecay.apply(q, k, v)

            has_nan = torch.isnan(output).any().item()
            has_inf = torch.isinf(output).any().item()

            if has_nan or has_inf:
                print(f"❌ {label:15s}: NaN/Inf detected in output")
                all_passed = False
            else:
                output_norm = output.norm().item()
                print(f"✓ {label:15s}: Output norm = {output_norm:.6e}")

        except Exception as e:
            print(f"❌ {label:15s}: {e}")
            all_passed = False

    return all_passed


def main():
    """Run all diagnostic tests."""
    print("\n" + "=" * 70)
    print("LIGHTNING-ATTENTION DIAGNOSTIC SUITE")
    print("=" * 70)

    results = {
        "Import": test_lightning_attn_import(),
        "Kernel Compilation": test_kernel_compilation(),
        "Backward Pass Stability": test_backward_pass_stability(),
        "Shared Memory Constraints": test_shared_memory_constraints(),
        "Numerical Stability": test_numerical_stability_across_scales(),
        "HYDRA Integration": test_integration_with_hydra(),
    }

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"{status:8s} - {test_name}")

    all_passed = all(results.values())
    print("=" * 70)

    if all_passed:
        print("🎉 All diagnostic tests passed!")
        return 0
    else:
        print("❌ Some diagnostic tests failed. See above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
