#!/usr/bin/env python3
"""Profile the value_shift operations in CCGQA attention.

The value_shift feature (lines 217-220 of attention.py) creates a new tensor
and does indexed assignments. This script measures its overhead and explores
if a more efficient implementation is possible.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import time
import gc


def reset_memory():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def benchmark_value_shift_implementations():
    """Compare different implementations of value shift."""
    print("=" * 70)
    print("BENCHMARKING VALUE_SHIFT IMPLEMENTATIONS")
    print("=" * 70)

    B, S = 4, 1024
    kv_dim = 48  # 3 kv_heads * 16 head_dim
    half_kv_dim = kv_dim // 2
    n_iters = 1000

    # Simulate inputs (like from qkv projection)
    v_curr = torch.randn(B, S, half_kv_dim, device="cuda", dtype=torch.bfloat16)
    v_prev = torch.randn(B, S, half_kv_dim, device="cuda", dtype=torch.bfloat16)

    # === Implementation 1: Current (torch.empty + indexed assignments) ===
    def value_shift_current(v_curr, v_prev):
        B, S, half_dim = v_curr.shape
        v = torch.empty((B, S, half_dim * 2), device=v_curr.device, dtype=v_curr.dtype)
        v[..., :half_dim] = v_curr
        v[..., half_dim:] = 0
        v[:, 1:, half_dim:] = v_prev[:, :-1, :]
        return v

    # === Implementation 2: torch.cat + pad ===
    def value_shift_cat_pad(v_curr, v_prev):
        B, S, half_dim = v_curr.shape
        # Shift v_prev by 1 position (prepend zeros)
        v_prev_shifted = torch.cat([
            torch.zeros(B, 1, half_dim, device=v_prev.device, dtype=v_prev.dtype),
            v_prev[:, :-1, :]
        ], dim=1)
        return torch.cat([v_curr, v_prev_shifted], dim=-1)

    # === Implementation 3: F.pad + cat ===
    def value_shift_fpad(v_curr, v_prev):
        B, S, half_dim = v_curr.shape
        # Use F.pad to shift v_prev
        v_prev_shifted = torch.nn.functional.pad(v_prev[:, :-1, :], (0, 0, 1, 0))
        return torch.cat([v_curr, v_prev_shifted], dim=-1)

    # === Implementation 4: roll + mask ===
    def value_shift_roll(v_curr, v_prev):
        B, S, half_dim = v_curr.shape
        v_prev_shifted = torch.roll(v_prev, shifts=1, dims=1)
        v_prev_shifted[:, 0, :] = 0  # Zero out first position
        return torch.cat([v_curr, v_prev_shifted], dim=-1)

    # === Implementation 5: zeros + scatter (compile-friendly) ===
    def value_shift_zeros(v_curr, v_prev):
        B, S, half_dim = v_curr.shape
        v = torch.zeros((B, S, half_dim * 2), device=v_curr.device, dtype=v_curr.dtype)
        v[..., :half_dim] = v_curr
        v[:, 1:, half_dim:] = v_prev[:, :-1, :]
        return v

    implementations = [
        ("Current (empty+assign)", value_shift_current),
        ("cat+pad", value_shift_cat_pad),
        ("F.pad+cat", value_shift_fpad),
        ("roll+mask", value_shift_roll),
        ("zeros+assign", value_shift_zeros),
    ]

    # Verify all implementations produce same result
    print("\nVerifying implementations produce identical results...")
    ref = value_shift_current(v_curr, v_prev)
    for name, impl in implementations:
        result = impl(v_curr, v_prev)
        if not torch.allclose(ref, result, atol=1e-5):
            print(f"  {name}: MISMATCH!")
        else:
            print(f"  {name}: OK")

    # Warmup
    print("\nWarmup...")
    for _ in range(100):
        for _, impl in implementations:
            _ = impl(v_curr, v_prev)
    torch.cuda.synchronize()

    # Benchmark each implementation
    print(f"\nBenchmarking ({n_iters} iterations each)...")
    print("-" * 50)

    times = {}
    for name, impl in implementations:
        reset_memory()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(n_iters):
            v_curr = torch.randn(B, S, half_kv_dim, device="cuda", dtype=torch.bfloat16)
            v_prev = torch.randn(B, S, half_kv_dim, device="cuda", dtype=torch.bfloat16)
            _ = impl(v_curr, v_prev)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / n_iters * 1000
        times[name] = elapsed
        print(f"{name:<25} {elapsed:.4f}ms")

    # Find fastest
    fastest = min(times, key=times.get)
    fastest_time = times[fastest]
    print(f"\nFastest: {fastest} ({fastest_time:.4f}ms)")
    print("\nRelative to current implementation:")
    current_time = times["Current (empty+assign)"]
    for name, t in times.items():
        diff = ((t - current_time) / current_time) * 100
        print(f"  {name:<25} {diff:+.1f}%")


def profile_value_shift_in_context():
    """Profile value_shift as part of full attention."""
    print("\n" + "=" * 70)
    print("VALUE_SHIFT IN FULL ATTENTION CONTEXT")
    print("=" * 70)

    from torch.profiler import profile, ProfilerActivity, record_function
    from hydra.attention.backends.ccgqa.attention import CCGQAAttention

    B, S = 4, 1024
    dim = 768

    attn = CCGQAAttention(
        dim=dim,
        n_heads=12,
        n_kv_heads=3,
        compression_factor=4,
        max_seq_len=2048,
        use_value_shift=True,  # Ensure value_shift is enabled
    ).cuda().to(torch.bfloat16)

    # Warmup
    x = torch.randn(B, S, dim, device="cuda", dtype=torch.bfloat16)
    for _ in range(10):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _ = attn(x)

    # Profile with record_function markers
    # We need to instrument the attention module temporarily
    original_forward = attn.forward

    def instrumented_forward(x, mask=None):
        B, S, _ = x.shape

        with record_function("qkv_proj"):
            qkv = attn.qkv_proj(x)

        with record_function("split_qk"):
            q = qkv[..., :attn.latent_dim]
            k = qkv[..., attn.latent_dim:attn.latent_dim + attn.kv_dim]

        with record_function("value_shift"):
            if attn.use_value_shift:
                v_curr = qkv[..., attn.latent_dim + attn.kv_dim:attn.latent_dim + attn.kv_dim + attn.kv_dim // 2]
                v_prev = qkv[..., attn.latent_dim + attn.kv_dim + attn.kv_dim // 2:]
                half_kv_dim = attn.kv_dim // 2
                v = torch.empty((B, S, attn.kv_dim), device=v_curr.device, dtype=v_curr.dtype)
                v[..., :half_kv_dim] = v_curr
                v[..., half_kv_dim:] = 0
                v[:, 1:, half_kv_dim:] = v_prev[:, :-1, :]
            else:
                v = qkv[..., attn.latent_dim + attn.kv_dim:]

        with record_function("convolutions"):
            if attn.use_convs:
                q = attn.q_conv(q)
                k = attn.k_conv(k)

        # Rest of attention (simplified for profiling)
        with record_function("reshape_and_sdpa"):
            q = q.view(B, S, attn.n_heads, attn.head_dim).transpose(1, 2)
            k = k.view(B, S, attn.n_kv_heads, attn.head_dim).transpose(1, 2)
            v = v.view(B, S, attn.n_kv_heads, attn.head_dim).transpose(1, 2)

            if attn.use_qk_norm:
                q = torch.nn.functional.normalize(q, p=2, dim=-1)
                k = torch.nn.functional.normalize(k, p=2, dim=-1) * attn.key_temperature

            if attn.use_rope:
                q = attn._apply_rope(q, S)
                k = attn._apply_rope(k, S)

            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=True, enable_gqa=(attn.n_kv_heads != attn.n_heads)
            )

        with record_function("output_proj"):
            out = out.transpose(1, 2).contiguous().view(B, S, attn.latent_dim)
            out = attn.o_proj(out) * attn.output_scale

        return out

    attn.forward = instrumented_forward

    # Profile
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        for _ in range(20):
            x = torch.randn(B, S, dim, device="cuda", dtype=torch.bfloat16)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                _ = attn(x)

    torch.cuda.synchronize()

    # Restore original forward
    attn.forward = original_forward

    # Analyze results
    print("\nAttention component breakdown:")
    print("-" * 60)

    key_avgs = prof.key_averages()
    components = ["qkv_proj", "split_qk", "value_shift", "convolutions", "reshape_and_sdpa", "output_proj"]

    total_time = 0
    component_times = {}
    for event in key_avgs:
        for comp in components:
            if event.key == comp:
                cuda_time = getattr(event, 'cuda_time_total', 0) or getattr(event, 'self_cuda_time_total', 0) or 0
                component_times[comp] = cuda_time / 1000  # Convert to ms
                total_time += cuda_time / 1000

    print(f"{'Component':<25} {'Time (ms)':>12} {'% of total':>12}")
    print("-" * 50)
    for comp in components:
        t = component_times.get(comp, 0)
        pct = (t / total_time * 100) if total_time > 0 else 0
        print(f"{comp:<25} {t:>10.3f}ms {pct:>10.1f}%")

    print(f"\nTotal attention time: {total_time:.3f}ms")

    vs_time = component_times.get('value_shift', 0)
    vs_pct = (vs_time / total_time * 100) if total_time > 0 else 0
    print(f"\nValue shift contribution: {vs_time:.3f}ms ({vs_pct:.1f}%)")

    if vs_pct < 5:
        print("=> Value shift overhead is minimal (<5%)")
    else:
        print("=> Value shift is significant - consider optimization")


def main():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    benchmark_value_shift_implementations()
    profile_value_shift_in_context()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
The value_shift operation:
1. Creates a new tensor with torch.empty()
2. Assigns v_curr to first half
3. Zeros second half
4. Copies shifted v_prev to second half (positions 1:)

This involves ~3 copy operations per forward pass.

Impact assessment:
- If <5% of attention time: Not worth optimizing
- If >5%: Consider fused implementation or torch.compile
""")


if __name__ == "__main__":
    main()
