#!/usr/bin/env python3
"""Benchmark script for fused RMSNorm and QK-Norm backward kernels.

Compares performance of:
1. PyTorch reference implementation (many kernel launches)
2. Fused Triton implementation (single/few kernel launches)

Usage:
    source /home/tim/venvs/llm/bin/activate && python diagnostics/benchmark_backward_kernels.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import time
import math


def benchmark_rms_norm_backward():
    """Comprehensive benchmark of RMSNorm backward implementations."""
    from hydra.kernels.fused_ops import (
        _fused_rms_norm_backward_triton,
        _rms_norm_backward_pytorch,
        get_kernel_status,
    )

    print("=" * 70)
    print("RMSNorm Backward Kernel Benchmark")
    print("=" * 70)
    print(f"\nKernel Status: {get_kernel_status()}")
    print()

    # Test configurations matching different model sizes
    configs = [
        # (name, batch, seq, dim)
        ("100M model", 16, 512, 768),
        ("250M model", 12, 512, 1024),
        ("500M model", 4, 1024, 1792),
        ("1B model", 2, 1024, 2048),
    ]

    warmup_iters = 20
    bench_iters = 100

    results = []

    for name, batch, seq, dim in configs:
        print(f"\n{name} (batch={batch}, seq={seq}, dim={dim})")
        print("-" * 50)

        torch.manual_seed(42)
        x = torch.randn(batch, seq, dim, device="cuda", dtype=torch.bfloat16)
        x_flat = x.contiguous().view(-1, dim)
        weight = torch.randn(dim, device="cuda", dtype=torch.bfloat16)
        grad_output = torch.randn(batch, seq, dim, device="cuda", dtype=torch.bfloat16)
        eps = 1e-6
        orig_shape = x.shape

        # Memory info
        elements = batch * seq * dim
        memory_mb = elements * 2 * 4 / 1024 / 1024  # x, grad_out, grad_x, x_norm
        print(f"Elements: {elements:,}, Memory: {memory_mb:.1f} MB")

        # Warmup PyTorch
        for _ in range(warmup_iters):
            _rms_norm_backward_pytorch(x_flat, weight, grad_output, eps, orig_shape)
        torch.cuda.synchronize()

        # Benchmark PyTorch
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for _ in range(bench_iters):
            _rms_norm_backward_pytorch(x_flat, weight, grad_output, eps, orig_shape)
        torch.cuda.synchronize()
        pytorch_time = (time.perf_counter() - start) / bench_iters * 1000
        pytorch_mem = torch.cuda.max_memory_allocated() / 1024 / 1024

        # Warmup Triton
        for _ in range(warmup_iters):
            _fused_rms_norm_backward_triton(x_flat, weight, grad_output, eps, orig_shape)
        torch.cuda.synchronize()

        # Benchmark Triton
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for _ in range(bench_iters):
            _fused_rms_norm_backward_triton(x_flat, weight, grad_output, eps, orig_shape)
        torch.cuda.synchronize()
        triton_time = (time.perf_counter() - start) / bench_iters * 1000
        triton_mem = torch.cuda.max_memory_allocated() / 1024 / 1024

        speedup = pytorch_time / triton_time
        mem_reduction = (pytorch_mem - triton_mem) / pytorch_mem * 100 if pytorch_mem > triton_mem else 0

        print(f"PyTorch:  {pytorch_time:.3f} ms  (peak mem: {pytorch_mem:.1f} MB)")
        print(f"Triton:   {triton_time:.3f} ms  (peak mem: {triton_mem:.1f} MB)")
        print(f"Speedup:  {speedup:.2f}x")
        print(f"Memory:   {mem_reduction:.1f}% reduction")

        results.append({
            "config": name,
            "pytorch_ms": pytorch_time,
            "triton_ms": triton_time,
            "speedup": speedup,
            "mem_reduction": mem_reduction,
        })

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY - RMSNorm Backward")
    print("=" * 70)
    print(f"{'Config':<15} {'PyTorch (ms)':<15} {'Triton (ms)':<15} {'Speedup':<10}")
    print("-" * 55)
    for r in results:
        print(f"{r['config']:<15} {r['pytorch_ms']:<15.3f} {r['triton_ms']:<15.3f} {r['speedup']:<10.2f}x")

    avg_speedup = sum(r["speedup"] for r in results) / len(results)
    print("-" * 55)
    print(f"{'Average':<15} {'':<15} {'':<15} {avg_speedup:<10.2f}x")

    return results


def benchmark_qk_norm_backward():
    """Comprehensive benchmark of QK-Norm backward implementations."""
    from hydra.kernels.fused_ops import (
        _fused_qk_norm_backward_triton,
        _qk_norm_backward_pytorch,
        get_kernel_status,
    )

    print("\n" + "=" * 70)
    print("QK-Norm Backward Kernel Benchmark")
    print("=" * 70)
    print()

    # Test configurations matching different model sizes
    # (name, batch, n_heads, seq, head_dim, n_kv_heads)
    configs = [
        ("100M model", 16, 12, 512, 64, 4),
        ("250M model", 12, 16, 512, 64, 4),
        ("500M model", 4, 28, 1024, 64, 4),
        ("1B model", 2, 32, 1024, 64, 8),
    ]

    warmup_iters = 20
    bench_iters = 100

    results = []

    for name, batch, n_heads, seq, head_dim, n_kv_heads in configs:
        print(f"\n{name} (batch={batch}, heads={n_heads}/{n_kv_heads}, seq={seq}, head_dim={head_dim})")
        print("-" * 50)

        torch.manual_seed(42)
        q = torch.randn(batch, n_heads, seq, head_dim, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(batch, n_kv_heads, seq, head_dim, device="cuda", dtype=torch.bfloat16)
        grad_q_out = torch.randn_like(q)
        grad_k_out = torch.randn_like(k)
        scale = math.sqrt(head_dim)
        temperature = 1.0

        # Memory info
        q_elements = batch * n_heads * seq * head_dim
        k_elements = batch * n_kv_heads * seq * head_dim
        memory_mb = (q_elements + k_elements) * 2 * 4 / 1024 / 1024  # q, k, grads
        print(f"Q elements: {q_elements:,}, K elements: {k_elements:,}, Memory: {memory_mb:.1f} MB")

        # Warmup PyTorch
        for _ in range(warmup_iters):
            _qk_norm_backward_pytorch(q, k, grad_q_out, grad_k_out, scale, temperature)
        torch.cuda.synchronize()

        # Benchmark PyTorch
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for _ in range(bench_iters):
            _qk_norm_backward_pytorch(q, k, grad_q_out, grad_k_out, scale, temperature)
        torch.cuda.synchronize()
        pytorch_time = (time.perf_counter() - start) / bench_iters * 1000
        pytorch_mem = torch.cuda.max_memory_allocated() / 1024 / 1024

        # Warmup Triton
        for _ in range(warmup_iters):
            _fused_qk_norm_backward_triton(q, k, grad_q_out, grad_k_out, scale, temperature)
        torch.cuda.synchronize()

        # Benchmark Triton
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for _ in range(bench_iters):
            _fused_qk_norm_backward_triton(q, k, grad_q_out, grad_k_out, scale, temperature)
        torch.cuda.synchronize()
        triton_time = (time.perf_counter() - start) / bench_iters * 1000
        triton_mem = torch.cuda.max_memory_allocated() / 1024 / 1024

        speedup = pytorch_time / triton_time
        mem_reduction = (pytorch_mem - triton_mem) / pytorch_mem * 100 if pytorch_mem > triton_mem else 0

        print(f"PyTorch:  {pytorch_time:.3f} ms  (peak mem: {pytorch_mem:.1f} MB)")
        print(f"Triton:   {triton_time:.3f} ms  (peak mem: {triton_mem:.1f} MB)")
        print(f"Speedup:  {speedup:.2f}x")
        print(f"Memory:   {mem_reduction:.1f}% reduction")

        results.append({
            "config": name,
            "pytorch_ms": pytorch_time,
            "triton_ms": triton_time,
            "speedup": speedup,
            "mem_reduction": mem_reduction,
        })

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY - QK-Norm Backward")
    print("=" * 70)
    print(f"{'Config':<15} {'PyTorch (ms)':<15} {'Triton (ms)':<15} {'Speedup':<10}")
    print("-" * 55)
    for r in results:
        print(f"{r['config']:<15} {r['pytorch_ms']:<15.3f} {r['triton_ms']:<15.3f} {r['speedup']:<10.2f}x")

    avg_speedup = sum(r["speedup"] for r in results) / len(results)
    print("-" * 55)
    print(f"{'Average':<15} {'':<15} {'':<15} {avg_speedup:<10.2f}x")

    return results


def benchmark_full_training_step():
    """Benchmark a full forward+backward pass with and without fused kernels."""
    from hydra.kernels import fused_ops
    from hydra.kernels.fused_ops import fused_rms_norm, fused_qk_norm

    print("\n" + "=" * 70)
    print("Full Forward+Backward Pass Benchmark")
    print("=" * 70)

    # 500M model config
    batch, seq, dim = 4, 1024, 1792
    n_heads, n_kv_heads, head_dim = 28, 4, 64

    warmup_iters = 10
    bench_iters = 50

    # Test RMSNorm full pass
    print("\n--- RMSNorm (Full Forward+Backward) ---")
    torch.manual_seed(42)
    x = torch.randn(batch, seq, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    # Benchmark with Triton backward
    fused_ops.USE_FUSED_RMS_NORM_BACKWARD = True

    for _ in range(warmup_iters):
        x.grad = None
        weight.grad = None
        out = fused_rms_norm(x, weight)
        out.sum().backward()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(bench_iters):
        x.grad = None
        weight.grad = None
        out = fused_rms_norm(x, weight)
        out.sum().backward()
    torch.cuda.synchronize()
    triton_time = (time.perf_counter() - start) / bench_iters * 1000

    # Benchmark with PyTorch backward
    fused_ops.USE_FUSED_RMS_NORM_BACKWARD = False

    for _ in range(warmup_iters):
        x.grad = None
        weight.grad = None
        out = fused_rms_norm(x, weight)
        out.sum().backward()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(bench_iters):
        x.grad = None
        weight.grad = None
        out = fused_rms_norm(x, weight)
        out.sum().backward()
    torch.cuda.synchronize()
    pytorch_time = (time.perf_counter() - start) / bench_iters * 1000

    # Re-enable
    fused_ops.USE_FUSED_RMS_NORM_BACKWARD = True

    rms_speedup = pytorch_time / triton_time
    print(f"RMSNorm (batch={batch}, seq={seq}, dim={dim}):")
    print(f"  PyTorch backward:  {pytorch_time:.3f} ms")
    print(f"  Triton backward:   {triton_time:.3f} ms")
    print(f"  Speedup:           {rms_speedup:.2f}x")

    # Test QK-Norm full pass
    print("\n--- QK-Norm (Full Forward+Backward) ---")
    torch.manual_seed(42)
    q = torch.randn(batch, n_heads, seq, head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(batch, n_kv_heads, seq, head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    scale = math.sqrt(head_dim)

    # Benchmark with Triton backward
    fused_ops.USE_FUSED_QK_NORM_BACKWARD = True

    for _ in range(warmup_iters):
        q.grad = None
        k.grad = None
        q_out, k_out = fused_qk_norm(q, k, scale)
        (q_out.sum() + k_out.sum()).backward()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(bench_iters):
        q.grad = None
        k.grad = None
        q_out, k_out = fused_qk_norm(q, k, scale)
        (q_out.sum() + k_out.sum()).backward()
    torch.cuda.synchronize()
    triton_time = (time.perf_counter() - start) / bench_iters * 1000

    # Benchmark with PyTorch backward
    fused_ops.USE_FUSED_QK_NORM_BACKWARD = False

    for _ in range(warmup_iters):
        q.grad = None
        k.grad = None
        q_out, k_out = fused_qk_norm(q, k, scale)
        (q_out.sum() + k_out.sum()).backward()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(bench_iters):
        q.grad = None
        k.grad = None
        q_out, k_out = fused_qk_norm(q, k, scale)
        (q_out.sum() + k_out.sum()).backward()
    torch.cuda.synchronize()
    pytorch_time = (time.perf_counter() - start) / bench_iters * 1000

    # Re-enable
    fused_ops.USE_FUSED_QK_NORM_BACKWARD = True

    qk_speedup = pytorch_time / triton_time
    print(f"QK-Norm (batch={batch}, heads={n_heads}/{n_kv_heads}, seq={seq}):")
    print(f"  PyTorch backward:  {pytorch_time:.3f} ms")
    print(f"  Triton backward:   {triton_time:.3f} ms")
    print(f"  Speedup:           {qk_speedup:.2f}x")

    return rms_speedup, qk_speedup


if __name__ == "__main__":
    print("CUDA Device:", torch.cuda.get_device_name())
    print()

    rms_results = benchmark_rms_norm_backward()
    qk_results = benchmark_qk_norm_backward()
    rms_full_speedup, qk_full_speedup = benchmark_full_training_step()

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    avg_rms_speedup = sum(r["speedup"] for r in rms_results) / len(rms_results)
    avg_qk_speedup = sum(r["speedup"] for r in qk_results) / len(qk_results)

    print(f"RMSNorm backward:  ~{avg_rms_speedup:.1f}x speedup (isolated), {rms_full_speedup:.1f}x (full pass)")
    print(f"QK-Norm backward:  ~{avg_qk_speedup:.1f}x speedup (isolated), {qk_full_speedup:.1f}x (full pass)")
    print()
    print("The fused kernels reduce memory traffic and kernel launch overhead,")
    print("providing meaningful improvements in training throughput.")
