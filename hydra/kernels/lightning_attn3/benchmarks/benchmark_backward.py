#!/usr/bin/env python3
"""
Lightning Attention-3 Backward Kernel Benchmark Suite

Compares:
1. Original LA3 backward (pre-Blackwell optimized)
2. Chunked LA3 backward (Blackwell-safe, recompute-heavy)
3. FlashAttention-2 baseline (if available)

Measures: throughput (iter/s), memory (MB), step time (ms)
Sequence lengths: 1024, 2048, 4096
"""

import gc
import time
import json
import argparse
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Callable

import torch
import torch.nn.functional as F

# Ensure repo root is on sys.path when executing this file directly.
# (Running `python hydra/.../benchmark_backward.py` sets sys.path[0] to the script dir.)
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Benchmark configuration
SEQ_LENGTHS = [1024, 2048, 4096]
BATCH_SIZE = 4
N_HEADS = 32
HEAD_DIM = 64
WARMUP_ITERS = 10
BENCH_ITERS = 50
DTYPE = torch.float16


@dataclass
class BenchmarkResult:
    """Single benchmark measurement."""
    name: str
    seq_len: int
    batch_size: int
    n_heads: int
    head_dim: int
    fwd_time_ms: float
    bwd_time_ms: float
    total_time_ms: float
    throughput_iters_per_sec: float
    peak_memory_mb: float
    allocated_memory_mb: float
    fwd_memory_mb: float  # Memory after forward only
    bwd_memory_mb: float  # Additional memory during backward
    activation_memory_mb: float  # Memory for saved tensors
    success: bool
    error: str = ""


def get_gpu_info() -> dict:
    """Get GPU information for the report."""
    props = torch.cuda.get_device_properties(0)
    return {
        "name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "total_memory_gb": props.total_memory / 1e9,
        "shared_memory_per_block": props.shared_memory_per_block,
        "shared_memory_per_block_optin": props.shared_memory_per_block_optin,
        "multiprocessor_count": props.multi_processor_count,
    }


def measure_memory() -> tuple[float, float]:
    """Returns (peak_mb, allocated_mb)."""
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 1e6
    allocated = torch.cuda.memory_allocated() / 1e6
    return peak, allocated


def reset_memory():
    """Reset memory stats and clear cache."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()


def benchmark_kernel(
    name: str,
    fn: Callable,
    seq_len: int,
    batch_size: int = BATCH_SIZE,
    n_heads: int = N_HEADS,
    head_dim: int = HEAD_DIM,
    warmup: int = WARMUP_ITERS,
    iters: int = BENCH_ITERS,
) -> BenchmarkResult:
    """Benchmark a single kernel configuration."""
    
    reset_memory()
    
    try:
        # Create tensors
        q = torch.randn(batch_size, n_heads, seq_len, head_dim, 
                       device='cuda', dtype=DTYPE, requires_grad=True)
        k = torch.randn_like(q, requires_grad=True)
        v = torch.randn_like(q, requires_grad=True)
        
        # Calculate theoretical input/output sizes
        tensor_bytes = batch_size * n_heads * seq_len * head_dim * 2  # 2 bytes for fp16
        input_size_mb = 3 * tensor_bytes / 1e6  # Q, K, V
        output_size_mb = tensor_bytes / 1e6  # Output
        
        # Warmup
        for _ in range(warmup):
            q.grad, k.grad, v.grad = None, None, None
            out = fn(q, k, v)
            loss = out.sum()
            loss.backward()
        torch.cuda.synchronize()
        
        # Reset for measurement
        reset_memory()
        q = torch.randn(batch_size, n_heads, seq_len, head_dim,
                       device='cuda', dtype=DTYPE, requires_grad=True)
        k = torch.randn_like(q, requires_grad=True)
        v = torch.randn_like(q, requires_grad=True)
        
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        mem_before_fwd = torch.cuda.memory_allocated() / 1e6
        
        # Measure forward memory (including saved tensors for backward)
        out = fn(q, k, v)
        torch.cuda.synchronize()
        mem_after_fwd = torch.cuda.memory_allocated() / 1e6
        fwd_peak = torch.cuda.max_memory_allocated() / 1e6
        fwd_memory = fwd_peak - mem_before_fwd  # Peak during forward
        
        # Measure backward memory
        torch.cuda.reset_peak_memory_stats()
        mem_before_bwd = torch.cuda.memory_allocated() / 1e6
        loss = out.sum()
        loss.backward()
        torch.cuda.synchronize()
        peak_during_bwd = torch.cuda.max_memory_allocated() / 1e6
        bwd_memory = peak_during_bwd - mem_before_bwd
        
        # Activation memory = what's held between forward and backward
        # This is the memory that stays allocated after forward, before backward
        activation_memory = mem_after_fwd - mem_before_fwd
        
        # Reset for timing
        reset_memory()
        q = torch.randn(batch_size, n_heads, seq_len, head_dim,
                       device='cuda', dtype=DTYPE, requires_grad=True)
        k = torch.randn_like(q, requires_grad=True)
        v = torch.randn_like(q, requires_grad=True)
        
        # Measure forward time
        torch.cuda.synchronize()
        fwd_start = time.perf_counter()
        for _ in range(iters):
            out = fn(q, k, v)
        torch.cuda.synchronize()
        fwd_end = time.perf_counter()
        fwd_time = (fwd_end - fwd_start) / iters * 1000  # ms
        
        # Measure backward time
        torch.cuda.synchronize()
        bwd_start = time.perf_counter()
        for _ in range(iters):
            q.grad, k.grad, v.grad = None, None, None
            out = fn(q, k, v)
            loss = out.sum()
            loss.backward()
        torch.cuda.synchronize()
        bwd_end = time.perf_counter()
        total_time = (bwd_end - bwd_start) / iters * 1000  # ms
        bwd_time = total_time - fwd_time
        
        # Final memory measurement
        peak_mb, allocated_mb = measure_memory()
        
        throughput = 1000 / total_time  # iter/s
        
        return BenchmarkResult(
            name=name,
            seq_len=seq_len,
            batch_size=batch_size,
            n_heads=n_heads,
            head_dim=head_dim,
            fwd_time_ms=fwd_time,
            bwd_time_ms=bwd_time,
            total_time_ms=total_time,
            throughput_iters_per_sec=throughput,
            peak_memory_mb=peak_mb,
            allocated_memory_mb=allocated_mb,
            fwd_memory_mb=fwd_memory,
            bwd_memory_mb=bwd_memory,
            activation_memory_mb=activation_memory,
            success=True,
        )
        
    except Exception as e:
        return BenchmarkResult(
            name=name,
            seq_len=seq_len,
            batch_size=batch_size,
            n_heads=n_heads,
            head_dim=head_dim,
            fwd_time_ms=0,
            bwd_time_ms=0,
            total_time_ms=0,
            throughput_iters_per_sec=0,
            peak_memory_mb=0,
            allocated_memory_mb=0,
            fwd_memory_mb=0,
            bwd_memory_mb=0,
            activation_memory_mb=0,
            success=False,
            error=str(e)[:200],
        )


def get_kernels() -> dict[str, Callable | None]:
    """Load available kernels."""
    kernels = {}
    
    # Lightning Attention-3 Original (uses hardware-aware selection)
    try:
        from hydra.kernels.lightning_attn3.ops import lightning_attn_func
        kernels["LA3-Original"] = lightning_attn_func
    except ImportError as e:
        print(f"⚠ LA3-Original not available: {e}")
        kernels["LA3-Original"] = None
    
    # Lightning Attention-3 Chunked (force chunked backward)
    try:
        from hydra.kernels.lightning_attn3.ops.triton import lightning_attn3_no_decay_chunked
        kernels["LA3-Chunked"] = lightning_attn3_no_decay_chunked
    except ImportError as e:
        print(f"⚠ LA3-Chunked not available: {e}")
        kernels["LA3-Chunked"] = None
    
    # FlashAttention-2
    try:
        from flash_attn import flash_attn_func
        # Wrap to match our API (flash_attn expects B, N, H, D not B, H, N, D)
        def flash_wrapper(q, k, v):
            # Transpose from (B, H, N, D) to (B, N, H, D)
            q_t = q.transpose(1, 2)
            k_t = k.transpose(1, 2)
            v_t = v.transpose(1, 2)
            out = flash_attn_func(q_t, k_t, v_t, causal=True)
            return out.transpose(1, 2)
        kernels["FlashAttn-2"] = flash_wrapper
    except ImportError as e:
        print(f"⚠ FlashAttention-2 not available: {e}")
        kernels["FlashAttn-2"] = None
    
    # PyTorch SDPA baseline
    def sdpa_wrapper(q, k, v):
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)
    kernels["PyTorch-SDPA"] = sdpa_wrapper
    
    return kernels


def run_benchmarks(seq_lengths: list[int] = SEQ_LENGTHS) -> list[BenchmarkResult]:
    """Run all benchmarks."""
    kernels = get_kernels()
    results = []
    
    print("\n" + "="*70)
    print("Lightning Attention-3 Backward Kernel Benchmark")
    print("="*70)
    
    gpu_info = get_gpu_info()
    print(f"\nGPU: {gpu_info['name']} (SM {gpu_info['compute_capability']})")
    print(f"Shared Memory: {gpu_info['shared_memory_per_block_optin']} bytes (opt-in)")
    print(f"Config: B={BATCH_SIZE}, H={N_HEADS}, D={HEAD_DIM}, dtype={DTYPE}")
    print(f"Warmup: {WARMUP_ITERS}, Iterations: {BENCH_ITERS}")
    print()
    
    for seq_len in seq_lengths:
        print(f"\n--- Sequence Length: {seq_len} ---")
        for name, fn in kernels.items():
            if fn is None:
                print(f"  {name}: SKIPPED (not available)")
                results.append(BenchmarkResult(
                    name=name, seq_len=seq_len, batch_size=BATCH_SIZE,
                    n_heads=N_HEADS, head_dim=HEAD_DIM,
                    fwd_time_ms=0, bwd_time_ms=0, total_time_ms=0,
                    throughput_iters_per_sec=0, peak_memory_mb=0,
                    allocated_memory_mb=0, fwd_memory_mb=0,
                    bwd_memory_mb=0, activation_memory_mb=0,
                    success=False, error="Not available"
                ))
                continue
            
            print(f"  {name}...", end=" ", flush=True)
            result = benchmark_kernel(name, fn, seq_len)
            results.append(result)
            
            if result.success:
                print(f"✓ {result.total_time_ms:.2f}ms ({result.throughput_iters_per_sec:.1f} iter/s), "
                      f"FwdMem: {result.fwd_memory_mb:.0f}MB, BwdMem: {result.bwd_memory_mb:.0f}MB")
            else:
                print(f"✗ FAILED: {result.error[:50]}")
    
    return results


def generate_table(results: list[BenchmarkResult]) -> str:
    """Generate ASCII table of results."""
    lines = []
    lines.append("\n" + "="*110)
    lines.append("BENCHMARK RESULTS SUMMARY")
    lines.append("="*110)
    
    # Header
    lines.append(f"{'Kernel':<15} {'SeqLen':>7} {'Fwd(ms)':>8} {'Bwd(ms)':>8} {'Total(ms)':>9} "
                f"{'Iter/s':>8} {'FwdMem':>8} {'BwdMem':>8} {'Peak':>8} {'Status':>8}")
    lines.append("-"*110)
    
    for r in results:
        status = "OK" if r.success else "FAIL"
        if r.success:
            lines.append(f"{r.name:<15} {r.seq_len:>7} {r.fwd_time_ms:>8.2f} {r.bwd_time_ms:>8.2f} "
                        f"{r.total_time_ms:>9.2f} {r.throughput_iters_per_sec:>8.1f} "
                        f"{r.fwd_memory_mb:>7.0f}M {r.bwd_memory_mb:>7.0f}M "
                        f"{r.peak_memory_mb:>7.0f}M {status:>8}")
        else:
            lines.append(f"{r.name:<15} {r.seq_len:>7} {'---':>8} {'---':>8} {'---':>9} "
                        f"{'---':>8} {'---':>8} {'---':>8} {'---':>8} {status:>8}")
    
    lines.append("="*110)
    lines.append("FwdMem = Memory allocated by forward pass")
    lines.append("BwdMem = Peak additional memory during backward pass")
    return "\n".join(lines)


def generate_comparison_table(results: list[BenchmarkResult]) -> str:
    """Generate comparison table showing relative performance."""
    lines = []
    lines.append("\n" + "="*70)
    lines.append("RELATIVE PERFORMANCE (vs PyTorch-SDPA baseline)")
    lines.append("="*70)
    
    # Group by seq_len
    seq_lens = sorted(set(r.seq_len for r in results))
    kernels = sorted(set(r.name for r in results))
    
    lines.append(f"{'Kernel':<15} " + " ".join(f"{'N='+str(s):>12}" for s in seq_lens))
    lines.append("-"*70)
    
    for kernel in kernels:
        row = [f"{kernel:<15}"]
        for seq_len in seq_lens:
            # Find baseline
            baseline = next((r for r in results if r.name == "PyTorch-SDPA" and r.seq_len == seq_len), None)
            current = next((r for r in results if r.name == kernel and r.seq_len == seq_len), None)
            
            if current and current.success and baseline and baseline.success:
                speedup = baseline.total_time_ms / current.total_time_ms
                row.append(f"{speedup:>10.2f}x  ")
            else:
                row.append(f"{'---':>12}")
        lines.append("".join(row))
    
    lines.append("="*70)
    lines.append("(>1.0x = faster than baseline)")
    return "\n".join(lines)


def generate_plots(results: list[BenchmarkResult], output_dir: Path):
    """Generate benchmark plots."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("⚠ matplotlib not available, skipping plots")
        return
    
    # Filter successful results
    results = [r for r in results if r.success]
    if not results:
        print("⚠ No successful results to plot")
        return
    
    seq_lens = sorted(set(r.seq_len for r in results))
    kernels = sorted(set(r.name for r in results))
    colors = {'LA3-Original': '#2ecc71', 'LA3-Chunked': '#3498db', 
              'FlashAttn-2': '#e74c3c', 'PyTorch-SDPA': '#95a5a6'}
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Total Time
    ax1 = axes[0]
    x = np.arange(len(seq_lens))
    width = 0.2
    for i, kernel in enumerate(kernels):
        times = [next((r.total_time_ms for r in results 
                      if r.name == kernel and r.seq_len == s), 0) for s in seq_lens]
        if any(t > 0 for t in times):
            ax1.bar(x + i*width, times, width, label=kernel, color=colors.get(kernel, '#333'))
    ax1.set_xlabel('Sequence Length')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('Forward + Backward Time')
    ax1.set_xticks(x + width * (len(kernels)-1) / 2)
    ax1.set_xticklabels(seq_lens)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Throughput
    ax2 = axes[1]
    for i, kernel in enumerate(kernels):
        throughputs = [next((r.throughput_iters_per_sec for r in results 
                           if r.name == kernel and r.seq_len == s), 0) for s in seq_lens]
        if any(t > 0 for t in throughputs):
            ax2.bar(x + i*width, throughputs, width, label=kernel, color=colors.get(kernel, '#333'))
    ax2.set_xlabel('Sequence Length')
    ax2.set_ylabel('Throughput (iter/s)')
    ax2.set_title('Training Throughput')
    ax2.set_xticks(x + width * (len(kernels)-1) / 2)
    ax2.set_xticklabels(seq_lens)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # Plot 3: Memory (stacked bar - forward + backward)
    ax3 = axes[2]
    for i, kernel in enumerate(kernels):
        fwd_mem = [next((r.fwd_memory_mb for r in results 
                        if r.name == kernel and r.seq_len == s), 0) for s in seq_lens]
        bwd_mem = [next((r.bwd_memory_mb for r in results 
                        if r.name == kernel and r.seq_len == s), 0) for s in seq_lens]
        if any(m > 0 for m in fwd_mem):
            bars1 = ax3.bar(x + i*width, fwd_mem, width, label=f'{kernel} fwd' if i == 0 else None, 
                           color=colors.get(kernel, '#333'), alpha=0.7)
            ax3.bar(x + i*width, bwd_mem, width, bottom=fwd_mem, 
                   label=f'{kernel} bwd' if i == 0 else None,
                   color=colors.get(kernel, '#333'), alpha=1.0, hatch='//')
    ax3.set_xlabel('Sequence Length')
    ax3.set_ylabel('Memory (MB)')
    ax3.set_title('GPU Memory: Forward (solid) + Backward (hatched)')
    ax3.set_xticks(x + width * (len(kernels)-1) / 2)
    ax3.set_xticklabels(seq_lens)
    ax3.legend(kernels, loc='upper left')
    ax3.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_dir / 'backward_benchmark_comparison.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved plot: {plot_path}")
    
    # Plot 4: Backward time breakdown
    fig2, ax4 = plt.subplots(figsize=(10, 6))
    x = np.arange(len(seq_lens))
    width = 0.2
    for i, kernel in enumerate(kernels):
        bwd_times = [next((r.bwd_time_ms for r in results 
                          if r.name == kernel and r.seq_len == s), 0) for s in seq_lens]
        if any(t > 0 for t in bwd_times):
            ax4.bar(x + i*width, bwd_times, width, label=kernel, color=colors.get(kernel, '#333'))
    ax4.set_xlabel('Sequence Length')
    ax4.set_ylabel('Backward Time (ms)')
    ax4.set_title('Backward Pass Time Comparison')
    ax4.set_xticks(x + width * (len(kernels)-1) / 2)
    ax4.set_xticklabels(seq_lens)
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    bwd_plot_path = output_dir / 'backward_time_comparison.png'
    plt.savefig(bwd_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved plot: {bwd_plot_path}")
    
    # Plot 5: Memory breakdown comparison
    fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5))
    
    # Forward memory
    ax5 = axes3[0]
    for i, kernel in enumerate(kernels):
        fwd_mem = [next((r.fwd_memory_mb for r in results 
                        if r.name == kernel and r.seq_len == s), 0) for s in seq_lens]
        if any(m > 0 for m in fwd_mem):
            ax5.bar(x + i*width, fwd_mem, width, label=kernel, color=colors.get(kernel, '#333'))
    ax5.set_xlabel('Sequence Length')
    ax5.set_ylabel('Memory (MB)')
    ax5.set_title('Forward Pass Memory Allocation')
    ax5.set_xticks(x + width * (len(kernels)-1) / 2)
    ax5.set_xticklabels(seq_lens)
    ax5.legend()
    ax5.grid(axis='y', alpha=0.3)
    
    # Backward memory
    ax6 = axes3[1]
    for i, kernel in enumerate(kernels):
        bwd_mem = [next((r.bwd_memory_mb for r in results 
                        if r.name == kernel and r.seq_len == s), 0) for s in seq_lens]
        if any(m > 0 for m in bwd_mem):
            ax6.bar(x + i*width, bwd_mem, width, label=kernel, color=colors.get(kernel, '#333'))
    ax6.set_xlabel('Sequence Length')
    ax6.set_ylabel('Memory (MB)')
    ax6.set_title('Backward Pass Additional Memory')
    ax6.set_xticks(x + width * (len(kernels)-1) / 2)
    ax6.set_xticklabels(seq_lens)
    ax6.legend()
    ax6.grid(axis='y', alpha=0.3)
    
    # Activation memory
    ax7 = axes3[2]
    for i, kernel in enumerate(kernels):
        act_mem = [next((r.activation_memory_mb for r in results 
                        if r.name == kernel and r.seq_len == s), 0) for s in seq_lens]
        if any(m > 0 for m in act_mem):
            ax7.bar(x + i*width, act_mem, width, label=kernel, color=colors.get(kernel, '#333'))
    ax7.set_xlabel('Sequence Length')
    ax7.set_ylabel('Memory (MB)')
    ax7.set_title('Activation Memory (Saved for Backward)')
    ax7.set_xticks(x + width * (len(kernels)-1) / 2)
    ax7.set_xticklabels(seq_lens)
    ax7.legend()
    ax7.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    mem_plot_path = output_dir / 'backward_memory_breakdown.png'
    plt.savefig(mem_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved plot: {mem_plot_path}")


def generate_report(results: list[BenchmarkResult], output_dir: Path) -> str:
    """Generate full markdown report."""
    gpu_info = get_gpu_info()
    
    report = f"""# Lightning Attention-3 Backward Kernel Benchmark Report

## System Configuration

| Property | Value |
|----------|-------|
| GPU | {gpu_info['name']} |
| Compute Capability | SM {gpu_info['compute_capability']} |
| Total Memory | {gpu_info['total_memory_gb']:.1f} GB |
| Shared Memory (opt-in) | {gpu_info['shared_memory_per_block_optin']:,} bytes |
| SMs | {gpu_info['multiprocessor_count']} |

## Benchmark Configuration

| Parameter | Value |
|-----------|-------|
| Batch Size | {BATCH_SIZE} |
| Attention Heads | {N_HEADS} |
| Head Dimension | {HEAD_DIM} |
| Data Type | {DTYPE} |
| Warmup Iterations | {WARMUP_ITERS} |
| Benchmark Iterations | {BENCH_ITERS} |
| Sequence Lengths | {SEQ_LENGTHS} |

## Kernels Tested

1. **LA3-Original**: Lightning Attention-3 with hardware-aware kernel selection
2. **LA3-Chunked**: Recompute-heavy chunked backward (Blackwell-optimized)
3. **FlashAttn-2**: FlashAttention-2 baseline (if available)
4. **PyTorch-SDPA**: PyTorch scaled_dot_product_attention baseline

## Results

### Detailed Measurements

| Kernel | SeqLen | Fwd (ms) | Bwd (ms) | Total (ms) | Throughput | Memory |
|--------|--------|----------|----------|------------|------------|--------|
"""
    
    for r in results:
        if r.success:
            report += f"| {r.name} | {r.seq_len} | {r.fwd_time_ms:.2f} | {r.bwd_time_ms:.2f} | {r.total_time_ms:.2f} | {r.throughput_iters_per_sec:.1f} iter/s | {r.peak_memory_mb:.0f} MB |\n"
        else:
            report += f"| {r.name} | {r.seq_len} | --- | --- | --- | FAILED | --- |\n"
    
    # Memory breakdown table
    report += "\n### Memory Breakdown\n\n"
    report += "| Kernel | SeqLen | Fwd Mem (MB) | Bwd Mem (MB) | Activation (MB) | Peak (MB) |\n"
    report += "|--------|--------|--------------|--------------|-----------------|----------|\n"
    
    for r in results:
        if r.success:
            report += f"| {r.name} | {r.seq_len} | {r.fwd_memory_mb:.0f} | {r.bwd_memory_mb:.0f} | {r.activation_memory_mb:.0f} | {r.peak_memory_mb:.0f} |\n"
        else:
            report += f"| {r.name} | {r.seq_len} | --- | --- | --- | --- |\n"
    
    report += """
**Memory Definitions:**
- **Fwd Mem**: Memory allocated during forward pass only
- **Bwd Mem**: Additional memory allocated during backward pass (gradients + temporaries)
- **Activation**: Memory used to save tensors for backward pass
- **Peak**: Maximum total memory usage

"""
    
    # Compute speedups
    report += "\n### Relative Performance\n\n"
    report += "Speedup vs PyTorch-SDPA baseline (higher = better):\n\n"
    report += "| Kernel | "
    seq_lens = sorted(set(r.seq_len for r in results))
    report += " | ".join(f"N={s}" for s in seq_lens) + " |\n"
    report += "|--------|" + "|".join(["-----" for _ in seq_lens]) + "|\n"
    
    kernels = sorted(set(r.name for r in results))
    for kernel in kernels:
        report += f"| {kernel} |"
        for seq_len in seq_lens:
            baseline = next((r for r in results if r.name == "PyTorch-SDPA" and r.seq_len == seq_len and r.success), None)
            current = next((r for r in results if r.name == kernel and r.seq_len == seq_len and r.success), None)
            if current and baseline:
                speedup = baseline.total_time_ms / current.total_time_ms
                report += f" {speedup:.2f}x |"
            else:
                report += " --- |"
        report += "\n"
    
    # Interpretation
    report += """
## Analysis

### Key Findings

"""
    
    # Find best kernel per seq_len
    for seq_len in seq_lens:
        seq_results = [r for r in results if r.seq_len == seq_len and r.success]
        if seq_results:
            best = min(seq_results, key=lambda r: r.total_time_ms)
            report += f"- **SeqLen {seq_len}**: Best kernel is **{best.name}** ({best.total_time_ms:.2f}ms, {best.throughput_iters_per_sec:.1f} iter/s)\n"
    
    report += """
### Tradeoff Summary

| Kernel | Pros | Cons |
|--------|------|------|
| LA3-Original | Fast on pre-Blackwell GPUs, optimized tile sizes | May OOM on Blackwell (SM 12.x) |
| LA3-Chunked | Works on all GPUs including Blackwell, lower memory | Slightly slower due to recomputation |
| FlashAttn-2 | Highly optimized, low memory | Softmax attention (different algorithm) |
| PyTorch-SDPA | No dependencies, reliable | Generally slower than specialized kernels |

### Recommendations

1. **Pre-Blackwell GPUs (SM < 12)**: Use LA3-Original for best performance
2. **Blackwell GPUs (SM 12.x)**: Use LA3-Chunked (auto-selected by hardware-aware dispatch)
3. **Memory-constrained**: LA3-Chunked or FlashAttn-2 offer lower memory footprint
4. **Long sequences**: Linear attention (LA3) scales better than quadratic (FlashAttn-2)

## Plots

![Benchmark Comparison](backward_benchmark_comparison.png)

![Backward Time Comparison](backward_time_comparison.png)

![Memory Breakdown](backward_memory_breakdown.png)
"""
    
    return report


def main():
    parser = argparse.ArgumentParser(description="Lightning Attention-3 Backward Benchmark")
    parser.add_argument("--seq-lengths", nargs="+", type=int, default=SEQ_LENGTHS,
                       help="Sequence lengths to benchmark")
    parser.add_argument("--output-dir", type=str, 
                       default="hydra/kernels/lightning_attn3/docs",
                       help="Output directory for report and plots")
    parser.add_argument("--json", action="store_true", help="Also save JSON results")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run benchmarks
    results = run_benchmarks(args.seq_lengths)
    
    # Print tables
    print(generate_table(results))
    print(generate_comparison_table(results))
    
    # Generate plots
    generate_plots(results, output_dir)
    
    # Generate and save report
    report = generate_report(results, output_dir)
    report_path = output_dir / "benchmark_report.md"
    report_path.write_text(report)
    print(f"\n✓ Saved report: {report_path}")
    
    # Save JSON if requested
    if args.json:
        ts = time.strftime("%Y%m%d_%H%M%S")
        json_path = output_dir / f"benchmark_results_{ts}.json"
        json_data = {
            "gpu_info": get_gpu_info(),
            "config": {
                "batch_size": BATCH_SIZE,
                "n_heads": N_HEADS,
                "head_dim": HEAD_DIM,
                "dtype": str(DTYPE),
                "warmup_iters": WARMUP_ITERS,
                "bench_iters": BENCH_ITERS,
            },
            "results": [asdict(r) for r in results],
        }
        json_path.write_text(json.dumps(json_data, indent=2))
        print(f"✓ Saved JSON: {json_path}")
    
    print("\n✓ Benchmark complete!")


if __name__ == "__main__":
    main()
