#!/usr/bin/env python3
"""
Lightning Attention-3 Comprehensive Benchmark

Compares all Lightning Attention-3 variants against baselines:
- LA3-Original: lightning_attn3 (with decay parameter)
- LA3-NoDecay: lightning_attn3_no_decay (original non-chunked)
- LA3-Chunked: lightning_attn3_no_decay_chunked (Blackwell-optimized)
- LA3-Parallel: lightning_attn3_parallel (parallel variant)
- FlashAttn-2: flash_attn_func (if available)
- PyTorch-SDPA: scaled_dot_product_attention

Usage:
    python benchmark.py                    # Full benchmark
    python benchmark.py --quick            # Quick test (fewer seq lengths)
    python benchmark.py --save             # Save results to docs/
    python benchmark.py --plot             # Generate plots
"""

import argparse
import json
import sys
from datetime import datetime
import importlib
from pathlib import Path

import torch
import triton

# Add repo root to path for imports when running as a script.
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Import Lightning Attention variants
from hydra.attention.backends.lightning_attn3.ops.triton import (
    lightning_attn3,
    lightning_attn3_no_decay,
    lightning_attn3_no_decay_chunked,
    lightning_attn3_parallel,
)

# Try to import Flash Attention
try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False
    print("Note: flash_attn not available, skipping FlashAttn-2 benchmark")


def get_gpu_info() -> dict:
    """Get GPU information."""
    props = torch.cuda.get_device_properties(0)
    return {
        "name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "total_memory_gb": round(props.total_memory / (1024**3), 2),
        "shared_memory_per_block": 49152,
        "shared_memory_per_block_optin": 101376,
        "multiprocessor_count": props.multi_processor_count,
    }


def benchmark_kernel(
    name: str,
    fwd_fn,
    bwd_fn,
    warmup: int = 10,
    rep: int = 50,
) -> dict:
    """Benchmark a kernel's forward and backward pass."""
    torch.cuda.reset_peak_memory_stats()
    
    # Warmup
    for _ in range(warmup):
        try:
            fwd_fn()
            bwd_fn()
        except Exception as e:
            return {"name": name, "error": str(e), "success": False}
    torch.cuda.synchronize()
    
    # Benchmark forward
    try:
        fwd_ms = triton.testing.do_bench(fwd_fn, warmup=warmup, rep=rep)
    except Exception as e:
        return {"name": name, "error": f"Forward failed: {e}", "success": False}
    
    # Benchmark forward+backward
    try:
        total_ms = triton.testing.do_bench(bwd_fn, warmup=warmup, rep=rep)
    except Exception as e:
        return {"name": name, "error": f"Backward failed: {e}", "success": False}
    
    bwd_ms = total_ms - fwd_ms
    peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
    
    return {
        "name": name,
        "fwd_time_ms": round(fwd_ms, 4),
        "bwd_time_ms": round(bwd_ms, 4),
        "total_time_ms": round(total_ms, 4),
        "throughput_iters_per_sec": round(1000 / total_ms, 1),
        "peak_memory_mb": round(peak_mem, 1),
        "success": True,
    }


def run_benchmark(
    batch_size: int = 4,
    n_heads: int = 32,
    head_dim: int = 64,
    seq_lengths: list[int] = None,
    dtype: torch.dtype = torch.bfloat16,
    warmup: int = 10,
    rep: int = 50,
) -> dict:
    """Run comprehensive benchmark across all kernels and sequence lengths."""
    
    if seq_lengths is None:
        seq_lengths = [1024, 2048, 4096, 8192]
    
    gpu_info = get_gpu_info()
    config = {
        "batch_size": batch_size,
        "n_heads": n_heads,
        "head_dim": head_dim,
        "dtype": str(dtype),
        "warmup_iters": warmup,
        "bench_iters": rep,
        "seq_lengths": seq_lengths,
    }
    
    results = []
    
    print("=" * 80)
    print(f"Lightning Attention-3 Benchmark")
    print(f"GPU: {gpu_info['name']} (SM {gpu_info['compute_capability']})")
    print(f"Config: B={batch_size}, H={n_heads}, D={head_dim}, dtype={dtype}")
    print("=" * 80)
    
    for seq_len in seq_lengths:
        print(f"\n{'='*60}")
        print(f"Sequence Length: {seq_len}")
        print(f"{'='*60}")
        
        # Create fresh tensors for each sequence length
        def make_tensors(requires_grad=True):
            q = torch.randn(batch_size, n_heads, seq_len, head_dim, 
                          device='cuda', dtype=dtype, requires_grad=requires_grad)
            k = torch.randn(batch_size, n_heads, seq_len, head_dim,
                          device='cuda', dtype=dtype, requires_grad=requires_grad)
            v = torch.randn(batch_size, n_heads, seq_len, head_dim,
                          device='cuda', dtype=dtype, requires_grad=requires_grad)
            return q, k, v
        
        # ===== LA3-Original (with decay) =====
        q, k, v = make_tensors()
        s = torch.ones(n_heads, device='cuda', dtype=dtype)  # decay parameter

        _printed_la3_orig_cfg = False
        
        def fwd_la3_orig():
            return lightning_attn3(q, k, v, s)
        
        def bwd_la3_orig():
            q.grad = k.grad = v.grad = None
            out = lightning_attn3(q, k, v, s)
            out.sum().backward()
            nonlocal _printed_la3_orig_cfg
            if not _printed_la3_orig_cfg:
                _printed_la3_orig_cfg = True
                try:
                    decay_mod = importlib.import_module(
                        'hydra.attention.backends.lightning_attn3.ops.triton.lightning_attn3'
                    )
                    device_idx = torch.cuda.current_device()
                    cfg = decay_mod._BWD_TILE_CACHE.get(device_idx)
                    if cfg is not None:
                        print(f"    [cfg] LA3-Original bwd tiles: BLOCK={cfg[0]}, CBLOCK={cfg[1]}")
                except Exception:
                    pass
        
        result = benchmark_kernel("LA3-Original", fwd_la3_orig, bwd_la3_orig, warmup, rep)
        result["seq_len"] = seq_len
        results.append(result)
        print(f"  LA3-Original:  {result.get('total_time_ms', 'N/A'):>8}ms  "
              f"(fwd: {result.get('fwd_time_ms', 'N/A')}ms, bwd: {result.get('bwd_time_ms', 'N/A')}ms)"
              f"{'  ERROR: ' + result.get('error', '') if not result['success'] else ''}")
        
        # ===== LA3-NoDecay (original non-chunked) =====
        q, k, v = make_tensors()

        _printed_la3_nodecay_cfg = False
        
        def fwd_la3_nodecay():
            return lightning_attn3_no_decay(q, k, v)
        
        def bwd_la3_nodecay():
            q.grad = k.grad = v.grad = None
            out = lightning_attn3_no_decay(q, k, v)
            out.sum().backward()
            nonlocal _printed_la3_nodecay_cfg
            if not _printed_la3_nodecay_cfg:
                _printed_la3_nodecay_cfg = True
                try:
                    nodecay_mod = importlib.import_module(
                        'hydra.attention.backends.lightning_attn3.ops.triton.lightning_attn3_no_decay'
                    )
                    device_idx = torch.cuda.current_device()
                    cfg = nodecay_mod._BWD_KERNEL_CACHE.get(device_idx)
                    if cfg is not None:
                        print(f"    [cfg] LA3-NoDecay bwd kernel: {cfg}")
                except Exception:
                    pass
        
        result = benchmark_kernel("LA3-NoDecay", fwd_la3_nodecay, bwd_la3_nodecay, warmup, rep)
        result["seq_len"] = seq_len
        results.append(result)
        print(f"  LA3-NoDecay:   {result.get('total_time_ms', 'N/A'):>8}ms  "
              f"(fwd: {result.get('fwd_time_ms', 'N/A')}ms, bwd: {result.get('bwd_time_ms', 'N/A')}ms)"
              f"{'  ERROR: ' + result.get('error', '') if not result['success'] else ''}")
        
        # ===== LA3-Chunked (Blackwell-optimized) =====
        q, k, v = make_tensors()

        _printed_la3_chunked_cfg = False
        
        def fwd_la3_chunked():
            return lightning_attn3_no_decay_chunked(q, k, v)
        
        def bwd_la3_chunked():
            q.grad = k.grad = v.grad = None
            out = lightning_attn3_no_decay_chunked(q, k, v)
            out.sum().backward()
            nonlocal _printed_la3_chunked_cfg
            if not _printed_la3_chunked_cfg:
                _printed_la3_chunked_cfg = True
                try:
                    chunked_mod = importlib.import_module(
                        'hydra.attention.backends.lightning_attn3.ops.triton.lightning_attn3_no_decay_chunked'
                    )
                    cfg = getattr(chunked_mod, '_LAST_INTER_CONFIG', None)
                    if cfg is not None:
                        print(f"    [cfg] LA3-Chunked inter: CBLOCK={cfg[0]}, stages={cfg[1]}, warps={cfg[2]}")
                    cfg = getattr(chunked_mod, '_LAST_INTRA_CONFIG', None)
                    if cfg is not None:
                        blk = getattr(chunked_mod, '_LAST_INTRA_BLOCK', None)
                        if blk is not None:
                            print(f"    [cfg] LA3-Chunked intra: BLOCK={blk}, CBLOCK={cfg[0]}, stages={cfg[1]}, warps={cfg[2]}")
                        else:
                            print(f"    [cfg] LA3-Chunked intra: CBLOCK={cfg[0]}, stages={cfg[1]}, warps={cfg[2]}")
                except Exception:
                    pass
        
        result = benchmark_kernel("LA3-Chunked", fwd_la3_chunked, bwd_la3_chunked, warmup, rep)
        result["seq_len"] = seq_len
        results.append(result)
        print(f"  LA3-Chunked:   {result.get('total_time_ms', 'N/A'):>8}ms  "
              f"(fwd: {result.get('fwd_time_ms', 'N/A')}ms, bwd: {result.get('bwd_time_ms', 'N/A')}ms)"
              f"{'  ERROR: ' + result.get('error', '') if not result['success'] else ''}")
        
        # ===== LA3-Parallel =====
        q, k, v = make_tensors()
        s = torch.ones(n_heads, device='cuda', dtype=dtype)

        _printed_la3_parallel_cfg = False
        
        def fwd_la3_parallel():
            return lightning_attn3_parallel(q, k, v, s)
        
        def bwd_la3_parallel():
            q.grad = k.grad = v.grad = None
            out = lightning_attn3_parallel(q, k, v, s)
            out.sum().backward()
            nonlocal _printed_la3_parallel_cfg
            if not _printed_la3_parallel_cfg:
                _printed_la3_parallel_cfg = True
                try:
                    par_mod = importlib.import_module(
                        'hydra.attention.backends.lightning_attn3.ops.triton.lightning_attn3_parallel'
                    )
                    cfg = getattr(par_mod, '_LAST_PARALLEL_CONFIG', None)
                    if cfg is not None:
                        print(
                            f"    [cfg] LA3-Parallel: BLOCK={cfg[0]}, CBLOCK={cfg[1]}, stages={cfg[2]}, warps={cfg[3]}"
                        )
                except Exception:
                    pass
        
        result = benchmark_kernel("LA3-Parallel", fwd_la3_parallel, bwd_la3_parallel, warmup, rep)
        result["seq_len"] = seq_len
        results.append(result)
        print(f"  LA3-Parallel:  {result.get('total_time_ms', 'N/A'):>8}ms  "
              f"(fwd: {result.get('fwd_time_ms', 'N/A')}ms, bwd: {result.get('bwd_time_ms', 'N/A')}ms)"
              f"{'  ERROR: ' + result.get('error', '') if not result['success'] else ''}")
        
        # ===== FlashAttn-2 =====
        if HAS_FLASH_ATTN:
            # Flash attention expects (B, N, H, D) layout
            q_flash = torch.randn(batch_size, seq_len, n_heads, head_dim,
                                 device='cuda', dtype=dtype, requires_grad=True)
            k_flash = torch.randn(batch_size, seq_len, n_heads, head_dim,
                                 device='cuda', dtype=dtype, requires_grad=True)
            v_flash = torch.randn(batch_size, seq_len, n_heads, head_dim,
                                 device='cuda', dtype=dtype, requires_grad=True)
            
            def fwd_flash():
                return flash_attn_func(q_flash, k_flash, v_flash, causal=True)
            
            def bwd_flash():
                q_flash.grad = k_flash.grad = v_flash.grad = None
                out = flash_attn_func(q_flash, k_flash, v_flash, causal=True)
                out.sum().backward()
            
            result = benchmark_kernel("FlashAttn-2", fwd_flash, bwd_flash, warmup, rep)
            result["seq_len"] = seq_len
            results.append(result)
            print(f"  FlashAttn-2:   {result.get('total_time_ms', 'N/A'):>8}ms  "
                  f"(fwd: {result.get('fwd_time_ms', 'N/A')}ms, bwd: {result.get('bwd_time_ms', 'N/A')}ms)"
                  f"{'  ERROR: ' + result.get('error', '') if not result['success'] else ''}")
        
        # ===== PyTorch SDPA =====
        q, k, v = make_tensors()
        
        def fwd_sdpa():
            return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        def bwd_sdpa():
            q.grad = k.grad = v.grad = None
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
            out.sum().backward()
        
        result = benchmark_kernel("PyTorch-SDPA", fwd_sdpa, bwd_sdpa, warmup, rep)
        result["seq_len"] = seq_len
        results.append(result)
        print(f"  PyTorch-SDPA:  {result.get('total_time_ms', 'N/A'):>8}ms  "
              f"(fwd: {result.get('fwd_time_ms', 'N/A')}ms, bwd: {result.get('bwd_time_ms', 'N/A')}ms)"
              f"{'  ERROR: ' + result.get('error', '') if not result['success'] else ''}")
        
        # Print speedups for this sequence length
        print(f"\n  Speedups vs SDPA:")
        sdpa_result = next((r for r in results if r["name"] == "PyTorch-SDPA" and r["seq_len"] == seq_len), None)
        if sdpa_result and sdpa_result["success"]:
            sdpa_time = sdpa_result["total_time_ms"]
            for kernel_name in ["LA3-Original", "LA3-NoDecay", "LA3-Chunked", "LA3-Parallel", "FlashAttn-2"]:
                kernel_result = next((r for r in results if r["name"] == kernel_name and r["seq_len"] == seq_len), None)
                if kernel_result and kernel_result["success"]:
                    speedup = sdpa_time / kernel_result["total_time_ms"]
                    print(f"    {kernel_name:15s}: {speedup:.2f}x")
    
    return {
        "gpu_info": gpu_info,
        "config": config,
        "results": results,
        "timestamp": datetime.now().isoformat(),
    }


def print_summary_table(data: dict):
    """Print a summary table of results."""
    results = data["results"]
    seq_lengths = data["config"]["seq_lengths"]
    
    print("\n" + "=" * 100)
    print("SUMMARY TABLE")
    print("=" * 100)
    
    # Group by kernel
    kernels = ["LA3-Original", "LA3-NoDecay", "LA3-Chunked", "LA3-Parallel", "FlashAttn-2", "PyTorch-SDPA"]
    
    # Header
    header = f"{'Kernel':<16}"
    for sl in seq_lengths:
        header += f" | N={sl:5d}"
    print(header)
    print("-" * len(header))
    
    # Data rows
    for kernel in kernels:
        row = f"{kernel:<16}"
        for sl in seq_lengths:
            result = next((r for r in results if r["name"] == kernel and r.get("seq_len") == sl), None)
            if result and result["success"]:
                row += f" | {result['total_time_ms']:7.2f}ms"
            elif result:
                row += f" |   ERROR  "
            else:
                row += f" |    N/A   "
        print(row)
    
    # Speedup vs SDPA
    print("\n" + "Speedup vs PyTorch-SDPA:")
    print("-" * len(header))
    for kernel in kernels[:-1]:  # Skip SDPA itself
        row = f"{kernel:<16}"
        for sl in seq_lengths:
            kernel_result = next((r for r in results if r["name"] == kernel and r.get("seq_len") == sl), None)
            sdpa_result = next((r for r in results if r["name"] == "PyTorch-SDPA" and r.get("seq_len") == sl), None)
            if kernel_result and kernel_result["success"] and sdpa_result and sdpa_result["success"]:
                speedup = sdpa_result["total_time_ms"] / kernel_result["total_time_ms"]
                row += f" | {speedup:7.2f}x "
            else:
                row += f" |    N/A   "
        print(row)


def generate_plots(data: dict, output_dir: Path):
    """Generate benchmark plots."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    results = data["results"]
    seq_lengths = data["config"]["seq_lengths"]
    kernels = ["LA3-Original", "LA3-NoDecay", "LA3-Chunked", "LA3-Parallel", "FlashAttn-2", "PyTorch-SDPA"]
    colors = {
        "LA3-Original": "#00d4aa",
        "LA3-NoDecay": "#00a088",
        "LA3-Chunked": "#00ff99",
        "LA3-Parallel": "#66ffcc",
        "FlashAttn-2": "#ffaa00",
        "PyTorch-SDPA": "#ff6b6b",
    }
    
    plt.style.use('dark_background')
    
    # Figure 1: Total time comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Total time bar chart
    ax1 = axes[0, 0]
    x = np.arange(len(seq_lengths))
    width = 0.12
    for i, kernel in enumerate(kernels):
        times = []
        for sl in seq_lengths:
            r = next((r for r in results if r["name"] == kernel and r.get("seq_len") == sl and r["success"]), None)
            times.append(r["total_time_ms"] if r else 0)
        if any(times):
            ax1.bar(x + i * width, times, width, label=kernel, color=colors[kernel])
    ax1.set_xlabel('Sequence Length')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('Total Time (Forward + Backward)')
    ax1.set_xticks(x + width * 2.5)
    ax1.set_xticklabels(seq_lengths)
    ax1.legend(loc='upper left', fontsize=8)
    ax1.set_yscale('log')
    
    # 2. Speedup vs SDPA
    ax2 = axes[0, 1]
    for kernel in kernels[:-1]:
        speedups = []
        for sl in seq_lengths:
            kernel_r = next((r for r in results if r["name"] == kernel and r.get("seq_len") == sl and r["success"]), None)
            sdpa_r = next((r for r in results if r["name"] == "PyTorch-SDPA" and r.get("seq_len") == sl and r["success"]), None)
            if kernel_r and sdpa_r:
                speedups.append(sdpa_r["total_time_ms"] / kernel_r["total_time_ms"])
            else:
                speedups.append(0)
        if any(speedups):
            ax2.plot(seq_lengths, speedups, 'o-', label=kernel, color=colors[kernel], linewidth=2, markersize=8)
    ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Sequence Length')
    ax2.set_ylabel('Speedup (x)')
    ax2.set_title('Speedup vs PyTorch-SDPA')
    ax2.set_xscale('log', base=2)
    ax2.set_xticks(seq_lengths)
    ax2.set_xticklabels(seq_lengths)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # 3. Forward time
    ax3 = axes[1, 0]
    for i, kernel in enumerate(kernels):
        times = []
        for sl in seq_lengths:
            r = next((r for r in results if r["name"] == kernel and r.get("seq_len") == sl and r["success"]), None)
            times.append(r["fwd_time_ms"] if r else 0)
        if any(times):
            ax3.bar(x + i * width, times, width, label=kernel, color=colors[kernel])
    ax3.set_xlabel('Sequence Length')
    ax3.set_ylabel('Time (ms)')
    ax3.set_title('Forward Pass Time')
    ax3.set_xticks(x + width * 2.5)
    ax3.set_xticklabels(seq_lengths)
    ax3.legend(loc='upper left', fontsize=8)
    ax3.set_yscale('log')
    
    # 4. Backward time
    ax4 = axes[1, 1]
    for i, kernel in enumerate(kernels):
        times = []
        for sl in seq_lengths:
            r = next((r for r in results if r["name"] == kernel and r.get("seq_len") == sl and r["success"]), None)
            times.append(r["bwd_time_ms"] if r else 0)
        if any(times):
            ax4.bar(x + i * width, times, width, label=kernel, color=colors[kernel])
    ax4.set_xlabel('Sequence Length')
    ax4.set_ylabel('Time (ms)')
    ax4.set_title('Backward Pass Time')
    ax4.set_xticks(x + width * 2.5)
    ax4.set_xticklabels(seq_lengths)
    ax4.legend(loc='upper left', fontsize=8)
    ax4.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'benchmark_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Saved {output_dir / 'benchmark_comparison.png'}")
    
    # Figure 2: Memory comparison
    fig2, ax = plt.subplots(figsize=(12, 6))
    for i, kernel in enumerate(kernels):
        mems = []
        for sl in seq_lengths:
            r = next((r for r in results if r["name"] == kernel and r.get("seq_len") == sl and r["success"]), None)
            mems.append(r["peak_memory_mb"] if r else 0)
        if any(mems):
            ax.bar(x + i * width, mems, width, label=kernel, color=colors[kernel])
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Peak Memory (MB)')
    ax.set_title('Memory Usage Comparison')
    ax.set_xticks(x + width * 2.5)
    ax.set_xticklabels(seq_lengths)
    ax.legend(fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'memory_breakdown.png', dpi=150, bbox_inches='tight')
    print(f"Saved {output_dir / 'memory_breakdown.png'}")
    
    # Figure 3: LA3 variants comparison (focused)
    fig3, ax = plt.subplots(figsize=(10, 6))
    la3_kernels = ["LA3-Original", "LA3-NoDecay", "LA3-Chunked", "LA3-Parallel"]
    x = np.arange(len(seq_lengths))
    width = 0.2
    for i, kernel in enumerate(la3_kernels):
        times = []
        for sl in seq_lengths:
            r = next((r for r in results if r["name"] == kernel and r.get("seq_len") == sl and r["success"]), None)
            times.append(r["total_time_ms"] if r else 0)
        if any(times):
            ax.bar(x + i * width, times, width, label=kernel, color=colors[kernel])
    ax.set_xlabel('Sequence Length', fontsize=12)
    ax.set_ylabel('Time (ms)', fontsize=12)
    ax.set_title('Lightning Attention-3 Variants Comparison', fontsize=14)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(seq_lengths)
    ax.legend(fontsize=10)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'la3_variants_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Saved {output_dir / 'la3_variants_comparison.png'}")


def micro_sweep_la3_chunked_inter_backward(
    *,
    batch_size: int,
    n_heads: int,
    head_dim: int,
    seq_len: int,
    dtype: torch.dtype = torch.bfloat16,
    warmup: int = 1,
    rep: int = 30,
):
    """Time the LA3-Chunked inter-backward kernel over the candidate grid.

    This is intended to answer questions like "is num_warps=2 ever best?" without
    running the full forward/backward benchmark loop or writing docs artifacts.
    """
    chunked_mod = importlib.import_module(
        'hydra.attention.backends.lightning_attn3.ops.triton.lightning_attn3_no_decay_chunked'
    )

    validate_inter_config = getattr(chunked_mod, 'validate_inter_config')
    inter_kernel = getattr(chunked_mod, '_bwd_inter_chunked_kernel')

    d = head_dim
    e = head_dim
    bh = batch_size * n_heads
    n = seq_len

    # Intentionally explore a broader space than the default runtime autotuner,
    # so we can answer questions like "is warps=2 ever competitive?".
    cblock_candidates = (64, 32, 16)
    stage_candidates = (2, 1, 3)
    warp_candidates = (8, 4, 2) if n <= 4096 else (8, 4)

    candidates: list[tuple[int, int, int]] = []
    for cblock in cblock_candidates:
        for stages in stage_candidates:
            is_valid, _ = validate_inter_config(cblock, d, e, num_stages=stages)
            if not is_valid:
                continue
            for warps in warp_candidates:
                if warps == 8 and cblock < 32:
                    continue
                if warps == 2 and cblock > 32:
                    continue
                candidates.append((cblock, stages, warps))

    if not candidates:
        print("No valid candidates found for micro sweep.")
        return

    # Representative tensors for inter-kernel timing.
    q = torch.randn((1, bh, n, d), device='cuda', dtype=dtype)
    k = torch.randn((1, bh, n, d), device='cuda', dtype=dtype)
    v = torch.randn((1, bh, n, e), device='cuda', dtype=dtype)
    do = torch.randn((1, bh, n, e), device='cuda', dtype=dtype)
    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)

    stride_qb, stride_qh, stride_qn, stride_qd = q.stride()
    stride_kb, stride_kh, stride_kn, stride_kd = k.stride()
    stride_vb, stride_vh, stride_vn, stride_ve = v.stride()
    stride_dob, stride_doh, stride_don, stride_doe = do.stride()

    grid = (bh,)

    def _run(cfg: tuple[int, int, int]):
        cblock, stages, warps = cfg
        num_cblocks = triton.cdiv(n, cblock)
        inter_kernel[grid](
            q, k, v, do,
            dq, dk, dv,
            stride_qb, stride_qh, stride_qn, stride_qd,
            stride_kb, stride_kh, stride_kn, stride_kd,
            stride_vb, stride_vh, stride_vn, stride_ve,
            stride_dob, stride_doh, stride_don, stride_doe,
            n=n, d=d, e=e,
            CBLOCK=cblock,
            NUM_CBLOCK=num_cblocks,
            num_warps=warps,
            num_stages=stages,
        )

    # Compile + warm each candidate once to avoid mixing compile latency into timings.
    for cfg in candidates:
        dq.zero_()
        dk.zero_()
        dv.zero_()
        _run(cfg)
    torch.cuda.synchronize()

    rows: list[tuple[float, tuple[int, int, int]]] = []
    for cfg in candidates:
        dq.zero_()
        dk.zero_()
        dv.zero_()
        ms = triton.testing.do_bench(lambda: _run(cfg), warmup=warmup, rep=rep)
        rows.append((float(ms), cfg))

    rows.sort(key=lambda x: x[0])
    best_ms, best_cfg = rows[0]

    print("\n" + "=" * 60)
    print("LA3-Chunked inter-backward micro sweep")
    print(f"Shape: bh={bh}, n={n}, d={d}, e={e}, dtype={dtype}")
    print(f"Bench: warmup={warmup}, rep={rep} (compile excluded)")
    print("=" * 60)
    for ms, (cblock, stages, warps) in rows:
        rel = ms / best_ms
        print(f"  {ms:8.4f} ms  (x{rel:5.2f})   CBLOCK={cblock:>2d}  stages={stages}  warps={warps}")
    print(f"\nBest: {best_ms:.4f} ms @ CBLOCK={best_cfg[0]}, stages={best_cfg[1]}, warps={best_cfg[2]}")


def micro_sweep_la3_chunked_intra_backward(
    *,
    batch_size: int,
    n_heads: int,
    head_dim: int,
    seq_len: int,
    dtype: torch.dtype = torch.bfloat16,
    warmup: int = 1,
    rep: int = 30,
):
    """Time the LA3-Chunked intra-backward kernel over a candidate grid.

    The intra kernel is launched over (b*h, ceil_div(n, BLOCK)) blocks and is
    sensitive to (CBLOCK, num_stages, num_warps) under the fixed BLOCK used by
    the implementation.
    """
    chunked_mod = importlib.import_module(
        'hydra.attention.backends.lightning_attn3.ops.triton.lightning_attn3_no_decay_chunked'
    )

    validate_config = getattr(chunked_mod, 'validate_config')
    intra_kernel = getattr(chunked_mod, '_bwd_intra_chunked_kernel')

    d = head_dim
    e = head_dim
    b = batch_size
    h = n_heads
    n = seq_len

    # Must match the implementation's BLOCK.
    block = 64

    # Intentionally explore a broader space than the default runtime autotuner.
    cblock_candidates = (64, 32, 16)
    stage_candidates = (2, 1, 3)
    warp_candidates = (8, 4, 2)

    candidates: list[tuple[int, int, int]] = []
    for cblock in cblock_candidates:
        if cblock > block or (block % cblock) != 0:
            continue
        for stages in stage_candidates:
            is_valid, _ = validate_config(cblock, d, num_stages=stages)
            if not is_valid:
                continue
            for warps in warp_candidates:
                if warps == 8 and cblock < 32:
                    continue
                candidates.append((cblock, stages, warps))

    if not candidates:
        print("No valid candidates found for micro sweep.")
        return

    q = torch.randn((b, h, n, d), device='cuda', dtype=dtype)
    k = torch.randn((b, h, n, d), device='cuda', dtype=dtype)
    v = torch.randn((b, h, n, e), device='cuda', dtype=dtype)
    do = torch.randn((b, h, n, e), device='cuda', dtype=dtype)
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)

    stride_qb, stride_qh, stride_qn, stride_qd = q.stride()
    stride_kb, stride_kh, stride_kn, stride_kd = k.stride()
    stride_vb, stride_vh, stride_vn, stride_ve = v.stride()
    stride_dob, stride_doh, stride_don, stride_doe = do.stride()

    bh = b * h
    num_block = triton.cdiv(n, block)
    grid = (bh, num_block)

    def _run(cfg: tuple[int, int, int]):
        cblock, stages, warps = cfg
        num_cblock = block // cblock
        intra_kernel[grid](
            q, k, v, do,
            dq, dk, dv,
            stride_qb, stride_qh, stride_qn, stride_qd,
            stride_kb, stride_kh, stride_kn, stride_kd,
            stride_vb, stride_vh, stride_vn, stride_ve,
            stride_dob, stride_doh, stride_don, stride_doe,
            n=n, d=d, e=e,
            BLOCK=block,
            CBLOCK=cblock,
            NUM_CBLOCK=num_cblock,
            num_warps=warps,
            num_stages=stages,
        )

    for cfg in candidates:
        _run(cfg)
    torch.cuda.synchronize()

    rows: list[tuple[float, tuple[int, int, int]]] = []
    for cfg in candidates:
        ms = triton.testing.do_bench(lambda: _run(cfg), warmup=warmup, rep=rep)
        rows.append((float(ms), cfg))

    rows.sort(key=lambda x: x[0])
    best_ms, best_cfg = rows[0]

    print("\n" + "=" * 60)
    print("LA3-Chunked intra-backward micro sweep")
    print(f"Shape: b={b}, h={h}, n={n}, d={d}, e={e}, dtype={dtype}, BLOCK={block}")
    print(f"Bench: warmup={warmup}, rep={rep} (compile excluded)")
    print("=" * 60)
    for ms, (cblock, stages, warps) in rows:
        rel = ms / best_ms
        print(f"  {ms:8.4f} ms  (x{rel:5.2f})   CBLOCK={cblock:>2d}  stages={stages}  warps={warps}")
    print(f"\nBest: {best_ms:.4f} ms @ CBLOCK={best_cfg[0]}, stages={best_cfg[1]}, warps={best_cfg[2]}")


def micro_sweep_la3_chunked_intra_block_backward(
    *,
    batch_size: int,
    n_heads: int,
    head_dim: int,
    seq_len: int,
    dtype: torch.dtype = torch.bfloat16,
    warmup: int = 1,
    rep: int = 20,
):
    """Time LA3-Chunked intra-backward over BLOCK and (CBLOCK, stages, warps).

    This is used to validate whether changing BLOCK (e.g., 64 vs 128) helps.
    """
    chunked_mod = importlib.import_module(
        'hydra.attention.backends.lightning_attn3.ops.triton.lightning_attn3_no_decay_chunked'
    )

    validate_config = getattr(chunked_mod, 'validate_config')
    intra_kernel = getattr(chunked_mod, '_bwd_intra_chunked_kernel')

    d = head_dim
    e = head_dim
    b = batch_size
    h = n_heads
    n = seq_len

    block_candidates = tuple(bc for bc in (128, 64) if bc <= max(32, n))
    stage_candidates = (2, 1)
    warp_candidates = (4, 2)

    candidates: list[tuple[int, int, int, int]] = []
    for block in block_candidates:
        for cblock in (32, 16):
            if cblock > block or (block % cblock) != 0:
                continue
            for stages in stage_candidates:
                is_valid, _ = validate_config(cblock, d, num_stages=stages)
                if not is_valid:
                    continue
                for warps in warp_candidates:
                    candidates.append((block, cblock, stages, warps))

    if not candidates:
        print("No valid candidates found for micro sweep.")
        return

    q = torch.randn((b, h, n, d), device='cuda', dtype=dtype)
    k = torch.randn((b, h, n, d), device='cuda', dtype=dtype)
    v = torch.randn((b, h, n, e), device='cuda', dtype=dtype)
    do = torch.randn((b, h, n, e), device='cuda', dtype=dtype)
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)

    stride_qb, stride_qh, stride_qn, stride_qd = q.stride()
    stride_kb, stride_kh, stride_kn, stride_kd = k.stride()
    stride_vb, stride_vh, stride_vn, stride_ve = v.stride()
    stride_dob, stride_doh, stride_don, stride_doe = do.stride()

    bh = b * h

    def _run(cfg: tuple[int, int, int, int]):
        block, cblock, stages, warps = cfg
        num_block = triton.cdiv(n, block)
        grid = (bh, num_block)
        num_cblock = block // cblock
        intra_kernel[grid](
            q, k, v, do,
            dq, dk, dv,
            stride_qb, stride_qh, stride_qn, stride_qd,
            stride_kb, stride_kh, stride_kn, stride_kd,
            stride_vb, stride_vh, stride_vn, stride_ve,
            stride_dob, stride_doh, stride_don, stride_doe,
            n=n, d=d, e=e,
            BLOCK=block,
            CBLOCK=cblock,
            NUM_CBLOCK=num_cblock,
            num_warps=warps,
            num_stages=stages,
        )

    for cfg in candidates:
        _run(cfg)
    torch.cuda.synchronize()

    rows: list[tuple[float, tuple[int, int, int, int]]] = []
    for cfg in candidates:
        ms = triton.testing.do_bench(lambda: _run(cfg), warmup=warmup, rep=rep)
        rows.append((float(ms), cfg))

    rows.sort(key=lambda x: x[0])
    best_ms, best_cfg = rows[0]

    print("\n" + "=" * 60)
    print("LA3-Chunked intra-backward BLOCK micro sweep")
    print(f"Shape: b={b}, h={h}, n={n}, d={d}, e={e}, dtype={dtype}")
    print(f"Bench: warmup={warmup}, rep={rep} (compile excluded)")
    print("=" * 60)
    for ms, (block, cblock, stages, warps) in rows:
        rel = ms / best_ms
        print(
            f"  {ms:8.4f} ms  (x{rel:5.2f})   BLOCK={block:>3d}  CBLOCK={cblock:>2d}  stages={stages}  warps={warps}"
        )
    print(
        f"\nBest: {best_ms:.4f} ms @ BLOCK={best_cfg[0]}, CBLOCK={best_cfg[1]}, stages={best_cfg[2]}, warps={best_cfg[3]}"
    )


def micro_sweep_la3_parallel(
    *,
    batch_size: int,
    n_heads: int,
    head_dim: int,
    seq_len: int,
    dtype: torch.dtype = torch.bfloat16,
    warmup: int = 1,
    rep: int = 20,
):
    """Time LA3-Parallel over a small candidate grid for one sequence length.

    Uses the internal `_FORCE_PARALLEL_CONFIG` escape hatch in the parallel module
    so we can measure (BLOCK, CBLOCK, num_stages, num_warps) directly.
    """
    par_mod = importlib.import_module(
        'hydra.attention.backends.lightning_attn3.ops.triton.lightning_attn3_parallel'
    )

    force_name = '_FORCE_PARALLEL_CONFIG'
    if not hasattr(par_mod, force_name):
        print('LA3-Parallel micro sweep unavailable: missing _FORCE_PARALLEL_CONFIG')
        return

    b, h, n, d = batch_size, n_heads, seq_len, head_dim
    e = head_dim

    # Candidate grid (kept small to limit compile overhead).
    block_candidates = tuple(bc for bc in (256, 128, 64, 32) if bc <= n and (n % bc) == 0)
    cblock_candidates = (64, 32, 16)
    stage_candidates = (2, 1)
    warp_candidates = (8, 4, 2) if n <= 2048 else (8, 4)

    candidates: list[tuple[int, int, int, int]] = []
    for block in block_candidates:
        for cblock in cblock_candidates:
            if cblock > block or (block % cblock) != 0:
                continue
            for stages in stage_candidates:
                for warps in warp_candidates:
                    if warps == 8 and cblock < 32:
                        continue
                    candidates.append((block, cblock, stages, warps))

    if not candidates:
        print('No valid LA3-Parallel candidates for this shape (needs n % BLOCK == 0).')
        return

    q = torch.randn((b, h, n, d), device='cuda', dtype=dtype, requires_grad=True)
    k = torch.randn((b, h, n, d), device='cuda', dtype=dtype, requires_grad=True)
    v = torch.randn((b, h, n, e), device='cuda', dtype=dtype, requires_grad=True)
    s = torch.ones((h,), device='cuda', dtype=dtype)

    def _run_one():
        q.grad = k.grad = v.grad = None
        out = lightning_attn3_parallel(q, k, v, s)
        out.sum().backward()

    # Compile + warm each candidate once to exclude compile from timing.
    with torch.enable_grad():
        for cfg in candidates:
            setattr(par_mod, force_name, cfg)
            try:
                _run_one()
            finally:
                setattr(par_mod, force_name, None)
        torch.cuda.synchronize()

        rows: list[tuple[float, tuple[int, int, int, int]]] = []
        for cfg in candidates:
            setattr(par_mod, force_name, cfg)
            try:
                ms = triton.testing.do_bench(_run_one, warmup=warmup, rep=rep)
            finally:
                setattr(par_mod, force_name, None)
            rows.append((float(ms), cfg))

    rows.sort(key=lambda x: x[0])
    best_ms, best_cfg = rows[0]

    print("\n" + "=" * 60)
    print("LA3-Parallel micro sweep")
    print(f"Shape: b={b}, h={h}, n={n}, d={d}, e={e}, dtype={dtype}")
    print(f"Bench: warmup={warmup}, rep={rep} (compile excluded)")
    print("=" * 60)
    for ms, (block, cblock, stages, warps) in rows:
        rel = ms / best_ms
        print(f"  {ms:8.4f} ms  (x{rel:5.2f})   BLOCK={block:>3d}  CBLOCK={cblock:>2d}  stages={stages}  warps={warps}")
    print(f"\nBest: {best_ms:.4f} ms @ BLOCK={best_cfg[0]}, CBLOCK={best_cfg[1]}, stages={best_cfg[2]}, warps={best_cfg[3]}")


def main():
    parser = argparse.ArgumentParser(description="Lightning Attention-3 Benchmark")
    parser.add_argument("--quick", action="store_true", help="Quick test with fewer sequence lengths")
    parser.add_argument(
        "--micro-sweep",
        action="store_true",
        help="Time each LA3-Chunked inter-backward candidate config for one seq length",
    )
    parser.add_argument(
        "--micro-sweep-chunked-intra",
        action="store_true",
        help="Time each LA3-Chunked intra-backward candidate config for one seq length",
    )
    parser.add_argument(
        "--micro-sweep-chunked-intra-block",
        action="store_true",
        help="Time LA3-Chunked intra-backward over BLOCK and config candidates for one seq length",
    )
    parser.add_argument(
        "--micro-sweep-parallel",
        action="store_true",
        help="Time each LA3-Parallel candidate config for one seq length",
    )
    parser.add_argument(
        "--micro-sweep-n",
        type=int,
        default=1024,
        help="Sequence length for --micro-sweep (default: 1024)",
    )
    parser.add_argument(
        "--micro-sweep-rep",
        type=int,
        default=30,
        help="Repetitions for --micro-sweep do_bench (default: 30)",
    )
    parser.add_argument("--save", action="store_true", help="Save results to docs/")
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--n-heads", type=int, default=32, help="Number of attention heads")
    parser.add_argument("--head-dim", type=int, default=64, help="Head dimension")
    parser.add_argument("--seq-lengths", type=int, nargs="+", default=None, help="Sequence lengths to test")
    args = parser.parse_args()

    if args.micro_sweep:
        micro_sweep_la3_chunked_inter_backward(
            batch_size=args.batch_size,
            n_heads=args.n_heads,
            head_dim=args.head_dim,
            seq_len=args.micro_sweep_n,
            dtype=torch.bfloat16,
            warmup=1,
            rep=args.micro_sweep_rep,
        )
        return None

    if args.micro_sweep_chunked_intra:
        micro_sweep_la3_chunked_intra_backward(
            batch_size=args.batch_size,
            n_heads=args.n_heads,
            head_dim=args.head_dim,
            seq_len=args.micro_sweep_n,
            dtype=torch.bfloat16,
            warmup=1,
            rep=args.micro_sweep_rep,
        )
        return None

    if args.micro_sweep_chunked_intra_block:
        micro_sweep_la3_chunked_intra_block_backward(
            batch_size=args.batch_size,
            n_heads=args.n_heads,
            head_dim=args.head_dim,
            seq_len=args.micro_sweep_n,
            dtype=torch.bfloat16,
            warmup=1,
            rep=args.micro_sweep_rep,
        )
        return None

    if args.micro_sweep_parallel:
        micro_sweep_la3_parallel(
            batch_size=args.batch_size,
            n_heads=args.n_heads,
            head_dim=args.head_dim,
            seq_len=args.micro_sweep_n,
            dtype=torch.bfloat16,
            warmup=1,
            rep=args.micro_sweep_rep,
        )
        return None
    
    if args.quick:
        seq_lengths = [1024, 4096]
    elif args.seq_lengths:
        seq_lengths = args.seq_lengths
    else:
        seq_lengths = [1024, 2048, 4096, 8192]
    
    # Run benchmark
    data = run_benchmark(
        batch_size=args.batch_size,
        n_heads=args.n_heads,
        head_dim=args.head_dim,
        seq_lengths=seq_lengths,
    )
    
    # Print summary
    print_summary_table(data)
    
    # Save results
    docs_dir = Path(__file__).parent / "docs"
    docs_dir.mkdir(exist_ok=True)
    
    if args.save:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = docs_dir / f"benchmark_results_{ts}.json"
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nSaved results to {out_path}")
    
    if args.plot:
        generate_plots(data, docs_dir)
    
    return data


if __name__ == "__main__":
    main()
