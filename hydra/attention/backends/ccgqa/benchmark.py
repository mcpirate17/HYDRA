#!/usr/bin/env python3
"""
CCGQA Attention Comprehensive Benchmark

Compares CCGQA against baselines:
- CCGQA: Compressed Convolutional Grouped Query Attention
- GQA: Standard Grouped Query Attention (PyTorch reference)
- MHA: Multi-Head Attention (baseline)
- FlashAttn-2: flash_attn_func (if available)
- PyTorch-SDPA: scaled_dot_product_attention

Usage:
    python benchmark.py                    # Full benchmark
    python benchmark.py --quick            # Quick test (fewer seq lengths)
    python benchmark.py --device cpu       # CPU benchmark
    python benchmark.py --save             # Save results to docs/
    python benchmark.py --plot             # Generate plots
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

# Add repo root to path for imports when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from hydra.attention.backends.ccgqa.attention import CCGQAAttention

# Try to import Flash Attention
try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False
    print("Note: flash_attn not available, skipping FlashAttn-2 benchmark")


def get_gpu_info() -> dict:
    """Get GPU information."""
    if not torch.cuda.is_available():
        return {"name": "CPU", "compute_capability": "N/A", "memory_gb": 0}
    
    props = torch.cuda.get_device_properties(0)
    return {
        "name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "memory_gb": round(props.total_memory / 1e9, 1),
    }


class ReferenceGQA(torch.nn.Module):
    """Reference Grouped Query Attention implementation."""
    
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        
        self.wq = torch.nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = torch.nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = torch.nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wo = torch.nn.Linear(n_heads * self.head_dim, dim, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        
        # Q, K, V projections
        q = self.wq(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, N, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, N, self.n_kv_heads, self.head_dim).transpose(1, 2)
        
        # Expand K, V to match Q heads (GQA)
        if self.n_kv_heads < self.n_heads:
            k = k.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)
            v = v.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        out = attn @ v
        
        # Output projection
        out = out.transpose(1, 2).contiguous().view(B, N, -1)
        return self.wo(out)


class ReferenceMHA(torch.nn.Module):
    """Reference Multi-Head Attention implementation."""
    
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        
        self.wq = torch.nn.Linear(dim, dim, bias=False)
        self.wk = torch.nn.Linear(dim, dim, bias=False)
        self.wv = torch.nn.Linear(dim, dim, bias=False)
        self.wo = torch.nn.Linear(dim, dim, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        
        q = self.wq(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        out = attn @ v
        
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.wo(out)


def benchmark_kernel(
    kernel_name: str,
    forward_fn,
    x: torch.Tensor,
    warmup: int = 10,
    rep: int = 50,
    device: str = "cuda",
) -> dict:
    """Benchmark a single kernel."""
    
    # Warmup - include backward to trigger Triton compilation for both passes
    for _ in range(warmup):
        try:
            out = forward_fn(x)
            loss = out.sum()
            loss.backward()
            x.grad = None
            if device == "cuda":
                torch.cuda.synchronize()
        except Exception as e:
            return {
                "kernel": kernel_name,
                "error": str(e),
                "success": False,
            }
    
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    
    # Forward timing
    t0 = time.perf_counter()
    for _ in range(rep):
        out = forward_fn(x)
        if device == "cuda":
            torch.cuda.synchronize()
    t1 = time.perf_counter()
    fwd_ms = (t1 - t0) * 1000 / rep

    peak_fwd_mem = 0.0
    if device == "cuda":
        peak_fwd_mem = torch.cuda.max_memory_allocated() / 1e6
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    
    # Backward timing
    t0 = time.perf_counter()
    for _ in range(rep):
        out = forward_fn(x)
        loss = out.sum()
        loss.backward()
        if device == "cuda":
            torch.cuda.synchronize()
        x.grad = None  # Clear gradients
    t1 = time.perf_counter()
    total_ms = (t1 - t0) * 1000 / rep
    bwd_ms = total_ms - fwd_ms

    peak_bwd_total_mem = torch.cuda.max_memory_allocated() / 1e6 if device == "cuda" else 0.0
    peak_bwd_extra_mem = max(0.0, peak_bwd_total_mem - peak_fwd_mem) if device == "cuda" else 0.0
    
    return {
        "kernel": kernel_name,
        "fwd_time_ms": round(fwd_ms, 4),
        "bwd_time_ms": round(bwd_ms, 4),
        "total_time_ms": round(total_ms, 4),
        "throughput_iters_per_sec": round(1000 / total_ms, 1),
        "peak_memory_mb": round(peak_bwd_total_mem, 1),
        "fwd_memory_mb": round(peak_fwd_mem, 1),
        "bwd_memory_mb": round(peak_bwd_extra_mem, 1),
        "success": True,
    }


def run_benchmark(
    batch_size: int = 4,
    n_heads: int = 8,
    n_kv_heads: int = 2,
    head_dim: int = 64,
    compression_factor: int = 4,
    seq_lengths: list[int] = None,
    dtype: torch.dtype = torch.bfloat16,
    warmup: int = 10,
    rep: int = 50,
    device: str = "cuda",
) -> dict:
    """Run comprehensive benchmark across all kernels and sequence lengths."""
    
    if seq_lengths is None:
        seq_lengths = [512, 1024, 2048, 4096] if device == "cuda" else [128, 256, 512]
    
    dim = n_heads * head_dim
    
    gpu_info = get_gpu_info()
    config = {
        "batch_size": batch_size,
        "n_heads": n_heads,
        "n_kv_heads": n_kv_heads,
        "head_dim": head_dim,
        "dim": dim,
        "compression_factor": compression_factor,
        "dtype": str(dtype),
        "warmup_iters": warmup,
        "bench_iters": rep,
        "seq_lengths": seq_lengths,
        "device": device,
    }
    
    results = []
    
    print("=" * 80)
    print(f"CCGQA Attention Benchmark")
    print(f"Device: {gpu_info['name']} ({device})")
    if device == "cuda":
        print(f"Compute: SM {gpu_info['compute_capability']}")
    print(f"Config: B={batch_size}, H={n_heads}, KV={n_kv_heads}, D={head_dim}, C={compression_factor}x")
    print("=" * 80)
    
    for seq_len in seq_lengths:
        print(f"\n{'='*60}")
        print(f"Sequence Length: {seq_len}")
        print(f"{'='*60}")
        
        # Create tensors
        x = torch.randn(batch_size, seq_len, dim,
                       device=device, dtype=dtype, requires_grad=True)
        
        # CCGQA Original
        print("\n[1/6] CCGQA Original (Unfused)...")
        ccgqa_orig = CCGQAAttention(
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            compression_factor=compression_factor,
            max_seq_len=seq_len * 2,
            use_fused_kernel=False,
        ).to(device=device, dtype=dtype)
        
        result = benchmark_kernel("CCGQA Original (Unfused)", lambda inp: ccgqa_orig(inp), x, warmup, rep, device)
        result["seq_len"] = seq_len
        results.append(result)
        if result["success"]:
            print(f"  ✓ {result['total_time_ms']:.2f}ms ({result['throughput_iters_per_sec']:.1f} iter/s)")
        else:
            print(f"  ✗ Error: {result.get('error', 'Unknown')}")
        
        # CCGQA2
        print("\n[2/6] CCGQA2 (Fused Path / SDPA-GQA on CUDA)...")
        ccgqa2 = CCGQAAttention(
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            compression_factor=compression_factor,
            max_seq_len=seq_len * 2,  # Extra headroom
            use_fused_kernel=True,
        ).to(device=device, dtype=dtype)
        
        result = benchmark_kernel("CCGQA2 (Fused Path)", lambda inp: ccgqa2(inp), x, warmup, rep, device)
        result["seq_len"] = seq_len
        results.append(result)
        if result["success"]:
            print(f"  ✓ {result['total_time_ms']:.2f}ms ({result['throughput_iters_per_sec']:.1f} iter/s)")
        else:
            print(f"  ✗ Error: {result.get('error', 'Unknown')}")
        
        # GQA Reference
        print("\n[3/6] GQA (Reference PyTorch)...")
        gqa = ReferenceGQA(dim, n_heads, n_kv_heads).to(device=device, dtype=dtype)
        result = benchmark_kernel("GQA-Ref", lambda inp: gqa(inp), x, warmup, rep, device)
        result["seq_len"] = seq_len
        results.append(result)
        if result["success"]:
            print(f"  ✓ {result['total_time_ms']:.2f}ms ({result['throughput_iters_per_sec']:.1f} iter/s)")
        else:
            print(f"  ✗ Error: {result.get('error', 'Unknown')}")
        
        # MHA Reference
        print("\n[3/5] MHA (Multi-Head Baseline)...")
        mha = ReferenceMHA(dim, n_heads).to(device=device, dtype=dtype)
        result = benchmark_kernel("MHA-Ref", lambda inp: mha(inp), x, warmup, rep, device)
        result["seq_len"] = seq_len
        results.append(result)
        if result["success"]:
            print(f"  ✓ {result['total_time_ms']:.2f}ms ({result['throughput_iters_per_sec']:.1f} iter/s)")
        else:
            print(f"  ✗ Error: {result.get('error', 'Unknown')}")
        
        # PyTorch SDPA
        print("\n[4/5] PyTorch SDPA...")
        def sdpa_forward(inp):
            q = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device, dtype=dtype, requires_grad=True)
            k = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device, dtype=dtype, requires_grad=True)
            v = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device, dtype=dtype, requires_grad=True)
            out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
            return out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        result = benchmark_kernel("PyTorch-SDPA", sdpa_forward, x, warmup, rep, device)
        result["seq_len"] = seq_len
        results.append(result)
        if result["success"]:
            print(f"  ✓ {result['total_time_ms']:.2f}ms ({result['throughput_iters_per_sec']:.1f} iter/s)")
        else:
            print(f"  ✗ Error: {result.get('error', 'Unknown')}")
        
        # FlashAttention-2
        if HAS_FLASH_ATTN and device == "cuda":
            print("\n[5/5] FlashAttention-2...")
            def flash_forward(inp):
                q = torch.randn(batch_size, seq_len, n_heads, head_dim, device=device, dtype=dtype, requires_grad=True)
                k = torch.randn(batch_size, seq_len, n_heads, head_dim, device=device, dtype=dtype, requires_grad=True)
                v = torch.randn(batch_size, seq_len, n_heads, head_dim, device=device, dtype=dtype, requires_grad=True)
                out = flash_attn_func(q, k, v, causal=False)
                return out.view(batch_size, seq_len, -1)
            
            result = benchmark_kernel("FlashAttn-2", flash_forward, x, warmup, rep, device)
            result["seq_len"] = seq_len
            results.append(result)
            if result["success"]:
                print(f"  ✓ {result['total_time_ms']:.2f}ms ({result['throughput_iters_per_sec']:.1f} iter/s)")
            else:
                print(f"  ✗ Error: {result.get('error', 'Unknown')}")
        else:
            print("\n[6/6] FlashAttention-2... (skipped)")
    
    return {
        "timestamp": datetime.now().isoformat(),
        "gpu_info": gpu_info,
        "config": config,
        "results": results,
    }


def print_summary_table(data: dict):
    """Print a formatted summary table."""
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    
    # Group by sequence length
    by_seqlen = {}
    for r in data["results"]:
        if not r["success"]:
            continue
        seq_len = r["seq_len"]
        if seq_len not in by_seqlen:
            by_seqlen[seq_len] = []
        by_seqlen[seq_len].append(r)
    
    for seq_len in sorted(by_seqlen.keys()):
        print(f"\nSequence Length: {seq_len}")
        print("-" * 80)
        print(f"{'Kernel':<20} {'Fwd (ms)':<12} {'Bwd (ms)':<12} {'Total (ms)':<12} {'Throughput':<15} {'Memory (MB)':<12}")
        print("-" * 80)
        
        for r in by_seqlen[seq_len]:
            print(f"{r['kernel']:<20} {r['fwd_time_ms']:<12.2f} {r['bwd_time_ms']:<12.2f} "
                  f"{r['total_time_ms']:<12.2f} {r['throughput_iters_per_sec']:<15.1f} {r['peak_memory_mb']:<12.1f}")


def generate_plots(data: dict, output_dir: Path):
    """Generate benchmark comparison plots."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available, skipping plot generation")
        return
    
    # Group results
    by_kernel = {}
    for r in data["results"]:
        if not r["success"]:
            continue
        kernel = r["kernel"]
        if kernel not in by_kernel:
            by_kernel[kernel] = {"seq_lens": [], "fwd": [], "bwd": [], "total": [], "memory": []}
        by_kernel[kernel]["seq_lens"].append(r["seq_len"])
        by_kernel[kernel]["fwd"].append(r["fwd_time_ms"])
        by_kernel[kernel]["bwd"].append(r["bwd_time_ms"])
        by_kernel[kernel]["total"].append(r["total_time_ms"])
        by_kernel[kernel]["memory"].append(r["peak_memory_mb"])
    
    # Plot 1: Total time comparison
    plt.figure(figsize=(12, 6))
    for kernel, vals in by_kernel.items():
        plt.plot(vals["seq_lens"], vals["total"], marker='o', label=kernel, linewidth=2)
    plt.xlabel("Sequence Length", fontsize=12)
    plt.ylabel("Total Time (ms)", fontsize=12)
    plt.title("CCGQA2 vs Baselines: Total Time", fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "benchmark_comparison.png", dpi=150)
    print(f"  ✓ Saved: {output_dir / 'benchmark_comparison.png'}")
    plt.close()
    
    # Plot 2: Forward vs Backward time
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    for kernel, vals in by_kernel.items():
        ax1.plot(vals["seq_lens"], vals["fwd"], marker='o', label=kernel, linewidth=2)
        ax2.plot(vals["seq_lens"], vals["bwd"], marker='o', label=kernel, linewidth=2)
    
    ax1.set_xlabel("Sequence Length", fontsize=11)
    ax1.set_ylabel("Forward Time (ms)", fontsize=11)
    ax1.set_title("Forward Pass Timing", fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel("Sequence Length", fontsize=11)
    ax2.set_ylabel("Backward Time (ms)", fontsize=11)
    ax2.set_title("Backward Pass Timing", fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "fwd_bwd_comparison.png", dpi=150)
    print(f"  ✓ Saved: {output_dir / 'fwd_bwd_comparison.png'}")
    plt.close()
    
    # Plot 3: Memory usage (log scale)
    plt.figure(figsize=(12, 6))
    for kernel, vals in by_kernel.items():
        plt.plot(vals["seq_lens"], vals["memory"], marker='o', label=kernel, linewidth=2)
    plt.xlabel("Sequence Length", fontsize=12)
    plt.ylabel("Peak Memory (MB, log scale)", fontsize=12)
    plt.yscale('log')  # Use logarithmic scale for y-axis
    plt.title("CCGQA2 vs Baselines: Memory Usage", fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3, which='both')  # Show grid for both major and minor ticks
    plt.tight_layout()
    plt.savefig(output_dir / "memory_comparison.png", dpi=150)
    print(f"  ✓ Saved: {output_dir / 'memory_comparison.png'}")
    plt.close()

    # Plot 4: LA3-style grouped bars (time / throughput / memory)
    seq_lens = sorted({r["seq_len"] for r in data["results"] if r.get("success")})
    kernels = sorted(by_kernel.keys())
    if not seq_lens or not kernels:
        return

    colors = {
        "CCGQA Original (Unfused)": "#e74c3c",
        "CCGQA2 (Fused Path)": "#3498db",
        "GQA-Ref": "#2ecc71",
        "MHA-Ref": "#9b59b6",
        "PyTorch-SDPA": "#95a5a6",
        "FlashAttn-2": "#f39c12",
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    x = np.arange(len(seq_lens))
    width = max(0.12, min(0.22, 0.8 / max(1, len(kernels))))

    # Bar 1: Total time
    ax1 = axes[0]
    for i, kernel in enumerate(kernels):
        times = [
            next((r["total_time_ms"] for r in data["results"]
                  if r.get("success") and r["kernel"] == kernel and r["seq_len"] == s), 0)
            for s in seq_lens
        ]
        if any(t > 0 for t in times):
            ax1.bar(x + i * width, times, width, label=kernel, color=colors.get(kernel, "#333"))
    ax1.set_xlabel("Sequence Length")
    ax1.set_ylabel("Time (ms)")
    ax1.set_title("Forward + Backward Time")
    ax1.set_xticks(x + width * (len(kernels) - 1) / 2)
    ax1.set_xticklabels(seq_lens)
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # Bar 2: Throughput
    ax2 = axes[1]
    for i, kernel in enumerate(kernels):
        throughputs = [
            next((r["throughput_iters_per_sec"] for r in data["results"]
                  if r.get("success") and r["kernel"] == kernel and r["seq_len"] == s), 0)
            for s in seq_lens
        ]
        if any(t > 0 for t in throughputs):
            ax2.bar(x + i * width, throughputs, width, label=kernel, color=colors.get(kernel, "#333"))
    ax2.set_xlabel("Sequence Length")
    ax2.set_ylabel("Throughput (iter/s)")
    ax2.set_title("Training Throughput")
    ax2.set_xticks(x + width * (len(kernels) - 1) / 2)
    ax2.set_xticklabels(seq_lens)
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    # Bar 3: Memory (stacked: forward + extra backward)
    ax3 = axes[2]
    for i, kernel in enumerate(kernels):
        fwd_mem = [
            next((r.get("fwd_memory_mb", r.get("peak_memory_mb", 0)) for r in data["results"]
                  if r.get("success") and r["kernel"] == kernel and r["seq_len"] == s), 0)
            for s in seq_lens
        ]
        bwd_mem = [
            next((r.get("bwd_memory_mb", 0) for r in data["results"]
                  if r.get("success") and r["kernel"] == kernel and r["seq_len"] == s), 0)
            for s in seq_lens
        ]
        if any(m > 0 for m in fwd_mem):
            ax3.bar(
                x + i * width,
                fwd_mem,
                width,
                label=kernel,
                color=colors.get(kernel, "#333"),
                alpha=0.7,
            )
            ax3.bar(
                x + i * width,
                bwd_mem,
                width,
                bottom=fwd_mem,
                color=colors.get(kernel, "#333"),
                alpha=1.0,
                hatch="//",
            )
    ax3.set_xlabel("Sequence Length")
    ax3.set_ylabel("Memory (MB)")
    ax3.set_title("GPU Memory: Forward (solid) + Backward (hatched)")
    ax3.set_xticks(x + width * (len(kernels) - 1) / 2)
    ax3.set_xticklabels(seq_lens)
    ax3.legend(kernels, loc="upper left")
    ax3.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "benchmark_bar_comparison.png", dpi=150, bbox_inches="tight")
    print(f"  ✓ Saved: {output_dir / 'benchmark_bar_comparison.png'}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="CCGQA Attention Benchmark")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--n-heads", type=int, default=8, help="Number of query heads")
    parser.add_argument("--n-kv-heads", type=int, default=2, help="Number of KV heads (GQA)")
    parser.add_argument("--head-dim", type=int, default=64, help="Head dimension")
    parser.add_argument("--compression", type=int, default=4, help="CCGQA compression factor")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations (increased for Triton compilation)")
    parser.add_argument("--rep", type=int, default=50, help="Benchmark iterations")
    parser.add_argument("--quick", action="store_true", help="Quick test (fewer seq lengths)")
    parser.add_argument("--save", action="store_true", help="Save results to JSON")
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    args = parser.parse_args()
    
    # Parse dtype
    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    dtype = dtype_map[args.dtype]
    
    # Set sequence lengths
    if args.quick:
        seq_lengths = [512, 1024, 2048, 4096, 8192] if args.device == "cuda" else [128, 256, 512, 1024]
    else:
        seq_lengths = [512, 1024, 2048, 4096, 8192, 16384] if args.device == "cuda" else [128, 256, 512, 1024, 2048]
    
    # Run benchmark
    data = run_benchmark(
        batch_size=args.batch_size,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        head_dim=args.head_dim,
        compression_factor=args.compression,
        seq_lengths=seq_lengths,
        dtype=dtype,
        warmup=args.warmup,
        rep=args.rep,
        device=args.device,
    )
    
    # Print summary
    print_summary_table(data)
    
    # Save results
    if args.save:
        output_dir = Path(__file__).parent / "docs"
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = output_dir / f"benchmark_results_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\n✓ Results saved to: {json_path}")
    
    # Generate plots
    if args.plot:
        output_dir = Path(__file__).parent / "docs"
        output_dir.mkdir(exist_ok=True)
        print("\nGenerating plots...")
        generate_plots(data, output_dir)


if __name__ == "__main__":
    main()
