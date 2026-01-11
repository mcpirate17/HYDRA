#!/usr/bin/env python3
"""Profile HYDRA 100M model training to find speed and memory bottlenecks.

Usage:
    source /home/tim/venvs/llm/bin/activate && python diagnostics/profile_100m_training.py

This script profiles:
1. Per-operation GPU time breakdown (forward, backward, optimizer)
2. Memory allocation patterns and peaks
3. CUDA kernel-level timing
4. Per-layer breakdown to identify which components are slowest
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity, schedule
from collections import defaultdict
import json
import gc

# Model size configuration for 100M (from hydra/training/config.py)
MODEL_CONFIG = {
    "mod_mor_dim": 768,
    "n_mor_blocks": 8,
    "mor_recursions": 4,
    "mod_mor_n_heads": 12,
    "mod_mor_n_kv_heads": 3,
    "vocab_size": 50257,
    "max_seq_len": 2048,
    "mod_capacity": 0.5,
    "mor_adaptive": True,
}

BATCH_SIZE = 4
SEQ_LEN = 1024
PROFILE_STEPS = 50
WARMUP_STEPS = 5


def get_memory_stats():
    """Get current CUDA memory statistics."""
    if not torch.cuda.is_available():
        return {}

    torch.cuda.synchronize()
    return {
        "allocated_mb": torch.cuda.memory_allocated() / 1024 / 1024,
        "reserved_mb": torch.cuda.memory_reserved() / 1024 / 1024,
        "max_allocated_mb": torch.cuda.max_memory_allocated() / 1024 / 1024,
        "max_reserved_mb": torch.cuda.max_memory_reserved() / 1024 / 1024,
    }


def reset_memory_stats():
    """Reset CUDA memory peak tracking."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        gc.collect()
        torch.cuda.empty_cache()


def setup_model():
    """Create 100M HYDRA model."""
    from hydra.model.framework import HydraModel

    # Enable Triton kernels for realistic profiling
    try:
        from hydra.kernels import set_use_triton_kernels, get_kernel_status
        set_use_triton_kernels(True)
        print("Kernel status:", get_kernel_status())
    except Exception as e:
        print(f"Warning: Could not configure Triton kernels: {e}")

    model = HydraModel(
        dim=MODEL_CONFIG["mod_mor_dim"],
        n_blocks=MODEL_CONFIG["n_mor_blocks"],
        n_heads=MODEL_CONFIG["mod_mor_n_heads"],
        n_kv_heads=MODEL_CONFIG["mod_mor_n_kv_heads"],
        vocab_size=MODEL_CONFIG["vocab_size"],
        max_seq_len=MODEL_CONFIG["max_seq_len"],
        n_recursions=MODEL_CONFIG["mor_recursions"],
        mod_capacity=MODEL_CONFIG["mod_capacity"],
        mor_adaptive=MODEL_CONFIG["mor_adaptive"],
    )

    model = model.to("cuda").to(torch.bfloat16)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params/1e6:.1f}M")

    return model


def create_synthetic_batch():
    """Create a synthetic training batch."""
    input_ids = torch.randint(0, MODEL_CONFIG["vocab_size"], (BATCH_SIZE, SEQ_LEN), device="cuda")
    targets = torch.randint(0, MODEL_CONFIG["vocab_size"], (BATCH_SIZE, SEQ_LEN), device="cuda")
    return input_ids, targets


def profile_memory_breakdown(model, optimizer):
    """Profile memory usage at different stages of training step."""
    print("\n" + "=" * 70)
    print("MEMORY BREAKDOWN ANALYSIS")
    print("=" * 70)

    reset_memory_stats()
    stages = []

    # Stage 1: Model loaded
    stages.append(("Model loaded", get_memory_stats()))

    # Stage 2: Batch created
    input_ids, targets = create_synthetic_batch()
    stages.append(("Batch created", get_memory_stats()))

    # Stage 3: After forward pass
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits = model(input_ids)
    stages.append(("After forward", get_memory_stats()))

    # Stage 4: After loss computation
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)
        loss = F.cross_entropy(logits_flat, targets_flat)
    stages.append(("After loss", get_memory_stats()))

    # Stage 5: After backward
    loss.backward()
    stages.append(("After backward", get_memory_stats()))

    # Stage 6: After optimizer step
    optimizer.step()
    stages.append(("After optimizer", get_memory_stats()))

    # Print breakdown
    print(f"\n{'Stage':<20} {'Allocated':>12} {'Reserved':>12} {'Delta':>12}")
    print("-" * 60)
    prev_alloc = 0
    for stage_name, stats in stages:
        alloc = stats["allocated_mb"]
        delta = alloc - prev_alloc
        print(f"{stage_name:<20} {alloc:>10.1f}MB {stats['reserved_mb']:>10.1f}MB {delta:>+10.1f}MB")
        prev_alloc = alloc

    print(f"\nPeak allocated: {stages[-1][1]['max_allocated_mb']:.1f}MB")
    print(f"Peak reserved: {stages[-1][1]['max_reserved_mb']:.1f}MB")

    return stages


def profile_detailed_timing():
    """Profile with detailed timing for each component."""
    print("\n" + "=" * 70)
    print("DETAILED TIMING ANALYSIS (100M Model)")
    print("=" * 70)
    print(f"Config: batch_size={BATCH_SIZE}, seq_len={SEQ_LEN}")
    print(f"Profiling {PROFILE_STEPS} steps (with {WARMUP_STEPS} warmup)")
    print()

    reset_memory_stats()
    model = setup_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Memory breakdown first
    profile_memory_breakdown(model, optimizer)

    # Reset for timing profiling
    reset_memory_stats()

    # Warmup
    print("\nRunning warmup steps...")
    for i in range(WARMUP_STEPS):
        input_ids, targets = create_synthetic_batch()
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids)
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = targets.view(-1)
            loss = F.cross_entropy(logits_flat, targets_flat)
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize()
    print("Warmup complete. Starting profiling...\n")

    # Manual timing for high-level breakdown
    import time
    forward_times = []
    backward_times = []
    optimizer_times = []
    total_times = []

    for step in range(PROFILE_STEPS):
        input_ids, targets = create_synthetic_batch()

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        t1 = time.perf_counter()

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids)
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = targets.view(-1)
            loss = F.cross_entropy(logits_flat, targets_flat)

        torch.cuda.synchronize()
        t2 = time.perf_counter()

        loss.backward()

        torch.cuda.synchronize()
        t3 = time.perf_counter()

        optimizer.step()

        torch.cuda.synchronize()
        t4 = time.perf_counter()

        forward_times.append((t2 - t1) * 1000)
        backward_times.append((t3 - t2) * 1000)
        optimizer_times.append((t4 - t3) * 1000)
        total_times.append((t4 - t0) * 1000)

        if (step + 1) % 10 == 0:
            print(f"  Step {step + 1}/{PROFILE_STEPS}")

    # Compute statistics
    import statistics

    print("\n" + "=" * 70)
    print("HIGH-LEVEL TIMING BREAKDOWN")
    print("=" * 70)

    def print_stats(name, times):
        avg = statistics.mean(times)
        std = statistics.stdev(times) if len(times) > 1 else 0
        total = sum(total_times)
        pct = (sum(times) / total) * 100 if total > 0 else 0
        print(f"{name:<20}: {avg:>8.2f}ms avg (std={std:.2f}ms) | {pct:>5.1f}% of step")

    print_stats("Forward pass", forward_times)
    print_stats("Backward pass", backward_times)
    print_stats("Optimizer step", optimizer_times)
    print(f"\nTotal step time: {statistics.mean(total_times):.2f}ms avg")
    print(f"Throughput: {BATCH_SIZE * SEQ_LEN / (statistics.mean(total_times) / 1000):.0f} tokens/sec")

    return model, optimizer


def profile_cuda_kernels(model, optimizer):
    """Profile CUDA kernels to identify bottlenecks."""
    print("\n" + "=" * 70)
    print("CUDA KERNEL PROFILING")
    print("=" * 70)

    reset_memory_stats()

    # Warmup inside profiler
    print("Running profiled steps...")

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for step in range(20):  # Fewer steps for kernel profiling
            input_ids, targets = create_synthetic_batch()

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                with record_function("FORWARD"):
                    logits = model(input_ids)
                with record_function("LOSS"):
                    logits_flat = logits.view(-1, logits.size(-1))
                    targets_flat = targets.view(-1)
                    loss = F.cross_entropy(logits_flat, targets_flat)

            with record_function("BACKWARD"):
                loss.backward()

            with record_function("OPTIMIZER"):
                optimizer.step()

    torch.cuda.synchronize()

    # Analyze results
    print("\n" + "=" * 70)
    print("TOP 20 CUDA OPERATIONS BY GPU TIME")
    print("=" * 70)

    key_averages = prof.key_averages()

    # Collect CUDA events
    cuda_events = []
    for event in key_averages:
        cuda_time = 0
        for attr in ['self_cuda_time_total', 'cuda_time_total', 'self_device_time_total', 'device_time_total']:
            val = getattr(event, attr, None)
            if val is not None and val > 0:
                cuda_time = val
                break
        if cuda_time > 0:
            cuda_events.append((event, cuda_time))

    cuda_events.sort(key=lambda x: x[1], reverse=True)

    total_cuda_time = sum(e[1] for e in cuda_events)

    results = []
    print(f"\n{'Rank':<5} {'Operation':<50} {'GPU Time':>12} {'%':>8} {'Calls':>8}")
    print("-" * 90)

    for i, (event, cuda_time) in enumerate(cuda_events[:20]):
        cuda_time_ms = cuda_time / 1000
        pct = (cuda_time / total_cuda_time) * 100 if total_cuda_time > 0 else 0
        name = event.key[:48]

        print(f"{i+1:<5} {name:<50} {cuda_time_ms:>10.2f}ms {pct:>7.1f}% {event.count:>8}")

        results.append({
            "rank": i + 1,
            "name": event.key,
            "cuda_time_ms": cuda_time_ms,
            "pct": pct,
            "count": event.count,
            "cpu_time_ms": getattr(event, 'cpu_time_total', 0) / 1000,
        })

    # Memory profiling summary
    print("\n" + "=" * 70)
    print("MEMORY ALLOCATION HOTSPOTS")
    print("=" * 70)

    mem_events = []
    for event in key_averages:
        mem = getattr(event, 'self_cuda_memory_usage', 0)
        if mem > 0:
            mem_events.append((event, mem))

    mem_events.sort(key=lambda x: x[1], reverse=True)

    print(f"\n{'Rank':<5} {'Operation':<50} {'Memory':>15}")
    print("-" * 75)

    for i, (event, mem) in enumerate(mem_events[:15]):
        mem_mb = mem / 1024 / 1024
        name = event.key[:48]
        print(f"{i+1:<5} {name:<50} {mem_mb:>13.2f}MB")

    # Export trace
    trace_file = Path(__file__).parent / "profile_100m_trace.json"
    prof.export_chrome_trace(str(trace_file))
    print(f"\nChrome trace saved to {trace_file}")
    print("Open chrome://tracing and load this file for detailed visualization")

    # Save results
    results_file = Path(__file__).parent / "profile_100m_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_file}")

    return prof, results


def profile_per_layer_breakdown(model, optimizer):
    """Profile each layer/block individually to find bottlenecks."""
    print("\n" + "=" * 70)
    print("PER-LAYER BREAKDOWN")
    print("=" * 70)

    # We'll use hooks to time each block
    block_times = defaultdict(list)

    def make_forward_hook(name):
        def hook(module, input, output):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        return hook

    # Time embedding, blocks, and output layers separately
    import time

    print("\nTiming individual components...")

    for step in range(10):
        input_ids, targets = create_synthetic_batch()
        optimizer.zero_grad(set_to_none=True)

        # Time embedding
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            # Access model internals
            x = model.tok_emb(input_ids)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        block_times["embedding"].append((t1 - t0) * 1000)

        # Time full forward (we already have embedding, so this measures rest)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        torch.cuda.synchronize()
        t2 = time.perf_counter()
        block_times["forward_rest"].append((t2 - t1) * 1000)

        # Time backward
        loss.backward()
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        block_times["backward"].append((t3 - t2) * 1000)

    # Print results
    print(f"\n{'Component':<30} {'Avg Time':>12} {'Std':>10}")
    print("-" * 55)

    import statistics
    for name, times in sorted(block_times.items()):
        avg = statistics.mean(times)
        std = statistics.stdev(times) if len(times) > 1 else 0
        print(f"{name:<30} {avg:>10.2f}ms {std:>8.2f}ms")


def generate_optimization_report(results):
    """Generate optimization recommendations based on profiling results."""
    print("\n" + "=" * 70)
    print("OPTIMIZATION RECOMMENDATIONS")
    print("=" * 70)

    recommendations = []

    for r in results[:10]:
        name = r["name"].lower()
        pct = r["pct"]

        if pct < 2:
            continue  # Skip minor contributors

        rec = None

        if "gemm" in name or "matmul" in name or "mm_" in name or "cutlass" in name:
            rec = {
                "op": r["name"],
                "pct": pct,
                "status": "OPTIMIZED",
                "note": "Matrix multiply already using cuBLAS/CUTLASS. Consider reducing model dim or using lower precision."
            }
        elif "flash" in name or "fmha" in name:
            rec = {
                "op": r["name"],
                "pct": pct,
                "status": "OPTIMIZED",
                "note": "Using Flash Attention. Already optimal."
            }
        elif "softmax" in name:
            rec = {
                "op": r["name"],
                "pct": pct,
                "status": "CHECK",
                "note": "Softmax can sometimes be fused with attention. Check if using Flash Attention."
            }
        elif "norm" in name or "layernorm" in name:
            rec = {
                "op": r["name"],
                "pct": pct,
                "status": "CHECK",
                "note": "Check if fused RMSNorm kernel is active (HYDRA_ENABLE_FUSED_RMS_NORM=1)."
            }
        elif "rope" in name or "rotary" in name:
            rec = {
                "op": r["name"],
                "pct": pct,
                "status": "CHECK",
                "note": "Check if fused RoPE kernel is active (HYDRA_ENABLE_FUSED_ROPE=1)."
            }
        elif "copy" in name or "contiguous" in name or "transpose" in name:
            rec = {
                "op": r["name"],
                "pct": pct,
                "status": "INVESTIGATE",
                "note": "Memory copy/layout operation. May indicate inefficient tensor layouts."
            }
        elif "adam" in name:
            rec = {
                "op": r["name"],
                "pct": pct,
                "status": "TIP",
                "note": "Consider 8-bit Adam (--8bit_adam) to reduce optimizer memory and time."
            }
        elif "cross_entropy" in name or "nll" in name:
            rec = {
                "op": r["name"],
                "pct": pct,
                "status": "TIP",
                "note": "Consider chunked cross-entropy (--chunked_ce) for memory savings."
            }
        elif "elementwise" in name or "pointwise" in name:
            rec = {
                "op": r["name"],
                "pct": pct,
                "status": "INVESTIGATE",
                "note": "Elementwise ops may be fusion candidates."
            }

        if rec:
            recommendations.append(rec)
            print(f"\n[{rec['status']}] {rec['op'][:50]}")
            print(f"  Time: {pct:.1f}% of total")
            print(f"  {rec['note']}")

    return recommendations


def main():
    """Run comprehensive profiling."""
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    print("=" * 70)
    print("HYDRA 100M MODEL PROFILING")
    print("=" * 70)
    print(f"Model config:")
    for k, v in MODEL_CONFIG.items():
        print(f"  {k}: {v}")
    print()

    # Run profiling
    model, optimizer = profile_detailed_timing()
    prof, results = profile_cuda_kernels(model, optimizer)
    profile_per_layer_breakdown(model, optimizer)
    generate_optimization_report(results)

    # Final memory summary
    print("\n" + "=" * 70)
    print("FINAL MEMORY SUMMARY")
    print("=" * 70)
    stats = get_memory_stats()
    print(f"Current allocated: {stats['allocated_mb']:.1f}MB")
    print(f"Peak allocated: {stats['max_allocated_mb']:.1f}MB")
    print(f"Peak reserved: {stats['max_reserved_mb']:.1f}MB")

    print("\n" + "=" * 70)
    print("PROFILING COMPLETE")
    print("=" * 70)
    print("\nFiles generated:")
    print(f"  - diagnostics/profile_100m_results.json (kernel timing data)")
    print(f"  - diagnostics/profile_100m_trace.json (Chrome trace)")
    print("\nTo visualize trace: open chrome://tracing and load profile_100m_trace.json")


if __name__ == "__main__":
    main()
