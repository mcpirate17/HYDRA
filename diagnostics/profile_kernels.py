#!/usr/bin/env python3
"""Profile HYDRA training to identify top CUDA kernels for optimization.

Usage:
    source /home/tim/venvs/llm/bin/activate && python diagnostics/profile_kernels.py

This script profiles HYDRA's 500M model for 100 training steps and identifies
the top CUDA kernels by GPU time for optimization opportunities.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity
from collections import defaultdict
import json

# Model size configuration for 500M (from hydra/training/config.py)
# Note: HYDRA's 500M is actually ~694M total params due to MoD/MoR architecture
MODEL_CONFIG = {
    "mod_mor_dim": 1792,
    "n_mor_blocks": 14,
    "mor_recursions": 4,
    "mod_mor_n_heads": 28,
    "mod_mor_n_kv_heads": 4,
    "vocab_size": 50257,
    "max_seq_len": 2048,
    "mod_capacity": 0.5,
    "mor_adaptive": False,  # Disable adaptive routing for profiling stability
}

BATCH_SIZE = 4
SEQ_LEN = 1024  # Shorter for profiling speed
PROFILE_STEPS = 100
WARMUP_STEPS = 10


def setup_model():
    """Create 500M HYDRA model."""
    from hydra.model.framework import HydraModel

    # Enable Triton kernels
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


def profile_training():
    """Profile training loop and identify top CUDA kernels."""
    print("=" * 70)
    print("HYDRA Kernel Profiling - 500M Model")
    print("=" * 70)
    print(f"Config: batch_size={BATCH_SIZE}, seq_len={SEQ_LEN}")
    print(f"Profiling {PROFILE_STEPS} steps (with {WARMUP_STEPS} warmup)")
    print()

    # Setup
    model = setup_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    # Note: BFloat16 doesn't need GradScaler (it has the same range as FP32)

    # Warmup (not profiled)
    print("Running warmup steps...")
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
    print(f"Warmup complete. Starting profiling...")

    # Profile
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for step in range(PROFILE_STEPS):
            input_ids, targets = create_synthetic_batch()

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                with record_function("forward"):
                    logits = model(input_ids)
                with record_function("loss_compute"):
                    logits_flat = logits.view(-1, logits.size(-1))
                    targets_flat = targets.view(-1)
                    loss = F.cross_entropy(logits_flat, targets_flat)

            with record_function("backward"):
                loss.backward()

            with record_function("optimizer_step"):
                optimizer.step()

            if (step + 1) % 20 == 0:
                print(f"  Step {step + 1}/{PROFILE_STEPS}")

    torch.cuda.synchronize()
    print(f"\nProfiling complete!")

    return prof


def analyze_profile(prof):
    """Analyze profiler results and identify top kernels."""
    print("\n" + "=" * 70)
    print("TOP 10 CUDA KERNELS BY GPU TIME")
    print("=" * 70)

    # Get CUDA kernel events
    key_averages = prof.key_averages()

    # Collect all events with their CUDA times
    # Try multiple attribute names for compatibility
    cuda_events = []
    for event in key_averages:
        # Try to get CUDA time from various attributes
        cuda_time = 0
        for attr in ['self_cuda_time_total', 'cuda_time_total', 'self_device_time_total', 'device_time_total']:
            val = getattr(event, attr, None)
            if val is not None and val > 0:
                cuda_time = val
                break

        if cuda_time > 0:
            cuda_events.append((event, cuda_time))

    # Sort by total CUDA time
    cuda_events.sort(key=lambda x: x[1], reverse=True)

    # Print top 10
    results = []
    for i, (event, cuda_time) in enumerate(cuda_events[:15]):  # Get top 15 for analysis
        cuda_time_ms = cuda_time / 1000  # Convert to ms
        count = event.count
        avg_time_us = cuda_time / count if count > 0 else 0

        result = {
            "rank": i + 1,
            "name": event.key,
            "total_time_ms": cuda_time_ms,
            "count": count,
            "avg_time_us": avg_time_us,
            "cpu_time_ms": event.cpu_time_total / 1000 if hasattr(event, 'cpu_time_total') else 0,
        }
        results.append(result)

        if i < 10:
            print(f"\n{i + 1}. {event.key}")
            print(f"   Total GPU time: {cuda_time_ms:.2f} ms ({count} calls, {avg_time_us:.1f} us/call)")

    # Also print the full table
    print("\n" + "=" * 70)
    print("FULL KERNEL TABLE (Top 30)")
    print("=" * 70)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))

    # Export to JSON
    output_file = Path(__file__).parent / "profile_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")

    # Export chrome trace
    trace_file = Path(__file__).parent / "profile_trace.json"
    prof.export_chrome_trace(str(trace_file))
    print(f"Chrome trace saved to {trace_file}")

    return results


def classify_kernel(name: str) -> dict:
    """Classify a kernel by implementation type and optimization potential."""
    name_lower = name.lower()

    info = {
        "implementation": "unknown",
        "optimizable": False,
        "effort": "unknown",
        "potential_gain": "unknown",
        "notes": "",
    }

    # Flash Attention / cuDNN attention
    if "flash" in name_lower or "fmha" in name_lower or "attention" in name_lower:
        if "flash" in name_lower or "cutlass" in name_lower or "fmha" in name_lower:
            info["implementation"] = "Flash Attention (optimized)"
            info["optimizable"] = False
            info["notes"] = "Already using highly optimized Flash Attention"
        else:
            info["implementation"] = "PyTorch native attention"
            info["optimizable"] = True
            info["effort"] = "low"
            info["potential_gain"] = "high (2-4x)"
            info["notes"] = "Consider Flash Attention 2/3 or xFormers"

    # Triton kernels
    elif "triton" in name_lower or "_fused" in name_lower:
        info["implementation"] = "Triton kernel"
        info["optimizable"] = False
        info["notes"] = "Already using fused Triton kernel"

    # GEMM operations (matmul)
    elif "gemm" in name_lower or "matmul" in name_lower or "mm_" in name_lower or "cutlass" in name_lower:
        info["implementation"] = "cuBLAS/CUTLASS GEMM"
        info["optimizable"] = False  # cuBLAS is already optimal
        info["notes"] = "cuBLAS GEMM is highly optimized; consider kernel fusion around GEMM"

    # Elementwise operations
    elif any(x in name_lower for x in ["elementwise", "pointwise", "vectorized", "copy", "fill"]):
        info["implementation"] = "PyTorch native elementwise"
        info["optimizable"] = True
        info["effort"] = "medium"
        info["potential_gain"] = "medium (1.3-2x)"
        info["notes"] = "Candidate for fusion into adjacent kernels"

    # Softmax
    elif "softmax" in name_lower:
        if "native" in name_lower or "cudnn" not in name_lower:
            info["implementation"] = "PyTorch native softmax"
            info["optimizable"] = True
            info["effort"] = "low"
            info["potential_gain"] = "medium (1.2-1.5x)"
            info["notes"] = "Liger/Triton fused softmax available"
        else:
            info["implementation"] = "cuDNN softmax"
            info["optimizable"] = False
            info["notes"] = "Already using cuDNN"

    # Layer normalization / RMS norm
    elif "norm" in name_lower or "layernorm" in name_lower or "rmsnorm" in name_lower:
        if "liger" in name_lower or "triton" in name_lower:
            info["implementation"] = "Liger/Triton RMSNorm"
            info["optimizable"] = False
            info["notes"] = "Already using fused kernel"
        else:
            info["implementation"] = "PyTorch native norm"
            info["optimizable"] = True
            info["effort"] = "low"
            info["potential_gain"] = "medium (1.5-2x)"
            info["notes"] = "Liger LigerRMSNorm provides fused kernel"

    # SiLU / SwiGLU
    elif "silu" in name_lower or "swiglu" in name_lower or "swish" in name_lower:
        if "liger" in name_lower or "triton" in name_lower or "fused" in name_lower:
            info["implementation"] = "Liger/Triton SwiGLU"
            info["optimizable"] = False
            info["notes"] = "Already using fused kernel"
        else:
            info["implementation"] = "PyTorch native SiLU"
            info["optimizable"] = True
            info["effort"] = "low"
            info["potential_gain"] = "medium (1.3x)"
            info["notes"] = "Liger LigerSiLUMulFunction provides fusion"

    # Cross entropy
    elif "cross_entropy" in name_lower or "nll" in name_lower or "log_softmax" in name_lower:
        if "liger" in name_lower or "chunked" in name_lower:
            info["implementation"] = "Liger/chunked cross-entropy"
            info["optimizable"] = False
            info["notes"] = "Already using memory-efficient CE"
        else:
            info["implementation"] = "PyTorch native cross-entropy"
            info["optimizable"] = True
            info["effort"] = "low"
            info["potential_gain"] = "high (memory: 60%, speed: 2x)"
            info["notes"] = "Liger fused linear + CE eliminates logits materialization"

    # RoPE
    elif "rope" in name_lower or "rotary" in name_lower:
        if "liger" in name_lower or "triton" in name_lower or "fused" in name_lower:
            info["implementation"] = "Liger/Triton RoPE"
            info["optimizable"] = False
            info["notes"] = "Already using fused kernel"
        else:
            info["implementation"] = "PyTorch native RoPE"
            info["optimizable"] = True
            info["effort"] = "low"
            info["potential_gain"] = "medium (2-3x)"
            info["notes"] = "Triton fused_rope available in hydra.kernels"

    # Indexing operations
    elif any(x in name_lower for x in ["index", "gather", "scatter", "embedding"]):
        info["implementation"] = "PyTorch native indexing"
        info["optimizable"] = False  # Memory-bound, hard to optimize
        info["notes"] = "Memory-bound operation; limited optimization potential"

    # Reduction operations
    elif any(x in name_lower for x in ["reduce", "sum", "mean", "max", "min"]):
        info["implementation"] = "PyTorch native reduction"
        info["optimizable"] = True
        info["effort"] = "medium"
        info["potential_gain"] = "low (1.1-1.3x)"
        info["notes"] = "Can fuse with adjacent operations"

    # Transpose/permute
    elif any(x in name_lower for x in ["transpose", "permute", "contiguous"]):
        info["implementation"] = "PyTorch native transpose"
        info["optimizable"] = True
        info["effort"] = "high"
        info["potential_gain"] = "medium (avoid entirely via layout changes)"
        info["notes"] = "Consider tensor layout reorganization to eliminate"

    # Adam optimizer
    elif "adam" in name_lower:
        info["implementation"] = "PyTorch Adam"
        info["optimizable"] = True
        info["effort"] = "low"
        info["potential_gain"] = "low-medium"
        info["notes"] = "Consider fused Adam or 8-bit Adam for memory"

    return info


def generate_report(results):
    """Generate detailed optimization report."""
    print("\n" + "=" * 70)
    print("OPTIMIZATION ANALYSIS")
    print("=" * 70)

    total_time = sum(r["total_time_ms"] for r in results)

    for result in results:
        name = result["name"]
        time_ms = result["total_time_ms"]
        pct = (time_ms / total_time) * 100 if total_time > 0 else 0

        info = classify_kernel(name)

        print(f"\n{result['rank']}. {name}")
        print(f"   Total time: {time_ms:.2f} ms ({pct:.1f}% of top-10)")
        print(f"   Implementation: {info['implementation']}")

        if info["optimizable"]:
            print(f"   OPTIMIZATION POTENTIAL:")
            print(f"     Effort: {info['effort']}")
            print(f"     Potential gain: {info['potential_gain']}")
            print(f"     Notes: {info['notes']}")
        else:
            print(f"   Already optimized: {info['notes']}")


if __name__ == "__main__":
    # Set environment for optimal profiling
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    prof = profile_training()
    results = analyze_profile(prof)
    generate_report(results)
