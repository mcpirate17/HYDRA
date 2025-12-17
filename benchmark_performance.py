#!/usr/bin/env python3
"""
HYDRA Performance Benchmark Script

Measures training throughput (tokens/sec), step time, and memory usage
to establish baseline before optimizations and verify improvements.

Usage:
    python benchmark_performance.py --warmup 5 --steps 20
    python benchmark_performance.py --warmup 5 --steps 20 --use-8bit
    python benchmark_performance.py --warmup 5 --steps 20 --aggressive
"""

import argparse
import gc
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# ============================================
# CUDA/cuDNN Optimizations (same as training script)
# ============================================
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')

# ============================================
# torch._dynamo fixes
# ============================================
torch._dynamo.config.allow_unspec_int_on_nn_module = True
torch._dynamo.config.cache_size_limit = 64
torch._dynamo.config.recompile_limit = 32

from hydra.model.ccgqa import CCGQAMoDMoRModel
from hydra.kernels import get_kernel_status


def get_gpu_memory_stats() -> Dict[str, float]:
    """Get current GPU memory usage in GB."""
    if not torch.cuda.is_available():
        return {"allocated": 0, "reserved": 0, "max_allocated": 0}
    
    return {
        "allocated_gb": torch.cuda.memory_allocated() / 1e9,
        "reserved_gb": torch.cuda.memory_reserved() / 1e9,
        "max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
    }


def reset_memory_stats():
    """Reset GPU memory statistics."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        gc.collect()


def create_model(
    device: torch.device,
    vocab_size: int = 50257,
    max_seq_len: int = 512,
    dim: int = 1280,
    n_layers: int = 8,
    n_heads: int = 20,
    n_kv_heads: int = 5,
    mor_recursions: int = 3,
    mod_capacity: float = 0.5,
    use_compile: bool = True,
    aggressive: bool = False,
    no_grad_checkpoint: bool = False,
) -> nn.Module:
    """Create HYDRA model for benchmarking."""
    
    use_checkpointing = not no_grad_checkpoint
    
    model = CCGQAMoDMoRModel(
        vocab_size=vocab_size,
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        max_seq_len=max_seq_len,
        ff_mult=4.0,
        dropout=0.0,
        capacity_ratio=mod_capacity,
        mor_recursions=mor_recursions,
        mor_adaptive=True,
        mor_threshold=0.1,
        use_gradient_checkpointing=use_checkpointing,
        checkpoint_every_n=2,
    )
    
    model = model.to(device)
    
    if use_compile:
        print("Compiling model with torch.compile...")
        import torch._inductor.config as inductor_config
        inductor_config.triton.cudagraphs = False
        
        if aggressive:
            # Aggressive optimizations for max performance
            inductor_config.coordinate_descent_tuning = True
            inductor_config.aggressive_fusion = True
            inductor_config.max_autotune = True
            print("  Enabled aggressive autotune optimizations")
        
        model = torch.compile(model, mode="max-autotune-no-cudagraphs")
        print("Model compiled!")
    
    return model


def create_optimizer(
    model: nn.Module,
    lr: float = 3e-4,
    weight_decay: float = 0.1,
    use_8bit: bool = False,
) -> torch.optim.Optimizer:
    """Create optimizer for benchmarking."""
    
    # Separate decay/no-decay params
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'weight' in name and 'norm' not in name and 'embed' not in name:
                decay_params.append(param)
            else:
                no_decay_params.append(param)
    
    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    
    if use_8bit:
        try:
            import bitsandbytes as bnb
            print("Using bitsandbytes 8-bit AdamW optimizer")
            return bnb.optim.AdamW8bit(param_groups, lr=lr, betas=(0.9, 0.95))
        except ImportError:
            print("WARNING: bitsandbytes not installed, falling back to standard AdamW")
            print("Install with: pip install bitsandbytes")
    
    print("Using fused AdamW optimizer")
    return torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95), fused=True)


def create_synthetic_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create synthetic batch for benchmarking."""
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    y = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    return x, y


def run_benchmark(
    warmup_steps: int = 5,
    benchmark_steps: int = 20,
    batch_size: int = 8,
    seq_len: int = 512,
    grad_accum_steps: int = 2,
    use_compile: bool = True,
    use_8bit: bool = False,
    device_str: str = "cuda",
    aggressive: bool = False,
    no_grad_checkpoint: bool = False,
) -> Dict:
    """Run benchmark and return results."""
    
    print("\n" + "="*70)
    print("HYDRA PERFORMANCE BENCHMARK")
    print("="*70)
    
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("WARNING: Running on CPU, results will not be representative")
    
    dtype = torch.bfloat16
    vocab_size = 50257
    
    # Configuration
    config = {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "grad_accum_steps": grad_accum_steps,
        "vocab_size": vocab_size,
        "warmup_steps": warmup_steps,
        "benchmark_steps": benchmark_steps,
        "use_compile": use_compile,
        "use_8bit_optimizer": use_8bit,
        "aggressive": aggressive,
        "no_grad_checkpoint": no_grad_checkpoint,
        "dtype": str(dtype),
        "device": str(device),
    }
    
    tokens_per_step = batch_size * grad_accum_steps * seq_len
    print(f"\nConfiguration:")
    print(f"  Batch size:        {batch_size}")
    print(f"  Sequence length:   {seq_len}")
    print(f"  Grad accum steps:  {grad_accum_steps}")
    print(f"  Tokens per step:   {tokens_per_step:,}")
    print(f"  Use torch.compile: {use_compile}")
    print(f"  Use 8-bit optim:   {use_8bit}")
    print(f"  Aggressive mode:   {aggressive}")
    print(f"  No grad checkpoint:{no_grad_checkpoint}")
    print(f"  dtype:             {dtype}")
    
    # Reset memory stats
    reset_memory_stats()
    
    # Create model
    print("\nCreating model...")
    model = create_model(device=device, vocab_size=vocab_size, max_seq_len=seq_len, use_compile=use_compile, aggressive=aggressive, no_grad_checkpoint=no_grad_checkpoint)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params:      {n_params:,} ({n_params/1e6:.1f}M)")
    print(f"  Trainable params:  {n_trainable:,} ({n_trainable/1e6:.1f}M)")
    config["n_params"] = n_params
    config["n_trainable_params"] = n_trainable
    
    # Create optimizer
    print("\nCreating optimizer...")
    optimizer = create_optimizer(model, use_8bit=use_8bit)
    
    # Print kernel status
    print("\nTriton kernel status:")
    kernel_status = get_kernel_status()
    for key, val in kernel_status.items():
        print(f"  {key}: {val}")
    config["kernel_status"] = kernel_status
    
    # Memory after model creation
    model_memory = get_gpu_memory_stats()
    print(f"\nMemory after model creation:")
    print(f"  Allocated: {model_memory['allocated_gb']:.2f} GB")
    print(f"  Reserved:  {model_memory['reserved_gb']:.2f} GB")
    config["memory_after_model_gb"] = model_memory["allocated_gb"]
    
    # Warmup
    print(f"\n{'='*70}")
    print(f"WARMUP ({warmup_steps} steps)")
    print(f"{'='*70}")
    
    model.train()
    
    for step in range(warmup_steps):
        optimizer.zero_grad(set_to_none=True)
        
        for _ in range(grad_accum_steps):
            x, y = create_synthetic_batch(batch_size, seq_len, vocab_size, device)
            
            with autocast("cuda", dtype=dtype):
                logits, aux_losses = model(x, return_losses=True)
                ce_loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
                aux_loss = aux_losses.get("aux_loss", 0.0)
                ponder_loss = aux_losses.get("ponder_loss", 0.0)
                loss = ce_loss + 0.1 * aux_loss + 0.01 * ponder_loss
                scaled_loss = loss / grad_accum_steps
            
            scaled_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        print(f"  Warmup step {step+1}/{warmup_steps} - Loss: {loss.item():.4f}")
    
    # Synchronize before benchmark
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    # Reset memory stats after warmup
    reset_memory_stats()
    
    # Benchmark
    print(f"\n{'='*70}")
    print(f"BENCHMARK ({benchmark_steps} steps)")
    print(f"{'='*70}")
    
    step_times: List[float] = []
    tps_values: List[float] = []
    losses: List[float] = []
    
    for step in range(benchmark_steps):
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        step_start = time.perf_counter()
        
        optimizer.zero_grad(set_to_none=True)
        
        for _ in range(grad_accum_steps):
            x, y = create_synthetic_batch(batch_size, seq_len, vocab_size, device)
            
            with autocast("cuda", dtype=dtype):
                logits, aux_losses = model(x, return_losses=True)
                ce_loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
                aux_loss = aux_losses.get("aux_loss", 0.0)
                ponder_loss = aux_losses.get("ponder_loss", 0.0)
                loss = ce_loss + 0.1 * aux_loss + 0.01 * ponder_loss
                scaled_loss = loss / grad_accum_steps
            
            scaled_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        step_time = time.perf_counter() - step_start
        tps = tokens_per_step / step_time
        
        step_times.append(step_time)
        tps_values.append(tps)
        losses.append(loss.item())
        
        print(f"  Step {step+1:3d}/{benchmark_steps} - "
              f"Time: {step_time*1000:.1f}ms - "
              f"TPS: {tps/1000:.1f}K - "
              f"Loss: {loss.item():.4f}")
    
    # Peak memory during benchmark
    peak_memory = get_gpu_memory_stats()
    
    # Calculate statistics
    avg_step_time = sum(step_times) / len(step_times)
    avg_tps = sum(tps_values) / len(tps_values)
    min_step_time = min(step_times)
    max_step_time = max(step_times)
    min_tps = min(tps_values)
    max_tps = max(tps_values)
    
    # Results
    results = {
        "config": config,
        "results": {
            "avg_step_time_ms": avg_step_time * 1000,
            "min_step_time_ms": min_step_time * 1000,
            "max_step_time_ms": max_step_time * 1000,
            "avg_tokens_per_sec": avg_tps,
            "min_tokens_per_sec": min_tps,
            "max_tokens_per_sec": max_tps,
            "avg_tokens_per_sec_k": avg_tps / 1000,
            "peak_memory_allocated_gb": peak_memory["max_allocated_gb"],
            "peak_memory_reserved_gb": peak_memory["reserved_gb"],
        },
        "step_times_ms": [t * 1000 for t in step_times],
        "tps_values": tps_values,
        "losses": losses,
        "timestamp": datetime.now().isoformat(),
    }
    
    # Print summary
    print(f"\n{'='*70}")
    print("BENCHMARK RESULTS")
    print(f"{'='*70}")
    print(f"  Average step time:     {avg_step_time*1000:.1f} ms")
    print(f"  Step time range:       [{min_step_time*1000:.1f}, {max_step_time*1000:.1f}] ms")
    print(f"  Average throughput:    {avg_tps:,.0f} tokens/sec ({avg_tps/1000:.1f}K)")
    print(f"  Throughput range:      [{min_tps/1000:.1f}K, {max_tps/1000:.1f}K] tokens/sec")
    print(f"  Peak memory allocated: {peak_memory['max_allocated_gb']:.2f} GB")
    print(f"  Peak memory reserved:  {peak_memory['reserved_gb']:.2f} GB")
    print(f"{'='*70}\n")
    
    return results


def compare_results(baseline: Dict, optimized: Dict) -> Dict:
    """Compare baseline and optimized benchmark results."""
    
    b_tps = baseline["results"]["avg_tokens_per_sec"]
    o_tps = optimized["results"]["avg_tokens_per_sec"]
    tps_improvement = (o_tps - b_tps) / b_tps * 100
    
    b_time = baseline["results"]["avg_step_time_ms"]
    o_time = optimized["results"]["avg_step_time_ms"]
    time_improvement = (b_time - o_time) / b_time * 100
    
    b_mem = baseline["results"]["peak_memory_allocated_gb"]
    o_mem = optimized["results"]["peak_memory_allocated_gb"]
    mem_improvement = (b_mem - o_mem) / b_mem * 100 if b_mem > 0 else 0
    
    comparison = {
        "baseline_tps": b_tps,
        "optimized_tps": o_tps,
        "tps_improvement_pct": tps_improvement,
        "baseline_step_time_ms": b_time,
        "optimized_step_time_ms": o_time,
        "time_improvement_pct": time_improvement,
        "baseline_memory_gb": b_mem,
        "optimized_memory_gb": o_mem,
        "memory_improvement_pct": mem_improvement,
    }
    
    print("\n" + "="*70)
    print("COMPARISON: BASELINE vs OPTIMIZED")
    print("="*70)
    print(f"  Throughput:  {b_tps/1000:.1f}K → {o_tps/1000:.1f}K tokens/sec ({tps_improvement:+.1f}%)")
    print(f"  Step time:   {b_time:.1f} → {o_time:.1f} ms ({time_improvement:+.1f}%)")
    print(f"  Memory:      {b_mem:.2f} → {o_mem:.2f} GB ({mem_improvement:+.1f}%)")
    
    if tps_improvement >= 30:
        print(f"\n  ✅ TARGET MET: {tps_improvement:.1f}% throughput improvement (>= 30%)")
    else:
        remaining = 30 - tps_improvement
        print(f"\n  ⚠️  TARGET NOT MET: Need {remaining:.1f}% more throughput improvement")
    
    print("="*70 + "\n")
    
    return comparison


def main():
    parser = argparse.ArgumentParser(description="HYDRA Performance Benchmark")
    parser.add_argument("--warmup", type=int, default=5, help="Number of warmup steps")
    parser.add_argument("--steps", type=int, default=20, help="Number of benchmark steps")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length")
    parser.add_argument("--grad-accum", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
    parser.add_argument("--use-8bit", action="store_true", help="Use 8-bit AdamW optimizer")
    parser.add_argument("--aggressive", action="store_true", help="Use aggressive torch.compile optimizations")
    parser.add_argument("--no-grad-checkpoint", action="store_true", help="Disable gradient checkpointing (faster but uses more memory)")
    parser.add_argument("--save", type=str, help="Save results to JSON file")
    
    args = parser.parse_args()
    
    results = run_benchmark(
        warmup_steps=args.warmup,
        benchmark_steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        grad_accum_steps=args.grad_accum,
        use_compile=not args.no_compile,
        use_8bit=args.use_8bit,
        aggressive=args.aggressive,
        no_grad_checkpoint=args.no_grad_checkpoint,
    )
    
    if args.save:
        with open(args.save, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.save}")
    
    return results


if __name__ == "__main__":
    main()
