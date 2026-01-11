#!/usr/bin/env python3
"""Profile data loading with actual HYDRA 100M model.

Tests whether the throughput gap is due to data loading or model compute.
"""

import time
import statistics
from typing import Dict, Any
import torch
import torch.nn as nn

import sys
sys.path.insert(0, "/home/tim/Projects/LLM/HYDRA")

from hydra import create_ccgqa_mod_mor_model
from hydra.training.config import TrainingConfig, MODEL_SIZE_CONFIGS
from hydra.data.universal_data_loader import (
    create_universal_loader,
    SyntheticDataLoader,
)


def profile_training_loop(
    loader,
    model: nn.Module,
    device: str,
    num_steps: int = 50,
    warmup_steps: int = 5,
    label: str = "Training",
) -> Dict[str, Any]:
    """Profile a training loop with data loading."""

    print(f"\n{'='*60}")
    print(f"Profiling: {label}")
    print(f"{'='*60}")

    model = model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    # Warmup
    print(f"Warming up ({warmup_steps} steps)...", end=" ", flush=True)
    for _ in range(warmup_steps):
        batch = loader.get_batch()
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids)
        loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize()
    print("done")

    # Profile
    data_times = []
    h2d_times = []
    forward_times = []
    backward_times = []
    step_times = []
    tokens_per_step = []
    buffer_sizes = []

    print(f"Profiling ({num_steps} steps)...", end=" ", flush=True)

    for i in range(num_steps):
        step_start = time.perf_counter()

        # Data loading (CPU side)
        data_start = time.perf_counter()
        batch = loader.get_batch()
        data_end = time.perf_counter()

        # H2D transfer
        h2d_start = time.perf_counter()
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        torch.cuda.synchronize()
        h2d_end = time.perf_counter()

        # Forward
        forward_start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids)
        loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        torch.cuda.synchronize()
        forward_end = time.perf_counter()

        # Backward + step
        backward_start = time.perf_counter()
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        backward_end = time.perf_counter()

        step_end = time.perf_counter()

        data_times.append(data_end - data_start)
        h2d_times.append(h2d_end - h2d_start)
        forward_times.append(forward_end - forward_start)
        backward_times.append(backward_end - backward_start)
        step_times.append(step_end - step_start)
        tokens_per_step.append(input_ids.numel())

        # Record buffer size if available
        if hasattr(loader, "token_buffer"):
            buffer_sizes.append(len(loader.token_buffer))

        if (i + 1) % 10 == 0:
            print(f"{i+1}", end=" ", flush=True)

    print("done")

    # Calculate statistics
    total_tokens = sum(tokens_per_step)
    total_time = sum(step_times)

    stats = {
        "label": label,
        "num_steps": num_steps,
        "total_tokens": total_tokens,
        "total_time_s": total_time,
        "throughput_tok_per_s": total_tokens / total_time,
        "avg_step_time_ms": statistics.mean(step_times) * 1000,
        "avg_data_time_ms": statistics.mean(data_times) * 1000,
        "avg_h2d_time_ms": statistics.mean(h2d_times) * 1000,
        "avg_forward_time_ms": statistics.mean(forward_times) * 1000,
        "avg_backward_time_ms": statistics.mean(backward_times) * 1000,
        "p50_data_time_ms": statistics.median(data_times) * 1000,
        "p99_data_time_ms": sorted(data_times)[int(0.99 * len(data_times))] * 1000,
        "max_data_time_ms": max(data_times) * 1000,
        "data_pct_of_step": statistics.mean(data_times) / statistics.mean(step_times) * 100,
    }

    if buffer_sizes:
        stats["avg_buffer_size"] = statistics.mean(buffer_sizes)
        stats["min_buffer_size"] = min(buffer_sizes)

    return stats


def main():
    print("=" * 70)
    print("HYDRA 100M Model Dataflow Profiler")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("CUDA not available, exiting")
        return

    device = "cuda"
    batch_size = 16  # 100M model batch size from config
    seq_len = 512
    num_steps = 50

    print(f"\nConfiguration:")
    print(f"  batch_size: {batch_size}")
    print(f"  seq_len: {seq_len}")
    print(f"  tokens_per_batch: {batch_size * seq_len:,}")

    # Create 100M model
    print("\nCreating HYDRA 100M model...", end=" ", flush=True)
    model_config = MODEL_SIZE_CONFIGS["100M"]
    model = create_ccgqa_mod_mor_model(
        vocab_size=50257,
        dim=model_config["mod_mor_dim"],
        n_mor_blocks=model_config["n_mor_blocks"],
        recursions_per_block=model_config["mor_recursions"],
        n_heads=model_config["mod_mor_n_heads"],
        n_kv_heads=model_config["mod_mor_n_kv_heads"],
        max_seq_len=seq_len,
    )
    num_params = sum(p.numel() for p in model.parameters())
    print(f"done ({num_params/1e6:.1f}M params)")

    # Profile with synthetic data
    print("\n--- Testing with SYNTHETIC data ---")
    synthetic_loader = SyntheticDataLoader(
        batch_size=batch_size,
        seq_len=seq_len,
        vocab_size=50257,
        device="cpu",
    )
    synthetic_stats = profile_training_loop(
        synthetic_loader,
        model,
        device,
        num_steps=num_steps,
        label="SyntheticDataLoader + HYDRA 100M",
    )

    # Profile with real data
    print("\n--- Testing with REAL data (finefineweb-local) ---")
    try:
        real_loader = create_universal_loader(
            dataset="finefineweb-local",
            batch_size=batch_size,
            seq_len=seq_len,
            vocab_size=50257,
            device="cpu",
            num_workers=4,
            prefetch_factor=2,
        )

        # Let buffer fill
        print("Waiting for buffer to fill...", end=" ", flush=True)
        time.sleep(2)
        print(f"done (buffer: {len(real_loader.token_buffer):,})")

        real_stats = profile_training_loop(
            real_loader,
            model,
            device,
            num_steps=num_steps,
            label="finefineweb-local + HYDRA 100M",
        )
        real_loader.close()
    except Exception as e:
        print(f"\nFailed to load finefineweb-local: {e}")
        import traceback
        traceback.print_exc()
        real_stats = None

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n{'Metric':<35} {'Synthetic':>15} {'Real Data':>15}")
    print("-" * 70)

    print(f"{'Throughput (tok/s)':<35} {synthetic_stats['throughput_tok_per_s']:>15,.0f}", end="")
    if real_stats:
        print(f" {real_stats['throughput_tok_per_s']:>15,.0f}")
    else:
        print(f" {'N/A':>15}")

    print(f"{'Avg step time (ms)':<35} {synthetic_stats['avg_step_time_ms']:>15.2f}", end="")
    if real_stats:
        print(f" {real_stats['avg_step_time_ms']:>15.2f}")
    else:
        print(f" {'N/A':>15}")

    print(f"{'Avg data time (ms)':<35} {synthetic_stats['avg_data_time_ms']:>15.2f}", end="")
    if real_stats:
        print(f" {real_stats['avg_data_time_ms']:>15.2f}")
    else:
        print(f" {'N/A':>15}")

    print(f"{'Avg H2D time (ms)':<35} {synthetic_stats['avg_h2d_time_ms']:>15.2f}", end="")
    if real_stats:
        print(f" {real_stats['avg_h2d_time_ms']:>15.2f}")
    else:
        print(f" {'N/A':>15}")

    print(f"{'Avg forward time (ms)':<35} {synthetic_stats['avg_forward_time_ms']:>15.2f}", end="")
    if real_stats:
        print(f" {real_stats['avg_forward_time_ms']:>15.2f}")
    else:
        print(f" {'N/A':>15}")

    print(f"{'Avg backward time (ms)':<35} {synthetic_stats['avg_backward_time_ms']:>15.2f}", end="")
    if real_stats:
        print(f" {real_stats['avg_backward_time_ms']:>15.2f}")
    else:
        print(f" {'N/A':>15}")

    print(f"{'Data % of step':<35} {synthetic_stats['data_pct_of_step']:>14.1f}%", end="")
    if real_stats:
        print(f" {real_stats['data_pct_of_step']:>14.1f}%")
    else:
        print(f" {'N/A':>15}")

    if real_stats:
        slowdown = synthetic_stats['throughput_tok_per_s'] / real_stats['throughput_tok_per_s']
        data_overhead = real_stats['avg_data_time_ms'] - synthetic_stats['avg_data_time_ms']

        print(f"\n{'Real data slowdown':<35} {slowdown:>15.2f}x")
        print(f"{'Data loading overhead (ms)':<35} {data_overhead:>15.2f}")

        if real_stats.get('avg_buffer_size'):
            print(f"\nBuffer stats:")
            print(f"  Avg size: {real_stats['avg_buffer_size']:,.0f} tokens")
            print(f"  Min size: {real_stats['min_buffer_size']:,.0f} tokens")

        # Bottleneck analysis
        print(f"\nBottleneck analysis:")
        total_per_step = real_stats['avg_step_time_ms']
        data_ms = real_stats['avg_data_time_ms']
        h2d_ms = real_stats['avg_h2d_time_ms']
        fwd_ms = real_stats['avg_forward_time_ms']
        bwd_ms = real_stats['avg_backward_time_ms']

        print(f"  Data fetch: {data_ms:.2f}ms ({data_ms/total_per_step*100:.1f}%)")
        print(f"  H2D transfer: {h2d_ms:.2f}ms ({h2d_ms/total_per_step*100:.1f}%)")
        print(f"  Forward: {fwd_ms:.2f}ms ({fwd_ms/total_per_step*100:.1f}%)")
        print(f"  Backward: {bwd_ms:.2f}ms ({bwd_ms/total_per_step*100:.1f}%)")


if __name__ == "__main__":
    main()
