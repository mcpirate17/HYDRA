#!/usr/bin/env python3
"""Profile data loading during actual training to understand throughput drop.

This simulates the training loop to measure data loading behavior under load.
"""

import time
import statistics
import threading
from typing import Dict, List, Any
import torch
import torch.nn as nn

import sys
sys.path.insert(0, "/home/tim/Projects/LLM/HYDRA")

from hydra.data.universal_data_loader import (
    create_universal_loader,
    SyntheticDataLoader,
)


def create_simple_model(vocab_size: int, hidden_dim: int, num_layers: int) -> nn.Module:
    """Create a simple transformer-like model for testing."""

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, hidden_dim)
            self.layers = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=8,
                    dim_feedforward=hidden_dim * 4,
                    batch_first=True,
                )
                for _ in range(num_layers)
            ])
            self.head = nn.Linear(hidden_dim, vocab_size, bias=False)

        def forward(self, x):
            x = self.embed(x)
            for layer in self.layers:
                x = layer(x)
            return self.head(x)

    return SimpleModel()


def profile_training_loop(
    loader,
    model: nn.Module,
    device: str,
    num_steps: int = 100,
    warmup_steps: int = 10,
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
    forward_times = []
    backward_times = []
    step_times = []
    tokens_per_step = []
    buffer_sizes = []

    print(f"Profiling ({num_steps} steps)...", end=" ", flush=True)

    for i in range(num_steps):
        step_start = time.perf_counter()

        # Data loading
        data_start = time.perf_counter()
        batch = loader.get_batch()
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        torch.cuda.synchronize()
        data_end = time.perf_counter()

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
        forward_times.append(forward_end - forward_start)
        backward_times.append(backward_end - backward_start)
        step_times.append(step_end - step_start)
        tokens_per_step.append(input_ids.numel())

        # Record buffer size if available
        if hasattr(loader, "token_buffer"):
            buffer_sizes.append(len(loader.token_buffer))

        if (i + 1) % 25 == 0:
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
        "avg_forward_time_ms": statistics.mean(forward_times) * 1000,
        "avg_backward_time_ms": statistics.mean(backward_times) * 1000,
        "p50_data_time_ms": statistics.median(data_times) * 1000,
        "p99_data_time_ms": sorted(data_times)[int(0.99 * len(data_times))] * 1000,
        "min_data_time_ms": min(data_times) * 1000,
        "max_data_time_ms": max(data_times) * 1000,
        "data_pct_of_step": statistics.mean(data_times) / statistics.mean(step_times) * 100,
    }

    if buffer_sizes:
        stats["avg_buffer_size"] = statistics.mean(buffer_sizes)
        stats["min_buffer_size"] = min(buffer_sizes)
        stats["max_buffer_size"] = max(buffer_sizes)

    return stats


def profile_prefetch_behavior(loader, num_batches: int = 50) -> Dict[str, Any]:
    """Profile prefetch thread behavior."""

    print(f"\n{'='*60}")
    print("Profiling prefetch behavior")
    print(f"{'='*60}")

    if not hasattr(loader, "token_buffer"):
        print("Loader doesn't have token_buffer (not HFStreamingDataLoader)")
        return {}

    # Let buffer fill
    print("Waiting for buffer to fill...", end=" ", flush=True)
    time.sleep(2)
    print("done")

    initial_buffer = len(loader.token_buffer)
    print(f"Initial buffer size: {initial_buffer:,} tokens")

    # Drain rapidly
    batch_times = []
    buffer_sizes = []

    print(f"Rapid drain test ({num_batches} batches)...", end=" ", flush=True)
    for i in range(num_batches):
        start = time.perf_counter()
        batch = loader.get_batch()
        end = time.perf_counter()

        batch_times.append(end - start)
        buffer_sizes.append(len(loader.token_buffer))

        if (i + 1) % 10 == 0:
            print(f"{i+1}", end=" ", flush=True)

    print("done")

    return {
        "initial_buffer": initial_buffer,
        "final_buffer": buffer_sizes[-1] if buffer_sizes else 0,
        "min_buffer_during_drain": min(buffer_sizes) if buffer_sizes else 0,
        "avg_batch_time_ms": statistics.mean(batch_times) * 1000,
        "max_batch_time_ms": max(batch_times) * 1000,
        "buffer_trajectory": buffer_sizes[:10] + ["..."] + buffer_sizes[-5:] if len(buffer_sizes) > 15 else buffer_sizes,
    }


def main():
    print("=" * 70)
    print("HYDRA Training Dataflow Profiler")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("CUDA not available, exiting")
        return

    device = "cuda"
    batch_size = 32
    seq_len = 512
    hidden_dim = 256  # Smaller model for faster profiling
    num_layers = 4
    vocab_size = 50257
    num_steps = 100

    print(f"\nConfiguration:")
    print(f"  batch_size: {batch_size}")
    print(f"  seq_len: {seq_len}")
    print(f"  tokens_per_batch: {batch_size * seq_len:,}")
    print(f"  model: {num_layers} layers, {hidden_dim} hidden")

    # Create model
    print("\nCreating model...", end=" ", flush=True)
    model = create_simple_model(vocab_size, hidden_dim, num_layers)
    print("done")

    # Profile with synthetic data
    synthetic_loader = SyntheticDataLoader(
        batch_size=batch_size,
        seq_len=seq_len,
        vocab_size=vocab_size,
        device="cpu",
    )
    synthetic_stats = profile_training_loop(
        synthetic_loader,
        model,
        device,
        num_steps=num_steps,
        label="SyntheticDataLoader Training",
    )

    # Profile with real data
    try:
        real_loader = create_universal_loader(
            dataset="finefineweb-local",
            batch_size=batch_size,
            seq_len=seq_len,
            vocab_size=vocab_size,
            device="cpu",
            num_workers=4,
            prefetch_factor=2,
        )

        # First check prefetch behavior
        prefetch_stats = profile_prefetch_behavior(real_loader, num_batches=50)

        # Re-create loader for clean training test
        real_loader.close()
        real_loader = create_universal_loader(
            dataset="finefineweb-local",
            batch_size=batch_size,
            seq_len=seq_len,
            vocab_size=vocab_size,
            device="cpu",
            num_workers=4,
            prefetch_factor=2,
        )

        real_stats = profile_training_loop(
            real_loader,
            model,
            device,
            num_steps=num_steps,
            label="finefineweb-local Training",
        )
        real_loader.close()
    except Exception as e:
        print(f"\nFailed to load finefineweb-local: {e}")
        real_stats = None
        prefetch_stats = {}

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

    print(f"{'Data % of step':<35} {synthetic_stats['data_pct_of_step']:>14.1f}%", end="")
    if real_stats:
        print(f" {real_stats['data_pct_of_step']:>14.1f}%")
    else:
        print(f" {'N/A':>15}")

    print(f"{'P99 data time (ms)':<35} {synthetic_stats['p99_data_time_ms']:>15.2f}", end="")
    if real_stats:
        print(f" {real_stats['p99_data_time_ms']:>15.2f}")
    else:
        print(f" {'N/A':>15}")

    if real_stats:
        print(f"\nTime breakdown for real data:")
        print(f"  Data loading: {real_stats['avg_data_time_ms']:.2f}ms ({real_stats['data_pct_of_step']:.1f}%)")
        print(f"  Forward: {real_stats['avg_forward_time_ms']:.2f}ms")
        print(f"  Backward: {real_stats['avg_backward_time_ms']:.2f}ms")
        print(f"  Total: {real_stats['avg_step_time_ms']:.2f}ms")

        if real_stats.get('avg_buffer_size'):
            print(f"\nBuffer statistics:")
            print(f"  Avg size: {real_stats['avg_buffer_size']:,.0f} tokens")
            print(f"  Min size: {real_stats['min_buffer_size']:,.0f} tokens")
            print(f"  Max size: {real_stats['max_buffer_size']:,.0f} tokens")

    if prefetch_stats:
        print(f"\nPrefetch behavior:")
        print(f"  Initial buffer: {prefetch_stats.get('initial_buffer', 'N/A'):,} tokens")
        print(f"  Min during drain: {prefetch_stats.get('min_buffer_during_drain', 'N/A'):,} tokens")
        print(f"  Max batch time: {prefetch_stats.get('max_batch_time_ms', 'N/A'):.2f}ms")


if __name__ == "__main__":
    main()
