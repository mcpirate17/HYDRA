#!/usr/bin/env python3
"""Profile data loading pipeline to identify throughput bottlenecks.

Compares synthetic vs real data loading to understand the 76K vs 31K tok/s gap.
"""

import time
import statistics
from typing import Dict, List, Any
import torch

# Must be before hydra imports
import sys
sys.path.insert(0, "/home/tim/Projects/LLM/HYDRA")

from hydra.data.universal_data_loader import (
    create_universal_loader,
    SyntheticDataLoader,
    HFStreamingDataLoader,
    get_tokenizer,
)


def profile_data_loader(
    loader,
    num_batches: int = 100,
    warmup_batches: int = 10,
    label: str = "DataLoader",
) -> Dict[str, float]:
    """Profile a data loader and return timing statistics."""

    # Warmup
    print(f"\n{'='*60}")
    print(f"Profiling: {label}")
    print(f"{'='*60}")
    print(f"Warming up ({warmup_batches} batches)...", end=" ", flush=True)

    for _ in range(warmup_batches):
        batch = loader.get_batch()
    print("done")

    # Profile
    batch_times = []
    tokens_per_batch = []

    print(f"Profiling ({num_batches} batches)...", end=" ", flush=True)

    for i in range(num_batches):
        start = time.perf_counter()
        batch = loader.get_batch()
        end = time.perf_counter()

        batch_times.append(end - start)
        tokens_per_batch.append(batch["input_ids"].numel())

        if (i + 1) % 25 == 0:
            print(f"{i+1}", end=" ", flush=True)

    print("done")

    # Calculate statistics
    total_tokens = sum(tokens_per_batch)
    total_time = sum(batch_times)

    stats = {
        "label": label,
        "num_batches": num_batches,
        "total_tokens": total_tokens,
        "total_time_s": total_time,
        "throughput_tok_per_s": total_tokens / total_time,
        "avg_batch_time_ms": statistics.mean(batch_times) * 1000,
        "std_batch_time_ms": statistics.stdev(batch_times) * 1000 if len(batch_times) > 1 else 0,
        "min_batch_time_ms": min(batch_times) * 1000,
        "max_batch_time_ms": max(batch_times) * 1000,
        "p50_batch_time_ms": statistics.median(batch_times) * 1000,
        "p99_batch_time_ms": sorted(batch_times)[int(0.99 * len(batch_times))] * 1000,
    }

    return stats


def profile_tokenizer(
    tokenizer_name: str = "gpt2",
    num_texts: int = 1000,
    text_length: int = 500,
) -> Dict[str, float]:
    """Profile tokenizer speed."""
    print(f"\n{'='*60}")
    print(f"Profiling tokenizer: {tokenizer_name}")
    print(f"{'='*60}")

    tokenizer = get_tokenizer(tokenizer_name)

    # Generate synthetic texts
    import random
    words = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
             "machine", "learning", "neural", "network", "transformer", "attention",
             "language", "model", "training", "optimization", "gradient", "descent"]

    texts = []
    for _ in range(num_texts):
        text = " ".join(random.choices(words, k=text_length // 5))
        texts.append(text)

    # Profile single tokenization
    print("Profiling single-text tokenization...", end=" ", flush=True)
    start = time.perf_counter()
    for text in texts[:100]:
        tokenizer(text, add_special_tokens=False)
    single_time = time.perf_counter() - start
    print(f"{single_time*1000:.1f}ms for 100 texts")

    # Profile batched tokenization
    print("Profiling batched tokenization...", end=" ", flush=True)
    start = time.perf_counter()
    tokenizer(texts, add_special_tokens=False)
    batched_time = time.perf_counter() - start
    print(f"{batched_time*1000:.1f}ms for {num_texts} texts")

    return {
        "tokenizer": tokenizer_name,
        "num_texts": num_texts,
        "single_time_ms_per_100": single_time * 1000,
        "batched_time_ms": batched_time * 1000,
        "speedup": (single_time * 10) / batched_time,  # Extrapolate single to 1000
        "texts_per_second_batched": num_texts / batched_time,
    }


def profile_buffer_operations(batch_size: int = 32, seq_len: int = 512) -> Dict[str, float]:
    """Profile token buffer operations."""
    from collections import deque

    print(f"\n{'='*60}")
    print(f"Profiling buffer operations")
    print(f"{'='*60}")

    buffer_size = batch_size * (seq_len + 1) * 20
    token_buffer = deque(maxlen=buffer_size)

    # Profile extending
    tokens = list(range(batch_size * seq_len * 5))

    print("Profiling buffer.extend()...", end=" ", flush=True)
    start = time.perf_counter()
    for _ in range(100):
        token_buffer.extend(tokens)
    extend_time = time.perf_counter() - start
    print(f"{extend_time*1000:.1f}ms for 100 extends")

    # Profile popleft
    needed = batch_size * (seq_len + 1)
    print("Profiling buffer.popleft() loop...", end=" ", flush=True)

    # Refill
    token_buffer.extend(tokens * 10)

    start = time.perf_counter()
    for _ in range(100):
        for _ in range(needed):
            token_buffer.popleft()
        token_buffer.extend(tokens)  # Refill
    popleft_time = time.perf_counter() - start
    print(f"{popleft_time*1000:.1f}ms for 100 batch extractions")

    # Profile list comprehension version
    print("Profiling list comprehension extraction...", end=" ", flush=True)
    token_buffer.extend(tokens * 10)

    start = time.perf_counter()
    for _ in range(100):
        batch_tokens = [token_buffer.popleft() for _ in range(needed)]
        token_buffer.extend(tokens)
    listcomp_time = time.perf_counter() - start
    print(f"{listcomp_time*1000:.1f}ms for 100 extractions")

    return {
        "extend_100_ms": extend_time * 1000,
        "popleft_loop_100_ms": popleft_time * 1000,
        "listcomp_100_ms": listcomp_time * 1000,
        "tokens_per_extend": len(tokens),
        "tokens_per_extraction": needed,
    }


def profile_tensor_creation(batch_size: int = 32, seq_len: int = 512) -> Dict[str, float]:
    """Profile tensor creation from token lists."""
    print(f"\n{'='*60}")
    print(f"Profiling tensor creation")
    print(f"{'='*60}")

    tokens = list(range(batch_size * (seq_len + 1)))

    # Profile basic tensor creation
    print("Profiling torch.tensor()...", end=" ", flush=True)
    start = time.perf_counter()
    for _ in range(100):
        t = torch.tensor(tokens, dtype=torch.long)
    basic_time = time.perf_counter() - start
    print(f"{basic_time*1000:.1f}ms for 100 creations")

    # Profile with pin_memory
    print("Profiling torch.tensor() + pin_memory...", end=" ", flush=True)
    start = time.perf_counter()
    for _ in range(100):
        t = torch.tensor(tokens, dtype=torch.long, pin_memory=True)
    pinned_time = time.perf_counter() - start
    print(f"{pinned_time*1000:.1f}ms for 100 creations")

    # Profile view operation
    print("Profiling .view()...", end=" ", flush=True)
    t = torch.tensor(tokens, dtype=torch.long)
    start = time.perf_counter()
    for _ in range(10000):
        v = t.view(batch_size, seq_len + 1)
    view_time = time.perf_counter() - start
    print(f"{view_time*1000:.1f}ms for 10k views")

    # Profile H2D transfer
    if torch.cuda.is_available():
        print("Profiling H2D transfer...", end=" ", flush=True)
        t = torch.tensor(tokens, dtype=torch.long, pin_memory=True)
        t = t.view(batch_size, seq_len + 1)

        # Warmup
        for _ in range(10):
            t.cuda(non_blocking=True)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(100):
            t_gpu = t.cuda(non_blocking=True)
        torch.cuda.synchronize()
        h2d_time = time.perf_counter() - start
        print(f"{h2d_time*1000:.1f}ms for 100 transfers")
    else:
        h2d_time = 0

    return {
        "tensor_creation_100_ms": basic_time * 1000,
        "pinned_creation_100_ms": pinned_time * 1000,
        "view_10k_ms": view_time * 1000,
        "h2d_100_ms": h2d_time * 1000 if torch.cuda.is_available() else None,
    }


def main():
    print("=" * 70)
    print("HYDRA Data Loading Pipeline Profiler")
    print("=" * 70)

    batch_size = 32
    seq_len = 512
    num_batches = 100

    print(f"\nConfiguration:")
    print(f"  batch_size: {batch_size}")
    print(f"  seq_len: {seq_len}")
    print(f"  tokens_per_batch: {batch_size * seq_len:,}")

    # 1. Profile synthetic data loader
    synthetic_loader = SyntheticDataLoader(
        batch_size=batch_size,
        seq_len=seq_len,
        vocab_size=50257,
        device="cpu",
    )
    synthetic_stats = profile_data_loader(
        synthetic_loader,
        num_batches=num_batches,
        label="SyntheticDataLoader",
    )

    # 2. Profile real data loader (finefineweb-local)
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
        real_stats = profile_data_loader(
            real_loader,
            num_batches=num_batches,
            label="finefineweb-local (HFStreamingDataLoader)",
        )
        if hasattr(real_loader, "close"):
            real_loader.close()
    except Exception as e:
        print(f"\nFailed to load finefineweb-local: {e}")
        real_stats = None

    # 3. Profile tokenizer
    tokenizer_stats = profile_tokenizer()

    # 4. Profile buffer operations
    buffer_stats = profile_buffer_operations(batch_size, seq_len)

    # 5. Profile tensor creation
    tensor_stats = profile_tensor_creation(batch_size, seq_len)

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

    print(f"{'Avg batch time (ms)':<35} {synthetic_stats['avg_batch_time_ms']:>15.2f}", end="")
    if real_stats:
        print(f" {real_stats['avg_batch_time_ms']:>15.2f}")
    else:
        print(f" {'N/A':>15}")

    print(f"{'P99 batch time (ms)':<35} {synthetic_stats['p99_batch_time_ms']:>15.2f}", end="")
    if real_stats:
        print(f" {real_stats['p99_batch_time_ms']:>15.2f}")
    else:
        print(f" {'N/A':>15}")

    if real_stats:
        speedup = synthetic_stats['throughput_tok_per_s'] / real_stats['throughput_tok_per_s']
        print(f"\n{'Synthetic speedup over real':<35} {speedup:>15.1f}x")

        # Estimate where time is spent
        print(f"\n{'Time breakdown estimate':}")

        synthetic_batch_ms = synthetic_stats['avg_batch_time_ms']
        real_batch_ms = real_stats['avg_batch_time_ms']
        overhead_ms = real_batch_ms - synthetic_batch_ms

        print(f"  Base (tensor creation): {synthetic_batch_ms:.2f}ms")
        print(f"  Data loading overhead: {overhead_ms:.2f}ms")
        print(f"  Total per batch: {real_batch_ms:.2f}ms")

    print(f"\nTokenizer stats:")
    print(f"  Batched throughput: {tokenizer_stats['texts_per_second_batched']:,.0f} texts/s")
    print(f"  Batch speedup: {tokenizer_stats['speedup']:.1f}x over sequential")

    print(f"\nBuffer operations (100 iterations):")
    print(f"  extend(): {buffer_stats['extend_100_ms']:.1f}ms")
    print(f"  popleft() loop: {buffer_stats['popleft_loop_100_ms']:.1f}ms")
    print(f"  list comprehension: {buffer_stats['listcomp_100_ms']:.1f}ms")

    print(f"\nTensor creation (100 iterations):")
    print(f"  torch.tensor(): {tensor_stats['tensor_creation_100_ms']:.1f}ms")
    print(f"  with pin_memory: {tensor_stats['pinned_creation_100_ms']:.1f}ms")
    if tensor_stats.get('h2d_100_ms'):
        print(f"  H2D transfer: {tensor_stats['h2d_100_ms']:.1f}ms")


if __name__ == "__main__":
    main()
