#!/usr/bin/env python3
"""Profile Python-level overhead in HYDRA training.

Usage:
    source /home/tim/venvs/llm/bin/activate && python diagnostics/profile_python_overhead.py

This script profiles:
1. Data loading pipeline (CPU usage, tokenization)
2. Python hot paths in training loop
3. Memory/GC overhead
4. Caching opportunities (masks, positions, allocations)
"""

import gc
import os
import sys
import time
import tracemalloc
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.profiler import profile, record_function, ProfilerActivity

# Disable GC during profiling to measure its impact later
gc.disable()


def profile_data_loading():
    """Profile data loading pipeline."""
    print("\n" + "=" * 70)
    print("1. DATA LOADING PIPELINE ANALYSIS")
    print("=" * 70)

    from hydra.data.universal_data_loader import (
        create_universal_loader,
        LocalDataLoader,
        SyntheticDataLoader,
        AVAILABLE_DATASETS,
    )

    batch_size = 4
    seq_len = 1024
    n_batches = 50

    results = {}

    # Test synthetic data (baseline - no I/O)
    print("\n--- Synthetic Data (baseline) ---")
    loader = SyntheticDataLoader(batch_size=batch_size, seq_len=seq_len, vocab_size=50257)

    gc.collect()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.perf_counter()
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        _ = batch["input_ids"]
    elapsed = time.perf_counter() - start
    results["synthetic"] = elapsed / n_batches * 1000
    print(f"  Time per batch: {results['synthetic']:.3f} ms")

    # Test local pre-tokenized data
    print("\n--- Local Pre-tokenized Data ---")
    # Find a local dataset
    local_dataset = None
    for name, cfg in AVAILABLE_DATASETS.items():
        if cfg.get("local") and os.path.exists(str(cfg.get("path", ""))):
            local_dataset = name
            print(f"  Using: {name}")
            break

    if local_dataset:
        try:
            loader = create_universal_loader(
                dataset=local_dataset,
                batch_size=batch_size,
                seq_len=seq_len,
            )
            gc.collect()
            start = time.perf_counter()
            for i, batch in enumerate(loader):
                if i >= n_batches:
                    break
                _ = batch["input_ids"]
            elapsed = time.perf_counter() - start
            results["local_pretokenized"] = elapsed / n_batches * 1000
            print(f"  Time per batch: {results['local_pretokenized']:.3f} ms")
            print(f"  Overhead vs synthetic: {results['local_pretokenized'] / results['synthetic']:.2f}x")
        except Exception as e:
            print(f"  Skipped: {e}")
    else:
        print("  No local datasets found")

    # Test HF streaming with tokenization (if available)
    print("\n--- HF Streaming (with tokenization) ---")
    try:
        loader = create_universal_loader(
            dataset="wikitext2",
            batch_size=batch_size,
            seq_len=seq_len,
        )
        gc.collect()
        start = time.perf_counter()
        for i, batch in enumerate(loader):
            if i >= min(n_batches, 20):  # Fewer batches due to network
                break
            _ = batch["input_ids"]
        elapsed = time.perf_counter() - start
        results["hf_streaming"] = elapsed / min(n_batches, 20) * 1000
        print(f"  Time per batch: {results['hf_streaming']:.3f} ms")
        if "synthetic" in results:
            print(f"  Overhead vs synthetic: {results['hf_streaming'] / results['synthetic']:.2f}x")
    except Exception as e:
        print(f"  Skipped: {e}")

    return results


def profile_training_step():
    """Profile Python overhead in training step."""
    print("\n" + "=" * 70)
    print("2. TRAINING STEP PYTHON OVERHEAD")
    print("=" * 70)

    from hydra.model.framework import HydraModel
    from hydra.kernels import fused_chunked_cross_entropy

    # Create small model for profiling
    dim = 768
    model = HydraModel(
        dim=dim,
        n_blocks=4,
        n_heads=12,
        n_kv_heads=4,
        n_recursions=2,
        vocab_size=50257,
        max_seq_len=2048,
        mod_enabled=True,
        mor_enabled=True,
    ).cuda().bfloat16()

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    batch_size, seq_len = 4, 512

    # Warmup
    for _ in range(3):
        x = torch.randint(0, 50257, (batch_size, seq_len), device="cuda")
        y = torch.randint(0, 50257, (batch_size, seq_len), device="cuda")
        hidden, aux = model.forward_hidden_with_losses(x)
        loss = fused_chunked_cross_entropy(hidden, model.output.weight, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    torch.cuda.synchronize()

    # Profile with torch.profiler
    print("\n--- Detailed CPU/GPU breakdown ---")

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for step in range(10):
            with record_function("full_step"):
                with record_function("data_prep"):
                    x = torch.randint(0, 50257, (batch_size, seq_len), device="cuda")
                    y = torch.randint(0, 50257, (batch_size, seq_len), device="cuda")

                with record_function("forward"):
                    hidden, aux = model.forward_hidden_with_losses(x)

                with record_function("loss_compute"):
                    loss = fused_chunked_cross_entropy(hidden, model.output.weight, y)

                with record_function("backward"):
                    loss.backward()

                with record_function("optimizer_step"):
                    optimizer.step()
                    optimizer.zero_grad()

    # Print CPU time breakdown
    print("\nTop 20 CPU operations:")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))

    # Identify Python overhead
    print("\n--- Python-only operations (potential optimization targets) ---")
    events = prof.key_averages()
    python_ops = []
    for e in events:
        # Look for operations that are pure Python
        if "aten::" not in e.key and "cuda" not in e.key.lower():
            if e.cpu_time_total > 100:  # > 0.1ms
                python_ops.append((e.key, e.cpu_time_total / 1000))  # Convert to ms

    python_ops.sort(key=lambda x: -x[1])
    for name, time_ms in python_ops[:15]:
        print(f"  {name}: {time_ms:.2f} ms")


def profile_memory_allocations():
    """Profile memory allocation patterns."""
    print("\n" + "=" * 70)
    print("3. MEMORY ALLOCATION ANALYSIS")
    print("=" * 70)

    from hydra.model.framework import HydraModel
    from hydra.kernels import fused_chunked_cross_entropy

    dim = 768
    model = HydraModel(
        dim=dim,
        n_blocks=4,
        n_heads=12,
        n_kv_heads=4,
        n_recursions=2,
        vocab_size=50257,
        max_seq_len=2048,
        mod_enabled=True,
        mor_enabled=True,
    ).cuda().bfloat16()

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    batch_size, seq_len = 4, 512

    # Warmup
    for _ in range(3):
        x = torch.randint(0, 50257, (batch_size, seq_len), device="cuda")
        y = torch.randint(0, 50257, (batch_size, seq_len), device="cuda")
        hidden, aux = model.forward_hidden_with_losses(x)
        loss = fused_chunked_cross_entropy(hidden, model.output.weight, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    torch.cuda.synchronize()

    # Track Python memory allocations
    tracemalloc.start()

    for step in range(10):
        x = torch.randint(0, 50257, (batch_size, seq_len), device="cuda")
        y = torch.randint(0, 50257, (batch_size, seq_len), device="cuda")
        hidden, aux = model.forward_hidden_with_losses(x)
        loss = fused_chunked_cross_entropy(hidden, model.output.weight, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    snapshot = tracemalloc.take_snapshot()
    tracemalloc.stop()

    print("\nTop Python memory allocations (by size):")
    top_stats = snapshot.statistics("lineno")
    for stat in top_stats[:15]:
        print(f"  {stat}")

    # CUDA memory analysis
    print("\n--- CUDA Memory Fragmentation ---")
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    fragmentation = (reserved - allocated) / reserved * 100 if reserved > 0 else 0
    print(f"  Allocated: {allocated:.2f} GB")
    print(f"  Reserved:  {reserved:.2f} GB")
    print(f"  Fragmentation: {fragmentation:.1f}%")


def profile_gc_impact():
    """Profile GC pause impact."""
    print("\n" + "=" * 70)
    print("4. GARBAGE COLLECTION IMPACT")
    print("=" * 70)

    from hydra.model.framework import HydraModel
    from hydra.kernels import fused_chunked_cross_entropy

    dim = 768
    model = HydraModel(
        dim=dim,
        n_blocks=4,
        n_heads=12,
        n_kv_heads=4,
        n_recursions=2,
        vocab_size=50257,
        max_seq_len=2048,
        mod_enabled=True,
        mor_enabled=True,
    ).cuda().bfloat16()

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    batch_size, seq_len = 4, 512
    n_steps = 50

    # Warmup
    for _ in range(5):
        x = torch.randint(0, 50257, (batch_size, seq_len), device="cuda")
        y = torch.randint(0, 50257, (batch_size, seq_len), device="cuda")
        hidden, aux = model.forward_hidden_with_losses(x)
        loss = fused_chunked_cross_entropy(hidden, model.output.weight, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    torch.cuda.synchronize()

    # Test WITHOUT GC
    gc.disable()
    gc.collect()  # Clean slate
    torch.cuda.synchronize()

    times_no_gc = []
    for step in range(n_steps):
        start = time.perf_counter()
        x = torch.randint(0, 50257, (batch_size, seq_len), device="cuda")
        y = torch.randint(0, 50257, (batch_size, seq_len), device="cuda")
        hidden, aux = model.forward_hidden_with_losses(x)
        loss = fused_chunked_cross_entropy(hidden, model.output.weight, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.synchronize()
        times_no_gc.append(time.perf_counter() - start)

    # Test WITH GC
    gc.enable()
    gc.collect()
    torch.cuda.synchronize()

    times_with_gc = []
    for step in range(n_steps):
        start = time.perf_counter()
        x = torch.randint(0, 50257, (batch_size, seq_len), device="cuda")
        y = torch.randint(0, 50257, (batch_size, seq_len), device="cuda")
        hidden, aux = model.forward_hidden_with_losses(x)
        loss = fused_chunked_cross_entropy(hidden, model.output.weight, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.synchronize()
        times_with_gc.append(time.perf_counter() - start)

    avg_no_gc = sum(times_no_gc) / len(times_no_gc) * 1000
    avg_with_gc = sum(times_with_gc) / len(times_with_gc) * 1000
    max_no_gc = max(times_no_gc) * 1000
    max_with_gc = max(times_with_gc) * 1000

    print(f"\n  Without GC: {avg_no_gc:.2f} ms avg, {max_no_gc:.2f} ms max")
    print(f"  With GC:    {avg_with_gc:.2f} ms avg, {max_with_gc:.2f} ms max")
    print(f"  GC overhead: {(avg_with_gc - avg_no_gc) / avg_no_gc * 100:.1f}%")
    print(f"  Max spike:   {max_with_gc - max_no_gc:.2f} ms")


def check_caching_opportunities():
    """Check for repeated computations that could be cached."""
    print("\n" + "=" * 70)
    print("5. CACHING OPPORTUNITIES ANALYSIS")
    print("=" * 70)

    from hydra.model.framework import HydraModel
    from hydra.layers import precompute_rope_freqs

    dim = 768
    model = HydraModel(
        dim=dim,
        n_blocks=4,
        n_heads=12,
        n_kv_heads=4,
        n_recursions=2,
        vocab_size=50257,
        max_seq_len=2048,
        mod_enabled=True,
        mor_enabled=True,
    ).cuda().bfloat16()

    print("\n--- RoPE Frequencies ---")
    # Check if RoPE is precomputed
    if hasattr(model, "rope_freqs") or hasattr(model, "freqs_cis"):
        print("  Status: CACHED (precomputed at init)")
    else:
        print("  Status: CHECK - may be recomputed per forward")
        # Look for rope computation
        for name, module in model.named_modules():
            if "rope" in name.lower() or "rotary" in name.lower():
                print(f"    Found: {name}")

    print("\n--- Attention Masks ---")
    # Check if causal mask is cached
    causal_cached = False
    for name, buf in model.named_buffers():
        if "mask" in name.lower() or "causal" in name.lower():
            print(f"  Buffer: {name} shape={buf.shape if hasattr(buf, 'shape') else 'N/A'}")
            causal_cached = True

    if not causal_cached:
        print("  Status: No cached masks found - may recompute per batch")
        print("  OPTIMIZATION: Could cache causal mask as buffer")

    print("\n--- Position IDs ---")
    # Check if position IDs are cached
    pos_cached = False
    for name, buf in model.named_buffers():
        if "position" in name.lower() or "pos_id" in name.lower():
            print(f"  Buffer: {name}")
            pos_cached = True

    if not pos_cached:
        print("  Status: No cached position IDs - generated per batch")
        print("  NOTE: This is fine if using torch.arange which is fast")

    print("\n--- Repeated Tensor Allocations ---")
    # Check forward for repeated allocations
    x = torch.randint(0, 50257, (4, 512), device="cuda")

    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated()

    with torch.no_grad():
        for _ in range(5):
            _ = model(x)

    mem_after = torch.cuda.memory_allocated()
    peak = torch.cuda.max_memory_allocated()

    print(f"  Memory before: {mem_before / 1e6:.1f} MB")
    print(f"  Memory after:  {mem_after / 1e6:.1f} MB")
    print(f"  Peak memory:   {peak / 1e6:.1f} MB")
    print(f"  Temp allocations: {(peak - mem_before) / 1e6:.1f} MB")


def check_compilation_candidates():
    """Identify code that could benefit from compilation."""
    print("\n" + "=" * 70)
    print("6. COMPILATION CANDIDATES")
    print("=" * 70)

    print("\n--- Already Compiled/Optimized ---")
    print("  - Tokenizer: Using HuggingFace fast tokenizers (Rust)")
    print("  - Triton kernels: RoPE, RMSNorm, SwiGLU, QK-Norm")
    print("  - Liger CE: Fused cross-entropy with Triton")
    print("  - CUTLASS: Matrix multiplications")

    print("\n--- Potential torch.compile Candidates ---")

    # Check if model is compiled
    from hydra.model.framework import HydraModel

    model = HydraModel(
        dim=768,
        n_blocks=4,
        n_heads=12,
        n_kv_heads=4,
        n_recursions=2,
        vocab_size=50257,
        max_seq_len=2048,
    ).cuda()

    # Check if forward is compilable
    try:
        compiled = torch.compile(model, mode="reduce-overhead")
        x = torch.randint(0, 50257, (2, 64), device="cuda")
        with torch.no_grad():
            _ = compiled(x)
        print("  - Model forward: COMPILABLE with torch.compile")
    except Exception as e:
        print(f"  - Model forward: NOT COMPILABLE - {type(e).__name__}")

    print("\n--- Python Loops in Hot Path ---")
    # Check training loop for Python loops
    import inspect
    from hydra.training.loop import compute_microbatch_loss

    source = inspect.getsource(compute_microbatch_loss)
    if "for " in source or "while " in source:
        print("  - compute_microbatch_loss: Contains Python loops")
        # Find the loops
        for i, line in enumerate(source.split("\n")):
            if "for " in line or "while " in line:
                print(f"      Line {i}: {line.strip()[:60]}...")
    else:
        print("  - compute_microbatch_loss: No Python loops (good!)")


def main():
    print("=" * 70)
    print("HYDRA PYTHON-LEVEL OPTIMIZATION AUDIT")
    print("=" * 70)

    try:
        profile_data_loading()
    except Exception as e:
        print(f"Data loading profile failed: {e}")

    try:
        profile_training_step()
    except Exception as e:
        print(f"Training step profile failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        profile_memory_allocations()
    except Exception as e:
        print(f"Memory allocation profile failed: {e}")

    try:
        profile_gc_impact()
    except Exception as e:
        print(f"GC impact profile failed: {e}")

    try:
        check_caching_opportunities()
    except Exception as e:
        print(f"Caching check failed: {e}")

    try:
        check_compilation_candidates()
    except Exception as e:
        print(f"Compilation check failed: {e}")

    print("\n" + "=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
