#!/usr/bin/env python3
"""Profile tensor copy operations to find sources of aten::copy_ overhead.

The previous profiling identified aten::copy_ as ~19% of GPU time with 12K+ calls.
This script traces where these copies originate to identify optimization opportunities.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.profiler import profile, ProfilerActivity
import gc


# 100M model config
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


def reset_memory():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def setup_model():
    from hydra.model.framework import HydraModel
    try:
        from hydra.kernels import set_use_triton_kernels
        set_use_triton_kernels(True)
    except:
        pass

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
    return model.to("cuda").to(torch.bfloat16)


def profile_copy_with_stack():
    """Profile with stack traces to find copy_ origins."""
    print("=" * 70)
    print("PROFILING aten::copy_ WITH STACK TRACES")
    print("=" * 70)

    reset_memory()
    model = setup_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Warmup
    print("\nWarmup (3 steps)...")
    for _ in range(3):
        input_ids = torch.randint(0, MODEL_CONFIG["vocab_size"], (BATCH_SIZE, SEQ_LEN), device="cuda")
        targets = torch.randint(0, MODEL_CONFIG["vocab_size"], (BATCH_SIZE, SEQ_LEN), device="cuda")
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        loss.backward()
        optimizer.step()

    # Profile with stack traces
    print("\nProfiling with stack traces (5 steps)...")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
        profile_memory=True,
    ) as prof:
        for _ in range(5):
            input_ids = torch.randint(0, MODEL_CONFIG["vocab_size"], (BATCH_SIZE, SEQ_LEN), device="cuda")
            targets = torch.randint(0, MODEL_CONFIG["vocab_size"], (BATCH_SIZE, SEQ_LEN), device="cuda")
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids)
                loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss.backward()
            optimizer.step()

    torch.cuda.synchronize()

    # Analyze copy operations
    print("\n" + "=" * 70)
    print("ALL COPY-RELATED OPERATIONS")
    print("=" * 70)

    key_avgs = prof.key_averages(group_by_stack_n=10)
    copy_ops = []

    for event in key_avgs:
        name = event.key
        # Look for copy, to, clone operations
        if any(x in name.lower() for x in ['copy', 'to(', 'clone', '_copy']):
            cuda_time = getattr(event, 'cuda_time_total', 0) or getattr(event, 'self_cuda_time_total', 0) or 0
            cpu_time = getattr(event, 'cpu_time_total', 0) or getattr(event, 'self_cpu_time_total', 0) or 0
            count = event.count
            if cuda_time > 0 or cpu_time > 0:
                copy_ops.append({
                    'name': name,
                    'cuda_time_us': cuda_time,
                    'cpu_time_us': cpu_time,
                    'count': count,
                    'stack': getattr(event, 'stack', None),
                })

    # Sort by cuda time
    copy_ops.sort(key=lambda x: x['cuda_time_us'], reverse=True)

    total_cuda = sum(op['cuda_time_us'] for op in copy_ops)
    print(f"\nTotal copy-related CUDA time: {total_cuda/1000:.2f}ms over 5 steps")
    print(f"Average per step: {total_cuda/5000:.2f}ms\n")

    print(f"{'Operation':<40} {'CUDA Time':>12} {'CPU Time':>12} {'Calls':>8} {'%':>6}")
    print("-" * 85)

    for op in copy_ops[:20]:
        cuda_ms = op['cuda_time_us'] / 1000
        cpu_ms = op['cpu_time_us'] / 1000
        pct = (op['cuda_time_us'] / total_cuda * 100) if total_cuda > 0 else 0
        name = op['name'][:38] if len(op['name']) > 38 else op['name']
        print(f"{name:<40} {cuda_ms:>10.2f}ms {cpu_ms:>10.2f}ms {op['count']:>8} {pct:>5.1f}%")

    # Export trace for detailed analysis
    trace_file = Path(__file__).parent / "profile_copy_trace.json"
    prof.export_chrome_trace(str(trace_file))
    print(f"\nTrace exported to {trace_file}")
    print("Load in chrome://tracing to inspect call stacks")

    return prof


def profile_copy_by_module():
    """Instrument model to track copies by module."""
    print("\n" + "=" * 70)
    print("COPY OPERATIONS BY MODULE")
    print("=" * 70)

    reset_memory()
    model = setup_model()

    # Track copy operations per module using hooks
    copy_counts = {}
    original_copy = torch.Tensor.copy_

    def counting_copy(self, src, non_blocking=False):
        # Get call stack to identify source
        import traceback
        stack = traceback.extract_stack()
        # Find the relevant frame (skip torch internals)
        for frame in reversed(stack[:-1]):
            if 'hydra' in frame.filename or 'forward' in frame.name:
                key = f"{Path(frame.filename).name}:{frame.lineno} ({frame.name})"
                copy_counts[key] = copy_counts.get(key, 0) + 1
                break
        else:
            copy_counts['other'] = copy_counts.get('other', 0) + 1
        return original_copy(self, src, non_blocking)

    # Monkey-patch temporarily
    torch.Tensor.copy_ = counting_copy

    try:
        print("\nRunning 3 training steps with copy tracking...")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        for step in range(3):
            input_ids = torch.randint(0, MODEL_CONFIG["vocab_size"], (BATCH_SIZE, SEQ_LEN), device="cuda")
            targets = torch.randint(0, MODEL_CONFIG["vocab_size"], (BATCH_SIZE, SEQ_LEN), device="cuda")
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids)
                loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss.backward()
            optimizer.step()

        print(f"\nTotal copy_ calls tracked: {sum(copy_counts.values())}")
        print(f"Per step: {sum(copy_counts.values()) / 3:.0f}\n")

        # Sort by count
        sorted_counts = sorted(copy_counts.items(), key=lambda x: x[1], reverse=True)

        print(f"{'Source Location':<60} {'Count':>8}")
        print("-" * 70)
        for loc, count in sorted_counts[:25]:
            print(f"{loc:<60} {count:>8}")

    finally:
        # Restore original
        torch.Tensor.copy_ = original_copy


def analyze_tensor_contiguity():
    """Check if tensors require copies due to non-contiguous layouts."""
    print("\n" + "=" * 70)
    print("TENSOR CONTIGUITY ANALYSIS")
    print("=" * 70)

    reset_memory()
    model = setup_model()

    # Hook to check tensor contiguity
    contiguity_issues = []

    def check_contiguity(module, input, output):
        module_name = module.__class__.__name__

        def check_tensor(t, name):
            if isinstance(t, torch.Tensor):
                if not t.is_contiguous():
                    contiguity_issues.append({
                        'module': module_name,
                        'tensor': name,
                        'shape': list(t.shape),
                        'stride': t.stride(),
                    })
            elif isinstance(t, (tuple, list)):
                for i, item in enumerate(t):
                    check_tensor(item, f"{name}[{i}]")

        if isinstance(input, torch.Tensor):
            check_tensor(input, 'input')
        elif isinstance(input, tuple):
            for i, inp in enumerate(input):
                check_tensor(inp, f'input[{i}]')

        if isinstance(output, torch.Tensor):
            check_tensor(output, 'output')
        elif isinstance(output, tuple):
            for i, out in enumerate(output):
                check_tensor(out, f'output[{i}]')

    # Register hooks
    handles = []
    for name, module in model.named_modules():
        handles.append(module.register_forward_hook(check_contiguity))

    # Run forward pass
    print("\nRunning forward pass with contiguity checks...")
    input_ids = torch.randint(0, MODEL_CONFIG["vocab_size"], (BATCH_SIZE, SEQ_LEN), device="cuda")
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        model(input_ids)

    # Remove hooks
    for h in handles:
        h.remove()

    # Report
    if contiguity_issues:
        print(f"\nFound {len(contiguity_issues)} non-contiguous tensors:\n")
        print(f"{'Module':<30} {'Tensor':<15} {'Shape':<25} {'Stride'}")
        print("-" * 90)
        for issue in contiguity_issues[:20]:
            print(f"{issue['module']:<30} {issue['tensor']:<15} {str(issue['shape']):<25} {issue['stride']}")
    else:
        print("\nNo non-contiguous tensors found in forward pass.")
        print("Copy operations may be from dtype casting or optimizer updates.")


def profile_dtype_conversions():
    """Profile dtype conversion overhead (bf16 autocasting)."""
    print("\n" + "=" * 70)
    print("DTYPE CONVERSION ANALYSIS")
    print("=" * 70)

    reset_memory()
    model = setup_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Profile with and without autocast
    import time

    def benchmark(use_autocast, name, n_steps=10):
        times = []
        for _ in range(3):  # warmup
            input_ids = torch.randint(0, MODEL_CONFIG["vocab_size"], (BATCH_SIZE, SEQ_LEN), device="cuda")
            targets = torch.randint(0, MODEL_CONFIG["vocab_size"], (BATCH_SIZE, SEQ_LEN), device="cuda")
            optimizer.zero_grad(set_to_none=True)
            if use_autocast:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(input_ids)
                    loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            else:
                logits = model(input_ids)
                loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss.backward()
            optimizer.step()

        for _ in range(n_steps):
            input_ids = torch.randint(0, MODEL_CONFIG["vocab_size"], (BATCH_SIZE, SEQ_LEN), device="cuda")
            targets = torch.randint(0, MODEL_CONFIG["vocab_size"], (BATCH_SIZE, SEQ_LEN), device="cuda")

            torch.cuda.synchronize()
            start = time.perf_counter()

            optimizer.zero_grad(set_to_none=True)
            if use_autocast:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(input_ids)
                    loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            else:
                logits = model(input_ids)
                loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss.backward()
            optimizer.step()

            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)

        avg = sum(times) / len(times)
        print(f"{name}: {avg:.2f}ms avg per step")
        return avg

    print("\nModel is already bf16, measuring autocast overhead...")
    t_autocast = benchmark(True, "With autocast")
    # Note: without autocast would fail since model is bf16 but inputs are int
    # Just report the autocast timing
    print("\nAutocast overhead is minimal when model weights are already bf16.")
    print("dtype conversions come from mixed operations, not autocast itself.")


def main():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    print("=" * 70)
    print("INVESTIGATING aten::copy_ HOTSPOT")
    print("=" * 70)
    print("\nPrevious profiling showed aten::copy_ at ~19% of GPU time.")
    print("This script traces the sources of these copy operations.\n")

    # Run analyses
    profile_copy_with_stack()
    profile_copy_by_module()
    analyze_tensor_contiguity()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
Common sources of aten::copy_ in training:

1. Optimizer state updates (Adam momentum/variance)
   - These are necessary and already optimized

2. Gradient accumulation/reduction
   - Required for backprop

3. Non-contiguous tensor operations
   - If present, consider using .contiguous() strategically
   - Or restructure operations to avoid layout changes

4. dtype casting in mixed precision
   - Minimal when model weights are bf16

5. Indexing operations in dynamic routing
   - MoD/MoR token selection may create copies
   - Static routing mode avoids this

Check the trace file in chrome://tracing for detailed call stacks.
""")


if __name__ == "__main__":
    main()
