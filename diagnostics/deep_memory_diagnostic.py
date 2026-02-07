#!/usr/bin/env python3
"""
Deep Memory Diagnostic - Identifies exactly where GPU memory is going.

This script profiles memory usage at each stage:
1. Model creation
2. Checkpoint loading
3. Optimizer creation
4. First forward pass
5. First backward pass
6. Optimizer step

Usage:
    source /home/tim/venvs/llm/bin/activate && python diagnostics/deep_memory_diagnostic.py \
        --checkpoint checkpoints/reasoning/reasoning_checkpoint.pt

    # With memory snapshot for visualization
    source /home/tim/venvs/llm/bin/activate && python diagnostics/deep_memory_diagnostic.py \
        --checkpoint checkpoints/reasoning/reasoning_checkpoint.pt --snapshot
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Set CUDA allocator config before importing torch
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class MemoryMeasurement:
    """Single memory measurement."""
    stage: str
    allocated_gb: float
    reserved_gb: float
    peak_allocated_gb: float
    delta_gb: float = 0.0
    details: str = ""

    def __str__(self) -> str:
        delta_str = f" (+{self.delta_gb:.2f}GB)" if self.delta_gb > 0 else ""
        return (
            f"{self.stage:40s}: "
            f"Allocated={self.allocated_gb:6.2f}GB, "
            f"Reserved={self.reserved_gb:6.2f}GB, "
            f"Peak={self.peak_allocated_gb:6.2f}GB"
            f"{delta_str}"
        )


class MemoryProfiler:
    """Profiles GPU memory usage at each stage."""

    def __init__(self, enable_snapshots: bool = False):
        self.measurements: List[MemoryMeasurement] = []
        self.enable_snapshots = enable_snapshots
        self._last_allocated = 0.0

        if enable_snapshots:
            torch.cuda.memory._record_memory_history(max_entries=100000)

    def measure(self, stage: str, details: str = "") -> MemoryMeasurement:
        """Take a memory measurement."""
        torch.cuda.synchronize()

        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        peak = torch.cuda.max_memory_allocated() / 1e9
        delta = allocated - self._last_allocated

        m = MemoryMeasurement(
            stage=stage,
            allocated_gb=allocated,
            reserved_gb=reserved,
            peak_allocated_gb=peak,
            delta_gb=delta,
            details=details,
        )
        self.measurements.append(m)
        self._last_allocated = allocated
        return m

    def reset_peak(self):
        """Reset peak memory stats."""
        torch.cuda.reset_peak_memory_stats()

    def save_snapshot(self, path: str = "memory_snapshot.pickle"):
        """Save memory snapshot for visualization."""
        if self.enable_snapshots:
            torch.cuda.memory._dump_snapshot(path)
            print(f"Saved memory snapshot to {path}")
            print("Visualize at: https://pytorch.org/memory_viz")

    def report(self):
        """Print full memory report."""
        print("\n" + "="*80)
        print("MEMORY PROFILE REPORT")
        print("="*80)

        for m in self.measurements:
            print(m)
            if m.details:
                print(f"  └─ {m.details}")

        print("\n" + "-"*80)
        print("SUMMARY")
        print("-"*80)

        if self.measurements:
            peak = max(m.peak_allocated_gb for m in self.measurements)
            final = self.measurements[-1].allocated_gb
            print(f"Peak allocated: {peak:.2f} GB")
            print(f"Final allocated: {final:.2f} GB")

            # Find biggest jumps
            jumps = sorted(self.measurements, key=lambda x: x.delta_gb, reverse=True)[:5]
            print(f"\nTop 5 memory increases:")
            for m in jumps:
                if m.delta_gb > 0:
                    print(f"  {m.stage}: +{m.delta_gb:.2f} GB")


def analyze_model_memory(model: nn.Module) -> Dict[str, float]:
    """Analyze memory usage of model components."""
    results = {}

    # Total parameters
    total_params = sum(p.numel() for p in model.parameters())
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    results["total_params_millions"] = total_params / 1e6
    results["param_memory_gb"] = param_bytes / 1e9

    # Buffers (RoPE caches, etc.)
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    results["buffer_memory_gb"] = buffer_bytes / 1e9

    # By component
    component_params = {}
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            params = sum(p.numel() for p in module.parameters(recurse=False))
            if params > 0:
                prefix = name.split(".")[0] if "." in name else name
                component_params[prefix] = component_params.get(prefix, 0) + params

    results["params_by_component"] = {
        k: v / 1e6 for k, v in sorted(component_params.items(), key=lambda x: -x[1])[:10]
    }

    return results


def analyze_optimizer_memory(optimizer) -> Dict[str, float]:
    """Analyze optimizer state memory usage."""
    results = {}

    state_bytes = 0
    state_count = 0

    for param, state in optimizer.state.items():
        if isinstance(state, dict):
            for key, val in state.items():
                if isinstance(val, torch.Tensor):
                    state_bytes += val.numel() * val.element_size()
                    state_count += 1

    results["optimizer_state_gb"] = state_bytes / 1e9
    results["optimizer_state_tensors"] = state_count

    return results


def run_diagnostic(checkpoint_path: Optional[str], model_size: str, enable_snapshots: bool):
    """Run the full memory diagnostic."""
    from hydra.training.config import TrainingConfig, MODEL_SIZE_CONFIGS
    from hydra.model.framework import HydraModel

    profiler = MemoryProfiler(enable_snapshots=enable_snapshots)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"Model size: {model_size}")
    print(f"Checkpoint: {checkpoint_path or 'None (fresh model)'}")
    print()

    # Clear GPU
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    profiler.measure("0. Initial (empty GPU)")

    # Get model config
    size_config = MODEL_SIZE_CONFIGS.get(model_size, MODEL_SIZE_CONFIGS["500M"])

    # Check if checkpoint has MoE
    moe_enabled = False
    moe_num_layers = 0
    moe_num_experts = 4

    # Architecture params - will be extracted from checkpoint if available
    dim = size_config.get("mod_mor_dim", 1792)
    n_mor_blocks = size_config.get("n_mor_blocks", 14)
    mor_recursions = size_config.get("mor_recursions", 4)
    n_heads = size_config.get("mod_mor_n_heads", 28)
    n_kv_heads = size_config.get("mod_mor_n_kv_heads", 4)
    ffn_mult = 3.72  # Default SwiGLU multiplier

    if checkpoint_path:
        print("Analyzing checkpoint structure...")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Extract architecture from checkpoint config
        ckpt_config = ckpt.get("config", {})

        def get_cfg(key, default):
            if hasattr(ckpt_config, key):
                return getattr(ckpt_config, key)
            elif isinstance(ckpt_config, dict):
                return ckpt_config.get(key, default)
            return default

        # Override with checkpoint config
        dim = get_cfg("mod_mor_dim", dim)
        n_mor_blocks = get_cfg("n_mor_blocks", n_mor_blocks)
        mor_recursions = get_cfg("mor_recursions", mor_recursions)
        n_heads = get_cfg("mod_mor_n_heads", n_heads)
        n_kv_heads = get_cfg("mod_mor_n_kv_heads", n_kv_heads)
        moe_enabled = get_cfg("moe_enabled", False)
        moe_num_layers = get_cfg("moe_num_layers", 0)
        moe_num_experts = get_cfg("moe_num_experts", 4)

        print(f"  Config from checkpoint:")
        print(f"    dim={dim}, blocks={n_mor_blocks}, recursions={mor_recursions}")
        print(f"    heads={n_heads}, kv_heads={n_kv_heads}")
        print(f"    MoE: enabled={moe_enabled}, layers={moe_num_layers}, experts={moe_num_experts}")

        # Check for MoE in state dict if not in config
        state_dict = ckpt.get("model", ckpt.get("model_state_dict", {}))
        moe_keys = [k for k in state_dict.keys() if "moe" in k.lower()]
        if moe_keys and not moe_enabled:
            moe_enabled = True
            layer_indices = set()
            for k in moe_keys:
                parts = k.split(".")
                for i, p in enumerate(parts):
                    if p == "moe_layers" and i + 1 < len(parts):
                        try:
                            layer_indices.add(int(parts[i + 1]))
                        except ValueError:
                            pass
            moe_num_layers = len(layer_indices) if layer_indices else 6
            print(f"  Detected MoE from state_dict: {moe_num_layers} layers")

        # Detect MLP hidden dim from checkpoint weights
        for k, v in state_dict.items():
            if "mlp.gate_up.weight" in k and isinstance(v, torch.Tensor):
                # gate_up is [hidden*2, dim], so hidden = shape[0] / 2
                hidden_dim = v.shape[0] // 2
                ffn_mult = hidden_dim / dim
                print(f"  Detected FFN multiplier: {ffn_mult:.2f}x (hidden={hidden_dim})")
                break

        # Estimate checkpoint size
        model_state = ckpt.get("model", {})
        total_params = sum(t.numel() for t in model_state.values() if isinstance(t, torch.Tensor))
        total_bytes = sum(t.numel() * t.element_size() for t in model_state.values() if isinstance(t, torch.Tensor))
        print(f"  Checkpoint params: {total_params / 1e6:.1f}M ({total_bytes / 1e9:.2f} GB)")

        del ckpt
        gc.collect()

    profiler.measure("1. After checkpoint analysis")

    # Create model
    print("\nCreating model...")
    model = HydraModel(
        vocab_size=50257,
        dim=dim,
        n_mor_blocks=n_mor_blocks,
        recursions_per_block=mor_recursions,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        max_seq_len=1024,
        mod_capacity=0.75,
        mlp_ratio=ffn_mult,
        moe_enabled=moe_enabled,
        moe_num_experts=moe_num_experts,
        moe_num_layers=moe_num_layers,
    )

    # Analyze model structure
    model_info = analyze_model_memory(model)
    print(f"  Parameters: {model_info['total_params_millions']:.1f}M")
    print(f"  Param memory (CPU): {model_info['param_memory_gb']:.2f} GB")
    print(f"  Buffer memory (CPU): {model_info['buffer_memory_gb']:.3f} GB")

    profiler.measure("2. Model created (CPU)")

    # Move to GPU
    print("\nMoving model to GPU...")
    model = model.cuda()

    m = profiler.measure("3. Model on GPU",
        f"Expected: {model_info['param_memory_gb']:.2f}GB params")

    # Enable gradient checkpointing
    print("Enabling gradient checkpointing...")
    if hasattr(model, "enable_gradient_checkpointing"):
        model.enable_gradient_checkpointing(every_n=1)

    profiler.measure("4. After gradient checkpointing enabled")

    # Load checkpoint if provided
    if checkpoint_path:
        print("\nLoading checkpoint...")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Handle duplicate keys
        if "model" in ckpt and "model_state_dict" in ckpt:
            print("  Removing duplicate model_state_dict...")
            del ckpt["model_state_dict"]
            gc.collect()

        state_dict = ckpt.get("model", {})

        # Drop RoPE caches
        rope_keys = [k for k in state_dict.keys() if "cos_cached" in k or "sin_cached" in k]
        for k in rope_keys:
            del state_dict[k]

        profiler.measure("5a. Checkpoint loaded to CPU")

        model.load_state_dict(state_dict, strict=False)

        profiler.measure("5b. State dict loaded to model")

        # Clean up checkpoint from CPU memory
        del ckpt, state_dict
        gc.collect()

        profiler.measure("5c. Checkpoint cleaned from CPU")

    # Create optimizer
    print("\nCreating optimizer (8-bit Adam)...")
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.Adam8bit(model.parameters(), lr=1e-4)
        opt_type = "8-bit Adam"
    except ImportError:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        opt_type = "AdamW (32-bit)"

    profiler.measure(f"6. Optimizer created ({opt_type})")

    # Create dummy batch
    print("\nRunning forward pass...")
    batch_size = 4
    seq_len = 1024
    x = torch.randint(0, 50257, (batch_size, seq_len), device="cuda")
    y = torch.randint(0, 50257, (batch_size, seq_len), device="cuda")

    profiler.measure("7. Dummy batch created")

    # Forward pass
    profiler.reset_peak()
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        outputs = model(x)

        # Handle both dict and tensor outputs
        if isinstance(outputs, dict):
            logits = outputs["logits"]
        else:
            logits = outputs

        # Compute loss
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1),
            ignore_index=-100,
        )

        # Add auxiliary losses if available
        if isinstance(outputs, dict):
            if "aux_loss" in outputs and outputs["aux_loss"] is not None:
                loss = loss + 0.01 * outputs["aux_loss"]
            if "ponder_loss" in outputs and outputs["ponder_loss"] is not None:
                loss = loss + 0.01 * outputs["ponder_loss"]

    profiler.measure("8. Forward pass complete",
        f"Loss: {loss.item():.4f}")

    # Backward pass
    print("Running backward pass...")
    profiler.reset_peak()
    loss.backward()

    m = profiler.measure("9. Backward pass complete")

    # Check gradient memory
    grad_bytes = sum(
        p.grad.numel() * p.grad.element_size()
        for p in model.parameters()
        if p.grad is not None
    )
    print(f"  Gradient memory: {grad_bytes / 1e9:.2f} GB")

    # Optimizer step
    print("Running optimizer step...")
    profiler.reset_peak()
    optimizer.step()

    profiler.measure("10. Optimizer step complete")

    # Analyze optimizer state
    opt_info = analyze_optimizer_memory(optimizer)
    print(f"  Optimizer state: {opt_info['optimizer_state_gb']:.2f} GB ({opt_info['optimizer_state_tensors']} tensors)")

    # Clean up
    optimizer.zero_grad(set_to_none=True)
    del x, y, logits, loss, outputs
    gc.collect()
    torch.cuda.empty_cache()

    profiler.measure("11. After cleanup")

    # Second forward/backward to check for leaks
    print("\nSecond iteration (check for memory leaks)...")
    x = torch.randint(0, 50257, (batch_size, seq_len), device="cuda")
    y = torch.randint(0, 50257, (batch_size, seq_len), device="cuda")

    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        outputs = model(x)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1),
        )
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    del x, y, loss, outputs
    gc.collect()
    torch.cuda.empty_cache()

    profiler.measure("12. After second iteration + cleanup")

    # Final report
    profiler.report()

    # Memory breakdown estimate
    print("\n" + "="*80)
    print("ESTIMATED MEMORY BREAKDOWN")
    print("="*80)

    param_gb = model_info['param_memory_gb']
    grad_gb = grad_bytes / 1e9
    opt_gb = opt_info['optimizer_state_gb']

    print(f"Model parameters (bf16):      {param_gb:.2f} GB")
    print(f"Gradients (bf16):             {grad_gb:.2f} GB")
    print(f"Optimizer state (8-bit):      {opt_gb:.2f} GB")
    print(f"                              --------")
    print(f"Subtotal (static):            {param_gb + grad_gb + opt_gb:.2f} GB")
    print()
    print(f"Activations & buffers:        ~{profiler.measurements[-1].allocated_gb - param_gb - grad_gb - opt_gb:.2f} GB (estimated)")
    print(f"                              --------")
    print(f"Total allocated:              {profiler.measurements[-1].allocated_gb:.2f} GB")
    print(f"Total reserved:               {profiler.measurements[-1].reserved_gb:.2f} GB")

    if enable_snapshots:
        profiler.save_snapshot()

    return profiler


def main():
    parser = argparse.ArgumentParser(description="Deep memory diagnostic")
    parser.add_argument("--checkpoint", type=str, help="Checkpoint path to load")
    parser.add_argument("--model_size", type=str, default="500M", help="Model size")
    parser.add_argument("--snapshot", action="store_true", help="Save memory snapshot for visualization")
    args = parser.parse_args()

    run_diagnostic(args.checkpoint, args.model_size, args.snapshot)


if __name__ == "__main__":
    main()
