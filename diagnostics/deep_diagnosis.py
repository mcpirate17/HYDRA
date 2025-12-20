"""
Deep diagnosis of CCGQA+MoD+MoR - check gradient flow layer by layer.

Industry-standard gradient analysis for debugging training instability.
Use with memory profiler: mprof run python diagnostics/deep_diagnosis.py

Optional: Install memory-profiler for detailed analysis:
    pip install memory-profiler
    mprof run python diagnostics/deep_diagnosis.py  
    mprof plot
"""

import os
import torch
import torch.nn as nn
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hydra.model.ccgqa import create_ccgqa_mod_mor_model

# Device selection with CPU fallback
if torch.cuda.is_available():
    device = "cuda"
    # Default attention pattern (per MoR block): 3:1 macro-block [LLA2, LLA2, LLA2, CCQA].
    # Users can override via HYDRA_MOR_ATTENTION_PATTERN / HYDRA_MOR_ATTENTION_PATTERN_NAME.
    os.environ.setdefault("HYDRA_MOR_ATTENTION_PATTERN_NAME", "lla2x3+ccqa")
else:
    device = "cpu"
    print("⚠️  WARNING: CUDA not available, running on CPU.")
    print("   Results may differ from GPU due to numerical precision differences.")
    print("   For accurate diagnostics, run on a CUDA-enabled system.\n")

# Create model
config = {
    "vocab_size": 50257,
    "dim": 1536,
    "n_mor_blocks": 16,
    "recursions_per_block": 4,
    "n_heads": 24,
    "n_kv_heads": 4,
    "compression_factor": 4,
    "mlp_ratio": 4.0,
    "max_seq_len": 2048,
    "mod_capacity": 0.75,
    "adaptive": True,
}

print("Creating 570M model...")
model = create_ccgqa_mod_mor_model(**config).to(device)

# Create batch
batch_size = 4
seq_len = 128
input_ids = torch.randint(0, 50257, (batch_size, seq_len)).to(device)
inputs = input_ids[:, :-1].contiguous()
targets = input_ids[:, 1:].contiguous()

model.train()

print("\n" + "=" * 60)
print("TEST 1: Verify each layer's output changes with input")
print("=" * 60)

# Hook to capture intermediate outputs
layer_outputs = {}


def make_hook(name):
    def hook(module, input, output):
        if isinstance(output, tuple):
            layer_outputs[name] = output[0].detach().clone()
        else:
            layer_outputs[name] = output.detach().clone()

    return hook


# Register hooks on key layers
handles = []
handles.append(model.tok_emb.register_forward_hook(make_hook("tok_emb")))
for i, layer in enumerate(model.layers):
    handles.append(layer.register_forward_hook(make_hook(f"layer_{i}")))
handles.append(model.norm.register_forward_hook(make_hook("final_norm")))
handles.append(model.output.register_forward_hook(make_hook("output")))

# Forward pass
with torch.no_grad():
    logits, losses = model(inputs, return_losses=True)

# Check layer outputs
print(f"\nLayer-by-layer output statistics:")
print("-" * 70)
prev_output = None
for name, output in layer_outputs.items():
    mean = output.mean().item()
    std = output.std().item()
    min_val = output.min().item()
    max_val = output.max().item()

    # Check if output is all zeros or constant
    is_constant = std < 1e-6
    is_zero = mean < 1e-6 and std < 1e-6

    status = ""
    if is_zero:
        status = " ⚠️ ALL ZEROS!"
    elif is_constant:
        status = " ⚠️ CONSTANT!"

    print(
        f"  {name:15s}: mean={mean:8.4f}, std={std:8.4f}, range=[{min_val:.4f}, {max_val:.4f}]{status}"
    )

    # Check if this layer changed from previous
    if prev_output is not None and prev_output.shape == output.shape:
        diff = (output - prev_output).abs().mean().item()
        if diff < 1e-6:
            print(f"    ⚠️ No change from previous layer!")

    prev_output = output

# Clean up hooks
for handle in handles:
    handle.remove()

print("\n" + "=" * 60)
print("TEST 2: Gradient magnitude by layer after backward")
print("=" * 60)

# Fresh forward
logits, losses = model(inputs, return_losses=True)
loss = nn.functional.cross_entropy(
    logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
)

# Backward
loss.backward()

# Check gradients by layer
print("\nGradient magnitudes by component:")
print("-" * 70)

grad_stats = {}
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        grad_mean = param.grad.abs().mean().item()

        # Group by component
        parts = name.split(".")
        component = parts[0] if len(parts) > 0 else name
        if component == "layers":
            if len(parts) > 1:
                component = f"layers.{parts[1]}"

        if component not in grad_stats:
            grad_stats[component] = {"norms": [], "means": []}
        grad_stats[component]["norms"].append(grad_norm)
        grad_stats[component]["means"].append(grad_mean)

for component, stats in sorted(grad_stats.items()):
    avg_norm = sum(stats["norms"]) / len(stats["norms"])
    avg_mean = sum(stats["means"]) / len(stats["means"])
    max_norm = max(stats["norms"])

    status = ""
    if avg_mean < 1e-8:
        status = " ⚠️ VANISHING!"
    elif avg_mean > 1.0:
        status = " ⚠️ EXPLODING!"

    print(
        f"  {component:20s}: avg_norm={avg_norm:10.6f}, avg_mean={avg_mean:10.6f}, max={max_norm:10.6f}{status}"
    )

print("\n" + "=" * 60)
print("TEST 3: Check if model actually updates after optimizer step")
print("=" * 60)

# Save original weights
original_weights = {}
for name, param in model.named_parameters():
    if "block" in name and param.requires_grad:
        original_weights[name] = param.data.clone()

# Optimizer step
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)  # Higher LR to see effect
optimizer.zero_grad()

logits, _ = model(inputs, return_losses=True)
loss = nn.functional.cross_entropy(
    logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
)
loss.backward()
optimizer.step()

# Check weight changes
print("\nWeight changes after optimizer step (sample):")
print("-" * 70)
changes = []
for name, param in model.named_parameters():
    if name in original_weights:
        diff = (param.data - original_weights[name]).abs().mean().item()
        changes.append((name, diff))

# Sort by change magnitude and show top/bottom
changes.sort(key=lambda x: x[1], reverse=True)

print("Largest changes:")
for name, diff in changes[:5]:
    print(f"  {name}: {diff:.8f}")

print("\nSmallest changes:")
for name, diff in changes[-5:]:
    status = " ⚠️ NOT UPDATING!" if diff < 1e-10 else ""
    print(f"  {name}: {diff:.8f}{status}")

print("\n" + "=" * 60)
print("TEST 4: Check MoR routing - are tokens going to different depths?")
print("=" * 60)

# Get routing stats
if hasattr(model, "get_routing_stats"):
    stats = model.get_routing_stats()
    print("\nRouting statistics:")
    if stats.get("mor_layers"):
        for i, mor_stats in enumerate(stats["mor_layers"]):
            avg_depth = mor_stats.get("avg_depth", "N/A")
            depth_std = mor_stats.get("depth_std", "N/A")
            probs_mean = mor_stats.get("probs_mean", "N/A")
            print(
                f"  MoR Layer {i}: avg_depth={avg_depth}, std={depth_std}, router_probs={probs_mean}"
            )
    else:
        print("  No MoR routing stats available")

    if stats.get("mod_layers"):
        for i, mod_stats in enumerate(stats["mod_layers"]):
            probs_mean = mod_stats.get("probs_mean", "N/A")
            print(f"  MoD Layer {i}: probs_mean={probs_mean}")
    else:
        print("  No MoD routing stats available")
else:
    print("  Model doesn't have get_routing_stats method")

print("\n" + "=" * 60)
print("DIAGNOSIS COMPLETE")
print("=" * 60)
