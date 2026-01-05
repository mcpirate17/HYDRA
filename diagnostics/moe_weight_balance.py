#!/usr/bin/env python3
"""
Diagnose MoE weight balance issues.

The forced routing phase can cause expert weights to grow much faster than
backbone weights, creating gradient flow imbalances post-transition.

Usage:
    python diagnostics/moe_weight_balance.py checkpoints/hydra_500m_step_80500.pt
"""

import sys
import math
import torch


def analyze_checkpoint(ckpt_path: str):
    """Analyze weight balance in a checkpoint."""
    print(f"Loading {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_state = ckpt.get("model", ckpt)
    step = ckpt.get("step", "unknown")
    
    print(f"\n{'='*60}")
    print(f"MoE Weight Balance Analysis - Step {step}")
    print(f"{'='*60}\n")
    
    # Collect weight norms by category
    backbone_mlp_norms = []
    expert_norms = []
    router_norms = []
    attention_norms = []
    
    for name, w in model_state.items():
        if not isinstance(w, torch.Tensor) or not w.is_floating_point():
            continue
        norm = w.float().norm().item()
        
        if "moe_layers" in name:
            if "router" in name:
                router_norms.append((name, norm))
            elif "expert" in name:
                expert_norms.append((name, norm))
        elif "mlp" in name and "gate_up" in name:
            backbone_mlp_norms.append((name, norm))
        elif "attention" in name and "proj" in name:
            attention_norms.append((name, norm))
    
    # Summary statistics
    if backbone_mlp_norms:
        avg_mlp = sum(n for _, n in backbone_mlp_norms) / len(backbone_mlp_norms)
        print(f"Backbone MLP (gate_up) average norm: {avg_mlp:.2f}")
    
    if expert_norms:
        avg_expert = sum(n for _, n in expert_norms) / len(expert_norms)
        print(f"MoE Expert (all params) average norm: {avg_expert:.2f}")
        
        # Filter just gate_up for fair comparison
        expert_gate_up = [(n, v) for n, v in expert_norms if "gate_up" in n]
        if expert_gate_up:
            avg_expert_gateup = sum(n for _, n in expert_gate_up) / len(expert_gate_up)
            print(f"MoE Expert (gate_up only) average norm: {avg_expert_gateup:.2f}")
            
            if backbone_mlp_norms:
                ratio = avg_expert_gateup / avg_mlp
                print(f"\n⚠️  Expert/Backbone ratio: {ratio:.1f}x")
                if ratio > 5:
                    print(f"   CRITICAL: Experts are {ratio:.0f}x larger than backbone!")
                    print(f"   This causes gradient flow imbalance post-MoE-transition.")
    
    if router_norms:
        avg_router = sum(n for _, n in router_norms) / len(router_norms)
        print(f"\nRouter gate average norm: {avg_router:.4f}")
        if expert_norms:
            expert_router_ratio = avg_expert_gateup / avg_router if avg_router > 0 else float('inf')
            print(f"Expert/Router ratio: {expert_router_ratio:.0f}x")
    
    # Per-layer breakdown
    print(f"\n{'='*60}")
    print("Per-Layer Expert Norms (gate_up projection)")
    print(f"{'='*60}")
    
    for layer_idx in range(20):  # Assume max 20 MoE layers
        layer_expert_norms = []
        for expert_idx in range(10):  # Assume max 10 experts
            key = f"moe_layers.{layer_idx}.experts.{expert_idx}.gate_up.weight"
            if key in model_state:
                layer_expert_norms.append(model_state[key].float().norm().item())
        
        if layer_expert_norms:
            print(f"  Layer {layer_idx}: {[f'{n:.0f}' for n in layer_expert_norms]}")
    
    # Recommendations
    print(f"\n{'='*60}")
    print("Recommendations")
    print(f"{'='*60}")
    
    if expert_norms and backbone_mlp_norms:
        ratio = avg_expert_gateup / avg_mlp if avg_mlp > 0 else float('inf')
        
        if ratio > 10:
            print("""
⚠️  SEVERE imbalance detected! Options:

1. WEIGHT RESCALING (immediate fix):
   - Scale expert weights down by {:.1f}x to match backbone
   - Scale router weights up by sqrt({:.1f})x for balance
   
2. GRADIENT SCALING (training fix):
   - Add expert_grad_scale parameter (0.1-0.3) to dampen expert gradients
   - Or use separate learning rates: expert_lr = base_lr * 0.1
   
3. RESTART TRAINING (cleanest fix):
   - Restart from pre-MoE checkpoint with balanced initialization
   - Use identity_init=True with slow warmup
""".format(ratio, ratio))
        elif ratio > 5:
            print("""
⚠️  Moderate imbalance. Consider:
   - Reducing learning rate for expert parameters
   - Using gradient clipping specifically on expert gradients
""")
        else:
            print("✅ Weight balance looks reasonable.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python diagnostics/moe_weight_balance.py <checkpoint.pt>")
        sys.exit(1)
    
    analyze_checkpoint(sys.argv[1])
