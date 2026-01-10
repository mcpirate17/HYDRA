#!/usr/bin/env python3
"""
Evaluate MoE expert specialization by running actual inference.

This diagnostic loads a checkpoint, runs inference on domain-specific test data,
and measures which experts activate for which domains.

Notes:
 - RECOMMENDED: Run inference-based routing analysis on CUDA for speed; weight-based analysis works on CPU.

Usage:
    # Full routing analysis (runs inference):
    python diagnostics/moe_specialization.py checkpoints/hydra_500m_step_58000.pt
    python diagnostics/moe_specialization.py checkpoints/hydra_500m_step_58000.pt --samples 100
    
    # Quick weight-based comparison (no inference):
    python diagnostics/moe_specialization.py --compare checkpoints/hydra_500m_step_*.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

import torch
import torch.nn.functional as F


# =============================================================================
# WEIGHT-BASED ANALYSIS (works without inference)
# =============================================================================

def analyze_expert_weights(state: dict, num_experts: int = 4) -> Dict:
    """Analyze expert weight divergence from checkpoint state dict."""
    
    # Find MoE layer indices
    moe_layer_indices = set()
    for k in state.keys():
        if k.startswith("moe_layers.") and ".experts." in k:
            idx = int(k.split(".")[1])
            moe_layer_indices.add(idx)
    
    if not moe_layer_indices:
        return {"error": "No MoE layers found"}
    
    results = {"moe_layers": []}
    
    for layer_idx in sorted(moe_layer_indices):
        layer_stats = {"layer": layer_idx}
        
        # Collect expert weights
        experts = []
        for exp_idx in range(num_experts):
            key = f"moe_layers.{layer_idx}.experts.{exp_idx}.gate_up.weight"
            if key in state:
                experts.append(state[key].float().flatten())
        
        if len(experts) >= 2:
            # Pairwise cosine distance
            cos_dists = []
            for i in range(len(experts)):
                for j in range(i + 1, len(experts)):
                    cos_sim = F.cosine_similarity(
                        experts[i].unsqueeze(0), experts[j].unsqueeze(0)
                    ).item()
                    cos_dists.append(1 - cos_sim)  # 0=identical, 1=orthogonal
            
            layer_stats["expert_cosine_dist"] = sum(cos_dists) / len(cos_dists)
            layer_stats["specialized"] = layer_stats["expert_cosine_dist"] > 0.5
        
        results["moe_layers"].append(layer_stats)
    
    # Aggregate
    if results["moe_layers"]:
        avg_div = sum(l.get("expert_cosine_dist", 0) for l in results["moe_layers"]) / len(results["moe_layers"])
        results["avg_divergence"] = avg_div
        results["any_specialized"] = any(l.get("specialized", False) for l in results["moe_layers"])
    
    return results


# =============================================================================
# INFERENCE-BASED ANALYSIS (shows actual routing behavior)
# =============================================================================

def load_model_from_checkpoint(ckpt_path: str, device: str = "cuda"):
    """Load model from checkpoint for inference."""
    from hydra.model.framework.model import HydraModel
    
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt.get("config")
    state = ckpt.get("model", ckpt.get("model_state_dict", {}))
    
    if config is None:
        raise ValueError("Checkpoint missing config")
    
    # Config can be dict or object - normalize access
    def cfg_get(key, default):
        if isinstance(config, dict):
            return config.get(key, default)
        return getattr(config, key, default)
    
    # Infer mlp_ratio from actual weights if possible
    mlp_ratio = cfg_get("mlp_ratio", 2.67)
    mlp_key = "layers.0.mlp.gate_up.weight"
    if mlp_key in state:
        w = state[mlp_key]
        inferred_dim = w.shape[1]
        inferred_int = w.shape[0] // 2  # gate_up is [intermediate*2, dim]
        mlp_ratio = inferred_int / inferred_dim
    
    # Build model with parameters from checkpoint config
    model = HydraModel(
        vocab_size=cfg_get("vocab_size", 50257),
        dim=cfg_get("mod_mor_dim", cfg_get("dim", 1792)),  # mod_mor models use mod_mor_dim
        n_mor_blocks=cfg_get("n_mor_blocks", 8),
        recursions_per_block=cfg_get("mor_recursions", 4),
        n_heads=cfg_get("mod_mor_n_heads", cfg_get("n_heads", 28)),
        n_kv_heads=cfg_get("mod_mor_n_kv_heads", cfg_get("n_kv_heads", 4)),
        compression_factor=cfg_get("compression_factor", 4),
        mlp_ratio=mlp_ratio,
        max_seq_len=cfg_get("max_seq_len", 8192),
        mod_capacity=cfg_get("mod_capacity", 0.5),
        moe_enabled=cfg_get("moe_enabled", True),
        moe_num_experts=cfg_get("moe_num_experts", 4),
        moe_num_layers=cfg_get("moe_num_layers", 6),
    )
    
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    
    return model, config, ckpt.get("step", 0)


def get_test_samples(domain: str, num_samples: int = 50, seq_len: int = 512) -> List[torch.Tensor]:
    """Get test text samples for a domain."""
    from hydra.data.universal_data_loader import create_universal_loader
    
    try:
        loader = create_universal_loader(
            dataset=domain,
            batch_size=1,
            seq_len=seq_len,
            vocab_size=32000,
            device="cpu",
        )
        
        samples = []
        for _ in range(num_samples):
            batch = loader.get_batch()
            if batch and "input_ids" in batch:
                samples.append(batch["input_ids"])
        
        loader.close()
        return samples
    except Exception as e:
        print(f"  Warning: Could not load '{domain}': {e}")
        return []


def measure_routing_for_domain(
    model,
    samples: List[torch.Tensor],
    device: str = "cuda",
    num_experts: int = 4,
) -> Dict[int, float]:
    """Run inference and measure which experts are activated.
    
    Returns dict mapping expert_idx -> fraction of tokens routed to it.
    """
    if not samples:
        return {}
    
    # Accumulate expert counts across all samples
    total_counts = torch.zeros(num_experts)
    
    # Hook to capture routing decisions
    routing_counts = defaultdict(lambda: torch.zeros(num_experts))
    
    def make_hook(layer_idx):
        def hook(module, inputs, outputs):
            # outputs from router: (top_k_indices, top_k_weights, aux_loss)
            # or (top_k_indices, top_k_weights, aux_loss, logits) if return_logits
            if isinstance(outputs, tuple) and len(outputs) >= 2:
                top_k_indices = outputs[0]  # [B, L, K]
                # Count top-1 assignments
                top1 = top_k_indices[..., 0]  # [B, L]
                for exp_idx in range(num_experts):
                    count = (top1 == exp_idx).sum().item()
                    routing_counts[layer_idx][exp_idx] += count
        return hook
    
    # Register hooks on all routers
    hooks = []
    if hasattr(model, "moe_layers"):
        for idx, moe_layer in enumerate(model.moe_layers):
            if hasattr(moe_layer, "router"):
                h = moe_layer.router.register_forward_hook(make_hook(idx))
                hooks.append(h)
    
    try:
        with torch.no_grad():
            for sample in samples:
                if isinstance(sample, torch.Tensor):
                    x = sample.to(device)
                else:
                    continue
                
                if x.dim() == 1:
                    x = x.unsqueeze(0)
                
                # Forward pass (captures routing via hooks)
                try:
                    _ = model(x)
                except Exception:
                    pass  # Some samples may fail, that's ok
    finally:
        # Remove hooks
        for h in hooks:
            h.remove()
    
    # Aggregate across layers
    for layer_idx, counts in routing_counts.items():
        total_counts += counts
    
    if total_counts.sum() > 0:
        fractions = (total_counts / total_counts.sum()).tolist()
        return {i: fractions[i] for i in range(num_experts)}
    return {}


def evaluate_moe_routing(
    ckpt_path: str,
    domains: List[str] = None,
    samples_per_domain: int = 50,
    device: str = "cuda",
) -> Dict:
    """Full evaluation: load model, run inference on each domain, measure routing."""
    
    if domains is None:
        domains = ["math", "code", "chat", "finefineweb-local"]
    
    print(f"\nLoading model from {ckpt_path}...")
    model, config, step = load_model_from_checkpoint(ckpt_path, device)
    
    num_experts = getattr(config, "moe_num_experts", 4)
    
    # Also get weight-based divergence
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt.get("model_state_dict", {}))
    weight_analysis = analyze_expert_weights(state, num_experts)
    
    results = {
        "step": step,
        "ema_loss": ckpt.get("metrics", {}).get("ema_loss", 0),
        "num_experts": num_experts,
        "weight_divergence": weight_analysis.get("avg_divergence", 0),
        "domains": {},
    }
    
    print(f"Running inference on {len(domains)} domains ({samples_per_domain} samples each)...\n")
    
    for domain in domains:
        print(f"  Testing '{domain}'...", end=" ", flush=True)
        samples = get_test_samples(domain, samples_per_domain)
        
        if not samples:
            print("SKIPPED (no data)")
            continue
        
        routing = measure_routing_for_domain(model, samples, device, num_experts)
        results["domains"][domain] = routing
        
        if routing:
            # Show which expert "won" for this domain
            winner = max(routing.items(), key=lambda x: x[1])
            pcts = [f"{routing.get(i, 0)*100:.0f}%" for i in range(num_experts)]
            print(f"-> Expert {winner[0]} ({winner[1]*100:.0f}%)  [{'/'.join(pcts)}]")
        else:
            print("-> No routing data")
    
    # Clean up
    del model
    torch.cuda.empty_cache()
    
    return results


def format_routing_report(results: Dict) -> str:
    """Format routing results as a readable report."""
    lines = []
    lines.append(f"\n{'='*70}")
    lines.append(f"MoE ROUTING ANALYSIS - Step {results['step']}")
    lines.append(f"{'='*70}")
    lines.append(f"EMA Loss: {results['ema_loss']:.4f}")
    lines.append(f"Weight Divergence: {results['weight_divergence']:.4f} (1.0 = orthogonal)")
    lines.append("")
    
    num_experts = results["num_experts"]
    domains = results.get("domains", {})
    
    if not domains:
        lines.append("No domain routing data available.")
        return "\n".join(lines)
    
    # Header
    expert_cols = "  ".join([f"Exp{i:d}" for i in range(num_experts)])
    lines.append(f"{'Domain':<20} {expert_cols}   Winner")
    lines.append("-" * 60)
    
    # Build matrix for analysis
    routing_matrix = {}
    for domain, routing in domains.items():
        routing_matrix[domain] = routing
        pcts = [f"{routing.get(i, 0)*100:5.1f}%" for i in range(num_experts)]
        winner_idx = max(routing.items(), key=lambda x: x[1])[0] if routing else -1
        winner_str = f"Expert {winner_idx}" if winner_idx >= 0 else "N/A"
        lines.append(f"{domain:<20} {'  '.join(pcts)}   {winner_str}")
    
    lines.append("")
    
    # Specialization analysis
    lines.append("SPECIALIZATION ANALYSIS:")
    
    # Check if each expert has a preferred domain
    expert_preferences = {}
    for exp_idx in range(num_experts):
        best_domain = None
        best_pct = 0
        for domain, routing in routing_matrix.items():
            pct = routing.get(exp_idx, 0)
            if pct > best_pct:
                best_pct = pct
                best_domain = domain
        expert_preferences[exp_idx] = (best_domain, best_pct)
    
    for exp_idx, (domain, pct) in expert_preferences.items():
        if domain:
            status = "✅" if pct > 0.35 else "⚠️" if pct > 0.25 else "❌"
            lines.append(f"  Expert {exp_idx}: {domain} ({pct*100:.1f}%) {status}")
    
    # Check for collapse (one expert dominates all domains)
    all_winners = [max(r.items(), key=lambda x: x[1])[0] for r in routing_matrix.values() if r]
    if len(set(all_winners)) == 1 and len(all_winners) > 1:
        lines.append("")
        lines.append(f"  ⚠️  WARNING: Expert {all_winners[0]} dominates ALL domains!")
        lines.append("     Consider: longer forced routing, stronger teacher loss, or balanced data")
    elif len(set(all_winners)) == len(domains):
        lines.append("")
        lines.append("  ✅ GOOD: Each domain routes to a different expert!")
    
    lines.append(f"{'='*70}")
    
    return "\n".join(lines)


# =============================================================================
# COMPARISON MODE (weight-based only, for speed)
# =============================================================================

def compare_checkpoints_weights_only(paths: List[str]) -> str:
    """Quick comparison using weight divergence only (no inference)."""
    results = []
    
    for p in sorted(paths):
        try:
            ckpt = torch.load(p, map_location="cpu", weights_only=False)
            state = ckpt.get("model", ckpt.get("model_state_dict", {}))
            step = ckpt.get("step", 0)
            ema = ckpt.get("metrics", {}).get("ema_loss", 0)
            
            analysis = analyze_expert_weights(state)
            
            if "error" in analysis:
                results.append({"step": step, "ema_loss": ema, "error": analysis["error"]})
            else:
                results.append({
                    "step": step,
                    "ema_loss": ema,
                    "divergence": analysis.get("avg_divergence", 0),
                    "specialized": analysis.get("any_specialized", False),
                })
        except Exception as e:
            print(f"Error loading {p}: {e}")
    
    if not results:
        return "No checkpoints loaded"
    
    lines = []
    lines.append(f"\n{'='*70}")
    lines.append("MoE WEIGHT DIVERGENCE PROGRESSION")
    lines.append("(For routing analysis, run on single checkpoint without --compare)")
    lines.append(f"{'='*70}")
    lines.append(f"{'Step':<10} {'EMA Loss':<12} {'Divergence':<14} {'Status'}")
    lines.append("-" * 50)
    
    for r in results:
        step = r.get("step", 0)
        ema = r.get("ema_loss", 0)
        if "error" in r:
            lines.append(f"{step:<10} {ema:<12.4f} {'N/A':<14} ❌ {r['error']}")
        else:
            div = r.get("divergence", 0)
            status = "✅ Specialized" if r.get("specialized") else "⏳ Learning"
            lines.append(f"{step:<10} {ema:<12.4f} {div:<14.4f} {status}")
    
    # Trend
    valid = [r for r in results if "divergence" in r]
    if len(valid) >= 2:
        first, last = valid[0], valid[-1]
        delta = last["divergence"] - first["divergence"]
        lines.append("")
        lines.append(f"Trend: {first['divergence']:.4f} → {last['divergence']:.4f} ({delta:+.4f})")
    
    lines.append(f"{'='*70}")
    return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MoE expert specialization")
    parser.add_argument("checkpoints", nargs="+", help="Checkpoint path(s)")
    parser.add_argument("--compare", action="store_true", 
                       help="Quick comparison mode (weight divergence only)")
    parser.add_argument("--samples", type=int, default=50,
                       help="Samples per domain for routing analysis")
    parser.add_argument("--domains", nargs="+", 
                       default=["math", "code", "chat", "finefineweb-local"],
                       help="Domains to test routing on")
    parser.add_argument("--device", default="cuda", help="Device for inference")
    parser.add_argument("--cpu", action="store_true", help="Use CPU (slow)")
    
    args = parser.parse_args()
    
    if args.cpu:
        args.device = "cpu"
    
    if args.compare or len(args.checkpoints) > 1:
        # Quick weight-based comparison
        print(compare_checkpoints_weights_only(args.checkpoints))
    else:
        # Full routing analysis on single checkpoint
        ckpt_path = args.checkpoints[0]
        results = evaluate_moe_routing(
            ckpt_path,
            domains=args.domains,
            samples_per_domain=args.samples,
            device=args.device,
        )
        print(format_routing_report(results))


if __name__ == "__main__":
    main()
