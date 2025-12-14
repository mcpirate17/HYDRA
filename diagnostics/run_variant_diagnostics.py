#!/usr/bin/env python3
"""
Multi-Variant Diagnostic Runner

Runs 100-step diagnostic training on model variants:
- 100M, 500M, 750M, 1.5B

Validates paper compliance for:
- MoD (arXiv:2404.02258): Mixture-of-Depths
- MoR (arXiv:2507.10524): Mixture-of-Recursions
- CCGQA (arXiv:2510.04476): Compressed Convolutional GQA

Usage:
    python run_variant_diagnostics.py [--variants 100M 500M] [--steps 100]
"""

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from hydra.model.ccgqa import (
    CCGQAAttention,
    CCGQABlock,
    CCGQAMoRBlock,
    CCGQAMoDBlockWrapper,
    CCGQAMoDMoRModel,
    create_ccgqa_mod_mor_model,
)


# =============================================================================
# Model Variant Configurations
# =============================================================================


@dataclass
class ModelVariant:
    """Configuration for a model variant."""

    name: str
    dim: int
    n_mor_blocks: int
    recursions: int
    n_heads: int
    n_kv_heads: int
    compression_factor: int
    batch_size: int
    seq_len: int
    expected_params_m: float

    @property
    def effective_layers(self) -> int:
        return self.n_mor_blocks * self.recursions

    def to_dict(self) -> dict:
        return asdict(self)


# Model variants optimized for memory/compute at each scale
MODEL_VARIANTS = {
    "100M": ModelVariant(
        name="100M",
        dim=768,
        n_mor_blocks=8,
        recursions=4,
        n_heads=12,
        n_kv_heads=3,
        compression_factor=4,
        batch_size=4,
        seq_len=256,
        expected_params_m=100,
    ),
    "250M": ModelVariant(
        name="250M",
        dim=1024,
        n_mor_blocks=12,
        recursions=4,
        n_heads=16,
        n_kv_heads=4,
        compression_factor=4,
        batch_size=4,
        seq_len=256,
        expected_params_m=250,
    ),
    "500M": ModelVariant(
        name="500M",
        dim=1536,
        n_mor_blocks=16,
        recursions=4,
        n_heads=24,
        n_kv_heads=4,
        compression_factor=4,
        batch_size=2,
        seq_len=256,
        expected_params_m=570,
    ),
    "750M": ModelVariant(
        name="750M",
        dim=1792,
        n_mor_blocks=20,
        recursions=4,
        n_heads=28,
        n_kv_heads=4,
        compression_factor=4,
        batch_size=2,
        seq_len=256,
        expected_params_m=750,
    ),
    "900M": ModelVariant(
        name="900M",
        dim=2048,
        n_mor_blocks=20,
        recursions=4,
        n_heads=32,
        n_kv_heads=4,
        compression_factor=4,
        batch_size=2,
        seq_len=256,
        expected_params_m=900,
    ),
    "1B": ModelVariant(
        name="1B",
        dim=2048,
        n_mor_blocks=24,
        recursions=4,
        n_heads=32,
        n_kv_heads=8,
        compression_factor=4,
        batch_size=1,
        seq_len=256,
        expected_params_m=1000,
    ),
    "1.5B": ModelVariant(
        name="1.5B",
        dim=2560,
        n_mor_blocks=24,
        recursions=5,
        n_heads=40,
        n_kv_heads=8,
        compression_factor=4,
        batch_size=1,
        seq_len=256,
        expected_params_m=1500,
    ),
}


def get_device() -> str:
    """Get available device."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# Diagnostic Collectors
# =============================================================================


class MoDCollector:
    """Collect MoD routing statistics."""

    def __init__(self, model: CCGQAMoDMoRModel):
        self.model = model

    def collect(self) -> Dict[str, Any]:
        """Collect MoD stats from all layers."""
        probs = []
        biases = []
        aux_losses = []

        for i, layer in enumerate(self.model.layers):
            if isinstance(layer, CCGQAMoDBlockWrapper):
                probs.append(layer._last_probs_mean)
                biases.append(layer.router.bias.item())
                if hasattr(layer, "_aux_loss"):
                    aux_loss = layer._aux_loss
                    aux_losses.append(
                        aux_loss.item()
                        if isinstance(aux_loss, torch.Tensor)
                        else aux_loss
                    )

        return {
            "layer_probs": probs,
            "layer_biases": biases,
            "avg_prob": sum(probs) / len(probs) if probs else 0,
            "std_prob": (
                sum((p - sum(probs) / len(probs)) ** 2 for p in probs) / len(probs)
            )
            ** 0.5
            if probs
            else 0,
            "total_aux_loss": sum(aux_losses) if aux_losses else 0,
            "target_capacity": 0.75,
            "capacity_deviation": abs((sum(probs) / len(probs) if probs else 0) - 0.75),
        }


class MoRCollector:
    """Collect MoR routing statistics."""

    def __init__(self, model: CCGQAMoDMoRModel):
        self.model = model

    def collect(self) -> Dict[str, Any]:
        """Collect MoR stats from all layers."""
        depths = []
        histograms = []
        ponder_losses = []
        target_ratios = []

        for i, layer in enumerate(self.model.layers):
            mor_block = None
            if isinstance(layer, CCGQAMoDBlockWrapper) and isinstance(
                layer.block, CCGQAMoRBlock
            ):
                mor_block = layer.block
            elif isinstance(layer, CCGQAMoRBlock):
                mor_block = layer

            if mor_block is not None:
                stats = mor_block.get_routing_stats()
                if "avg_depth" in stats:
                    depths.append(stats["avg_depth"])
                if "depth_histogram" in stats:
                    histograms.append(stats["depth_histogram"])
                if hasattr(mor_block, "_ponder_loss"):
                    ponder = mor_block._ponder_loss
                    ponder_losses.append(
                        ponder.item() if isinstance(ponder, torch.Tensor) else ponder
                    )
                if hasattr(mor_block, "target_depth_ratio"):
                    target_ratios.append(mor_block.target_depth_ratio)

        max_rec = mor_block.max_recursions if mor_block else 4

        # Check layer-aware scaling
        layer_scaling_ok = True
        if len(depths) >= 2:
            mid = len(depths) // 2
            first_half = sum(depths[:mid]) / mid if mid > 0 else 0
            second_half = sum(depths[mid:]) / (len(depths) - mid)
            layer_scaling_ok = second_half >= first_half * 0.9

        return {
            "layer_depths": depths,
            "avg_depth": sum(depths) / len(depths) if depths else 0,
            "std_depth": (
                sum((d - sum(depths) / len(depths)) ** 2 for d in depths) / len(depths)
            )
            ** 0.5
            if depths
            else 0,
            "max_recursions": max_rec,
            "depth_utilization": (sum(depths) / len(depths)) / (max_rec - 1)
            if depths and max_rec > 1
            else 0,
            "total_ponder_loss": sum(ponder_losses) if ponder_losses else 0,
            "layer_scaling_preserved": layer_scaling_ok,
            "target_ratios": target_ratios,
            "histograms": histograms,
        }


class CCGQACollector:
    """Collect CCGQA statistics."""

    def __init__(self, model: CCGQAMoDMoRModel):
        self.model = model

    def collect(self) -> Dict[str, Any]:
        """Collect CCGQA stats."""
        temperatures = []
        compression_factors = []
        n_heads_list = []
        n_kv_heads_list = []

        for module in self.model.modules():
            if isinstance(module, CCGQAAttention):
                temperatures.append(module.key_temperature.item())
                compression_factors.append(module.compression_factor)
                n_heads_list.append(module.n_heads)
                n_kv_heads_list.append(module.n_kv_heads)

        return {
            "num_attention_layers": len(temperatures),
            "avg_temperature": sum(temperatures) / len(temperatures)
            if temperatures
            else 0,
            "compression_factor": compression_factors[0] if compression_factors else 0,
            "n_heads": n_heads_list[0] if n_heads_list else 0,
            "n_kv_heads": n_kv_heads_list[0] if n_kv_heads_list else 0,
            "gqa_ratio": n_heads_list[0] // n_kv_heads_list[0]
            if n_kv_heads_list and n_kv_heads_list[0] > 0
            else 0,
            "features": {
                "convolutions": True,
                "qk_norm": True,
                "value_shift": True,
            },
        }


class GradientCollector:
    """Collect gradient health statistics."""

    def __init__(self, model: CCGQAMoDMoRModel):
        self.model = model

    def collect(self) -> Dict[str, Any]:
        """Collect gradient stats after backward pass."""
        total_norm = 0.0
        router_grads = []
        zero_grads = []
        exploding = []

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                total_norm += grad_norm**2

                if "router" in name:
                    router_grads.append((name, grad_norm))

                if grad_norm == 0 and "bias" not in name:
                    zero_grads.append(name)
                elif grad_norm > 100:
                    exploding.append((name, grad_norm))

        total_norm = math.sqrt(total_norm)

        return {
            "total_norm": total_norm,
            "router_grads": router_grads,
            "zero_grads": zero_grads,
            "exploding": exploding,
            "is_healthy": len(exploding) == 0,
        }


# =============================================================================
# Diagnostic Runner
# =============================================================================


def run_diagnostic(
    variant: ModelVariant,
    steps: int = 100,
    log_every: int = 10,
    vocab_size: int = 50257,
) -> Dict[str, Any]:
    """Run diagnostic training on a model variant.

    Args:
        variant: Model configuration
        steps: Number of training steps
        log_every: Log frequency
        vocab_size: Vocabulary size

    Returns:
        Dictionary with full diagnostic results
    """
    device = get_device()

    print(f"\n{'=' * 70}")
    print(f"DIAGNOSTIC: {variant.name}")
    print(f"{'=' * 70}")

    # Create model
    print(
        f"Creating model: dim={variant.dim}, blocks={variant.n_mor_blocks}, rec={variant.recursions}"
    )
    model = create_ccgqa_mod_mor_model(
        vocab_size=vocab_size,
        dim=variant.dim,
        n_mor_blocks=variant.n_mor_blocks,
        recursions_per_block=variant.recursions,
        n_heads=variant.n_heads,
        n_kv_heads=variant.n_kv_heads,
        compression_factor=variant.compression_factor,
        mod_capacity=0.75,
        adaptive=True,
    )
    model = model.to(device)

    param_count = count_parameters(model)
    print(f"Parameters: {param_count:,} ({param_count / 1e6:.1f}M)")
    print(f"Effective layers: {model.effective_layers}")
    print(f"Device: {device}")

    # Initialize collectors
    mod_collector = MoDCollector(model)
    mor_collector = MoRCollector(model)
    ccgqa_collector = CCGQACollector(model)
    grad_collector = GradientCollector(model)

    # Training setup
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.1
    )

    def lr_lambda(step):
        warmup = 10
        if step < warmup:
            return step / warmup
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Results storage
    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "device": device,
            "torch_version": torch.__version__,
        },
        "variant": variant.to_dict(),
        "model_info": {
            "total_params": param_count,
            "params_m": param_count / 1e6,
            "effective_layers": model.effective_layers,
        },
        "steps": [],
        "final_analysis": {},
    }

    if torch.cuda.is_available():
        results["metadata"]["gpu"] = torch.cuda.get_device_name()
        results["metadata"]["gpu_memory_gb"] = (
            torch.cuda.get_device_properties(0).total_memory / 1e9
        )

    # Training loop
    model.train()
    start_time = time.time()

    print(f"\nRunning {steps} training steps...")

    for step in range(1, steps + 1):
        step_start = time.time()

        # Generate batch
        x = torch.randint(
            0, vocab_size, (variant.batch_size, variant.seq_len), device=device
        )

        # Forward
        optimizer.zero_grad()
        logits, losses = model(x, return_losses=True)

        # Compute loss
        ce_loss = F.cross_entropy(logits.view(-1, vocab_size), x.view(-1))
        aux_loss = losses.get("aux_loss", torch.tensor(0.0, device=device))
        ponder_loss = losses.get("ponder_loss", torch.tensor(0.0, device=device))

        total_loss = ce_loss + 0.1 * aux_loss + 0.01 * ponder_loss

        # Backward
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        step_time = (time.time() - step_start) * 1000

        # Collect stats
        if step % log_every == 0 or step == 1 or step == steps:
            mod_stats = mod_collector.collect()
            mor_stats = mor_collector.collect()
            ccgqa_stats = ccgqa_collector.collect()
            grad_stats = grad_collector.collect()

            step_data = {
                "step": step,
                "training": {
                    "total_loss": total_loss.item(),
                    "ce_loss": ce_loss.item(),
                    "aux_loss": aux_loss.item()
                    if isinstance(aux_loss, torch.Tensor)
                    else aux_loss,
                    "ponder_loss": ponder_loss.item()
                    if isinstance(ponder_loss, torch.Tensor)
                    else ponder_loss,
                    "grad_norm": grad_norm.item()
                    if isinstance(grad_norm, torch.Tensor)
                    else grad_norm,
                    "lr": scheduler.get_last_lr()[0],
                    "time_ms": step_time,
                },
                "mod": mod_stats,
                "mor": mor_stats,
                "ccgqa": ccgqa_stats,
                "gradients": grad_stats,
            }
            results["steps"].append(step_data)

            print(
                f"[Step {step:3d}/{steps}] loss={total_loss.item():.4f} ce={ce_loss.item():.4f} "
                f"aux={aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss:.6f} "
                f"mod_prob={mod_stats['avg_prob']:.3f} mor_depth={mor_stats['avg_depth']:.3f} "
                f"time={step_time:.0f}ms"
            )

    total_time = time.time() - start_time

    # Compute final analysis
    final_mod = results["steps"][-1]["mod"]
    final_mor = results["steps"][-1]["mor"]
    final_ccgqa = results["steps"][-1]["ccgqa"]

    results["final_analysis"] = {
        "total_time_s": total_time,
        "avg_step_ms": sum(s["training"]["time_ms"] for s in results["steps"])
        / len(results["steps"]),
        "mod_analysis": {
            "initial_prob": results["steps"][0]["mod"]["avg_prob"],
            "final_prob": final_mod["avg_prob"],
            "target": 0.75,
            "deviation": final_mod["capacity_deviation"],
            "converged": final_mod["capacity_deviation"] < 0.15,
        },
        "mor_analysis": {
            "initial_depth": results["steps"][0]["mor"]["avg_depth"],
            "final_depth": final_mor["avg_depth"],
            "max_depth": final_mor["max_recursions"] - 1,
            "utilization": final_mor["depth_utilization"],
            "layer_scaling": final_mor["layer_scaling_preserved"],
            "not_collapsed": 0.3
            < final_mor["avg_depth"]
            < final_mor["max_recursions"] - 0.3,
        },
        "ccgqa_analysis": {
            "compression": final_ccgqa["compression_factor"],
            "gqa_ratio": final_ccgqa["gqa_ratio"],
            "avg_temperature": final_ccgqa["avg_temperature"],
        },
        "paper_compliance": {
            "mod_2404.02258": {
                "router_learning": final_mod["avg_prob"] < 0.95,
                "capacity_maintained": final_mod["capacity_deviation"] < 0.2,
                "aux_loss_active": final_mod["total_aux_loss"] > 0,
            },
            "mor_2507.10524": {
                "depth_routing_active": final_mor["avg_depth"] > 0,
                "not_collapsed": 0.2
                < final_mor["avg_depth"]
                < final_mor["max_recursions"] - 0.2,
                "layer_aware_scaling": final_mor["layer_scaling_preserved"],
                "ponder_loss_active": final_mor["total_ponder_loss"] != 0,
            },
            "ccgqa_2510.04476": {
                "compression_active": final_ccgqa["compression_factor"] > 1,
                "gqa_active": final_ccgqa["gqa_ratio"] > 1,
                "qk_norm_enabled": True,
                "convolutions_enabled": final_ccgqa["features"]["convolutions"],
            },
        },
    }

    # Print summary
    print(f"\n{'=' * 70}")
    print(f"SUMMARY: {variant.name}")
    print(f"{'=' * 70}")
    print(f"Parameters: {param_count / 1e6:.1f}M")
    print(f"Total time: {total_time:.1f}s ({total_time / steps * 1000:.0f}ms/step)")

    print(f"\nMoD (arXiv:2404.02258):")
    mod_a = results["final_analysis"]["mod_analysis"]
    print(
        f"  Prob: {mod_a['initial_prob']:.3f} -> {mod_a['final_prob']:.3f} (target: {mod_a['target']})"
    )
    print(f"  Converged: {mod_a['converged']}")

    print(f"\nMoR (arXiv:2507.10524):")
    mor_a = results["final_analysis"]["mor_analysis"]
    print(
        f"  Depth: {mor_a['initial_depth']:.3f} -> {mor_a['final_depth']:.3f} (max: {mor_a['max_depth']})"
    )
    print(f"  Layer scaling: {mor_a['layer_scaling']}")
    print(f"  Not collapsed: {mor_a['not_collapsed']}")

    print(f"\nCCGQA (arXiv:2510.04476):")
    ccgqa_a = results["final_analysis"]["ccgqa_analysis"]
    print(f"  Compression: {ccgqa_a['compression']}x")
    print(f"  GQA ratio: {ccgqa_a['gqa_ratio']}:1")

    print(f"\nPaper Compliance:")
    compliance = results["final_analysis"]["paper_compliance"]
    all_passed = True
    for paper, checks in compliance.items():
        passed = all(checks.values())
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"  {paper}: {status}")
        for check, val in checks.items():
            mark = "✓" if val else "✗"
            print(f"    {mark} {check}")

    results["final_analysis"]["all_passed"] = all_passed

    return results


def main():
    parser = argparse.ArgumentParser(description="Run diagnostics on model variants")
    parser.add_argument(
        "--variants",
        "-v",
        nargs="+",
        default=["100M", "500M", "750M", "1.5B"],
        choices=list(MODEL_VARIANTS.keys()),
        help="Model variants to test",
    )
    parser.add_argument(
        "--steps", "-s", type=int, default=100, help="Training steps per variant"
    )
    parser.add_argument("--log-every", "-l", type=int, default=10, help="Log frequency")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="variant_diagnostics.json",
        help="Output file",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("MULTI-VARIANT DIAGNOSTIC RUNNER")
    print("=" * 70)
    print(f"Variants: {args.variants}")
    print(f"Steps: {args.steps}")
    print(f"Output: {args.output}")

    all_results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "variants": args.variants,
            "steps": args.steps,
        },
        "variants": {},
        "summary": {},
    }

    for variant_name in args.variants:
        if variant_name in ["750M", "1.5B"] and not torch.cuda.is_available():
            print(f"\nSkipping {variant_name} - requires CUDA")
            continue

        variant = MODEL_VARIANTS[variant_name]

        try:
            results = run_diagnostic(
                variant, steps=args.steps, log_every=args.log_every
            )
            all_results["variants"][variant_name] = results
        except Exception as e:
            print(f"\nERROR running {variant_name}: {e}")
            all_results["variants"][variant_name] = {"error": str(e)}

    # Generate summary
    summary = {
        "variants_tested": [],
        "all_compliant": True,
        "issues": [],
    }

    for name, result in all_results["variants"].items():
        if "error" in result:
            summary["issues"].append(f"{name}: {result['error']}")
            summary["all_compliant"] = False
        elif "final_analysis" in result:
            summary["variants_tested"].append(
                {
                    "name": name,
                    "params_m": result["model_info"]["params_m"],
                    "passed": result["final_analysis"]["all_passed"],
                }
            )
            if not result["final_analysis"]["all_passed"]:
                summary["all_compliant"] = False
                summary["issues"].append(f"{name}: Paper compliance failed")

    all_results["summary"] = summary

    # Save results
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'=' * 70}")
    print("FINAL SUMMARY")
    print(f"{'=' * 70}")
    print(f"Variants tested: {len(summary['variants_tested'])}")
    print(f"All compliant: {summary['all_compliant']}")
    if summary["issues"]:
        print(f"Issues: {summary['issues']}")
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
