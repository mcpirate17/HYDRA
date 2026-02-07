#!/usr/bin/env python3
"""
Rescale MoE expert weights to fix weight explosion.

Problem: MoE expert weights can explode when trained at too-high LR,
causing gradient norms of 3000-5000+ and requiring very low LR to train.

This script rescales expert weights back to a healthy range (matching MLP norms)
while preserving the learned direction/structure of the weights.

Usage:
    python scripts/rescale_moe_weights.py --checkpoint checkpoints/hydra_500m_step_262381.pt
    python scripts/rescale_moe_weights.py --checkpoint checkpoints/hydra_500m_step_262381.pt --target-norm 40.0
    python scripts/rescale_moe_weights.py --checkpoint checkpoints/hydra_500m_step_262381.pt --dry-run
"""

import argparse
import torch
from pathlib import Path
from datetime import datetime


def analyze_checkpoint(ckpt_path: Path) -> dict:
    """Analyze weight norms in a checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_state = ckpt.get("model", {})

    stats = {
        "moe_expert_norms": [],
        "moe_expert_names": [],
        "mlp_norms": [],
        "mlp_names": [],
    }

    for name, tensor in model_state.items():
        if tensor.dtype not in [torch.float16, torch.float32, torch.bfloat16]:
            continue

        norm = tensor.float().norm().item()

        if "moe" in name and "expert" in name and "weight" in name:
            stats["moe_expert_norms"].append(norm)
            stats["moe_expert_names"].append(name)
        elif ("mlp" in name or "ffn" in name) and "weight" in name:
            if "moe" not in name:  # Exclude MoE from MLP stats
                stats["mlp_norms"].append(norm)
                stats["mlp_names"].append(name)

    return stats, ckpt


def rescale_moe_weights(
    ckpt_path: Path,
    output_path: Path | None = None,
    target_norm: float | None = None,
    dry_run: bool = False,
    reset_optimizer: bool = True,
) -> None:
    """
    Rescale MoE expert weights to target norm.

    Args:
        ckpt_path: Path to checkpoint
        output_path: Output path (default: adds _rescaled suffix)
        target_norm: Target average norm for experts (default: match MLP avg)
        dry_run: If True, only analyze without modifying
        reset_optimizer: If True, clear optimizer state for rescaled params
    """
    print(f"Loading checkpoint: {ckpt_path}")
    stats, ckpt = analyze_checkpoint(ckpt_path)

    if not stats["moe_expert_norms"]:
        print("No MoE expert weights found in checkpoint!")
        return

    moe_avg = sum(stats["moe_expert_norms"]) / len(stats["moe_expert_norms"])
    mlp_avg = (
        sum(stats["mlp_norms"]) / len(stats["mlp_norms"])
        if stats["mlp_norms"]
        else 40.0
    )

    if target_norm is None:
        target_norm = mlp_avg

    print(f"\n{'=' * 60}")
    print("CURRENT STATE")
    print(f"{'=' * 60}")
    print(f"MoE Expert weights: {len(stats['moe_expert_norms'])} tensors")
    print(f"  Average norm: {moe_avg:.2f}")
    print(f"  Min norm: {min(stats['moe_expert_norms']):.2f}")
    print(f"  Max norm: {max(stats['moe_expert_norms']):.2f}")
    print(f"\nMLP weights: {len(stats['mlp_norms'])} tensors")
    print(f"  Average norm: {mlp_avg:.2f}")
    print(f"\nExplosion factor: {moe_avg / mlp_avg:.1f}x")

    scale_factor = target_norm / moe_avg
    print(f"\n{'=' * 60}")
    print("RESCALING PLAN")
    print(f"{'=' * 60}")
    print(f"Target norm: {target_norm:.2f}")
    print(f"Scale factor: {scale_factor:.4f} ({1/scale_factor:.1f}x reduction)")
    print(f"New average norm: {moe_avg * scale_factor:.2f}")

    if dry_run:
        print("\n[DRY RUN] No changes made.")
        return

    # Apply rescaling
    print(f"\n{'=' * 60}")
    print("APPLYING RESCALING")
    print(f"{'=' * 60}")

    model_state = ckpt["model"]
    rescaled_count = 0

    for name in stats["moe_expert_names"]:
        if name in model_state:
            old_norm = model_state[name].float().norm().item()
            model_state[name] = model_state[name] * scale_factor
            new_norm = model_state[name].float().norm().item()
            rescaled_count += 1
            print(f"  {name}: {old_norm:.2f} -> {new_norm:.2f}")

    print(f"\nRescaled {rescaled_count} weight tensors")

    # Handle optimizer state
    if reset_optimizer and "optimizer" in ckpt:
        print("\nResetting optimizer state (momentum will restart fresh)...")
        # Clear all optimizer state - safest option after rescaling
        ckpt["optimizer"]["state"] = {}
        print("  Optimizer state cleared")

    # Update checkpoint metadata
    if "rescale_history" not in ckpt:
        ckpt["rescale_history"] = []

    ckpt["rescale_history"].append(
        {
            "timestamp": datetime.now().isoformat(),
            "operation": "moe_weight_rescale",
            "scale_factor": scale_factor,
            "target_norm": target_norm,
            "original_moe_avg": moe_avg,
            "original_mlp_avg": mlp_avg,
            "tensors_rescaled": rescaled_count,
        }
    )

    # Save
    if output_path is None:
        stem = ckpt_path.stem
        if "_rescaled" not in stem:
            output_path = ckpt_path.parent / f"{stem}_moe_rescaled.pt"
        else:
            output_path = ckpt_path.parent / f"{stem}_v2.pt"

    print(f"\nSaving to: {output_path}")
    torch.save(ckpt, output_path)
    print("Done!")

    # Verify
    print(f"\n{'=' * 60}")
    print("VERIFICATION")
    print(f"{'=' * 60}")
    verify_stats, _ = analyze_checkpoint(output_path)
    new_moe_avg = sum(verify_stats["moe_expert_norms"]) / len(
        verify_stats["moe_expert_norms"]
    )
    print(f"New MoE expert average norm: {new_moe_avg:.2f}")
    print(f"Target was: {target_norm:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Rescale MoE expert weights to fix weight explosion"
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="Path to checkpoint"
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="Output path (default: add suffix)"
    )
    parser.add_argument(
        "--target-norm",
        type=float,
        default=None,
        help="Target average norm (default: match MLP)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Analyze without modifying"
    )
    parser.add_argument(
        "--keep-optimizer",
        action="store_true",
        help="Keep optimizer state (not recommended)",
    )

    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"Checkpoint not found: {args.checkpoint}")
        return 1

    rescale_moe_weights(
        ckpt_path=args.checkpoint,
        output_path=args.output,
        target_norm=args.target_norm,
        dry_run=args.dry_run,
        reset_optimizer=not args.keep_optimizer,
    )

    return 0


if __name__ == "__main__":
    exit(main())
