#!/usr/bin/env python3
"""Fix step counter in a checkpoint that was corrupted during test runs.

Usage:
    # Set step to specific value
    python scripts/fix_checkpoint_step.py checkpoints/hydra_500m_final.pt --step 282150

    # Copy step from another checkpoint
    python scripts/fix_checkpoint_step.py checkpoints/hydra_500m_final.pt --from-checkpoint checkpoints/hydra_500m_step_282000.pt

    # Add offset to current step (e.g., if you know you trained 1000 more steps)
    python scripts/fix_checkpoint_step.py checkpoints/hydra_500m_final.pt --add-steps 1000

    # Dry run (show what would change without modifying)
    python scripts/fix_checkpoint_step.py checkpoints/hydra_500m_final.pt --step 282150 --dry-run
"""

import argparse
import shutil
from pathlib import Path

import torch


def fix_checkpoint_step(
    checkpoint_path: str,
    new_step: int | None = None,
    from_checkpoint: str | None = None,
    add_steps: int | None = None,
    dry_run: bool = False,
    no_backup: bool = False,
) -> None:
    """Fix step counter in checkpoint."""
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    old_step = ckpt.get("step", 0)
    print(f"Current step: {old_step}")

    # Determine new step value
    if new_step is not None:
        target_step = new_step
    elif from_checkpoint is not None:
        ref_path = Path(from_checkpoint)
        if not ref_path.exists():
            raise FileNotFoundError(f"Reference checkpoint not found: {ref_path}")
        print(f"Loading reference checkpoint: {ref_path}")
        ref_ckpt = torch.load(ref_path, map_location="cpu", weights_only=False)
        target_step = ref_ckpt.get("step", 0)
        print(f"Reference checkpoint step: {target_step}")
    elif add_steps is not None:
        target_step = old_step + add_steps
    else:
        raise ValueError("Must specify --step, --from-checkpoint, or --add-steps")

    print(f"Target step: {target_step}")

    if target_step == old_step:
        print("Step already correct, nothing to do.")
        return

    # Find all step-related keys in model state
    step_keys = []
    for key in ckpt.get("model", {}).keys():
        if "_global_step" in key:
            step_keys.append(key)

    print(f"\nChanges to apply:")
    print(f"  step: {old_step} -> {target_step}")
    for key in step_keys:
        old_val = ckpt["model"][key].item()
        new_val = target_step - 1  # _global_step is typically step - 1
        print(f"  {key}: {old_val} -> {new_val}")

    if dry_run:
        print("\n[DRY RUN] No changes made.")
        return

    # Create backup
    if not no_backup:
        backup_path = ckpt_path.with_suffix(".pt.backup")
        # If backup exists, add number
        i = 1
        while backup_path.exists():
            backup_path = ckpt_path.with_suffix(f".pt.backup{i}")
            i += 1
        print(f"\nCreating backup: {backup_path}")
        shutil.copy(ckpt_path, backup_path)

    # Apply fixes
    ckpt["step"] = target_step
    for key in step_keys:
        ckpt["model"][key] = torch.tensor(
            target_step - 1, dtype=ckpt["model"][key].dtype
        )

    # Save
    print(f"Saving fixed checkpoint: {ckpt_path}")
    torch.save(ckpt, ckpt_path)
    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Fix step counter in a corrupted checkpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("checkpoint", help="Path to checkpoint to fix")
    parser.add_argument("--step", type=int, help="Set step to this value")
    parser.add_argument(
        "--from-checkpoint", help="Copy step value from this checkpoint"
    )
    parser.add_argument(
        "--add-steps", type=int, help="Add this many steps to current value"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would change without modifying"
    )
    parser.add_argument(
        "--no-backup", action="store_true", help="Skip creating backup file"
    )

    args = parser.parse_args()

    # Validate that exactly one step source is provided
    sources = sum([args.step is not None, args.from_checkpoint is not None, args.add_steps is not None])
    if sources == 0:
        parser.error("Must specify one of: --step, --from-checkpoint, --add-steps")
    if sources > 1:
        parser.error("Specify only one of: --step, --from-checkpoint, --add-steps")

    fix_checkpoint_step(
        checkpoint_path=args.checkpoint,
        new_step=args.step,
        from_checkpoint=args.from_checkpoint,
        add_steps=args.add_steps,
        dry_run=args.dry_run,
        no_backup=args.no_backup,
    )


if __name__ == "__main__":
    main()
