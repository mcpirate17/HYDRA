#!/usr/bin/env python3
"""Fix step counters corrupted during test run.

The test run reset step counters to ~150 instead of continuing from 282150.
This script fixes:
1. Checkpoint step counters
2. Diagnostics JSON files
3. training.db entries

Usage:
    python scripts/fix_corrupted_run.py --dry-run  # Preview changes
    python scripts/fix_corrupted_run.py            # Apply fixes
"""

import argparse
import json
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path

import torch

# The offset to add to corrupted steps
# True step = corrupted step + STEP_OFFSET
STEP_OFFSET = 282000

# Corrupted step range (steps below this threshold are from the corrupted run)
CORRUPTED_THRESHOLD = 10000


def fix_checkpoint(ckpt_path: Path, dry_run: bool = False) -> dict:
    """Fix step counters in a checkpoint."""
    if not ckpt_path.exists():
        return {"skipped": True, "reason": "not found"}

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    old_step = ckpt.get("step", 0)

    # Only fix if step is in corrupted range
    if old_step >= CORRUPTED_THRESHOLD:
        return {"skipped": True, "reason": f"step {old_step} not in corrupted range"}

    new_step = old_step + STEP_OFFSET
    changes = {"step": (old_step, new_step)}

    # Fix metrics best_loss_step
    if "metrics" in ckpt and "best_loss_step" in ckpt["metrics"]:
        old_best = ckpt["metrics"]["best_loss_step"]
        if old_best < CORRUPTED_THRESHOLD:
            new_best = old_best + STEP_OFFSET
            changes["best_loss_step"] = (old_best, new_best)
            if not dry_run:
                ckpt["metrics"]["best_loss_step"] = new_best

    # Fix moe global_step counters
    moe_fixes = []
    for key in ckpt.get("model", {}).keys():
        if "_global_step" in key:
            old_val = ckpt["model"][key].item()
            if old_val < CORRUPTED_THRESHOLD:
                new_val = old_val + STEP_OFFSET
                moe_fixes.append((key, old_val, new_val))
                if not dry_run:
                    ckpt["model"][key] = torch.tensor(
                        new_val, dtype=ckpt["model"][key].dtype
                    )
    if moe_fixes:
        changes["moe_global_steps"] = moe_fixes

    if not dry_run:
        ckpt["step"] = new_step
        # Create backup
        backup_path = ckpt_path.with_suffix(".pt.prefixfix")
        if not backup_path.exists():
            shutil.copy(ckpt_path, backup_path)
        torch.save(ckpt, ckpt_path)

    return {"fixed": True, "changes": changes}


def fix_diagnostics_json(json_path: Path, dry_run: bool = False) -> dict:
    """Fix step values in diagnostics JSON."""
    if not json_path.exists():
        return {"skipped": True, "reason": "not found"}

    with open(json_path) as f:
        data = json.load(f)

    if not isinstance(data, list) or not data:
        return {"skipped": True, "reason": "empty or wrong format"}

    first_step = data[0].get("step", 0)
    last_step = data[-1].get("step", 0)

    # Only fix if steps are in corrupted range
    if first_step >= CORRUPTED_THRESHOLD:
        return {"skipped": True, "reason": f"steps {first_step}-{last_step} not in corrupted range"}

    fixed_count = 0
    for entry in data:
        if "step" in entry and entry["step"] < CORRUPTED_THRESHOLD:
            if not dry_run:
                entry["step"] = entry["step"] + STEP_OFFSET
            fixed_count += 1

    if not dry_run and fixed_count > 0:
        # Backup
        backup_path = json_path.with_suffix(".json.prefixfix")
        if not backup_path.exists():
            shutil.copy(json_path, backup_path)
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)

    return {
        "fixed": True,
        "entries_fixed": fixed_count,
        "new_range": (first_step + STEP_OFFSET, last_step + STEP_OFFSET),
    }


def fix_training_db(db_path: Path, dry_run: bool = False) -> dict:
    """Fix step values in training.db."""
    if not db_path.exists():
        return {"skipped": True, "reason": "not found"}

    # Backup first
    if not dry_run:
        backup_path = db_path.with_suffix(".db.prefixfix")
        if not backup_path.exists():
            shutil.copy(db_path, backup_path)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    results = {}

    # Fix steps table
    cur.execute(f"SELECT COUNT(*) FROM steps WHERE step < {CORRUPTED_THRESHOLD}")
    count = cur.fetchone()[0]
    if count > 0:
        if not dry_run:
            cur.execute(
                f"UPDATE steps SET step = step + {STEP_OFFSET} WHERE step < {CORRUPTED_THRESHOLD}"
            )
        results["steps"] = count

    # Fix routing_mod table
    cur.execute(f"SELECT COUNT(*) FROM routing_mod WHERE step < {CORRUPTED_THRESHOLD}")
    count = cur.fetchone()[0]
    if count > 0:
        if not dry_run:
            cur.execute(
                f"UPDATE routing_mod SET step = step + {STEP_OFFSET} WHERE step < {CORRUPTED_THRESHOLD}"
            )
        results["routing_mod"] = count

    # Fix routing_mor table
    cur.execute(f"SELECT COUNT(*) FROM routing_mor WHERE step < {CORRUPTED_THRESHOLD}")
    count = cur.fetchone()[0]
    if count > 0:
        if not dry_run:
            cur.execute(
                f"UPDATE routing_mor SET step = step + {STEP_OFFSET} WHERE step < {CORRUPTED_THRESHOLD}"
            )
        results["routing_mor"] = count

    # Fix routing_moe table
    cur.execute(f"SELECT COUNT(*) FROM routing_moe WHERE step < {CORRUPTED_THRESHOLD}")
    count = cur.fetchone()[0]
    if count > 0:
        if not dry_run:
            cur.execute(
                f"UPDATE routing_moe SET step = step + {STEP_OFFSET} WHERE step < {CORRUPTED_THRESHOLD}"
            )
        results["routing_moe"] = count

    # Fix adaptive_lr table
    cur.execute(f"SELECT COUNT(*) FROM adaptive_lr WHERE step < {CORRUPTED_THRESHOLD}")
    count = cur.fetchone()[0]
    if count > 0:
        if not dry_run:
            cur.execute(
                f"UPDATE adaptive_lr SET step = step + {STEP_OFFSET} WHERE step < {CORRUPTED_THRESHOLD}"
            )
        results["adaptive_lr"] = count

    # Fix runs table - start_step, end_step, best_loss_step
    cur.execute(
        f"SELECT run_id, start_step, end_step, best_loss_step FROM runs WHERE end_step < {CORRUPTED_THRESHOLD}"
    )
    runs_to_fix = cur.fetchall()
    if runs_to_fix:
        for run_id, start_step, end_step, best_loss_step in runs_to_fix:
            new_start = start_step + STEP_OFFSET if start_step < CORRUPTED_THRESHOLD else start_step
            new_end = end_step + STEP_OFFSET if end_step < CORRUPTED_THRESHOLD else end_step
            new_best = best_loss_step + STEP_OFFSET if best_loss_step and best_loss_step < CORRUPTED_THRESHOLD else best_loss_step
            if not dry_run:
                cur.execute(
                    "UPDATE runs SET start_step = ?, end_step = ?, best_loss_step = ? WHERE run_id = ?",
                    (new_start, new_end, new_best, run_id),
                )
        results["runs"] = len(runs_to_fix)

    if not dry_run:
        conn.commit()
    conn.close()

    return {"fixed": True, "tables_updated": results}


def main():
    parser = argparse.ArgumentParser(description="Fix corrupted step counters")
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview changes without applying"
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=Path("/home/tim/Projects/LLM/HYDRA/checkpoints"),
        help="Checkpoints directory",
    )
    args = parser.parse_args()

    print(f"Step offset: {STEP_OFFSET}")
    print(f"Corrupted threshold: < {CORRUPTED_THRESHOLD}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'APPLYING FIXES'}\n")

    # Fix checkpoints
    print("=" * 60)
    print("CHECKPOINTS")
    print("=" * 60)

    for ckpt_name in ["hydra_500m_final.pt", "hydra_500m_step_4780.pt",
                       "hydra_500m_step_4500.pt", "hydra_500m_step_4000.pt"]:
        ckpt_path = args.checkpoints_dir / ckpt_name
        print(f"\n{ckpt_name}:")
        result = fix_checkpoint(ckpt_path, args.dry_run)
        if result.get("skipped"):
            print(f"  Skipped: {result['reason']}")
        else:
            for key, val in result["changes"].items():
                if key == "moe_global_steps":
                    print(f"  {len(val)} MoE global_step counters: +{STEP_OFFSET}")
                else:
                    old, new = val
                    print(f"  {key}: {old} -> {new}")

    # Fix diagnostics JSON
    print("\n" + "=" * 60)
    print("DIAGNOSTICS JSON")
    print("=" * 60)

    json_files = list(args.checkpoints_dir.glob("diagnostics_500m_*.json"))
    for json_path in sorted(json_files):
        print(f"\n{json_path.name}:")
        result = fix_diagnostics_json(json_path, args.dry_run)
        if result.get("skipped"):
            print(f"  Skipped: {result['reason']}")
        else:
            print(f"  Fixed {result['entries_fixed']} entries")
            print(f"  New range: {result['new_range'][0]} - {result['new_range'][1]}")

    # Fix training.db
    print("\n" + "=" * 60)
    print("TRAINING DATABASE")
    print("=" * 60)

    db_path = args.checkpoints_dir / "training.db"
    print(f"\n{db_path.name}:")
    result = fix_training_db(db_path, args.dry_run)
    if result.get("skipped"):
        print(f"  Skipped: {result['reason']}")
    else:
        for table, count in result["tables_updated"].items():
            print(f"  {table}: {count} rows updated")

    if args.dry_run:
        print("\n[DRY RUN] No changes were made. Run without --dry-run to apply.")
    else:
        print("\nAll fixes applied. Backups created with .prefixfix extension.")
        print("\nNext: Regenerate reports with:")
        print("  python scripts/generate_training_report.py --model-id 500m")


if __name__ == "__main__":
    main()
