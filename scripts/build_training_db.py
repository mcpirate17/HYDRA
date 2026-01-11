#!/usr/bin/env python3
"""
Build training database from existing JSON files.

Loads all diagnostics and training reports into SQLite for fast querying.

Usage:
    python scripts/build_training_db.py [--db-path PATH] [--model-id ID]
    
Examples:
    # Build from all 500m data (default)
    python scripts/build_training_db.py
    
    # Specify custom DB location
    python scripts/build_training_db.py --db-path /path/to/training.db
    
    # Build for specific model
    python scripts/build_training_db.py --model-id 626m
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hydra.training.db import TrainingDB


def build_db(db_path: Path = None, model_id: str = "500m", verbose: bool = True) -> None:
    """Build training database from existing JSON files."""
    
    checkpoints_dir = ROOT / "checkpoints"
    reports_dir = ROOT / "reports"
    
    db = TrainingDB(db_path)
    
    if verbose:
        print("=" * 60)
        print("Building Training Database")
        print("=" * 60)
        print(f"Database: {db.db_path}")
        print(f"Model ID: {model_id}")
    
    # ==========================================================================
    # Load diagnostics files
    # ==========================================================================
    diag_pattern = str(checkpoints_dir / f"diagnostics_{model_id}_*.json")
    diag_files = sorted(glob.glob(diag_pattern))
    
    if verbose:
        print(f"\n📊 Loading diagnostics files...")
        print(f"   Pattern: {diag_pattern}")
        print(f"   Found: {len(diag_files)} files")
    
    total_steps = 0
    for f in diag_files:
        path = Path(f)
        run_id = path.stem.replace("diagnostics_", "")
        count = db.load_diagnostics_json(path, model_id=model_id, run_id=run_id)
        total_steps += count
        if verbose:
            print(f"   ✓ {path.name}: {count} records")
    
    if verbose:
        print(f"   Total step records: {total_steps}")
    
    # ==========================================================================
    # Load training reports
    # ==========================================================================
    report_pattern = str(reports_dir / "training_report_*.json")
    report_files = sorted(glob.glob(report_pattern))
    
    if verbose:
        print(f"\n📋 Loading training reports...")
        print(f"   Pattern: {report_pattern}")
        print(f"   Found: {len(report_files)} files")
    
    runs_loaded = 0
    for f in report_files:
        path = Path(f)
        # Only load reports for this model
        import json
        with open(path) as fp:
            data = json.load(fp)
        run_id = data.get("configuration", {}).get("run_id", "")
        if not run_id.startswith(f"{model_id}_"):
            continue
        
        if db.load_training_report(path, model_id=model_id):
            runs_loaded += 1
            if verbose:
                print(f"   ✓ {path.name}")
    
    if verbose:
        print(f"   Runs loaded: {runs_loaded}")
    
    # ==========================================================================
    # Summary
    # ==========================================================================
    stats = db.get_model_stats(model_id)
    
    if verbose:
        print("\n" + "=" * 60)
        print("Database Summary")
        print("=" * 60)
        print(f"  Model:      {model_id}")
        print(f"  Steps:      {stats.get('step_count', 0):,}")
        print(f"  Step range: {stats.get('min_step', 0):,} - {stats.get('max_step', 0):,}")
        print(f"  Best loss:  {stats.get('best_loss', 'N/A')}")
        print(f"  Runs:       {stats.get('run_count', 0)}")
        print(f"  DB size:    {db.db_path.stat().st_size / 1024:.1f} KB")
        print("=" * 60)
    
    return db


def main():
    parser = argparse.ArgumentParser(description="Build training database from JSON files")
    parser.add_argument("--db-path", type=Path, default=None, help="Database path")
    parser.add_argument("--model-id", type=str, default="500m", help="Model ID to load")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")
    args = parser.parse_args()
    
    build_db(
        db_path=args.db_path,
        model_id=args.model_id,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
