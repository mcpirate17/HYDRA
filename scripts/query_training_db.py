#!/usr/bin/env python3
"""
Query training database for analysis.

Usage:
    python scripts/query_training_db.py --model 500m --milestones
    python scripts/query_training_db.py --model 500m --ema --start 140000
    python scripts/query_training_db.py --model 500m --runs
    python scripts/query_training_db.py --model 500m --stats
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hydra.training.db import TrainingDB


def main():
    parser = argparse.ArgumentParser(description="Query training database")
    parser.add_argument("--model", type=str, default="500m", help="Model ID")
    parser.add_argument("--milestones", action="store_true", help="Show loss milestones")
    parser.add_argument("--ema", action="store_true", help="Show multi-scale EMA")
    parser.add_argument("--runs", action="store_true", help="Show run history")
    parser.add_argument("--stats", action="store_true", help="Show model stats")
    parser.add_argument("--start", type=int, default=0, help="Start step for EMA")
    parser.add_argument("--end", type=int, default=None, help="End step for EMA")
    parser.add_argument("--db", type=Path, default=None, help="Database path")
    args = parser.parse_args()
    
    db = TrainingDB(args.db)
    model_id = args.model
    
    if args.stats or (not args.milestones and not args.ema and not args.runs):
        stats = db.get_model_stats(model_id)
        print(f"\n=== {model_id} Stats ===")
        print(f"  Steps:      {stats.get('step_count', 0):,}")
        print(f"  Step range: {stats.get('min_step', 0):,} - {stats.get('max_step', 0):,}")
        print(f"  Best loss:  {stats.get('best_loss', 'N/A'):.4f}" if stats.get('best_loss') else "  Best loss:  N/A")
        print(f"  Avg loss:   {stats.get('avg_loss', 'N/A'):.4f}" if stats.get('avg_loss') else "  Avg loss:   N/A")
        print(f"  Runs:       {stats.get('run_count', 0)}")
        
        latest = db.get_latest_step(model_id)
        ema = db.get_latest_ema(model_id)
        print(f"\n  Latest step: {latest:,}" if latest else "  Latest step: None")
        print(f"  Latest EMA (short/med/long): {ema[0]:.4f} / {ema[1]:.4f} / {ema[2]:.4f}")
    
    if args.milestones:
        milestones = db.get_loss_milestones(model_id)
        print(f"\n=== {model_id} Loss Milestones ===")
        prev_loss = None
        for step, loss in milestones.items():
            if prev_loss:
                delta = loss - prev_loss
                pct = 100 * delta / prev_loss
                arrow = "↓" if delta < 0 else "↑"
                print(f"  {step//1000:3d}K: {loss:.4f}  ({arrow} {abs(pct):.1f}%)")
            else:
                print(f"  {step//1000:3d}K: {loss:.4f}")
            prev_loss = loss
    
    if args.ema:
        ema_data = db.get_ema_series(model_id, start_step=args.start, end_step=args.end)
        print(f"\n=== {model_id} Multi-scale EMA ===")
        print(f"  {'Step':>8}  {'Raw':>8}  {'Short':>8}  {'Medium':>8}  {'Long':>8}")
        print("  " + "-" * 48)
        # Show at most 20 rows
        raw = ema_data["raw"]
        step_size = max(1, len(raw) // 20)
        for i in range(0, len(raw), step_size):
            step, loss = raw[i]
            short = ema_data["short"][i][1]
            medium = ema_data["medium"][i][1]
            long = ema_data["long"][i][1]
            print(f"  {step:>8,}  {loss:>8.4f}  {short:>8.4f}  {medium:>8.4f}  {long:>8.4f}")
    
    if args.runs:
        runs = db.get_runs(model_id)
        print(f"\n=== {model_id} Runs ({len(runs)} total) ===")
        print(f"  {'Run ID':<30} {'Steps':>10} {'Best Loss':>10}")
        print("  " + "-" * 52)
        for r in runs[-20:]:  # Last 20
            best = r.get('best_loss')
            best_str = f"{best:.4f}" if best else "N/A"
            print(f"  {r['run_id']:<30} {r.get('end_step', 0):>10,} {best_str:>10}")


if __name__ == "__main__":
    main()
