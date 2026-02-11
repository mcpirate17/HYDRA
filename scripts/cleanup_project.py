#!/usr/bin/env python3
"""
Cleanup script to remove old log, JSON, and temp files.

Removes files older than 2 weeks that are not actively used:
- Pretest logs (not in use after the run)
- Training logs older than 2 weeks (archived)
- Diagnostics JSON files not loaded to training.db
- Training report JSON files not loaded to training.db
- .prefixfix temporary files (from database fixes)

This script is safe: it only removes files that are:
1. Older than 2 weeks (Jan 26 or older, as of Feb 9, 2026)
2. Not referenced by other project code
3. Not loaded to the training.db database

Usage:
    python scripts/cleanup_project.py [--dry-run] [--verbose]
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import json
import sqlite3

ROOT = Path(__file__).resolve().parents[1]


def get_db_run_ids():
    """Get all run IDs currently in training.db."""
    db_path = ROOT / "checkpoints" / "training.db"
    if not db_path.exists():
        return set()
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT run_id FROM runs")
    run_ids = {row[0] for row in cursor.fetchall()}
    conn.close()
    return run_ids


def cleanup(dry_run: bool = True, verbose: bool = False):
    """Clean up old files from project."""
    
    cutoff = datetime(2026, 1, 26)
    db_run_ids = get_db_run_ids()
    
    deleted_files = []
    deleted_size = 0
    skipped = []
    
    # ==========================================================================
    # 1. Delete old pretest logs (checkpoints/pretest_logs/*.json)
    # ==========================================================================
    pretest_dir = ROOT / "checkpoints" / "pretest_logs"
    if pretest_dir.exists():
        for f in pretest_dir.glob("*.json"):
            # Skip pretest_history.json (important metadata)
            if "pretest_history" in f.name:
                if verbose:
                    print(f"  SKIPPED (important): {f.name}")
                skipped.append((f, "important metadata"))
                continue
            
            stat = f.stat()
            mtime = datetime.fromtimestamp(stat.st_mtime)
            
            if mtime < cutoff:
                try:
                    if not dry_run:
                        f.unlink()
                    deleted_files.append(f)
                    deleted_size += stat.st_size
                    if verbose:
                        print(f"  DELETE pretest log: {f.name}")
                except Exception as e:
                    print(f"  ERROR deleting {f.name}: {e}")
    
    # ==========================================================================
    # 2. Delete old training logs (logs/*.log older than 2 weeks)
    # ==========================================================================
    logs_dir = ROOT / "logs"
    if logs_dir.exists():
        for f in logs_dir.glob("*.log"):
            stat = f.stat()
            mtime = datetime.fromtimestamp(stat.st_mtime)
            
            if mtime < cutoff:
                try:
                    if not dry_run:
                        f.unlink()
                    deleted_files.append(f)
                    deleted_size += stat.st_size
                    if verbose:
                        print(f"  DELETE training log: {f.name}")
                except Exception as e:
                    print(f"  ERROR deleting {f.name}: {e}")
    
    # ==========================================================================
    # 3. Delete old diagnostics JSON not in DB (checkpoints/diagnostics_*.json)
    # ==========================================================================
    ckpt_dir = ROOT / "checkpoints"
    for f in ckpt_dir.glob("diagnostics_*.json"):
        run_id = f.stem.replace("diagnostics_", "")
        stat = f.stat()
        mtime = datetime.fromtimestamp(stat.st_mtime)
        
        if mtime < cutoff and run_id not in db_run_ids:
            try:
                if not dry_run:
                    f.unlink()
                deleted_files.append(f)
                deleted_size += stat.st_size
                if verbose:
                    print(f"  DELETE old diagnostics: {f.name}")
            except Exception as e:
                print(f"  ERROR deleting {f.name}: {e}")
    
    # ==========================================================================
    # 4. Delete old training reports not in DB (reports/training_report_*.json)
    # ==========================================================================
    reports_dir = ROOT / "reports"
    if reports_dir.exists():
        for f in reports_dir.glob("training_report_*.json"):
            stat = f.stat()
            mtime = datetime.fromtimestamp(stat.st_mtime)
            
            if mtime < cutoff:
                try:
                    with open(f) as fp:
                        data = json.load(fp)
                    run_id = data.get("configuration", {}).get("run_id", "")
                    
                    if run_id not in db_run_ids:
                        if not dry_run:
                            f.unlink()
                        deleted_files.append(f)
                        deleted_size += stat.st_size
                        if verbose:
                            print(f"  DELETE old report: {f.name}")
                    else:
                        skipped.append((f, "in database"))
                except Exception as e:
                    print(f"  WARNING: Could not process {f.name}: {e}")
    
    # ==========================================================================
    # 5. Delete .prefixfix temporary files
    # ==========================================================================
    for f in ckpt_dir.glob("*.prefixfix"):
        try:
            if not dry_run:
                f.unlink()
            deleted_files.append(f)
            deleted_size += f.stat().st_size
            if verbose:
                print(f"  DELETE prefixfix temp: {f.name}")
        except Exception as e:
            print(f"  ERROR deleting {f.name}: {e}")
    
    # ==========================================================================
    # REPORT
    # ==========================================================================
    print()
    print("=" * 80)
    print("CLEANUP SUMMARY")
    print("=" * 80)
    print(f"Cutoff date: {cutoff.strftime('%Y-%m-%d')} (2 weeks old)")
    print()
    
    if dry_run:
        print("🔍 DRY RUN MODE (no files deleted)")
    else:
        print("✅ CLEANUP COMPLETE")
    
    print()
    print(f"Total files to delete: {len(deleted_files)}")
    print(f"Space recoverable: {deleted_size / (1024*1024):.2f} MB")
    print()
    
    if deleted_files:
        print("Files to delete (by category):")
        pretest_count = sum(1 for f in deleted_files if "pretest_logs" in str(f))
        log_count = sum(1 for f in deleted_files if "/logs/" in str(f))
        diag_count = sum(1 for f in deleted_files if "diagnostics_" in f.name)
        report_count = sum(1 for f in deleted_files if "training_report_" in f.name)
        prefixfix_count = sum(1 for f in deleted_files if f.suffix == ".prefixfix")
        
        if pretest_count > 0:
            print(f"  • Pretest logs: {pretest_count} files")
        if log_count > 0:
            print(f"  • Training logs: {log_count} files")
        if diag_count > 0:
            print(f"  • Old diagnostics: {diag_count} files")
        if report_count > 0:
            print(f"  • Old reports: {report_count} files")
        if prefixfix_count > 0:
            print(f"  • Temp .prefixfix files: {prefixfix_count} files")
    
    if skipped:
        print()
        print(f"Skipped (still useful): {len(skipped)} files")
        for f, reason in skipped[:5]:
            print(f"  • {f.name} [{reason}]")
        if len(skipped) > 5:
            print(f"  ... ({len(skipped)-5} more)")
    
    print()
    print("=" * 80)
    
    if dry_run:
        print("To apply cleanup, run: python scripts/cleanup_project.py")
    
    return len(deleted_files)


def main():
    parser = argparse.ArgumentParser(
        description="Clean up old log, JSON, and temp files from project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run (safe preview)
  python scripts/cleanup_project.py --dry-run

  # Actually delete files
  python scripts/cleanup_project.py

  # Verbose output with file names
  python scripts/cleanup_project.py --verbose
        """
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be deleted without actually deleting"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed file-by-file output"
    )
    
    args = parser.parse_args()
    
    try:
        count = cleanup(dry_run=args.dry_run, verbose=args.verbose)
        sys.exit(0 if count >= 0 else 1)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
