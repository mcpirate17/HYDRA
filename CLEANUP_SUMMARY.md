# HYDRA Project Cleanup Summary

## Problem: 7 Output Locations

| Location | Purpose | Status |
|----------|---------|--------|
| `/logs/` | Training log files | ✅ Keep (gitignored) |
| `/reports/` | Training reports JSON | ✅ Keep (gitignored) |
| `/checkpoints/*.pt` | Model weights | ✅ Keep (gitignored) |
| `/checkpoints/diagnostics_*.json` | Step metrics | ❌ **Move to diagnostics/output/** |
| `/nohup.out` | Console output | ❌ **Delete** (duplicate of logs) |
| Root-level artifacts | Stray files | ❌ **Delete** |
| Nested `docs/benchmark_*.json` | Duplicates | ❌ **Delete** |

## Files Created

1. **`cleanup_hydra.sh`** - Run this to clean up the project:
   ```bash
   chmod +x cleanup_hydra.sh
   ./cleanup_hydra.sh
   ```

2. **`GITIGNORE_ADDITIONS.txt`** - Append to `.gitignore`:
   ```bash
   cat GITIGNORE_ADDITIONS.txt >> .gitignore
   ```

3. **`CHECKPOINTING_PATCH.py`** - Manual code fix for diagnostics location
   - Edit `hydra/training/checkpointing.py` following the instructions

## After Cleanup: New Structure

```
HYDRA/
├── checkpoints/           # Only .pt files (model weights)
├── diagnostics/
│   ├── output/            # NEW: diagnostics_*.json files go here
│   ├── *.py               # Diagnostic scripts (unchanged)
│   └── README.md
├── logs/                  # Training logs (unchanged)
├── reports/               # Training reports (unchanged)
└── ...
```

## Checkpoint Cleanup (Optional)

You have 12 × 8.3GB = **~100GB** of 500M checkpoints. To free ~75GB:

**Keep:**
- `hydra_500m_step_70000.pt` (milestone)
- `hydra_500m_step_95000.pt` (latest)
- `hydra_500m_final.pt` (final)

**Delete (uncomment in cleanup_hydra.sh Phase 5):**
- All others: 89000, 89500, 90000, 90500, 91000, 91500, 92000, 92262, 94500

## Git Preparation

After running cleanup:
```bash
# Review changes
git status

# Stage and commit
git add -A
git commit -m "Clean up project structure: consolidate logging outputs"

# Optional: clean up the cleanup files themselves
rm cleanup_hydra.sh GITIGNORE_ADDITIONS.txt CHECKPOINTING_PATCH.py CLEANUP_SUMMARY.md
```
