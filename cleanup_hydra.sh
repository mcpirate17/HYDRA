#!/bin/bash
# HYDRA Project Cleanup Script
# Generated: 2026-01-05
# Updated: Added phases 7-10 (benchmark consolidation)
# 
# This script consolidates logging outputs and removes noise files.
# Review before running!

set -e
cd /home/tim/Projects/LLM/HYDRA

echo "=== HYDRA Cleanup Script ==="
echo ""

# ============================================================
# PHASE 1: Root-level garbage
# ============================================================
echo "Phase 1: Removing root-level noise files..."

# Stray file (likely from a shell mishap)
[ -f "4}" ] && rm -v "4}"

# nohup.out duplicates log files
[ -f "nohup.out" ] && rm -v "nohup.out"

# One-time install/recovery scripts
[ -f "venv_install.log" ] && rm -v "venv_install.log"
[ -f "recover_venv_complete.sh" ] && rm -v "recover_venv_complete.sh"

# Package file (should not be in repo)
[ -f "cuda-keyring_1.1-1_all.deb" ] && rm -v "cuda-keyring_1.1-1_all.deb"

# Stray results file (should be in reports/)
[ -f "scaling_analysis_results.json" ] && mv -v "scaling_analysis_results.json" "reports/"

echo ""

# ============================================================
# PHASE 2: Move checkpoint diagnostics to diagnostics/output/
# ============================================================
echo "Phase 2: Consolidating diagnostics..."

mkdir -p diagnostics/output

# Move diagnostics_*.json from checkpoints/ to diagnostics/output/
for f in checkpoints/diagnostics_*.json; do
    [ -f "$f" ] && mv -v "$f" diagnostics/output/
done

echo ""

# ============================================================
# PHASE 3: Clean up nested benchmark results (keep canonical only)
# ============================================================
echo "Phase 3: Cleaning nested benchmark results..."

# CCGQA: Keep only canonical files, remove timestamped duplicates
CCGQA_DOCS="hydra/attention/backends/ccgqa/docs"
if [ -d "$CCGQA_DOCS" ]; then
    # Remove timestamped benchmark results (keep profile_results.json)
    for f in "$CCGQA_DOCS"/benchmark_results_*.json; do
        [ -f "$f" ] && rm -v "$f"
    done
fi

echo ""

# ============================================================
# PHASE 4: Clean up logs/ directory
# ============================================================
echo "Phase 4: Cleaning logs directory..."

# Remove empty log files
find logs/ -name "*.log" -size 0 -delete -print 2>/dev/null || true

# Remove debug logs (keep training logs)
for f in logs/training_debug_*.log; do
    [ -f "$f" ] && rm -v "$f"
done

echo ""

# ============================================================
# PHASE 5: Checkpoint cleanup (OPTIONAL - uncomment to enable)
# ============================================================
echo "Phase 5: Checkpoint cleanup (showing what WOULD be deleted)..."
echo "WARNING: This would free ~75GB but is commented out for safety!"
echo ""

# These are intermediate checkpoints that can be deleted if you have final/best:
# Uncomment the rm commands to actually delete

echo "Would delete (uncomment to enable):"
# Intermediate 500M checkpoints (keep 70000, 95000, final)
for step in 89000 89500 90000 90500 91000 91500 92000 92262 94500; do
    f="checkpoints/hydra_500m_step_${step}.pt"
    [ -f "$f" ] && echo "  $f (8.3GB)"
    # [ -f "$f" ] && rm -v "$f"  # UNCOMMENT TO DELETE
done

echo ""

# ============================================================
# PHASE 6: Run the reports cleanup (from previous session)
# ============================================================
echo "Phase 6: Reports cleanup..."

if [ -f "reports/cleanup_reports.sh" ]; then
    echo "Running reports/cleanup_reports.sh..."
    chmod +x reports/cleanup_reports.sh
    cd reports && ./cleanup_reports.sh && cd ..
else
    echo "reports/cleanup_reports.sh not found, skipping"
fi

echo ""

# ============================================================
# PHASE 7: Remove duplicate test files
# ============================================================
echo "Phase 7: Removing duplicate test files..."

# Duplicate pytest file (keep the one in tests/)
if [ -f "diagnostics/mod_mor_routing_healthcheck.py" ]; then
    rm -v "diagnostics/mod_mor_routing_healthcheck.py"
    echo "  (kept tests/test_mod_mor_routing_healthcheck.py)"
fi

echo ""

# ============================================================
# PHASE 8: Remove vendored test_small_transformer
# ============================================================
echo "Phase 8: Removing vendored lightning_attn3 test (fix integrated)..."

if [ -d "hydra/attention/backends/lightning_attn3/test_small_transformer" ]; then
    rm -rf "hydra/attention/backends/lightning_attn3/test_small_transformer"
    echo "  Removed test_small_transformer/ (~50KB duplicate ops/)"
fi

echo ""

# ============================================================
# PHASE 9: Remove build artifacts
# ============================================================
echo "Phase 9: Removing build artifacts..."

# egg-info should be gitignored and regenerated on install
if [ -d "hydra_transformer.egg-info" ]; then
    rm -rf "hydra_transformer.egg-info"
    echo "  Removed hydra_transformer.egg-info/"
fi

echo ""

# ============================================================
# PHASE 10: Consolidate CCGQA benchmarks
# ============================================================
echo "Phase 10: Consolidating CCGQA benchmarks..."

# Remove old diagnostics benchmark files (consolidated into hydra/attention/backends/ccgqa/benchmarks/)
if [ -f "diagnostics/benchmark_ccgqa.py" ]; then
    rm -v "diagnostics/benchmark_ccgqa.py"
    echo "  (replaced by diagnostics/benchmark_hydra_models.py for model benchmarks)"
fi

if [ -f "diagnostics/benchmark_ccgqa_cpu.py" ]; then
    rm -v "diagnostics/benchmark_ccgqa_cpu.py"
    echo "  (consolidated into hydra/attention/backends/ccgqa/benchmarks/)"
fi

# Remove old benchmark.py at ccgqa root (moved to benchmarks/ folder)
if [ -f "hydra/attention/backends/ccgqa/benchmark.py" ]; then
    rm -v "hydra/attention/backends/ccgqa/benchmark.py"
    echo "  (moved to hydra/attention/backends/ccgqa/benchmarks/benchmark_attention.py)"
fi

echo ""

# ============================================================
# SUMMARY
# ============================================================
echo "=== Cleanup Complete ==="
echo ""
echo "Space freed (estimated):"
echo "  - Root garbage: ~20KB"
echo "  - Nested benchmarks: ~50KB"
echo "  - Duplicate test file: ~7KB"
echo "  - Vendored test_small_transformer: ~50KB"
echo "  - Build artifacts (egg-info): ~5KB"
echo "  - Consolidated benchmarks: ~50KB"
echo "  - Empty logs: variable"
echo "  - Reports (if script ran): ~2-3MB"
echo ""
echo "To free ~75GB more, uncomment the checkpoint deletion in Phase 5"
echo ""
echo "Benchmark locations after cleanup:"
echo "  - Model benchmarks:     python -m diagnostics.benchmark_hydra_models"
echo "  - Attention benchmarks: python -m hydra.attention.backends.ccgqa.benchmarks.benchmark_attention --save --plot"
echo ""
echo "Next steps:"
echo "  1. Review changes with: git status"
echo "  2. Update .gitignore: cat GITIGNORE_ADDITIONS.txt >> .gitignore"
echo "  3. Commit: git add -A && git commit -m 'Clean up project structure'"
