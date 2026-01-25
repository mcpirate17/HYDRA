#!/usr/bin/env python3
"""
Training trends analysis and plotting - reads from SQLite database.

Generates plots and a comprehensive training report comparing to industry norms.

Usage:
    python scripts/plot_training_trends.py [--model 500m]
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add parent to path for hydra imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hydra.training.db import TrainingDB

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
PLOTS_DIR = REPORTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# INDUSTRY BENCHMARKS (for comparison)
# =============================================================================

# Chinchilla-optimal: ~20 tokens per parameter
# GPT-3 paper: ~300B tokens for 175B model (~1.7 tokens/param, undertrained)
# LLaMA-2: ~2T tokens for 7B model (~285 tokens/param, overtrained)
# Modern consensus: 20-100 tokens/param depending on compute budget

INDUSTRY_NORMS = {
    "tokens_per_param": {
        "chinchilla_optimal": 20,
        "typical_range": (20, 100),
        "llama2_ratio": 285,
    },
    "loss_benchmarks": {
        # Cross-entropy loss on web text at various scales
        "1B_converged": 2.5,      # ~1B params, well-trained
        "500M_converged": 2.8,    # ~500M params, well-trained
        "100M_converged": 3.5,    # ~100M params, well-trained
    },
    "throughput": {
        # Tokens/sec on single GPU (varies by hardware)
        "a100_500m": 8000,        # A100 80GB, 500M model
        "rtx4090_500m": 6000,     # RTX 4090, 500M model
        "rtx3090_500m": 4000,     # RTX 3090, 500M model
    },
    "gradient_norms": {
        "healthy_range": (0.1, 10),   # Typical healthy range
        "warning_threshold": 100,      # Might indicate instability
        "critical_threshold": 1000,    # Likely training issues
    },
    "learning_rate": {
        "typical_peak": (1e-4, 6e-4),  # For transformer LMs
        "warmup_steps": (1000, 5000),  # Typical warmup
    },
}


# =============================================================================
# DATA ACCESS
# =============================================================================

def get_step_data(db: TrainingDB, model_id: str):
    """Get all step data for plotting."""
    return db.get_steps(model_id=model_id)


def get_run_data(db: TrainingDB, model_id: str):
    """Get run summaries."""
    return db.get_runs(model_id=model_id)


def compute_total_tokens(max_step: int, model_id: str = "500m") -> int:
    """
    Compute total tokens trained from step count - NOT from summing runs.
    
    Summing runs double-counts tokens when resuming from checkpoints.
    Instead, we calculate: steps × batch × grad_accum × seq_len.
    
    For 500m model:
    - Steps 0-30K: batch=2, accum=8, seq=512 → 8,192 tokens/step
    - Steps 30K+:  batch=2, accum=8, seq=1024 → 16,384 tokens/step
    """
    if model_id == "500m":
        # Account for seq_len change at step 30K
        seq_change_step = 30_000
        tokens_per_step_early = 2 * 8 * 512   # 8,192
        tokens_per_step_late = 2 * 8 * 1024   # 16,384
        
        if max_step <= seq_change_step:
            return max_step * tokens_per_step_early
        else:
            early_tokens = seq_change_step * tokens_per_step_early
            late_tokens = (max_step - seq_change_step) * tokens_per_step_late
            return early_tokens + late_tokens
    else:
        # Default: assume current config (batch=2, accum=8, seq=1024)
        return max_step * 2 * 8 * 1024


# =============================================================================
# PLOTTING FUNCTIONS (now using DB data)
# =============================================================================

def plot_loss_over_steps(db: TrainingDB, model_id: str):
    """Loss progression over all steps with multi-scale EMA."""
    steps = get_step_data(db, model_id)
    if len(steps) < 2:
        print("Skipping loss plot (insufficient data)")
        return
    
    x = [s["step"] for s in steps]
    raw = [s["loss_total"] for s in steps]
    ema_short = [s["ema_short"] for s in steps]
    ema_medium = [s["ema_medium"] for s in steps]
    ema_long = [s["ema_long"] for s in steps]
    
    plt.figure(figsize=(14, 6))
    
    plt.plot(x, raw, alpha=0.3, linewidth=0.5, color='gray', label='Raw')
    plt.plot(x, ema_short, linewidth=1, color='blue', label='EMA Short (α=0.99)')
    plt.plot(x, ema_medium, linewidth=1.5, color='green', label='EMA Medium (α=0.999)')
    plt.plot(x, ema_long, linewidth=2, color='red', label='EMA Long (α=0.9999)')
    
    # Add industry benchmark line
    if model_id.startswith("500"):
        plt.axhline(y=INDUSTRY_NORMS["loss_benchmarks"]["500M_converged"], 
                   color='purple', linestyle='--', alpha=0.7, 
                   label=f'500M Converged Target ({INDUSTRY_NORMS["loss_benchmarks"]["500M_converged"]})')
    
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title(f'{model_id.upper()} Training Loss with Multi-Scale EMA')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    out = PLOTS_DIR / f"loss_ema_{model_id}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")


def plot_loss_components(db: TrainingDB, model_id: str):
    """CE vs auxiliary losses."""
    steps = get_step_data(db, model_id)
    if len(steps) < 2:
        print("Skipping components plot (insufficient data)")
        return
    
    x = [s["step"] for s in steps]
    ce = [s["loss_ce"] for s in steps]
    aux = [s["loss_aux"] for s in steps]
    ponder = [s["loss_ponder"] for s in steps]
    advantage = [s["loss_advantage"] for s in steps]
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # CE loss
    axes[0].plot(x, ce, linewidth=1, color='blue', label='CE Loss')
    axes[0].set_ylabel('Cross-Entropy Loss')
    axes[0].set_title('Primary Loss (CE)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Auxiliary losses
    if any(v != 0 for v in aux):
        axes[1].plot(x, aux, linewidth=1, label='Aux (Load Balance)', alpha=0.8)
    if any(v != 0 for v in ponder):
        axes[1].plot(x, ponder, linewidth=1, label='Ponder', alpha=0.8)
    if any(v != 0 for v in advantage):
        axes[1].plot(x, advantage, linewidth=1, label='Advantage', alpha=0.8)
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Auxiliary Losses')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    out = PLOTS_DIR / f"loss_components_{model_id}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")


def plot_gradient_norms(db: TrainingDB, model_id: str):
    """Gradient norm progression with health indicators."""
    steps = get_step_data(db, model_id)
    if len(steps) < 2:
        print("Skipping gradient plot (insufficient data)")
        return
    
    x = [s["step"] for s in steps]
    grad = [s["grad_norm"] for s in steps]
    grad_pre = [s["grad_norm_pre_clip"] for s in steps]
    
    plt.figure(figsize=(14, 5))
    
    plt.plot(x, grad_pre, alpha=0.5, linewidth=0.8, color='orange', label='Pre-clip')
    plt.plot(x, grad, linewidth=1, color='blue', label='Post-clip')
    
    # Health zone
    healthy_lo, healthy_hi = INDUSTRY_NORMS["gradient_norms"]["healthy_range"]
    plt.axhspan(healthy_lo, healthy_hi, alpha=0.1, color='green', label='Healthy Range')
    plt.axhline(y=INDUSTRY_NORMS["gradient_norms"]["warning_threshold"], 
               color='orange', linestyle='--', alpha=0.7, label='Warning Threshold')
    
    plt.yscale('log', base=10)
    plt.xlabel('Step')
    plt.ylabel('Gradient Norm')
    plt.title(f'{model_id.upper()} Gradient Norms')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    
    out = PLOTS_DIR / f"grad_norms_{model_id}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")


def plot_learning_rate(db: TrainingDB, model_id: str):
    """Learning rate schedule."""
    steps = get_step_data(db, model_id)
    if len(steps) < 2:
        print("Skipping LR plot (insufficient data)")
        return
    
    x = [s["step"] for s in steps]
    lr = [s["lr"] for s in steps]
    
    plt.figure(figsize=(14, 4))
    plt.plot(x, lr, linewidth=1.5, color='purple')
    
    # Industry typical range
    lo, hi = INDUSTRY_NORMS["learning_rate"]["typical_peak"]
    plt.axhspan(lo, hi, alpha=0.1, color='blue', label=f'Typical Peak LR ({lo:.0e} - {hi:.0e})')
    
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.title(f'{model_id.upper()} Learning Rate Schedule')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    out = PLOTS_DIR / f"learning_rate_{model_id}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")


def plot_training_dashboard(db: TrainingDB, model_id: str):
    """Comprehensive dashboard with all key metrics."""
    steps = get_step_data(db, model_id)
    runs = get_run_data(db, model_id)
    stats = db.get_model_stats(model_id)
    
    if len(steps) < 2:
        print("Skipping dashboard (insufficient data)")
        return
    
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f'HYDRA {model_id.upper()} Training Dashboard', fontsize=16, fontweight='bold')
    
    # Grid layout
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    x = [s["step"] for s in steps]
    
    # 1. Loss with EMA (large)
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(x, [s["ema_short"] for s in steps], label='Short EMA', linewidth=1.5)
    ax1.plot(x, [s["ema_medium"] for s in steps], label='Medium EMA', linewidth=1.5)
    ax1.plot(x, [s["ema_long"] for s in steps], label='Long EMA', linewidth=2)
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss (Multi-Scale EMA)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # 2. Summary stats (text box)
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    
    # Compute summary
    latest = steps[-1]
    max_step = stats.get("max_step", 0)
    total_tokens = compute_total_tokens(max_step, model_id)
    best_loss = stats.get("best_loss", 0)
    
    # Industry comparison
    target_loss = INDUSTRY_NORMS["loss_benchmarks"].get(f"{model_id}_converged", 
                                                         INDUSTRY_NORMS["loss_benchmarks"]["500M_converged"])
    loss_gap = ((latest["ema_long"] - target_loss) / target_loss) * 100
    
    summary = f"""
    Model: {model_id.upper()}
    ─────────────────────────
    Steps: {stats.get('max_step', 0):,}
    Runs: {stats.get('run_count', 0)}
    Records: {stats.get('step_count', 0):,}
    Tokens: {total_tokens/1e9:.2f}B
    
    Current Loss:
      Short EMA: {latest['ema_short']:.4f}
      Medium EMA: {latest['ema_medium']:.4f}
      Long EMA: {latest['ema_long']:.4f}
      Best: {best_loss:.4f}
    
    Industry Comparison:
      Target: {target_loss:.2f}
      Gap: {loss_gap:+.1f}%
      Status: {'✅ On track' if loss_gap < 20 else '⚠️ Above target'}
    """
    ax2.text(0.05, 0.95, summary, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 3. Gradient norms
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(x, [s["grad_norm"] for s in steps], linewidth=1, color='blue')
    ax3.set_yscale('log', base=10)
    ax3.set_ylabel('Grad Norm')
    ax3.set_title('Gradient Norms')
    ax3.grid(True, alpha=0.3)
    
    # 4. Learning rate
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(x, [s["lr"] for s in steps], linewidth=1.5, color='purple')
    ax4.set_ylabel('LR')
    ax4.set_title('Learning Rate')
    ax4.grid(True, alpha=0.3)
    
    # 5. Loss components
    ax5 = fig.add_subplot(gs[1, 2])
    ce = [s["loss_ce"] for s in steps]
    ax5.plot(x, ce, linewidth=1, label='CE', color='blue')
    aux = [s["loss_aux"] for s in steps]
    if any(v != 0 for v in aux):
        ax5.plot(x, aux, linewidth=1, label='Aux', color='orange', alpha=0.7)
    ax5.set_ylabel('Loss')
    ax5.set_title('Loss Components')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Loss milestones
    ax6 = fig.add_subplot(gs[2, 0])
    milestones = db.get_loss_milestones(model_id, milestone_interval=5000)
    if milestones:
        ms_x = list(milestones.keys())
        ms_y = list(milestones.values())
        ax6.bar(range(len(ms_x)), ms_y, color='steelblue', alpha=0.8)
        ax6.set_xticks(range(len(ms_x)))
        ax6.set_xticklabels([f"{x//1000}K" for x in ms_x], rotation=45)
        ax6.set_ylabel('Best Loss')
        ax6.set_title('Loss at Milestones')
        ax6.grid(True, alpha=0.3, axis='y')
    
    # 7. Recent trend (last 10K steps)
    ax7 = fig.add_subplot(gs[2, 1])
    recent_steps = [s for s in steps if s["step"] >= stats.get("max_step", 0) - 10000]
    if len(recent_steps) > 10:
        rx = [s["step"] for s in recent_steps]
        ry_short = [s["ema_short"] for s in recent_steps]
        ry_med = [s["ema_medium"] for s in recent_steps]
        ax7.plot(rx, ry_short, label='Short', linewidth=1.5)
        ax7.plot(rx, ry_med, label='Medium', linewidth=1.5)
        ax7.set_ylabel('Loss')
        ax7.set_title('Recent Trend (Last 10K Steps)')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
    
    # 8. Run history
    ax8 = fig.add_subplot(gs[2, 2])
    if runs:
        run_steps = [(r.get("end_step", 0) or 0) - (r.get("start_step", 0) or 0) for r in runs]
        ax8.bar(range(len(run_steps)), run_steps, color='teal', alpha=0.7)
        ax8.set_xlabel('Run Index')
        ax8.set_ylabel('Steps')
        ax8.set_title(f'Run Lengths ({len(runs)} runs)')
        ax8.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    out = PLOTS_DIR / f"dashboard_{model_id}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")


def plot_throughput_vram(db: TrainingDB, model_id: str):
    """Tokens/sec and VRAM usage over time."""
    steps = get_step_data(db, model_id)
    if len(steps) < 2:
        print("Skipping throughput/VRAM plot (insufficient data)")
        return

    x = [s["step"] for s in steps]
    tps = [s.get("tokens_per_sec", 0.0) or 0.0 for s in steps]
    vram = [s.get("vram_gb", 0.0) or 0.0 for s in steps]

    if not any(tps) and not any(vram):
        print("Skipping throughput/VRAM plot (no data)")
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    axes[0].plot(x, tps, linewidth=1.2, color="teal")
    axes[0].set_ylabel("Tokens/sec")
    axes[0].set_title(f"{model_id.upper()} Throughput")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x, vram, linewidth=1.2, color="purple")
    axes[1].set_ylabel("VRAM (GB)")
    axes[1].set_xlabel("Step")
    axes[1].set_title(f"{model_id.upper()} VRAM Usage")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out = PLOTS_DIR / f"throughput_vram_{model_id}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")


def plot_moe_routing_health(db: TrainingDB, model_id: str):
    """MoE routing entropy, divergence, and expert utilization."""
    with db._conn() as conn:
        rows = conn.execute(
            """
            SELECT step, entropy, divergence,
                   util_expert_0, util_expert_1, util_expert_2, util_expert_3
            FROM routing_moe
            WHERE model_id = ?
            ORDER BY step
            """,
            (model_id,),
        ).fetchall()

    if len(rows) < 2:
        print("Skipping MoE routing plot (insufficient data)")
        return

    x = [r["step"] for r in rows]
    entropy = [r["entropy"] for r in rows]
    divergence = [r["divergence"] for r in rows]
    util0 = [r["util_expert_0"] for r in rows]
    util1 = [r["util_expert_1"] for r in rows]
    util2 = [r["util_expert_2"] for r in rows]
    util3 = [r["util_expert_3"] for r in rows]

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    axes[0].plot(x, entropy, label="Entropy", linewidth=1.2)
    axes[0].plot(x, divergence, label="Divergence", linewidth=1.2)
    axes[0].set_ylabel("Score")
    axes[0].set_title(f"{model_id.upper()} MoE Routing Health")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x, util0, label="Expert 0")
    axes[1].plot(x, util1, label="Expert 1")
    axes[1].plot(x, util2, label="Expert 2")
    axes[1].plot(x, util3, label="Expert 3")
    axes[1].set_ylabel("Utilization (%)")
    axes[1].set_xlabel("Step")
    axes[1].legend(ncol=4, fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out = PLOTS_DIR / f"moe_routing_{model_id}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")


def plot_mor_depth_health(db: TrainingDB, model_id: str):
    """MoR average vs expected depth over time (avg across layers)."""
    with db._conn() as conn:
        rows = conn.execute(
            """
            SELECT step,
                   AVG(avg_depth) AS avg_depth,
                   AVG(expected_depth) AS expected_depth
            FROM routing_mor
            WHERE model_id = ?
            GROUP BY step
            ORDER BY step
            """,
            (model_id,),
        ).fetchall()

    if len(rows) < 2:
        print("Skipping MoR depth plot (insufficient data)")
        return

    x = [r["step"] for r in rows]
    avg_depth = [r["avg_depth"] for r in rows]
    exp_depth = [r["expected_depth"] for r in rows]

    plt.figure(figsize=(14, 5))
    plt.plot(x, avg_depth, label="Avg Depth", linewidth=1.5)
    plt.plot(x, exp_depth, label="Expected Depth", linewidth=1.5)
    plt.xlabel("Step")
    plt.ylabel("Depth")
    plt.title(f"{model_id.upper()} MoR Depth Health")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out = PLOTS_DIR / f"mor_depth_{model_id}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")


def plot_mod_utilization(db: TrainingDB, model_id: str):
    """MoD utilization and compute savings (avg across layers)."""
    with db._conn() as conn:
        rows = conn.execute(
            """
            SELECT step,
                   AVG(selected_frac) AS selected_frac,
                   AVG(compute_savings_pct) AS compute_savings_pct
            FROM routing_mod
            WHERE model_id = ?
            GROUP BY step
            ORDER BY step
            """,
            (model_id,),
        ).fetchall()

    if len(rows) < 2:
        print("Skipping MoD utilization plot (insufficient data)")
        return

    x = [r["step"] for r in rows]
    selected_frac = [r["selected_frac"] for r in rows]
    savings = [r["compute_savings_pct"] for r in rows]

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    axes[0].plot(x, selected_frac, linewidth=1.2, color="blue")
    axes[0].set_ylabel("Selected Fraction")
    axes[0].set_title(f"{model_id.upper()} MoD Utilization")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x, savings, linewidth=1.2, color="green")
    axes[1].set_ylabel("Compute Savings (%)")
    axes[1].set_xlabel("Step")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out = PLOTS_DIR / f"mod_utilization_{model_id}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")


def plot_adaptive_lr_state(db: TrainingDB, model_id: str):
    """Adaptive LR state (EMA trends, patience, cooldown)."""
    with db._conn() as conn:
        rows = conn.execute(
            """
            SELECT step, loss_ema_short, loss_ema_long,
                   patience_counter, cooldown_triggered
            FROM adaptive_lr
            WHERE model_id = ?
            ORDER BY step
            """,
            (model_id,),
        ).fetchall()

    if len(rows) < 2:
        print("Skipping adaptive LR plot (insufficient data)")
        return

    x = [r["step"] for r in rows]
    ema_short = [r["loss_ema_short"] for r in rows]
    ema_long = [r["loss_ema_long"] for r in rows]
    patience = [r["patience_counter"] for r in rows]
    cooldown = [r["cooldown_triggered"] for r in rows]

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    axes[0].plot(x, ema_short, label="EMA Short", linewidth=1.2)
    axes[0].plot(x, ema_long, label="EMA Long", linewidth=1.2)
    axes[0].set_ylabel("Loss EMA")
    axes[0].set_title(f"{model_id.upper()} Adaptive LR State")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x, patience, label="Patience", linewidth=1.2)
    axes[1].plot(x, cooldown, label="Cooldown Triggered", linewidth=1.2)
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("State")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out = PLOTS_DIR / f"adaptive_lr_{model_id}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")


def plot_grad_clip_pressure(db: TrainingDB, model_id: str):
    """Grad clip pressure: post/pre ratio over time."""
    steps = get_step_data(db, model_id)
    if len(steps) < 2:
        print("Skipping grad clip pressure plot (insufficient data)")
        return

    x = [s["step"] for s in steps]
    pre = [s["grad_norm_pre_clip"] for s in steps]
    post = [s["grad_norm"] for s in steps]

    eps = 1e-9
    ratio = [p / max(pp, eps) for p, pp in zip(post, pre)]

    plt.figure(figsize=(14, 4))
    plt.plot(x, ratio, linewidth=1.2, color="orange")
    plt.axhline(y=1.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
    plt.xlabel("Step")
    plt.ylabel("Post/Pre")
    plt.title(f"{model_id.upper()} Grad Clip Pressure (ratio)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out = PLOTS_DIR / f"grad_clip_pressure_{model_id}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")


def plot_run_health(db: TrainingDB, model_id: str):
    """Run lengths and best loss per run (cross-run health)."""
    runs = get_run_data(db, model_id)
    if not runs:
        print("Skipping run health plot (insufficient data)")
        return

    run_labels = list(range(len(runs)))
    run_lengths = [(r.get("end_step", 0) or 0) - (r.get("start_step", 0) or 0) for r in runs]
    best_losses = [r.get("best_loss", 0.0) or 0.0 for r in runs]

    fig, ax1 = plt.subplots(figsize=(14, 5))
    ax1.bar(run_labels, run_lengths, color="steelblue", alpha=0.7, label="Run Length (steps)")
    ax1.set_xlabel("Run Index")
    ax1.set_ylabel("Steps")
    ax1.grid(True, alpha=0.3, axis="y")

    ax2 = ax1.twinx()
    ax2.plot(run_labels, best_losses, color="darkred", linewidth=1.5, label="Best Loss")
    ax2.set_ylabel("Best Loss")

    ax1.set_title(f"{model_id.upper()} Run Health Summary")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper right")

    plt.tight_layout()
    out = PLOTS_DIR / f"run_health_{model_id}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")


# =============================================================================
# COMPREHENSIVE TRAINING REPORT
# =============================================================================

def generate_training_report(db: TrainingDB, model_id: str) -> str:
    """Generate a comprehensive markdown report comparing to industry norms."""
    steps = get_step_data(db, model_id)
    runs = get_run_data(db, model_id)
    stats = db.get_model_stats(model_id)
    milestones = db.get_loss_milestones(model_id, milestone_interval=10000)
    
    if not steps:
        return "No training data available."
    
    latest = steps[-1]
    max_step = stats.get("max_step", 0)
    total_tokens = compute_total_tokens(max_step, model_id)
    
    # Model size estimation (from model_id)
    model_size_str = model_id.replace("m", "M").replace("b", "B")
    try:
        if "m" in model_id.lower():
            model_params = float(model_id.lower().replace("m", "")) * 1e6
        elif "b" in model_id.lower():
            model_params = float(model_id.lower().replace("b", "")) * 1e9
        else:
            model_params = 500e6  # Default assumption
    except:
        model_params = 500e6
    
    # Compute metrics
    tokens_per_param = total_tokens / model_params if model_params > 0 else 0
    chinchilla_tokens = model_params * INDUSTRY_NORMS["tokens_per_param"]["chinchilla_optimal"]
    chinchilla_progress = (total_tokens / chinchilla_tokens) * 100 if chinchilla_tokens > 0 else 0
    
    # Loss analysis
    target_loss = INDUSTRY_NORMS["loss_benchmarks"].get(
        f"{model_id}_converged", 
        INDUSTRY_NORMS["loss_benchmarks"]["500M_converged"]
    )
    current_loss = latest["ema_long"]
    loss_gap_pct = ((current_loss - target_loss) / target_loss) * 100
    
    # Gradient health
    recent_grads = [s["grad_norm"] for s in steps[-100:]]
    avg_grad = sum(recent_grads) / len(recent_grads) if recent_grads else 0
    max_grad = max(recent_grads) if recent_grads else 0
    grad_healthy_lo, grad_healthy_hi = INDUSTRY_NORMS["gradient_norms"]["healthy_range"]
    grad_status = "✅ Healthy" if grad_healthy_lo <= avg_grad <= grad_healthy_hi else "⚠️ Check"
    
    # Recent trend (last 5K steps)
    recent_5k = [s for s in steps if s["step"] >= stats.get("max_step", 0) - 5000]
    if len(recent_5k) >= 2:
        trend_start = recent_5k[0]["ema_medium"]
        trend_end = recent_5k[-1]["ema_medium"]
        trend_change = ((trend_end - trend_start) / trend_start) * 100
        trend_status = "📉 Improving" if trend_change < -1 else "📈 Worsening" if trend_change > 1 else "➡️ Plateau"
    else:
        trend_change = 0
        trend_status = "❓ Insufficient data"
    
    # LR analysis
    current_lr = latest["lr"]
    lr_lo, lr_hi = INDUSTRY_NORMS["learning_rate"]["typical_peak"]
    lr_status = "✅ In range" if lr_lo <= current_lr <= lr_hi else "⚠️ Outside typical"
    
    # Build report
    report = f"""# HYDRA Training Report: {model_size_str}

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 📊 Training Overview

| Metric | Value | Industry Norm | Status |
|--------|-------|---------------|--------|
| Total Steps | {stats.get('max_step', 0):,} | — | — |
| Total Runs | {stats.get('run_count', 0)} | — | — |
| Tokens Trained | {total_tokens/1e9:.2f}B | {chinchilla_tokens/1e9:.1f}B (Chinchilla) | {chinchilla_progress:.0f}% of optimal |
| Tokens/Param | {tokens_per_param:.1f} | 20-100 | {'✅' if 20 <= tokens_per_param <= 100 else '⚠️'} |

---

## 📉 Loss Analysis

| Metric | Value | Target | Gap |
|--------|-------|--------|-----|
| Current (Long EMA) | {current_loss:.4f} | {target_loss:.2f} | {loss_gap_pct:+.1f}% |
| Short EMA | {latest['ema_short']:.4f} | — | — |
| Medium EMA | {latest['ema_medium']:.4f} | — | — |
| Best Ever | {stats.get('best_loss', 0):.4f} | — | — |

### Loss Milestones
"""
    
    if milestones:
        report += "\n| Step | Best Loss | Δ from Previous |\n|------|-----------|------------------|\n"
        prev_loss = None
        for step, loss in sorted(milestones.items()):
            if prev_loss:
                delta = ((loss - prev_loss) / prev_loss) * 100
                delta_str = f"{delta:+.1f}%"
            else:
                delta_str = "—"
            report += f"| {step//1000}K | {loss:.4f} | {delta_str} |\n"
            prev_loss = loss
    
    report += f"""
---

## 🔬 Training Health

### Gradient Norms
- **Recent Average:** {avg_grad:.2f} {grad_status}
- **Recent Max:** {max_grad:.2f}
- **Healthy Range:** {grad_healthy_lo} - {grad_healthy_hi}
- **Warning Threshold:** {INDUSTRY_NORMS['gradient_norms']['warning_threshold']}

### Learning Rate
- **Current:** {current_lr:.2e} {lr_status}
- **Typical Peak Range:** {lr_lo:.0e} - {lr_hi:.0e}

### Recent Trend (Last 5K Steps)
- **Direction:** {trend_status}
- **Change:** {trend_change:+.2f}%

---

## 🎯 Recommendations

"""
    
    # Generate recommendations
    recommendations = []
    
    if chinchilla_progress < 50:
        recommendations.append("- **Continue training:** Only {:.0f}% of Chinchilla-optimal tokens trained. Model has room to improve.".format(chinchilla_progress))
    elif chinchilla_progress > 150:
        recommendations.append("- **Consider stopping:** Training beyond Chinchilla-optimal. Diminishing returns likely.")
    
    if loss_gap_pct > 30:
        recommendations.append("- **Loss is high:** {:.1f}% above target. Check learning rate, data quality, or architecture.".format(loss_gap_pct))
    elif loss_gap_pct < 10:
        recommendations.append("- **Loss is good:** Within 10% of target. Training is on track.")
    
    if "Plateau" in trend_status:
        recommendations.append("- **Plateau detected:** Consider reducing LR or checking for data issues.")
    elif "Worsening" in trend_status:
        recommendations.append("- **Loss increasing:** Check for gradient explosion, bad data batch, or LR too high.")
    
    if not (grad_healthy_lo <= avg_grad <= grad_healthy_hi):
        if avg_grad < grad_healthy_lo:
            recommendations.append("- **Gradients too small:** May indicate vanishing gradients. Check residual connections.")
        else:
            recommendations.append("- **Gradients elevated:** Consider reducing LR or increasing grad clip.")
    
    if not recommendations:
        recommendations.append("- **All metrics healthy.** Continue training as planned.")
    
    report += "\n".join(recommendations)
    
    report += f"""

---

## 📈 Plots

See `reports/plots/` for detailed visualizations:
- `dashboard_{model_id}.png` - Comprehensive dashboard
- `loss_ema_{model_id}.png` - Loss with multi-scale EMA
- `loss_components_{model_id}.png` - CE vs auxiliary losses
- `grad_norms_{model_id}.png` - Gradient health
- `learning_rate_{model_id}.png` - LR schedule
- `throughput_vram_{model_id}.png` - Throughput & VRAM over time
- `moe_routing_{model_id}.png` - MoE routing entropy/utilization
- `mor_depth_{model_id}.png` - MoR depth health
- `mod_utilization_{model_id}.png` - MoD utilization & savings
- `adaptive_lr_{model_id}.png` - Adaptive LR state
- `grad_clip_pressure_{model_id}.png` - Grad clip pressure ratio
- `run_health_{model_id}.png` - Run length & best loss summary

---

*Report generated by HYDRA Training Analysis System*
"""
    
    return report


def save_training_report(db: TrainingDB, model_id: str):
    """Save the training report to a markdown file."""
    report = generate_training_report(db, model_id)
    
    # Save as both latest and timestamped
    report_path = REPORTS_DIR / f"training_status_{model_id}.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Saved {report_path}")
    
    return report_path


# =============================================================================
# MAIN
# =============================================================================

def analyze_training(model_id: str = "500m"):
    """Run full analysis and generate all outputs."""
    print("=" * 60)
    print(f"HYDRA Training Analysis - {model_id.upper()}")
    print("=" * 60)
    
    db = TrainingDB()
    stats = db.get_model_stats(model_id)
    
    if stats.get("step_count", 0) == 0:
        print(f"\n⚠️  No data found for model '{model_id}'")
        print("   Run training first, or check model_id")
        return
    
    print(f"\n📊 Found {stats.get('step_count', 0):,} step records")
    print(f"   Steps: {stats.get('min_step', 0):,} - {stats.get('max_step', 0):,}")
    print(f"   Runs: {stats.get('run_count', 0)}")
    
    # Generate plots
    print("\n📈 Generating plots...")
    plot_loss_over_steps(db, model_id)
    plot_loss_components(db, model_id)
    plot_gradient_norms(db, model_id)
    plot_learning_rate(db, model_id)
    plot_training_dashboard(db, model_id)
    plot_throughput_vram(db, model_id)
    plot_moe_routing_health(db, model_id)
    plot_mor_depth_health(db, model_id)
    plot_mod_utilization(db, model_id)
    plot_adaptive_lr_state(db, model_id)
    plot_grad_clip_pressure(db, model_id)
    plot_run_health(db, model_id)
    
    # Generate report
    print("\n📝 Generating training report...")
    report_path = save_training_report(db, model_id)
    
    print(f"\n✅ Analysis complete!")
    print(f"   Plots: {PLOTS_DIR}")
    print(f"   Report: {report_path}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="HYDRA Training Analysis")
    parser.add_argument("--model", "-m", default="500m", help="Model ID (e.g., 500m, 626m)")
    args = parser.parse_args()
    
    analyze_training(args.model)


if __name__ == "__main__":
    main()
