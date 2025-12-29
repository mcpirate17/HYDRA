#!/usr/bin/env python3
"""
Compare MoD/MoR effectiveness: Vanilla vs Full Routing

This script runs controlled comparisons to measure:
1. Training loss convergence (target: < 4.0)
2. MoD token routing efficiency (% tokens skipping per layer)
3. MoR depth distribution (tokens exiting at each recursion level)
4. Compute savings from dynamic routing
5. Throughput impact

Usage:
    python scripts/compare_mod_mor_effectiveness.py --model_size 50M --max_steps 5000
    python scripts/compare_mod_mor_effectiveness.py --model_size debug --max_steps 1000 --quick
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def run_training(
    model_size: str,
    max_steps: int,
    mod_off: bool = False,
    mor_off: bool = False,
    run_name: str = "",
    extra_args: list[str] | None = None,
) -> dict[str, Any]:
    """Run a single training experiment and return results."""
    
    cmd = [
        sys.executable, "trainer.py",
        "--model_size", model_size,
        "--attention", "ccgqa",
        "--max_steps", str(max_steps),
        "--no_short_run_override",
        "--mod_enable_pct", "0.10",
        "--mod_force_enable_pct", "0.15",
        "--mor_enable_pct", "0.20",
        "--no-wandb",
        "--mode", "testing",
    ]
    
    if mod_off:
        cmd.append("--mod_off")
    if mor_off:
        cmd.append("--mor_off")
    if run_name:
        cmd.extend(["--run_name", run_name])
    if extra_args:
        cmd.extend(extra_args)
    
    print(f"\n{'='*70}")
    print(f"🚀 Running: {run_name or 'experiment'}")
    print(f"   Command: {' '.join(cmd)}")
    print(f"{'='*70}\n")
    
    # Run training
    result = subprocess.run(
        cmd,
        capture_output=False,
        text=True,
        cwd=Path(__file__).parent.parent,
    )
    
    # Load diagnostics
    diag_path = Path("checkpoints/training_diagnostics.json")
    if diag_path.exists():
        with open(diag_path) as f:
            diagnostics = json.load(f)
    else:
        diagnostics = []
    
    # Load latest report
    reports_dir = Path("reports")
    report_files = sorted(reports_dir.glob("training_report_*.json"), reverse=True)
    report = {}
    if report_files:
        with open(report_files[0]) as f:
            report = json.load(f)
    
    return {
        "run_name": run_name,
        "mod_off": mod_off,
        "mor_off": mor_off,
        "diagnostics": diagnostics,
        "report": report,
        "exit_code": result.returncode,
    }


def extract_metrics(results: dict[str, Any]) -> dict[str, Any]:
    """Extract key metrics from training results."""
    diag = results.get("diagnostics", [])
    report = results.get("report", {})
    
    metrics = {
        "run_name": results.get("run_name", "unknown"),
        "mod_enabled": not results.get("mod_off", False),
        "mor_enabled": not results.get("mor_off", False),
        "initial_loss": report.get("loss_analysis", {}).get("initial_loss", 0),
        "final_loss": report.get("loss_analysis", {}).get("final_loss", 0),
        "best_loss": report.get("loss_analysis", {}).get("best_loss", 0),
        "reduction_pct": report.get("loss_analysis", {}).get("reduction_percent", 0),
        "avg_throughput": report.get("performance", {}).get("avg_throughput_tok_s", 0),
        "training_time_s": report.get("performance", {}).get("training_time_seconds", 0),
    }
    
    # Extract MoD metrics from final diagnostics
    if diag:
        final_diag = diag[-1]
        mod_layers = final_diag.get("mod_layers", [])
        mor_layers = final_diag.get("mor_layers", [])
        
        # MoD metrics
        if mod_layers:
            compute_savings = [l.get("compute_savings_pct", 0) for l in mod_layers]
            metrics["mod_avg_compute_savings"] = np.mean(compute_savings) if compute_savings else 0
            metrics["mod_per_layer_savings"] = compute_savings
            metrics["mod_routing_modes"] = [l.get("routing_mode", "unknown") for l in mod_layers]
        else:
            metrics["mod_avg_compute_savings"] = 0
            metrics["mod_per_layer_savings"] = []
            metrics["mod_routing_modes"] = []
        
        # MoR metrics
        if mor_layers:
            avg_depths = [l.get("avg_depth", 0) for l in mor_layers if "avg_depth" in l]
            depth_histograms = [l.get("depth_histogram", []) for l in mor_layers]
            tokens_per_rec = [l.get("tokens_per_recursion", []) for l in mor_layers]
            
            metrics["mor_avg_depth"] = np.mean(avg_depths) if avg_depths else 0
            metrics["mor_depth_histograms"] = depth_histograms
            metrics["mor_tokens_per_recursion"] = tokens_per_rec
            
            # Calculate early exit ratio
            total_early_exits = 0
            total_tokens = 0
            for hist in depth_histograms:
                if hist:
                    # Tokens exiting before final recursion
                    total_early_exits += sum(hist[:-1]) if len(hist) > 1 else 0
                    total_tokens += sum(hist)
            metrics["mor_early_exit_ratio"] = total_early_exits / max(1, total_tokens)
        else:
            metrics["mor_avg_depth"] = 0
            metrics["mor_depth_histograms"] = []
            metrics["mor_tokens_per_recursion"] = []
            metrics["mor_early_exit_ratio"] = 0
    
    # Extract loss history
    losses = []
    steps = []
    for d in diag:
        step = d.get("step", 0)
        loss = d.get("losses", {}).get("ce", d.get("losses", {}).get("total", 0))
        if loss > 0:
            losses.append(loss)
            steps.append(step)
    metrics["loss_history"] = losses
    metrics["step_history"] = steps
    
    return metrics


def plot_comparison(all_metrics: list[dict], output_dir: Path):
    """Generate comparison charts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Loss curves comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss over time
    ax1 = axes[0, 0]
    for m in all_metrics:
        label = m["run_name"]
        if m["loss_history"]:
            ax1.plot(m["step_history"], m["loss_history"], label=label, linewidth=2)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_title("Training Loss Convergence")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=4.0, color='r', linestyle='--', alpha=0.5, label='Target (4.0)')
    
    # Final metrics comparison
    ax2 = axes[0, 1]
    names = [m["run_name"] for m in all_metrics]
    final_losses = [m["final_loss"] for m in all_metrics]
    colors = ['#2ecc71' if m["mod_enabled"] and m["mor_enabled"] else '#e74c3c' for m in all_metrics]
    bars = ax2.bar(names, final_losses, color=colors)
    ax2.set_ylabel("Final Loss")
    ax2.set_title("Final Loss Comparison")
    ax2.axhline(y=4.0, color='r', linestyle='--', alpha=0.5)
    for bar, loss in zip(bars, final_losses):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                 f'{loss:.3f}', ha='center', va='bottom', fontsize=10)
    
    # MoD compute savings
    ax3 = axes[1, 0]
    mod_savings = [m.get("mod_avg_compute_savings", 0) for m in all_metrics]
    bars = ax3.bar(names, mod_savings, color=['#3498db' if s > 0 else '#95a5a6' for s in mod_savings])
    ax3.set_ylabel("Compute Savings (%)")
    ax3.set_title("MoD Compute Savings per Run")
    ax3.set_ylim(0, 60)
    for bar, savings in zip(bars, mod_savings):
        if savings > 0:
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                     f'{savings:.1f}%', ha='center', va='bottom', fontsize=10)
    
    # MoR depth distribution (for runs with MoR)
    ax4 = axes[1, 1]
    mor_run = next((m for m in all_metrics if m["mor_enabled"] and m["mor_depth_histograms"]), None)
    if mor_run and mor_run["mor_depth_histograms"]:
        # Aggregate histograms across layers
        all_hists = mor_run["mor_depth_histograms"]
        max_len = max(len(h) for h in all_hists if h)
        aggregated = np.zeros(max_len)
        for h in all_hists:
            if h:
                for i, v in enumerate(h):
                    if i < max_len:
                        aggregated[i] += v
        if aggregated.sum() > 0:
            aggregated = aggregated / aggregated.sum() * 100
            x = np.arange(len(aggregated))
            ax4.bar(x, aggregated, color='#9b59b6')
            ax4.set_xlabel("Recursion Depth")
            ax4.set_ylabel("Token Distribution (%)")
            ax4.set_title(f"MoR Depth Distribution ({mor_run['run_name']})")
            ax4.set_xticks(x)
            ax4.set_xticklabels([f"r{i}" for i in x])
            for i, v in enumerate(aggregated):
                if v > 1:
                    ax4.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontsize=9)
    else:
        ax4.text(0.5, 0.5, "No MoR data available", ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title("MoR Depth Distribution")
    
    plt.tight_layout()
    chart_path = output_dir / "comparison_charts.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"\n📊 Charts saved to: {chart_path}")
    
    # 2. Per-layer MoD savings chart
    fig, ax = plt.subplots(figsize=(12, 5))
    for m in all_metrics:
        if m["mod_per_layer_savings"]:
            x = np.arange(len(m["mod_per_layer_savings"]))
            ax.plot(x, m["mod_per_layer_savings"], marker='o', label=m["run_name"], linewidth=2)
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Compute Savings (%)")
    ax.set_title("MoD Compute Savings per Layer")
    ax.legend()
    ax.grid(True, alpha=0.3)
    layer_chart_path = output_dir / "mod_per_layer_savings.png"
    plt.savefig(layer_chart_path, dpi=150)
    plt.close()
    print(f"📊 Per-layer MoD chart saved to: {layer_chart_path}")


def print_summary(all_metrics: list[dict]):
    """Print a summary table of results."""
    print("\n" + "="*80)
    print("📊 COMPARISON SUMMARY")
    print("="*80)
    
    # Header
    print(f"\n{'Run Name':<25} {'MoD':<5} {'MoR':<5} {'Final Loss':<12} {'Best Loss':<12} {'Throughput':<12} {'MoD Savings':<12}")
    print("-"*95)
    
    for m in all_metrics:
        mod_str = "✓" if m["mod_enabled"] else "✗"
        mor_str = "✓" if m["mor_enabled"] else "✗"
        savings_str = f"{m['mod_avg_compute_savings']:.1f}%" if m["mod_avg_compute_savings"] > 0 else "N/A"
        throughput_str = f"{m['avg_throughput']/1000:.1f}K" if m['avg_throughput'] > 0 else "N/A"
        
        print(f"{m['run_name']:<25} {mod_str:<5} {mor_str:<5} {m['final_loss']:<12.4f} {m['best_loss']:<12.4f} {throughput_str:<12} {savings_str:<12}")
    
    print("\n" + "-"*95)
    
    # Analysis
    full_routing = next((m for m in all_metrics if m["mod_enabled"] and m["mor_enabled"]), None)
    vanilla = next((m for m in all_metrics if not m["mod_enabled"] and not m["mor_enabled"]), None)
    
    if full_routing and vanilla:
        loss_diff = vanilla["final_loss"] - full_routing["final_loss"]
        loss_pct = (loss_diff / vanilla["final_loss"]) * 100 if vanilla["final_loss"] > 0 else 0
        
        print(f"\n🎯 KEY FINDINGS:")
        print(f"   • Loss difference: {loss_diff:+.4f} ({loss_pct:+.1f}%)")
        if loss_diff > 0:
            print(f"   • MoD/MoR IMPROVES loss by {loss_pct:.1f}%")
        elif loss_diff < 0:
            print(f"   • MoD/MoR HURTS loss by {abs(loss_pct):.1f}%")
        else:
            print(f"   • No significant difference")
        
        if full_routing["mod_avg_compute_savings"] > 0:
            print(f"   • MoD compute savings: {full_routing['mod_avg_compute_savings']:.1f}%")
        if full_routing["mor_early_exit_ratio"] > 0:
            print(f"   • MoR early exit ratio: {full_routing['mor_early_exit_ratio']*100:.1f}%")
        
        # Target check
        target = 4.0
        if full_routing["best_loss"] < target:
            print(f"   ✅ MoD/MoR achieved target loss < {target} (best: {full_routing['best_loss']:.4f})")
        else:
            print(f"   ⚠️  MoD/MoR did NOT achieve target loss < {target} (best: {full_routing['best_loss']:.4f})")
    
    print("\n" + "="*80)


def save_results(all_metrics: list[dict], output_dir: Path):
    """Save detailed results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        return obj
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "experiments": all_metrics,
    }
    
    results_path = output_dir / "mod_mor_comparison_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=convert)
    print(f"\n💾 Results saved to: {results_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare MoD/MoR effectiveness")
    parser.add_argument("--model_size", type=str, default="50M", help="Model size preset")
    parser.add_argument("--max_steps", type=int, default=5000, help="Training steps per run")
    parser.add_argument("--quick", action="store_true", help="Quick test with fewer steps")
    parser.add_argument("--output_dir", type=str, default="reports/mod_mor_comparison", help="Output directory")
    args = parser.parse_args()
    
    if args.quick:
        args.max_steps = min(args.max_steps, 1000)
    
    output_dir = Path(args.output_dir)
    all_results = []
    
    # Define experiments
    experiments = [
        {"mod_off": True, "mor_off": True, "name": "vanilla"},
        {"mod_off": False, "mor_off": True, "name": "mod_only"},
        {"mod_off": True, "mor_off": False, "name": "mor_only"},
        {"mod_off": False, "mor_off": False, "name": "full_routing"},
    ]
    
    print(f"\n{'#'*70}")
    print(f"# MoD/MoR EFFECTIVENESS COMPARISON")
    print(f"# Model: {args.model_size}, Steps: {args.max_steps}")
    print(f"# Experiments: {len(experiments)}")
    print(f"{'#'*70}\n")
    
    for exp in experiments:
        run_name = f"{args.model_size}_{exp['name']}"
        results = run_training(
            model_size=args.model_size,
            max_steps=args.max_steps,
            mod_off=exp["mod_off"],
            mor_off=exp["mor_off"],
            run_name=run_name,
        )
        metrics = extract_metrics(results)
        all_results.append(metrics)
    
    # Generate outputs
    print_summary(all_results)
    plot_comparison(all_results, output_dir)
    save_results(all_results, output_dir)
    
    print(f"\n✅ Comparison complete! Check {output_dir} for results and charts.")


if __name__ == "__main__":
    main()
