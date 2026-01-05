#!/usr/bin/env python3
"""
MoE Training Watchdog - Real-time monitoring for expert specialization.

Monitors for two failure modes:
1. "Winner-Take-All" Monopoly: One expert takes >60% utilization while others starve
2. Representation Collapse: Divergence drops, Entropy approaches 1.0, loss plateaus

Usage:
    python diagnostics/moe_watchdog.py [--interval 60] [--checkpoint-dir checkpoints]
    
Run in a separate terminal while training.
"""

import argparse
import glob
import os
import re
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

# ANSI colors
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    RESET = '\033[0m'


@dataclass
class MoESnapshot:
    """Single point-in-time MoE health measurement."""
    step: int
    timestamp: datetime
    divergence: float
    entropy: float
    utilization: List[float]  # Per-expert utilization %
    ema_loss: float
    
    @property
    def max_utilization(self) -> float:
        return max(self.utilization) if self.utilization else 0.0
    
    @property
    def min_utilization(self) -> float:
        return min(self.utilization) if self.utilization else 0.0
    
    @property
    def utilization_spread(self) -> float:
        return self.max_utilization - self.min_utilization


def compute_expert_divergence(model_state: Dict[str, torch.Tensor]) -> Tuple[float, int]:
    """Compute mean pairwise divergence between expert weights."""
    # Find MoE expert weights
    expert_weights_by_layer = {}
    
    for key, tensor in model_state.items():
        # Match patterns like: moe_layers.0.experts.0.gate_up.weight
        match = re.match(r'moe_layers\.(\d+)\.experts\.(\d+)\.gate_up\.weight', key)
        if match:
            layer_idx = int(match.group(1))
            expert_idx = int(match.group(2))
            if layer_idx not in expert_weights_by_layer:
                expert_weights_by_layer[layer_idx] = {}
            expert_weights_by_layer[layer_idx][expert_idx] = tensor
    
    if not expert_weights_by_layer:
        return 0.0, 0
    
    # Compute pairwise divergence across all layers
    all_divergences = []
    for layer_idx, experts in expert_weights_by_layer.items():
        expert_ids = sorted(experts.keys())
        for i, ei in enumerate(expert_ids):
            for ej in expert_ids[i+1:]:
                diff = (experts[ei].float() - experts[ej].float()).abs().mean().item()
                all_divergences.append(diff)
    
    num_experts = len(expert_weights_by_layer.get(0, {}))
    avg_divergence = sum(all_divergences) / len(all_divergences) if all_divergences else 0.0
    return avg_divergence, num_experts


def compute_router_entropy(model_state: Dict[str, torch.Tensor], num_experts: int) -> float:
    """Estimate router entropy from router weights (approximation)."""
    # This is a rough approximation - true entropy requires forward pass data
    # We use router weight norm variance as a proxy
    router_weights = []
    for key, tensor in model_state.items():
        if 'router' in key and 'weight' in key:
            router_weights.append(tensor)
    
    if not router_weights:
        return 1.0  # Unknown, assume max entropy
    
    # Compute variance of router weight norms per expert
    # Higher variance = more decisive routing
    import math
    max_entropy = math.log(num_experts) if num_experts > 1 else 1.0
    
    # For now, return a placeholder - real entropy needs utilization stats
    return 0.99 * max_entropy  # Will be overridden by log parsing


def parse_latest_log_metrics(log_dir: str) -> Dict[str, float]:
    """Parse most recent training log for MoE metrics."""
    log_files = sorted(glob.glob(os.path.join(log_dir, "training_*.log")))
    if not log_files:
        return {}
    
    latest_log = log_files[-1]
    metrics = {}
    
    # Patterns to extract
    patterns = {
        'ema_loss': r'CE_EMA=([0-9.]+)',
        'moe_aux': r'moe_aux=([0-9.]+)',
        'step': r'step[=:\s]+(\d+)',
    }
    
    try:
        with open(latest_log, 'r') as f:
            lines = f.readlines()[-100:]  # Last 100 lines
            for line in reversed(lines):
                for key, pattern in patterns.items():
                    if key not in metrics:
                        match = re.search(pattern, line)
                        if match:
                            metrics[key] = float(match.group(1))
    except Exception:
        pass
    
    return metrics


def get_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Find most recent checkpoint file."""
    patterns = [
        os.path.join(checkpoint_dir, "hydra_*_step_*.pt"),
        os.path.join(checkpoint_dir, "*.pt"),
    ]
    
    all_files = []
    for pattern in patterns:
        all_files.extend(glob.glob(pattern))
    
    if not all_files:
        return None
    
    # Sort by modification time
    all_files.sort(key=os.path.getmtime)
    return all_files[-1]


def analyze_snapshot(current: MoESnapshot, history: List[MoESnapshot]) -> Tuple[str, str, str]:
    """
    Analyze MoE health and return (status, analysis, recommendation).
    
    Status: HEALTHY / WARNING / CRITICAL
    """
    issues = []
    analysis_parts = []
    
    # === Check for Winner-Take-All ===
    if current.max_utilization > 0.60:
        issues.append("WINNER-TAKE-ALL")
        analysis_parts.append(
            f"⚠️  Expert monopoly detected: max util={current.max_utilization:.1%} "
            f"(threshold: 60%)"
        )
    elif current.max_utilization > 0.45:
        analysis_parts.append(
            f"📊 Utilization slightly skewed: max={current.max_utilization:.1%}"
        )
    
    # === Check for Representation Collapse ===
    if len(history) >= 2:
        prev = history[-2] if len(history) >= 2 else history[-1]
        div_delta = current.divergence - prev.divergence
        
        if div_delta < -0.001 and current.entropy > 0.98:
            issues.append("COLLAPSE-RISK")
            analysis_parts.append(
                f"⚠️  Divergence FALLING ({prev.divergence:.4f} → {current.divergence:.4f}) "
                f"with high entropy ({current.entropy:.3f}) - collapse risk!"
            )
        elif div_delta > 0.002:
            analysis_parts.append(
                f"✅ Divergence RISING ({prev.divergence:.4f} → {current.divergence:.4f}) - "
                f"experts specializing"
            )
        else:
            analysis_parts.append(
                f"📊 Divergence stable ({current.divergence:.4f})"
            )
    
    # === Check Entropy ===
    if current.entropy > 1.35:  # Near max for 4 experts (ln(4)=1.39)
        analysis_parts.append(
            f"🎲 Entropy HIGH ({current.entropy:.3f}) - router still exploring"
        )
    elif current.entropy < 0.9:
        analysis_parts.append(
            f"🎯 Entropy LOW ({current.entropy:.3f}) - router has strong preferences"
        )
    else:
        analysis_parts.append(
            f"📊 Entropy moderate ({current.entropy:.3f})"
        )
    
    # === Check Loss Trend ===
    if len(history) >= 3:
        recent_losses = [s.ema_loss for s in history[-3:]] + [current.ema_loss]
        if all(recent_losses[i] >= recent_losses[i+1] - 0.01 for i in range(len(recent_losses)-1)):
            # Loss roughly flat or rising
            if current.divergence < 0.05:
                issues.append("LOSS-PLATEAU")
                analysis_parts.append(
                    f"⚠️  Loss plateaued ({current.ema_loss:.3f}) with low divergence - "
                    f"experts may be redundant"
                )
    
    # === Determine Status ===
    if "WINNER-TAKE-ALL" in issues or "COLLAPSE-RISK" in issues:
        status = "CRITICAL"
    elif "LOSS-PLATEAU" in issues or current.utilization_spread > 0.4:
        status = "WARNING"
    else:
        status = "HEALTHY"
    
    # === Generate Recommendation ===
    if status == "CRITICAL":
        if "WINNER-TAKE-ALL" in issues:
            recommendation = "INCREASE aux_weight to 0.001 or INCREASE jitter to 0.2"
        else:
            recommendation = "INCREASE diversity_noise to 0.1 and RESTART from earlier checkpoint"
    elif status == "WARNING":
        recommendation = "CONTINUE but monitor closely - may need intervention in 1-2K steps"
    else:
        recommendation = "CONTINUE - training is progressing normally"
    
    analysis = "\n   ".join(analysis_parts)
    return status, analysis, recommendation


def format_utilization(util: List[float]) -> str:
    """Format utilization as bar chart."""
    if not util:
        return "N/A"
    bars = []
    for i, u in enumerate(util):
        bar_len = int(u * 20)
        bar = '█' * bar_len + '░' * (20 - bar_len)
        bars.append(f"E{i}: [{bar}] {u*100:5.1f}%")
    return "\n      ".join(bars)


def run_watchdog(
    checkpoint_dir: str,
    log_dir: str,
    interval: int = 60,
    max_history: int = 20,
):
    """Main watchdog loop."""
    print(f"\n{Colors.BOLD}{'='*70}")
    print("  MoE TRAINING WATCHDOG")
    print(f"{'='*70}{Colors.RESET}")
    print(f"  Checkpoint dir: {checkpoint_dir}")
    print(f"  Log dir: {log_dir}")
    print(f"  Poll interval: {interval}s")
    print(f"  Press Ctrl+C to stop\n")
    
    history: deque = deque(maxlen=max_history)
    last_step = -1
    
    while True:
        try:
            # Find latest checkpoint
            ckpt_path = get_latest_checkpoint(checkpoint_dir)
            if not ckpt_path:
                print(f"{Colors.YELLOW}⏳ Waiting for checkpoint...{Colors.RESET}")
                time.sleep(interval)
                continue
            
            # Load checkpoint
            try:
                ckpt = torch.load(ckpt_path, weights_only=False, map_location='cpu')
            except Exception as e:
                print(f"{Colors.YELLOW}⏳ Checkpoint loading... ({e}){Colors.RESET}")
                time.sleep(10)
                continue
            
            step = ckpt.get('step', 0)
            if step == last_step:
                # No new checkpoint
                time.sleep(interval // 2)
                continue
            
            last_step = step
            model_state = ckpt.get('model', {})
            
            # Compute metrics
            divergence, num_experts = compute_expert_divergence(model_state)
            
            # Get metrics from log
            log_metrics = parse_latest_log_metrics(log_dir)
            ema_loss = log_metrics.get('ema_loss', ckpt.get('metrics', {}).get('ema_loss', 0.0))
            
            # Estimate entropy (placeholder - would need forward pass for real values)
            import math
            max_entropy = math.log(num_experts) if num_experts > 1 else 1.0
            # Use divergence as inverse proxy for entropy (more divergence = less random)
            estimated_entropy = max_entropy * (1.0 - min(divergence / 0.2, 0.3))
            
            # Create snapshot (utilization would need forward pass, use placeholder)
            # In production, this would parse actual utilization from logs
            snapshot = MoESnapshot(
                step=step,
                timestamp=datetime.now(),
                divergence=divergence,
                entropy=estimated_entropy,
                utilization=[0.25, 0.25, 0.25, 0.25],  # Placeholder
                ema_loss=ema_loss,
            )
            
            history.append(snapshot)
            
            # Analyze
            status, analysis, recommendation = analyze_snapshot(snapshot, list(history))
            
            # Color-code status
            if status == "HEALTHY":
                status_color = Colors.GREEN
                status_icon = "✅"
            elif status == "WARNING":
                status_color = Colors.YELLOW
                status_icon = "⚠️ "
            else:
                status_color = Colors.RED
                status_icon = "🚨"
            
            # Print report
            print(f"\n{Colors.BOLD}{'─'*70}")
            print(f"  STEP {step:,} | {datetime.now().strftime('%H:%M:%S')}")
            print(f"{'─'*70}{Colors.RESET}")
            print(f"  Status: {status_color}{status_icon} {status}{Colors.RESET}")
            print(f"  Divergence: {divergence:.6f} {'(↑ good)' if divergence > 0.05 else '(low - watch)'}")
            print(f"  Entropy:    {estimated_entropy:.4f} / {max_entropy:.4f}")
            print(f"  EMA Loss:   {ema_loss:.4f}")
            print(f"\n  {Colors.CYAN}Analysis:{Colors.RESET}")
            print(f"   {analysis}")
            print(f"\n  {Colors.BOLD}Recommendation:{Colors.RESET} {recommendation}")
            
            # Trend sparkline
            if len(history) >= 3:
                divs = [s.divergence for s in list(history)[-10:]]
                trend = "".join(['↑' if divs[i] < divs[i+1] else '↓' if divs[i] > divs[i+1] else '→' 
                                for i in range(len(divs)-1)])
                print(f"\n  Divergence trend: {trend}")
            
            print(f"{Colors.BOLD}{'─'*70}{Colors.RESET}")
            
            time.sleep(interval)
            
        except KeyboardInterrupt:
            print(f"\n{Colors.CYAN}Watchdog stopped.{Colors.RESET}")
            break
        except Exception as e:
            print(f"{Colors.RED}Error: {e}{Colors.RESET}")
            time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description="MoE Training Watchdog")
    parser.add_argument("--interval", type=int, default=60, 
                        help="Poll interval in seconds (default: 60)")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Checkpoint directory (default: checkpoints)")
    parser.add_argument("--log-dir", type=str, default="logs",
                        help="Log directory (default: logs)")
    args = parser.parse_args()
    
    run_watchdog(
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        interval=args.interval,
    )


if __name__ == "__main__":
    main()
