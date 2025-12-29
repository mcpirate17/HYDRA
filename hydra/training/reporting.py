from __future__ import annotations

import json
import math
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch

from .config import TrainingConfig
from .metrics import TrainingMetrics


def assess_model_performance(*, config: TrainingConfig, metrics: TrainingMetrics) -> dict[str, str]:
    assessment: dict[str, str] = {}
    reduction = (
        (metrics.initial_loss - metrics.final_loss) / metrics.initial_loss * 100
        if metrics.initial_loss > 0
        else 0
    )

    if reduction >= 60:
        assessment["learning_quality"] = "Excellent - Strong convergence"
    elif reduction >= 40:
        assessment["learning_quality"] = "Good - Solid learning"
    elif reduction >= 20:
        assessment["learning_quality"] = "Fair - Moderate progress"
    else:
        assessment["learning_quality"] = "Poor - May need more steps or tuning"

    random_baseline = math.log(config.vocab_size)
    final_vs_random = (random_baseline - metrics.final_loss) / random_baseline * 100
    if metrics.final_loss < 5.0:
        assessment["vs_random_baseline"] = f"{final_vs_random:.1f}% better - Approaching usable"
    elif metrics.final_loss < 7.0:
        assessment["vs_random_baseline"] = f"{final_vs_random:.1f}% better - Learning patterns"
    else:
        assessment["vs_random_baseline"] = f"{final_vs_random:.1f}% better - Early training"

    if len(metrics.losses) >= 10:
        last_10 = metrics.losses[-10:]
        trend = (last_10[0] - last_10[-1]) / last_10[0] * 100 if last_10[0] > 0 else 0
        if trend > 5:
            assessment["convergence_trend"] = "Still improving - Continue training"
        elif trend > 0:
            assessment["convergence_trend"] = "Slowing down - Near plateau"
        else:
            assessment["convergence_trend"] = "Plateaued - Consider LR adjustment"

    return assessment


def assess_training_quality(*, metrics: TrainingMetrics) -> dict[str, str]:
    assessment: dict[str, str] = {}
    if metrics.grad_norms:
        max_grad = max(metrics.grad_norms)
        if max_grad <= 1.0:
            assessment["gradient_stability"] = "Excellent - Well controlled"
        elif max_grad <= 5.0:
            assessment["gradient_stability"] = "Good - Occasional spikes"
        else:
            assessment["gradient_stability"] = f"Warning - Max grad {max_grad:.1f}"

    if metrics.tokens_per_sec:
        avg_tps = sum(metrics.tokens_per_sec) / len(metrics.tokens_per_sec)
        if avg_tps >= 30000:
            assessment["throughput"] = f"Excellent - {avg_tps/1000:.1f}K tok/s"
        elif avg_tps >= 15000:
            assessment["throughput"] = f"Good - {avg_tps/1000:.1f}K tok/s"
        else:
            assessment["throughput"] = f"Moderate - {avg_tps/1000:.1f}K tok/s"

    if len(metrics.losses) >= 20:
        loss_changes = [
            abs(metrics.losses[i] - metrics.losses[i - 1]) for i in range(1, len(metrics.losses))
        ]
        avg_change = sum(loss_changes) / len(loss_changes)
        if avg_change < 0.2:
            assessment["loss_stability"] = "Very smooth training"
        elif avg_change < 0.5:
            assessment["loss_stability"] = "Normal variance"
        else:
            assessment["loss_stability"] = "High variance - Consider LR or batch filtering"

    return assessment


def generate_report(
    *,
    config: TrainingConfig,
    metrics: TrainingMetrics,
    device: str,
    logger,
    format_time,
) -> Path:
    report_dir = Path(config.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    training_time = metrics.end_time - metrics.start_time
    avg_tps = metrics.total_tokens / training_time if training_time > 0 else 0

    loss_reduction = (
        (metrics.initial_loss - metrics.final_loss) / metrics.initial_loss * 100
        if metrics.initial_loss > 0
        else 0
    )

    losses = metrics.losses
    if losses:
        avg_loss = sum(losses) / len(losses)
        min_loss = min(losses)
        max_loss = max(losses)
        n = len(losses)
        loss_at_25 = losses[n // 4] if n > 4 else losses[-1]
        loss_at_50 = losses[n // 2] if n > 2 else losses[-1]
        loss_at_75 = losses[3 * n // 4] if n > 4 else losses[-1]
    else:
        avg_loss = min_loss = max_loss = 0
        loss_at_25 = loss_at_50 = loss_at_75 = 0

    tps_list = metrics.tokens_per_sec
    if tps_list:
        warmup_skip = min(10, len(tps_list) // 10)
        steady_tps = tps_list[warmup_skip:] if len(tps_list) > warmup_skip else tps_list
        avg_tps_steady = sum(steady_tps) / len(steady_tps) if steady_tps else 0
        peak_tps = max(tps_list)
    else:
        avg_tps_steady = peak_tps = 0

    report = {
        "metadata": {
            "timestamp": timestamp,
            "model": "HYDRA 100M",
            "dataset": config.dataset_name,
            "device": device,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
        },
        "configuration": asdict(config),
        "training_summary": {
            "total_steps": config.max_steps,
            "total_tokens": metrics.total_tokens,
            "training_time_seconds": training_time,
            "training_time_formatted": format_time(training_time),
        },
        "loss_analysis": {
            "initial_loss": metrics.initial_loss,
            "final_loss": metrics.final_loss,
            "best_loss": metrics.best_loss,
            "best_loss_step": metrics.best_loss_step,
            "loss_reduction_percent": loss_reduction,
            "average_loss": avg_loss,
            "min_loss": min_loss,
            "max_loss": max_loss,
            "loss_at_25_percent": loss_at_25,
            "loss_at_50_percent": loss_at_50,
            "loss_at_75_percent": loss_at_75,
        },
        "performance": {
            "average_tokens_per_second": avg_tps,
            "average_tokens_per_second_steady": avg_tps_steady,
            "peak_tokens_per_second": peak_tps,
            "average_step_time_ms": sum(metrics.step_times) / len(metrics.step_times) * 1000
            if metrics.step_times
            else 0,
        },
        "gradient_analysis": {
            "average_grad_norm": sum(metrics.grad_norms) / len(metrics.grad_norms)
            if metrics.grad_norms
            else 0,
            "max_grad_norm": max(metrics.grad_norms) if metrics.grad_norms else 0,
            "min_grad_norm": min(metrics.grad_norms) if metrics.grad_norms else 0,
        },
        "model_assessment": assess_model_performance(config=config, metrics=metrics),
        "training_assessment": assess_training_quality(metrics=metrics),
        "raw_metrics": metrics.to_dict(),
    }

    report_path = report_dir / f"training_report_{timestamp}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("\n" + "=" * 70)
    logger.info("TRAINING REPORT")
    logger.info("=" * 70)
    logger.info(f"\n📊 Loss Analysis:")
    logger.info(f"   Initial: {metrics.initial_loss:.4f}")
    logger.info(f"   Final:   {metrics.final_loss:.4f}")
    logger.info(f"   Best:    {metrics.best_loss:.4f} (step {metrics.best_loss_step})")
    logger.info(f"   Reduction: {loss_reduction:.1f}%")
    logger.info(f"\n⚡ Performance:")
    logger.info(f"   Training time: {format_time(training_time)}")
    logger.info(f"   Total tokens: {metrics.total_tokens:,}")
    logger.info(f"   Avg throughput: {avg_tps/1000:.1f}K tok/s")
    logger.info(f"   Peak throughput: {peak_tps/1000:.1f}K tok/s")
    logger.info(f"\n📈 Model Assessment:")
    for key, value in report["model_assessment"].items():
        logger.info(f"   {key}: {value}")
    logger.info(f"\n✅ Training Assessment:")
    for key, value in report["training_assessment"].items():
        logger.info(f"   {key}: {value}")
    logger.info(f"\n📁 Report saved: {report_path}")
    logger.info("=" * 70)

    return report_path
