#!/usr/bin/env python3
"""HYDRA training knob sweep (short runs).

Runs several short training jobs (few steps) and compares:
- Loss after N steps (proxy for "learn faster" under identical data/steps)
- Throughput (tokens/sec)
- Peak GPU memory (allocated + reserved)

Outputs:
- JSON results to reports/training_sweeps/
- LA3-style 3-panel bar chart to reports/training_sweeps/

Notes
- Uses the real training stack (hydra.training.Trainer).
- Keep steps small; compile + dataloader warmup can dominate at tiny step counts.
- Use env vars for attention pattern / LA3 variant without code edits:
  - HYDRA_MOR_ATTENTION_PATTERN_NAME=lla3_only | ccqa_only | lla3x3+ccqa
  - HYDRA_LLA3_VARIANT=chunk_loop (or other supported LA3 variants)
  - HYDRA_CCQA_USE_FUSED_KERNEL=1/0
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from hydra.training import TrainingConfig, Trainer, MODEL_SIZE_CONFIGS


def _parse_int_list(csv: str | None) -> list[int]:
    if not csv:
        return []
    out: list[int] = []
    for part in csv.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def _apply_cfg_overrides(cfg: TrainingConfig, overrides: dict[str, Any]) -> None:
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise ValueError(f"Unknown TrainingConfig field override: {k}")
        setattr(cfg, k, v)


def _set_env(temp_env: dict[str, str | None]) -> dict[str, str | None]:
    prev: dict[str, str | None] = {}
    for k, v in temp_env.items():
        prev[k] = os.environ.get(k)
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = str(v)
    return prev


def _restore_env(prev: dict[str, str | None]) -> None:
    for k, v in prev.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _peak_mem_gb() -> dict[str, float]:
    if not torch.cuda.is_available():
        return {"max_allocated_gb": 0.0, "max_reserved_gb": 0.0}
    return {
        "max_allocated_gb": float(torch.cuda.max_memory_allocated()) / 1e9,
        "max_reserved_gb": float(torch.cuda.max_memory_reserved()) / 1e9,
    }


def _reset_cuda_stats() -> None:
    if not torch.cuda.is_available():
        return
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()


def _make_config(args: argparse.Namespace) -> TrainingConfig:
    size_config = MODEL_SIZE_CONFIGS.get(args.model_size, MODEL_SIZE_CONFIGS["100M"])

    cfg = TrainingConfig(
        architecture="mod_mor",
        mode=args.mode,
        resume_from=None,
        resume_ignore_ckpt_lr=False,
        resume_lr_override=0.0,
        mor_enable_pct=args.mor_enable_pct,
        mor_already_enabled=False,
        mod_capacity=args.mod_capacity,
        mod_enable_pct=args.mod_enable_pct,
        mod_enable_loss_threshold=args.mod_enable_loss_threshold,
        mod_force_enable_pct=args.mod_force_enable_pct,
        mod_loss_aware_weight=args.mod_loss_aware_weight,
        mor_adaptive=args.mor_adaptive,
        aux_scale=args.aux_scale,
        ponder_scale=args.ponder_scale,
        mor_advantage_loss_scale=args.mor_advantage_loss_scale,
        adaptive_lr=args.adaptive_lr,
        adaptive_metric=args.adaptive_metric,
        adaptive_min_trigger_pct=args.adaptive_min_trigger_pct,
        use_swa=False,
        swa_start_pct=0.75,
        batch_filter=False,
        batch_filter_threshold=2.5,
        model_size=args.model_size,
        mod_mor_dim=size_config["mod_mor_dim"],
        n_mor_blocks=size_config["n_mor_blocks"],
        mor_recursions=size_config["mor_recursions"],
        mod_mor_n_heads=size_config["mod_mor_n_heads"],
        mod_mor_n_kv_heads=size_config["mod_mor_n_kv_heads"],
        dim=768,
        n_macro_blocks=3,
        n_heads=12,
        n_kv_heads=3,
        use_triton_kernels=True,
        use_chunked_ce=args.chunked_ce,
        chunked_ce_size=args.chunked_ce_size,
        dataset_name=args.dataset,
        use_compile=args.compile,
        dtype=args.dtype,
        gradient_checkpointing=args.gradient_checkpointing,
        checkpoint_every_n=args.checkpoint_every,
        halt_on_spike=False,
        use_8bit_adam=args.use_8bit_adam,
        log_interval=max(1, args.log_interval),
        save_interval=max(10_000, args.steps + 1),
        seed=args.seed,
    )

    # NOTE: Trainer uses `max_seq_len` (and `seq_steps`) to drive the active
    # dataloader seq_len. We keep seq_steps empty and set max_seq_len directly.
    cfg.seq_steps = ()
    cfg.max_steps = int(args.steps)
    cfg.max_seq_len = int(args.seq_len)

    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.grad_accum is not None:
        cfg.grad_accum_steps = args.grad_accum

    return cfg


def _run_one(name: str, cfg: TrainingConfig, env: dict[str, str | None]) -> dict[str, Any]:
    prev = _set_env(env)
    try:
        _reset_cuda_stats()
        start = time.time()
        trainer = Trainer(cfg)
        try:
            metrics = trainer.train()
        finally:
            trainer.close()
        wall = time.time() - start

        mem = _peak_mem_gb()
        # Prefer last reported throughput; also compute mean over last 20%.
        tps_series = list(metrics.tokens_per_sec)
        tail = tps_series[max(0, int(len(tps_series) * 0.8)) :]
        tps_tail_mean = float(sum(tail) / max(1, len(tail)))

        losses = list(metrics.losses)
        loss_final = float(metrics.final_loss) if getattr(metrics, "final_loss", 0.0) else (float(losses[-1]) if losses else float("nan"))
        loss_initial = float(metrics.initial_loss) if getattr(metrics, "initial_loss", 0.0) else (float(losses[0]) if losses else float("nan"))

        out = {
            "name": name,
            "env": {k: (None if v is None else str(v)) for k, v in env.items()},
            "config": {
                "model_size": cfg.model_size,
                "mode": cfg.mode,
                "dataset": cfg.dataset_name,
                "seq_len": int(getattr(cfg, "max_seq_len", 0)),
                "batch_size": cfg.batch_size,
                "grad_accum": cfg.grad_accum_steps,
                "compile": cfg.use_compile,
                "dtype": str(cfg.dtype),
                "gradient_checkpointing": cfg.gradient_checkpointing,
                "chunked_ce": cfg.use_chunked_ce,
                "mod_capacity": cfg.mod_capacity,
                "mor_adaptive": cfg.mor_adaptive,
            },
            "summary": {
                "steps": cfg.max_steps,
                "loss_initial": loss_initial,
                "loss_final": loss_final,
                "loss_delta": (loss_initial - loss_final) if (loss_initial == loss_initial and loss_final == loss_final) else float("nan"),
                "best_loss": float(metrics.best_loss),
                "tokens_per_sec_tail_mean": tps_tail_mean,
                "wall_time_s": float(wall),
                **mem,
            },
            "series": {
                "loss": losses,
                "tokens_per_sec": tps_series,
                "step_time_s": list(metrics.step_times),
            },
        }
        return out
    finally:
        _restore_env(prev)


def _pick_best(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not results:
        return None

    def key_fn(r: dict[str, Any]) -> tuple[float, float, float]:
        # Primary: lowest final loss
        loss_final = float(r["summary"]["loss_final"])
        # Secondary: lower reserved mem
        mem_reserved = float(r["summary"]["max_reserved_gb"])
        # Tertiary: higher throughput
        tps = float(r["summary"]["tokens_per_sec_tail_mean"])
        return (loss_final, mem_reserved, -tps)

    return sorted(results, key=key_fn)[0]


def _plot(results: list[dict[str, Any]], out_png: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        print("matplotlib not available, skipping plot")
        return

    names = [r["name"] for r in results]
    x = np.arange(len(names))

    loss_final = [r["summary"]["loss_final"] for r in results]
    tps = [r["summary"]["tokens_per_sec_tail_mean"] for r in results]

    alloc = [r["summary"]["max_allocated_gb"] for r in results]
    reserv = [r["summary"]["max_reserved_gb"] for r in results]
    reserv_extra = [max(0.0, reserv[i] - alloc[i]) for i in range(len(results))]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    colors = ["#3498db", "#2ecc71", "#e74c3c", "#95a5a6", "#f39c12", "#9b59b6"]
    palette = [colors[i % len(colors)] for i in range(len(results))]

    ax1 = axes[0]
    ax1.bar(x, loss_final, color=palette)
    ax1.set_title("Loss After N Steps (lower is better)")
    ax1.set_ylabel("Loss")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=20, ha="right")
    ax1.grid(axis="y", alpha=0.3)

    ax2 = axes[1]
    ax2.bar(x, tps, color=palette)
    ax2.set_title("Training Throughput")
    ax2.set_ylabel("Tokens/sec (tail mean)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=20, ha="right")
    ax2.grid(axis="y", alpha=0.3)

    ax3 = axes[2]
    ax3.bar(x, alloc, color=palette, alpha=0.7)
    ax3.bar(x, reserv_extra, bottom=alloc, color=palette, alpha=1.0, hatch="//")
    ax3.set_title("GPU Memory: allocated (solid) + reserved extra (hatched)")
    ax3.set_ylabel("GB")
    ax3.set_xticks(x)
    ax3.set_xticklabels(names, rotation=20, ha="right")
    ax3.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    p = argparse.ArgumentParser(description="HYDRA knob sweep (short training runs)")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--mode", type=str, default="testing", choices=["testing", "production", "chinchilla_third"])
    p.add_argument("--model_size", type=str, default="100M")
    p.add_argument("--dataset", type=str, default="finefineweb-sequential")
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument(
        "--seq_lens",
        type=str,
        default=None,
        help="Optional comma-separated list (e.g. 256,512,1024) to sweep multiple sequence lengths.",
    )
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--grad_accum", type=int, default=None)

    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--checkpoint_every", type=int, default=2)
    p.add_argument("--chunked_ce", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--chunked_ce_size", type=int, default=4096)
    p.add_argument("--use_8bit_adam", action="store_true")

    p.add_argument("--mod_capacity", type=float, default=0.5)
    p.add_argument("--mod_enable_pct", type=float, default=0.10)
    p.add_argument("--mod_force_enable_pct", type=float, default=0.20)
    p.add_argument("--mod_enable_loss_threshold", type=float, default=5.0)
    p.add_argument("--mod_loss_aware_weight", type=float, default=0.0)

    p.add_argument("--mor_adaptive", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--mor_enable_pct", type=float, default=0.30)
    p.add_argument("--aux_scale", type=float, default=0.1)
    p.add_argument("--ponder_scale", type=float, default=0.01)
    p.add_argument("--mor_advantage_loss_scale", type=float, default=0.1)

    p.add_argument("--adaptive_lr", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--adaptive_metric", type=str, default="eval", choices=["train", "eval"])
    p.add_argument("--adaptive_min_trigger_pct", type=float, default=0.50)

    p.add_argument("--log_interval", type=int, default=5)
    p.add_argument("--seed", type=int, default=1234)

    p.add_argument(
        "--variants",
        type=str,
        default="baseline,lla3_only,ccqa_only",
        help="Comma-separated preset variants: baseline, lla3_only, ccqa_only, ccqa_unfused",
    )

    p.add_argument(
        "--emit_seq_policy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write a seq-len policy JSON mapping seq_len -> runtime knob overrides.",
    )

    args = p.parse_args()

    out_dir = Path("reports") / "training_sweeps" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = _make_config(args)

    # Variant specs can override env vars (rebuild-required knobs) and cfg fields (runtime knobs).
    # NOTE: env overrides that change the model structure (e.g. attention pattern) cannot be
    # applied mid-run during stepped seq_len; those variants are still useful for per-seq sweeps.
    presets: dict[str, dict[str, Any]] = {
        "baseline": {
            "env": {
                "HYDRA_MOR_ATTENTION_PATTERN_NAME": None,
                "HYDRA_LLA3_VARIANT": None,
                "HYDRA_CCQA_USE_FUSED_KERNEL": None,
            },
            "cfg": {},
            "policy": {},
        },
        "lla3_only": {
            "env": {"HYDRA_MOR_ATTENTION_PATTERN_NAME": "lla3_only"},
            "cfg": {},
            "policy": {},
        },
        "ccqa_only": {
            "env": {"HYDRA_MOR_ATTENTION_PATTERN_NAME": "ccqa_only"},
            "cfg": {},
            "policy": {},
        },
        "ccqa_unfused": {
            "env": {
                "HYDRA_MOR_ATTENTION_PATTERN_NAME": "ccqa_only",
                "HYDRA_CCQA_USE_FUSED_KERNEL": "0",
            },
            "cfg": {},
            "policy": {"ccqa_use_fused_kernel": False},
        },
        # Runtime-only knobs (apply during stepped seq_len without rebuilding model)
        "ckpt_all": {
            "env": {},
            "cfg": {"gradient_checkpointing": True, "checkpoint_every_n": 1},
            "policy": {"gradient_checkpointing": True, "checkpoint_every_n": 1},
        },
        "ckpt_every4": {
            "env": {},
            "cfg": {"gradient_checkpointing": True, "checkpoint_every_n": 4},
            "policy": {"gradient_checkpointing": True, "checkpoint_every_n": 4},
        },
        "chunked_ce_off": {
            "env": {},
            "cfg": {"use_chunked_ce": False},
            "policy": {"use_chunked_ce": False},
        },
    }

    requested = [v.strip() for v in args.variants.split(",") if v.strip()]

    seq_lens = _parse_int_list(args.seq_lens) if args.seq_lens else [int(args.seq_len)]

    results_by_seq: dict[int, list[dict[str, Any]]] = {}
    policy_by_seq: dict[int, dict[str, Any]] = {}
    recommended_by_seq: dict[int, str] = {}

    for seq_len in seq_lens:
        print("\n" + "#" * 90)
        print(f"SEQ_LEN SWEEP: {seq_len}")
        print("#" * 90)
        cfg_for_seq = copy.deepcopy(base_cfg)
        cfg_for_seq.seq_steps = ()
        cfg_for_seq.max_steps = int(args.steps)
        cfg_for_seq.max_seq_len = int(seq_len)

        seq_results: list[dict[str, Any]] = []
        for name in requested:
            spec = presets.get(name)
            if spec is None:
                raise SystemExit(f"Unknown variant '{name}'. Known: {sorted(presets.keys())}")
            env = dict(spec.get("env", {}))
            cfg_over = dict(spec.get("cfg", {}))
            cfg = copy.deepcopy(cfg_for_seq)
            _apply_cfg_overrides(cfg, cfg_over)
            print("\n" + "=" * 80)
            print(f"Variant: {name}  (seq_len={seq_len})")
            print("=" * 80)
            seq_results.append(_run_one(name, cfg, env))

        results_by_seq[int(seq_len)] = seq_results

        best = _pick_best(seq_results)
        if best is not None:
            recommended_by_seq[int(seq_len)] = str(best["name"])
            best_spec = presets.get(str(best["name"]), {})
            policy_by_seq[int(seq_len)] = dict(best_spec.get("policy", {}))

        out_png = out_dir / f"sweep_comparison_seq{seq_len}.png"
        _plot(seq_results, out_png)
        if out_png.exists():
            print(f"✓ Wrote: {out_png}")

    out_json = out_dir / "sweep_results.json"
    with open(out_json, "w") as f:
        json.dump(
            {
                "seq_lens": seq_lens,
                "results_by_seq_len": {str(k): v for k, v in results_by_seq.items()},
                "recommended_by_seq_len": {str(k): v for k, v in recommended_by_seq.items()},
                "policy_by_seq_len": {str(k): v for k, v in policy_by_seq.items()},
            },
            f,
            indent=2,
        )
    print(f"\n✓ Wrote: {out_json}")

    if args.emit_seq_policy and policy_by_seq:
        out_policy = out_dir / "seq_len_policy.json"
        with open(out_policy, "w") as f:
            json.dump(
                {
                    "version": 1,
                    "default": {},
                    "by_seq_len": {str(k): v for k, v in policy_by_seq.items()},
                },
                f,
                indent=2,
            )
        print(f"✓ Wrote: {out_policy}")


if __name__ == "__main__":
    main()
