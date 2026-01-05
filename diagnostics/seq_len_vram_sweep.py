#!/usr/bin/env python3
"""
Auto VRAM/throughput sweep across seq_len for HYDRA model sizes using Trainer.

Purpose:
- For given `model_size` presets (e.g., 500M, 750M, 1B), find the largest microbatch
    (batch_size) that fits in VRAM at sequence lengths 1024 and 2048 with recommended
    runtime knobs (AMP, torch.compile, gradient checkpointing, chunked CE, 8-bit Adam when needed).
- Measure tokens/sec and peak GPU memory.
- Emit JSON + Markdown summary under reports/.

Notes:
- REQUIRES: CUDA to perform VRAM sweeps. Do not run while other GPU workloads are active.

Usage:
    source /home/tim/venvs/llm/bin/activate && \
    python diagnostics/seq_len_vram_sweep.py \
        --model_sizes 500M,750M,1B --seq_lens 1024,2048 --device cuda \
        --compile --dtype bfloat16 --gradient_checkpointing --chunked_ce_size 4096 \
        --steps 30 --warmup 5

Outputs:
- reports/seq_len_vram_sweep_<timestamp>/summary.json
- reports/seq_len_vram_sweep_<timestamp>/report.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

# Ensure repo root on path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hydra.training import TrainingConfig, Trainer, MODEL_SIZE_CONFIGS


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _cuda_mem_info() -> Tuple[Optional[int], Optional[int]]:
    if not torch.cuda.is_available():
        return None, None
    free_b, total_b = torch.cuda.mem_get_info()
    return int(free_b), int(total_b)


def _peak_mem_bytes() -> Tuple[int, int]:
    alloc = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
    reserv = int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else 0
    return alloc, reserv


def _reset_cuda_stats() -> None:
    if not torch.cuda.is_available():
        return
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()


def _make_cfg(
    model_size: str,
    seq_len: int,
    *,
    compile: bool,
    dtype: str,
    gradient_checkpointing: bool,
    chunked_ce_size: int,
    use_8bit_adam: bool,
    steps: int,
    batch_size: Optional[int] = None,
    grad_accum: Optional[int] = None,
) -> TrainingConfig:
    size_cfg = MODEL_SIZE_CONFIGS.get(model_size)
    if not size_cfg:
        raise SystemExit(f"Unknown model_size '{model_size}'. Known: {sorted(MODEL_SIZE_CONFIGS.keys())}")

    cfg = TrainingConfig(
        architecture="mod_mor",
        mode="testing",
        resume_from=None,
        resume_ignore_ckpt_lr=False,
        resume_lr_override=0.0,
        mor_enable_pct=0.30,
        mor_already_enabled=False,
        mod_capacity=0.5,
        mod_enable_min_step=3000,
        mod_enable_mor_early_exit_threshold=0.38,
        mod_enable_loss_threshold=4.5,
        mod_loss_aware_weight=0.0,
        mor_adaptive=True,
        aux_scale=float(size_cfg.get("aux_scale", 0.02)),
        ponder_scale=0.01,
        mor_advantage_loss_scale=0.10,
        adaptive_lr=True,
        adaptive_metric="eval",
        adaptive_min_trigger_pct=0.50,
        use_swa=False,
        swa_start_pct=0.75,
        batch_filter=False,
        batch_filter_threshold=2.5,
        model_size=model_size,
        mod_mor_dim=int(size_cfg["mod_mor_dim"]),
        n_mor_blocks=int(size_cfg["n_mor_blocks"]),
        mor_recursions=int(size_cfg["mor_recursions"]),
        mod_mor_n_heads=int(size_cfg["mod_mor_n_heads"]),
        mod_mor_n_kv_heads=int(size_cfg["mod_mor_n_kv_heads"]),
        dim=768,
        n_macro_blocks=3,
        n_heads=12,
        n_kv_heads=3,
        use_triton_kernels=True,
        use_chunked_ce=True,
        chunked_ce_size=int(chunked_ce_size),
        dataset_name="finefineweb-sequential",
        use_compile=bool(compile),
        dtype=str(dtype),
        gradient_checkpointing=bool(gradient_checkpointing),
        checkpoint_every_n=2,
        halt_on_spike=False,
        use_8bit_adam=bool(use_8bit_adam),
        log_interval=5,
        save_interval=max(10_000, steps + 1),
        seed=1234,
    )

    cfg.seq_steps = ()
    cfg.max_steps = int(steps)
    cfg.max_seq_len = int(seq_len)

    if batch_size is not None:
        cfg.batch_size = int(batch_size)
    else:
        cfg.batch_size = int(size_cfg.get("default_batch_size", 4))

    if grad_accum is not None:
        cfg.grad_accum_steps = int(grad_accum)
    else:
        cfg.grad_accum_steps = int(size_cfg.get("default_grad_accum", 4))

    return cfg


def _run_try(cfg: TrainingConfig) -> Tuple[bool, Dict[str, Any]]:
    _reset_cuda_stats()
    try:
        start = time.time()
        trainer = Trainer(cfg)
        try:
            metrics = trainer.train()
        finally:
            trainer.close()
        wall = time.time() - start
        alloc_b, reserv_b = _peak_mem_bytes()
        tps_series = list(metrics.tokens_per_sec)
        tail = tps_series[max(0, int(len(tps_series) * 0.8)) :]
        tps_tail_mean = float(sum(tail) / max(1, len(tail)))
        out = {
            "ok": True,
            "final_loss": float(getattr(metrics, "final_loss", 0.0) or (metrics.losses[-1] if metrics.losses else float("nan"))),
            "tokens_per_sec_tail_mean": tps_tail_mean,
            "wall_time_s": float(wall),
            "peak_alloc_bytes": int(alloc_b),
            "peak_reserved_bytes": int(reserv_b),
            "batch_size": int(cfg.batch_size),
            "grad_accum": int(cfg.grad_accum_steps),
        }
        return True, out
    except RuntimeError as e:
        msg = str(e).lower()
        is_oom = ("out of memory" in msg) or ("cuda" in msg and "memory" in msg)
        return False, {"ok": False, "error": "oom" if is_oom else "runtime_error", "message": str(e)}


def _search_max_batch(cfg_base: TrainingConfig, target_frac: float, max_try: int = 256) -> Tuple[int, List[Dict[str, Any]]]:
    # Target bytes based on current free VRAM and total VRAM
    free_b, total_b = _cuda_mem_info()
    if isinstance(total_b, int) and isinstance(free_b, int):
        target_b = min(int(total_b * target_frac), int(free_b * 0.98))
    elif isinstance(total_b, int):
        target_b = int(total_b * target_frac)
    else:
        target_b = None

    trials: List[Dict[str, Any]] = []

    def under_target(rec: Dict[str, Any]) -> bool:
        if target_b is None:
            return True
        peak = rec.get("peak_reserved_bytes", rec.get("peak_alloc_bytes", 0))
        return isinstance(peak, int) and peak <= target_b

    # Exponential then binary search
    lo = 0
    hi = None
    bs = 1
    while bs <= max_try:
        cfg = _make_cfg(
            cfg_base.model_size,
            cfg_base.max_seq_len,
            compile=cfg_base.use_compile,
            dtype=str(cfg_base.dtype).replace("torch.", ""),
            gradient_checkpointing=cfg_base.gradient_checkpointing,
            chunked_ce_size=cfg_base.chunked_ce_size,
            use_8bit_adam=cfg_base.use_8bit_adam,
            steps=cfg_base.max_steps,
            batch_size=bs,
            grad_accum=cfg_base.grad_accum_steps,
        )
        ok, rec = _run_try(cfg)
        rec["seq_len"] = int(cfg_base.max_seq_len)
        trials.append(rec)
        if ok and under_target(rec):
            lo = bs
            bs *= 2
            continue
        hi = bs
        break

    if lo == 0:
        return 0, trials
    if hi is None:
        return lo, trials

    # Binary search refinement
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        cfg = _make_cfg(
            cfg_base.model_size,
            cfg_base.max_seq_len,
            compile=cfg_base.use_compile,
            dtype=str(cfg_base.dtype).replace("torch.", ""),
            gradient_checkpointing=cfg_base.gradient_checkpointing,
            chunked_ce_size=cfg_base.chunked_ce_size,
            use_8bit_adam=cfg_base.use_8bit_adam,
            steps=cfg_base.max_steps,
            batch_size=mid,
            grad_accum=cfg_base.grad_accum_steps,
        )
        ok, rec = _run_try(cfg)
        rec["seq_len"] = int(cfg_base.max_seq_len)
        trials.append(rec)
        if ok and under_target(rec):
            lo = mid
        else:
            hi = mid

    return lo, trials


def main() -> int:
    p = argparse.ArgumentParser(description="Seq-len VRAM sweep for HYDRA Trainer presets")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--model_sizes", default="500M")
    p.add_argument("--seq_lens", default="1024,2048")
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--chunked_ce_size", type=int, default=4096)
    p.add_argument("--use_8bit_adam", action="store_true")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--target_vram_frac", type=float, default=0.92)
    p.add_argument("--out_dir", default="")
    args = p.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available")

    sizes = [s.strip() for s in args.model_sizes.split(",") if s.strip()]
    seqs = [int(s.strip()) for s in args.seq_lens.split(",") if s.strip()]

    out_dir = Path(args.out_dir) if args.out_dir else (REPO_ROOT / "reports" / f"seq_len_vram_sweep_{_now_tag()}")
    out_dir.mkdir(parents=True, exist_ok=True)

    free_b, total_b = _cuda_mem_info() if args.device == "cuda" else (None, None)
    meta = {
        "timestamp": datetime.now().isoformat(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_total_mem_gb": float(total_b) / (1024**3) if isinstance(total_b, int) else None,
        "cuda_free_mem_gb": float(free_b) / (1024**3) if isinstance(free_b, int) else None,
        "target_vram_frac": float(args.target_vram_frac),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    summary: List[Dict[str, Any]] = []

    for size in sizes:
        for seq in seqs:
            print(f"\n=== Sweep: model_size={size} seq_len={seq} ===", flush=True)
            # Enable 8-bit Adam if required by preset or flag
            preset = MODEL_SIZE_CONFIGS.get(size, {})
            use_8bit = bool(args.use_8bit_adam or preset.get("requires_8bit_adam", False))
            cfg_base = _make_cfg(
                size,
                seq,
                compile=bool(args.compile),
                dtype=str(args.dtype),
                gradient_checkpointing=bool(args.gradient_checkpointing),
                chunked_ce_size=int(args.chunked_ce_size),
                use_8bit_adam=use_8bit,
                steps=int(args.steps),
            )

            max_bs, trials = _search_max_batch(cfg_base, float(args.target_vram_frac))
            best_rec = None
            # Validate at max_bs with compile flag (already using compile above)
            if max_bs > 0:
                cfg_final = _make_cfg(
                    size,
                    seq,
                    compile=bool(args.compile),
                    dtype=str(args.dtype),
                    gradient_checkpointing=bool(args.gradient_checkpointing),
                    chunked_ce_size=int(args.chunked_ce_size),
                    use_8bit_adam=use_8bit,
                    steps=int(args.steps),
                    batch_size=int(max_bs),
                    grad_accum=cfg_base.grad_accum_steps,
                )
                ok, rec = _run_try(cfg_final)
                rec["seq_len"] = int(seq)
                rec["model_size"] = size
                rec["compile"] = bool(args.compile)
                rec["dtype"] = str(args.dtype)
                rec["gradient_checkpointing"] = bool(args.gradient_checkpointing)
                rec["chunked_ce_size"] = int(args.chunked_ce_size)
                rec["use_8bit_adam"] = bool(use_8bit)
                rec["validated_at_max_bs"] = True
                trials.append(rec)
                best_rec = rec

            out_json = out_dir / f"{size}_seq{seq}_trials.json"
            out_json.write_text(json.dumps(trials, indent=2), encoding="utf-8")
            if best_rec:
                summary.append(best_rec)
                bs_v = best_rec.get("batch_size")
                tps_v = float(best_rec.get("tokens_per_sec_tail_mean", 0.0))
                peak_b = best_rec.get("peak_reserved_bytes", best_rec.get("peak_alloc_bytes", 0))
                peak_gb = (float(peak_b) / (1024**3)) if isinstance(peak_b, (int, float)) else 0.0
                print(
                    f"✓ {size} seq={seq}: bs={bs_v} tps={tps_v/1e3:.1f}K peakGB={peak_gb:.2f}",
                    flush=True,
                )
            else:
                summary.append({
                    "model_size": size,
                    "seq_len": int(seq),
                    "error": "no_batch_fit",
                })
                print(f"⚠ {size} seq={seq}: no batch size fit under target VRAM", flush=True)

    # Write summary
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Write human-friendly report
    md_lines = [f"# Seq-len VRAM Sweep ({datetime.now().isoformat()})\n\n"]
    md_lines.append(f"- Device: {meta.get('cuda_device')}\n")
    if isinstance(meta.get("cuda_total_mem_gb"), (int, float)):
        md_lines.append(f"- GPU VRAM: {float(meta['cuda_total_mem_gb']):.2f} GB total\n")
    md_lines.append("\n## Recommended CLI per model-size and seq_len\n")
    for rec in summary:
        if not isinstance(rec, dict) or rec.get("error"):
            ms = rec.get("model_size")
            seq = rec.get("seq_len")
            md_lines.append(f"- {ms} seq={seq}: no batch fit under target VRAM\n")
            continue
        ms = rec["model_size"]
        seq = rec["seq_len"]
        bs = rec["batch_size"]
        ga = rec["grad_accum"]
        use8 = rec.get("use_8bit_adam", False)
        cmd = (
            f"python trainer.py --model_size {ms} --batch_size {bs} --grad_accum {ga} "
            f"--seq_len {seq} --compile --dtype bfloat16 --gradient_checkpointing "
            f"--checkpoint_every 2 --chunked_ce --chunked_ce_size {int(rec['chunked_ce_size'])} "
        )
        if use8:
            cmd += "--8bit_adam "
        md_lines.append(
            f"- {ms} @ seq={seq}: tokens/sec≈{rec['tokens_per_sec_tail_mean']/1e3:.1f}K, peakVRAM≈{rec['peak_reserved_bytes']/(1024**3):.2f} GB\n  CLI: {cmd.strip()}\n"
        )

    (out_dir / "report.md").write_text("".join(md_lines), encoding="utf-8")
    print(f"\nWrote sweep report to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
