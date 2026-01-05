#!/usr/bin/env python3
"""Exhaustive-ish throughput sweep for HYDRA (MoD+MoR+CCGQA kept ON).

This script:
- Sweeps presets (100m/500m), seq lens, AMP dtype, compile on/off, grad-ckpt, grad-accum, chunked CE.
- Auto-finds a near-max microbatch size under a target VRAM fraction (OOM-safe).
- Writes raw results (JSONL), a CSV summary, a Markdown report, and plots.

It uses the same training-critical path as diagnostics/tall_skinny_bench.py:
  forward_hidden_with_losses + fused_chunked_cross_entropy

Run example:
  /home/tim/venvs/llm/bin/python diagnostics/throughput_sweep.py \
    --presets 100m,500m --seq_lens 512,1024,2048 --compile 1,0 \
    --grad_accum 1,2 --grad_ckpt_every_n 2,0 --chunked_ce 4096,2048
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import torch

# Ensure repo root is importable even when executed from diagnostics/
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _is_oom(err: BaseException) -> bool:
    msg = str(err).lower()
    return "out of memory" in msg or "cuda error" in msg and "memory" in msg


def _cleanup_cuda() -> None:
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _device_total_bytes() -> Optional[int]:
    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(0)
    return int(props.total_memory)


def _cuda_mem_info() -> Tuple[Optional[int], Optional[int]]:
    """Returns (free_bytes, total_bytes) if CUDA is available, else (None, None)."""
    if not torch.cuda.is_available():
        return None, None
    free_b, total_b = torch.cuda.mem_get_info()
    return int(free_b), int(total_b)


def _format_gb(nbytes: Optional[int]) -> str:
    if not isinstance(nbytes, int):
        return "n/a"
    return f"{nbytes / (1024**3):.2f}"


def _bench_via_subprocess(
    *,
    preset: str,
    device: str,
    batch_size: int,
    seq_len: int,
    use_compile: bool,
    compile_mode: str,
    use_amp: bool,
    dtype: str,
    grad_accum_steps: int,
    grad_checkpoint_every_n: int,
    chunked_ce_size: int,
    steps: int,
    warmup: int,
    mor_attention_pattern: str,
    use_ada_rmsnorm: bool,
    tmp_out: Path,
) -> Tuple[bool, Dict[str, Any]]:
    """Run one bench in a fresh Python subprocess.

    This guarantees the sweep never holds multiple models/compiled graphs in memory.
    """

    env = os.environ.copy()
    env["HYDRA_BENCH_OUT"] = str(tmp_out)

    # Optional: enable AdaRMSNorm (MoR blocks only).
    if use_ada_rmsnorm:
        env["HYDRA_USE_ADA_RMSNORM"] = "1"
    """
    Exhaustive-ish throughput sweep for HYDRA (MoD+MoR+CCGQA kept ON).

    This script:
    - Sweeps presets (100m/500m), seq lens, AMP dtype, compile on/off, grad-ckpt, grad-accum, chunked CE.
    - Auto-finds a near-max microbatch size under a target VRAM fraction (OOM-safe).
    - Writes raw results (JSONL), a CSV summary, a Markdown report, and plots.

    Notes:
    - REQUIRES: CUDA for meaningful throughput sweeps. Ensure no other processes occupy the GPU.

    It uses the same training-critical path as diagnostics/tall_skinny_bench.py:
        forward_hidden_with_losses + fused_chunked_cross_entropy

    Run example:
        /home/tim/venvs/llm/bin/python diagnostics/throughput_sweep.py \
            --presets 100m,500m --seq_lens 512,1024,2048 --compile 1,0 \
            --grad_accum 1,2 --grad_ckpt_every_n 2,0 --chunked_ce 4096,2048
    """
    cmd = [
        sys.executable,
        str(REPO_ROOT / "diagnostics" / "tall_skinny_bench.py"),
        "--preset",
        preset,
        "--device",
        device,
        "--compile",
        "1" if use_compile else "0",
        "--compile_mode",
        compile_mode,
        "--amp",
        "1" if use_amp else "0",
        "--dtype",
        dtype,
        "--batch_size",
        str(batch_size),
        "--seq_len",
        str(seq_len),
        "--steps",
        str(steps),
        "--warmup",
        str(warmup),
        "--grad_accum_steps",
        str(grad_accum_steps),
        "--grad_ckpt_every_n",
        str(grad_checkpoint_every_n),
        "--chunked_ce_size",
        str(chunked_ce_size),
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout or "").strip()
        if device == "cuda" and "out of memory" in msg.lower():
            return False, {"error": "oom", "message": msg}
        return False, {"error": "subprocess_failed", "message": msg, "returncode": proc.returncode}

    try:
        data = json.loads(tmp_out.read_text(encoding="utf-8"))
        if isinstance(data, list) and data:
            return True, data[0]
        return False, {"error": "no_output", "message": "HYDRA_BENCH_OUT written but empty"}
    except Exception as e:
        return False, {"error": "bad_output", "message": str(e)}


def _try_bench_once(
    preset: str,
    *,
    device: str,
    batch_size: int,
    seq_len: int,
    use_compile: bool,
    compile_mode: str,
    use_amp: bool,
    dtype: str,
    grad_accum_steps: int,
    grad_checkpoint_every_n: int,
    chunked_ce_size: int,
    steps: int,
    warmup: int,
    mor_attention_pattern: str,
    use_ada_rmsnorm: bool,
    tmp_dir: Path,
    run_id: int,
) -> Tuple[bool, Dict[str, Any]]:
    tmp_out = tmp_dir / f"run_{run_id}.json"
    if tmp_out.exists():
        tmp_out.unlink()

    ok, out = _bench_via_subprocess(
        preset=preset,
        device=device,
        batch_size=batch_size,
        seq_len=seq_len,
        use_compile=use_compile,
        compile_mode=compile_mode,
        use_amp=use_amp,
        dtype=dtype,
        grad_accum_steps=grad_accum_steps,
        grad_checkpoint_every_n=grad_checkpoint_every_n,
        chunked_ce_size=chunked_ce_size,
        steps=steps,
        warmup=warmup,
        mor_attention_pattern=mor_attention_pattern,
        use_ada_rmsnorm=use_ada_rmsnorm,
        tmp_out=tmp_out,
    )
    if not ok and device == "cuda" and out.get("error") == "oom":
        _cleanup_cuda()
    return ok, out


def _find_max_microbatch(
    preset: str,
    *,
    device: str,
    seq_len: int,
    target_bytes: Optional[int],
    max_batch: int,
    search_steps: int,
    search_warmup: int,
    compile_mode: str,
    use_amp: bool,
    dtype: str,
    grad_accum_steps: int,
    grad_checkpoint_every_n: int,
    chunked_ce_size: int,
    mor_attention_pattern: str,
    use_ada_rmsnorm: bool,
    tmp_dir: Path,
    run_id_start: int,
) -> Tuple[int, List[Dict[str, Any]]]:
    """Find the largest batch_size that stays under target_bytes (approx) without OOM.

    This uses compile=False for the search to avoid repeated compilation.
    The returned batch is later validated with the requested compile flag.
    """

    trials: List[Dict[str, Any]] = []

    def ok_under_target(out: Dict[str, Any]) -> bool:
        if target_bytes is None:
            return True
        peak = out.get("peak_mem_bytes")
        if not isinstance(peak, int):
            return True
        return peak <= target_bytes

    low_ok = 0
    high_fail = None

    bs = 1
    run_id = run_id_start
    while bs <= max_batch:
        ok, out = _try_bench_once(
            preset,
            device=device,
            batch_size=bs,
            seq_len=seq_len,
            use_compile=False,
            compile_mode=compile_mode,
            use_amp=use_amp,
            dtype=dtype,
            grad_accum_steps=grad_accum_steps,
            grad_checkpoint_every_n=grad_checkpoint_every_n,
            chunked_ce_size=chunked_ce_size,
            steps=search_steps,
            warmup=search_warmup,
            mor_attention_pattern=mor_attention_pattern,
            use_ada_rmsnorm=use_ada_rmsnorm,
            tmp_dir=tmp_dir,
            run_id=run_id,
        )
        run_id += 1
        trials.append({"batch_size": bs, "ok": ok, "out": out})
        if ok and ok_under_target(out):
            low_ok = bs
            bs *= 2
            continue
        high_fail = bs
        break

    if low_ok == 0:
        return 0, trials

    if high_fail is None:
        return low_ok, trials

    lo = low_ok
    hi = high_fail
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        ok, out = _try_bench_once(
            preset,
            device=device,
            batch_size=mid,
            seq_len=seq_len,
            use_compile=False,
            compile_mode=compile_mode,
            use_amp=use_amp,
            dtype=dtype,
            grad_accum_steps=grad_accum_steps,
            grad_checkpoint_every_n=grad_checkpoint_every_n,
            chunked_ce_size=chunked_ce_size,
            steps=search_steps,
            warmup=search_warmup,
            mor_attention_pattern=mor_attention_pattern,
            use_ada_rmsnorm=use_ada_rmsnorm,
            tmp_dir=tmp_dir,
            run_id=run_id,
        )
        run_id += 1
        trials.append({"batch_size": mid, "ok": ok, "out": out})
        if ok and ok_under_target(out):
            lo = mid
        else:
            hi = mid

    return lo, trials


def _write_jsonl(path: Path, record: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj


def _record_key(rec: Dict[str, Any]) -> Optional[Tuple[Any, ...]]:
    """A stable key for de-duplication/resume.

    Keyed on the config dimensions that identify a unique benchmark run.
    """

    try:
        bench = rec.get("bench", {})
        if not isinstance(bench, dict):
            return None
        return (
            str(rec.get("preset")),
            str(rec.get("mor_attention_pattern", "baseline")),
            int(bool(rec.get("ada_rmsnorm", False))),
            int(bench.get("seq_len")),
            int(bool(bench.get("compile"))),
            str(bench.get("compile_mode")),
            str(bench.get("dtype")),
            int(bool(rec.get("amp"))),
            int(rec.get("grad_accum_steps")),
            int(rec.get("grad_ckpt_every_n")),
            int(rec.get("chunked_ce_size")),
        )
    except Exception:
        return None


def _load_existing_results(results_jsonl: Path) -> Tuple[List[Dict[str, Any]], Set[Tuple[Any, ...]]]:
    records: List[Dict[str, Any]] = []
    keys: Set[Tuple[Any, ...]] = set()
    for rec in _iter_jsonl(results_jsonl):
        k = _record_key(rec)
        if k is None:
            continue
        if k in keys:
            continue
        keys.add(k)
        records.append(rec)
    return records, keys


def _next_run_id(tmp_dir: Path) -> int:
    max_id = -1
    if tmp_dir.exists():
        for p in tmp_dir.glob("run_*.json"):
            stem = p.stem
            if not stem.startswith("run_"):
                continue
            try:
                rid = int(stem.split("_", 1)[1])
            except Exception:
                continue
            max_id = max(max_id, rid)
    return max_id + 1


def _summarize_csv(records: List[Dict[str, Any]], path: Path) -> None:
    fields = [
        "preset",
        "mor_attention_pattern",
        "ada_rmsnorm",
        "cfg_name",
        "seq_len",
        "compile",
        "compile_mode",
        "dtype",
        "amp",
        "grad_accum_steps",
        "grad_ckpt_every_n",
        "chunked_ce_size",
        "batch_size",
        "tok_per_sec",
        "peak_mem_gb",
        "seconds",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            out = r.get("bench", {})
            peak = out.get("peak_mem_bytes")
            w.writerow(
                {
                    "preset": r.get("preset"),
                    "mor_attention_pattern": r.get("mor_attention_pattern", "baseline"),
                    "ada_rmsnorm": int(bool(r.get("ada_rmsnorm", False))),
                    "cfg_name": out.get("name"),
                    "seq_len": out.get("seq_len"),
                    "compile": int(bool(out.get("compile"))),
                    "compile_mode": out.get("compile_mode"),
                    "dtype": out.get("dtype"),
                    "amp": int(bool(r.get("amp"))),
                    "grad_accum_steps": r.get("grad_accum_steps"),
                    "grad_ckpt_every_n": r.get("grad_ckpt_every_n"),
                    "chunked_ce_size": r.get("chunked_ce_size"),
                    "batch_size": out.get("batch_size"),
                    "tok_per_sec": out.get("tok_per_sec"),
                    "peak_mem_gb": float(peak) / (1024**3) if isinstance(peak, int) else None,
                    "seconds": out.get("seconds"),
                }
            )


def _make_plots(records: List[Dict[str, Any]], out_dir: Path) -> List[str]:
    """Returns list of filenames written."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []

    written: List[str] = []

    # Plot: tok/s vs seq_len for each preset and compile flag (taking best tok/s per seq)
    by_key: Dict[Tuple[str, int, int], float] = {}
    for r in records:
        out = r.get("bench", {})
        if not isinstance(out, dict):
            continue
        if out.get("tok_per_sec") is None or out.get("seq_len") is None:
            continue
        preset = str(r.get("preset"))
        seq = int(out.get("seq_len"))
        comp = int(bool(out.get("compile")))
        tps = float(out.get("tok_per_sec"))
        k = (preset, comp, seq)
        by_key[k] = max(by_key.get(k, 0.0), tps)

    for preset in sorted({str(r.get("preset")) for r in records}):
        for comp in [0, 1]:
            xs = sorted(
                {
                    int(rr.get("bench", {}).get("seq_len"))
                    for rr in records
                    if rr.get("bench")
                    and str(rr.get("preset")) == preset
                    and int(bool(rr.get("bench", {}).get("compile"))) == comp
                    and rr.get("bench", {}).get("seq_len") is not None
                }
            )
            if not xs:
                continue
            ys = [by_key.get((preset, comp, x), 0.0) / 1e3 for x in xs]
            plt.figure(figsize=(7, 4))
            plt.plot(xs, ys, marker="o")
            plt.title(f"{preset}: best tok/s vs seq_len (compile={comp})")
            plt.xlabel("Sequence length")
            plt.ylabel("Tokens/sec (K)")
            plt.grid(True, alpha=0.3)
            fn = f"{preset}_compile{comp}_tps_vs_seq.png"
            plt.tight_layout()
            plt.savefig(out_dir / fn, dpi=140)
            plt.close()
            written.append(fn)

    return written


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--presets", default="100m")
    p.add_argument("--seq_lens", default="2048")
    p.add_argument("--compile", default="1")
    p.add_argument("--compile_mode", default="max-autotune")
    p.add_argument("--amp", type=int, default=1)
    p.add_argument("--dtypes", default="bfloat16")
    p.add_argument("--grad_accum", default="1")
    p.add_argument("--grad_ckpt_every_n", default="2")
    p.add_argument("--chunked_ce", default="4096")
    p.add_argument(
        "--mor_patterns",
        default="lla2x3+ccqa",
        help=(
            "MoR attention pattern (only 'lla2x3+ccqa' supported). "
            "Pattern format: literal token list containing commas (e.g. 'lla2,lla2,lla2,ccqa')."
        ),
    )
    p.add_argument("--target_vram_frac", type=float, default=0.92)
    p.add_argument("--max_batch", type=int, default=1024)
    p.add_argument("--search_steps", type=int, default=6)
    p.add_argument("--search_warmup", type=int, default=2)
    p.add_argument("--final_steps", type=int, default=20)
    p.add_argument("--final_warmup", type=int, default=5)
    p.add_argument(
        "--ada_rmsnorm",
        type=int,
        default=1,
        help="If 1, run one additional compiled benchmark with HYDRA_USE_ADA_RMSNORM=1 for the chosen pattern.",
    )
    p.add_argument(
        "--ada_rmsnorm_pattern",
        default="lla2x3+ccqa",
        help="Which MoR pattern to run with AdaRMSNorm enabled (single extra run).",
    )
    p.add_argument("--out_dir", default="")
    p.add_argument("--resume", type=int, default=0)
    args = p.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available")

    presets = [s.strip().lower() for s in args.presets.split(",") if s.strip()]
    seq_lens = [int(s.strip()) for s in args.seq_lens.split(",") if s.strip()]
    compile_flags = [int(s.strip()) for s in args.compile.split(",") if s.strip()]
    dtypes = [s.strip() for s in args.dtypes.split(",") if s.strip()]
    grad_accums = [int(s.strip()) for s in args.grad_accum.split(",") if s.strip()]
    grad_ckpts = [int(s.strip()) for s in args.grad_ckpt_every_n.split(",") if s.strip()]
    chunked_ce_sizes = [int(s.strip()) for s in args.chunked_ce.split(",") if s.strip()]
    mor_patterns = [s.strip() for s in args.mor_patterns.split(";") if s.strip()]

    out_dir = Path(args.out_dir) if args.out_dir else (REPO_ROOT / "reports" / f"throughput_sweep_{_now_tag()}")
    out_dir.mkdir(parents=True, exist_ok=True)

    tmp_dir = out_dir / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    print(f"Writing sweep artifacts under: {out_dir}", flush=True)

    free_bytes, total_bytes = _cuda_mem_info() if args.device == "cuda" else (None, None)
    if args.device == "cuda" and isinstance(free_bytes, int) and free_bytes < 2 * (1024**3):
        raise SystemExit(
            f"CUDA device has only {free_bytes/(1024**3):.2f} GB free. "
            "A training job or another process is likely occupying VRAM; stop it (or use a different GPU) before sweeping."
        )

    if args.device == "cuda" and isinstance(free_bytes, int) and isinstance(total_bytes, int):
        print(
            f"CUDA mem: free={free_bytes/(1024**3):.2f} GB total={total_bytes/(1024**3):.2f} GB",
            flush=True,
        )

    # Use total * frac, but never exceed the currently free VRAM (with a small safety margin).
    if isinstance(total_bytes, int) and isinstance(free_bytes, int):
        target_bytes = min(int(total_bytes * float(args.target_vram_frac)), int(free_bytes * 0.98))
    elif isinstance(total_bytes, int):
        target_bytes = int(total_bytes * float(args.target_vram_frac))
    else:
        target_bytes = None

    meta_path = out_dir / "meta.json"
    if bool(args.resume) and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
    else:
        meta = {
            "timestamp": datetime.now().isoformat(),
            "repo_root": str(REPO_ROOT),
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cuda_total_mem_gb": float(total_bytes) / (1024**3) if isinstance(total_bytes, int) else None,
            "cuda_free_mem_gb": float(free_bytes) / (1024**3) if isinstance(free_bytes, int) else None,
            "target_vram_frac": float(args.target_vram_frac),
            "target_vram_gb": float(target_bytes) / (1024**3) if isinstance(target_bytes, int) else None,
            "grid": {
                "presets": presets,
                "seq_lens": seq_lens,
                "compile": compile_flags,
                "dtypes": dtypes,
                "grad_accum": grad_accums,
                "grad_ckpt_every_n": grad_ckpts,
                "chunked_ce": chunked_ce_sizes,
                "mor_patterns": mor_patterns,
            },
            "search": {"steps": args.search_steps, "warmup": args.search_warmup},
            "final": {"steps": args.final_steps, "warmup": args.final_warmup},
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    results_jsonl = out_dir / "results.jsonl"

    records: List[Dict[str, Any]] = []
    completed_keys: Set[Tuple[Any, ...]] = set()
    if bool(args.resume) and results_jsonl.exists():
        records, completed_keys = _load_existing_results(results_jsonl)
        print(f"Resume enabled: loaded {len(records)} existing records", flush=True)

    run_id = _next_run_id(tmp_dir)

    def _finalize_reports() -> None:
        all_records, _ = _load_existing_results(results_jsonl)
        _summarize_csv(all_records, out_dir / "summary.csv")
        plots = _make_plots(all_records, out_dir)

        best_by_preset: Dict[str, Dict[str, Any]] = {}
        best_by_preset_pattern_ada: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
        for r in all_records:
            out = r.get("bench", {})
            if "tok_per_sec" not in out:
                continue
            preset = str(r.get("preset"))
            pat = str(r.get("mor_attention_pattern", "baseline"))
            ada = int(bool(r.get("ada_rmsnorm", False)))
            best = best_by_preset.get(preset)
            if best is None or float(out["tok_per_sec"]) > float(best["bench"]["tok_per_sec"]):
                best_by_preset[preset] = r

            k = (preset, pat, ada)
            best_pp = best_by_preset_pattern_ada.get(k)
            if best_pp is None or float(out["tok_per_sec"]) > float(best_pp["bench"]["tok_per_sec"]):
                best_by_preset_pattern_ada[k] = r

        md_lines: List[str] = []
        ts = meta.get("timestamp", datetime.now().isoformat())
        md_lines.append(f"# HYDRA Throughput Sweep ({ts})\n")
        md_lines.append(f"- Device: {meta.get('cuda_device')}\n")
        md_lines.append(f"- Torch: {meta.get('torch')}\n")
        if isinstance(meta.get("target_vram_gb"), (int, float)) and isinstance(meta.get("target_vram_frac"), (int, float)):
            md_lines.append(
                f"- Target VRAM: {float(meta['target_vram_gb']):.2f} GB (frac={float(meta['target_vram_frac'])})\n"
            )
        md_lines.append("\n## Best configs (by preset)\n")
        for preset, r in best_by_preset.items():
            out = r["bench"]
            md_lines.append(
                f"- **{preset}**: tok/s={out['tok_per_sec']/1e3:.1f}K  seq={out['seq_len']}  bs={out['batch_size']}  compile={int(bool(out['compile']))}  dtype={out['dtype']}  ga={r['grad_accum_steps']}  ckpt={r['grad_ckpt_every_n']}  ce={r['chunked_ce_size']}  peakGB={float(out['peak_mem_bytes'])/(1024**3):.2f}\n"
            )

        md_lines.append("\n## Best configs (by preset + MoR pattern + AdaRMSNorm)\n")
        for (preset, pat, ada) in sorted(best_by_preset_pattern_ada.keys()):
            r = best_by_preset_pattern_ada[(preset, pat, ada)]
            out = r["bench"]
            md_lines.append(
                f"- **{preset} / {pat} / ada={ada}**: tok/s={out['tok_per_sec']/1e3:.1f}K  seq={out['seq_len']}  bs={out['batch_size']}  compile={int(bool(out['compile']))}  mode={out.get('compile_mode')}  dtype={out['dtype']}  ga={r['grad_accum_steps']}  ckpt={r['grad_ckpt_every_n']}  ce={r['chunked_ce_size']}  peakGB={float(out['peak_mem_bytes'])/(1024**3):.2f}\n"
            )

        md_lines.append("\n## Artifacts\n")
        md_lines.append("- results.jsonl (raw per-run records)\n")
        md_lines.append("- summary.csv (flat table)\n")
        if plots:
            md_lines.append("\n## Plots\n")
            for fn in plots:
                md_lines.append(f"- {fn}\n")

        (out_dir / "report.md").write_text("".join(md_lines), encoding="utf-8")

    try:
        for preset in presets:
            for mor_pattern in mor_patterns:
                for seq_len in seq_lens:
                    for dtype in dtypes:
                        for grad_accum_steps in grad_accums:
                            for grad_ckpt in grad_ckpts:
                                for chunked_ce in chunked_ce_sizes:
                                    # If all compile variants already exist, skip the whole config (including batch search).
                                    need_any = False
                                    for compile_flag in compile_flags:
                                        probe = {
                                            "preset": preset,
                                            "mor_attention_pattern": mor_pattern,
                                            "ada_rmsnorm": False,
                                            "amp": bool(args.amp),
                                            "grad_accum_steps": grad_accum_steps,
                                            "grad_ckpt_every_n": grad_ckpt,
                                            "chunked_ce_size": chunked_ce,
                                            "bench": {
                                                "seq_len": int(seq_len),
                                                "compile": bool(compile_flag),
                                                "compile_mode": args.compile_mode,
                                                "dtype": dtype,
                                            },
                                        }
                                        k = _record_key(probe)
                                        if k is None or k not in completed_keys:
                                            need_any = True
                                            break
                                    if not need_any:
                                        continue

                                # Find max microbatch (eager search). Search uses compile=0.
                                bs_guess, trials = _find_max_microbatch(
                                    preset,
                                    device=args.device,
                                    seq_len=seq_len,
                                    target_bytes=target_bytes,
                                    max_batch=args.max_batch,
                                    search_steps=args.search_steps,
                                    search_warmup=args.search_warmup,
                                    compile_mode=args.compile_mode,
                                    use_amp=bool(args.amp),
                                    dtype=dtype,
                                    grad_accum_steps=grad_accum_steps,
                                    grad_checkpoint_every_n=grad_ckpt,
                                    chunked_ce_size=chunked_ce,
                                    mor_attention_pattern=mor_pattern,
                                    use_ada_rmsnorm=False,
                                    tmp_dir=tmp_dir,
                                    run_id_start=run_id,
                                )
                                run_id += len(trials)

                                for compile_flag in compile_flags:
                                    probe = {
                                        "preset": preset,
                                        "mor_attention_pattern": mor_pattern,
                                        "ada_rmsnorm": False,
                                        "amp": bool(args.amp),
                                        "grad_accum_steps": grad_accum_steps,
                                        "grad_ckpt_every_n": grad_ckpt,
                                        "chunked_ce_size": chunked_ce,
                                        "bench": {
                                            "seq_len": int(seq_len),
                                            "compile": bool(compile_flag),
                                            "compile_mode": args.compile_mode,
                                            "dtype": dtype,
                                        },
                                    }
                                    k = _record_key(probe)
                                    if k is not None and k in completed_keys:
                                        continue

                                    # Validate/measure with requested compile flag
                                    bs_final = bs_guess
                                    bench_out: Dict[str, Any] = {}
                                    if bs_final <= 0:
                                        bench_out = {"error": "no_batch_fit"}
                                    else:
                                        # Backoff search to avoid repeated OOMs (which can poison allocator state).
                                        bs_try = int(bs_final)
                                        while bs_try > 0:
                                            ok, out = _try_bench_once(
                                                preset,
                                                device=args.device,
                                                batch_size=bs_try,
                                                seq_len=seq_len,
                                                use_compile=bool(compile_flag),
                                                compile_mode=args.compile_mode,
                                                use_amp=bool(args.amp),
                                                dtype=dtype,
                                                grad_accum_steps=grad_accum_steps,
                                                grad_checkpoint_every_n=grad_ckpt,
                                                chunked_ce_size=chunked_ce,
                                                steps=args.final_steps,
                                                warmup=args.final_warmup,
                                                mor_attention_pattern=mor_pattern,
                                                use_ada_rmsnorm=False,
                                                tmp_dir=tmp_dir,
                                                run_id=run_id,
                                            )
                                            run_id += 1
                                            if ok:
                                                bench_out = out
                                                bs_final = bs_try
                                                break
                                            bench_out = out
                                            bs_try //= 2

                                        if not bench_out:
                                            bench_out = {"error": "oom_even_at_bs1"}

                                    # Ensure basic fields exist even on failures (helps reporting/plotting).
                                    if isinstance(bench_out, dict):
                                        bench_out.setdefault("seq_len", int(seq_len))
                                        bench_out.setdefault("compile", bool(compile_flag))
                                        bench_out.setdefault("compile_mode", args.compile_mode)
                                        bench_out.setdefault("dtype", dtype)
                                        bench_out.setdefault("batch_size", int(max(1, bs_final)))

                                    rec = {
                                        "preset": preset,
                                        "mor_attention_pattern": mor_pattern,
                                        "ada_rmsnorm": False,
                                        "amp": bool(args.amp),
                                        "dtype": dtype,
                                        "grad_accum_steps": grad_accum_steps,
                                        "grad_ckpt_every_n": grad_ckpt,
                                        "chunked_ce_size": chunked_ce,
                                        "batch_search": {
                                            "target_mem_bytes": target_bytes,
                                            "target_mem_gb": float(target_bytes) / (1024**3) if isinstance(target_bytes, int) else None,
                                            "bs_guess_eager": bs_guess,
                                            "trials": trials,
                                        },
                                        "bench": bench_out,
                                    }
                                    records.append(rec)
                                    _write_jsonl(results_jsonl, rec)
                                    k2 = _record_key(rec)
                                    if k2 is not None:
                                        completed_keys.add(k2)

                                    # Console progress
                                    if "tok_per_sec" in bench_out:
                                        print(
                                            f"[{preset}] pat={mor_pattern} ada=0 seq={seq_len} compile={compile_flag} mode={args.compile_mode} dtype={dtype} ga={grad_accum_steps} ckpt={grad_ckpt} ce={chunked_ce}  bs={bench_out.get('batch_size')}  tok/s={bench_out.get('tok_per_sec')/1e3:.1f}K  peakGB={_format_gb(bench_out.get('peak_mem_bytes'))}"
                                        )
                                    else:
                                        print(
                                            f"[{preset}] pat={mor_pattern} ada=0 seq={seq_len} compile={compile_flag} mode={args.compile_mode} dtype={dtype} ga={grad_accum_steps} ckpt={grad_ckpt} ce={chunked_ce}  ERROR={bench_out.get('error')}"
                                        )

        # One extra AdaRMSNorm run (compiled) for a single chosen pattern, at the same seq.
        if bool(args.ada_rmsnorm) and presets and seq_lens:
            preset = presets[0]
            seq_len = int(seq_lens[0])
            ada_pat = str(args.ada_rmsnorm_pattern).strip() or "lla2x3+ccqa"
            dtype = dtypes[0] if dtypes else "bfloat16"
            grad_accum_steps = grad_accums[0] if grad_accums else 1
            grad_ckpt = grad_ckpts[0] if grad_ckpts else 2
            chunked_ce = chunked_ce_sizes[0] if chunked_ce_sizes else 4096

            probe = {
                "preset": preset,
                "mor_attention_pattern": ada_pat,
                "ada_rmsnorm": True,
                "amp": bool(args.amp),
                "grad_accum_steps": grad_accum_steps,
                "grad_ckpt_every_n": grad_ckpt,
                "chunked_ce_size": chunked_ce,
                "bench": {
                    "seq_len": int(seq_len),
                    "compile": True,
                    "compile_mode": args.compile_mode,
                    "dtype": dtype,
                },
            }
            k = _record_key(probe)
            if k is None or k not in completed_keys:
                bs_guess, trials = _find_max_microbatch(
                    preset,
                    device=args.device,
                    seq_len=seq_len,
                    target_bytes=target_bytes,
                    max_batch=args.max_batch,
                    search_steps=args.search_steps,
                    search_warmup=args.search_warmup,
                    compile_mode=args.compile_mode,
                    use_amp=bool(args.amp),
                    dtype=dtype,
                    grad_accum_steps=grad_accum_steps,
                    grad_checkpoint_every_n=grad_ckpt,
                    chunked_ce_size=chunked_ce,
                    mor_attention_pattern=ada_pat,
                    use_ada_rmsnorm=True,
                    tmp_dir=tmp_dir,
                    run_id_start=run_id,
                )
                run_id += len(trials)

                bs_try = int(max(1, bs_guess))
                bench_out: Dict[str, Any] = {}
                while bs_try > 0:
                    ok, out = _try_bench_once(
                        preset,
                        device=args.device,
                        batch_size=bs_try,
                        seq_len=seq_len,
                        use_compile=True,
                        compile_mode=args.compile_mode,
                        use_amp=bool(args.amp),
                        dtype=dtype,
                        grad_accum_steps=grad_accum_steps,
                        grad_checkpoint_every_n=grad_ckpt,
                        chunked_ce_size=chunked_ce,
                        steps=args.final_steps,
                        warmup=args.final_warmup,
                        mor_attention_pattern=ada_pat,
                        use_ada_rmsnorm=True,
                        tmp_dir=tmp_dir,
                        run_id=run_id,
                    )
                    run_id += 1
                    if ok:
                        bench_out = out
                        break
                    bench_out = out
                    bs_try //= 2

                if isinstance(bench_out, dict):
                    bench_out.setdefault("seq_len", int(seq_len))
                    bench_out.setdefault("compile", True)
                    bench_out.setdefault("compile_mode", args.compile_mode)
                    bench_out.setdefault("dtype", dtype)
                    bench_out.setdefault("batch_size", int(max(1, bs_try)))

                rec = {
                    "preset": preset,
                    "mor_attention_pattern": ada_pat,
                    "ada_rmsnorm": True,
                    "amp": bool(args.amp),
                    "dtype": dtype,
                    "grad_accum_steps": grad_accum_steps,
                    "grad_ckpt_every_n": grad_ckpt,
                    "chunked_ce_size": chunked_ce,
                    "batch_search": {
                        "target_mem_bytes": target_bytes,
                        "target_mem_gb": float(target_bytes) / (1024**3) if isinstance(target_bytes, int) else None,
                        "bs_guess_eager": bs_guess,
                        "trials": trials,
                    },
                    "bench": bench_out,
                }
                records.append(rec)
                _write_jsonl(results_jsonl, rec)
                k2 = _record_key(rec)
                if k2 is not None:
                    completed_keys.add(k2)

                if "tok_per_sec" in bench_out:
                    print(
                        f"[{preset}] pat={ada_pat} ada=1 seq={seq_len} compile=1 mode={args.compile_mode} dtype={dtype} ga={grad_accum_steps} ckpt={grad_ckpt} ce={chunked_ce}  bs={bench_out.get('batch_size')}  tok/s={bench_out.get('tok_per_sec')/1e3:.1f}K  peakGB={_format_gb(bench_out.get('peak_mem_bytes'))}"
                    )
                else:
                    print(
                        f"[{preset}] pat={ada_pat} ada=1 seq={seq_len} compile=1 mode={args.compile_mode} dtype={dtype} ga={grad_accum_steps} ckpt={grad_ckpt} ce={chunked_ce}  ERROR={bench_out.get('error')}"
                    )
    except KeyboardInterrupt:
        print("\nInterrupted; finalizing report from results.jsonl ...", flush=True)
    finally:
        _finalize_reports()

    print(f"\nWrote sweep report to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
