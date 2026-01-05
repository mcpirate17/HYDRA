#!/usr/bin/env python3
"""Tall/skinny + small-model diagnostics for HYDRA (MoD + MoR + frontier opts).

Goals:
- Keep MoD+MoR enabled.
- Exercise the training-critical path (AMP, torch.compile, chunked CE, grad ckpt).
- Report tok/s, step time, peak VRAM, and routing/regularizer stats.

This script is intentionally self-contained and uses only repo code.

Run examples:
  /home/tim/venvs/llm/bin/python diagnostics/tall_skinny_bench.py --device cuda --compile 1
  /home/tim/venvs/llm/bin/python diagnostics/tall_skinny_bench.py --device cuda --compile 1 --steps 50
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
import contextlib
import io
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Tuple

import torch
import torch.nn.functional as F

# Ensure repo root is on sys.path so `import hydra...` works when executing
# this script directly from the diagnostics/ folder.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if TYPE_CHECKING:
    from hydra.model.framework import HydraModel


@dataclass
class BenchConfig:
    name: str
    vocab_size: int
    dim: int
    n_mor_blocks: int
    recursions_per_block: int
    n_heads: int
    n_kv_heads: int
    max_seq_len: int
    mod_capacity: float


def _sync_if_cuda(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def _format_gb(bytes_val: int) -> str:
    return f"{bytes_val / (1024**3):.2f} GB"


def _get_peak_mem(device: str) -> Optional[int]:
    if device != "cuda":
        return None
    return int(torch.cuda.max_memory_allocated())


def _reset_peak_mem(device: str) -> None:
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()


def _get_first_layer_stats(model: HydraModel) -> Dict[str, Any]:
    # Use compile-disabled helpers.
    layer0 = model.layers[0]
    out: Dict[str, Any] = {}
    if hasattr(layer0, "get_routing_stats"):
        """
        Tall/skinny + small-model diagnostics for HYDRA (MoD + MoR + frontier opts).

        Goals:
        - Keep MoD+MoR enabled.
        - Exercise the training-critical path (AMP, torch.compile, chunked CE, grad ckpt).
        - Report tok/s, step time, peak VRAM, and routing/regularizer stats.

        This script is intentionally self-contained and uses only repo code.

        Notes:
        - RECOMMENDED: Run on CUDA for meaningful throughput and VRAM measurements.
            It can run on CPU for quick functional checks but will be much slower.

        Run examples:
            /home/tim/venvs/llm/bin/python diagnostics/tall_skinny_bench.py --device cuda --compile 1
            /home/tim/venvs/llm/bin/python diagnostics/tall_skinny_bench.py --device cuda --compile 1 --steps 50
        """
    device: str,
    dtype: torch.dtype,
    batch_size: int,
    n_heads: int,
    seq_len: int,
    head_dim: int,
    is_causal: bool = True,
) -> Optional[Dict[str, Any]]:
    """Best-effort SDPA backend probe.

    PyTorch SDPA dispatch is dynamic; this probe records:
    - Global enable flags (flash/efficient/math/cudnn)
    - can_use_* decisions where PyTorch exposes them
    - Whether forcing a backend succeeds for representative shapes
    - Debug strings explaining why flash/cudnn are unavailable (when available)
    """

    if device != "cuda" or not torch.cuda.is_available():
        return None

    from torch.backends import cuda as bc

    probe_b = min(2, max(1, int(batch_size)))
    q = torch.randn(probe_b, n_heads, seq_len, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(probe_b, n_heads, seq_len, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(probe_b, n_heads, seq_len, head_dim, device="cuda", dtype=dtype)

    # torch.backends.cuda.SDPAParams is a pybind type and its constructor signature
    # varies slightly across torch versions. In torch 2.9+, it takes 2 trailing bools.
    # The final bool is backend-internal; False is a safe default for probing.
    params = bc.SDPAParams(q, k, v, None, 0.0, is_causal, False)

    out: Dict[str, Any] = {
        "shape": {
            "probe_batch": probe_b,
            "n_heads": int(n_heads),
            "seq_len": int(seq_len),
            "head_dim": int(head_dim),
            "dtype": str(dtype).replace("torch.", ""),
            "is_causal": bool(is_causal),
        },
        "enabled": {
            "flash": bool(bc.flash_sdp_enabled()),
            "mem_efficient": bool(bc.mem_efficient_sdp_enabled()),
            "math": bool(bc.math_sdp_enabled()),
            "cudnn": bool(bc.cudnn_sdp_enabled()),
        },
        "can_use": {},
        "debug": {},
        "forced": {},
        "predicted_backend": None,
    }

    def _capture_debug(fn) -> str:
        buf_out = io.StringIO()
        buf_err = io.StringIO()
        with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
            try:
                _ = fn()
            except Exception as e:
                return f"<exception: {type(e).__name__}: {e}>"
        s = (buf_err.getvalue() + buf_out.getvalue()).strip()
        return s

    if hasattr(bc, "can_use_flash_attention"):
        try:
            out["can_use"]["flash"] = bool(bc.can_use_flash_attention(params, debug=False))
            out["debug"]["flash"] = _capture_debug(lambda: bc.can_use_flash_attention(params, debug=True))
        except Exception as e:
            out["can_use"]["flash_error"] = f"{type(e).__name__}: {e}"

    if hasattr(bc, "can_use_cudnn_attention"):
        try:
            out["can_use"]["cudnn"] = bool(bc.can_use_cudnn_attention(params, debug=False))
            out["debug"]["cudnn"] = _capture_debug(lambda: bc.can_use_cudnn_attention(params, debug=True))
        except Exception as e:
            out["can_use"]["cudnn_error"] = f"{type(e).__name__}: {e}"

    def _try_force(name: str, backend) -> None:
        try:
            with torch.nn.attention.sdpa_kernel(backend):
                _ = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=is_causal)
            out["forced"][name] = {"ok": True}
        except Exception as e:
            out["forced"][name] = {"ok": False, "error": f"{type(e).__name__}: {e}"}

    # Try forcing each backend. This is a practical proxy for "could SDPA run via X".
    sdpb = torch.nn.attention.SDPBackend
    _try_force("flash", sdpb.FLASH_ATTENTION)
    _try_force("cudnn", sdpb.CUDNN_ATTENTION)
    _try_force("mem_efficient", sdpb.EFFICIENT_ATTENTION)
    _try_force("math", sdpb.MATH)

    # Heuristic prediction matching typical SDPA priority.
    # (This is still a prediction; the real dispatch is internal and dynamic.)
    for name in ("flash", "cudnn", "mem_efficient", "math"):
        forced = out["forced"].get(name, {})
        if forced.get("ok"):
            out["predicted_backend"] = name
            break

    return out


def run_one(
    cfg: BenchConfig,
    *,
    device: str,
    use_compile: bool,
    compile_mode: str,
    use_amp: bool,
    dtype: str,
    batch_size: int,
    seq_len: int,
    steps: int,
    warmup: int,
    grad_accum_steps: int,
    grad_checkpoint_every_n: int,
    chunked_ce_size: int,
) -> Dict[str, Any]:
    from hydra.kernels import fused_chunked_cross_entropy
    from hydra.kernels import get_kernel_status
    from hydra.model.framework import HydraModel

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    torch.manual_seed(0)

    dev = torch.device(device)

    model = HydraModel(
        vocab_size=cfg.vocab_size,
        dim=cfg.dim,
        n_mor_blocks=cfg.n_mor_blocks,
        recursions_per_block=cfg.recursions_per_block,
        n_heads=cfg.n_heads,
        n_kv_heads=cfg.n_kv_heads,
        max_seq_len=cfg.max_seq_len,
        mod_capacity=cfg.mod_capacity,
        adaptive=True,
        hybrid_attention=True,
        tie_weights=True,
    ).to(dev)
    model.train()
    model.enable_gradient_checkpointing(every_n=max(1, grad_checkpoint_every_n))

    # A lightweight optimizer to include backward + step cost.
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.1)

    # torch.compile (frontier optimization). Keep fullgraph=False for robustness.
    compiled_model = model
    compile_t = None
    compile_fallback_used = False
    compile_mode_effective = compile_mode
    if use_compile:
        # MoD/MoR introduces control-flow + token routing which can make CUDA graph capture unstable.
        # We attempt max-autotune with cudagraphs (best case), but fall back to no-cudagraphs if it fails.

        def _set_cudagraphs(enabled: bool) -> None:
            try:
                import torch._inductor.config as inductor_config  # type: ignore

                if hasattr(inductor_config, "triton") and hasattr(inductor_config.triton, "cudagraphs"):
                    inductor_config.triton.cudagraphs = bool(enabled)
            except Exception:
                return

        def _compile_with(mode: str, *, cudagraphs: bool) -> Tuple[torch.nn.Module, float, str]:
            _set_cudagraphs(cudagraphs)
            _sync_if_cuda(device)
            t0 = time.perf_counter()
            cm = torch.compile(model, mode=mode, fullgraph=False)
            _sync_if_cuda(device)
            return cm, (time.perf_counter() - t0), mode

        want_no_cudagraphs = "no-cudagraphs" in str(compile_mode).lower()
        if want_no_cudagraphs:
            compiled_model, compile_t, compile_mode_effective = _compile_with(
                compile_mode, cudagraphs=False
            )
        else:
            try:
                compiled_model, compile_t, compile_mode_effective = _compile_with(
                    compile_mode, cudagraphs=True
                )
            except Exception as e:
                msg = str(e).lower()
                if "cudagraph" in msg or "cuda graph" in msg or "capture" in msg:
                    compiled_model, compile_t, compile_mode_effective = _compile_with(
                        "max-autotune-no-cudagraphs", cudagraphs=False
                    )
                    compile_fallback_used = True
                else:
                    raise

    amp_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }.get(dtype, torch.bfloat16)

    # Best-effort SDPA backend probe (records fallback reasons when possible).
    head_dim = cfg.dim // max(1, cfg.n_heads)
    sdpa_probe = _sdpa_backend_probe(
        device=device,
        dtype=amp_dtype if (use_amp and device == "cuda") else torch.float32,
        batch_size=batch_size,
        n_heads=cfg.n_heads,
        seq_len=seq_len,
        head_dim=head_dim,
        is_causal=True,
    )

    # Synthetic batch. Keep a single microbatch to avoid skewing results with RNG overhead.
    x = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device=dev, dtype=torch.long)
    y = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device=dev, dtype=torch.long)

    def step_fn() -> Dict[str, float]:
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        ce_val = 0.0
        aux_val = 0.0
        ponder_val = 0.0
        for _ in range(max(1, grad_accum_steps)):
            with torch.autocast(device_type=device, dtype=amp_dtype, enabled=(use_amp and device == "cuda")):
                hidden, aux = compiled_model.forward_hidden_with_losses(x)
                weight = compiled_model.output.weight
                ce = fused_chunked_cross_entropy(
                    hidden,
                    weight,
                    y,
                    ignore_index=-100,
                    chunk_size=chunked_ce_size,
                )
                aux_loss = aux.get("aux_loss", 0.0)
                ponder_loss = aux.get("ponder_loss", 0.0)
                # Keep both losses active; these are exactly what training uses.
                loss = ce + 0.1 * aux_loss + 0.01 * ponder_loss
                scaled = loss / max(1, grad_accum_steps)

            scaled.backward()
            total_loss = total_loss + loss.detach()
            ce_val = float(ce.detach().float().cpu())
            aux_val = float(torch.as_tensor(aux_loss).detach().float().cpu())
            ponder_val = float(torch.as_tensor(ponder_loss).detach().float().cpu())

        optimizer.step()
        return {
            "loss": float(total_loss.float().cpu()),
            "ce": ce_val,
            "aux": aux_val,
            "ponder": ponder_val,
        }

    # Warmup to amortize compile/caches.
    _reset_peak_mem(device)
    for _ in range(max(0, warmup)):
        _ = step_fn()
    _sync_if_cuda(device)

    # Timed region
    losses: List[Dict[str, float]] = []
    _reset_peak_mem(device)
    _sync_if_cuda(device)
    t0 = time.perf_counter()
    for _ in range(steps):
        losses.append(step_fn())
    _sync_if_cuda(device)
    dt = time.perf_counter() - t0

    tokens = int(steps * batch_size * seq_len * max(1, grad_accum_steps))
    tps = tokens / dt if dt > 0 else 0.0

    peak_mem = _get_peak_mem(device)

    # Pull a small amount of routing stats (compile-disabled calls).
    stats = _get_first_layer_stats(model)

    # Summarize last step losses
    last = losses[-1] if losses else {}
    avg_loss = sum(d["loss"] for d in losses) / len(losses) if losses else 0.0

    return {
        "name": cfg.name,
        "cfg": cfg.__dict__,
        "device": device,
        "compile": use_compile,
        "compile_mode": compile_mode_effective,
        "compile_seconds": compile_t,
        "compile_fallback_used": bool(compile_fallback_used),
        "batch_size": batch_size,
        "seq_len": seq_len,
        "steps": steps,
        "seconds": dt,
        "tok_per_sec": tps,
        "peak_mem_bytes": peak_mem,
        "avg_loss": avg_loss,
        "last_losses": last,
        "layer0_stats": stats,
        "kernel_status": get_kernel_status(),
        "sdpa_probe": sdpa_probe,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--compile", type=int, default=1)
    p.add_argument("--compile_mode", default="max-autotune")
    p.add_argument("--amp", type=int, default=1)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument(
        "--seq_lens",
        type=str,
        default="",
        help="Optional comma-separated seq lens to sweep (e.g. 512,1024,2048). If set, --seq_len is ignored.",
    )
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--grad_ckpt_every_n", type=int, default=2)
    p.add_argument("--chunked_ce_size", type=int, default=4096)
    p.add_argument(
        "--preset",
        default="small",
        choices=["small", "100m", "500m"],
        help="Model preset list to benchmark. 'small' = quick tall/skinny micro-models.",
    )
    p.add_argument(
        "--also_compile0",
        type=int,
        default=0,
        help="If 1, run each config again with torch.compile disabled for A/B.",
    )
    args = p.parse_args()

    # Default attention pattern (per MoR block): 3:1 macro-block [LLA2, LLA2, LLA2, CCQA].
    # Users can override via HYDRA_MOR_ATTENTION_PATTERN / HYDRA_MOR_ATTENTION_PATTERN_NAME.
    # Keep this CUDA-only: LLA2 requires the external lightning-attention CUDA kernels.
    if args.device == "cuda":
        os.environ.setdefault("HYDRA_MOR_ATTENTION_PATTERN_NAME", "lla2x3+ccqa")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available")

    # Sequence sweep
    if args.seq_lens.strip():
        seq_lens = [int(s.strip()) for s in args.seq_lens.split(",") if s.strip()]
        if not seq_lens:
            raise SystemExit("--seq_lens was provided but parsed empty")
    else:
        seq_lens = [int(args.seq_len)]

    # Presets
    # Keep MoD+MoR on via mod_capacity < 1 and adaptive=True.
    if args.preset == "small":
        configs = [
            BenchConfig(
                name="tall_skinny_256d_16b_r2",
                vocab_size=50257,
                dim=256,
                n_mor_blocks=16,
                recursions_per_block=2,
                n_heads=4,
                n_kv_heads=2,
                max_seq_len=2048,
                mod_capacity=0.5,
            ),
            BenchConfig(
                name="tall_skinny_384d_12b_r2",
                vocab_size=50257,
                dim=384,
                n_mor_blocks=12,
                recursions_per_block=2,
                n_heads=6,
                n_kv_heads=2,
                max_seq_len=2048,
                mod_capacity=0.5,
            ),
            BenchConfig(
                name="shorter_wide_512d_8b_r3",
                vocab_size=50257,
                dim=512,
                n_mor_blocks=8,
                recursions_per_block=3,
                n_heads=8,
                n_kv_heads=2,
                max_seq_len=2048,
                mod_capacity=0.5,
            ),
        ]
    elif args.preset == "100m":
        # Mirrors the training defaults for mod_mor 100M preset.
        configs = [
            BenchConfig(
                name="hydra_100m_like_1280d_8b_r3",
                vocab_size=50257,
                dim=1280,
                n_mor_blocks=8,
                recursions_per_block=3,
                n_heads=20,
                n_kv_heads=5,
                max_seq_len=2048,
                mod_capacity=0.5,
            )
        ]
    else:  # 500m
        # Based on code comments in train config.
        configs = [
            BenchConfig(
                name="hydra_500m_like_2048d_10b_r3",
                vocab_size=50257,
                dim=2048,
                n_mor_blocks=10,
                recursions_per_block=3,
                n_heads=32,
                n_kv_heads=4,
                max_seq_len=2048,
                mod_capacity=0.5,
            )
        ]

    results: List[Dict[str, Any]] = []
    for seq in seq_lens:
        for cfg in configs:
            cfg_run = BenchConfig(**{**cfg.__dict__, "max_seq_len": max(cfg.max_seq_len, seq)})
            for compile_flag in ([bool(args.compile), False] if args.also_compile0 else [bool(args.compile)]):
                print("=" * 80)
                print(
                    f"Running {cfg_run.name}  dim={cfg_run.dim} blocks={cfg_run.n_mor_blocks} rec={cfg_run.recursions_per_block}  seq={seq} bs={args.batch_size}  compile={int(compile_flag)}"
                )
                out = run_one(
                    cfg_run,
                    device=args.device,
                    use_compile=compile_flag,
                    compile_mode=args.compile_mode,
                    use_amp=bool(args.amp),
                    dtype=args.dtype,
                    batch_size=args.batch_size,
                    seq_len=seq,
                    steps=args.steps,
                    warmup=args.warmup,
                    grad_accum_steps=args.grad_accum_steps,
                    grad_checkpoint_every_n=args.grad_ckpt_every_n,
                    chunked_ce_size=args.chunked_ce_size,
                )
                results.append(out)

                peak = out.get("peak_mem_bytes")
                peak_s = _format_gb(peak) if isinstance(peak, int) else "n/a"
                print(f"tok/s: {out['tok_per_sec'] / 1e3:.1f}K   seconds: {out['seconds']:.3f}   peak_mem: {peak_s}")
                if out.get("compile_seconds") is not None:
                    print(f"compile_seconds: {out['compile_seconds']:.2f}")
                print(f"avg_loss: {out['avg_loss']:.4f}  last: {out['last_losses']}")
                print(f"layer0_stats: {out['layer0_stats']}")

    # Optional: write JSON if requested
    out_path = os.environ.get("HYDRA_BENCH_OUT")
    if out_path:
        import json

        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
