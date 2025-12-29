"""Opt-in per-step deep diagnostics.

This module handles the HYDRA_DIAG_STEPS environment variable functionality,
which enables detailed diagnostic collection at specific training steps.

This is separate from spike_diagnostics.py which handles gradient spike
detection and response. This module is for explicit opt-in step-by-step
analysis regardless of whether a spike occurred.

Usage:
  HYDRA_DIAG_STEPS=100,200,300-350 python trainer.py  # specific steps/ranges
  HYDRA_DIAG_FIRST_N=10 python trainer.py              # first N steps from resume
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import torch
import torch.nn.functional as F


def parse_diag_steps(spec: str) -> Set[int]:
    """Parse HYDRA_DIAG_STEPS specification into a set of step numbers.

    Supports:
      - Single steps: "100,200,300"
      - Ranges with dash: "100-200"
      - Ranges with colon: "100:200"
      - Mixed: "50,100-150,200,300:350"
    """
    out: Set[int] = set()
    s = (spec or "").strip()
    if not s:
        return out

    for part in s.split(","):
        p = part.strip()
        if not p:
            continue

        if "-" in p:
            a, b = p.split("-", 1)
        elif ":" in p:
            a, b = p.split(":", 1)
        else:
            a, b = p, ""

        try:
            if b == "":
                out.add(int(a))
            else:
                lo = int(a)
                hi = int(b)
                if hi < lo:
                    lo, hi = hi, lo
                out.update(range(lo, hi + 1))
        except Exception:
            continue

    return out


def get_diag_steps_from_env(start_step: int = 0) -> Set[int]:
    """Get diagnostic steps from environment variables.

    Combines HYDRA_DIAG_STEPS and HYDRA_DIAG_FIRST_N.
    """
    diag_steps = parse_diag_steps(os.environ.get("HYDRA_DIAG_STEPS", ""))
    diag_first_n = int(os.environ.get("HYDRA_DIAG_FIRST_N", "0") or 0)

    if diag_first_n > 0:
        diag_steps.update(range(start_step, start_step + diag_first_n))

    return diag_steps


@dataclass
class StepDiagnostics:
    """Container for step diagnostic data collected across phases."""

    step: int
    active: bool = False

    # Phase 1: Forward-pass and batch stats
    batch: Dict[str, Any] = field(default_factory=dict)
    logits: Dict[str, Any] = field(default_factory=dict)
    ce: Dict[str, Any] = field(default_factory=dict)
    weights: Dict[str, Any] = field(default_factory=dict)

    # References for gradient analysis (cleared after use)
    _emb_w_ref: Optional[torch.Tensor] = field(default=None, repr=False)
    _uniq_in: Optional[torch.Tensor] = field(default=None, repr=False)

    # Phase 2: Pre-clip gradient stats
    grad_preclip: Dict[str, Any] = field(default_factory=dict)
    global_preclip: float = 0.0

    # Phase 3: Post-clip gradient stats
    grad_postclip: Dict[str, Any] = field(default_factory=dict)
    global_postclip: float = 0.0

    def clear(self) -> None:
        """Clear all data and references."""
        self.batch.clear()
        self.logits.clear()
        self.ce.clear()
        self.weights.clear()
        self.grad_preclip.clear()
        self.grad_postclip.clear()
        self._emb_w_ref = None
        self._uniq_in = None
        self.global_preclip = 0.0
        self.global_postclip = 0.0


def collect_phase1_batch_stats(
    diag: StepDiagnostics,
    x: torch.Tensor,
    y: torch.Tensor,
    base_model: Any,
    logger: Any,
) -> None:
    """Phase 1: Collect forward-pass and batch stats BEFORE AMP unscale.

    Collects:
    - Batch token statistics (ranges, special tokens, distribution)
    - Logits statistics (shape, finiteness, distribution)
    - Per-token CE loss distribution
    - Embedding/output weight statistics
    """
    if not diag.active:
        return

    try:
        with torch.no_grad():
            # --- Batch token stats ---
            x_flat = x.view(-1)
            y_flat = y.view(-1)
            y_valid_mask = y_flat != -100
            y_valid = y_flat[y_valid_mask]

            x_min, x_max = int(x_flat.min().item()), int(x_flat.max().item())
            y_min = int(y_valid.min().item()) if y_valid.numel() > 0 else 0
            y_max = int(y_valid.max().item()) if y_valid.numel() > 0 else 0
            y_valid_count = int(y_valid.numel())

            # Special token counts (GPT-2: 50256=EOS, 50257=PAD usually)
            eos_id = 50256
            eos_count = int((x_flat == eos_id).sum().item())
            pad_count = int((x_flat == 50257).sum().item()) if x_max >= 50257 else 0
            unk_count = int((x_flat == 0).sum().item())

            # Top-10 token ids in batch + singleton count
            unique_toks, counts = x_flat.unique(return_counts=True)
            top10_idx = counts.argsort(descending=True)[:10]
            top10_toks = [(int(unique_toks[i].item()), int(counts[i].item())) for i in top10_idx]
            singleton_count = int((counts == 1).sum().item())

            diag.batch = {
                "x_min": x_min,
                "x_max": x_max,
                "y_min": y_min,
                "y_max": y_max,
                "y_valid": y_valid_count,
                "eos_count": eos_count,
                "pad_count": pad_count,
                "unk_count": unk_count,
                "top10_toks": top10_toks,
                "singleton_count": singleton_count,
                "unique_count": int(unique_toks.numel()),
            }

            # Store unique input tokens for row-slice grad analysis later
            diag._uniq_in = unique_toks.clone()

            # --- Compute logits explicitly (chunked CE doesn't expose them) ---
            was_training = bool(getattr(base_model, "training", True))
            base_model.eval()
            try:
                _diag_logits = base_model(x)  # [B, S, V]
            finally:
                base_model.train(was_training)

            B, S, V = _diag_logits.shape
            logits_shape = (B, S, V)
            logits_dtype = str(_diag_logits.dtype)

            # Finite check
            finite_frac = float(torch.isfinite(_diag_logits).float().mean().item())
            num_nan = int(torch.isnan(_diag_logits).sum().item())
            num_inf = int(torch.isinf(_diag_logits).sum().item())

            # Safe stats on float32 slice
            safe_slice = _diag_logits[:, : min(S, 256), : min(V, 8192)].float()
            safe_finite = safe_slice[torch.isfinite(safe_slice)]
            if safe_finite.numel() > 0:
                logits_std = float(safe_finite.std().item())
                logits_max = float(safe_finite.abs().max().item())
                logits_mean = float(safe_finite.mean().item())
            else:
                logits_std = logits_max = logits_mean = float("nan")

            diag.logits = {
                "shape": logits_shape,
                "dtype": logits_dtype,
                "finite_frac": finite_frac,
                "num_nan": num_nan,
                "num_inf": num_inf,
                "std": logits_std,
                "max": logits_max,
                "mean": logits_mean,
            }

            # --- Per-token CE losses ---
            ce_per_token = F.cross_entropy(
                _diag_logits.view(-1, V),
                y_flat,
                ignore_index=-100,
                reduction="none",
            )  # [B*S]
            ce_valid = ce_per_token[y_valid_mask]

            if ce_valid.numel() > 0:
                ce_mean = float(ce_valid.mean().item())
                ce_max = float(ce_valid.max().item())
                ce_p95 = float(torch.quantile(ce_valid.float(), 0.95).item())
                ce_p99 = float(torch.quantile(ce_valid.float(), 0.99).item())
                ce_gt20 = int((ce_valid > 20).sum().item())
                ce_gt30 = int((ce_valid > 30).sum().item())
                ce_gt40 = int((ce_valid > 40).sum().item())

                # argmax CE: find (batch, position)
                max_ce_flat_idx = int(ce_per_token.argmax().item())
                max_ce_b = max_ce_flat_idx // S
                max_ce_t = max_ce_flat_idx % S
                max_ce_label_id = int(y_flat[max_ce_flat_idx].item())
            else:
                ce_mean = ce_max = ce_p95 = ce_p99 = 0.0
                ce_gt20 = ce_gt30 = ce_gt40 = 0
                max_ce_b = max_ce_t = max_ce_label_id = -1

            diag.ce = {
                "mean": ce_mean,
                "p95": ce_p95,
                "p99": ce_p99,
                "max": ce_max,
                "gt20": ce_gt20,
                "gt30": ce_gt30,
                "gt40": ce_gt40,
                "argmax_pos": (max_ce_b, max_ce_t),
                "argmax_label": max_ce_label_id,
            }

            del _diag_logits, ce_per_token

            # --- Embedding/Output weight stats ---
            emb_w = base_model.tok_emb.weight
            emb_std = float(emb_w.float().std().item())
            emb_max = float(emb_w.float().abs().max().item())
            out_w = base_model.output.weight
            tied = emb_w.data_ptr() == out_w.data_ptr()

            diag.weights = {
                "emb_std": emb_std,
                "emb_max": emb_max,
                "tied": tied,
            }
            diag._emb_w_ref = emb_w  # Keep reference for grad analysis

    except Exception as e:
        import traceback

        logger.warning(f"Diagnostic phase1 at step {diag.step} failed: {e}\n{traceback.format_exc()}")


def collect_phase2_preclip_grads(
    diag: StepDiagnostics,
    model: Any,
    logger: Any,
) -> None:
    """Phase 2: Collect gradient stats AFTER AMP unscale, BEFORE clip."""
    if not diag.active:
        return

    try:
        emb_w = diag._emb_w_ref
        if emb_w is not None and emb_w.grad is not None:
            g = emb_w.grad.float()
            diag.grad_preclip = {
                "emb_norm": float(g.norm().item()),
                "emb_max": float(g.abs().max().item()),
                "state": "after_unscale_before_clip",
            }
            # Row-slice stats: gradients for unique input tokens only
            uniq_in = diag._uniq_in
            if uniq_in is not None and uniq_in.numel() > 0:
                g_rows = g[uniq_in]  # [num_unique, dim]
                diag.grad_preclip["uniq_in_norm"] = float(g_rows.norm().item())
                diag.grad_preclip["uniq_in_maxabs"] = float(g_rows.abs().max().item())

        # Global grad norm (before clip)
        base = model._orig_mod if hasattr(model, "_orig_mod") else model
        global_preclip = 0.0
        for p in base.parameters():
            if p.grad is not None:
                global_preclip += p.grad.float().norm().item() ** 2
        diag.global_preclip = float(global_preclip**0.5)

    except Exception as e:
        logger.warning(f"Diagnostic phase2 pre-clip at step {diag.step} failed: {e}")


def collect_phase3_postclip_grads(
    diag: StepDiagnostics,
    model: Any,
    logger: Any,
) -> None:
    """Phase 3: Collect gradient stats AFTER clip."""
    if not diag.active:
        return

    try:
        emb_w = diag._emb_w_ref
        if emb_w is not None and emb_w.grad is not None:
            g = emb_w.grad.float()
            diag.grad_postclip = {
                "emb_norm": float(g.norm().item()),
                "emb_max": float(g.abs().max().item()),
            }
            uniq_in = diag._uniq_in
            if uniq_in is not None and uniq_in.numel() > 0:
                g_rows = g[uniq_in]
                diag.grad_postclip["uniq_in_norm"] = float(g_rows.norm().item())
                diag.grad_postclip["uniq_in_maxabs"] = float(g_rows.abs().max().item())

        # Global grad norm (after clip)
        base = model._orig_mod if hasattr(model, "_orig_mod") else model
        global_postclip = 0.0
        for p in base.parameters():
            if p.grad is not None:
                global_postclip += p.grad.float().norm().item() ** 2
        diag.global_postclip = float(global_postclip**0.5)

    except Exception as e:
        import traceback

        logger.warning(f"Diagnostic phase3 at step {diag.step} failed: {e}\n{traceback.format_exc()}")


def log_step_diagnostics(
    diag: StepDiagnostics,
    accum_loss: float,
    logger: Any,
) -> None:
    """Log the collected step diagnostics."""
    if not diag.active:
        return

    try:
        ld = diag.logits
        ce = diag.ce
        bt = diag.batch
        wt = diag.weights
        gp = diag.grad_preclip
        gq = diag.grad_postclip

        logger.warning(
            f"\n{'='*80}\n"
            f"📊 DIAG Step {diag.step}: loss={accum_loss:.4f}\n"
            f"{'='*80}\n"
            f"LOGITS: shape={ld.get('shape')}, dtype={ld.get('dtype')}\n"
            f"   finite_frac={ld.get('finite_frac', 0):.6f}, num_nan={ld.get('num_nan', 0)}, num_inf={ld.get('num_inf', 0)}\n"
            f"   (safe slice) std={ld.get('std', 0):.2f}, max={ld.get('max', 0):.1f}, mean={ld.get('mean', 0):.2f}\n"
            f"CE DISTRIBUTION:\n"
            f"   ce_mean={ce.get('mean', 0):.3f}, ce_p95={ce.get('p95', 0):.3f}, ce_p99={ce.get('p99', 0):.3f}, ce_max={ce.get('max', 0):.3f}\n"
            f"   count(ce>20)={ce.get('gt20', 0)}, count(ce>30)={ce.get('gt30', 0)}, count(ce>40)={ce.get('gt40', 0)}\n"
            f"   argmax_CE: pos=(b={ce.get('argmax_pos', (-1,-1))[0]}, t={ce.get('argmax_pos', (-1,-1))[1]}), label_id={ce.get('argmax_label', -1)}\n"
            f"BATCH TOKENS:\n"
            f"   x: min_id={bt.get('x_min', 0)}, max_id={bt.get('x_max', 0)}\n"
            f"   y: min_id={bt.get('y_min', 0)}, max_id={bt.get('y_max', 0)}, valid_count={bt.get('y_valid', 0)}\n"
            f"   top10_tokens={bt.get('top10_toks', [])}\n"
            f"   unique_count={bt.get('unique_count', 0)}, singleton_count={bt.get('singleton_count', 0)}\n"
            f"   special: eos={bt.get('eos_count', 0)}, pad={bt.get('pad_count', 0)}, unk(id=0)={bt.get('unk_count', 0)}\n"
            f"WEIGHTS: emb_std={wt.get('emb_std', 0):.5f}, emb_max={wt.get('emb_max', 0):.5f}, tied={wt.get('tied', False)}\n"
            f"GRADIENTS (after AMP unscale):\n"
            f"   global_norm: pre_clip={diag.global_preclip:.2e}, post_clip={diag.global_postclip:.2e}\n"
            f"   emb/lm_head: pre_clip(norm={gp.get('emb_norm', 0):.2e}, max={gp.get('emb_max', 0):.2e})\n"
            f"               post_clip(norm={gq.get('emb_norm', 0):.2e}, max={gq.get('emb_max', 0):.2e})\n"
            f"   row-slice [uniq_in]: pre_clip(norm={gp.get('uniq_in_norm', 0):.2e}, maxabs={gp.get('uniq_in_maxabs', 0):.2e})\n"
            f"                        post_clip(norm={gq.get('uniq_in_norm', 0):.2e}, maxabs={gq.get('uniq_in_maxabs', 0):.2e})\n"
            f"{'='*80}"
        )

        # Clean up after logging
        diag.clear()

    except Exception as e:
        import traceback

        logger.warning(f"Diagnostic logging at step {diag.step} failed: {e}\n{traceback.format_exc()}")
        diag.clear()
