"""Gradient spike diagnostics and response handling.

This module centralizes all gradient spike detection, analysis, logging, and
response logic. The Trainer calls into this module when a spike is detected,
keeping the training loop clean and readable.

Spike diagnostics include:
- Global gradient norm analysis (pre/post clip)
- Per-layer gradient attribution (top offenders)
- Attention output RMS analysis (early layers)
- Batch/data analysis (token distribution, outliers)
- Probable spike source identification
"""

from __future__ import annotations

import math
import os
import heapq
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch


@dataclass
class SpikeContext:
    """Container for all spike-relevant diagnostic data."""

    step: int
    pre_clip_norm: float
    grad_norm: float  # post-clip
    grad_clip: float
    threshold: float
    accum_loss: float
    ce_loss: float = 0.0
    aux_loss: float = 0.0
    ponder_loss: float = 0.0
    advantage_loss: float = 0.0
    lr: float = 0.0
    lr_effective: float = 0.0
    scaler_scale: Optional[float] = None
    clip_coef: float = 1.0
    clip_scale: float = 1.0

    # Populated during analysis
    grad_info: List[Tuple[str, float, float, bool, bool]] = field(default_factory=list)
    batch_stats: Dict[str, Any] = field(default_factory=dict)
    layer_attn_stats: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    probable_causes: List[str] = field(default_factory=list)
    top_offenders: List[Tuple[str, float, float]] = field(default_factory=list)


def compute_spike_threshold(
    n_params: int,
    model_size_hint: str = "",
    env_threshold: float = 0.0,
    env_scale: float = 2000.0,
    env_min: Optional[float] = None,
) -> float:
    """Compute gradient spike threshold scaled by model size.

    For 500M+ models, use higher minimum to avoid constant spike logging.
    """
    is_large = (
        model_size_hint.lower() in ("500m", "0.5b", "500_m")
        or n_params >= 450_000_000
    )
    default_min = 1e9 if is_large else 1e6

    if env_min is not None:
        spike_min = env_min
    else:
        spike_min = default_min

    if env_threshold > 0.0:
        return env_threshold
    elif n_params > 0:
        return max(spike_min, env_scale * math.sqrt(float(n_params)))
    else:
        return spike_min


def is_spike(pre_clip_norm: float, threshold: float) -> bool:
    """Check if gradient norm constitutes a spike."""
    return math.isfinite(pre_clip_norm) and (pre_clip_norm > threshold)


def collect_gradient_info(
    base_model,
    clip_scale: float,
) -> List[Tuple[str, float, float, bool, bool]]:
    """Collect per-parameter gradient info (norm, max, nan, inf).

    Returns list of (name, norm, max, has_nan, has_inf) tuples.
    Undoes clip scaling to estimate pre-clip magnitudes.
    """
    grad_info = []
    denom = clip_scale if clip_scale > 0.0 else 1.0

    try:
        for name, param in base_model.named_parameters():
            if param.grad is None:
                continue
            g = param.grad.detach()
            g_norm = float(g.float().norm().item()) / denom
            g_max = float(g.float().abs().max().item()) / denom
            has_nan = bool(torch.isnan(g).any().item())
            has_inf = bool(torch.isinf(g).any().item())
            grad_info.append((name, g_norm, g_max, has_nan, has_inf))
    except Exception:
        pass

    return grad_info


def collect_batch_stats(
    x: torch.Tensor,
    y: torch.Tensor,
    vocab_size: int,
) -> Dict[str, Any]:
    """Collect batch token statistics for spike source analysis."""
    stats = {}
    try:
        with torch.no_grad():
            x_flat = x.view(-1)
            y_flat = y.view(-1)
            y_valid_mask = y_flat != -100
            y_valid = y_flat[y_valid_mask]

            # Basic token range
            stats["x_min"] = int(x_flat.min().item())
            stats["x_max"] = int(x_flat.max().item())
            stats["y_min"] = int(y_valid.min().item()) if y_valid.numel() > 0 else 0
            stats["y_max"] = int(y_valid.max().item()) if y_valid.numel() > 0 else 0
            stats["y_valid_count"] = int(y_valid.numel())
            stats["batch_size"] = int(x.shape[0])
            stats["seq_len"] = int(x.shape[1])
            stats["total_tokens"] = int(x_flat.numel())

            # Out-of-bounds detection
            x_oob = int(((x_flat < 0) | (x_flat >= vocab_size)).sum().item())
            y_oob = int((y_valid_mask & ((y_flat < 0) | (y_flat >= vocab_size))).sum().item())
            stats["x_oob"] = x_oob
            stats["y_oob"] = y_oob

            # Special tokens (GPT-2: 50256=EOS, 50257=PAD typically)
            eos_id = 50256
            stats["eos_count"] = int((x_flat == eos_id).sum().item())
            stats["pad_count"] = int((x_flat == 50257).sum().item()) if stats["x_max"] >= 50257 else 0
            stats["unk_count"] = int((x_flat == 0).sum().item())  # Token 0 often unused/UNK

            # Token distribution analysis
            unique_toks, counts = x_flat.unique(return_counts=True)
            stats["unique_count"] = int(unique_toks.numel())
            stats["singleton_count"] = int((counts == 1).sum().item())

            # Highly repeated tokens (potential padding/anomaly)
            top_idx = counts.argsort(descending=True)[:5]
            stats["top_tokens"] = [
                (int(unique_toks[i].item()), int(counts[i].item()))
                for i in top_idx
            ]

            # Check for suspicious patterns
            stats["high_repeat_ratio"] = float(counts.max().item() / max(1, x_flat.numel()))
            stats["low_diversity"] = stats["unique_count"] < stats["total_tokens"] * 0.1

    except Exception as e:
        stats["error"] = str(e)

    return stats


def collect_attention_layer_stats(
    base_model,
    layer_indices: List[int] = [0, 1],
) -> Dict[int, Dict[str, Any]]:
    """Collect attention output stats from early layers."""
    layer_stats = {}
    try:
        layers = getattr(base_model, "layers", None)
        if layers is None:
            return layer_stats

        for li in layer_indices:
            if li >= len(layers):
                continue

            layer = layers[li]
            stats = {}

            def _to_float(t) -> Optional[float]:
                if torch.is_tensor(t) and t.numel() == 1:
                    v = t.detach().float().item()
                    return float(v) if math.isfinite(v) else None
                return None

            def _to_int(t) -> Optional[int]:
                if torch.is_tensor(t) and t.numel() == 1:
                    return int(t.detach().to(dtype=torch.int64).item())
                return None

            stats["rms"] = _to_float(getattr(layer, "_last_attn_out_rms_t", None))
            stats["finite_frac"] = _to_float(getattr(layer, "_last_attn_out_finite_frac_t", None))
            stats["nan_count"] = _to_int(getattr(layer, "_last_attn_out_nan_ct_t", None))
            stats["inf_count"] = _to_int(getattr(layer, "_last_attn_out_inf_ct_t", None))
            stats["dtype"] = getattr(layer, "_last_attn_out_dtype", None)
            stats["shape"] = getattr(layer, "_last_attn_out_shape", None)

            layer_stats[li] = stats

    except Exception:
        pass

    return layer_stats


def trigger_attention_diagnostics(
    base_model,
    x: torch.Tensor,
    device: str,
    dtype: torch.dtype,
    layer_indices: List[int] = [0, 1],
) -> None:
    """Temporarily enable attention output diagnostics and run forward pass.

    This populates layer._last_attn_out_* fields for spike analysis without
    keeping the expensive diagnostics enabled for every step.
    """
    try:
        from torch.amp import autocast

        layers = getattr(base_model, "layers", None)
        if layers is None or len(layers) == 0:
            return

        # Save previous state
        prev_state = {}
        for li in layer_indices:
            if li < len(layers):
                prev_state[li] = bool(getattr(layers[li], "_enable_attn_out_rms", False))
                layers[li]._enable_attn_out_rms = True

        # Run forward pass to collect stats
        try:
            with torch.no_grad():
                was_training = bool(getattr(base_model, "training", True))
                base_model.eval()
                try:
                    with autocast(device, dtype=dtype):
                        _ = base_model(x)
                finally:
                    base_model.train(was_training)
        finally:
            # Restore previous state
            for li, prev in prev_state.items():
                layers[li]._enable_attn_out_rms = prev

    except Exception:
        pass


def identify_top_gradient_offenders(
    grad_info: List[Tuple[str, float, float, bool, bool]],
    top_k: int = 5,
) -> List[Tuple[str, float, float]]:
    """Identify parameters with largest gradient norms."""
    def key_fn(t):
        name, g_norm, g_max, has_nan, has_inf = t
        if has_nan or has_inf:
            return float("inf")
        return g_norm

    top = heapq.nlargest(top_k, grad_info, key=key_fn)
    return [(name, g_norm, g_max) for (name, g_norm, g_max, _, _) in top]


def analyze_spike_causes(ctx: SpikeContext) -> List[str]:
    """Analyze spike context and identify probable causes."""
    causes = []

    # Check for data issues
    bs = ctx.batch_stats
    if bs.get("x_oob", 0) > 0 or bs.get("y_oob", 0) > 0:
        causes.append(f"OUT_OF_BOUNDS_TOKENS: x_oob={bs.get('x_oob')}, y_oob={bs.get('y_oob')}")

    if bs.get("high_repeat_ratio", 0) > 0.5:
        causes.append(f"HIGH_TOKEN_REPETITION: {bs.get('high_repeat_ratio', 0):.1%} of batch is single token")

    if bs.get("low_diversity", False):
        causes.append(f"LOW_TOKEN_DIVERSITY: only {bs.get('unique_count', 0)} unique tokens")

    if bs.get("y_valid_count", 0) < bs.get("total_tokens", 1) * 0.1:
        causes.append(f"FEW_VALID_TARGETS: only {bs.get('y_valid_count', 0)} valid (non-ignored) targets")

    # Check for attention layer issues
    for li, stats in ctx.layer_attn_stats.items():
        nan_ct = stats.get("nan_count") or 0
        inf_ct = stats.get("inf_count") or 0
        ff = stats.get("finite_frac")

        if nan_ct > 0 or inf_ct > 0:
            causes.append(f"LAYER_{li}_NONFINITE_ATTN: nan={nan_ct}, inf={inf_ct}")
        elif ff is not None and ff < 1.0:
            causes.append(f"LAYER_{li}_PARTIAL_NONFINITE: finite_frac={ff:.4f}")

        rms = stats.get("rms")
        if rms is not None and rms > 1e4:
            causes.append(f"LAYER_{li}_HIGH_ATTN_RMS: {rms:.2e}")

    # Check gradient structure
    if ctx.top_offenders:
        # Embedding gradients dominating often indicates data issue
        emb_in_top = any("tok_emb" in name or "embed" in name.lower() for name, _, _ in ctx.top_offenders[:3])
        if emb_in_top:
            causes.append("EMBEDDING_GRADIENT_SPIKE: tok_emb in top-3 offenders (likely data/tokenization issue)")

        # Early layer attention dominating
        early_attn = any("layers.0.attention" in name or "layers.1.attention" in name for name, _, _ in ctx.top_offenders[:3])
        if early_attn:
            causes.append("EARLY_ATTENTION_GRADIENT_SPIKE: layer 0/1 attention in top offenders")

    # Loss-related
    if ctx.accum_loss > 20:
        causes.append(f"HIGH_LOSS: {ctx.accum_loss:.2f} (very high CE often triggers embedding grad spikes)")

    if not causes:
        causes.append("CAUSE_UNKNOWN: spike threshold exceeded but no clear anomaly detected")

    return causes


def format_spike_summary(ctx: SpikeContext) -> str:
    """Format a comprehensive one-line spike summary for logging."""
    # Format attention stats
    def fmt_attn(stats: Dict[str, Any]) -> str:
        if not stats:
            return "n/a"
        rms = stats.get("rms")
        ff = stats.get("finite_frac")
        nan = stats.get("nan_count") or 0
        inf = stats.get("inf_count") or 0
        dtype = stats.get("dtype") or "?"
        shape = stats.get("shape") or "?"
        rms_s = f"{rms:.2e}" if rms is not None else "n/a"
        ff_s = f"{ff:.6f}" if ff is not None else "n/a"
        return f"rms={rms_s},ff={ff_s},nan={nan},inf={inf},dtype={dtype},shape={shape}"

    l0_stats = ctx.layer_attn_stats.get(0, {})
    l1_stats = ctx.layer_attn_stats.get(1, {})

    # Format top offenders
    top_str = ", ".join(
        f"{name}:{norm:.2e}|max{mx:.2e}"
        for name, norm, mx in ctx.top_offenders[:3]
    )

    # Format layer 0/1 o_proj grad norms (grad_info is 5-tuple: name, norm, max, has_nan, has_inf)
    l0_o_proj = next((norm for name, norm, *_ in ctx.grad_info if "layers.0.attention.o_proj" in name), 0.0)
    l1_o_proj = next((norm for name, norm, *_ in ctx.grad_info if "layers.1.attention.o_proj" in name), 0.0)

    scale_s = f"{ctx.scaler_scale:.1f}" if ctx.scaler_scale is not None and math.isfinite(ctx.scaler_scale) else "n/a"

    return (
        f"Step {ctx.step}: grad={ctx.pre_clip_norm:.2e}→{ctx.grad_norm:.2e}, "
        f"amp_scale={scale_s}, "
        f"L0/L1_o_proj_grad=[{l0_o_proj:.2e},{l1_o_proj:.2e}], "
        f"L0_attn=[{fmt_attn(l0_stats)}], "
        f"L1_attn=[{fmt_attn(l1_stats)}], "
        f"top_grad=[{top_str}], "
        f"loss={ctx.accum_loss:.4f}, LR={ctx.lr:.2e}→{ctx.lr_effective:.2e}"
    )


def format_spike_detail(ctx: SpikeContext) -> str:
    """Format detailed multi-line spike analysis."""
    lines = [
        "",
        "=" * 80,
        f"🔥 GRADIENT SPIKE ANALYSIS - Step {ctx.step}",
        "=" * 80,
        "",
        "GRADIENT STATE:",
        f"  Pre-clip norm: {ctx.pre_clip_norm:.4e} (threshold: {ctx.threshold:.2e})",
        f"  Post-clip norm: {ctx.grad_norm:.4e}",
        f"  Clip scale: {ctx.clip_scale:.6f}",
        f"  AMP scaler: {ctx.scaler_scale if ctx.scaler_scale else 'n/a'}",
        "",
        "LOSS COMPONENTS:",
        f"  Total: {ctx.accum_loss:.4f}",
        f"  CE: {ctx.ce_loss:.4f}",
        f"  Aux: {ctx.aux_loss:.4f}",
        f"  Ponder: {ctx.ponder_loss:.4f}",
        f"  Advantage: {ctx.advantage_loss:.4f}",
        "",
        "TOP GRADIENT OFFENDERS:",
    ]

    for i, (name, norm, mx) in enumerate(ctx.top_offenders[:5]):
        lines.append(f"  {i+1}. {name}: norm={norm:.2e}, max={mx:.2e}")

    bs = ctx.batch_stats
    if bs:
        lines.extend([
            "",
            "BATCH STATISTICS:",
            f"  Shape: ({bs.get('batch_size', '?')}, {bs.get('seq_len', '?')})",
            f"  Token range: x=[{bs.get('x_min', '?')}, {bs.get('x_max', '?')}], y=[{bs.get('y_min', '?')}, {bs.get('y_max', '?')}]",
            f"  Valid targets: {bs.get('y_valid_count', '?')} / {bs.get('total_tokens', '?')}",
            f"  OOB tokens: x={bs.get('x_oob', 0)}, y={bs.get('y_oob', 0)}",
            f"  Unique tokens: {bs.get('unique_count', '?')} (singletons: {bs.get('singleton_count', '?')})",
            f"  Special tokens: eos={bs.get('eos_count', 0)}, pad={bs.get('pad_count', 0)}, unk={bs.get('unk_count', 0)}",
            f"  Top tokens: {bs.get('top_tokens', [])}",
        ])

    if ctx.layer_attn_stats:
        lines.extend(["", "ATTENTION LAYER DIAGNOSTICS:"])
        for li, stats in sorted(ctx.layer_attn_stats.items()):
            rms = stats.get("rms")
            ff = stats.get("finite_frac")
            nan = stats.get("nan_count") or 0
            inf = stats.get("inf_count") or 0
            lines.append(
                f"  Layer {li}: rms={rms if rms else 'n/a':.2e}, "
                f"finite_frac={ff if ff else 'n/a':.6f}, nan={nan}, inf={inf}"
            )

    lines.extend(["", "PROBABLE CAUSES:"])
    for cause in ctx.probable_causes:
        lines.append(f"  • {cause}")

    lines.extend(["", "=" * 80, ""])

    return "\n".join(lines)


def dump_bad_batch(
    checkpoint_dir: str,
    step: int,
    reason: str,
    x: torch.Tensor,
    y: torch.Tensor,
    ctx: Optional[SpikeContext] = None,
) -> str:
    """Save batch data for post-mortem analysis."""
    path = os.path.join(checkpoint_dir, f"bad_batch_step_{step}.pt")
    data = {
        "step": step,
        "reason": reason,
        "x": x.detach().cpu(),
        "y": y.detach().cpu(),
        "timestamp": datetime.now().isoformat(),
    }
    if ctx:
        data["spike_context"] = {
            "pre_clip_norm": ctx.pre_clip_norm,
            "threshold": ctx.threshold,
            "loss": ctx.accum_loss,
            "batch_stats": ctx.batch_stats,
            "probable_causes": ctx.probable_causes,
            "top_offenders": [(n, norm, mx) for n, norm, mx in ctx.top_offenders[:5]],
        }
    torch.save(data, path)
    return path


class SpikeTracker:
    """Track spike rate over a rolling window for halt policy decisions."""

    def __init__(self, window_size: int = 200):
        self.window_size = window_size
        self._spike_steps: List[int] = []

    def record_spike(self, step: int) -> None:
        self._spike_steps.append(step)
        # Prune old entries
        while self._spike_steps and (step - self._spike_steps[0]) >= self.window_size:
            self._spike_steps.pop(0)

    def spike_count_in_window(self) -> int:
        return len(self._spike_steps)

    def should_halt(self, threshold_count: int) -> bool:
        return len(self._spike_steps) >= threshold_count


def handle_spike(
    *,
    step: int,
    pre_clip_norm: float,
    grad_norm: float,
    grad_clip: float,
    threshold: float,
    accum_loss: float,
    ce_loss: float,
    aux_loss: float,
    ponder_loss: float,
    advantage_loss: float,
    lr: float,
    lr_effective: float,
    scaler_scale: Optional[float],
    clip_coef: float,
    clip_scale: float,
    base_model,
    x: torch.Tensor,
    y: torch.Tensor,
    vocab_size: int,
    device: str,
    dtype: torch.dtype,
    checkpoint_dir: str,
    logger: Any,
    spike_tracker: Optional[SpikeTracker] = None,
    verbose: bool = False,
    dump_batch: bool = False,
) -> SpikeContext:
    """Central spike handling: analyze, log, and optionally dump batch.

    Returns populated SpikeContext for further decision-making by caller.
    """
    ctx = SpikeContext(
        step=step,
        pre_clip_norm=pre_clip_norm,
        grad_norm=grad_norm,
        grad_clip=grad_clip,
        threshold=threshold,
        accum_loss=accum_loss,
        ce_loss=ce_loss,
        aux_loss=aux_loss,
        ponder_loss=ponder_loss,
        advantage_loss=advantage_loss,
        lr=lr,
        lr_effective=lr_effective,
        scaler_scale=scaler_scale,
        clip_coef=clip_coef,
        clip_scale=clip_scale,
    )

    # Collect gradient info
    ctx.grad_info = collect_gradient_info(base_model, clip_scale)
    ctx.top_offenders = identify_top_gradient_offenders(ctx.grad_info)

    # Collect batch stats
    ctx.batch_stats = collect_batch_stats(x, y, vocab_size)

    # Trigger and collect attention diagnostics
    trigger_attention_diagnostics(base_model, x, device, dtype)
    ctx.layer_attn_stats = collect_attention_layer_stats(base_model)

    # Analyze causes
    ctx.probable_causes = analyze_spike_causes(ctx)

    # Log summary (always)
    logger.warning(f"  ⚠️  {format_spike_summary(ctx)}")

    # Log detailed analysis if verbose
    if verbose:
        logger.warning(format_spike_detail(ctx))

    # Track spike
    if spike_tracker:
        spike_tracker.record_spike(step)

    # Optionally dump batch
    if dump_batch:
        path = dump_bad_batch(checkpoint_dir, step, "gradient_spike", x, y, ctx)
        logger.warning(f"  📦 Dumped spike batch to: {path}")

    return ctx


def check_attention_nonfinite(
    base_model,
    step: int,
    checkpoint_dir: str,
    x: torch.Tensor,
    y: torch.Tensor,
    logger: Any,
) -> Optional[str]:
    """Check for non-finite attention outputs.

    Returns path to dumped batch if non-finite detected, else None.
    """
    try:
        layers = getattr(base_model, "layers", None)
        if layers is None or len(layers) == 0:
            return None

        # Check if diagnostics were populated
        has_diag = False
        for layer in layers:
            if (getattr(layer, "_last_attn_out_dtype", None) is not None or
                getattr(layer, "_last_attn_out_shape", None) is not None):
                has_diag = True
                break

        if not has_diag:
            return None

        bad_layers = []
        for li, layer in enumerate(layers):
            def _to_float(t):
                if torch.is_tensor(t) and t.numel() == 1:
                    return float(t.detach().float().item())
                return None

            def _to_int(t):
                if torch.is_tensor(t) and t.numel() == 1:
                    return int(t.detach().to(dtype=torch.int64).item())
                return None

            ff = _to_float(getattr(layer, "_last_attn_out_finite_frac_t", None))
            nan_ct = _to_int(getattr(layer, "_last_attn_out_nan_ct_t", None))
            inf_ct = _to_int(getattr(layer, "_last_attn_out_inf_ct_t", None))

            is_bad = False
            if ff is not None and ff < 1.0:
                is_bad = True
            if (nan_ct is not None and nan_ct > 0) or (inf_ct is not None and inf_ct > 0):
                is_bad = True

            if is_bad:
                bad_layers.append({
                    "layer": li,
                    "finite_frac": ff,
                    "nan": nan_ct,
                    "inf": inf_ct,
                })

        if bad_layers:
            path = dump_bad_batch(checkpoint_dir, step, "non_finite_attn_out", x, y)
            logger.warning(
                f"  🛑 Non-finite attention output detected (layers={len(bad_layers)}); dumped batch to {path}"
            )
            return path

    except Exception:
        pass

    return None
