from __future__ import annotations

from typing import Any, Callable, Optional

import time
import math
import heapq

import torch


def log_gradient_pathology_diagnostic(
    *,
    logger: Any,
    step: int,
    pre_clip_norm: float,
    grad_norm: float,
    grad_clip: float,
    clip_coef: float,
    clip_scale: float,
    scaler_scale: Optional[float] = None,
    lr: float,
    lr_effective: float,
    spike_detected: bool,
    grad_spike_lr_factor: float,
    accum_loss: float,
    last_ce_loss,
    last_aux_loss,
    last_ponder_loss,
    micro_diag: list[dict],
    grad_info_pre_clip: list[tuple[str, float, float, bool, bool]],
    base_model,
    verbose: bool = False,
) -> None:
    """Log gradient spike diagnostic.
    
    Prints compact spike summary: global pre/post norm, early-layer o_proj grad norms,
    and (optionally) early-layer attention output RMS if the model exposes it.
    """
    import torch
    
    # Get layer 0/1 attention o_proj *grad* norms (pre-clip stats list).
    l0_o_proj_grad_norm = 0.0
    l1_o_proj_grad_norm = 0.0
    for name, g_norm, g_max, has_nan, has_inf in grad_info_pre_clip:
        if "layers.0.attention.o_proj" in name:
            l0_o_proj_grad_norm = g_norm
        elif "layers.1.attention.o_proj" in name:
            l1_o_proj_grad_norm = g_norm

    def _fmt_opt_scalar(v: Optional[float]) -> str:
        if v is None:
            return "n/a"
        if not math.isfinite(v):
            return "n/a"
        return f"{v:.2e}"

    # Optional: attention output RMS + finiteness stats stored by blocks.
    l0_attn_out_rms: Optional[float] = None
    l1_attn_out_rms: Optional[float] = None
    l0_attn_finite_frac: Optional[float] = None
    l1_attn_finite_frac: Optional[float] = None
    l0_attn_nan: Optional[int] = None
    l1_attn_nan: Optional[int] = None
    l0_attn_inf: Optional[int] = None
    l1_attn_inf: Optional[int] = None
    l0_attn_dtype: Optional[str] = None
    l1_attn_dtype: Optional[str] = None
    l0_attn_shape: Optional[tuple] = None
    l1_attn_shape: Optional[tuple] = None
    try:
        layers = getattr(base_model, "layers", None)
        if layers is not None and len(layers) > 0:
            if len(layers) > 0:
                t0 = getattr(layers[0], "_last_attn_out_rms_t", None)
                # Don't require `_enable_attn_out_rms` here: the trainer may populate
                # these fields on-demand (e.g., only on spike steps) and then restore
                # the enable flag to avoid per-step overhead.
                if torch.is_tensor(t0) and t0.numel() == 1:
                    t0f = t0.detach().float()
                    if bool(torch.isfinite(t0f).item()):
                        l0_attn_out_rms = float(t0f.item())
                    l0_attn_dtype = getattr(layers[0], "_last_attn_out_dtype", None)
                    l0_attn_shape = getattr(layers[0], "_last_attn_out_shape", None)
                    ff0 = getattr(layers[0], "_last_attn_out_finite_frac_t", None)
                    if torch.is_tensor(ff0) and ff0.numel() == 1:
                        l0_attn_finite_frac = float(ff0.detach().float().item())
                    n0 = getattr(layers[0], "_last_attn_out_nan_ct_t", None)
                    if torch.is_tensor(n0) and n0.numel() == 1:
                        l0_attn_nan = int(n0.detach().to(dtype=torch.int64).item())
                    i0 = getattr(layers[0], "_last_attn_out_inf_ct_t", None)
                    if torch.is_tensor(i0) and i0.numel() == 1:
                        l0_attn_inf = int(i0.detach().to(dtype=torch.int64).item())
            if len(layers) > 1:
                t1 = getattr(layers[1], "_last_attn_out_rms_t", None)
                if torch.is_tensor(t1) and t1.numel() == 1:
                    t1f = t1.detach().float()
                    if bool(torch.isfinite(t1f).item()):
                        l1_attn_out_rms = float(t1f.item())
                    l1_attn_dtype = getattr(layers[1], "_last_attn_out_dtype", None)
                    l1_attn_shape = getattr(layers[1], "_last_attn_out_shape", None)
                    ff1 = getattr(layers[1], "_last_attn_out_finite_frac_t", None)
                    if torch.is_tensor(ff1) and ff1.numel() == 1:
                        l1_attn_finite_frac = float(ff1.detach().float().item())
                    n1 = getattr(layers[1], "_last_attn_out_nan_ct_t", None)
                    if torch.is_tensor(n1) and n1.numel() == 1:
                        l1_attn_nan = int(n1.detach().to(dtype=torch.int64).item())
                    i1 = getattr(layers[1], "_last_attn_out_inf_ct_t", None)
                    if torch.is_tensor(i1) and i1.numel() == 1:
                        l1_attn_inf = int(i1.detach().to(dtype=torch.int64).item())
    except Exception:
        pass

    def _fmt_attn_stats(rms: Optional[float], ff: Optional[float], n_nan: Optional[int], n_inf: Optional[int], dtype: Optional[str], shape: Optional[tuple]) -> str:
        rms_s = _fmt_opt_scalar(rms)
        ff_s = "n/a" if ff is None or not math.isfinite(float(ff)) else f"{float(ff):.6f}"
        nan_s = "n/a" if n_nan is None else str(int(n_nan))
        inf_s = "n/a" if n_inf is None else str(int(n_inf))
        dt_s = dtype if isinstance(dtype, str) and dtype else "n/a"
        sh_s = str(shape) if isinstance(shape, tuple) and shape else "n/a"
        inconsistent = False
        if ff is not None and math.isfinite(float(ff)) and float(ff) >= 1.0:
            if rms is not None and not math.isfinite(float(rms)):
                inconsistent = True
        tag = "(!)" if inconsistent else ""
        return f"rms={rms_s},ff={ff_s},nan={nan_s},inf={inf_s},dtype={dt_s},shape={sh_s}{tag}"

    # Identify the parameters driving the global norm on spike steps.
    # Note: grad_info_pre_clip is only collected on spike/non-finite steps.
    def _off_key(t: tuple[str, float, float, bool, bool]) -> float:
        _name, g_norm, _g_max, has_nan, has_inf = t
        if has_nan or has_inf:
            return float("inf")
        return float(g_norm)

    top_offenders = heapq.nlargest(3, grad_info_pre_clip, key=_off_key)
    top_str = ", ".join(
        [
            f"{n}:{(float('inf') if (nan or inf) else g):.2e}|max{mx:.2e}{'(!)' if (nan or inf) else ''}"
            for (n, g, mx, nan, inf) in top_offenders
        ]
    )
    
    # Compact one-line spike summary
    scale_s = "n/a"
    if scaler_scale is not None and math.isfinite(float(scaler_scale)):
        scale_s = f"{float(scaler_scale):.1f}"
    logger.warning(
        f"  ⚠️  Step {step}: grad={pre_clip_norm:.2e}→{grad_norm:.2e}, "
        f"amp_scale={scale_s}, "
        f"L0/L1_o_proj_grad=[{l0_o_proj_grad_norm:.2e},{l1_o_proj_grad_norm:.2e}], "
        f"L0_attn=[{_fmt_attn_stats(l0_attn_out_rms, l0_attn_finite_frac, l0_attn_nan, l0_attn_inf, l0_attn_dtype, l0_attn_shape)}], "
        f"L1_attn=[{_fmt_attn_stats(l1_attn_out_rms, l1_attn_finite_frac, l1_attn_nan, l1_attn_inf, l1_attn_dtype, l1_attn_shape)}], "
        f"top_grad=[{top_str}], "
        f"loss={accum_loss:.4f}, LR={lr:.2e}→{lr_effective:.2e}"
    )


def skip_update_for_nonfinite_gradients(
    *,
    nonfinite_grads: bool,
    optimizer,
    use_scaler: bool,
    scaler,
) -> bool:
    if not nonfinite_grads:
        return False
    optimizer.zero_grad(set_to_none=True)
    if use_scaler:
        scaler.update()
    return True


def reset_optimizer_moments_for_gradient_spike(
    *,
    spike_detected: bool,
    grad_spike_reset_moments: bool,
    grad_spike_topk: int,
    grad_info_pre_clip: list[tuple[str, float, float, bool, bool]],
    base_model,
    optimizer,
    logger: Optional[Any] = None,
) -> None:
    """Reset Adam/AdamW moments for top-k gradient offenders during a spike.
    
    If logger is None, operates silently (caller logs consolidated message).
    """
    if not spike_detected or not grad_spike_reset_moments:
        return

    try:
        offenders = sorted(
            grad_info_pre_clip,
            key=lambda t: float("inf") if (t[3] or t[4]) else t[1],
            reverse=True,
        )[:grad_spike_topk]
        offender_names = {n for (n, *_rest) in offenders}

        for name, p in base_model.named_parameters():
            if p.grad is None or name not in offender_names:
                continue
            st = optimizer.state.get(p)
            if not st:
                continue
            if "exp_avg" in st and torch.is_tensor(st["exp_avg"]):
                st["exp_avg"].zero_()
            if "exp_avg_sq" in st and torch.is_tensor(st["exp_avg_sq"]):
                st["exp_avg_sq"].zero_()
            if "max_exp_avg_sq" in st and torch.is_tensor(st["max_exp_avg_sq"]):
                st["max_exp_avg_sq"].zero_()

        if logger is not None:
            logger.warning(f"  🔧 Spike response: reset Adam moments for top {len(offenders)} params")
    except Exception as e:
        if logger is not None:
            logger.warning(f"  ⚠️  Spike response: moment reset failed ({e})")


def maybe_prepare_halt_on_spike(
    *,
    spike_detected: bool,
    halt_on_spike: bool,
    step: int,
    accum_loss: float,
    lr_effective: float,
    grad_norm: float,
    tokens_per_step: int,
    step_start: float,
    metrics,
    save_checkpoint: Callable[[int], None],
    logger: Any,
) -> bool:
    if not (spike_detected and halt_on_spike):
        return False

    try:
        halt_step_time = max(1e-9, time.time() - step_start)
        halt_tps = tokens_per_step / halt_step_time
        metrics.update(step, accum_loss, lr_effective, grad_norm, halt_tps, halt_step_time)
        metrics.total_tokens += tokens_per_step
        metrics.final_loss = accum_loss
        save_checkpoint(step)
    except Exception as e:
        logger.warning(f"  ⚠️  Halt-on-spike: failed to record/save state ({e})")

    return True
