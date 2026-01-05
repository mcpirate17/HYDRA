from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

import torch
import torch.nn.functional as F

from hydra.kernels import fused_chunked_cross_entropy

from .config import TrainingConfig


def update_scalar_ema(*, ema: float, value: float, alpha: float) -> float:
    if ema == 0.0:
        return float(value)
    return float(alpha) * float(value) + (1.0 - float(alpha)) * float(ema)


def resolve_micro_diag_tensors(micro_diag: list[dict]) -> None:
    if not micro_diag:
        return
    for md in micro_diag:
        for k, v in list(md.items()):
            if isinstance(v, torch.Tensor):
                md[k] = v.item()


@torch.no_grad()
def compute_token_losses_from_hidden(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
    token_chunk_size: int = 2048,
    vocab_chunk_size: int = 4096,
) -> torch.Tensor:
    """Compute per-token cross-entropy losses without storing full logits.

    This is used to support loss-driven routing (MoR) and loss-aware MoD
    supervision when training uses chunked CE (logits may be omitted).
    Returns [B, L] float32 losses with ignored positions set to 0.
    """
    B, L, D = hidden.shape
    V = weight.shape[0]
    N = B * L

    h = hidden.view(N, D).float()
    t = targets.view(N)
    valid = t != ignore_index

    # Defensive bounds check: clamp target IDs to valid vocab range.
    # Out-of-bounds targets cause CUDA illegal memory access.
    valid_targets = t[valid]
    if valid_targets.numel() > 0:
        oob_mask = (valid_targets < 0) | (valid_targets >= V)
        if oob_mask.any():
            n_oob = oob_mask.sum().item()
            t_max = valid_targets.max().item()
            t_min = valid_targets.min().item()
            import logging
            logging.getLogger("dmta.training").warning(
                f"compute_token_losses_from_hidden: {n_oob} targets out of vocab range "
                f"[0, {V}). min={t_min}, max={t_max}. Clamping to valid range."
            )
            # Clamp in-place on the flat view
            t.clamp_(min=0, max=V - 1)
            # Recompute valid mask after clamping (ignore_index might have changed)
            valid = t != ignore_index

    # Correct-class logits (only for valid positions)
    correct_logits = torch.zeros((N,), device=hidden.device, dtype=torch.float32)
    if valid.any():
        w_y = weight[t[valid]].float()  # [M, D]
        correct_logits[valid] = (h[valid] * w_y).sum(dim=1)

    losses = torch.zeros((N,), device=hidden.device, dtype=torch.float32)
    for t0 in range(0, N, token_chunk_size):
        t1 = min(N, t0 + token_chunk_size)
        h_chunk = h[t0:t1]  # [T, D]
        lse = torch.full((t1 - t0,), -float("inf"), device=hidden.device, dtype=torch.float32)
        for v0 in range(0, V, vocab_chunk_size):
            v1 = min(V, v0 + vocab_chunk_size)
            w_chunk = weight[v0:v1].float()  # [C, D]
            logits_chunk = h_chunk @ w_chunk.t()  # [T, C]
            lse = torch.logaddexp(lse, torch.logsumexp(logits_chunk, dim=1))
        losses[t0:t1] = lse - correct_logits[t0:t1]

    losses = losses * valid.float()
    return losses.view(B, L)


def compute_microbatch_loss(
    *,
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    config: TrainingConfig,
    device: str,
    dtype: torch.dtype,
    use_mod_mor: bool,
    track_loss_scalars: bool,
    compute_token_losses_from_hidden: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Compute training loss for one micro-batch.

    Returns:
      loss, ce_loss, logits, aux_loss, ponder_loss, advantage_loss, moe_aux_loss

    `logits` may be None when using chunked CE.
    The *_loss values returned are tensors (or None) suitable for detach() without sync.
    """

    logits: Optional[torch.Tensor]
    ce_loss: torch.Tensor
    aux_loss_t: Optional[torch.Tensor] = None
    ponder_loss_t: Optional[torch.Tensor] = None
    advantage_loss_t: Optional[torch.Tensor] = None

    if use_mod_mor:
        logits = None
        if config.use_chunked_ce and hasattr(model, "forward_hidden_with_losses"):
            hidden, aux_losses = model.forward_hidden_with_losses(x, mask=mask)
            base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            weight = base_model.output.weight
            ce_loss = fused_chunked_cross_entropy(
                hidden,
                weight,
                y,
                ignore_index=-100,
                chunk_size=config.chunked_ce_size,
            )
            if hasattr(base_model, "compute_advantage_loss_from_token_losses"):
                token_losses = compute_token_losses_from_hidden(hidden, weight, y)
                advantage_loss_t = base_model.compute_advantage_loss_from_token_losses(
                    token_losses,
                    y,
                    ignore_index=-100,
                )
            else:
                advantage_loss_t = None
        else:
            logits, aux_losses = model(x, mask=mask, return_losses=True)
            ce_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
                ignore_index=-100,
            )
            advantage_loss_t = None

        aux_loss = aux_losses.get("aux_loss", 0.0)
        ponder_loss = aux_losses.get("ponder_loss", 0.0)
        moe_aux_loss = aux_losses.get("moe_aux_loss", 0.0)
        moe_teacher_loss = aux_losses.get("moe_teacher_loss", 0.0)

        base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        if logits is not None and hasattr(base_model, "compute_advantage_loss"):
            advantage_loss_t = base_model.compute_advantage_loss(logits, y, ignore_index=-100)

        # Curriculum consistency: ponder_loss is gated/ramped inside the model
        # via MoR curriculum. Advantage loss is computed post-CE and must be
        # ramped in the same way, otherwise router gradients can dominate right
        # at MoR enable and cause depth-0 collapse.
        if advantage_loss_t is not None and hasattr(base_model, "get_mor_status"):
            try:
                mor_status = base_model.get_mor_status()
                phase = str(mor_status.get("phase", ""))
                if phase == "fixed-depth":
                    rampup_scale = 0.0
                elif phase == "full-adaptive":
                    rampup_scale = 1.0
                else:
                    # Match the model's quantized rampup behavior (10 discrete values)
                    progress = float(mor_status.get("rampup_progress", 0.0) or 0.0)
                    progress = max(0.0, min(1.0, progress))
                    rampup_scale = max(0.1, round(progress * 10.0) / 10.0)
                advantage_loss_t = advantage_loss_t * float(rampup_scale)
            except Exception:
                # If anything goes wrong, fall back to unscaled advantage.
                pass

        # Optional runtime multiplier (used by trainer auto-nudge).
        if advantage_loss_t is not None:
            try:
                advantage_loss_t = advantage_loss_t * float(getattr(config, "mor_advantage_loss_mult", 1.0))
            except Exception:
                pass

        if hasattr(aux_loss, "clamp"):
            aux_loss = aux_loss.clamp(max=100.0)
        if hasattr(ponder_loss, "clamp"):
            ponder_loss = ponder_loss.clamp(max=100.0)
        if hasattr(advantage_loss_t, "clamp"):
            advantage_loss_t = advantage_loss_t.clamp(max=10.0)
        if hasattr(moe_aux_loss, "clamp"):
            moe_aux_loss = moe_aux_loss.clamp(max=100.0)
        if hasattr(moe_teacher_loss, "clamp"):
            moe_teacher_loss = moe_teacher_loss.clamp(max=100.0)

        aux_loss_t = aux_loss if isinstance(aux_loss, torch.Tensor) else torch.tensor(aux_loss, device=device)
        ponder_loss_t = ponder_loss if isinstance(ponder_loss, torch.Tensor) else torch.tensor(ponder_loss, device=device)
        moe_aux_loss_t = moe_aux_loss if isinstance(moe_aux_loss, torch.Tensor) else torch.tensor(moe_aux_loss, device=device)
        moe_teacher_loss_t = (
            moe_teacher_loss
            if isinstance(moe_teacher_loss, torch.Tensor)
            else torch.tensor(moe_teacher_loss, device=device)
        )
        advantage_loss_t = (
            advantage_loss_t
            if isinstance(advantage_loss_t, torch.Tensor)
            else (torch.tensor(advantage_loss_t, device=device) if advantage_loss_t is not None else None)
        )

        # aux_loss_t is already scaled by aux_loss_weight in the model/router.
        # ponder_loss_t is unscaled (raw) from the router, so we scale it here.
        # moe_aux_loss_t is already scaled by moe_aux_weight in the MoE router.
        loss = ce_loss + aux_loss_t + config.ponder_scale * ponder_loss_t + moe_aux_loss_t
        # Domain-teacher loss: unscaled CE from router(s), scaled here by alpha.
        teacher_alpha = float(getattr(config, "moe_teacher_weight", 0.0) or 0.0)
        if teacher_alpha != 0.0:
            loss = loss + teacher_alpha * moe_teacher_loss_t
        if advantage_loss_t is not None:
            loss = loss + advantage_loss_t

        if track_loss_scalars:
            # Keep tensors; caller can store .detach() for later logging.
            pass

        return loss, ce_loss, logits, aux_loss_t, ponder_loss_t, advantage_loss_t, moe_aux_loss_t

    # Non-mod_mor
    if config.use_chunked_ce and hasattr(model, "forward_hidden"):
        hidden = model.forward_hidden(x, mask=mask)
        base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        weight = base_model.output.weight
        loss = fused_chunked_cross_entropy(
            hidden,
            weight,
            y,
            ignore_index=-100,
            chunk_size=config.chunked_ce_size,
        )
        logits = None
    else:
        logits = model(x, mask=mask)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1),
            ignore_index=-100,
        )

    return loss, loss, logits, None, None, None, None


@torch.no_grad()
def evaluate_fixed_batches(
    *,
    base_model,
    fixed_eval_batches: list[dict],
    device: str,
    dtype: torch.dtype,
    config: TrainingConfig,
    use_mod_mor: bool,
    eval_debug: bool = False,
) -> float:
    """Compute mean eval loss over a fixed set of batches.

    Uses chunked CE when enabled and supported; otherwise falls back to logits CE.
    """

    eval_loss = 0.0
    total_valid_tokens = 0
    device_type = "cuda" if device == "cuda" else "cpu"
    with torch.amp.autocast(device_type=device_type, dtype=dtype):
        for b_idx, b in enumerate(fixed_eval_batches):
            x_eval = b["input_ids"].to(device, non_blocking=True)
            y_eval = b["labels"].to(device, non_blocking=True)
            mask_eval = b.get("attention_mask")
            if mask_eval is not None:
                mask_eval = mask_eval.to(device, non_blocking=True)
            
            # Count valid tokens (non-ignored)
            valid_mask = y_eval != -100
            n_valid = int(valid_mask.sum().item())
            total_valid_tokens += n_valid
            
            if use_mod_mor:
                if config.use_chunked_ce and hasattr(base_model, "forward_hidden_with_losses"):
                    # Use forward_hidden_with_losses which accepts mask (forward_hidden doesn't)
                    hidden, _aux = base_model.forward_hidden_with_losses(x_eval, mask=mask_eval)
                    weight = base_model.output.weight
                    loss_eval = fused_chunked_cross_entropy(
                        hidden,
                        weight,
                        y_eval,
                        ignore_index=-100,
                        chunk_size=config.chunked_ce_size,
                    )
                    if eval_debug and b_idx == 0:
                        print(f"  [EVAL_DEBUG] batch 0: chunked_ce, n_valid={n_valid}/{y_eval.numel()}, loss={loss_eval.item():.4f}")
                    eval_loss += float(loss_eval.item())
                    continue
                logits, _aux = base_model(x_eval, mask=mask_eval, return_losses=True)
            else:
                logits = base_model(x_eval, mask=mask_eval) if mask_eval is not None else base_model(x_eval)
            loss_eval = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y_eval.view(-1),
                ignore_index=-100,
            )
            if eval_debug and b_idx == 0:
                print(f"  [EVAL_DEBUG] batch 0: logits_ce, n_valid={n_valid}/{y_eval.numel()}, loss={loss_eval.item():.4f}")
            eval_loss += float(loss_eval.item())
    
    if eval_debug:
        print(f"  [EVAL_DEBUG] total_valid_tokens across {len(fixed_eval_batches)} batches: {total_valid_tokens}")

    return eval_loss / max(1, len(fixed_eval_batches))


@torch.no_grad()
def eval_sanity_check_on_train_batch(
    *,
    base_model,
    train_batch: dict,
    device: str,
    dtype: torch.dtype,
    config: TrainingConfig,
    use_mod_mor: bool,
    current_train_loss: float,
    current_ema: float,
    logger: Any,
    last_ce_loss: Optional[float] = None,
) -> dict:
    """Run eval codepath on a training batch to verify consistency.
    
    This sanity check ensures the eval loss computation matches training CE.
    Expected: eval_on_train_batch ≈ last_ce_loss (within ~0.3).
    
    NOTE: When MoR/MoD are active, total training loss includes auxiliary terms
    (aux_loss, ponder_loss, advantage_loss) that eval does not compute.
    This check compares CE-to-CE, not total_loss-to-CE.
    
    Returns dict with diagnostics.
    """
    x = train_batch["input_ids"].to(device, non_blocking=True)
    y = train_batch["labels"].to(device, non_blocking=True)
    mask = train_batch.get("attention_mask")
    if mask is not None:
        mask = mask.to(device, non_blocking=True)
    
    # Count valid tokens (non-ignored)
    valid_mask = y != -100
    n_valid = int(valid_mask.sum().item())
    n_total = y.numel()
    
    device_type = "cuda" if device == "cuda" else "cpu"
    with torch.amp.autocast(device_type=device_type, dtype=dtype):
        # Method 1: Eval path (same as evaluate_fixed_batches)
        if use_mod_mor:
            if config.use_chunked_ce and hasattr(base_model, "forward_hidden_with_losses"):
                # Match evaluate_fixed_batches: forward_hidden_with_losses is mask-aware.
                hidden, _aux = base_model.forward_hidden_with_losses(x, mask=mask)
                weight = base_model.output.weight
                eval_loss = fused_chunked_cross_entropy(
                    hidden,
                    weight,
                    y,
                    ignore_index=-100,
                    chunk_size=config.chunked_ce_size,
                )
            else:
                logits, _aux = base_model(x, mask=mask, return_losses=True)
                eval_loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1),
                    ignore_index=-100,
                )
        else:
            logits = base_model(x, mask=mask) if mask is not None else base_model(x)
            eval_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
                ignore_index=-100,
            )
        
        eval_loss_val = float(eval_loss.item())
        
        # Method 2: Manual CE (reference implementation)
        # Force logits computation for comparison
        if use_mod_mor:
            logits_ref, _ = base_model(x, mask=mask, return_losses=True)
        else:
            logits_ref = base_model(x, mask=mask) if mask is not None else base_model(x)
        
        # Per-token CE (no reduction)
        ce_per_token = F.cross_entropy(
            logits_ref.view(-1, logits_ref.size(-1)),
            y.view(-1),
            ignore_index=-100,
            reduction="none",
        )
        
        # Manual mean over valid tokens
        valid_flat = valid_mask.view(-1)
        if valid_flat.any():
            manual_mean = ce_per_token[valid_flat].mean().item()
            manual_sum = ce_per_token[valid_flat].sum().item()
        else:
            manual_mean = 0.0
            manual_sum = 0.0
        
        # PyTorch mean reduction
        pytorch_mean = F.cross_entropy(
            logits_ref.view(-1, logits_ref.size(-1)),
            y.view(-1),
            ignore_index=-100,
            reduction="mean",
        ).item()
    
    # IMPORTANT: `train_batch` is a fresh batch fetched after the training step.
    # `last_ce_loss` (when provided) comes from the *previous* training step and
    # generally corresponds to a different batch. Comparing eval CE on this batch
    # against `last_ce_loss` will frequently (and falsely) fail.
    #
    # So the sanity check validates *internal CE consistency on the same batch*:
    #   eval path CE  ≈  PyTorch mean CE  ≈  manual mean CE
    delta_eval_vs_pytorch = abs(eval_loss_val - pytorch_mean)
    delta_eval_vs_manual = abs(eval_loss_val - manual_mean)
    delta_ema = abs(eval_loss_val - current_ema)
    
    result = {
        "eval_on_train_batch": eval_loss_val,
        "manual_mean": manual_mean,
        "pytorch_mean": pytorch_mean,
        "current_train_loss": current_train_loss,
        "last_step_ce_loss": (float(last_ce_loss) if last_ce_loss is not None else None),
        "current_ema": current_ema,
        "n_valid_tokens": n_valid,
        "n_total_tokens": n_total,
        "delta_eval_vs_pytorch": delta_eval_vs_pytorch,
        "delta_eval_vs_manual": delta_eval_vs_manual,
        "delta_vs_ema": delta_ema,
        "sane": (delta_eval_vs_pytorch < 1e-3) and (delta_eval_vs_manual < 1e-3),
    }
    
    last_step_line = (
        f"  last_step_ce_loss   = {float(last_ce_loss):.4f}\n" if last_ce_loss is not None else ""
    )
    logger.info(
        f"\n{'='*70}\n"
        f"[EVAL SANITY CHECK] Eval codepath on TRAINING batch (CE-to-CE):\n"
        f"  eval_on_train_batch = {eval_loss_val:.4f}\n"
        f"  manual_mean (ref)   = {manual_mean:.4f}\n"
        f"  pytorch_mean        = {pytorch_mean:.4f}\n"
        f"{last_step_line}"
        f"  current_total_loss  = {current_train_loss:.4f}\n"
        f"  current_EMA         = {current_ema:.4f}\n"
        f"  valid_tokens        = {n_valid}/{n_total} ({100*n_valid/max(1,n_total):.1f}%)\n"
        f"  delta_eval_vs_ref   = {delta_eval_vs_pytorch:.6f} {'✓' if delta_eval_vs_pytorch < 1e-3 else '⚠️ HIGH'}\n"
        f"  delta_vs_ema        = {delta_ema:.4f} {'✓' if delta_ema < 0.5 else '⚠️ HIGH'}\n"
        f"{'='*70}"
    )
    
    if not result["sane"]:
        logger.warning(
            "⚠️  EVAL SANITY CHECK FAILED: eval CE mismatch on the same batch.\n"
            "   This suggests a bug in the eval codepath (shift, mask, or normalization mismatch)."
        )
    
    return result


def maybe_run_fixed_eval(
    *,
    step: int,
    eval_interval: int,
    model,
    fixed_eval_batches: list[dict],
    device: str,
    dtype: torch.dtype,
    config: TrainingConfig,
    use_mod_mor: bool,
    logger: Any,
    train_loss: float,
    eval_debug: bool = False,
) -> Optional[float]:
    if eval_interval <= 0 or step % eval_interval != 0:
        return None

    base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    base_model.eval()
    eval_loss = evaluate_fixed_batches(
        base_model=base_model,
        fixed_eval_batches=fixed_eval_batches,
        device=device,
        dtype=dtype,
        config=config,
        use_mod_mor=use_mod_mor,
        eval_debug=eval_debug,
    )
    base_model.train()

    logger.info(f"[EVAL] step={step}  eval_loss={eval_loss:.4f}  train_loss={train_loss:.4f}")
    return eval_loss
