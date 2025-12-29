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
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Compute training loss for one micro-batch.

    Returns:
      loss, ce_loss, logits, aux_loss, ponder_loss, advantage_loss

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

        base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        if logits is not None and hasattr(base_model, "compute_advantage_loss"):
            advantage_loss_t = base_model.compute_advantage_loss(logits, y, ignore_index=-100)

        if hasattr(aux_loss, "clamp"):
            aux_loss = aux_loss.clamp(max=100.0)
        if hasattr(ponder_loss, "clamp"):
            ponder_loss = ponder_loss.clamp(max=100.0)
        if hasattr(advantage_loss_t, "clamp"):
            advantage_loss_t = advantage_loss_t.clamp(max=10.0)

        aux_loss_t = aux_loss if isinstance(aux_loss, torch.Tensor) else torch.tensor(aux_loss, device=device)
        ponder_loss_t = ponder_loss if isinstance(ponder_loss, torch.Tensor) else torch.tensor(ponder_loss, device=device)
        advantage_loss_t = (
            advantage_loss_t
            if isinstance(advantage_loss_t, torch.Tensor)
            else (torch.tensor(advantage_loss_t, device=device) if advantage_loss_t is not None else None)
        )

        # aux_loss_t is already scaled by aux_loss_weight in the model/router.
        # ponder_loss_t is unscaled (raw) from the router, so we scale it here.
        loss = ce_loss + aux_loss_t + config.ponder_scale * ponder_loss_t
        if advantage_loss_t is not None:
            loss = loss + advantage_loss_t

        if track_loss_scalars:
            # Keep tensors; caller can store .detach() for later logging.
            pass

        return loss, ce_loss, logits, aux_loss_t, ponder_loss_t, advantage_loss_t

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

    return loss, loss, logits, None, None, None


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
) -> dict:
    """Run eval codepath on a training batch to verify consistency.
    
    This sanity check ensures the eval loss computation matches training.
    Expected: eval_on_train_batch ≈ current_train_loss (within ~0.5).
    
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
            if config.use_chunked_ce and hasattr(base_model, "forward_hidden"):
                hidden = base_model.forward_hidden(x, mask)
                weight = base_model.output.weight
                eval_loss = fused_chunked_cross_entropy(
                    hidden, weight, y,
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
    
    # Compute deltas
    delta_train = abs(eval_loss_val - current_train_loss)
    delta_ema = abs(eval_loss_val - current_ema)
    
    result = {
        "eval_on_train_batch": eval_loss_val,
        "manual_mean": manual_mean,
        "pytorch_mean": pytorch_mean,
        "current_train_loss": current_train_loss,
        "current_ema": current_ema,
        "n_valid_tokens": n_valid,
        "n_total_tokens": n_total,
        "delta_vs_train": delta_train,
        "delta_vs_ema": delta_ema,
        "sane": delta_train < 0.5 and delta_ema < 0.5,
    }
    
    logger.info(
        f"\n{'='*70}\n"
        f"[EVAL SANITY CHECK] Eval codepath on TRAINING batch:\n"
        f"  eval_on_train_batch = {eval_loss_val:.4f}\n"
        f"  manual_mean (ref)   = {manual_mean:.4f}\n"
        f"  pytorch_mean        = {pytorch_mean:.4f}\n"
        f"  current_train_loss  = {current_train_loss:.4f}\n"
        f"  current_EMA         = {current_ema:.4f}\n"
        f"  valid_tokens        = {n_valid}/{n_total} ({100*n_valid/max(1,n_total):.1f}%)\n"
        f"  delta_vs_train      = {delta_train:.4f} {'✓' if delta_train < 0.5 else '⚠️ HIGH'}\n"
        f"  delta_vs_ema        = {delta_ema:.4f} {'✓' if delta_ema < 0.5 else '⚠️ HIGH'}\n"
        f"{'='*70}"
    )
    
    if not result["sane"]:
        logger.warning(
            "⚠️  EVAL SANITY CHECK FAILED: eval_on_train_batch differs significantly from train_loss.\n"
            "   This indicates a bug in the eval codepath (shift, mask, or normalization mismatch)."
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
