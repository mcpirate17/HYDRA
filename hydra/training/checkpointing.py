from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import torch

from .config import TrainingConfig
from .metrics import TrainingMetrics


def _capture_rng_state() -> dict:
    """Best-effort capture of RNG state for deterministic resume."""
    import random

    state: dict = {"python": random.getstate(), "torch": torch.random.get_rng_state()}
    if torch.cuda.is_available():
        try:
            state["cuda"] = torch.cuda.random.get_rng_state_all()
        except Exception:
            pass
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except Exception:
        pass
    return state


def load_diagnostics(
    *,
    checkpoint_dir: str,
    run_id: Optional[str] = None,
    logger=None,
) -> List[dict]:
    """Load existing diagnostics from a run-specific JSON file for resuming.
    
    Returns empty list if file doesn't exist or can't be loaded.
    """
    ckpt_dir = Path(checkpoint_dir)
    
    if run_id:
        diag_path = ckpt_dir / f"diagnostics_{run_id}.json"
    else:
        diag_path = ckpt_dir / "training_diagnostics.json"
    
    if not diag_path.exists():
        return []
    
    try:
        with open(diag_path, "r") as f:
            data = json.load(f)
        if logger:
            logger.info(f"📊 Loaded {len(data)} existing diagnostic records from {diag_path}")
        return data if isinstance(data, list) else []
    except Exception as e:
        if logger:
            logger.warning(f"⚠️  Failed to load existing diagnostics: {e}")
        return []


def save_diagnostics(
    *,
    checkpoint_dir: str,
    diagnostics_data: list[dict],
    logger,
    run_id: Optional[str] = None,
) -> None:
    """Save diagnostics to a run-specific JSON file.
    
    Files are named: diagnostics_{run_id}.json
    If run_id is None, falls back to generic training_diagnostics.json
    """
    if not diagnostics_data:
        return
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # Use run-specific filename if run_id provided
    if run_id:
        diag_path = ckpt_dir / f"diagnostics_{run_id}.json"
    else:
        diag_path = ckpt_dir / "training_diagnostics.json"
    
    try:
        with open(diag_path, "w") as f:
            json.dump(diagnostics_data, f, indent=2)
        logger.debug(f"📊 Diagnostics saved to {diag_path}")
    except Exception as e:
        logger.warning(f"⚠️  Failed to save diagnostics: {e}")


def save_checkpoint(
    *,
    step: int,
    config: TrainingConfig,
    model: torch.nn.Module,
    optimizer,
    scaler,
    metrics: TrainingMetrics,
    checkpoint_history: list[Path],
    adaptive_lr_manager=None,
    final: bool = False,
    best: bool = False,
    logger,
    trainer_state: Optional[dict] = None,
) -> Path:
    ckpt_dir = Path(config.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if best:
        suffix = "best"
    elif final:
        suffix = "final"
    else:
        suffix = f"step_{step}"

    # Use model_size from config (e.g., "debug", "50M", "100M")
    model_size = getattr(config, "model_size", "100M").lower().replace(".", "_")
    ckpt_path = ckpt_dir / f"hydra_{model_size}_{suffix}.pt"
    if (not best) and (not final) and ckpt_path.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        ckpt_path = ckpt_dir / f"hydra_{model_size}_{suffix}_{ts}.pt"

    if hasattr(model, "_orig_mod"):
        model = model._orig_mod

    checkpoint: dict = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "config": asdict(config),
        "rng_state": _capture_rng_state(),
        "metrics": {
            "initial_loss": metrics.initial_loss,
            "current_loss": metrics.losses[-1] if metrics.losses else 0,
            "best_loss": metrics.best_loss,
            "best_loss_step": metrics.best_loss_step,
            "total_tokens": metrics.total_tokens,
            "ema_loss": metrics.ema_loss,
        },
    }

    if trainer_state is not None:
        checkpoint["trainer_state"] = trainer_state

    if adaptive_lr_manager is not None:
        checkpoint["adaptive_lr_state"] = adaptive_lr_manager.get_state()
        if hasattr(adaptive_lr_manager, "_swa_model") and adaptive_lr_manager._swa_model is not None:
            checkpoint["swa_model"] = adaptive_lr_manager._swa_model.state_dict()

    # Write atomically to avoid producing a truncated/invalid checkpoint if the
    # process is interrupted mid-save.
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            delete=False,
            dir=str(ckpt_dir),
            prefix=ckpt_path.name + ".tmp.",
        ) as f:
            tmp_path = Path(f.name)
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, ckpt_path)
    finally:
        if tmp_path is not None and tmp_path.exists() and tmp_path != ckpt_path:
            try:
                tmp_path.unlink()
            except Exception:
                pass

    if best:
        logger.info(f"🏆 New best! Loss: {metrics.best_loss:.4f} → {ckpt_path}")
    else:
        logger.info(f"Checkpoint saved: {ckpt_path}")

    if (not best) and (not final):
        checkpoint_history.append(ckpt_path)
        _cleanup_old_checkpoints(
            checkpoint_history=checkpoint_history,
            max_checkpoints=config.max_checkpoints,
            logger=logger,
        )

    return ckpt_path


def _cleanup_old_checkpoints(*, checkpoint_history: list[Path], max_checkpoints: int, logger) -> None:
    while len(checkpoint_history) > max_checkpoints:
        old_ckpt = checkpoint_history.pop(0)
        if old_ckpt.exists():
            old_ckpt.unlink()
            logger.info(f"   Removed old checkpoint: {old_ckpt.name}")


def maybe_save_best_checkpoint(
    *,
    step: int,
    prev_best_loss: float,
    best_loss: float,
    save_checkpoint_fn,
) -> bool:
    """Implements the periodic best-checkpoint heuristic used during training."""
    if step < 1000 or step % 500 != 0:
        return False

    improvement = (prev_best_loss - best_loss) / prev_best_loss if prev_best_loss < float("inf") else 0.0
    if improvement > 0.20:
        save_checkpoint_fn(step, best=True)
        return True
    if step % 1000 == 0 and best_loss < prev_best_loss:
        save_checkpoint_fn(step, best=True)
        return True
    return False
