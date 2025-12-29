from __future__ import annotations

import math
import os
import time
from collections import deque
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import json

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
import torch.profiler

from hydra.model.framework import HydraModel
from hydra.data.universal_data_loader import create_universal_loader
from hydra.data.data_filter import BatchFilter, FilterConfig

from .config import TrainingConfig
from .metrics import TrainingMetrics
from .lr import get_lr, ProgressAwareLRManager
from .lr_step import compute_step_lr
from .runtime import configure_runtime
from hydra.logging import HydraLogger

from . import checkpointing as _checkpointing
from . import reporting as _reporting
from .loop import (
    compute_microbatch_loss,
    eval_sanity_check_on_train_batch,
    maybe_run_fixed_eval,
    resolve_micro_diag_tensors,
    update_scalar_ema,
)
from .gradients import skip_update_for_nonfinite_gradients
from . import spike_diagnostics as _spike
from . import step_diagnostics as _step_diag

_RUNTIME_STATUS = configure_runtime()
def _load_seq_len_policy_from_env() -> dict | None:
    path = os.environ.get("HYDRA_SEQ_LEN_POLICY_JSON", "").strip()
    if not path:
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        warnings.warn(f"Failed to load HYDRA_SEQ_LEN_POLICY_JSON='{path}': {e}")
        return None


def _policy_for_seq_len(policy: dict, seq_len: int) -> dict:
    by_seq = policy.get("by_seq_len", {}) if isinstance(policy, dict) else {}
    if isinstance(by_seq, dict):
        exact = by_seq.get(str(seq_len))
        if isinstance(exact, dict):
            return exact
    default = policy.get("default", {}) if isinstance(policy, dict) else {}
    return default if isinstance(default, dict) else {}


def _auto_policy_for_seq_len(seq_len: int) -> dict:
    """Built-in heuristic policy keyed on current sequence length.

    This is intentionally conservative (safe) and only touches runtime knobs.
    It does NOT attempt to change architecture (e.g. LA3 vs CCQA) mid-run.
    """

    L = int(seq_len)

    # Short context: prioritize throughput (checkpointing is overhead; chunked CE can be slower).
    if L <= 512:
        return {
            "use_chunked_ce": False,
            "gradient_checkpointing": True,
            "checkpoint_every_n": 4,
        }

    # Mid context: balanced
    if L <= 1024:
        return {
            "use_chunked_ce": True,
            "gradient_checkpointing": True,
            "checkpoint_every_n": 2,
        }

    # Long context: prioritize memory
    return {
        "use_chunked_ce": True,
        "gradient_checkpointing": True,
        "checkpoint_every_n": 1,
    }


class Trainer:
    """Optimized trainer with all performance best practices."""

    __slots__ = (
        "config",
        "device",
        "logger",
        "model",
        "optimizer",
        "scaler",
        "train_loader",
        "metrics",
        "dtype",
        "_cached_get_lr",
        "_param_groups",
        "_log_interval",
        "_tokens_per_step",
        "_use_scaler",
        "_checkpoint_history",
        "_checkpoint_losses",
        "_early_stop_counter",
        "_current_seq_len",
        "_use_mod_mor",
        "_start_step",
        "_mor_enable_step",
        "_mod_triggered",
        "_mor_triggered_by_loss",
        "_diagnostics_file",
        "_diagnostics_data",
        "_last_ce_loss",
        "_last_aux_loss",
        "_last_ponder_loss",
        "_last_advantage_loss",
        "_last_pre_clip_norm",
        "_ce_ema",
        "_adaptive_lr",
        "_use_progress_aware_lr",
        "_batch_filter",
        "_checkpoint_seq_len",
        "_checkpoint_config",
        "_checkpoint_lr",
        "_resume_lr_scale",
        "_resume_lr_override_target",
        "_kernel_status",
        "_checkpoint_adaptive_state",
        "_seed_set",
        "_seq_len_policy",
        "_skip_lr_schedule",
        "_embed_lr_scale",
        "_resume_rewarmup_steps",
        "_resume_rewarmup_start_step",
        "_resume_state_incomplete",
        "_prev_run_best_loss",
        "_resume_ce_ema",
    )

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.logger = HydraLogger(config)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.metrics = TrainingMetrics()

        # Quality-of-life: if the user sets a stepped schedule but forgets to
        # bump max_seq_len, make it consistent with the largest phase length.
        # The Trainer drives effective seq_len via dataloader batch shapes; this
        # just ensures the config is self-consistent.
        if config.seq_steps:
            try:
                max_phase_len = max(int(s) for _, s in config.seq_steps)
                if int(getattr(config, "max_seq_len", 0)) < max_phase_len:
                    self.logger.info(
                        f"Adjusting max_seq_len={config.max_seq_len} -> {max_phase_len} to match seq_steps"
                    )
                    config.max_seq_len = max_phase_len
            except Exception:
                pass

        self._seq_len_policy = _load_seq_len_policy_from_env()
        if self._seq_len_policy is None:
            # Optional built-in auto-tune (no external policy file).
            # Opt-in to avoid surprising config changes.
            env_auto = os.environ.get("HYDRA_SEQ_LEN_AUTO_TUNE", "").strip().lower()
            if env_auto in ("1", "true", "yes", "on"):
                self._seq_len_policy = {"version": 1, "default": {}, "by_seq_len": {}}
                self._seq_len_policy["_auto"] = True

        if self._seq_len_policy:
            try:
                initial_seq_len = config.seq_steps[0][1] if config.seq_steps else config.max_seq_len
                patch = _policy_for_seq_len(self._seq_len_policy, int(initial_seq_len))
                if not patch and bool(self._seq_len_policy.get("_auto", False)):
                    patch = _auto_policy_for_seq_len(int(initial_seq_len))
                if isinstance(patch, dict) and patch:
                    if "use_chunked_ce" in patch:
                        config.use_chunked_ce = bool(patch["use_chunked_ce"])
                    if "chunked_ce_size" in patch:
                        config.chunked_ce_size = int(patch["chunked_ce_size"])
                    if "gradient_checkpointing" in patch:
                        config.gradient_checkpointing = bool(patch["gradient_checkpointing"])
                    if "checkpoint_every_n" in patch:
                        setattr(config, "checkpoint_every_n", int(patch["checkpoint_every_n"]))
            except Exception:
                # Keep startup robust; policy application is best-effort.
                pass

        self._seed_set = False
        if config.seed is not None:
            import random
            import numpy as np

            random.seed(config.seed)
            np.random.seed(config.seed)
            torch.manual_seed(config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(config.seed)
            self._seed_set = True

        self._log_interval = config.log_interval
        self._tokens_per_step = config.tokens_per_step

        self._checkpoint_history: List[Path] = []
        self._checkpoint_losses: List[float] = []
        self._early_stop_counter: int = 0
        self._start_step: int = 0

        self._diagnostics_file: Optional[Path] = None
        self._diagnostics_data: List[dict] = []
        self._last_ce_loss: float = 0.0
        self._last_aux_loss: float = 0.0
        self._last_ponder_loss: float = 0.0
        self._last_advantage_loss: float = 0.0
        self._last_pre_clip_norm: float = 0.0  # Raw gradient norm before clipping

        self._adaptive_lr: Optional[ProgressAwareLRManager] = None
        self._use_progress_aware_lr = config.adaptive_lr or config.use_swa or config.lr_schedule == "wsd_adaptive"

        # If set during resume, this requests a one-time LR override at resume
        # which is converted into a scale factor once the adaptive LR manager
        # has been initialized and can provide the true base LR.
        self._resume_lr_override_target: float = 0.0

        self._batch_filter: Optional[BatchFilter] = None
        if config.batch_filter:
            filter_cfg = FilterConfig(
                loss_spike_threshold=config.batch_filter_threshold,
                max_skips_per_epoch=config.batch_filter_max_skip,
            )
            self._batch_filter = BatchFilter(filter_cfg)

        self.dtype = getattr(torch, config.dtype) if config.dtype != "float32" else torch.float32

        self._kernel_status = None
        try:
            from hydra.kernels import set_use_triton_kernels, get_kernel_status

            set_use_triton_kernels(bool(getattr(config, "use_triton_kernels", False)))
            self._kernel_status = get_kernel_status()
        except Exception as e:
            self.logger.warning(f"Failed to configure Triton kernels ({e})")

        self._checkpoint_seq_len = None
        self._checkpoint_config = {}
        self._checkpoint_lr = None
        self._resume_lr_scale: float = 1.0
        self._resume_rewarmup_steps: int = 0
        self._resume_rewarmup_start_step: int = 0
        self._resume_state_incomplete: bool = False
        self._prev_run_best_loss: float = float("inf")
        self._resume_ce_ema: float = 0.0
        if config.resume_from:
            self._checkpoint_config = self._peek_checkpoint_config(config.resume_from)
            self._checkpoint_seq_len = self._checkpoint_config.get("max_seq_len", config.max_seq_len)
            if "architecture" in self._checkpoint_config:
                ckpt_arch = self._checkpoint_config["architecture"]
                if ckpt_arch != config.architecture:
                    self.logger.warning(f"Overriding architecture: {config.architecture} -> {ckpt_arch} (from checkpoint)")
                    config.architecture = ckpt_arch
            if "mod_mor_dim" in self._checkpoint_config:
                config.mod_mor_dim = self._checkpoint_config["mod_mor_dim"]
            if "n_mor_blocks" in self._checkpoint_config:
                config.n_mor_blocks = self._checkpoint_config["n_mor_blocks"]
            if "mor_recursions" in self._checkpoint_config:
                config.mor_recursions = self._checkpoint_config["mor_recursions"]
            if "mod_mor_n_heads" in self._checkpoint_config:
                config.mod_mor_n_heads = self._checkpoint_config["mod_mor_n_heads"]
            if "mod_mor_n_kv_heads" in self._checkpoint_config:
                config.mod_mor_n_kv_heads = self._checkpoint_config["mod_mor_n_kv_heads"]
            # NOTE: attention_backend is NOT restored from checkpoint - use config/CLI value
            # This allows switching backends on resume (e.g., la3 -> ccgqa)
            # Preserve run_id from checkpoint for continuity of diagnostics/logs
            if "run_id" in self._checkpoint_config:
                config.run_id = self._checkpoint_config["run_id"]
                self.logger.info(f"Resuming run: {config.run_id}")
                # Load existing diagnostics for this run to append to
                existing_diag = _checkpointing.load_diagnostics(
                    checkpoint_dir=config.checkpoint_dir,
                    run_id=config.run_id,
                    logger=self.logger,
                )
                if existing_diag:
                    self._diagnostics_data = existing_diag
            if "optimizer" in torch.load(config.resume_from, weights_only=False, map_location="cpu"):
                ckpt_full = torch.load(config.resume_from, weights_only=False, map_location="cpu")
                if "optimizer" in ckpt_full and "param_groups" in ckpt_full["optimizer"]:
                    last_lr = ckpt_full["optimizer"]["param_groups"][0].get("lr", config.max_lr)
                    self.logger.info(f"Checkpoint final LR: {last_lr:.6f}")
                    self._checkpoint_lr = last_lr

        self._setup_model()
        self._setup_optimizer()
        self._setup_data()

        if config.resume_from:
            self._load_checkpoint(config.resume_from)

        self.logger.info(f"\n{'='*70}")
        self.logger.info("HYDRA 100M Optimized Trainer")
        self.logger.info(f"{'='*70}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Model: {self._count_params()/1e6:.1f}M parameters")
        self.logger.info(f"Batch: {config.batch_size} micro × {config.grad_accum_steps} accum = {config.effective_batch_size} effective")
        self.logger.info(f"Sequence length: {config.max_seq_len}")
        self.logger.info(f"Tokens/step: {self._tokens_per_step:,} ({self._tokens_per_step/1e6:.2f}M per optimizer step)")
        self.logger.info(f"Dataset: {config.dataset_name}")
        self.logger.info(f"torch.compile: {config.use_compile} (mode={config.compile_mode})")
        if self._kernel_status is not None:
            ks = self._kernel_status
            triton_enabled = ks.get("use_triton_kernels", False)
            self.logger.info(f"Triton kernels: {triton_enabled}" + (f" (v{ks.get('triton_version', 'N/A')})" if triton_enabled else ""))
            if triton_enabled:
                self.logger.info(f"  ├─ fused_swiglu:  {ks.get('fused_swiglu', False)}")
                self.logger.info(f"  ├─ fused_qk_norm: {ks.get('fused_qk_norm', False)}")
                self.logger.info(f"  ├─ fused_rope:    {ks.get('fused_rope', False)}" + ("" if ks.get('fused_rope', False) else " (opt-in: HYDRA_ENABLE_FUSED_ROPE=1)"))
                self.logger.info(f"  └─ fused_rms_norm:{ks.get('fused_rms_norm', False)}" + ("" if ks.get('fused_rms_norm', False) else " (opt-in: HYDRA_ENABLE_FUSED_RMS_NORM=1)"))
        _chunked_ce_active = config.use_chunked_ce and hasattr(self.model, "forward_hidden")
        self.logger.info(f"Chunked CE: {_chunked_ce_active}" + (" (model supports forward_hidden)" if _chunked_ce_active else f" (disabled: use_chunked_ce={config.use_chunked_ce}, has forward_hidden={hasattr(self.model, 'forward_hidden')})"))
        self.logger.info(f"AMP dtype: {config.dtype}")
        if self._seed_set:
            self.logger.info(f"Seed: {config.seed} (reproducible training enabled)")

        if self._skip_lr_schedule:
            self.logger.info("LR Schedule: Disabled (Adafactor uses internal 1/\u221at schedule)")
        elif config.lr_schedule in ("wsd", "wsd_adaptive"):
            stable_steps = config.decay_start_step - config.warmup_steps
            self.logger.info("LR Schedule: WSD (Warmup-Stable-Decay)")
            self.logger.info(f"  Warmup: {config.warmup_steps} steps ({config.warmup_steps/config.max_steps*100:.1f}%)")
            self.logger.info(f"  Stable: {stable_steps} steps ({stable_steps/config.max_steps*100:.1f}%) at LR={config.max_lr}")
            self.logger.info(f"  Decay:  {config.decay_steps} steps ({config.decay_steps/config.max_steps*100:.1f}%) -> LR={config.min_lr}")
            if config.adaptive_lr:
                self.logger.info(f"  Adaptive: ENABLED (patience={config.adaptive_patience}, threshold={config.adaptive_threshold:.0%})")
            if config.use_swa:
                self.logger.info(f"  SWA: ENABLED (starts at {config.swa_start_pct:.0%} of training)")
        else:
            self.logger.info("LR Schedule: Cosine with warmup")
            self.logger.info(f"  Warmup: {config.warmup_steps} steps, Max LR: {config.max_lr}, Min LR: {config.min_lr}")
        if config.batch_filter:
            self.logger.info(f"Batch Filter: ENABLED (threshold={config.batch_filter_threshold}x, max_skip={config.batch_filter_max_skip:.0%})")
        self.logger.info(f"{'='*70}\n")

    def _peek_checkpoint_config(self, checkpoint_path: str) -> dict:
        try:
            checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
        except Exception as e:
            size_mb = None
            try:
                size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
            except Exception:
                pass
            hint = (
                f"Failed to read checkpoint '{checkpoint_path}'"
                + (f" (size={size_mb:.1f} MiB)" if size_mb is not None else "")
                + ". The file is likely truncated/corrupt (common if a run was interrupted during saving).\n"
                + "Try resuming from the previous step checkpoint (e.g. step_71000/70500) or from *_final.pt."
            )
            raise RuntimeError(hint) from e
        ckpt_config = checkpoint.get("config", {})
        result = {}
        for key, tensor in checkpoint["model"].items():
            if "cos_cached" in key:
                result["max_seq_len"] = tensor.shape[2]
                self.logger.info(f"Checkpoint RoPE cache seq_len: {result['max_seq_len']}")
                break
        arch_keys = [
            "architecture",
            "mod_mor_dim",
            "n_mor_blocks",
            "mor_recursions",
            "mod_mor_n_heads",
            "mod_mor_n_kv_heads",
            "mod_capacity",
            "vocab_size",
            "mor_adaptive",
        ]
        for key in arch_keys:
            if key in ckpt_config:
                result[key] = ckpt_config[key]
        if result:
            self.logger.info(
                f"Checkpoint architecture: dim={result.get('mod_mor_dim')}, "
                f"blocks={result.get('n_mor_blocks')}, "
                f"recursions={result.get('mor_recursions')}, "
                f"heads={result.get('mod_mor_n_heads')}"
            )
        return result

    def _peek_checkpoint_seq_len(self, checkpoint_path: str) -> int:
        try:
            checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
        except Exception:
            return self.config.max_seq_len
        for key, tensor in checkpoint["model"].items():
            if "cos_cached" in key:
                seq_len = tensor.shape[2]
                self.logger.info(f"Checkpoint RoPE cache seq_len: {seq_len}")
                return seq_len
        return self.config.max_seq_len

    def _setup_model(self) -> None:
        config = self.config
        max_needed_seq_len = config.max_seq_len
        for _, seq_len in config.seq_steps:
            max_needed_seq_len = max(max_needed_seq_len, seq_len)
        model_seq_len = max_needed_seq_len
        self.logger.info(f"Creating model with max_seq_len={model_seq_len} (covers all training phases)")
        if config.architecture != "mod_mor":
            raise ValueError(f"Unsupported architecture '{config.architecture}'. Only 'mod_mor' is supported.")
        
        # MoR warmup defaults to 30% of training (ponder loss ramps up to this point)
        mor_warmup = int(config.max_steps * 0.30)
        
        # MoD is now MoR-informed: starts disabled, triggers when MoR early_exit crosses threshold
        # We set a very high warmup initially so MoD stays disabled until we trigger it dynamically
        mod_mlp_warmup = config.max_steps + 1  # MoD disabled initially
        mod_force_enable_step = config.max_steps + 1  # No step-based forcing
        self._mod_triggered = False  # Track if MoD has been dynamically enabled
        
        # HydraModel uses CCGQA attention (Compressed Convolutional GQA)
        attention_backend = "ccgqa"  # Only CCGQA is supported
        self.model = HydraModel(
            vocab_size=config.vocab_size,
            dim=config.mod_mor_dim,
            n_mor_blocks=config.n_mor_blocks,
            recursions_per_block=config.mor_recursions,
            n_heads=config.mod_mor_n_heads,
            n_kv_heads=config.mod_mor_n_kv_heads,
            compression_factor=4,
            mlp_ratio=3.6,
            max_seq_len=model_seq_len,
            mod_capacity=config.mod_capacity,
            aux_loss_weight=config.aux_scale,
            mod_loss_aware_weight=getattr(config, "mod_loss_aware_weight", 0.0),
            adaptive=config.mor_adaptive,
            tie_weights=True,
            mod_mlp_warmup=mod_mlp_warmup,
            mod_enable_loss_threshold=getattr(config, "mod_enable_loss_threshold", None),
            mod_force_enable_step=mod_force_enable_step,
            mor_warmup=mor_warmup,
            mor_advantage_loss_scale=getattr(config, "mor_advantage_loss_scale", 0.1),
            attention_backend=attention_backend,
        ).to(self.device)
        self._use_mod_mor = True
        mod_status = "OFF (capacity=1.0)" if config.mod_capacity >= 1.0 else f"{config.mod_capacity:.0%} capacity (MoR-informed)"
        mor_status = "adaptive" if config.mor_adaptive else "fixed-depth (no routing)"
        self.logger.info(f"MoD: {mod_status}")
        self.logger.info(f"MoR: {mor_status}, {config.mor_recursions} recursions/block")
        if config.mor_adaptive:
            # Apply floor: MoR needs minimum steps to stabilize regardless of pct
            mor_enable_min = getattr(config, "mor_enable_min_steps", 3000)
            mor_enable_step = max(
                mor_enable_min,
                int(config.max_steps * config.mor_enable_pct)
            )
            if config.mor_already_enabled:
                mor_enable_step = 0
                self.logger.info("MoR RESTART MODE: Adaptive routing enabled from start (resumed after enable point)")
            remaining_steps = config.max_steps - mor_enable_step
            default_rampup = min(config.mor_rampup_steps, 2 * mor_enable_step)
            actual_rampup = min(default_rampup, remaining_steps)
            actual_rampup = max(actual_rampup, min(100, int(config.max_steps * 0.1)))
            self.model.set_mor_curriculum(enable_step=mor_enable_step, rampup_steps=actual_rampup)
            self._mor_enable_step = mor_enable_step
            mor_loss_thr = getattr(config, "mor_enable_loss_threshold", 0.0) or 0.0
            if mor_enable_step > 0:
                if mor_loss_thr > 0:
                    self.logger.info(f"MoR CURRICULUM: Fixed-depth until step {mor_enable_step:,} (min={mor_enable_min}, pct={config.mor_enable_pct:.0%}) OR CE_EMA < {mor_loss_thr:.1f}, then {actual_rampup:,} step rampup")
                else:
                    self.logger.info(f"MoR CURRICULUM: Fixed-depth until step {mor_enable_step:,} (min={mor_enable_min}, pct={config.mor_enable_pct:.0%}), then {actual_rampup:,} step rampup")
        else:
            self._mor_enable_step = 0
            self.logger.info("MoR CURRICULUM: Disabled (adaptive=False, running pure fixed-depth)")
        if config.gradient_checkpointing:
            if hasattr(self.model, "enable_gradient_checkpointing"):
                every_n = getattr(config, "checkpoint_every_n", 1)
                self.model.enable_gradient_checkpointing(every_n=every_n)
                if every_n == 1:
                    self.logger.info("Gradient checkpointing: ENABLED (all layers, ~50% memory, ~30% overhead)")
                else:
                    self.logger.info(f"Gradient checkpointing: ENABLED (every {every_n} layers, ~35% memory, ~15% overhead)")
            else:
                self.logger.warning("WARNING: Model doesn't support gradient checkpointing")
        if config.use_compile and self.device == "cuda":
            mode = config.compile_mode
            # if mode == "max-autotune":
            #     mode = "max-autotune-no-cudagraphs"
            self.logger.info(f"Compiling model with mode='{mode}'...")
            self.model = torch.compile(self.model, mode=mode, fullgraph=False, dynamic=False)
            self.logger.info("Model compiled successfully!")

    def _setup_optimizer(self) -> None:
        config = self.config
        decay_params = []
        no_decay_params = []
        embed_params = []  # Separate group for embeddings - they need lower LR
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if "embed" in name or "tok_emb" in name:
                    embed_params.append(param)
                elif "weight" in name and "norm" not in name:
                    decay_params.append(param)
                else:
                    no_decay_params.append(param)
        
        # Embedding LR scale: 0.1x base LR to prevent gradient spikes
        embed_lr_scale = 0.1
        embed_lr = config.max_lr * embed_lr_scale
        
        self._skip_lr_schedule = False
        
        if config.use_adafactor:
            # PyTorch Adafactor: lr is the MAX for relative step size ρ_t, not a direct LR
            # Paper and PyTorch default: lr=0.01 (used as: ρ_t = min(lr, 1/√t))
            # The optimizer internally adapts step sizes - no manual scaling needed
            # Reference: https://docs.pytorch.org/docs/stable/generated/torch.optim.Adafactor.html
            adafactor_lr = 0.01  # Paper/PyTorch default, acts as ceiling for ρ_t
            adafactor_embed_lr = adafactor_lr * embed_lr_scale  # Lower for embeddings
            
            self.optimizer = torch.optim.Adafactor(
                [
                    {"params": decay_params, "weight_decay": config.weight_decay},
                    {"params": no_decay_params, "weight_decay": 0.0},
                    {"params": embed_params, "weight_decay": 0.0, "lr": adafactor_embed_lr},
                ],
                lr=adafactor_lr,
                beta2_decay=-0.8,  # Paper default
                eps=(None, 1e-3),  # (eps1, eps2) - stabilization terms
                d=1.0,  # Clipping threshold for update/weight ratio
                weight_decay=0.0,  # Handled by param groups
                foreach=True,  # Batch ops for speed
            )
            # Skip external LR scheduling - Adafactor manages its own step sizes
            self._skip_lr_schedule = True
            self.logger.info(f"Using PyTorch Adafactor - lr={adafactor_lr:.2e} (internal 1/√t schedule, external LR schedule disabled)")
            self.logger.info(f"  Embedding LR: {adafactor_embed_lr:.2e} ({embed_lr_scale}x base)")
            self._embed_lr_scale = embed_lr_scale
        elif config.use_8bit_adam:
            import bitsandbytes as bnb

            self.optimizer = bnb.optim.AdamW8bit(
                [
                    {"params": decay_params, "weight_decay": config.weight_decay},
                    {"params": no_decay_params, "weight_decay": 0.0},
                    {"params": embed_params, "weight_decay": 0.0, "lr": embed_lr},
                ],
                lr=config.max_lr,
                betas=(0.9, 0.95),
            )
            self.logger.info("Using 8-bit AdamW (bitsandbytes) - ~75% optimizer memory savings")
            self.logger.info(f"  Embedding LR: {embed_lr:.2e} ({embed_lr_scale}x base)")
            self._embed_lr_scale = embed_lr_scale
        else:
            self.optimizer = torch.optim.AdamW(
                [
                    {"params": decay_params, "weight_decay": config.weight_decay},
                    {"params": no_decay_params, "weight_decay": 0.0},
                    {"params": embed_params, "weight_decay": 0.0, "lr": embed_lr},
                ],
                lr=config.max_lr,
                betas=(0.9, 0.95),
                fused=True,
            )
            self.logger.info(f"  Embedding LR: {embed_lr:.2e} ({embed_lr_scale}x base)")
            self._embed_lr_scale = embed_lr_scale
        self._param_groups = self.optimizer.param_groups
        use_scaler = config.dtype == "float16"
        self.scaler = GradScaler("cuda", enabled=use_scaler)
        self._use_scaler = use_scaler

    def _setup_data(self) -> None:
        config = self.config
        initial_seq_len = config.seq_steps[0][1] if config.seq_steps else config.max_seq_len
        self._current_seq_len = initial_seq_len
        self._apply_seq_len_policy(initial_seq_len)
        self.logger.info(f"Loading {config.dataset_name} dataset...")
        self.logger.info(f"Stepped sequence schedule: {config.seq_steps} + final @ {config.max_seq_len}")
        self.logger.info(f"Starting with seq_len={initial_seq_len}")
        self.train_loader = create_universal_loader(
            dataset=config.dataset_name,
            batch_size=config.batch_size,
            seq_len=initial_seq_len,
            vocab_size=config.vocab_size,
            device="cpu",
            tokenizer_name=config.tokenizer_name,
            max_steps=config.max_steps,
        )
        if hasattr(self.train_loader, "set_max_steps"):
            self.train_loader.set_max_steps(config.max_steps)
        self._tokens_per_step = config.batch_size * config.grad_accum_steps * initial_seq_len
        self.logger.info(f"Tokens per step: {self._tokens_per_step:,}")
        self.logger.info("Dataset ready!")

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"RESUMING FROM CHECKPOINT: {checkpoint_path}")
        self.logger.info(f"{'='*70}")
        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=self.device)
        model = self.model
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod

        # RoPE caches are derived buffers whose shape depends on max_seq_len.
        # When resuming with a different --seq_len (or seq schedule), the checkpoint
        # may contain cos/sin caches with incompatible shapes. Drop them and rebuild.
        try:
            state_dict = checkpoint.get("model", {})
            if isinstance(state_dict, dict):
                drop_keys = [
                    k
                    for k in state_dict.keys()
                    if ("cos_cached" in k) or ("sin_cached" in k)
                ]
                if drop_keys:
                    for k in drop_keys:
                        state_dict.pop(k, None)
                    self.logger.info(
                        f"  Dropped {len(drop_keys)} RoPE cache tensors from checkpoint state_dict (will be rebuilt)"
                    )
        except Exception:
            # Keep resume robust even if checkpoint structure differs.
            state_dict = checkpoint["model"]

        # strict=False because we intentionally exclude cache buffers.
        model.load_state_dict(state_dict, strict=False)
        optimizer_loaded = False
        scaler_loaded = False
        rng_loaded = False
        try:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            optimizer_loaded = True
        except ValueError as e:
            # Common after changing optimizer grouping or adding/removing params.
            # Keep training going: resume weights + step, but re-init optimizer state.
            self.logger.warning("  ⚠️  Optimizer state incompatible with current parameter groups; skipping optimizer resume")
            self.logger.warning(f"  Reason: {e}")
            self.logger.warning("  Continuing with freshly initialized optimizer (momentum/Adam moments reset)")
            # Avoid LR scaling based on a checkpoint optimizer LR we didn't apply.
            self._resume_lr_scale = 1.0
        try:
            self.scaler.load_state_dict(checkpoint["scaler"])
            scaler_loaded = True
        except Exception as e:
            self.logger.warning(f"  ⚠️  Failed to load GradScaler state; continuing with fresh scaler. Reason: {e}")

        # Restore RNG state (best-effort). This helps keep resume deterministic and
        # avoids subtle distribution shifts that can look like "quality" regressions.
        try:
            rng_state = checkpoint.get("rng_state")
            if isinstance(rng_state, dict):
                import random

                if "python" in rng_state:
                    random.setstate(rng_state["python"])
                try:
                    import numpy as np

                    if "numpy" in rng_state:
                        np.random.set_state(rng_state["numpy"])
                except Exception:
                    pass
                if "torch" in rng_state:
                    torch.random.set_rng_state(rng_state["torch"])
                if torch.cuda.is_available() and "cuda" in rng_state:
                    try:
                        torch.cuda.random.set_rng_state_all(rng_state["cuda"])
                    except Exception:
                        pass
                rng_loaded = True
        except Exception:
            rng_loaded = False
        ckpt_config = checkpoint.get("config", {})
        ckpt_max_steps = ckpt_config.get("max_steps", 0)
        # Resume LR behavior:
        # - default: align scheduled LR at resume step to checkpoint optimizer LR
        # - resume_ignore_ckpt_lr: skip alignment (use schedule/adaptive as-is)
        # - resume_lr_override: set LR at resume, but still allow schedule/adaptive cooldown afterward
        if float(getattr(self.config, "resume_lr_override", 0.0) or 0.0) > 0.0:
            desired_lr = float(getattr(self.config, "resume_lr_override"))
            self._resume_lr_override_target = desired_lr
            self._resume_lr_scale = 1.0
            for pg in self.optimizer.param_groups:
                pg["lr"] = desired_lr
            self.logger.info(f"  Resume LR override: lr={desired_lr:.6f} (cooldown/schedule remains active)")
        elif bool(getattr(self.config, "resume_ignore_ckpt_lr", False)):
            self._resume_lr_scale = 1.0
            self.logger.info("  Resume LR: using scheduled LR (checkpoint LR alignment disabled)")
        elif ckpt_max_steps > 0 and ckpt_max_steps != self.config.max_steps:
            self.logger.warning(f"  ⚠️  max_steps changed: {ckpt_max_steps} → {self.config.max_steps}")
            self.logger.warning("  LR schedule will be recalculated for new training length")
            self._resume_lr_scale = 1.0
        else:
            try:
                if optimizer_loaded:
                    ckpt_lr = float(self.optimizer.param_groups[0].get("lr", 0.0))
                    sched_lr_at_resume = float(get_lr(checkpoint["step"], self.config))
                    if ckpt_lr > 0.0 and sched_lr_at_resume > 0.0:
                        self._resume_lr_scale = ckpt_lr / sched_lr_at_resume
                        if abs(self._resume_lr_scale - 1.0) > 1e-6:
                            self.logger.info(
                                f"  Resume LR alignment: ckpt_lr={ckpt_lr:.6f}, "
                                f"sched_lr(step={checkpoint['step']})={sched_lr_at_resume:.6f}, "
                                f"scale={self._resume_lr_scale:.4f}"
                            )
            except Exception:
                self._resume_lr_scale = 1.0
        self._start_step = checkpoint["step"]
        if hasattr(self.train_loader, "set_step"):
            self.train_loader.set_step(self._start_step)
        ckpt_metrics = checkpoint.get("metrics", {})
        self.logger.info(f"  Loaded step: {self._start_step}")
        self._prev_run_best_loss = float(ckpt_metrics.get("best_loss", float("inf")) or float("inf"))
        self.logger.info(f"  Previous best loss (from checkpoint): {self._prev_run_best_loss if math.isfinite(self._prev_run_best_loss) else 'N/A'}")
        self.logger.info(f"  Previous total tokens: {ckpt_metrics.get('total_tokens', 'N/A'):,}")
        self.logger.info(f"  Will continue to step: {self.config.max_steps}")
        self.logger.info(f"  Remaining steps: {self.config.max_steps - self._start_step:,}")
        if ckpt_metrics:
            # Resume semantics: keep EMA continuity, but do NOT compare new losses
            # against the prior run's best.
            try:
                self.metrics.ema_loss = float(ckpt_metrics.get("ema_loss", 0.0) or 0.0)
            except Exception:
                self.metrics.ema_loss = 0.0
            self.metrics.best_loss = float("inf")
            self.metrics.best_loss_step = int(self._start_step)
            self.metrics.total_tokens = ckpt_metrics.get("total_tokens", 0)

        # Restore trainer-side EMA used for curriculum/gating (best-effort)
        try:
            extra = checkpoint.get("trainer_state", {})
            if isinstance(extra, dict) and "ce_ema" in extra:
                self._resume_ce_ema = float(extra.get("ce_ema", 0.0) or 0.0)
        except Exception:
            self._resume_ce_ema = 0.0
        self._checkpoint_adaptive_state = None
        if "adaptive_lr_state" in checkpoint:
            self._checkpoint_adaptive_state = checkpoint["adaptive_lr_state"]
            self.logger.info("  Found adaptive LR state in checkpoint (will apply after manager init)")

        # If we failed to restore key state, request a brief LR re-warmup after resume.
        self._resume_state_incomplete = not (optimizer_loaded and scaler_loaded and rng_loaded)
        self._resume_rewarmup_start_step = int(self._start_step)
        if self._resume_state_incomplete and int(self._start_step) > 0:
            rewarm_steps = int(os.environ.get("HYDRA_RESUME_REWARMUP_STEPS", "300") or 300)
            self._resume_rewarmup_steps = max(0, rewarm_steps)
            if self._resume_rewarmup_steps > 0:
                self.logger.warning(
                    f"  ⚠️  Resume state incomplete (optimizer={optimizer_loaded}, scaler={scaler_loaded}, rng={rng_loaded}); "
                    f"enabling LR re-warmup for {self._resume_rewarmup_steps} steps"
                )
        else:
            self._resume_rewarmup_steps = 0
        self.logger.info(f"{'='*70}\n")

    def _count_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def _get_seq_len_for_step(self, step: int) -> int:
        config = self.config
        cumulative_steps = 0
        for phase_steps, seq_len in config.seq_steps:
            cumulative_steps += phase_steps
            if step < cumulative_steps:
                return seq_len
        return config.max_seq_len

    def _recreate_dataloader(self, seq_len: int) -> None:
        config = self.config
        if hasattr(self, "train_loader") and self.train_loader:
            self.train_loader.close()
        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"SEQUENCE LENGTH TRANSITION: {self._current_seq_len} -> {seq_len}")
        self.logger.info(f"{'='*70}")
        model = self.model
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
        if hasattr(model, "resize_rope_cache"):
            model.resize_rope_cache(seq_len)

        self._apply_seq_len_policy(seq_len)
        self.train_loader = create_universal_loader(
            dataset=config.dataset_name,
            batch_size=config.batch_size,
            seq_len=seq_len,
            vocab_size=config.vocab_size,
            device="cpu",
            tokenizer_name=config.tokenizer_name,
            max_steps=config.max_steps,
        )
        if hasattr(self.train_loader, "set_max_steps"):
            self.train_loader.set_max_steps(config.max_steps)
        self._current_seq_len = seq_len
        self._tokens_per_step = config.batch_size * config.grad_accum_steps * seq_len
        self.logger.info(f"New tokens/step: {self._tokens_per_step:,}")
        self.logger.info(f"{'='*70}\n")

    def _apply_seq_len_policy(self, seq_len: int) -> None:
        policy = self._seq_len_policy
        if not policy:
            return

        patch = _policy_for_seq_len(policy, int(seq_len))
        if not patch and bool(policy.get("_auto", False)):
            patch = _auto_policy_for_seq_len(int(seq_len))
        if not patch:
            return

        try:
            self.logger.info(f"Seq-len policy applied (seq_len={int(seq_len)}): {patch}")
        except Exception:
            pass

        # Apply dynamic TrainingConfig knobs
        config = self.config
        if "use_chunked_ce" in patch:
            config.use_chunked_ce = bool(patch["use_chunked_ce"])
        if "chunked_ce_size" in patch:
            config.chunked_ce_size = int(patch["chunked_ce_size"])

        base_model = self.model
        if hasattr(base_model, "_orig_mod"):
            base_model = base_model._orig_mod

        if "gradient_checkpointing" in patch:
            want_gc = bool(patch["gradient_checkpointing"])
            config.gradient_checkpointing = want_gc
            if want_gc and hasattr(base_model, "enable_gradient_checkpointing"):
                every_n = int(patch.get("checkpoint_every_n", getattr(config, "checkpoint_every_n", 1)))
                setattr(config, "checkpoint_every_n", every_n)
                base_model.enable_gradient_checkpointing(every_n=every_n)
            elif (not want_gc) and hasattr(base_model, "disable_gradient_checkpointing"):
                base_model.disable_gradient_checkpointing()
        elif "checkpoint_every_n" in patch:
            every_n = int(patch["checkpoint_every_n"])
            setattr(config, "checkpoint_every_n", every_n)
            if hasattr(base_model, "enable_gradient_checkpointing") and bool(getattr(config, "gradient_checkpointing", False)):
                base_model.enable_gradient_checkpointing(every_n=every_n)

        # Apply CCQA/CCGQA fused kernel toggle if safe
        if "ccqa_use_fused_kernel" in patch:
            want_fused = bool(patch["ccqa_use_fused_kernel"])
            triton_ok = False
            if want_fused:
                try:
                    from hydra.attention.backends.ccgqa.attention import TRITON_ATTENTION_AVAILABLE

                    triton_ok = bool(TRITON_ATTENTION_AVAILABLE)
                except Exception:
                    triton_ok = False
            for m in base_model.modules():
                if hasattr(m, "use_fused_kernel"):
                    if want_fused and triton_ok:
                        setattr(m, "use_fused_kernel", True)
                    elif not want_fused:
                        setattr(m, "use_fused_kernel", False)

    def _get_batch(self) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        batch = self.train_loader.get_batch()
        mask = batch.get("attention_mask")
        if mask is not None:
            mask = mask.to(self.device, non_blocking=True)
        return (
            batch["input_ids"].to(self.device, non_blocking=True),
            batch["labels"].to(self.device, non_blocking=True),
            mask,
        )

    @torch.no_grad()
    def _compute_token_losses_from_hidden(
        self,
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

    def train(self) -> TrainingMetrics:
        config = self.config
        model = self.model
        optimizer = self.optimizer
        scaler = self.scaler
        metrics = self.metrics

        grad_accum = config.grad_accum_steps
        grad_clip = config.grad_clip
        max_steps = config.max_steps
        log_interval = self._log_interval
        eval_interval = config.eval_interval
        save_interval = config.save_interval
        tokens_per_step = self._tokens_per_step
        vocab_size = config.vocab_size
        dtype = self.dtype
        device = self.device
        param_groups = self._param_groups
        loss_scale = 1.0 / grad_accum

        start_step = self._start_step
        use_mod_mor = bool(getattr(self, "_use_mod_mor", False))
        use_scaler = bool(getattr(self, "_use_scaler", False))

        # Skip LR manager for Adafactor (uses internal scheduling)
        if self._use_progress_aware_lr and not self._skip_lr_schedule:
            self._adaptive_lr = ProgressAwareLRManager(config, start_step=start_step)
            if hasattr(self, "_checkpoint_adaptive_state") and self._checkpoint_adaptive_state:
                self._adaptive_lr.load_state(self._checkpoint_adaptive_state)
        if start_step > 0:
            self.logger.info(f"Resuming training from step {start_step}...")
        else:
            self.logger.info("Starting training...")
        self.logger.info("-" * 70)
        model.train()
        metrics.start_time = time.time()
        self._last_aux_loss = 0.0
        self._ce_ema = float(getattr(self, "_resume_ce_ema", 0.0) or 0.0) if start_step > 0 else 0.0
        eval_batches = 25
        eval_dataset = config.dataset_name
        if config.dataset_name.startswith("pretrain_") or config.dataset_name in ["sft_chat"]:
            eval_dataset = "wikitext2"
            self.logger.info(f"📊 Using wikitext2 for evaluation (mixed dataset: {config.dataset_name})")
        eval_loader = create_universal_loader(
            dataset=eval_dataset,
            batch_size=config.batch_size,
            seq_len=self._current_seq_len,
            vocab_size=config.vocab_size,
            device="cpu",
            tokenizer_name=config.tokenizer_name,
        )
        fixed_eval_batches = [eval_loader.get_batch() for _ in range(eval_batches)]
        
        # Sanity check flag: run once at first eval to verify eval codepath
        _eval_sanity_done = False
        _eval_debug = getattr(config, "eval_debug", False) or os.environ.get("HYDRA_EVAL_DEBUG", "0") == "1"

        if start_step >= max_steps:
            self.logger.warning(f"⚠️  No steps to run: start_step={start_step} >= max_steps={max_steps}")
            self.logger.warning("   Increase --max_steps to continue training from this checkpoint.")
            metrics.end_time = time.time()
            return metrics

        # Observability setup (optional)
        tb_writer = None
        if getattr(config, "use_tensorboard", False):
            try:
                from torch.utils.tensorboard import SummaryWriter  # type: ignore

                tb_writer = SummaryWriter(log_dir=getattr(config, "tensorboard_dir", "runs"))
                self.logger.info(
                    f"📈 TensorBoard enabled. Logs will be saved to {getattr(config, 'tensorboard_dir', 'runs')}"
                )
            except Exception as e:
                self.logger.warning(
                    f"⚠️  TensorBoard requested but unavailable: {type(e).__name__}: {e}"
                )
                config.use_tensorboard = False

        wandb_mod = None
        wandb_run = None
        if getattr(config, "use_wandb", False):
            try:
                import wandb as wandb_mod  # type: ignore

                wandb_run = wandb_mod.init(
                    project=getattr(config, "wandb_project", "hydra-llm"),
                    entity=getattr(config, "wandb_entity", None),
                    name=getattr(config, "run_name", None),
                    config={
                        "mode": config.mode,
                        "model_size": config.model_size,
                        "max_steps": config.max_steps,
                        "batch_size": config.batch_size,
                        "grad_accum_steps": config.grad_accum_steps,
                        "max_seq_len": config.max_seq_len,
                        "dataset_name": config.dataset_name,
                        "max_lr": config.max_lr,
                        "min_lr": config.min_lr,
                    },
                )
                self.logger.info("📡 W&B enabled.")
            except Exception as e:
                self.logger.warning(
                    f"⚠️  W&B requested but unavailable: {type(e).__name__}: {e}"
                )
                config.use_wandb = False
                wandb_mod = None
                wandb_run = None

        # Profiler setup
        profiler = None
        if config.use_profiler:
            self.logger.info(f"🔍 Profiling enabled. Traces will be saved to {config.profiler_dir}")
            profiler = torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=10, warmup=5, active=5, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(config.profiler_dir),
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            )
            profiler.start()

        # Gradient-spike threshold: scale with model size via spike_diagnostics module.
        _base_for_params = model._orig_mod if hasattr(model, "_orig_mod") else model
        try:
            _n_params = int(sum(p.numel() for p in _base_for_params.parameters()))
        except Exception:
            _n_params = 0

        _model_size = str(getattr(config, "model_size", "") or "").strip().lower()
        _env_spike_thr = float(os.environ.get("HYDRA_GRAD_SPIKE_THRESHOLD", "0") or 0.0)
        _env_spike_scale = float(os.environ.get("HYDRA_GRAD_SPIKE_SCALE", "2000") or 2000.0)
        _spike_min_env = os.environ.get("HYDRA_GRAD_SPIKE_MIN", None)
        _env_spike_min = float(_spike_min_env) if _spike_min_env is not None else None

        _default_grad_spike_threshold = _spike.compute_spike_threshold(
            n_params=_n_params,
            model_size_hint=_model_size,
            env_threshold=_env_spike_thr,
            env_scale=_env_spike_scale,
            env_min=_env_spike_min,
        )

        # Spike tracker for rolling window rate analysis
        _halt_spike_window = int(os.environ.get("HYDRA_HALT_SPIKE_WINDOW", "200") or 200)
        _halt_spike_count = int(os.environ.get("HYDRA_HALT_SPIKE_COUNT", "10") or 10)
        _spike_tracker = _spike.SpikeTracker(window_size=_halt_spike_window)

        # Halt policy (no spike-based halts):
        # - Halt on NaN/Inf loss or gradients
        # - Halt on sustained EMA degradation (but only if EMA is also high)
        _halt_ema_window = int(os.environ.get("HYDRA_HALT_EMA_WINDOW", "200") or 200)
        # Relaxed default: 0.5 (was 0.3, too sensitive for MoR curriculum learning)
        _halt_ema_delta = float(os.environ.get("HYDRA_HALT_EMA_DELTA", "0.5") or 0.5)
        # Absolute EMA threshold: don't halt if EMA is still below this (model is learning fine)
        _halt_ema_abs = float(os.environ.get("HYDRA_HALT_EMA_ABS", "8.0") or 8.0)
        _ema_hist = deque(maxlen=max(2, _halt_ema_window + 1))

        if os.environ.get("HYDRA_SMART_HALT_ON_SPIKE", "0") == "1":
            self.logger.warning("HYDRA_SMART_HALT_ON_SPIKE is deprecated/ignored (spikes never halt training).")

        # Rolling clip-rate tracking (do NOT treat occasional clips as failures).
        _clip_window_n = int(os.environ.get("HYDRA_GRAD_CLIP_WINDOW", "200") or 200)
        _clip_hist = deque(maxlen=max(1, _clip_window_n))

        # LR cooldown factor on spike
        _spike_lr_factor_raw = float(os.environ.get("HYDRA_GRAD_SPIKE_LR_FACTOR", "0.5") or 0.5)
        _grad_spike_lr_factor = max(0.01, min(1.0, _spike_lr_factor_raw))

        # Opt-in step diagnostics (HYDRA_DIAG_STEPS, HYDRA_DIAG_FIRST_N)
        _diag_steps = _step_diag.get_diag_steps_from_env(start_step)

        step = int(start_step)

        while step < max_steps:
            step_start = time.time()
            optimizer.zero_grad(set_to_none=True)
            accum_loss = torch.zeros((), device=self.device)
            accum_ce = torch.zeros((), device=self.device)
            micro_diag: List[dict] = []
            collect_micro_diag = os.environ.get("HYDRA_ENABLE_MICRO_DIAG", "0") == "1"
            if collect_micro_diag and step == start_step:
                self.logger.warning("\n" + "⚠" * 35)
                self.logger.warning("  WARNING: HYDRA_ENABLE_MICRO_DIAG=1 is active.")
                self.logger.warning("  This adds ~20 .item() calls per micro-batch, causing")
                self.logger.warning("  significant CPU-GPU sync overhead. Use for debugging only.")
                self.logger.warning("⚠" * 35 + "\n")
            track_loss_scalars = collect_micro_diag or ((log_interval > 0 and step % log_interval == 0) or (step % 500 == 0))

            if use_mod_mor and hasattr(model, "set_global_step"):
                base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
                if hasattr(base_model, "set_global_step"):
                    base_model.set_global_step(step)

            for micro_step in range(grad_accum):
                x, y, mask = self._get_batch()
                if self.device == "cuda" and self.config.use_compile:
                    torch.compiler.cudagraph_mark_step_begin()
                with autocast(device, dtype=dtype):
                    loss, ce_loss, logits, aux_loss_t, ponder_loss_t, advantage_loss_t = compute_microbatch_loss(
                        model=model,
                        x=x,
                        y=y,
                        mask=mask,
                        config=self.config,
                        device=self.device,
                        dtype=dtype,
                        use_mod_mor=use_mod_mor,
                        track_loss_scalars=track_loss_scalars,
                        compute_token_losses_from_hidden=lambda hidden, weight, targets: self._compute_token_losses_from_hidden(
                            hidden,
                            weight,
                            targets,
                            ignore_index=-100,
                        ),
                    )
                    if track_loss_scalars:
                        self._last_ce_loss = ce_loss.detach()
                        self._last_aux_loss = (
                            aux_loss_t.detach()
                            if isinstance(aux_loss_t, torch.Tensor)
                            else torch.tensor(0.0, device=self.device)
                        )
                        self._last_ponder_loss = (
                            ponder_loss_t.detach()
                            if isinstance(ponder_loss_t, torch.Tensor)
                            else torch.tensor(0.0, device=self.device)
                        )
                        self._last_advantage_loss = (
                            advantage_loss_t.detach()
                            if isinstance(advantage_loss_t, torch.Tensor)
                            else torch.tensor(0.0, device=self.device)
                        )
                if collect_micro_diag:
                    try:
                        with torch.no_grad():
                            y_flat = y.view(-1)
                            y_is_ignore = y_flat == -100
                            y_valid = ~y_is_ignore
                            # Store tensors directly to avoid sync
                            y_oob = (y_valid & ((y_flat < 0) | (y_flat >= vocab_size))).sum()
                            y_ignore = y_is_ignore.sum()
                            y_min = y_flat.min() if y_flat.numel() else torch.tensor(0, device=self.device)
                            y_max = y_flat.max() if y_flat.numel() else torch.tensor(0, device=self.device)
                            x_flat = x.view(-1)
                            x_oob = ((x_flat < 0) | (x_flat >= vocab_size)).sum()
                            x_min = x_flat.min() if x_flat.numel() else torch.tensor(0, device=self.device)
                            x_max = x_flat.max() if x_flat.numel() else torch.tensor(0, device=self.device)
                            if logits is None:
                                logits_isfinite = torch.tensor(True, device=self.device)
                                logits_absmax = torch.tensor(0.0, device=self.device)
                                logits_mean = torch.tensor(0.0, device=self.device)
                                logits_std = torch.tensor(0.0, device=self.device)
                            else:
                                logits_f = logits.detach()
                                logits_f32 = logits_f.float()
                                logits_isfinite = torch.isfinite(logits_f32).all()
                                logits_absmax = logits_f32.abs().max() if logits_f32.numel() else torch.tensor(0.0, device=self.device)
                                logits_mean = logits_f32.mean() if logits_f32.numel() else torch.tensor(0.0, device=self.device)
                                logits_std = logits_f32.std(unbiased=False) if logits_f32.numel() else torch.tensor(0.0, device=self.device)
                            micro_diag.append(
                                {
                                    "micro_step": micro_step,
                                    "loss": loss.detach(),
                                    "accum_enabled": True,
                                    "x_min": x_min,
                                    "x_max": x_max,
                                    "x_oob": x_oob,
                                    "y_min": y_min,
                                    "y_max": y_max,
                                    "y_oob": y_oob,
                                    "y_ignore": y_ignore,
                                    "y_valid": int(y_flat.numel()) - y_ignore,
                                    "logits_isfinite": logits_isfinite,
                                    "logits_absmax": logits_absmax,
                                    "logits_mean": logits_mean,
                                    "logits_std": logits_std,
                                }
                            )
                    except Exception:
                        pass
                if self._batch_filter is not None:
                    should_skip, skip_reason = self._batch_filter.should_skip_batch(loss.item(), step)
                    if should_skip:
                        if micro_diag:
                            micro_diag[-1]["accum_enabled"] = False
                            micro_diag[-1]["skip_reason"] = str(skip_reason)
                        if step % 500 == 0:
                            stats = self._batch_filter.get_stats()
                            self.logger.info(
                                f"  [BatchFilter] Skipped batch: {skip_reason}, "
                                f"total skipped: {stats['n_skipped']}/{stats['n_total']} "
                                f"({stats['skip_ratio']*100:.1f}%)"
                            )
                        continue
                
                # In testing mode, check for NaN/Inf immediately (causes sync)
                if self.config.mode == "testing":
                    loss_val = loss.item()
                    if not math.isfinite(loss_val):
                        self.logger.warning(f"  ⚠️  Step {step}: NaN/Inf loss detected, skipping batch")
                        if micro_diag:
                            micro_diag[-1]["accum_enabled"] = False
                            micro_diag[-1]["skip_reason"] = "non_finite_loss"
                        continue
                
                scaled_loss = loss * loss_scale
                if use_scaler:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                accum_loss += loss.detach() * loss_scale
                # Track CE separately so MoD gating can follow the same objective as eval.
                # For non-mod_mor, loss==CE; for mod_mor, ce_loss is defined above.
                try:
                    accum_ce += ce_loss.detach() * loss_scale  # type: ignore[name-defined]
                except Exception:
                    # Fallback: treat total loss as CE if ce_loss isn't available.
                    accum_ce += loss.detach() * loss_scale

            # Resolve accumulated losses (sync points)
            accum_loss = accum_loss.item()
            accum_ce_f = float(accum_ce.item())

            # Opt-in step diagnostics: Phase 1 (before unscale)
            _step_diag_ctx = _step_diag.StepDiagnostics(step=step, active=(step in _diag_steps))
            if _step_diag_ctx.active:
                _base_for_diag = model._orig_mod if hasattr(model, "_orig_mod") else model
                _step_diag.collect_phase1_batch_stats(_step_diag_ctx, x, y, _base_for_diag, self.logger)

            # Maintain an EMA over CE for curriculum decisions (mirrors eval objective).
            # Use the same alpha as TrainingMetrics.ema_alpha (0.05) unless overridden.
            ce_alpha = float(getattr(self.metrics, "ema_alpha", 0.05))
            self._ce_ema = update_scalar_ema(ema=float(self._ce_ema), value=float(accum_ce_f), alpha=float(ce_alpha))

            # Resolve micro_diag tensors if collected
            resolve_micro_diag_tensors(micro_diag)

            if use_scaler:
                scaler.unscale_(optimizer)

            # Opt-in step diagnostics: Phase 2 (after unscale, before clip)
            if _step_diag_ctx.active:
                _step_diag.collect_phase2_preclip_grads(_step_diag_ctx, model, self.logger)

            base = model._orig_mod if hasattr(model, "_orig_mod") else model
            pre_clip_norm_t = torch.nn.utils.clip_grad_norm_(base.parameters(), grad_clip)
            pre_clip_norm = float(pre_clip_norm_t) if torch.is_tensor(pre_clip_norm_t) else float(pre_clip_norm_t)

            clipped_this_step = math.isfinite(pre_clip_norm) and (pre_clip_norm > float(grad_clip))
            _clip_hist.append(1 if clipped_this_step else 0)
            clip_pct = 100.0 * (sum(_clip_hist) / max(1, len(_clip_hist)))

            # Opt-in step diagnostics: Phase 3 (after clip) + log output
            if _step_diag_ctx.active:
                _step_diag.collect_phase3_postclip_grads(_step_diag_ctx, model, self.logger)
                _step_diag.log_step_diagnostics(_step_diag_ctx, accum_loss, self.logger)

            self._last_pre_clip_norm = pre_clip_norm  # Store for diagnostics
            grad_norm = min(pre_clip_norm, float(grad_clip)) if math.isfinite(pre_clip_norm) else float("nan")

            # Clip coefficient implied by global norm (uniform scaling factor).
            clip_coef = grad_clip / (pre_clip_norm + 1e-12)
            clip_scale = min(1.0, clip_coef) if math.isfinite(clip_coef) else 0.0
            try:
                scaler_scale = float(scaler.get_scale()) if use_scaler else None
            except Exception:
                scaler_scale = None

            base_lr, lr, self._resume_lr_scale, self._resume_lr_override_target = compute_step_lr(
                step=step,
                config=self.config,
                adaptive_lr=self._adaptive_lr,
                adaptive_metric=str(getattr(self.config, "adaptive_metric", "train")),
                accum_loss=float(accum_loss),
                resume_lr_scale=float(getattr(self, "_resume_lr_scale", 1.0) or 1.0),
                resume_lr_override_target=float(getattr(self, "_resume_lr_override_target", 0.0) or 0.0),
                resume_rewarmup_start_step=int(getattr(self, "_resume_rewarmup_start_step", 0) or 0),
                resume_rewarmup_steps=int(getattr(self, "_resume_rewarmup_steps", 0) or 0),
                get_lr=get_lr,
            )
            lr_effective = lr
            spike_detected = _spike.is_spike(pre_clip_norm, _default_grad_spike_threshold)
            nonfinite_grads = not math.isfinite(pre_clip_norm)

            # ============================================================
            # SPIKE/NONFINITE HANDLING - Single unified branch
            # ============================================================
            if spike_detected or nonfinite_grads:
                # Apply LR cooldown for spikes
                if spike_detected:
                    lr_effective = lr * _grad_spike_lr_factor

                # Get loss component values
                ce_val = float(self._last_ce_loss.item() if hasattr(self._last_ce_loss, "item") else self._last_ce_loss)
                aux_val = float(self._last_aux_loss.item() if hasattr(self._last_aux_loss, "item") else self._last_aux_loss)
                ponder_val = float(self._last_ponder_loss.item() if hasattr(self._last_ponder_loss, "item") else self._last_ponder_loss)
                adv_val = float(self._last_advantage_loss.item() if hasattr(self._last_advantage_loss, "item") else self._last_advantage_loss)

                # Comprehensive spike analysis via spike_diagnostics module
                # Returns SpikeContext with probable_causes, batch_stats, etc.
                _spike.handle_spike(
                    step=step,
                    pre_clip_norm=pre_clip_norm,
                    grad_norm=grad_norm,
                    grad_clip=float(grad_clip),
                    threshold=_default_grad_spike_threshold,
                    accum_loss=float(accum_loss),
                    ce_loss=ce_val,
                    aux_loss=aux_val,
                    ponder_loss=ponder_val,
                    advantage_loss=adv_val,
                    lr=float(lr),
                    lr_effective=float(lr_effective),
                    scaler_scale=scaler_scale,
                    clip_coef=float(clip_coef),
                    clip_scale=float(clip_scale),
                    base_model=base,
                    x=x,
                    y=y,
                    vocab_size=vocab_size,
                    device=self.device,
                    dtype=dtype,
                    checkpoint_dir=self.config.checkpoint_dir,
                    logger=self.logger,
                    spike_tracker=_spike_tracker,
                    verbose=True,  # Always log detailed spike analysis
                    dump_batch=self.config.halt_on_spike,  # Only dump batch file on explicit halt_on_spike
                )

                # Check for non-finite attention outputs -> halt
                bad_batch_path = _spike.check_attention_nonfinite(
                    base_model=base,
                    step=step,
                    checkpoint_dir=self.config.checkpoint_dir,
                    x=x,
                    y=y,
                    logger=self.logger,
                )
                if bad_batch_path:
                    halt_step_time = max(1e-9, time.time() - step_start)
                    halt_tps = tokens_per_step / halt_step_time
                    metrics.update(step, accum_loss, lr_effective, grad_norm, halt_tps, halt_step_time)
                    metrics.total_tokens += tokens_per_step
                    metrics.final_loss = accum_loss
                    self._save_checkpoint(step)
                    if profiler:
                        profiler.stop()
                    metrics.end_time = time.time()
                    return metrics

                # Handle non-finite gradients -> halt
                if skip_update_for_nonfinite_gradients(
                    nonfinite_grads=nonfinite_grads,
                    optimizer=optimizer,
                    use_scaler=use_scaler,
                    scaler=scaler,
                ):
                    self.logger.warning(f"  🛑 HALT at step {step}: NaN/Inf gradients")
                    try:
                        halt_step_time = max(1e-9, time.time() - step_start)
                        halt_tps = tokens_per_step / halt_step_time
                        metrics.update(step, accum_loss, lr_effective, grad_norm, halt_tps, halt_step_time)
                        metrics.total_tokens += tokens_per_step
                        metrics.final_loss = accum_loss
                        self._save_checkpoint(step)
                    except Exception as e:
                        self.logger.warning(f"  ⚠️  Halt: failed to save state ({e})")
                    if profiler:
                        profiler.stop()
                    metrics.end_time = time.time()
                    return metrics

                # Spike detected (not halting): apply LR cooldown and continue
                if spike_detected:
                    lr = lr_effective
                    self.logger.warning(f"  🔻 Spike response: mild LR cooldown x{_grad_spike_lr_factor}")

            # Skip LR updates for Adafactor (uses internal 1/√t schedule)
            if not self._skip_lr_schedule:
                embed_lr_scale = getattr(self, "_embed_lr_scale", 1.0)
                for i, pg in enumerate(param_groups):
                    # Param group 2 (index 2) is embeddings - apply lower LR scale
                    if i == 2 and embed_lr_scale < 1.0:
                        pg["lr"] = lr * embed_lr_scale
                    else:
                        pg["lr"] = lr
            if use_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            if self._adaptive_lr is not None:
                base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
                self._adaptive_lr.update_swa(base_model)
            step += 1
            if profiler:
                profiler.step()

            if hasattr(self.train_loader, "set_step"):
                self.train_loader.set_step(step)
            step_time = time.time() - step_start
            tps = tokens_per_step / step_time
            target_seq_len = self._get_seq_len_for_step(step)
            if target_seq_len != self._current_seq_len:
                self._recreate_dataloader(target_seq_len)
                tokens_per_step = self._tokens_per_step
            if step == start_step + 1:
                metrics.initial_loss = accum_loss
                if start_step == 0:
                    self.logger.info(f"Initial loss: {accum_loss:.4f}")
                else:
                    prev_best_str = f"{self._prev_run_best_loss:.4f}" if math.isfinite(self._prev_run_best_loss) else "N/A"
                    self.logger.info(
                        f"Resume loss: {accum_loss:.4f} (prev run best: {prev_best_str}; best tracking reset on resume)"
                    )
            prev_best = metrics.best_loss
            metrics.update(step, accum_loss, lr, grad_norm, tps, step_time)
            metrics.total_tokens += tokens_per_step

            # EMA debug: trace loss -> EMA flow
            _ema_debug = getattr(self.config, "ema_debug", False)
            if _ema_debug and step % 25 == 0:
                self.logger.info(
                    f"  [EMA_DEBUG] step={step} mode=train input_loss={accum_loss:.4f} -> ema_loss={metrics.ema_loss:.4f} "
                    f"(ce_ema={self._ce_ema:.4f})"
                )

            # Halt policy: sustained EMA degradation or spike explosion + EMA blow-up.
            try:
                if not math.isfinite(float(accum_loss)):
                    raise ValueError("non-finite loss")
            except Exception:
                self.logger.warning(f"  🛑 HALT at step {step}: NaN/Inf loss")
                self._save_checkpoint(step)
                if profiler:
                    profiler.stop()
                metrics.end_time = time.time()
                metrics.final_loss = accum_loss
                return metrics

            try:
                _ema_hist.append(float(metrics.ema_loss))
                ema_degrade = False
                ema_delta = 0.0
                if len(_ema_hist) >= _ema_hist.maxlen:
                    ema_delta = float(_ema_hist[-1]) - float(_ema_hist[0])
                    current_ema = float(_ema_hist[-1])
                    # Only halt if:
                    # 1. EMA increased by more than threshold over window, AND
                    # 2. Current EMA is above the absolute threshold (model is actually struggling)
                    # This prevents false halts during normal variance when loss is still reasonable
                    delta_exceeded = math.isfinite(ema_delta) and (ema_delta > float(_halt_ema_delta))
                    ema_is_high = current_ema > float(_halt_ema_abs)
                    ema_degrade = delta_exceeded and ema_is_high
                    
                    # EMA debug: show halt check state
                    if _ema_debug and step % 25 == 0:
                        self.logger.info(
                            f"  [EMA_DEBUG] halt_check: window_len={len(_ema_hist)} "
                            f"oldest={_ema_hist[0]:.4f} newest={current_ema:.4f} "
                            f"delta={ema_delta:.4f} (thr={_halt_ema_delta}) "
                            f"ema_high={ema_is_high} (thr={_halt_ema_abs})"
                        )
                spikes_in_window = _spike_tracker.spike_count_in_window()
                if ema_degrade:
                    # Print full diagnostic before halting
                    self.logger.warning(
                        f"  [EMA_DEBUG] HALT TRIGGER: delta={ema_delta:.4f} > threshold={_halt_ema_delta} "
                        f"window=[{_ema_hist[0]:.4f}...{_ema_hist[-1]:.4f}] "
                        f"current_loss={accum_loss:.4f}"
                    )
                    if spikes_in_window >= int(_halt_spike_count):
                        halt_reason = f"spike-rate explosion ({spikes_in_window}/{_halt_spike_window}) + EMA blow-up (Δ={ema_delta:.3f} over {_halt_ema_window})"
                    else:
                        halt_reason = f"sustained EMA degradation (Δ={ema_delta:.3f} over {_halt_ema_window} steps)"
                    self.logger.warning(f"  🛑 HALT at step {step}: {halt_reason}")
                    self._save_checkpoint(step)
                    if profiler:
                        profiler.stop()
                    metrics.end_time = time.time()
                    metrics.final_loss = accum_loss
                    return metrics
            except Exception:
                pass
            if use_mod_mor:
                base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
                if hasattr(base_model, "update_mod_loss_ema"):
                    # IMPORTANT: MoD enable threshold is based on EMA CE (not total loss),
                    # so it mirrors eval and doesn't get dominated by routing auxiliaries.
                    base_model.update_mod_loss_ema(self._ce_ema)
                # Dynamic MoR enable: trigger when loss drops below threshold
                # BUT respect mor_enable_min_steps as a hard floor (loss can't bypass it)
                if not getattr(self, "_mor_triggered_by_loss", False):
                    mor_loss_thr = getattr(self.config, "mor_enable_loss_threshold", 0.0) or 0.0
                    mor_enable_step = getattr(self, "_mor_enable_step", 0)
                    mor_min_steps = getattr(self.config, "mor_enable_min_steps", 1000)
                    # Loss threshold can only trigger AFTER mor_enable_min_steps (hard floor)
                    if (mor_loss_thr > 0 and self._ce_ema > 0 and self._ce_ema < mor_loss_thr 
                        and step >= mor_min_steps and step < mor_enable_step):
                        # Loss threshold reached before scheduled enable step - trigger MoR now
                        rampup = min(1000, max(100, int(self.config.max_steps * 0.05)))
                        if hasattr(base_model, "trigger_mor_early"):
                            base_model.trigger_mor_early(step, rampup_steps=rampup)
                            self._mor_enable_step = step
                            self._mor_triggered_by_loss = True
                            self.logger.info(f"🚀 MoR EARLY TRIGGER: CE_EMA={self._ce_ema:.3f} < threshold={mor_loss_thr:.1f} at step {step}. Starting {rampup}-step rampup now.")
                
                # MoR-informed MoD: Enable MoD when MoR early_exit_ratio exceeds threshold
                # This ensures MoD only activates once the model demonstrates it can route confidently
                if not getattr(self, "_mod_triggered", False):
                    mod_min_step = getattr(self.config, "mod_enable_min_step", 3000)
                    mod_mor_threshold = getattr(self.config, "mod_enable_mor_early_exit_threshold", 0.38)
                    mod_loss_thr = getattr(self.config, "mod_enable_loss_threshold", 4.5)
                    # Check every 100 steps to avoid overhead
                    # Require minimum steps for routing stats to stabilize
                    if step % 100 == 0 and step >= mod_min_step and mod_mor_threshold < 1.0:
                        # Get MoR early_exit_ratio from routing stats
                        if hasattr(base_model, "get_routing_stats"):
                            stats = base_model.get_routing_stats()
                            mor_stats = stats.get("mor_layers", [])
                            if mor_stats:
                                # Aggregate depth histograms across all MoR layers
                                all_hists = [s.get("depth_histogram", []) for s in mor_stats if s.get("depth_histogram")]
                                if all_hists:
                                    max_len = max(len(h) for h in all_hists)
                                    agg_hist = [0.0] * max_len
                                    for h in all_hists:
                                        for i, v in enumerate(h):
                                            agg_hist[i] += v
                                    total = sum(agg_hist) or 1
                                    # Early exit ratio = tokens that exit before max depth
                                    # Last bucket is max depth, so early_exit = 1 - (last_bucket / total)
                                    if max_len > 0:
                                        early_exit_ratio = 1.0 - (agg_hist[-1] / total)
                                        # Trigger MoD if: early_exit > threshold AND loss < loss_threshold
                                        loss_ok = (mod_loss_thr <= 0) or (self._ce_ema > 0 and self._ce_ema < mod_loss_thr)
                                        if early_exit_ratio > mod_mor_threshold and loss_ok:
                                            if hasattr(base_model, "trigger_mod_from_mor"):
                                                base_model.trigger_mod_from_mor(step)
                                                self._mod_triggered = True
                                                self.logger.info(
                                                    f"🎯 MoD ENABLED: MoR early_exit_ratio={early_exit_ratio:.1%} > threshold={mod_mor_threshold:.0%}, "
                                                    f"CE_EMA={self._ce_ema:.3f} < {mod_loss_thr:.1f} at step {step}"
                                                )
            if step >= 1000 and step % 500 == 0:
                _checkpointing.maybe_save_best_checkpoint(
                    step=step,
                    prev_best_loss=float(prev_best),
                    best_loss=float(metrics.best_loss),
                    save_checkpoint_fn=lambda s, best=False: self._save_checkpoint(s, best=best),
                )
            eval_loss = maybe_run_fixed_eval(
                step=step,
                eval_interval=eval_interval,
                model=model,
                fixed_eval_batches=fixed_eval_batches,
                device=device,
                dtype=dtype,
                config=self.config,
                use_mod_mor=use_mod_mor,
                logger=self.logger,
                train_loss=accum_loss,
                eval_debug=_eval_debug,
            )
            if eval_loss is not None:
                # One-time sanity check: run eval codepath on a training batch
                if not _eval_sanity_done:
                    _eval_sanity_done = True
                    # Get a fresh training batch for the sanity check
                    _sanity_batch = self.train_loader.get_batch()
                    _sanity_batch_with_mask = {
                        "input_ids": _sanity_batch["input_ids"],
                        "labels": _sanity_batch["labels"],
                        "attention_mask": _sanity_batch.get("attention_mask"),
                    }
                    _base_for_sanity = model._orig_mod if hasattr(model, "_orig_mod") else model
                    _base_for_sanity.eval()
                    eval_sanity_check_on_train_batch(
                        base_model=_base_for_sanity,
                        train_batch=_sanity_batch_with_mask,
                        device=device,
                        dtype=dtype,
                        config=self.config,
                        use_mod_mor=use_mod_mor,
                        current_train_loss=accum_loss,
                        current_ema=metrics.ema_loss,
                        logger=self.logger,
                    )
                    _base_for_sanity.train()
                
                # EMA debug: confirm eval does NOT update train EMA
                if _ema_debug:
                    self.logger.info(
                        f"  [EMA_DEBUG] step={step} mode=EVAL eval_loss={eval_loss:.4f} "
                        f"(train_ema unchanged at {metrics.ema_loss:.4f})"
                    )
                if self._adaptive_lr is not None and getattr(self.config, "adaptive_metric", "train") == "eval":
                    self._adaptive_lr.update(step, float(eval_loss))
                if tb_writer is not None:
                    tb_writer.add_scalar("eval/loss", float(eval_loss), step)
                if self.config.use_wandb and wandb_mod is not None:
                    wandb_mod.log({"eval_loss": float(eval_loss), "step": step})
            if step % log_interval == 0:
                elapsed = time.time() - metrics.start_time
                steps_this_session = step - start_step
                tokens_this_session = steps_this_session * tokens_per_step
                avg_tps = tokens_this_session / elapsed if elapsed > 0 else 0
                steps_per_sec = steps_this_session / elapsed if elapsed > 0 else 0
                self.logger.info(
                    f"Step {step:5d}/{max_steps} | "
                    f"Loss: {accum_loss:.4f} (EMA: {metrics.ema_loss:.4f}) | "
                    f"LR: {lr:.2e} | "
                    f"Grad: pre {pre_clip_norm:.2e} | post {grad_norm:.2e} | clipped {clip_pct:4.1f}% | "
                    f"{tps/1000:.1f}K tok/s | "
                    f"Avg: {avg_tps/1000:.1f}K tok/s ({steps_per_sec:.2f} steps/s)"
                )

                if tb_writer is not None:
                    tb_writer.add_scalar("train/loss", float(accum_loss), step)
                    tb_writer.add_scalar("train/ema_loss", float(metrics.ema_loss), step)
                    tb_writer.add_scalar("train/lr", float(lr), step)
                    tb_writer.add_scalar("train/grad_norm", float(grad_norm), step)
                    tb_writer.add_scalar("train/grad_norm_pre_clip", float(pre_clip_norm), step)
                    tb_writer.add_scalar("train/grad_clip_pct", float(clip_pct), step)
                    tb_writer.add_scalar("train/grad_clipped", 1.0 if clipped_this_step else 0.0, step)
                    tb_writer.add_scalar("train/tps", float(tps), step)
                
                if self.config.use_wandb and wandb_mod is not None:
                    wandb_mod.log(
                        {
                            "train_loss": float(accum_loss),
                            "ema_loss": float(metrics.ema_loss),
                            "lr": float(lr),
                            "grad_norm": float(grad_norm),
                            "grad_norm_pre_clip": float(pre_clip_norm),
                            "grad_clip_pct": float(clip_pct),
                            "grad_clipped": 1.0 if clipped_this_step else 0.0,
                            "tps": float(tps),
                            "avg_tps": float(avg_tps),
                            "step": step,
                        }
                    )
                if use_mod_mor and step % 100 == 0:
                    self._log_layer_diagnostics(step, accum_loss, lr, grad_norm)
            if step % save_interval == 0:
                self._save_checkpoint(step)
                if self._should_early_stop(accum_loss if accum_loss > 0 else metrics.losses[-1], step, self._count_params()):
                    self.logger.warning(f"\n⚠️  EARLY STOPPING at step {step}")
                    self.logger.warning(f"   Loss has increased for {config.early_stop_patience} consecutive checkpoints")
                    self.logger.warning(f"   Checkpoint losses: {self._checkpoint_losses[-4:]}")
                    break
            accum_loss = 0.0

        if profiler:
            profiler.stop()

        metrics.end_time = time.time()
        metrics.final_loss = metrics.losses[-1] if metrics.losses else 0.0

        self.logger.info("-" * 70)
        self.logger.info("Training complete!")

        if self._adaptive_lr is not None and self._adaptive_lr.swa_n > 0:
            base_model = self.model
            if hasattr(base_model, "_orig_mod"):
                base_model = base_model._orig_mod
            self._adaptive_lr.apply_swa(base_model)

        self._save_checkpoint(step, final=True)
        self._generate_report()
        self._save_diagnostics()

        if tb_writer is not None:
            try:
                tb_writer.flush()
                tb_writer.close()
            except Exception:
                pass

        if wandb_run is not None and wandb_mod is not None:
            try:
                wandb_mod.finish()
            except Exception:
                pass
        return metrics

    def _log_layer_diagnostics(self, step: int, loss: float, lr: float, grad_norm: float) -> None:
        model = self.model
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
        if hasattr(model, "get_routing_stats"):
            stats = model.get_routing_stats()
        else:
            return
        record = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "losses": {
                "total": float(loss) if hasattr(loss, "item") else float(loss),
                "ce": float(self._last_ce_loss.item()) if hasattr(self._last_ce_loss, "item") else float(self._last_ce_loss),
                "aux": float(self._last_aux_loss.item()) if hasattr(self._last_aux_loss, "item") else float(self._last_aux_loss),
                "ponder": float(self._last_ponder_loss.item()) if hasattr(self._last_ponder_loss, "item") else float(self._last_ponder_loss),
                "advantage": float(self._last_advantage_loss.item()) if hasattr(self._last_advantage_loss, "item") else float(self._last_advantage_loss),
            },
            "lr": lr,
            "grad_norm": grad_norm,
            "grad_norm_pre_clip": getattr(self, "_last_pre_clip_norm", grad_norm),  # Raw norm before clipping
            "mod_layers": [],
            "mor_layers": [],
        }
        if self._adaptive_lr is not None:
            adaptive_state = self._adaptive_lr.get_state()
            record["adaptive_lr"] = {
                "loss_ema_short": adaptive_state.get("loss_ema_short", 0.0),
                "loss_ema_long": adaptive_state.get("loss_ema_long", 0.0),
                "cooldown_triggered": adaptive_state.get("cooldown_triggered", False),
                "decay_start_step": adaptive_state.get("decay_start_step", 0),
                "decay_steps": adaptive_state.get("decay_steps", 0),
                "patience_counter": adaptive_state.get("patience_counter", 0),
            }
        mod_stats = stats.get("mod_layers", [])
        mod_issues = []
        for layer_stat in mod_stats:
            layer_idx = layer_stat.get("layer", -1)
            probs_mean = layer_stat.get("probs_mean", 0.5)
            target = layer_stat.get("target_capacity", 0.5)
            tokens_processed = layer_stat.get("tokens_processed", 0)
            tokens_total = layer_stat.get("tokens_total", 0)
            compute_ratio = layer_stat.get("compute_ratio", 1.0)
            compute_savings = layer_stat.get("compute_savings_pct", 0.0)
            routing_mode = layer_stat.get("routing_mode", "unknown")
            global_step = layer_stat.get("global_step", 0)
            warmup_steps = layer_stat.get("warmup_steps", 100)
            force_enable_step = layer_stat.get("force_enable_step", warmup_steps)
            deviation = abs(probs_mean - target)
            status = "OK"
            # MoD can be intentionally disabled before its curriculum enable step.
            # In that phase, probs_mean may be 0.0 by design and should not be
            # reported as router collapse.
            mod_active = (routing_mode != "disabled") and (global_step >= warmup_steps)
            if probs_mean < 0.1:
                status = "DISABLED_PRE_ENABLE" if not mod_active else "COLLAPSED_LOW"
                if mod_active:
                    mod_issues.append(f"Layer {layer_idx}: probs={probs_mean:.3f} (collapsed to skip)")
            elif probs_mean > 0.9:
                status = "DISABLED_PRE_ENABLE" if not mod_active else "COLLAPSED_HIGH"
                if mod_active:
                    mod_issues.append(f"Layer {layer_idx}: probs={probs_mean:.3f} (collapsed to process all)")
            elif deviation > 0.2:
                status = "DISABLED_PRE_ENABLE" if not mod_active else "DRIFTING"
                if mod_active:
                    mod_issues.append(f"Layer {layer_idx}: probs={probs_mean:.3f} vs target={target:.2f}")
            record["mod_layers"].append(
                {
                    "layer": layer_idx,
                    "probs_mean": probs_mean,
                    "probs_std": layer_stat.get("probs_std", 0),
                    "target": target,
                    "k_selected": tokens_processed,
                    "k_total": tokens_total,
                    "selected_frac": compute_ratio,
                    "compute_savings_pct": compute_savings,
                    "routing_mode": routing_mode,
                    "global_step": global_step,
                    "warmup_steps": warmup_steps,
                    "force_enable_step": force_enable_step,
                    "status": status,
                }
            )
        mor_phase_raw = "unknown"
        mor_status = {}
        if hasattr(model, "get_mor_status"):
            mor_status = model.get_mor_status()
            mor_phase_raw = mor_status.get("phase", "unknown")

        mor_stats = stats.get("mor_layers", [])
        mor_issues = []
        for layer_stat in mor_stats:
            layer_idx = layer_stat.get("layer", -1)
            avg_depth = layer_stat.get("avg_depth", 0)
            expected_depth = layer_stat.get("expected_avg_depth", 1.5)
            router_probs = layer_stat.get("router_probs_mean", 0.5)
            depth_hist = layer_stat.get("depth_histogram", [])
            status = "OK"
            
            # Only flag router collapse if we are in adaptive phase
            # Use depth_histogram (actual routing) not router_probs_mean (raw sigmoid)
            if mor_phase_raw in ["full-adaptive", "rampup"] and depth_hist:
                total = sum(depth_hist)
                if total > 0:
                    # Check if >90% of tokens at depth 0 (early exit collapse)
                    if depth_hist[0] / total > 0.9:
                        status = "DEPTH_COLLAPSED_EARLY"
                        mor_issues.append(f"Layer {layer_idx}: {depth_hist[0]/total*100:.0f}% at depth 0 (early exit collapse)")
                    # Check if >90% of tokens at max depth (max depth collapse)
                    elif depth_hist[-1] / total > 0.9:
                        status = "DEPTH_COLLAPSED_MAX"
                        mor_issues.append(f"Layer {layer_idx}: {depth_hist[-1]/total*100:.0f}% at max depth (no early exit)")
                    # Check if >90% at any single depth (collapsed to fixed)
                    elif max(depth_hist) / total > 0.9:
                        dominant_idx = depth_hist.index(max(depth_hist))
                        status = "DEPTH_COLLAPSED"
                        mor_issues.append(f"Layer {layer_idx}: {max(depth_hist)/total*100:.0f}% at depth {dominant_idx}")
            
            record["mor_layers"].append(
                {
                    "layer": layer_idx,
                    "avg_depth": avg_depth,
                    "expected_depth": expected_depth,
                    "router_probs_mean": router_probs,
                    "depth_histogram": depth_hist,
                    "status": status,
                }
            )
        self._diagnostics_data.append(record)
        
        if mor_phase_raw == "fixed-depth":
            mor_phase = f"FIXED ({mor_status.get('rampup_progress', 0)*100:.0f}%)"
        elif mor_phase_raw == "rampup":
            mor_phase = f"RAMP ({mor_status.get('rampup_progress', 0)*100:.0f}%)"
        else:
            mor_phase = "FULL"
        mod_mode = "N/A"
        mod_savings = 0.0
        if mod_stats:
            mod_mode = mod_stats[0].get("routing_mode", "?").upper()
            mod_savings = sum(s.get("compute_savings_pct", 0) for s in mod_stats) / max(1, len(mod_stats))
        mor_depth = 0.0
        depth_dist = ""
        if mor_stats:
            depths = [s.get("avg_depth", 0) for s in mor_stats if "avg_depth" in s]
            mor_depth = sum(depths) / len(depths) if depths else 0
            all_hists = [s.get("depth_histogram", []) for s in mor_stats if s.get("depth_histogram")]
            if all_hists:
                max_len = max(len(h) for h in all_hists)
                agg_hist = [0] * max_len
                for h in all_hists:
                    for i, v in enumerate(h):
                        agg_hist[i] += v
                total = sum(agg_hist) or 1
                depth_dist = "/".join([f"{100*v/total:.0f}" for v in agg_hist])
        show_mod_issues = bool(mod_issues) and (float(getattr(self.config, "mod_capacity", 1.0)) < 1.0)
        show_mor_issues = bool(mor_issues) and bool(getattr(self.config, "mor_adaptive", False))
        if show_mod_issues or show_mor_issues:
            self.logger.warning(
                f"  ⚠️  Issues: MoD={len(mod_issues) if show_mod_issues else 0}, MoR={len(mor_issues) if show_mor_issues else 0}"
            )
            if show_mod_issues:
                self.logger.warning(f"      MoD Issues: {mod_issues[:3]}...")
            if show_mor_issues:
                self.logger.warning(f"      MoR Issues: {mor_issues[:3]}...")
        if step % 500 == 0:
            adaptive_info = ""
            if self._adaptive_lr is not None:
                state = self._adaptive_lr.get_state()
                ema_s = state.get("loss_ema_short", 0)
                ema_l = state.get("loss_ema_long", 0)
                trend = "↑" if ema_s > ema_l * 1.02 else ("↓" if ema_s < ema_l * 0.98 else "→")
                adaptive_info = f" | EMA: {ema_s:.3f}/{ema_l:.3f} {trend}"
            adv_str = f" adv={self._last_advantage_loss:.4f}" if self._last_advantage_loss != 0 else ""
            lr_info = f" | LR={lr:.2e} (min={self.config.min_lr:.2e})"
            mod_min_step = getattr(self.config, "mod_enable_min_step", 3000)
            mod_loss_thr = float(getattr(self.config, "mod_enable_loss_threshold", 4.5) or 0.0)
            mod_mor_thr = getattr(self.config, "mod_enable_mor_early_exit_threshold", 0.38)
            mod_triggered = getattr(self, "_mod_triggered", False)
            # Build waiting condition string
            if mod_triggered:
                mod_status = "MoD:ON"
            else:
                conditions = []
                if step < mod_min_step:
                    conditions.append(f"step≥{mod_min_step}")
                conditions.append(f"MoR>{mod_mor_thr:.0%}")
                if mod_loss_thr > 0:
                    conditions.append(f"loss<{mod_loss_thr:.1f}")
                mod_status = f"MoD:wait({','.join(conditions)})"
            mod_gate = f" | CE_EMA={getattr(self, '_ce_ema', 0.0):.3f} {mod_status}"
            self.logger.info(
                f"  [DIAG] MoD:{mod_mode} save={mod_savings:.0f}% | MoR:{mor_phase} d={mor_depth:.2f} [{depth_dist}%] | "
                f"CE={self._last_ce_loss:.3f} aux={self._last_aux_loss:.4f} ponder={self._last_ponder_loss:.3f}{adv_str}{adaptive_info}{lr_info}{mod_gate}"
            )
            self._save_diagnostics()

    def _save_diagnostics(self) -> None:
        _checkpointing.save_diagnostics(
            checkpoint_dir=self.config.checkpoint_dir,
            diagnostics_data=self._diagnostics_data,
            logger=self.logger,
            run_id=self.config.run_id,
        )

    def _save_checkpoint(self, step: int, final: bool = False, best: bool = False) -> None:
        _checkpointing.save_checkpoint(
            step=step,
            config=self.config,
            model=self.model,
            optimizer=self.optimizer,
            scaler=self.scaler,
            metrics=self.metrics,
            checkpoint_history=self._checkpoint_history,
            adaptive_lr_manager=self._adaptive_lr,
            final=final,
            best=best,
            logger=self.logger,
            trainer_state={
                "ce_ema": float(getattr(self, "_ce_ema", 0.0) or 0.0),
            },
        )

    def _cleanup_old_checkpoints(self) -> None:
        # Backwards-compatible wrapper (implementation lives in hydra.training.checkpointing)
        _checkpointing._cleanup_old_checkpoints(
            checkpoint_history=self._checkpoint_history,
            max_checkpoints=self.config.max_checkpoints,
            logger=self.logger,
        )

    def _should_early_stop(self, current_loss: float, current_step: int, total_params: int) -> bool:
        config = self.config
        self._checkpoint_losses.append(current_loss)
        chinchilla_tokens = total_params * config.chinchilla_multiplier
        tokens_so_far = current_step * config.tokens_per_step
        progress = tokens_so_far / chinchilla_tokens
        if progress < config.early_stop_min_progress:
            return False
        if len(self._checkpoint_losses) < 2:
            return False
        prev_loss = self._checkpoint_losses[-2]
        relative_increase = (current_loss - prev_loss) / prev_loss if prev_loss > 0 else 0
        if relative_increase > config.early_stop_threshold:
            self._early_stop_counter += 1
            self.logger.warning(
                f"   ⚠️  Loss increased: {prev_loss:.4f} → {current_loss:.4f} (+{relative_increase*100:.1f}%) "
                f"[{self._early_stop_counter}/{config.early_stop_patience}]"
            )
            self.logger.warning(f"       (Chinchilla progress: {progress*100:.1f}%, early stop active)")
        else:
            self._early_stop_counter = 0
        return self._early_stop_counter >= config.early_stop_patience

    def _generate_report(self) -> None:
        _reporting.generate_report(
            config=self.config,
            metrics=self.metrics,
            device=self.device,
            logger=self.logger,
            format_time=self._format_time,
        )

    def _assess_model_performance(self, metrics: TrainingMetrics) -> Dict[str, str]:
        return _reporting.assess_model_performance(config=self.config, metrics=metrics)

    def _assess_training_quality(self, metrics: TrainingMetrics) -> Dict[str, str]:
        return _reporting.assess_training_quality(metrics=metrics)

    @staticmethod
    def _format_time(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.1f}s"
        if seconds < 3600:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins}m {secs}s"
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"

    def close(self) -> None:
        if hasattr(self, "train_loader") and self.train_loader:
            self.train_loader.close()
