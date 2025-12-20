from __future__ import annotations

import math
import os
import time
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

from hydra.model.ccgqa import CCGQAMoDMoRModel
from hydra.kernels import chunked_cross_entropy, fused_chunked_cross_entropy
from universal_data_loader import create_universal_loader
from data_filter import BatchFilter, FilterConfig

from .config import TrainingConfig
from .metrics import TrainingMetrics
from .lr import get_lr, ProgressAwareLRManager
from .runtime import configure_runtime

_RUNTIME_STATUS = configure_runtime()


class Trainer:
    """Optimized trainer with all performance best practices."""

    __slots__ = (
        "config",
        "device",
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
        "_diagnostics_file",
        "_diagnostics_data",
        "_last_ce_loss",
        "_last_aux_loss",
        "_last_ponder_loss",
        "_last_advantage_loss",
        "_adaptive_lr",
        "_use_progress_aware_lr",
        "_batch_filter",
        "_checkpoint_seq_len",
        "_checkpoint_config",
        "_checkpoint_lr",
        "_resume_lr_scale",
        "_kernel_status",
        "_checkpoint_adaptive_state",
        "_seed_set",
    )

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.metrics = TrainingMetrics()

        if self.device == "cuda":
            os.environ.setdefault("HYDRA_MOR_ATTENTION_PATTERN_NAME", "lla2x3+ccqa")

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

        self._adaptive_lr: Optional[ProgressAwareLRManager] = None
        self._use_progress_aware_lr = config.adaptive_lr or config.use_swa or config.lr_schedule == "wsd_adaptive"

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
            print(f"WARNING: Failed to configure Triton kernels ({e})")

        self._checkpoint_seq_len = None
        self._checkpoint_config = {}
        self._checkpoint_lr = None
        self._resume_lr_scale: float = 1.0
        if config.resume_from:
            self._checkpoint_config = self._peek_checkpoint_config(config.resume_from)
            self._checkpoint_seq_len = self._checkpoint_config.get("max_seq_len", config.max_seq_len)
            if "architecture" in self._checkpoint_config:
                ckpt_arch = self._checkpoint_config["architecture"]
                if ckpt_arch != config.architecture:
                    print(f"Overriding architecture: {config.architecture} -> {ckpt_arch} (from checkpoint)")
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
            if "optimizer" in torch.load(config.resume_from, weights_only=False, map_location="cpu"):
                ckpt_full = torch.load(config.resume_from, weights_only=False, map_location="cpu")
                if "optimizer" in ckpt_full and "param_groups" in ckpt_full["optimizer"]:
                    last_lr = ckpt_full["optimizer"]["param_groups"][0].get("lr", config.max_lr)
                    print(f"Checkpoint final LR: {last_lr:.6f}")
                    self._checkpoint_lr = last_lr

        self._setup_model()
        self._setup_optimizer()
        self._setup_data()

        if config.resume_from:
            self._load_checkpoint(config.resume_from)

        print(f"\n{'='*70}")
        print("HYDRA 100M Optimized Trainer")
        print(f"{'='*70}")
        print(f"Device: {self.device}")
        print(f"Model: {self._count_params()/1e6:.1f}M parameters")
        print(f"Batch: {config.batch_size} micro × {config.grad_accum_steps} accum = {config.effective_batch_size} effective")
        print(f"Sequence length: {config.max_seq_len}")
        print(f"Tokens/step: {self._tokens_per_step:,} ({self._tokens_per_step/1e6:.2f}M per optimizer step)")
        print(f"Dataset: {config.dataset_name}")
        print(f"torch.compile: {config.use_compile} (mode={config.compile_mode})")
        if self._kernel_status is not None:
            ks = self._kernel_status
            triton_enabled = ks.get("use_triton_kernels", False)
            print(f"Triton kernels: {triton_enabled}" + (f" (v{ks.get('triton_version', 'N/A')})" if triton_enabled else ""))
            if triton_enabled:
                print(f"  ├─ fused_swiglu:  {ks.get('fused_swiglu', False)}")
                print(f"  ├─ fused_qk_norm: {ks.get('fused_qk_norm', False)}")
                print(f"  ├─ fused_rope:    {ks.get('fused_rope', False)}" + ("" if ks.get('fused_rope', False) else " (opt-in: HYDRA_ENABLE_FUSED_ROPE=1)"))
                print(f"  └─ fused_rms_norm:{ks.get('fused_rms_norm', False)}" + ("" if ks.get('fused_rms_norm', False) else " (opt-in: HYDRA_ENABLE_FUSED_RMS_NORM=1)"))
        _chunked_ce_active = config.use_chunked_ce and hasattr(self.model, "forward_hidden")
        print(f"Chunked CE: {_chunked_ce_active}" + (" (model supports forward_hidden)" if _chunked_ce_active else f" (disabled: use_chunked_ce={config.use_chunked_ce}, has forward_hidden={hasattr(self.model, 'forward_hidden')})"))
        print(f"AMP dtype: {config.dtype}")
        if self._seed_set:
            print(f"Seed: {config.seed} (reproducible training enabled)")

        if config.lr_schedule in ("wsd", "wsd_adaptive"):
            stable_steps = config.decay_start_step - config.warmup_steps
            print("LR Schedule: WSD (Warmup-Stable-Decay)")
            print(f"  Warmup: {config.warmup_steps} steps ({config.warmup_steps/config.max_steps*100:.1f}%)")
            print(f"  Stable: {stable_steps} steps ({stable_steps/config.max_steps*100:.1f}%) at LR={config.max_lr}")
            print(f"  Decay:  {config.decay_steps} steps ({config.decay_steps/config.max_steps*100:.1f}%) -> LR={config.min_lr}")
            if config.adaptive_lr:
                print(f"  Adaptive: ENABLED (patience={config.adaptive_patience}, threshold={config.adaptive_threshold:.0%})")
            if config.use_swa:
                print(f"  SWA: ENABLED (starts at {config.swa_start_pct:.0%} of training)")
        else:
            print("LR Schedule: Cosine with warmup")
            print(f"  Warmup: {config.warmup_steps} steps, Max LR: {config.max_lr}, Min LR: {config.min_lr}")
        if config.batch_filter:
            print(f"Batch Filter: ENABLED (threshold={config.batch_filter_threshold}x, max_skip={config.batch_filter_max_skip:.0%})")
        print(f"{'='*70}\n")

    def _peek_checkpoint_config(self, checkpoint_path: str) -> dict:
        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
        ckpt_config = checkpoint.get("config", {})
        result = {}
        for key, tensor in checkpoint["model"].items():
            if "cos_cached" in key:
                result["max_seq_len"] = tensor.shape[2]
                print(f"Checkpoint RoPE cache seq_len: {result['max_seq_len']}")
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
            print(
                f"Checkpoint architecture: dim={result.get('mod_mor_dim')}, "
                f"blocks={result.get('n_mor_blocks')}, "
                f"recursions={result.get('mor_recursions')}, "
                f"heads={result.get('mod_mor_n_heads')}"
            )
        return result

    def _peek_checkpoint_seq_len(self, checkpoint_path: str) -> int:
        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
        for key, tensor in checkpoint["model"].items():
            if "cos_cached" in key:
                seq_len = tensor.shape[2]
                print(f"Checkpoint RoPE cache seq_len: {seq_len}")
                return seq_len
        return self.config.max_seq_len

    def _setup_model(self) -> None:
        config = self.config
        max_needed_seq_len = config.max_seq_len
        for _, seq_len in config.seq_steps:
            max_needed_seq_len = max(max_needed_seq_len, seq_len)
        model_seq_len = max_needed_seq_len
        print(f"Creating model with max_seq_len={model_seq_len} (covers all training phases)")
        if config.architecture != "mod_mor":
            raise ValueError(f"Unsupported architecture '{config.architecture}'. Only 'mod_mor' is supported.")
        self.model = CCGQAMoDMoRModel(
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
            adaptive=config.mor_adaptive,
            tie_weights=True,
        ).to(self.device)
        self._use_mod_mor = True
        mod_status = "OFF (capacity=1.0)" if config.mod_capacity >= 1.0 else f"{config.mod_capacity:.0%} capacity"
        mor_status = "adaptive" if config.mor_adaptive else "fixed-depth (no routing)"
        print(f"MoD: {mod_status}")
        print(f"MoR: {mor_status}, {config.mor_recursions} recursions/block")
        if config.mor_adaptive:
            mor_enable_step = int(config.max_steps * config.mor_enable_pct)
            if config.mor_already_enabled:
                mor_enable_step = 0
                print("MoR RESTART MODE: Adaptive routing enabled from start (resumed after enable point)")
            remaining_steps = config.max_steps - mor_enable_step
            default_rampup = min(config.mor_rampup_steps, 2 * mor_enable_step)
            actual_rampup = min(default_rampup, remaining_steps)
            actual_rampup = max(actual_rampup, min(100, int(config.max_steps * 0.1)))
            self.model.set_mor_curriculum(enable_step=mor_enable_step, rampup_steps=actual_rampup)
            self._mor_enable_step = mor_enable_step
            if mor_enable_step > 0:
                print(f"MoR CURRICULUM: Fixed-depth until step {mor_enable_step:,} ({config.mor_enable_pct:.0%}), then {actual_rampup:,} step rampup")
        else:
            self._mor_enable_step = 0
            print("MoR CURRICULUM: Disabled (adaptive=False, running pure fixed-depth)")
        if config.gradient_checkpointing:
            if hasattr(self.model, "enable_gradient_checkpointing"):
                every_n = getattr(config, "checkpoint_every_n", 1)
                self.model.enable_gradient_checkpointing(every_n=every_n)
                if every_n == 1:
                    print("Gradient checkpointing: ENABLED (all layers, ~50% memory, ~30% overhead)")
                else:
                    print(f"Gradient checkpointing: ENABLED (every {every_n} layers, ~35% memory, ~15% overhead)")
            else:
                print("WARNING: Model doesn't support gradient checkpointing")
        if config.use_compile and self.device == "cuda":
            mode = config.compile_mode
            if mode == "max-autotune":
                mode = "max-autotune-no-cudagraphs"
            print(f"Compiling model with mode='{mode}'...")
            self.model = torch.compile(self.model, mode=mode, fullgraph=False, dynamic=False)
            print("Model compiled successfully!")

    def _setup_optimizer(self) -> None:
        config = self.config
        decay_params = []
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if "weight" in name and "norm" not in name and "embed" not in name:
                    decay_params.append(param)
                else:
                    no_decay_params.append(param)
        if config.use_8bit_adam:
            try:
                import bitsandbytes as bnb

                self.optimizer = bnb.optim.AdamW8bit(
                    [
                        {"params": decay_params, "weight_decay": config.weight_decay},
                        {"params": no_decay_params, "weight_decay": 0.0},
                    ],
                    lr=config.max_lr,
                    betas=(0.9, 0.95),
                )
                print("Using 8-bit AdamW (bitsandbytes) - ~75% optimizer memory savings")
            except ImportError:
                print("WARNING: bitsandbytes not installed, falling back to standard AdamW")
                print("Install with: pip install bitsandbytes")
                config.use_8bit_adam = False
        if not config.use_8bit_adam:
            self.optimizer = torch.optim.AdamW(
                [
                    {"params": decay_params, "weight_decay": config.weight_decay},
                    {"params": no_decay_params, "weight_decay": 0.0},
                ],
                lr=config.max_lr,
                betas=(0.9, 0.95),
                fused=True,
            )
        self._param_groups = self.optimizer.param_groups
        use_scaler = config.dtype == "float16"
        self.scaler = GradScaler("cuda", enabled=use_scaler)
        self._use_scaler = use_scaler

    def _setup_data(self) -> None:
        config = self.config
        initial_seq_len = config.seq_steps[0][1] if config.seq_steps else config.max_seq_len
        self._current_seq_len = initial_seq_len
        print(f"Loading {config.dataset_name} dataset...")
        print(f"Stepped sequence schedule: {config.seq_steps} + final @ {config.max_seq_len}")
        print(f"Starting with seq_len={initial_seq_len}")
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
        print(f"Tokens per step: {self._tokens_per_step:,}")
        print("Dataset ready!")

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        print(f"\n{'='*70}")
        print(f"RESUMING FROM CHECKPOINT: {checkpoint_path}")
        print(f"{'='*70}")
        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=self.device)
        model = self.model
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
        model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scaler.load_state_dict(checkpoint["scaler"])
        ckpt_config = checkpoint.get("config", {})
        ckpt_max_steps = ckpt_config.get("max_steps", 0)
        if ckpt_max_steps > 0 and ckpt_max_steps != self.config.max_steps:
            print(f"  ⚠️  max_steps changed: {ckpt_max_steps} → {self.config.max_steps}")
            print("  LR schedule will be recalculated for new training length")
            self._resume_lr_scale = 1.0
        else:
            try:
                ckpt_lr = float(self.optimizer.param_groups[0].get("lr", 0.0))
                sched_lr_at_resume = float(get_lr(checkpoint["step"], self.config))
                if ckpt_lr > 0.0 and sched_lr_at_resume > 0.0:
                    self._resume_lr_scale = ckpt_lr / sched_lr_at_resume
                    if abs(self._resume_lr_scale - 1.0) > 1e-6:
                        print(
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
        print(f"  Loaded step: {self._start_step}")
        print(f"  Previous best loss: {ckpt_metrics.get('best_loss', 'N/A')}")
        print(f"  Previous total tokens: {ckpt_metrics.get('total_tokens', 'N/A'):,}")
        print(f"  Will continue to step: {self.config.max_steps}")
        print(f"  Remaining steps: {self.config.max_steps - self._start_step:,}")
        if ckpt_metrics:
            self.metrics.best_loss = ckpt_metrics.get("best_loss", float("inf"))
            self.metrics.best_loss_step = ckpt_metrics.get("best_loss_step", 0)
            self.metrics.total_tokens = ckpt_metrics.get("total_tokens", 0)
        self._checkpoint_adaptive_state = None
        if "adaptive_lr_state" in checkpoint:
            self._checkpoint_adaptive_state = checkpoint["adaptive_lr_state"]
            print("  Found adaptive LR state in checkpoint (will apply after manager init)")
        print(f"{'='*70}\n")

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
        print(f"\n{'='*70}")
        print(f"SEQUENCE LENGTH TRANSITION: {self._current_seq_len} -> {seq_len}")
        print(f"{'='*70}")
        model = self.model
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
        if hasattr(model, "resize_rope_cache"):
            model.resize_rope_cache(seq_len)
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
        print(f"New tokens/step: {self._tokens_per_step:,}")
        print(f"{'='*70}\n")

    def _get_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        batch = self.train_loader.get_batch()
        return (
            batch["input_ids"].to(self.device, non_blocking=True),
            batch["labels"].to(self.device, non_blocking=True),
        )

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

        if self._use_progress_aware_lr:
            self._adaptive_lr = ProgressAwareLRManager(config, start_step=start_step)
            if hasattr(self, "_checkpoint_adaptive_state") and self._checkpoint_adaptive_state:
                self._adaptive_lr.load_state(self._checkpoint_adaptive_state)

        if start_step > 0:
            print(f"Resuming training from step {start_step}...")
        else:
            print("Starting training...")
        print("-" * 70)

        model.train()
        metrics.start_time = time.time()
        step = start_step
        accum_loss = 0.0
        use_scaler = self._use_scaler
        use_mod_mor = self._use_mod_mor

        self._last_ce_loss = 0.0
        self._last_aux_loss = 0.0
        self._last_ponder_loss = 0.0
        self._last_advantage_loss = 0.0

        eval_batches = 25
        eval_dataset = config.dataset_name
        if config.dataset_name.startswith("pretrain_") or config.dataset_name in ["sft_chat"]:
            eval_dataset = "wikitext2"
            print(f"📊 Using wikitext2 for evaluation (mixed dataset: {config.dataset_name})")
        eval_loader = create_universal_loader(
            dataset=eval_dataset,
            batch_size=config.batch_size,
            seq_len=self._current_seq_len,
            vocab_size=config.vocab_size,
            device="cpu",
            tokenizer_name=config.tokenizer_name,
        )
        fixed_eval_batches = [eval_loader.get_batch() for _ in range(eval_batches)]

        if start_step >= max_steps:
            print(f"⚠️  No steps to run: start_step={start_step} >= max_steps={max_steps}")
            print("   Increase --max_steps to continue training from this checkpoint.")
            metrics.end_time = time.time()
            return metrics

        while step < max_steps:
            step_start = time.time()
            optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0
            micro_diag: List[dict] = []
            collect_micro_diag = os.environ.get("HYDRA_ENABLE_MICRO_DIAG", "0") == "1"
            if collect_micro_diag and step == start_step:
                print("\n" + "⚠" * 35)
                print("  WARNING: HYDRA_ENABLE_MICRO_DIAG=1 is active.")
                print("  This adds ~20 .item() calls per micro-batch, causing")
                print("  significant CPU-GPU sync overhead. Use for debugging only.")
                print("⚠" * 35 + "\n")
            next_step = step + 1
            track_loss_scalars = collect_micro_diag or ((log_interval > 0 and next_step % log_interval == 0) or (next_step % 500 == 0))

            grad_spike_threshold = 1e6
            grad_spike_lr_factor = 0.1
            grad_spike_topk = 32
            grad_spike_reset_moments = True

            if use_mod_mor and hasattr(model, "set_global_step"):
                base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
                if hasattr(base_model, "set_global_step"):
                    base_model.set_global_step(step)

            for micro_step in range(grad_accum):
                x, y = self._get_batch()
                if self.device == "cuda" and self.config.use_compile:
                    torch.compiler.cudagraph_mark_step_begin()
                with autocast(device, dtype=dtype):
                    if use_mod_mor:
                        logits = None
                        if self.config.use_chunked_ce and hasattr(model, "forward_hidden_with_losses"):
                            hidden, aux_losses = model.forward_hidden_with_losses(x)
                            base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
                            weight = base_model.output.weight
                            ce_loss = fused_chunked_cross_entropy(
                                hidden,
                                weight,
                                y,
                                ignore_index=-100,
                                chunk_size=self.config.chunked_ce_size,
                            )
                        else:
                            logits, aux_losses = model(x, return_losses=True)
                            ce_loss = F.cross_entropy(
                                logits.view(-1, logits.size(-1)),
                                y.view(-1),
                                ignore_index=-100,
                            )
                        aux_loss = aux_losses.get("aux_loss", 0.0)
                        ponder_loss = aux_losses.get("ponder_loss", 0.0)
                        advantage_loss = 0.0
                        base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
                        if logits is not None and hasattr(base_model, "compute_advantage_loss"):
                            advantage_loss = base_model.compute_advantage_loss(logits, y, ignore_index=-100)
                        if hasattr(aux_loss, "clamp"):
                            aux_loss = aux_loss.clamp(max=100.0)
                        if hasattr(ponder_loss, "clamp"):
                            ponder_loss = ponder_loss.clamp(max=100.0)
                        if hasattr(advantage_loss, "clamp"):
                            advantage_loss = advantage_loss.clamp(max=10.0)
                        loss = ce_loss + self.config.aux_scale * aux_loss + self.config.ponder_scale * ponder_loss + advantage_loss
                        if track_loss_scalars:
                            self._last_ce_loss = ce_loss.item() if hasattr(ce_loss, "item") else float(ce_loss)
                            self._last_aux_loss = aux_loss.item() if hasattr(aux_loss, "item") else float(aux_loss)
                            self._last_ponder_loss = ponder_loss.item() if hasattr(ponder_loss, "item") else float(ponder_loss)
                            self._last_advantage_loss = advantage_loss.item() if hasattr(advantage_loss, "item") else float(advantage_loss)
                    else:
                        if self.config.use_chunked_ce and hasattr(model, "forward_hidden"):
                            hidden = model.forward_hidden(x)
                            base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
                            weight = base_model.output.weight
                            loss = fused_chunked_cross_entropy(
                                hidden,
                                weight,
                                y,
                                ignore_index=-100,
                                chunk_size=self.config.chunked_ce_size,
                            )
                            logits = None
                        else:
                            logits = model(x)
                            loss = F.cross_entropy(
                                logits.view(-1, logits.size(-1)),
                                y.view(-1),
                                ignore_index=-100,
                            )
                        if track_loss_scalars:
                            self._last_ce_loss = loss.item() if hasattr(loss, "item") else float(loss)
                            self._last_aux_loss = 0.0
                            self._last_ponder_loss = 0.0
                if collect_micro_diag:
                    try:
                        with torch.no_grad():
                            y_flat = y.view(-1)
                            y_is_ignore = y_flat == -100
                            y_valid = ~y_is_ignore
                            y_oob = (y_valid & ((y_flat < 0) | (y_flat >= vocab_size))).sum().item()
                            y_ignore = y_is_ignore.sum().item()
                            y_min = int(y_flat.min().item()) if y_flat.numel() else 0
                            y_max = int(y_flat.max().item()) if y_flat.numel() else 0
                            x_flat = x.view(-1)
                            x_oob = ((x_flat < 0) | (x_flat >= vocab_size)).sum().item()
                            x_min = int(x_flat.min().item()) if x_flat.numel() else 0
                            x_max = int(x_flat.max().item()) if x_flat.numel() else 0
                            if logits is None:
                                logits_isfinite = True
                                logits_absmax = 0.0
                                logits_mean = 0.0
                                logits_std = 0.0
                            else:
                                logits_f = logits.detach()
                                logits_f32 = logits_f.float()
                                logits_isfinite = torch.isfinite(logits_f32).all().item()
                                logits_absmax = logits_f32.abs().max().item() if logits_f32.numel() else 0.0
                                logits_mean = logits_f32.mean().item() if logits_f32.numel() else 0.0
                                logits_std = logits_f32.std(unbiased=False).item() if logits_f32.numel() else 0.0
                            micro_diag.append(
                                {
                                    "micro_step": micro_step,
                                    "loss": float(loss.item()) if hasattr(loss, "item") else float(loss),
                                    "accum_enabled": True,
                                    "x_min": x_min,
                                    "x_max": x_max,
                                    "x_oob": int(x_oob),
                                    "y_min": y_min,
                                    "y_max": y_max,
                                    "y_oob": int(y_oob),
                                    "y_ignore": int(y_ignore),
                                    "y_valid": int(y_flat.numel() - y_ignore),
                                    "logits_isfinite": bool(logits_isfinite),
                                    "logits_absmax": float(logits_absmax),
                                    "logits_mean": float(logits_mean),
                                    "logits_std": float(logits_std),
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
                            print(
                                f"  [BatchFilter] Skipped batch: {skip_reason}, "
                                f"total skipped: {stats['n_skipped']}/{stats['n_total']} "
                                f"({stats['skip_ratio']*100:.1f}%)"
                            )
                        continue
                loss_val = loss.item()
                if not math.isfinite(loss_val):
                    print(f"  ⚠️  Step {step}: NaN/Inf loss detected, skipping batch")
                    if micro_diag:
                        micro_diag[-1]["accum_enabled"] = False
                        micro_diag[-1]["skip_reason"] = "non_finite_loss"
                    continue
                scaled_loss = loss * loss_scale
                if use_scaler:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                accum_loss += loss_val * loss_scale

            if use_scaler:
                scaler.unscale_(optimizer)
            base = model._orig_mod if hasattr(model, "_orig_mod") else model
            pre_clip_norm_t = torch.nn.utils.clip_grad_norm_(base.parameters(), grad_clip)
            pre_clip_norm = float(pre_clip_norm_t) if torch.is_tensor(pre_clip_norm_t) else float(pre_clip_norm_t)
            grad_norm = min(pre_clip_norm, float(grad_clip)) if math.isfinite(pre_clip_norm) else float("nan")
            grad_info_pre_clip = None
            spike_detected = math.isfinite(pre_clip_norm) and (pre_clip_norm > grad_spike_threshold)
            nonfinite_grads = not math.isfinite(pre_clip_norm)
            if self._adaptive_lr is not None:
                if step % 100 == 0:
                    self._adaptive_lr.update(step, accum_loss)
                lr = self._adaptive_lr.get_lr(step)
            else:
                lr = get_lr(step, config)
            lr = lr * getattr(self, "_resume_lr_scale", 1.0)
            lr_effective = lr
            if spike_detected:
                lr_effective = lr * grad_spike_lr_factor
            if nonfinite_grads or spike_detected:
                grad_info_pre_clip = []
                try:
                    for name, param in base.named_parameters():
                        if param.grad is None:
                            continue
                        g = param.grad.detach()
                        grad_info_pre_clip.append(
                            (
                                name,
                                float(g.float().norm().item()),
                                float(g.float().abs().max().item()),
                                bool(torch.isnan(g).any().item()),
                                bool(torch.isinf(g).any().item()),
                            )
                        )
                except Exception:
                    grad_info_pre_clip = []
                clip_coef = grad_clip / (pre_clip_norm + 1e-12)
                clip_scale = min(1.0, clip_coef) if math.isfinite(clip_coef) else 0.0
                print(f"\n{'='*60}")
                print(f"  🔍 GRADIENT EXPLOSION DIAGNOSTIC - Step {step}")
                print(f"{'='*60}")
                print(f"  Pre-clip grad_norm:  {pre_clip_norm:.2e}")
                print(f"  Post-clip grad_norm: {grad_norm:.2e}")
                print(f"  Clip coefficient:    {clip_coef:.2e}")
                print(f"  Clip scale applied:  {clip_scale:.2e}")
                print(f"  LR (scheduled):      {lr:.2e}")
                if spike_detected:
                    print(f"  LR (effective):      {lr_effective:.2e} (cooldown x{grad_spike_lr_factor})")
                print(f"  Accumulated loss this step: {accum_loss:.4f}")
                ce = getattr(self, "_last_ce_loss", 0.0)
                aux = getattr(self, "_last_aux_loss", 0.0)
                ponder = getattr(self, "_last_ponder_loss", 0.0)
                print(f"  Last micro-batch losses: CE={ce:.4f}, aux={aux:.4f}, ponder={ponder:.4f}")
                if micro_diag:
                    print("\n  Micro-batch sanity (this step):")
                    for md in micro_diag:
                        skip_note = "" if md.get("accum_enabled", True) else f" SKIPPED({md.get('skip_reason', 'unknown')})"
                        print(
                            f"    micro={md.get('micro_step')} loss={md.get('loss', 0.0):.4f}{skip_note} | "
                            f"x[min,max]=[{md.get('x_min')},{md.get('x_max')}] x_oob={md.get('x_oob')} | "
                            f"y[min,max]=[{md.get('y_min')},{md.get('y_max')}] y_oob={md.get('y_oob')} "
                            f"valid={md.get('y_valid')} ignore={md.get('y_ignore')} | "
                            f"logits finite={md.get('logits_isfinite')} absmax={md.get('logits_absmax', 0.0):.2e} "
                            f"mean={md.get('logits_mean', 0.0):.2e} std={md.get('logits_std', 0.0):.2e}"
                        )
                def sort_key(x):
                    if x[3] or x[4]:
                        return float("inf")
                    return x[1]
                grad_info_pre_clip.sort(key=sort_key, reverse=True)
                print("  Top 15 gradients by norm (post-clip; est pre-clip in parentheses):")
                for name, g_norm, g_max, has_nan, has_inf in grad_info_pre_clip[:15]:
                    flag = ""
                    if has_nan:
                        flag += " [NaN!]"
                    if has_inf:
                        flag += " [Inf!]"
                    if clip_scale > 0.0:
                        g_norm_pre = g_norm / clip_scale
                        g_max_pre = g_max / clip_scale
                    else:
                        g_norm_pre = g_norm
                        g_max_pre = g_max
                    print(
                        f"    {name}: norm={g_norm:.2e} (pre~{g_norm_pre:.2e}), "
                        f"max={g_max:.2e} (pre~{g_max_pre:.2e}){flag}"
                    )
                try:
                    top_names = [n for (n, *_rest) in grad_info_pre_clip[:15]]
                    eps = 1e-12
                    ratios = []
                    with torch.no_grad():
                        for name, p in base.named_parameters():
                            if name not in top_names or p.grad is None:
                                continue
                            w = p.detach()
                            g = p.grad.detach()
                            w_norm = w.float().norm().item()
                            g_norm_post = g.float().norm().item()
                            w_absmax = w.float().abs().max().item()
                            g_absmax_post = g.float().abs().max().item()
                            ratio_l2 = (lr_effective * g_norm_post) / (w_norm + eps)
                            ratio_max = (lr_effective * g_absmax_post) / (w_absmax + eps)
                            ratios.append((name, ratio_l2, ratio_max, w_norm, g_norm_post))
                    ratios.sort(key=lambda t: t[1], reverse=True)
                    if ratios:
                        print("\n  Update/weight ratios (clipped grads, effective LR):")
                        for name, r_l2, r_max, w_norm, g_norm_post in ratios[:8]:
                            print(
                                f"    {name}: (lr*||g||/||w||)={r_l2:.2e}, (lr*gmax/wmax)={r_max:.2e} | "
                                f"||w||={w_norm:.2e}, ||g||_post={g_norm_post:.2e}"
                            )
                except Exception:
                    pass
                if hasattr(base, "layers"):
                    print("\n  Router diagnostics (first 5 layers):")
                    for i, layer in enumerate(base.layers[:5]):
                        if hasattr(layer, "router"):
                            w = layer.router.weight
                            b = layer.router.bias if hasattr(layer.router, "bias") and layer.router.bias is not None else None
                            print(f"    Layer {i} router.weight: mean={w.mean():.4f}, std={w.std():.4f}, max={w.abs().max():.4f}")
                            if b is not None:
                                print(f"    Layer {i} router.bias: {b.item():.4f}")
                            if w.grad is not None:
                                print(f"    Layer {i} router.weight.grad: norm={w.grad.norm():.2e}, max={w.grad.abs().max():.2e}")
                print(f"{'='*60}\n")
                if nonfinite_grads:
                    optimizer.zero_grad(set_to_none=True)
                    if use_scaler:
                        scaler.update()
                    print(f"  ⚠️  Step {step}: Skipping update - non-finite gradients")
                    step += 1
                    continue
                if spike_detected and grad_spike_reset_moments:
                    try:
                        offenders = sorted(
                            grad_info_pre_clip,
                            key=lambda t: float("inf") if (t[3] or t[4]) else t[1],
                            reverse=True,
                        )[:grad_spike_topk]
                        offender_names = {n for (n, *_rest) in offenders}
                        for name, p in base.named_parameters():
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
                        print(f"  🔧 Spike response: reset Adam moments for top {len(offenders)} params")
                    except Exception as e:
                        print(f"  ⚠️  Spike response: moment reset failed ({e})")
                if (spike_detected or nonfinite_grads) and getattr(self.config, "halt_on_spike", False):
                    try:
                        halt_step_time = max(1e-9, time.time() - step_start)
                        halt_tps = tokens_per_step / halt_step_time
                        metrics.update(step, accum_loss, lr_effective, grad_norm, halt_tps, halt_step_time)
                        metrics.total_tokens += tokens_per_step
                        metrics.final_loss = accum_loss
                        self._save_checkpoint(step)
                    except Exception as e:
                        print(f"  ⚠️  Halt-on-spike: failed to record/save state ({e})")
                    print(f"  🛑 Halt-on-spike enabled: stopping at step {step}")
                    metrics.end_time = time.time()
                    return metrics
            if spike_detected:
                lr = lr_effective
                print(f"  🔻 Spike response: LR cooldown applied (factor={grad_spike_lr_factor})")
            for pg in param_groups:
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
                    print(f"Initial loss: {accum_loss:.4f}")
                else:
                    print(f"Resume loss: {accum_loss:.4f} (previous best: {metrics.best_loss:.4f})")
            prev_best = metrics.best_loss
            metrics.update(step, accum_loss, lr, grad_norm, tps, step_time)
            metrics.total_tokens += tokens_per_step
            if step >= 1000 and step % 500 == 0:
                improvement = (prev_best - metrics.best_loss) / prev_best if prev_best < float("inf") else 0.0
                if improvement > 0.20:
                    self._save_checkpoint(step, best=True)
                elif step % 1000 == 0 and metrics.best_loss < prev_best:
                    self._save_checkpoint(step, best=True)
            if eval_interval > 0 and step % eval_interval == 0:
                base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
                base_model.eval()
                eval_loss = 0.0
                with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=dtype):
                    for b in fixed_eval_batches:
                        x_eval = b["input_ids"].to(device, non_blocking=True)
                        y_eval = b["labels"].to(device, non_blocking=True)
                        if use_mod_mor:
                            if self.config.use_chunked_ce and hasattr(base_model, "forward_hidden"):
                                hidden = base_model.forward_hidden(x_eval)
                                weight = base_model.output.weight
                                loss_eval = fused_chunked_cross_entropy(
                                    hidden,
                                    weight,
                                    y_eval,
                                    ignore_index=-100,
                                    chunk_size=self.config.chunked_ce_size,
                                )
                                eval_loss += loss_eval.item()
                                continue
                            logits, _aux = base_model(x_eval, return_losses=True)
                        else:
                            logits = base_model(x_eval)
                        loss_eval = F.cross_entropy(
                            logits.view(-1, logits.size(-1)),
                            y_eval.view(-1),
                            ignore_index=-100,
                        )
                        eval_loss += loss_eval.item()
                base_model.train()
                eval_loss /= len(fixed_eval_batches)
                print(f"[EVAL] step={step}  eval_loss={eval_loss:.4f}  train_loss={accum_loss:.4f}")
            if step % log_interval == 0:
                elapsed = time.time() - metrics.start_time
                steps_this_session = step - start_step
                tokens_this_session = steps_this_session * tokens_per_step
                avg_tps = tokens_this_session / elapsed if elapsed > 0 else 0
                steps_per_sec = steps_this_session / elapsed if elapsed > 0 else 0
                print(
                    f"Step {step:5d}/{max_steps} | "
                    f"Loss: {accum_loss:.4f} (EMA: {metrics.ema_loss:.4f}) | "
                    f"LR: {lr:.2e} | "
                    f"Grad: {grad_norm:.2f} | "
                    f"{tps/1000:.1f}K tok/s | "
                    f"Avg: {avg_tps/1000:.1f}K tok/s ({steps_per_sec:.2f} steps/s)"
                )
                if use_mod_mor and step % 100 == 0:
                    self._log_layer_diagnostics(step, accum_loss, lr, grad_norm)
            if step % save_interval == 0:
                self._save_checkpoint(step)
                if self._should_early_stop(accum_loss if accum_loss > 0 else metrics.losses[-1], step, self._count_params()):
                    print(f"\n⚠️  EARLY STOPPING at step {step}")
                    print(f"   Loss has increased for {config.early_stop_patience} consecutive checkpoints")
                    print(f"   Checkpoint losses: {self._checkpoint_losses[-4:]}")
                    break
            accum_loss = 0.0

        metrics.end_time = time.time()
        metrics.final_loss = metrics.losses[-1] if metrics.losses else 0.0

        print("-" * 70)
        print("Training complete!")

        if self._adaptive_lr is not None and self._adaptive_lr.swa_n > 0:
            base_model = self.model
            if hasattr(base_model, "_orig_mod"):
                base_model = base_model._orig_mod
            self._adaptive_lr.apply_swa(base_model)

        self._save_checkpoint(step, final=True)
        self._generate_report()
        self._save_diagnostics()
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
                "total": loss,
                "ce": self._last_ce_loss,
                "aux": self._last_aux_loss,
                "ponder": self._last_ponder_loss,
                "advantage": self._last_advantage_loss,
            },
            "lr": lr,
            "grad_norm": grad_norm,
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
            deviation = abs(probs_mean - target)
            status = "OK"
            if probs_mean < 0.1:
                status = "COLLAPSED_LOW"
                mod_issues.append(f"Layer {layer_idx}: probs={probs_mean:.3f} (collapsed to skip)")
            elif probs_mean > 0.9:
                status = "COLLAPSED_HIGH"
                mod_issues.append(f"Layer {layer_idx}: probs={probs_mean:.3f} (collapsed to process all)")
            elif deviation > 0.2:
                status = "DRIFTING"
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
                    "status": status,
                }
            )
        mor_stats = stats.get("mor_layers", [])
        mor_issues = []
        for layer_stat in mor_stats:
            layer_idx = layer_stat.get("layer", -1)
            avg_depth = layer_stat.get("avg_depth", 0)
            expected_depth = layer_stat.get("expected_avg_depth", 1.5)
            router_probs = layer_stat.get("router_probs_mean", 0.5)
            depth_hist = layer_stat.get("depth_histogram", [])
            status = "OK"
            if router_probs < 0.1:
                status = "ROUTER_COLLAPSED"
                mor_issues.append(f"Layer {layer_idx}: router_probs={router_probs:.3f} (always early exit)")
            elif router_probs > 0.9:
                status = "ROUTER_SATURATED"
                mor_issues.append(f"Layer {layer_idx}: router_probs={router_probs:.3f} (always max depth)")
            elif depth_hist and max(depth_hist) > 0.9 * sum(depth_hist):
                status = "DEPTH_COLLAPSED"
                mor_issues.append(f"Layer {layer_idx}: all tokens at same depth")
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
        mor_phase = "N/A"
        if hasattr(model, "get_mor_status"):
            mor_status = model.get_mor_status()
            phase = mor_status.get("phase", "unknown")
            if phase == "fixed-depth":
                mor_phase = f"FIXED ({mor_status.get('rampup_progress', 0)*100:.0f}%)"
            elif phase == "rampup":
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
        if mod_issues or mor_issues:
            print(f"  ⚠️  Issues: MoD={len(mod_issues)}, MoR={len(mor_issues)}")
        if step % 500 == 0:
            adaptive_info = ""
            if self._adaptive_lr is not None:
                state = self._adaptive_lr.get_state()
                ema_s = state.get("loss_ema_short", 0)
                ema_l = state.get("loss_ema_long", 0)
                trend = "↑" if ema_s > ema_l * 1.02 else ("↓" if ema_s < ema_l * 0.98 else "→")
                adaptive_info = f" | EMA: {ema_s:.3f}/{ema_l:.3f} {trend}"
            adv_str = f" adv={self._last_advantage_loss:.4f}" if self._last_advantage_loss != 0 else ""
            print(
                f"  [DIAG] MoD:{mod_mode} save={mod_savings:.0f}% | MoR:{mor_phase} d={mor_depth:.2f} [{depth_dist}%] | "
                f"CE={self._last_ce_loss:.3f} aux={self._last_aux_loss:.4f} ponder={self._last_ponder_loss:.3f}{adv_str}{adaptive_info}"
            )
            self._save_diagnostics()

    def _save_diagnostics(self) -> None:
        if not self._diagnostics_data:
            return
        ckpt_dir = Path(self.config.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        diag_path = ckpt_dir / "training_diagnostics.json"
        try:
            with open(diag_path, "w") as f:
                json.dump(self._diagnostics_data, f, indent=2)
            print(f"📊 Diagnostics saved to {diag_path}")
        except Exception as e:
            print(f"⚠️  Failed to save diagnostics: {e}")

    def _save_checkpoint(self, step: int, final: bool = False, best: bool = False) -> None:
        config = self.config
        ckpt_dir = Path(config.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        if best:
            suffix = "best"
        elif final:
            suffix = "final"
        else:
            suffix = f"step_{step}"
        ckpt_path = ckpt_dir / f"hydra_100m_{suffix}.pt"
        if (not best) and (not final) and ckpt_path.exists():
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            ckpt_path = ckpt_dir / f"hydra_100m_{suffix}_{ts}.pt"
        model = self.model
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
        checkpoint = {
            "step": step,
            "model": model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "config": asdict(self.config),
            "metrics": {
                "initial_loss": self.metrics.initial_loss,
                "current_loss": self.metrics.losses[-1] if self.metrics.losses else 0,
                "best_loss": self.metrics.best_loss,
                "best_loss_step": self.metrics.best_loss_step,
                "total_tokens": self.metrics.total_tokens,
            },
        }
        if self._adaptive_lr is not None:
            checkpoint["adaptive_lr_state"] = self._adaptive_lr.get_state()
            if hasattr(self._adaptive_lr, "_swa_model") and self._adaptive_lr._swa_model is not None:
                checkpoint["swa_model"] = self._adaptive_lr._swa_model.state_dict()
        torch.save(checkpoint, ckpt_path)
        if best:
            print(f"🏆 New best! Loss: {self.metrics.best_loss:.4f} → {ckpt_path}")
        elif final:
            print(f"Checkpoint saved: {ckpt_path}")
        else:
            print(f"Checkpoint saved: {ckpt_path}")
            self._checkpoint_history.append(ckpt_path)
            self._cleanup_old_checkpoints()

    def _cleanup_old_checkpoints(self) -> None:
        max_ckpts = self.config.max_checkpoints
        while len(self._checkpoint_history) > max_ckpts:
            old_ckpt = self._checkpoint_history.pop(0)
            if old_ckpt.exists():
                old_ckpt.unlink()
                print(f"   Removed old checkpoint: {old_ckpt.name}")

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
            print(
                f"   ⚠️  Loss increased: {prev_loss:.4f} → {current_loss:.4f} (+{relative_increase*100:.1f}%) "
                f"[{self._early_stop_counter}/{config.early_stop_patience}]"
            )
            print(f"       (Chinchilla progress: {progress*100:.1f}%, early stop active)")
        else:
            self._early_stop_counter = 0
        return self._early_stop_counter >= config.early_stop_patience

    def _generate_report(self) -> None:
        config = self.config
        metrics = self.metrics
        report_dir = Path(config.report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        training_time = metrics.end_time - metrics.start_time
        avg_tps = metrics.total_tokens / training_time if training_time > 0 else 0
        loss_reduction = (
            (metrics.initial_loss - metrics.final_loss) / metrics.initial_loss * 100
            if metrics.initial_loss > 0
            else 0
        )
        losses = metrics.losses
        if losses:
            avg_loss = sum(losses) / len(losses)
            min_loss = min(losses)
            max_loss = max(losses)
            n = len(losses)
            loss_at_25 = losses[n // 4] if n > 4 else losses[-1]
            loss_at_50 = losses[n // 2] if n > 2 else losses[-1]
            loss_at_75 = losses[3 * n // 4] if n > 4 else losses[-1]
        else:
            avg_loss = min_loss = max_loss = 0
            loss_at_25 = loss_at_50 = loss_at_75 = 0
        tps_list = metrics.tokens_per_sec
        if tps_list:
            warmup_skip = min(10, len(tps_list) // 10)
            steady_tps = tps_list[warmup_skip:] if len(tps_list) > warmup_skip else tps_list
            avg_tps_steady = sum(steady_tps) / len(steady_tps) if steady_tps else 0
            peak_tps = max(tps_list)
        else:
            avg_tps_steady = peak_tps = 0
        report = {
            "metadata": {
                "timestamp": timestamp,
                "model": "HYDRA 100M",
                "dataset": config.dataset_name,
                "device": self.device,
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
            },
            "configuration": asdict(config),
            "training_summary": {
                "total_steps": config.max_steps,
                "total_tokens": metrics.total_tokens,
                "training_time_seconds": training_time,
                "training_time_formatted": self._format_time(training_time),
            },
            "loss_analysis": {
                "initial_loss": metrics.initial_loss,
                "final_loss": metrics.final_loss,
                "best_loss": metrics.best_loss,
                "best_loss_step": metrics.best_loss_step,
                "loss_reduction_percent": loss_reduction,
                "average_loss": avg_loss,
                "min_loss": min_loss,
                "max_loss": max_loss,
                "loss_at_25_percent": loss_at_25,
                "loss_at_50_percent": loss_at_50,
                "loss_at_75_percent": loss_at_75,
            },
            "performance": {
                "average_tokens_per_second": avg_tps,
                "average_tokens_per_second_steady": avg_tps_steady,
                "peak_tokens_per_second": peak_tps,
                "average_step_time_ms": sum(metrics.step_times) / len(metrics.step_times) * 1000
                if metrics.step_times
                else 0,
            },
            "gradient_analysis": {
                "average_grad_norm": sum(metrics.grad_norms) / len(metrics.grad_norms) if metrics.grad_norms else 0,
                "max_grad_norm": max(metrics.grad_norms) if metrics.grad_norms else 0,
                "min_grad_norm": min(metrics.grad_norms) if metrics.grad_norms else 0,
            },
            "model_assessment": self._assess_model_performance(metrics),
            "training_assessment": self._assess_training_quality(metrics),
            "raw_metrics": metrics.to_dict(),
        }
        report_path = report_dir / f"training_report_{timestamp}.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print("\n" + "=" * 70)
        print("TRAINING REPORT")
        print("=" * 70)
        print(f"\n📊 Loss Analysis:")
        print(f"   Initial: {metrics.initial_loss:.4f}")
        print(f"   Final:   {metrics.final_loss:.4f}")
        print(f"   Best:    {metrics.best_loss:.4f} (step {metrics.best_loss_step})")
        print(f"   Reduction: {loss_reduction:.1f}%")
        print(f"\n⚡ Performance:")
        print(f"   Training time: {self._format_time(training_time)}")
        print(f"   Total tokens: {metrics.total_tokens:,}")
        print(f"   Avg throughput: {avg_tps/1000:.1f}K tok/s")
        print(f"   Peak throughput: {peak_tps/1000:.1f}K tok/s")
        print(f"\n📈 Model Assessment:")
        for key, value in report["model_assessment"].items():
            print(f"   {key}: {value}")
        print(f"\n✅ Training Assessment:")
        for key, value in report["training_assessment"].items():
            print(f"   {key}: {value}")
        print(f"\n📁 Report saved: {report_path}")
        print("=" * 70)

    def _assess_model_performance(self, metrics: TrainingMetrics) -> Dict[str, str]:
        assessment: Dict[str, str] = {}
        reduction = (
            (metrics.initial_loss - metrics.final_loss) / metrics.initial_loss * 100
            if metrics.initial_loss > 0
            else 0
        )
        if reduction >= 60:
            assessment["learning_quality"] = "Excellent - Strong convergence"
        elif reduction >= 40:
            assessment["learning_quality"] = "Good - Solid learning"
        elif reduction >= 20:
            assessment["learning_quality"] = "Fair - Moderate progress"
        else:
            assessment["learning_quality"] = "Poor - May need more steps or tuning"
        random_baseline = math.log(self.config.vocab_size)
        final_vs_random = (random_baseline - metrics.final_loss) / random_baseline * 100
        if metrics.final_loss < 5.0:
            assessment["vs_random_baseline"] = f"{final_vs_random:.1f}% better - Approaching usable"
        elif metrics.final_loss < 7.0:
            assessment["vs_random_baseline"] = f"{final_vs_random:.1f}% better - Learning patterns"
        else:
            assessment["vs_random_baseline"] = f"{final_vs_random:.1f}% better - Early training"
        if len(metrics.losses) >= 10:
            last_10 = metrics.losses[-10:]
            trend = (last_10[0] - last_10[-1]) / last_10[0] * 100 if last_10[0] > 0 else 0
            if trend > 5:
                assessment["convergence_trend"] = "Still improving - Continue training"
            elif trend > 0:
                assessment["convergence_trend"] = "Slowing down - Near plateau"
            else:
                assessment["convergence_trend"] = "Plateaued - Consider LR adjustment"
        return assessment

    def _assess_training_quality(self, metrics: TrainingMetrics) -> Dict[str, str]:
        assessment: Dict[str, str] = {}
        if metrics.grad_norms:
            avg_grad = sum(metrics.grad_norms) / len(metrics.grad_norms)
            max_grad = max(metrics.grad_norms)
            if max_grad <= 1.0:
                assessment["gradient_stability"] = "Excellent - Well controlled"
            elif max_grad <= 5.0:
                assessment["gradient_stability"] = "Good - Occasional spikes"
            else:
                assessment["gradient_stability"] = f"Warning - Max grad {max_grad:.1f}"
        if metrics.tokens_per_sec:
            avg_tps = sum(metrics.tokens_per_sec) / len(metrics.tokens_per_sec)
            if avg_tps >= 30000:
                assessment["throughput"] = f"Excellent - {avg_tps/1000:.1f}K tok/s"
            elif avg_tps >= 15000:
                assessment["throughput"] = f"Good - {avg_tps/1000:.1f}K tok/s"
            else:
                assessment["throughput"] = f"Moderate - {avg_tps/1000:.1f}K tok/s"
        if len(metrics.losses) >= 20:
            loss_changes = [abs(metrics.losses[i] - metrics.losses[i - 1]) for i in range(1, len(metrics.losses))]
            avg_change = sum(loss_changes) / len(loss_changes)
            if avg_change < 0.2:
                assessment["loss_stability"] = "Very smooth training"
            elif avg_change < 0.5:
                assessment["loss_stability"] = "Normal variance"
            else:
                assessment["loss_stability"] = "High variance - May need lower LR"
        return assessment

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
