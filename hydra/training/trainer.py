from __future__ import annotations

import math
import os
import signal
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
import torch.profiler

from hydra.model.framework import HydraModel
from hydra.data.universal_data_loader import DATASET_CONFIGS, create_universal_loader
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
    compute_token_losses_from_hidden,
    evaluate_fixed_batches,
    eval_sanity_check_on_train_batch,
    maybe_run_fixed_eval,
    resolve_micro_diag_tensors,
    update_scalar_ema,
)
from .gradients import skip_update_for_nonfinite_gradients
from . import spike_diagnostics as _spike
from .halt import HaltController
from .policy import SeqLenPolicy
from .curriculum import CurriculumController
from . import step_diagnostics as _step_diag
from . import db as _db
from .safe_optimizations import SafeOptimizations, OptimizationConfig
from .pretest_hook import PretestHook

_RUNTIME_STATUS = configure_runtime()


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
        "_last_moe_aux_loss",
        "_last_pre_clip_norm",
        "_ce_ema",
        "_adaptive_lr",
        "_use_progress_aware_lr",
        "_use_per_component_lr",
        "_expert_lr_scale",
        "_router_lr_scale",
        "_moe_rewarmup_start_step",
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
        "_curriculum_controller",
        "_mor_advantage_nudge_until_step",
        "_mor_advantage_nudge_cooldown_until_step",
        "_mor_advantage_nudge_active",
        "_last_moe_divergence",
        "_moe_domain_expert_map",
        "_last_batch_source_name",
        "_source_name_counts",
        "_source_name_counts_total",
        "_grad_norm_ema",  # For dynamic gradient clipping
        "_safe_opts",  # SafeOptimizations wrapper for experimental optimizations
        "_pretest_hook",  # Hook for automatic pretest runs on config/checkpoint changes
    )

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.logger = HydraLogger(config)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.metrics = TrainingMetrics()

        # Domain -> expert mapping (for InterleavedDataLoader batches).
        self._moe_domain_expert_map = {}
        self._last_batch_source_name = None
        self._source_name_counts: dict[str, int] = {}
        self._source_name_counts_total: dict[str, int] = {}
        try:
            raw = str(getattr(config, "moe_domain_expert_map", "") or "").strip()
            if raw:
                for part in raw.split(","):
                    part = part.strip()
                    if not part:
                        continue
                    k, v = part.split(":", 1)
                    self._moe_domain_expert_map[k.strip()] = int(v.strip())
        except Exception:
            self._moe_domain_expert_map = {}

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

        # SeqLenPolicy owns all HYDRA_SEQ_LEN_* env parsing and policy matching
        self._seq_len_policy = SeqLenPolicy.from_env()

        if self._seq_len_policy.is_active:
            try:
                initial_seq_len = config.seq_steps[0][1] if config.seq_steps else config.max_seq_len
                patch = self._seq_len_policy.get_patch(int(initial_seq_len))
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
        self._last_moe_aux_loss: float = 0.0
        self._last_moe_divergence: float = 0.0  # For tracking divergence trend
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

        # Initialize SafeOptimizations with config-based settings
        self._safe_opts = self._setup_safe_optimizations(config)

        # Initialize pretest hook for automatic pretest runs on config/checkpoint changes
        self._pretest_hook = PretestHook(
            log_dir=str(Path(config.checkpoint_dir) / "pretest_logs"),
            cache_results=True,
            verbose=True,
            logger=self.logger,
        )

        self._checkpoint_seq_len = None
        self._checkpoint_config = {}
        self._checkpoint_lr = None
        self._resume_lr_scale: float = 1.0
        self._resume_rewarmup_steps: int = 0
        self._resume_rewarmup_start_step: int = 0
        self._resume_state_incomplete: bool = False
        self._prev_run_best_loss: float = float("inf")
        self._resume_ce_ema: float = 0.0

        # MoE per-component LR state
        self._use_per_component_lr: bool = False
        self._expert_lr_scale: float = 1.0
        self._router_lr_scale: float = 1.0
        self._moe_rewarmup_start_step: int = 0  # Set on resume if moe_lr_rewarmup_steps > 0

        # MoR auto-nudge state (gentle, temporary damping of router advantage loss)
        self._mor_advantage_nudge_until_step: int = 0
        self._mor_advantage_nudge_cooldown_until_step: int = 0
        self._mor_advantage_nudge_active: bool = False
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
                self.logger.info(
                    f"  ├─ fused_rope:    {ks.get('fused_rope', False)}" + ("" if ks.get('fused_rope', False) else " (disabled; set HYDRA_DISABLE_FUSED_ROPE=0 / HYDRA_ENABLE_FUSED_ROPE=1)")
                )
                self.logger.info(
                    f"  └─ fused_rms_norm:{ks.get('fused_rms_norm', False)}" + ("" if ks.get('fused_rms_norm', False) else " (disabled; set HYDRA_DISABLE_FUSED_RMS_NORM=0 / HYDRA_ENABLE_FUSED_RMS_NORM=1)")
                )
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

    def _setup_safe_optimizations(self, config: TrainingConfig) -> SafeOptimizations:
        """Initialize SafeOptimizations with config-based settings.

        Creates OptimizationConfig from TrainingConfig experimental flags,
        handling attributes that may not exist on older configs.
        """
        opt_config = OptimizationConfig(
            # FA3 (Flash Attention 3)
            enable_fa3=bool(getattr(config, "experimental_fa3", True)),
            fa3_fallback_to_fa2=True,
            # CUDA Graphs
            enable_cuda_graphs=bool(getattr(config, "experimental_cuda_graphs", True)),
            cuda_graphs_warmup_steps=int(getattr(config, "cuda_graphs_warmup", 50)),
            # Blackwell Triton tuning
            enable_blackwell_tuning=bool(getattr(config, "experimental_blackwell_tuning", True)),
            triton_block_size_q=int(getattr(config, "triton_block_q", 128)),
            triton_block_size_kv=int(getattr(config, "triton_block_kv", 64)),
            triton_num_warps=int(getattr(config, "triton_num_warps", 8)),
            # Prefetch
            enable_prefetch_threads=int(getattr(config, "experimental_prefetch_threads", 4)),
            prefetch_buffer_size=int(getattr(config, "prefetch_buffer_size", 8)),
            # FP8
            enable_fp8=bool(getattr(config, "experimental_fp8", False)),
            fp8_format=str(getattr(config, "fp8_format", "e4m3")),
            # Safety monitoring
            safety_window_steps=int(getattr(config, "experimental_safety_window", 100)),
            loss_spike_threshold=float(getattr(config, "experimental_loss_spike_threshold", 2.0)),
            throughput_drop_threshold=float(getattr(config, "experimental_throughput_drop_threshold", 0.5)),
            # Pretest
            pretest_steps=int(getattr(config, "experimental_pretest_steps", 10)),
        )

        safe_opts = SafeOptimizations(
            config=opt_config,
            device=self.device,
            logger=self.logger,
        )

        # Log optimization status
        status = safe_opts.get_status_summary()
        active_opts = [k for k, v in status.items() if v in ("ENABLED", "MONITORING", "PRETESTING")]
        if active_opts:
            self.logger.info(f"SafeOptimizations: {', '.join(active_opts)}")

        return safe_opts

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
        
        # MoD is step-based: dense MLP during warmup, then hard top-k routing.
        # This avoids brittle MoR/CE-gated enabling.
        mod_mlp_warmup = int(getattr(config, "mod_mlp_warmup_steps", 1000) or 1000)
        mod_mlp_warmup = max(0, mod_mlp_warmup)
        mod_force_enable_step = None
        
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
            # Step-based MoD warmup: do not CE-gate enablement.
            mod_enable_loss_threshold=None,
            mod_force_enable_step=mod_force_enable_step,
            mor_warmup=mor_warmup,
            mor_advantage_loss_scale=getattr(config, "mor_advantage_loss_scale", 0.02),
            mor_min_depth=getattr(config, "mor_min_depth", 1),
            attention_backend=attention_backend,
            # MoE configuration
            moe_enabled=config.moe_enabled,
            moe_num_experts=config.moe_num_experts,
            moe_num_layers=config.moe_num_layers,
            moe_top_k=config.moe_top_k,
            moe_aux_weight=config.moe_aux_weight,
            moe_router_jitter=config.moe_router_jitter,
            moe_expert_diversity_noise=config.moe_expert_diversity_noise,
            moe_warmup_steps=config.moe_warmup_steps,
            moe_identity_init=config.moe_identity_init,
            moe_forced_routing_steps=getattr(config, "moe_forced_routing_steps", 0),
            moe_teacher_until_step=getattr(config, "moe_teacher_until_step", 0),
            # Static routing mode for CUDA graph compatibility
            static_routing_mode=getattr(config, "static_routing_mode", False),
        ).to(self.device)
        self._use_mod_mor = True
        mod_status = "OFF (capacity=1.0)" if config.mod_capacity >= 1.0 else f"{config.mod_capacity:.0%} capacity (warmup={mod_mlp_warmup} steps)"
        mor_status = "adaptive" if config.mor_adaptive else "fixed-depth (no routing)"
        self.logger.info(f"MoD: {mod_status}")
        self.logger.info(f"MoR: {mor_status}, {config.mor_recursions} recursions/block")
        
        # CurriculumController owns MoR/MoD gating state and decision logic
        self._curriculum_controller = CurriculumController.from_training_config(config, start_step=self._start_step)
        mor_enable_step, actual_rampup = self._curriculum_controller.get_mor_curriculum_params()
        
        if config.mor_adaptive:
            if config.mor_already_enabled:
                self.logger.info("MoR RESTART MODE: Adaptive routing enabled from start (resumed after enable point)")
            self.model.set_mor_curriculum(enable_step=mor_enable_step, rampup_steps=actual_rampup)
            self._mor_enable_step = mor_enable_step
            mor_loss_thr = getattr(config, "mor_enable_loss_threshold", 0.0) or 0.0
            mor_enable_min = getattr(config, "mor_enable_min_steps", 3000)
            if mor_enable_step > 0:
                if mor_loss_thr > 0:
                    self.logger.info(f"MoR CURRICULUM: Fixed-depth until step {mor_enable_step:,} (min={mor_enable_min}, pct={config.mor_enable_pct:.0%}) OR CE_EMA < {mor_loss_thr:.1f}, then {actual_rampup:,} step rampup")
                else:
                    self.logger.info(f"MoR CURRICULUM: Fixed-depth until step {mor_enable_step:,} (min={mor_enable_min}, pct={config.mor_enable_pct:.0%}), then {actual_rampup:,} step rampup")
        else:
            self._mor_enable_step = 0
            self.logger.info("MoR CURRICULUM: Disabled (adaptive=False, running pure fixed-depth)")

        # Static routing mode: enables CUDA graphs by using soft weights instead of dynamic routing
        if getattr(config, "static_routing_mode", False):
            self.logger.info("STATIC ROUTING: Enabled - using soft weights for CUDA graph compatibility")
            self.logger.info("  MoD: All tokens computed, router weights applied after")
            self.logger.info("  MoR: All recursions computed, soft depth weighting")
            self.logger.info("  CUDA Graphs: Forced enabled (no dynamic shapes)")
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
            self.logger.info(f"Compiling model with mode='{mode}'...")
            self.model = torch.compile(self.model, mode=mode, fullgraph=False, dynamic=False)
            self.logger.info("Model compiled successfully!")

    def _setup_optimizer(self) -> None:
        config = self.config
        
        # =====================================================================
        # Per-Component Parameter Groups for MoE Gradient Stabilization
        # =====================================================================
        # When MoE is enabled, split parameters into distinct groups:
        #   - Group A (Veterans/Experts): expert/mlp params -> slower LR (stabilize)
        #   - Group B (Rookies/Routers): router/gate params -> faster LR (accelerate)
        #   - Group C (Backbone): everything else -> base LR
        #   - Group D (Embeddings): embed params -> much slower LR (prevent spikes)
        #
        # This addresses gradient instability from upcycled pre-trained experts
        # exploding while new routers learn too slowly.
        # =====================================================================
        
        # Force sensible defaults for MoE - old checkpoints may have 1.0 which is wrong
        # If config has 1.0 (old default), override with new defaults
        _expert_lr = getattr(config, "moe_expert_lr_scale", None)
        _router_lr = getattr(config, "moe_router_lr_scale", None)
        _expert_wd = getattr(config, "moe_expert_weight_decay_scale", None)
        
        expert_lr_scale = 0.5 if (_expert_lr is None or _expert_lr == 1.0) else float(_expert_lr)
        router_lr_scale = 3.0 if (_router_lr is None or _router_lr == 1.0) else float(_router_lr)
        expert_wd_scale = 3.0 if (_expert_wd is None or _expert_wd == 1.0) else float(_expert_wd)
        embed_lr_scale = 0.1  # Always slower for embeddings
        
        # ALWAYS use per-component LR when MoE is enabled (experts need different treatment)
        use_per_component_lr = config.moe_enabled
        
        # Parameter categorization
        expert_decay_params = []      # Group A: expert/mlp with decay
        expert_no_decay_params = []   # Group A: expert/mlp without decay  
        router_decay_params = []      # Group B: router/gate with decay
        router_no_decay_params = []   # Group B: router/gate without decay
        backbone_decay_params = []    # Group C: backbone with decay
        backbone_no_decay_params = [] # Group C: backbone without decay
        embed_params = []             # Group D: embeddings
        
        expert_param_count = 0
        router_param_count = 0
        backbone_param_count = 0
        embed_param_count = 0
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            n_params = param.numel()
            name_lower = name.lower()
            is_weight = "weight" in name_lower
            is_norm = "norm" in name_lower
            has_decay = is_weight and not is_norm
            
            # Classification priority: embed > moe_expert > moe_router > backbone
            # Key insight: we need to distinguish:
            #   - MoE expert FFNs: moe_layers.X.experts.* -> expert group
            #   - MoE routers: moe_layers.X.router.* -> router group  
            #   - Backbone MLPs: layers.X.mlp.* -> backbone group (NOT expert!)
            #   - Other routers: mor_router, mod_router -> backbone (they're part of base model)
            
            is_moe_expert = "moe_layers" in name_lower and "experts" in name_lower
            is_moe_router = "moe_layers" in name_lower and "router" in name_lower and "experts" not in name_lower
            
            if "embed" in name_lower or "tok_emb" in name_lower:
                embed_params.append(param)
                embed_param_count += n_params
            elif use_per_component_lr and is_moe_expert:
                # MoE expert FFN weights (moe_layers.X.experts.*)
                if has_decay:
                    expert_decay_params.append(param)
                else:
                    expert_no_decay_params.append(param)
                expert_param_count += n_params
            elif use_per_component_lr and is_moe_router:
                # MoE router gate weights (moe_layers.X.router.*)
                if has_decay:
                    router_decay_params.append(param)
                else:
                    router_no_decay_params.append(param)
                router_param_count += n_params
            else:
                # Everything else is backbone: attention, backbone MLPs, norms, MoR/MoD routers, etc.
                if has_decay:
                    backbone_decay_params.append(param)
                else:
                    backbone_no_decay_params.append(param)
                backbone_param_count += n_params
        
        # Compute per-group LRs
        expert_lr = config.max_lr * expert_lr_scale
        router_lr = config.max_lr * router_lr_scale
        backbone_lr = config.max_lr
        embed_lr = config.max_lr * embed_lr_scale
        
        # Compute expert weight decay (allow higher WD to shrink blown-up weights)
        expert_weight_decay = config.weight_decay * expert_wd_scale
        
        self._skip_lr_schedule = False
        
        # Store LR scales for training loop to apply per-group scaling
        self._embed_lr_scale = embed_lr_scale
        self._expert_lr_scale = expert_lr_scale
        self._router_lr_scale = router_lr_scale
        self._use_per_component_lr = use_per_component_lr
        
        # Build parameter groups list
        if use_per_component_lr:
            # Full per-component groups (6 groups)
            param_groups = [
                # Group 0: Expert decay (with scaled weight decay to shrink blown-up weights)
                {"params": expert_decay_params, "weight_decay": expert_weight_decay, "lr": expert_lr, "lr_scale": expert_lr_scale, "name": "expert_decay"},
                # Group 1: Expert no-decay
                {"params": expert_no_decay_params, "weight_decay": 0.0, "lr": expert_lr, "lr_scale": expert_lr_scale, "name": "expert_nodecay"},
                # Group 2: Router decay
                {"params": router_decay_params, "weight_decay": config.weight_decay, "lr": router_lr, "lr_scale": router_lr_scale, "name": "router_decay"},
                # Group 3: Router no-decay
                {"params": router_no_decay_params, "weight_decay": 0.0, "lr": router_lr, "lr_scale": router_lr_scale, "name": "router_nodecay"},
                # Group 4: Backbone decay
                {"params": backbone_decay_params, "weight_decay": config.weight_decay, "lr": backbone_lr, "lr_scale": 1.0, "name": "backbone_decay"},
                # Group 5: Backbone no-decay
                {"params": backbone_no_decay_params, "weight_decay": 0.0, "lr": backbone_lr, "lr_scale": 1.0, "name": "backbone_nodecay"},
                # Group 6: Embeddings
                {"params": embed_params, "weight_decay": 0.0, "lr": embed_lr, "lr_scale": embed_lr_scale, "name": "embed"},
            ]
            # Remove empty groups
            param_groups = [g for g in param_groups if len(g["params"]) > 0]
        else:
            # Legacy 3-group setup (backward compatible)
            decay_params = backbone_decay_params + expert_decay_params + router_decay_params
            no_decay_params = backbone_no_decay_params + expert_no_decay_params + router_no_decay_params
            param_groups = [
                {"params": decay_params, "weight_decay": config.weight_decay, "lr": config.max_lr, "lr_scale": 1.0, "name": "decay"},
                {"params": no_decay_params, "weight_decay": 0.0, "lr": config.max_lr, "lr_scale": 1.0, "name": "nodecay"},
                {"params": embed_params, "weight_decay": 0.0, "lr": embed_lr, "lr_scale": embed_lr_scale, "name": "embed"},
            ]
            param_groups = [g for g in param_groups if len(g["params"]) > 0]
        
        if config.use_adafactor:
            adafactor_lr = 0.01
            for pg in param_groups:
                pg["lr"] = adafactor_lr * pg.get("lr_scale", 1.0)
            
            self.optimizer = torch.optim.Adafactor(
                param_groups,
                lr=adafactor_lr,
                beta2_decay=-0.8,
                eps=(None, 1e-3),
                d=1.0,
                weight_decay=0.0,
                foreach=True,
            )
            self._skip_lr_schedule = True
            self.logger.info(f"Using PyTorch Adafactor - lr={adafactor_lr:.2e} (internal 1/√t schedule)")
        elif config.use_8bit_adam:
            import bitsandbytes as bnb
            self.optimizer = bnb.optim.AdamW8bit(
                param_groups,
                lr=config.max_lr,
                betas=(0.9, 0.95),
            )
            self.logger.info("Using 8-bit AdamW (bitsandbytes) - ~75% optimizer memory savings")
        else:
            self.optimizer = torch.optim.AdamW(
                param_groups,
                lr=config.max_lr,
                betas=(0.9, 0.95),
                fused=(self.device == "cuda"),
            )
        
        self._param_groups = self.optimizer.param_groups
        use_scaler = config.dtype == "float16"
        self.scaler = GradScaler(self.device, enabled=use_scaler)
        self._use_scaler = use_scaler
        
        # Log parameter group summary
        total_params = expert_param_count + router_param_count + backbone_param_count + embed_param_count
        self.logger.info("=" * 60)
        self.logger.info("OPTIMIZER PARAMETER GROUPS")
        self.logger.info("=" * 60)
        if use_per_component_lr:
            self.logger.info(f"  🎯 Per-Component LR ENABLED (MoE gradient stabilization)")
            wd_info = f", WD scale: {expert_wd_scale}x" if expert_wd_scale != 1.0 else ""
            self.logger.info(f"  MoE Experts (moe_layers.*.experts.*): {expert_param_count:,} params ({100*expert_param_count/max(1,total_params):.1f}%) | LR scale: {expert_lr_scale}x{wd_info}")
            self.logger.info(f"  MoE Routers (moe_layers.*.router.*):  {router_param_count:,} params ({100*router_param_count/max(1,total_params):.1f}%) | LR scale: {router_lr_scale}x")
            self.logger.info(f"  Backbone (attn, MLP, norms, MoR/MoD): {backbone_param_count:,} params ({100*backbone_param_count/max(1,total_params):.1f}%) | LR scale: 1.0x")
            self.logger.info(f"  Embeddings (tok_emb, recursion_emb):  {embed_param_count:,} params ({100*embed_param_count/max(1,total_params):.1f}%) | LR scale: {embed_lr_scale}x")
        else:
            self.logger.info(f"  Standard parameter groups (per-component LR disabled)")
            self.logger.info(f"  Backbone params: {backbone_param_count + expert_param_count + router_param_count:,}")
            self.logger.info(f"  Embedding params: {embed_param_count:,}")
        self.logger.info(f"  Total trainable: {total_params:,}")
        self.logger.info("=" * 60)

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

        if self._moe_domain_expert_map:
            self.logger.info(f"MoE domain->expert map: {self._moe_domain_expert_map}")
            names = getattr(self.train_loader, "dataset_names", None)
            if isinstance(names, list) and names:
                self.logger.info(f"Interleaved sources: {names}")
                missing = [n for n in names if n not in self._moe_domain_expert_map]
                if missing:
                    self.logger.info(
                        "MoE domain forcing: no mapping for sources: " + ", ".join(str(x) for x in missing)
                    )
            # Log two-phase specialization strategy
            forced_steps = int(getattr(config, "moe_forced_routing_steps", 0) or 0)
            teacher_weight = float(getattr(config, "moe_teacher_weight", 0.0) or 0.0)
            teacher_until = int(getattr(config, "moe_teacher_until_step", 0) or 0)
            if forced_steps > 0 or teacher_weight > 0.0:
                self.logger.info("=" * 60)
                self.logger.info("🎯 MoE TWO-PHASE EXPERT SPECIALIZATION ENABLED")
                self.logger.info("=" * 60)
                if forced_steps > 0:
                    self.logger.info(f"  Phase 1 (HARD FORCING): steps 0-{forced_steps}")
                    self.logger.info("    → Domain batches forced to mapped expert (bypasses router)")
                if teacher_weight > 0.0:
                    until_str = f"step {teacher_until}" if teacher_until > 0 else "forever"
                    phase2_start = f"step {forced_steps}" if forced_steps > 0 else "step 0"
                    self.logger.info(f"  Phase 2 (TEACHER SUPERVISION): {phase2_start} to {until_str}")
                    self.logger.info("    → Router chooses freely, but CE loss guides toward domain expert")
                    self.logger.info(f"    → Teacher loss weight (alpha): {teacher_weight}")
                self.logger.info("=" * 60)
        if hasattr(self.train_loader, "set_max_steps"):
            self.train_loader.set_max_steps(config.max_steps)
        self._tokens_per_step = config.batch_size * config.grad_accum_steps * initial_seq_len
        self.logger.info(f"Tokens per step: {self._tokens_per_step:,}")
        self.logger.info("Dataset ready!")

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"RESUMING FROM CHECKPOINT: {checkpoint_path}")
        self.logger.info(f"{'='*70}")
        
        # Load checkpoint to CPU first to avoid OOM from loading large optimizer states
        # Model weights will be moved to device during load_state_dict
        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
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

        def _sanitize_bitsandbytes_state_for_step() -> bool:
            """Return True if bnb optimizer state looks usable; else clear it.

            bitsandbytes initializes per-parameter state only when `len(state)==0`.
            If we load a checkpoint from a *non-bnb* optimizer (e.g., torch AdamW),
            the loaded state dict can contain keys like `exp_avg`/`exp_avg_sq` and
            `step`, which makes `len(state)>0` but lacks bnb's required buffers
            (`state1`/`state2`, etc.), causing a KeyError on the first step.
            """

            if not bool(getattr(self.config, "use_8bit_adam", False)):
                return True

            try:
                if not str(getattr(self.optimizer.__class__, "__module__", "")).startswith("bitsandbytes"):
                    return True
            except Exception:
                return True

            cleared = 0
            # Iterate over the optimizer's current parameter states.
            for p, state in list(getattr(self.optimizer, "state", {}).items()):
                try:
                    if not isinstance(state, dict) or len(state) == 0:
                        continue
                    # Minimum required keys for `update_step`.
                    if ("step" not in state) or ("state1" not in state) or ("state2" not in state):
                        state.clear()
                        cleared += 1
                        continue
                    # Additional required keys when running in 8-bit mode.
                    s1 = state.get("state1")
                    if hasattr(s1, "dtype") and s1.dtype == torch.uint8:
                        if ("qmap1" not in state) or ("qmap2" not in state):
                            state.clear()
                            cleared += 1
                            continue
                        has_blockwise = ("absmax1" in state) and ("absmax2" in state)
                        has_non_blockwise = (
                            ("max1" in state)
                            and ("max2" in state)
                            and ("new_max1" in state)
                            and ("new_max2" in state)
                        )
                        if not (has_blockwise or has_non_blockwise):
                            state.clear()
                            cleared += 1
                            continue
                except Exception:
                    try:
                        state.clear()
                    except Exception:
                        pass
                    cleared += 1

            if cleared > 0:
                self.logger.warning(
                    f"  ⚠️  bitsandbytes optimizer state incompatible for {cleared} params; "
                    "resetting 8-bit optimizer state (moments will be re-initialized)"
                )
                return False
            return True

        optimizer_loaded = False
        scaler_loaded = False
        rng_loaded = False
        
        # Detect optimizer type mismatch BEFORE loading (avoids OOM from allocating old state)
        skip_optimizer_load = False
        if "optimizer" in checkpoint:
            ckpt_opt_state = checkpoint["optimizer"]
            # Check if checkpoint has AdamW-style state but we're using 8-bit or Adafactor
            if isinstance(ckpt_opt_state, dict) and "state" in ckpt_opt_state:
                ckpt_param_state = ckpt_opt_state.get("state", {})
                # Sample first param state to detect optimizer type
                for _, pstate in ckpt_param_state.items():
                    if isinstance(pstate, dict):
                        has_adamw_keys = "exp_avg" in pstate and "exp_avg_sq" in pstate
                        has_bnb_keys = "state1" in pstate and "state2" in pstate
                        
                        # Switching FROM AdamW TO 8-bit/Adafactor - skip to avoid OOM
                        if has_adamw_keys and not has_bnb_keys:
                            if getattr(self.config, "use_8bit_adam", False):
                                self.logger.warning("  ⚠️  Checkpoint has AdamW state but using 8-bit Adam - skipping optimizer load to avoid OOM")
                                skip_optimizer_load = True
                            elif getattr(self.config, "use_adafactor", False):
                                self.logger.warning("  ⚠️  Checkpoint has AdamW state but using Adafactor - skipping optimizer load")
                                skip_optimizer_load = True
                        break  # Only need to check one param
        
        if not skip_optimizer_load:
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
                optimizer_loaded = _sanitize_bitsandbytes_state_for_step()
                if not optimizer_loaded:
                    # Avoid LR scaling based on a checkpoint optimizer LR we didn't actually restore.
                    self._resume_lr_scale = 1.0
            except Exception as e:
                # Common after changing optimizer type/grouping or adding/removing params.
                # Keep training going: resume weights + step, but re-init optimizer state.
                self.logger.warning(
                    "  ⚠️  Optimizer state incompatible with current optimizer; skipping optimizer resume"
                )
                self.logger.warning(f"  Reason: {e}")
                self.logger.warning("  Continuing with freshly initialized optimizer (momentum/Adam moments reset)")
                optimizer_loaded = False
                # Avoid LR scaling based on a checkpoint optimizer LR we didn't apply.
                self._resume_lr_scale = 1.0
        else:
            self.logger.warning("  Continuing with freshly initialized optimizer (momentum/Adam moments reset)")
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
            if isinstance(extra, dict):
                if "ce_ema" in extra:
                    self._resume_ce_ema = float(extra.get("ce_ema", 0.0) or 0.0)
                if "grad_norm_ema" in extra:
                    self._grad_norm_ema = float(extra.get("grad_norm_ema", 0.0) or 0.0)
                    self.logger.info(f"  Restored grad_norm_ema: {self._grad_norm_ema:.1f}")
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
        
        # MoE per-component LR re-warmup (for mid-run optimizer reset / hot-fix)
        # This is triggered when moe_lr_rewarmup_steps > 0, indicating user intentionally
        # wants to re-warmup after applying new per-component LR scales (momentum states lost).
        moe_rewarmup_steps = int(getattr(self.config, "moe_lr_rewarmup_steps", 0) or 0)
        if moe_rewarmup_steps > 0 and int(self._start_step) > 0:
            self._moe_rewarmup_start_step = int(self._start_step)
            self.logger.info("=" * 60)
            self.logger.info("🔧 MoE LR RE-WARMUP ENABLED (hot-fix for gradient instability)")
            self.logger.info("=" * 60)
            self.logger.info(f"  Re-warmup steps: {moe_rewarmup_steps}")
            self.logger.info(f"  Starting from step: {self._start_step}")
            self.logger.info(f"  Will ramp LR from 0 -> target over {moe_rewarmup_steps} steps")
            if getattr(self, "_use_per_component_lr", False):
                self.logger.info(f"  Expert LR scale: {self._expert_lr_scale}x (stabilize upcycled experts)")
                self.logger.info(f"  Router LR scale: {self._router_lr_scale}x (accelerate router learning)")
            self.logger.info("=" * 60)
        
        # Apply MoE expert diversity noise post-load to break symmetry on resumed checkpoints
        # This is critical for identity-init MoE that hasn't specialized yet
        if hasattr(self.config, "moe_expert_diversity_noise") and self.config.moe_expert_diversity_noise > 0:
            noise_std = self.config.moe_expert_diversity_noise
            model = self.model
            if hasattr(model, "_orig_mod"):
                model = model._orig_mod
            if hasattr(model, "moe_layers") and len(model.moe_layers) > 0:
                with torch.no_grad():
                    perturbed_count = 0
                    for layer_idx, moe_layer in enumerate(model.moe_layers):
                        if hasattr(moe_layer, "experts"):
                            for expert_idx, expert in enumerate(moe_layer.experts):
                                # Deterministic seed per (layer, expert) for reproducibility
                                gen = torch.Generator(device=expert.gate_up.weight.device)
                                gen.manual_seed(42 + layer_idx * 1000 + expert_idx * 137)
                                for param in expert.parameters():
                                    noise = torch.randn_like(param, generator=gen) * noise_std
                                    param.add_(noise)
                                perturbed_count += 1
                self.logger.info(f"  ⚡ Applied diversity noise (std={noise_std}) to {perturbed_count} MoE experts to break symmetry")

        # Run pretest hook on checkpoint load (if enabled)
        if self._pretest_hook is not None:
            try:
                # Create sample batch for pretest
                sample_batch = torch.randint(
                    0, self.config.vocab_size, (2, 512), device=self.device
                )
                base_model = self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model
                pretest_record = self._pretest_hook.on_checkpoint_loaded(
                    model=base_model,
                    config=self.config,
                    checkpoint_path=checkpoint_path,
                    sample_batch=sample_batch,
                )
                if pretest_record is not None:
                    self.logger.info(
                        f"  Pretests: {pretest_record.passed_count} passed, "
                        f"{pretest_record.failed_count} failed, "
                        f"{pretest_record.skipped_count} skipped"
                    )
            except Exception as e:
                self.logger.warning(f"  ⚠️  Pretest hook failed: {e}")

        self.logger.info(f"{'='*70}\n")

    def _compute_moe_divergence(self) -> tuple[float, float, list[float]]:
        """Compute MoE expert divergence, routing entropy, and utilization.
        
        Divergence: 1 - avg(cosine_similarity) between expert weight pairs.
            0 = identical experts, 1 = orthogonal/maximally different.
        Entropy: Actual routing entropy from router stats (not placeholder).
        Utilization: Fraction of tokens routed to each expert (accumulated over interval).
        
        Returns:
            (divergence, entropy, utilization_list)
        """
        import math
        import torch.nn.functional as F
        
        model = self.model
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
        
        if not hasattr(model, "moe_layers") or len(model.moe_layers) == 0:
            return 0.0, 0.0, []
        
        # Compute pairwise cosine distance between expert weights
        all_cos_distances = []
        num_experts = 0
        all_utilizations = []
        all_entropies = []
        
        with torch.no_grad():
            for moe_layer in model.moe_layers:
                if not hasattr(moe_layer, "experts"):
                    continue
                experts = moe_layer.experts
                num_experts = len(experts)
                
                # Flatten expert weights for cosine similarity
                weights = [experts[i].gate_up.weight.float().flatten() for i in range(num_experts)]
                
                # Pairwise cosine distance: 1 - cos_sim (0=identical, 1=orthogonal, 2=opposite)
                for i in range(num_experts):
                    for j in range(i + 1, num_experts):
                        cos_sim = F.cosine_similarity(weights[i].unsqueeze(0), weights[j].unsqueeze(0)).item()
                        cos_distance = 1.0 - cos_sim  # 0=identical, 1=orthogonal
                        all_cos_distances.append(cos_distance)
                
                # Get real routing stats from router if available
                if hasattr(moe_layer, "router"):
                    router = moe_layer.router
                    # Prefer accumulated counts (better signal across multiple batches)
                    if hasattr(router, "_expert_counts_accum") and hasattr(router, "_expert_counts_n"):
                        n = router._expert_counts_n.item()
                        if n > 0:
                            counts = (router._expert_counts_accum / n).detach().cpu().float()
                        else:
                            counts = router._expert_counts.detach().cpu().float() if hasattr(router, "_expert_counts") else None
                    elif hasattr(router, "_expert_counts"):
                        counts = router._expert_counts.detach().cpu().float()
                    else:
                        counts = None
                    
                    if counts is not None and counts.sum() > 0:
                        probs = counts / counts.sum()
                        all_utilizations.append(probs.tolist())
                        # Compute actual entropy: -sum(p * log(p))
                        entropy = -sum(p * math.log(p + 1e-10) for p in probs.tolist())
                        all_entropies.append(entropy)
                    
                    # Reset accumulated counts after reading
                    if hasattr(router, "reset_accumulated_counts"):
                        router.reset_accumulated_counts()
        
        # Average divergence across all layers and pairs
        avg_divergence = sum(all_cos_distances) / len(all_cos_distances) if all_cos_distances else 0.0
        
        # Average entropy across layers (or estimate from uniform if no stats)
        if all_entropies:
            avg_entropy = sum(all_entropies) / len(all_entropies)
        else:
            # Fallback: max entropy for uniform routing
            avg_entropy = math.log(num_experts) if num_experts > 1 else 0.0
        
        # Average utilization across layers
        if all_utilizations:
            avg_util = [sum(u[i] for u in all_utilizations) / len(all_utilizations) 
                        for i in range(num_experts)]
        else:
            avg_util = [1.0 / num_experts] * num_experts if num_experts > 0 else []
        
        return avg_divergence, avg_entropy, avg_util

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
        if not self._seq_len_policy.is_active:
            return

        patch = self._seq_len_policy.get_patch(int(seq_len))
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
        # Keep source metadata (if present) for MoE domain forcing/teacher.
        try:
            self._last_batch_source_name = batch.get("source_name")
        except Exception:
            self._last_batch_source_name = None

        sn = self._last_batch_source_name
        if isinstance(sn, str) and sn:
            self._source_name_counts[sn] = self._source_name_counts.get(sn, 0) + 1
            self._source_name_counts_total[sn] = self._source_name_counts_total.get(sn, 0) + 1
        mask = batch.get("attention_mask")
        if mask is not None:
            mask = mask.to(self.device, non_blocking=True)
        return (
            batch["input_ids"].to(self.device, non_blocking=True),
            batch["labels"].to(self.device, non_blocking=True),
            mask,
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

        # Dynamic gradient clipping state
        _use_dynamic_clip = bool(getattr(config, "grad_clip_dynamic", True))
        _clip_k = float(getattr(config, "grad_clip_k", 2.0))
        _clip_min = float(getattr(config, "grad_clip_min", 50.0))
        _clip_max = float(getattr(config, "grad_clip_max", 3000.0))
        _clip_ema_alpha = float(getattr(config, "grad_clip_ema_alpha", 0.05))
        # Initialize EMA from checkpoint or static grad_clip (R2/R5 mitigation)
        _grad_norm_ema = float(getattr(self, "_grad_norm_ema", 0.0) or 0.0)
        if _grad_norm_ema <= 0:
            _grad_norm_ema = float(grad_clip) / _clip_k  # Start so dynamic_clip = grad_clip
        self._grad_norm_ema = _grad_norm_ema  # Initialize slot
        if _use_dynamic_clip:
            self.logger.info(f"Dynamic gradient clipping: ENABLED (k={_clip_k}, min={_clip_min}, max={_clip_max}, ema={_grad_norm_ema:.1f})")
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

        # Peak VRAM tracking (for quick smoke runs / capacity estimates)
        _is_cuda_device = False
        if torch.cuda.is_available():
            if isinstance(device, torch.device):
                _is_cuda_device = device.type == "cuda"
            elif isinstance(device, str):
                _is_cuda_device = device.startswith("cuda")

        _track_peak_vram = bool(_is_cuda_device)
        _cuda_device_idx = torch.cuda.current_device() if _track_peak_vram else None
        if _track_peak_vram:
            try:
                torch.cuda.reset_peak_memory_stats(device=_cuda_device_idx)
            except Exception:
                pass

        metrics.start_time = time.time()
        self._last_aux_loss = 0.0
        self._ce_ema = float(getattr(self, "_resume_ce_ema", 0.0) or 0.0) if start_step > 0 else 0.0
        eval_batches = 25
        eval_dataset = str(getattr(config, "eval_dataset_name", "") or "")
        if not eval_dataset:
            eval_dataset = config.dataset_name
            if config.dataset_name.startswith("pretrain_"):
                eval_dataset = "pretrain_default_eval"
                self.logger.info(
                    f"📊 Using pretrain_default_eval for evaluation (train dataset: {config.dataset_name})"
                )
            elif config.dataset_name in ["sft_chat"]:
                eval_dataset = "wikitext2"
                self.logger.info(
                    f"📊 Using wikitext2 for evaluation (train dataset: {config.dataset_name})"
                )
        else:
            self.logger.info(
                f"📊 Using {eval_dataset} for evaluation (override; train dataset: {config.dataset_name})"
            )

        try:
            eval_loader = create_universal_loader(
                dataset=eval_dataset,
                batch_size=config.batch_size,
                seq_len=self._current_seq_len,
                vocab_size=config.vocab_size,
                device="cpu",
                tokenizer_name=config.tokenizer_name,
            )
        except Exception as e:
            self.logger.warning(
                f"⚠️  Failed to create eval loader for dataset='{eval_dataset}': {type(e).__name__}: {e}"
            )
            self.logger.warning("   Falling back to wikitext2 for evaluation.")
            eval_dataset = "wikitext2"
            eval_loader = create_universal_loader(
                dataset=eval_dataset,
                batch_size=config.batch_size,
                seq_len=self._current_seq_len,
                vocab_size=config.vocab_size,
                device="cpu",
                tokenizer_name=config.tokenizer_name,
            )
        fixed_eval_batches = [eval_loader.get_batch() for _ in range(eval_batches)]

        # Optional: for mixed eval presets, also build per-component fixed eval batches
        # so we can report per-source eval loss (e.g. "math died, web improved").
        fixed_eval_batches_by_component: Dict[str, List[dict]] = {}
        eval_component_weights: Dict[str, float] = {}
        try:
            eval_cfg = DATASET_CONFIGS.get(eval_dataset, {}) if isinstance(DATASET_CONFIGS, dict) else {}
            sources = None
            if isinstance(eval_cfg, dict) and eval_cfg.get("mixed"):
                sources = eval_cfg.get("sources")
            elif isinstance(eval_cfg, dict) and eval_cfg.get("mixed_by_seq"):
                seq_key = str(int(self._current_seq_len))
                seq_cfg = eval_cfg.get("mixed_by_seq", {}).get(seq_key)
                if seq_cfg is None:
                    # Best-effort: pick nearest available seq
                    available = [int(k) for k in eval_cfg.get("mixed_by_seq", {}).keys() if str(k).isdigit()]
                    if available:
                        nearest = min(available, key=lambda v: abs(v - int(self._current_seq_len)))
                        seq_cfg = eval_cfg.get("mixed_by_seq", {}).get(str(nearest))
                if isinstance(seq_cfg, dict):
                    sources = seq_cfg.get("sources")

            if isinstance(sources, list) and sources:
                # Keep total eval work roughly constant: split eval_batches across components.
                per_comp_batches = max(1, int(math.ceil(eval_batches / max(1, len(sources)))))
                for s in sources:
                    name = str(s.get("name", ""))
                    if not name:
                        continue
                    try:
                        eval_component_weights[name] = float(s.get("weight", 0.0) or 0.0)
                    except Exception:
                        eval_component_weights[name] = 0.0
                    try:
                        comp_loader = create_universal_loader(
                            dataset=name,
                            batch_size=config.batch_size,
                            seq_len=self._current_seq_len,
                            vocab_size=config.vocab_size,
                            device="cpu",
                            tokenizer_name=config.tokenizer_name,
                        )
                        fixed_eval_batches_by_component[name] = [comp_loader.get_batch() for _ in range(per_comp_batches)]
                    except Exception as e:
                        self.logger.warning(
                            f"⚠️  Skipping eval component '{name}': failed to create loader ({type(e).__name__}: {e})"
                        )
        except Exception:
            fixed_eval_batches_by_component = {}
            eval_component_weights = {}
        
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

                # Log SafeOptimizations status to TensorBoard
                if self._safe_opts is not None:
                    try:
                        opt_status = self._safe_opts.get_status_summary()
                        triton_cfg = self._safe_opts.get_triton_config()

                        # Log optimization status as text
                        opt_text = "## SafeOptimizations Status\n\n"
                        opt_text += "| Optimization | Status |\n|---|---|\n"
                        for opt, status in opt_status.items():
                            emoji = "✓" if status in ("ENABLED", "MONITORING") else "✗"
                            opt_text += f"| {opt} | {emoji} {status} |\n"

                        opt_text += "\n## Triton Configuration\n\n"
                        opt_text += "| Parameter | Value |\n|---|---|\n"
                        for param, value in triton_cfg.items():
                            opt_text += f"| {param} | {value} |\n"

                        tb_writer.add_text("optimizations/status", opt_text, 0)

                        # Log pretest results if available
                        if self._pretest_hook is not None:
                            history = self._pretest_hook.get_history(
                                model_size=str(config.model_size), limit=1
                            )
                            if history:
                                latest = history[0]
                                pretest_text = "## Pretest Results\n\n"
                                pretest_text += f"- **Timestamp**: {latest.get('timestamp', 'N/A')}\n"
                                pretest_text += f"- **GPU**: {latest.get('gpu_name', 'N/A')} ({latest.get('gpu_arch', 'N/A')})\n"
                                pretest_text += f"- **Passed**: {latest.get('passed_count', 0)}\n"
                                pretest_text += f"- **Failed**: {latest.get('failed_count', 0)}\n\n"

                                pretest_text += "| Optimization | Status | Time (ms) |\n|---|---|---|\n"
                                for r in latest.get("results", []):
                                    status = r.get("status", "unknown")
                                    emoji = "✓" if status == "passed" else ("○" if status == "skipped" else "✗")
                                    time_ms = r.get("time_ms", 0)
                                    pretest_text += f"| {r.get('optimization', '')} | {emoji} {status} | {time_ms:.1f} |\n"

                                tb_writer.add_text("optimizations/pretests", pretest_text, 0)

                        self.logger.info("📊 Logged optimization status to TensorBoard")
                    except Exception:
                        pass
            except Exception as e:
                self.logger.warning(
                    f"⚠️  TensorBoard requested but unavailable: {type(e).__name__}: {e}"
                )
                config.use_tensorboard = False

        wandb_mod = None
        wandb_run = None

        # Get SafeOptimizations status for logging
        _safe_opts_status = {}
        _safe_opts_config = {}
        if self._safe_opts is not None:
            _safe_opts_status = self._safe_opts.get_status_summary()
            _safe_opts_config = self._safe_opts.get_triton_config()

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
                        # SafeOptimizations status
                        "optimizations": _safe_opts_status,
                        "triton_config": _safe_opts_config,
                        "opt_fa3": _safe_opts_status.get("fa3", "DISABLED"),
                        "opt_cuda_graphs": _safe_opts_status.get("cuda_graphs", "DISABLED"),
                        "opt_blackwell_tuning": _safe_opts_status.get("blackwell_tuning", "DISABLED"),
                        "opt_prefetch_threads": _safe_opts_status.get("prefetch_threads", "DISABLED"),
                        "opt_fp8": _safe_opts_status.get("fp8", "DISABLED"),
                    },
                )
                self.logger.info("📡 W&B enabled.")

                # Log pretest results as a table if available
                if self._pretest_hook is not None:
                    try:
                        history = self._pretest_hook.get_history(
                            model_size=str(config.model_size), limit=1
                        )
                        if history:
                            latest = history[0]
                            pretest_table = wandb_mod.Table(
                                columns=["optimization", "status", "time_ms", "error"]
                            )
                            for r in latest.get("results", []):
                                pretest_table.add_data(
                                    r.get("optimization", ""),
                                    r.get("status", ""),
                                    r.get("time_ms", 0),
                                    r.get("error_message", "")[:50],
                                )
                            wandb_run.log({"pretest_results": pretest_table})
                            self.logger.info("📊 Logged pretest results to W&B")
                    except Exception:
                        pass
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

        # HaltController owns all halt-related env vars, EMA tracking, and decision logic
        _halt_controller = HaltController.from_env(debug=getattr(self.config, "ema_debug", False))
        
        # Spike tracker for rolling window rate analysis (window size from HaltController)
        _spike_tracker = _spike.SpikeTracker(window_size=_halt_controller.spike_window)

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

        # Graceful shutdown handling for Ctrl+C
        _interrupt_requested = False
        _original_sigint_handler = signal.getsignal(signal.SIGINT)
        
        def _handle_interrupt(signum, frame):
            nonlocal _interrupt_requested
            if _interrupt_requested:
                # Second Ctrl+C: force exit
                self.logger.warning("\n⚠️  Second interrupt received - forcing exit without save")
                signal.signal(signal.SIGINT, _original_sigint_handler)
                raise KeyboardInterrupt
            _interrupt_requested = True
            self.logger.info("\n🛑 Interrupt received - will save checkpoint and exit after current step completes")
            self.logger.info("   (Press Ctrl+C again to force immediate exit without saving)")
        
        signal.signal(signal.SIGINT, _handle_interrupt)

        # Run SafeOptimizations pretests via hook (optional, can be skipped via config)
        # The hook automatically detects config changes and caches results
        _skip_pretest = bool(getattr(self.config, "experimental_skip_pretest", False))
        if not _skip_pretest and self._pretest_hook is not None:
            try:
                # Get a sample batch for pretesting
                sample_x, sample_y, _ = self._get_batch()
                base_model = model._orig_mod if hasattr(model, "_orig_mod") else model

                # Use on_config_changed which checks if config differs from last run
                pretest_record = self._pretest_hook.on_config_changed(
                    model=base_model,
                    config=self.config,
                    sample_batch=sample_x,
                )

                if pretest_record is not None:
                    self.logger.info(
                        f"SafeOptimizations pretests: {pretest_record.passed_count} passed, "
                        f"{pretest_record.failed_count} failed"
                    )
                else:
                    self.logger.info("SafeOptimizations pretests: using cached results")
            except Exception as e:
                self.logger.warning(f"SafeOptimizations pretest skipped: {e}")

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

                # Optional MoE domain forcing + router teacher target.
                # Done here (outside autocast/compile regions) to avoid sync/graph issues.
                #
                # Two-Phase Expert Specialization:
                # - Phase 1 (Hard Forcing): step < moe_forced_routing_steps
                #   All tokens from domain X are forced to expert X (bypasses router).
                #   Controlled by: moe_forced_routing_steps > 0 && step < that value.
                # - Phase 2 (Teacher Supervision): step >= moe_forced_routing_steps (or always if moe_teacher_weight > 0)
                #   Router chooses freely, but CE(router_logits, target_domain) is added to loss.
                #   Controlled by: moe_teacher_weight > 0.0 && (step < moe_teacher_until_step OR moe_teacher_until_step == 0 for "forever").
                #
                # Both phases use the same domain->expert mapping (moe_domain_expert_map).
                try:
                    base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
                    moe_layers = getattr(base_model, "moe_layers", None)
                    if moe_layers is not None and len(moe_layers) > 0:
                        source_name = self._last_batch_source_name
                        expert_id = -1
                        if isinstance(source_name, str) and self._moe_domain_expert_map:
                            expert_id = int(self._moe_domain_expert_map.get(source_name, -1))

                        # Force phase: expert_id >= 0 tells router to hard-route to this expert
                        # (only effective while global_step < forced_routing_steps in router).
                        forced_expert_id = expert_id if expert_id >= 0 else -1

                        # Teacher phase: expert_id >= 0 tells router to compute CE loss toward this expert.
                        # Active when moe_teacher_weight > 0. Step cutoff handled inside router
                        # (moe_teacher_until_step=0 means forever, >0 means until that step).
                        teacher_weight = float(getattr(self.config, "moe_teacher_weight", 0.0) or 0.0)
                        teacher_target_id = forced_expert_id if teacher_weight > 0.0 else -1

                        for moe_block in moe_layers:
                            if hasattr(moe_block, "set_forced_expert"):
                                moe_block.set_forced_expert(forced_expert_id)
                            if hasattr(moe_block, "set_teacher_target"):
                                moe_block.set_teacher_target(teacher_target_id)
                            # Pre-generate jitter noise outside CUDA graph capture
                            if hasattr(moe_block, "router") and hasattr(moe_block.router, "refresh_jitter_noise"):
                                # Shape: [batch, seq_len, num_experts]
                                jitter_shape = (x.shape[0], x.shape[1], moe_block.router.num_experts)
                                moe_block.router.refresh_jitter_noise(jitter_shape)
                except Exception:
                    pass

                # Mark CUDA graph step begin if CUDA graphs are enabled and SafeOptimizations allows it
                # Static routing mode bypasses the SafeOptimizations check because it's designed
                # to eliminate the dynamic shapes that break CUDA graphs
                _use_cuda_graphs = (
                    self.device == "cuda" and
                    self.config.use_compile and
                    (
                        getattr(self.config, "static_routing_mode", False) or
                        self._safe_opts is None or
                        self._safe_opts.should_use_cuda_graphs()
                    )
                )
                if _use_cuda_graphs:
                    torch.compiler.cudagraph_mark_step_begin()
                with autocast(device, dtype=dtype):
                    loss, ce_loss, logits, aux_loss_t, ponder_loss_t, advantage_loss_t, moe_aux_loss_t = compute_microbatch_loss(
                        model=model,
                        x=x,
                        y=y,
                        mask=mask,
                        config=self.config,
                        device=self.device,
                        dtype=dtype,
                        use_mod_mor=use_mod_mor,
                        track_loss_scalars=track_loss_scalars,
                        compute_token_losses_from_hidden=lambda hidden, weight, targets: compute_token_losses_from_hidden(
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
                        self._last_moe_aux_loss = (
                            moe_aux_loss_t.detach()
                            if isinstance(moe_aux_loss_t, torch.Tensor)
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

            # Dynamic gradient clipping: compute adaptive threshold
            # Always track an EMA-based dynamic clip; if static clipping is active,
            # use the larger of static and dynamic to avoid per-step saturation.
            dynamic_clip = min(_clip_max, max(_clip_min, _clip_k * _grad_norm_ema))
            if _use_dynamic_clip:
                effective_clip = dynamic_clip
            else:
                effective_clip = max(grad_clip, dynamic_clip)

            base = model._orig_mod if hasattr(model, "_orig_mod") else model
            pre_clip_norm_t = torch.nn.utils.clip_grad_norm_(base.parameters(), effective_clip)
            pre_clip_norm = float(pre_clip_norm_t) if torch.is_tensor(pre_clip_norm_t) else float(pre_clip_norm_t)

            # Update gradient norm EMA (R4 mitigation: skip NaN/Inf)
            if _use_dynamic_clip and math.isfinite(pre_clip_norm):
                _grad_norm_ema = _clip_ema_alpha * pre_clip_norm + (1 - _clip_ema_alpha) * _grad_norm_ema
                self._grad_norm_ema = _grad_norm_ema  # Store for checkpoint save

            clipped_this_step = math.isfinite(pre_clip_norm) and (pre_clip_norm > float(effective_clip))
            _clip_hist.append(1 if clipped_this_step else 0)
            clip_pct = 100.0 * (sum(_clip_hist) / max(1, len(_clip_hist)))

            # Opt-in step diagnostics: Phase 3 (after clip) + log output
            if _step_diag_ctx.active:
                _step_diag.collect_phase3_postclip_grads(_step_diag_ctx, model, self.logger)
                _step_diag.log_step_diagnostics(_step_diag_ctx, accum_loss, self.logger)

            self._last_pre_clip_norm = pre_clip_norm  # Store for diagnostics
            grad_norm = min(pre_clip_norm, float(effective_clip)) if math.isfinite(pre_clip_norm) else float("nan")

            # Clip coefficient implied by global norm (uniform scaling factor).
            clip_coef = effective_clip / (pre_clip_norm + 1e-12)
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
                    signal.signal(signal.SIGINT, _original_sigint_handler)
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
                    signal.signal(signal.SIGINT, _original_sigint_handler)
                    metrics.end_time = time.time()
                    return metrics

                # Spike detected (not halting): apply LR cooldown and continue
                if spike_detected:
                    lr = lr_effective
                    self.logger.warning(f"  🔻 Spike response: mild LR cooldown x{_grad_spike_lr_factor}")

            # =================================================================
            # MoE LR Re-Warmup Logic (for mid-run optimizer reset)
            # =================================================================
            # When resuming with new per-component LR groups (momentum states lost),
            # linearly ramp LR from 0 to target over moe_lr_rewarmup_steps.
            moe_rewarmup_steps = int(getattr(self.config, "moe_lr_rewarmup_steps", 0) or 0)
            moe_rewarmup_start = int(getattr(self, "_moe_rewarmup_start_step", 0) or 0)
            if moe_rewarmup_steps > 0 and moe_rewarmup_start > 0:
                rel_step = step - moe_rewarmup_start
                if 0 <= rel_step < moe_rewarmup_steps:
                    # Linear ramp from 0 to 1
                    rewarmup_frac = (rel_step + 1) / moe_rewarmup_steps
                    lr = lr * rewarmup_frac
                    if step % 25 == 0:
                        self.logger.info(f"  [MoE Re-Warmup] step {rel_step+1}/{moe_rewarmup_steps} | LR frac: {rewarmup_frac:.3f}")

            # =================================================================
            # Per-Component LR Application
            # =================================================================
            # Apply LR to each parameter group with its configured scale factor.
            # This ensures experts get slower LR and routers get faster LR.
            if not self._skip_lr_schedule:
                use_per_component = getattr(self, "_use_per_component_lr", False)
                for pg in param_groups:
                    # Each group has lr_scale stored from _setup_optimizer
                    lr_scale = float(pg.get("lr_scale", 1.0))
                    pg["lr"] = lr * lr_scale
            if use_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            # Log peak VRAM for the first couple optimizer steps (avoid log spam)
            if _track_peak_vram and step < (start_step + 2):
                try:
                    torch.cuda.synchronize(device=_cuda_device_idx)
                    peak_alloc = float(torch.cuda.max_memory_allocated(device=_cuda_device_idx))
                    peak_reserved = float(torch.cuda.max_memory_reserved(device=_cuda_device_idx))
                    gib = 1024.0 ** 3
                    self.logger.info(
                        f"[VRAM] peak_alloc={peak_alloc / gib:.2f} GiB, peak_reserved={peak_reserved / gib:.2f} GiB (after optimizer step {step + 1})"
                    )
                except Exception:
                    pass
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

            # Record step for SafeOptimizations monitoring (auto-disables failing optimizations)
            if self._safe_opts is not None:
                try:
                    safe_anomalies = self._safe_opts.record_step(
                        step=step,
                        loss=float(accum_loss),
                        grad_norm=float(grad_norm) if math.isfinite(grad_norm) else None,
                        tokens_per_sec=float(tps),
                    )
                    # Log anomalies that triggered optimization fallbacks
                    for anomaly in safe_anomalies:
                        if anomaly.anomaly_type in ("loss_spike", "nan_grad", "inf_grad"):
                            self.logger.warning(
                                f"  [SafeOpts] {anomaly.optimization} anomaly: "
                                f"{anomaly.anomaly_type} (value={anomaly.value:.4f})"
                            )
                            # Log to W&B
                            if wandb_run is not None and wandb_mod is not None:
                                try:
                                    wandb_mod.log({
                                        f"safe_opts/{anomaly.optimization}_anomaly": 1,
                                        f"safe_opts/{anomaly.optimization}_anomaly_value": anomaly.value,
                                    }, step=step)
                                except Exception:
                                    pass
                            # Log to TensorBoard
                            if tb_writer is not None:
                                try:
                                    tb_writer.add_scalar(
                                        f"safe_opts/{anomaly.optimization}_anomaly",
                                        anomaly.value,
                                        step,
                                    )
                                except Exception:
                                    pass
                except Exception:
                    pass  # Non-fatal, don't interrupt training

            # EMA debug: trace loss -> EMA flow
            _ema_debug = getattr(self.config, "ema_debug", False)
            if _ema_debug and step % 25 == 0:
                self.logger.info(
                    f"  [EMA_DEBUG] step={step} mode=train input_loss={accum_loss:.4f} -> ema_loss={metrics.ema_loss:.4f} "
                    f"(ce_ema={self._ce_ema:.4f})"
                )

            # Halt policy: NaN/Inf loss or sustained EMA degradation
            halt, halt_reason = _halt_controller.check_loss_finite(float(accum_loss))
            if halt:
                self.logger.warning(f"  🛑 HALT at step {step}: {halt_reason}")
                self._save_checkpoint(step)
                if profiler:
                    profiler.stop()
                signal.signal(signal.SIGINT, _original_sigint_handler)
                metrics.end_time = time.time()
                metrics.final_loss = accum_loss
                return metrics

            try:
                spikes_in_window = _spike_tracker.spike_count_in_window()
                halt, halt_reason = _halt_controller.update(
                    step=step,
                    ema_loss=metrics.ema_loss,
                    spikes_in_window=spikes_in_window,
                    logger=self.logger,
                )
                if halt:
                    self.logger.warning(f"  🛑 HALT at step {step}: {halt_reason}")
                    self._save_checkpoint(step)
                    if profiler:
                        profiler.stop()
                    signal.signal(signal.SIGINT, _original_sigint_handler)
                    metrics.end_time = time.time()
                    metrics.final_loss = accum_loss
                    return metrics
            except Exception:
                pass
            if use_mod_mor:
                base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
                
                # CurriculumController handles all MoR/MoD gating decisions
                routing_stats = None
                if step % 100 == 0 and hasattr(base_model, "get_routing_stats"):
                    routing_stats = base_model.get_routing_stats()
                
                curriculum_actions = self._curriculum_controller.step(
                    step=step,
                    ce_ema=self._ce_ema,
                    routing_stats=routing_stats,
                )
                
                for action in curriculum_actions:
                    if action.action_type == "update_mod_loss_ema":
                        if hasattr(base_model, "update_mod_loss_ema"):
                            base_model.update_mod_loss_ema(action.params["ce_ema"])
                    elif action.action_type == "trigger_mor_early":
                        if hasattr(base_model, "trigger_mor_early"):
                            base_model.trigger_mor_early(step, rampup_steps=action.params["rampup_steps"])
                            self._mor_enable_step = step
                            self._mor_triggered_by_loss = True
                        if action.log_message:
                            self.logger.info(action.log_message)
                    elif action.action_type == "trigger_mod_from_mor":
                        # Deprecated: MoD enablement is step-based warmup, not MoR-informed.
                        # Keep this branch as a no-op for backward compatibility with old
                        # CurriculumController configs.
                        if action.log_message:
                            self.logger.info(action.log_message)
                            
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
                # Per-component eval (only when eval preset is a mix and components were built)
                if fixed_eval_batches_by_component:
                    _base_for_eval = model._orig_mod if hasattr(model, "_orig_mod") else model
                    _base_for_eval.eval()
                    comp_losses: Dict[str, float] = {}
                    for comp_name, comp_batches in fixed_eval_batches_by_component.items():
                        try:
                            comp_loss = evaluate_fixed_batches(
                                base_model=_base_for_eval,
                                fixed_eval_batches=comp_batches,
                                device=device,
                                dtype=dtype,
                                config=self.config,
                                use_mod_mor=use_mod_mor,
                                eval_debug=False,
                            )
                            comp_losses[comp_name] = float(comp_loss)
                        except Exception as e:
                            self.logger.warning(
                                f"⚠️  Eval component '{comp_name}' failed: {type(e).__name__}: {e}"
                            )
                    _base_for_eval.train()

                    if comp_losses:
                        # Stable, readable ordering
                        comp_items = sorted(comp_losses.items(), key=lambda kv: kv[0])
                        parts = []
                        for n, v in comp_items:
                            w = eval_component_weights.get(n)
                            if w is None:
                                parts.append(f"{n}={v:.4f}")
                            else:
                                parts.append(f"{n}({w:.2f})={v:.4f}")
                        self.logger.info(f"[EVAL_COMPONENTS] step={step}  " + "  ".join(parts))
                        if self.config.use_wandb and wandb_mod is not None:
                            payload = {f"eval_loss/{k}": float(v) for k, v in comp_losses.items()}
                            payload["step"] = step
                            wandb_mod.log(payload)
                        if tb_writer is not None:
                            for k, v in comp_losses.items():
                                tb_writer.add_scalar(f"eval/loss_{k}", float(v), step)

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
                        last_ce_loss=float(self._last_ce_loss.item() if hasattr(self._last_ce_loss, "item") else self._last_ce_loss),
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
                if self._source_name_counts:
                    items = sorted(self._source_name_counts.items(), key=lambda kv: (-kv[1], kv[0]))
                    parts = [f"{k}={v}" for k, v in items[:12]]
                    suffix = "" if len(items) <= 12 else f" (+{len(items) - 12} more)"
                    self.logger.info("Batch sources (window): " + ", ".join(parts) + suffix)
                    self._source_name_counts = {}

                elapsed = time.time() - metrics.start_time
                steps_this_session = step - start_step
                tokens_this_session = steps_this_session * tokens_per_step
                avg_tps = tokens_this_session / elapsed if elapsed > 0 else 0
                steps_per_sec = steps_this_session / elapsed if elapsed > 0 else 0
                # Build dynamic clip info string if enabled
                clip_info = f"post {grad_norm:.2e}"
                if _use_dynamic_clip:
                    clip_info = f"post {grad_norm:.2e} (dyn={effective_clip:.0f})"
                self.logger.info(
                    f"Step {step:5d}/{max_steps} | "
                    f"Loss: {accum_loss:.4f} (EMA: {metrics.ema_loss:.4f}) | "
                    f"CE: {accum_ce_f:.4f} (EMA: {getattr(self, '_ce_ema', 0.0):.4f}) | "
                    f"LR: {lr:.2e} | "
                    f"Grad: pre {pre_clip_norm:.2e} | {clip_info} | clipped {clip_pct:4.1f}% | "
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
                    if _use_dynamic_clip:
                        tb_writer.add_scalar("train/grad_clip_dynamic", float(effective_clip), step)
                        tb_writer.add_scalar("train/grad_norm_ema", float(_grad_norm_ema), step)
                    tb_writer.add_scalar("train/tps", float(tps), step)
                    # Log MoE aux loss if MoE is enabled
                    if self.config.moe_enabled:
                        moe_val = float(self._last_moe_aux_loss.item() if hasattr(self._last_moe_aux_loss, "item") else self._last_moe_aux_loss)
                        tb_writer.add_scalar("train/moe_aux_loss", moe_val, step)
                
                if self.config.use_wandb and wandb_mod is not None:
                    wandb_payload = {
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
                    # Log MoE aux loss if MoE is enabled
                    if self.config.moe_enabled:
                        moe_val = float(self._last_moe_aux_loss.item() if hasattr(self._last_moe_aux_loss, "item") else self._last_moe_aux_loss)
                        wandb_payload["moe_aux_loss"] = moe_val
                    wandb_mod.log(wandb_payload)
                if use_mod_mor and step % 100 == 0:
                    self._log_layer_diagnostics(step, accum_loss, lr, grad_norm, ce_step=accum_ce_f)
            
            # MoE divergence tracking (optional, adds CPU overhead)
            if (self.config.moe_enabled and 
                getattr(self.config, "moe_track_divergence", False) and 
                step % getattr(self.config, "moe_divergence_interval", 100) == 0):
                try:
                    divergence, entropy, util = self._compute_moe_divergence()
                    util_str = "/".join([f"{u*100:.0f}" for u in util]) if util else "N/A"
                    # Cosine distance: 0=identical, 1=orthogonal (good), 2=opposite
                    # >0.5 means experts are reasonably different, >0.9 means nearly orthogonal
                    status = "✅ Spec" if divergence > 0.5 else "⏳ Learn"
                    self.logger.info(
                        f"  [MoE] step={step} div={divergence:.4f} ent={entropy:.4f} util=[{util_str}%] {status}"
                    )
                    # Warn if divergence is falling significantly (experts reconverging)
                    if hasattr(self, "_last_moe_divergence") and divergence < self._last_moe_divergence - 0.01:
                        self.logger.warning(
                            f"  ⚠️  MoE divergence FALLING: {self._last_moe_divergence:.4f} → {divergence:.4f} (experts reconverging!)"
                        )
                    self._last_moe_divergence = divergence
                    
                    # Log to tensorboard/wandb
                    if tb_writer is not None:
                        tb_writer.add_scalar("moe/divergence", divergence, step)
                        tb_writer.add_scalar("moe/entropy", entropy, step)
                    if self.config.use_wandb and wandb_mod is not None:
                        wandb_mod.log({"moe_divergence": divergence, "moe_entropy": entropy, "step": step})
                except Exception as e:
                    self.logger.warning(f"  MoE divergence check failed: {e}")
            
            if step % save_interval == 0:
                self._save_checkpoint(step)
                if self._should_early_stop(accum_loss if accum_loss > 0 else metrics.losses[-1], step, self._count_params()):
                    self.logger.warning(f"\n⚠️  EARLY STOPPING at step {step}")
                    self.logger.warning(f"   Loss has increased for {config.early_stop_patience} consecutive checkpoints")
                    self.logger.warning(f"   Checkpoint losses: {self._checkpoint_losses[-4:]}")
                    break
            
            # Check for graceful shutdown request (Ctrl+C)
            if _interrupt_requested:
                self.logger.info(f"\n🛑 Graceful shutdown at step {step}")
                self._save_checkpoint(step)
                self.logger.info(f"   Checkpoint saved. Exiting.")
                break
            
            accum_loss = 0.0
        
        # Restore original signal handler
        signal.signal(signal.SIGINT, _original_sigint_handler)
        
        if profiler:
            profiler.stop()

        metrics.end_time = time.time()
        metrics.final_loss = metrics.losses[-1] if metrics.losses else 0.0

        self.logger.info("-" * 70)
        self.logger.info("Training complete!")

        # Log SafeOptimizations final status
        if self._safe_opts is not None:
            try:
                status = self._safe_opts.get_status_summary()
                anomalies = self._safe_opts.get_anomaly_summary()
                self.logger.info(f"SafeOptimizations status: {status}")
                if anomalies:
                    self.logger.info(f"SafeOptimizations anomalies: {anomalies}")
            except Exception:
                pass

        if self._adaptive_lr is not None and self._adaptive_lr.swa_n > 0:
            base_model = self.model
            if hasattr(base_model, "_orig_mod"):
                base_model = base_model._orig_mod
            self._adaptive_lr.apply_swa(base_model)

        self._save_checkpoint(step, final=True)
        self._generate_report()
        self._save_diagnostics()
        self._update_training_db()

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

    def _log_layer_diagnostics(
        self, step: int, loss: float, lr: float, grad_norm: float, *, ce_step: float
    ) -> None:
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
                # Step-mean CE over the full grad-accum window (matches CE_EMA / curriculum objective).
                "ce": float(ce_step),
                # Last-microbatch CE for debugging only (can be noisy/misleading).
                "ce_micro_last": float(self._last_ce_loss.item()) if hasattr(self._last_ce_loss, "item") else float(self._last_ce_loss),
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

        # MoE health recording: divergence, entropy, utilization (averaged across MoE layers)
        try:
            moe_result = self._compute_moe_divergence()
            if isinstance(moe_result, tuple) and len(moe_result) == 3:
                moe_div, moe_ent, moe_util = moe_result
                util_pct = [round(u * 100.0, 1) for u in moe_util] if isinstance(moe_util, list) else []
                # Append a compact MoE summary to diagnostics data
                self._diagnostics_data[-1]["moe"] = {
                    "divergence": float(moe_div) if moe_div is not None else None,
                    "entropy": float(moe_ent) if moe_ent is not None else None,
                    "utilization_pct": util_pct,
                }
        except Exception:
            # Keep diagnostics robust; MoE metrics are optional.
            pass

        # =========================
        # MoR auto-nudge controller
        # =========================
        # If MoR collapses to min-depth in adaptive phases, temporarily damp the
        # advantage loss (router gradients) for a short window, then restore.
        # This is intentionally conservative: it does not change LR/optimizer
        # state and it is applied outside compiled regions.
        if bool(getattr(self.config, "mor_auto_nudge", True)) and bool(getattr(self.config, "mor_adaptive", False)):
            # Auto-reset when the window ends.
            if self._mor_advantage_nudge_active and step >= self._mor_advantage_nudge_until_step:
                self.config.mor_advantage_loss_mult = 1.0
                self._mor_advantage_nudge_active = False
                self.logger.info(
                    f"MoR auto-nudge ended at step {step:,}: mor_advantage_loss_mult reset to 1.0"
                )

            # Trigger if any layer shows collapse in adaptive phases.
            # Handles both depth-0 collapse AND min-depth collapse (when mor_min_depth > 0).
            collapse_thr = float(getattr(self.config, "mor_collapse_depth0_threshold", 0.90) or 0.90)
            min_depth = int(getattr(self.config, "mor_min_depth", 0) or 0)
            
            # Check for any type of shallow collapse:
            # - DEPTH_COLLAPSED_EARLY: stuck at depth 0
            # - DEPTH_COLLAPSED: >90% at any single depth (including min_depth)
            collapse_statuses = {"DEPTH_COLLAPSED_EARLY", "DEPTH_COLLAPSED"}
            collapse_detected = any(
                (layer.get("status") in collapse_statuses) for layer in record.get("mor_layers", [])
            )
            if (
                collapse_detected
                and (not self._mor_advantage_nudge_active)
                and step >= self._mor_advantage_nudge_cooldown_until_step
                and mor_phase_raw in ["full-adaptive", "rampup"]
            ):
                # Confirm collapse ratio if histogram present; this avoids nudging
                # on weak/noisy stats. Check if collapsed to depth 0 or min_depth.
                confirmed = False
                collapse_depth = -1
                for layer in record.get("mor_layers", []):
                    if layer.get("status") not in collapse_statuses:
                        continue
                    hist = layer.get("depth_histogram", []) or []
                    total = float(sum(hist) or 0.0)
                    if total <= 0.0:
                        continue
                    # Check collapse at depth 0
                    if len(hist) > 0 and float(hist[0]) / total >= collapse_thr:
                        confirmed = True
                        collapse_depth = 0
                        break
                    # Check collapse at min_depth (when min_depth > 0)
                    if min_depth > 0 and len(hist) > min_depth:
                        if float(hist[min_depth]) / total >= collapse_thr:
                            confirmed = True
                            collapse_depth = min_depth
                            break
                if confirmed:
                    # Choose action based on collapse type:
                    # - Depth-0 collapse: BOOST advantage to encourage deeper routing
                    # - Min-depth collapse: DAMPEN advantage to reduce penalty on depth
                    if collapse_depth == 0:
                        nudge_mult = float(getattr(self.config, "mor_advantage_nudge_mult", 2.0) or 2.0)
                        action = "boost"
                    else:
                        # Min-depth collapse: negative advantage penalizes depth
                        nudge_mult = float(getattr(self.config, "mor_advantage_nudge_damp", 0.1) or 0.1)
                        action = "damp"
                    # Clamp to safe range
                    nudge_mult = max(0.0, min(10.0, nudge_mult))
                    duration = int(getattr(self.config, "mor_advantage_nudge_duration_steps", 200) or 200)
                    cooldown = int(getattr(self.config, "mor_advantage_nudge_cooldown_steps", 500) or 500)
                    duration = max(1, duration)
                    cooldown = max(duration, cooldown)

                    self.config.mor_advantage_loss_mult = nudge_mult
                    self._mor_advantage_nudge_until_step = step + duration
                    self._mor_advantage_nudge_cooldown_until_step = step + cooldown
                    self._mor_advantage_nudge_active = True
                    self.logger.warning(
                        f"MoR auto-nudge: depth-{collapse_depth} collapse detected (phase={mor_phase_raw}). "
                        f"{action.upper()}ing advantage loss: mor_advantage_loss_mult={nudge_mult:.2f} "
                        f"for {duration} steps (cooldown {cooldown} steps)."
                    )
        
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
                _metric = str(getattr(self.config, "adaptive_metric", "train") or "train")
                adaptive_info = f" | LR_EMA({_metric}): {ema_s:.3f}/{ema_l:.3f} {trend}"
            adv_str = f" adv={self._last_advantage_loss:.4f}" if self._last_advantage_loss != 0 else ""
            moe_str = f" moe_aux={self._last_moe_aux_loss:.4f}" if self.config.moe_enabled and self._last_moe_aux_loss != 0 else ""
            lr_info = f" | LR={lr:.2e} (min={self.config.min_lr:.2e})"
            warmup = int(getattr(self.config, "mod_mlp_warmup_steps", 1000) or 1000)
            warmup = max(0, warmup)
            is_on = (mod_mode not in ("N/A", "DISABLED")) and (float(getattr(self.config, "mod_capacity", 1.0)) < 1.0)
            mod_status = "MoD:ON" if is_on else (f"MoD:WARMUP(<{warmup})" if step < warmup else "MoD:OFF")
            mod_gate = f" | CE_EMA={getattr(self, '_ce_ema', 0.0):.3f} {mod_status}"
            self.logger.info(
                f"  [DIAG] MoD:{mod_mode} save={mod_savings:.0f}% | MoR:{mor_phase} d={mor_depth:.2f} [{depth_dist}%] | "
                f"CE={float(ce_step):.3f} aux={self._last_aux_loss:.4f} ponder={self._last_ponder_loss:.3f}{adv_str}{moe_str}{adaptive_info}{lr_info}{mod_gate}"
            )
            self._save_diagnostics()

    def _save_diagnostics(self) -> None:
        _checkpointing.save_diagnostics(
            checkpoint_dir=self.config.checkpoint_dir,
            diagnostics_data=self._diagnostics_data,
            logger=self.logger,
            run_id=self.config.run_id,
        )

    def _update_training_db(self) -> None:
        """Load diagnostics JSON into training database for cross-run analysis."""
        try:
            # Extract model_id from run_id (e.g., "500m_20260109_144733" -> "500m")
            run_id = self.config.run_id
            model_id = run_id.split("_")[0] if "_" in run_id else run_id
            
            # Find the diagnostics file for this run
            diag_path = Path(self.config.checkpoint_dir) / f"diagnostics_{run_id}.json"
            if not diag_path.exists():
                self.logger.info(f"   No diagnostics file to load into DB: {diag_path}")
                return
            
            db = _db.TrainingDB()
            count = db.load_diagnostics_json(diag_path, model_id=model_id, run_id=run_id)
            
            # Also load the training report if it exists
            report_path = Path(self.config.log_dir) / f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            # Reports are saved after this method, so try to find the latest one
            reports_dir = Path(self.config.log_dir).parent / "reports"
            if reports_dir.exists():
                report_files = sorted(reports_dir.glob("training_report_*.json"))
                if report_files:
                    db.load_training_report(report_files[-1], model_id=model_id)
            
            self.logger.info(f"   📊 Training DB updated: {count} steps loaded for {model_id}")
            
            # Generate plots and report
            self._generate_training_analysis(model_id)
        except Exception as e:
            self.logger.warning(f"   ⚠️  Failed to update training DB: {e}")

    def _generate_training_analysis(self, model_id: str) -> None:
        """Generate training plots and comprehensive report."""
        try:
            # Import here to avoid circular imports and keep it optional
            import subprocess
            import sys
            
            script_path = Path(__file__).resolve().parents[2] / "scripts" / "plot_training_trends.py"
            if not script_path.exists():
                self.logger.warning(f"   Training analysis script not found: {script_path}")
                return
            
            self.logger.info(f"   📈 Generating training analysis for {model_id}...")
            
            # Run the analysis script
            result = subprocess.run(
                [sys.executable, str(script_path), "--model", model_id],
                capture_output=True,
                text=True,
                timeout=120,  # 2 minute timeout for plots
                cwd=str(Path(__file__).resolve().parents[2]),  # Run from repo root
            )
            
            if result.returncode == 0:
                self.logger.info(f"   ✅ Training analysis complete: reports/training_status_{model_id}.md")
            else:
                self.logger.warning(f"   ⚠️  Training analysis failed (exit {result.returncode})")
                if result.stderr:
                    # Show first meaningful error line
                    for line in result.stderr.split('\n'):
                        if 'Error' in line or 'Exception' in line or 'Traceback' in line:
                            self.logger.warning(f"      {line[:100]}")
                            break
        except subprocess.TimeoutExpired:
            self.logger.warning("   ⚠️  Training analysis timed out (>2min)")
        except Exception as e:
            self.logger.warning(f"   ⚠️  Training analysis error: {e}")

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
                "grad_norm_ema": float(getattr(self, "_grad_norm_ema", 0.0) or 0.0),
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
