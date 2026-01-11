"""HYDRA Training Configuration.

Contains TrainingConfig dataclass, model size presets, and config-building
utilities for constructing configs from CLI arguments.
"""
from __future__ import annotations

import argparse
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

_log = logging.getLogger(__name__)


def compute_auto_lr(params_millions: float, base_lr: float = 1e-4, base_params: float = 100.0) -> float:
    """Compute learning rate using μP-inspired scaling.
    
    LR scales as: base_lr / sqrt(params / base_params)
    
    This follows the observation that larger models need lower LR roughly proportional
    to 1/sqrt(params). Empirically validated across GPT-style models.
    
    Examples:
        100M params → 1e-4 (base)
        250M params → 6.3e-5
        500M params → 4.5e-5  
        1B params   → 3.2e-5
    """
    return base_lr / math.sqrt(params_millions / base_params)


@dataclass
class TrainingConfig:
    """Training configuration with optimized defaults for 100M variant."""

    # TRAINING MODE: "testing", "production", "chinchilla_third"
    mode: str = "testing"

    # Resume from checkpoint
    resume_from: Optional[str] = None
    start_step: int = 0

    # MODEL ARCHITECTURE
    architecture: str = "mod_mor"
    attention_backend: str = "ccgqa"  # Only CCGQA is supported
    mod_capacity: float = 0.5
    mor_adaptive: bool = True

    # MoD Warmup (step-based)
    # MoD runs dense MLP for the first `mod_mlp_warmup_steps`, then switches to
    # hard top-k routing (compute skipping). This is intentionally step-based
    # (not MoR/CE-gated) because the MoR-informed trigger was too brittle.
    mod_mlp_warmup_steps: int = 1000

    # MoR Curriculum (enables first, informs MoD)
    # MoR enables when: step >= max(mor_enable_min_steps, mor_enable_pct * max_steps) OR loss < mor_enable_loss_threshold
    mor_enable_pct: float = 0.10  # Enable MoR after 10% of training
    mor_enable_min_steps: int = 3000  # Minimum steps before MoR enables (floor for short runs)
    mor_enable_loss_threshold: float = 5.0  # OR enable when CE loss < 5.0
    mor_rampup_steps: int = 5000
    mor_already_enabled: bool = False

    # NOTE: The previous MoR-informed MoD trigger (early-exit + loss thresholds)
    # is kept for back-compat with older configs/CLI, but is no longer used to
    # enable MoD. MoD is enabled purely by step warmup + mod_capacity.
    mod_enable_min_step: int = 3000
    mod_enable_mor_early_exit_threshold: float = 0.38
    mod_enable_loss_threshold: float = 4.5

    # Loss-aware MoD supervision (teacher = top-k per-token CE loss)
    # 0.0 disables; typical starting point: 0.1 - 1.0
    mod_loss_aware_weight: float = 0.0

    # Auxiliary loss scales
    aux_scale: float = 0.1
    ponder_scale: float = 0.01

    # MoR: scales the loss-driven advantage signal that encourages deeper
    # recursions on hard tokens and shallower exits on easy tokens.
    # WARNING: Values above 0.05 can cause "advantage gaming" where the model
    # optimizes for efficiency over learning. Default 0.02 is conservative.
    mor_advantage_loss_scale: float = 0.02  # Reduced from 0.1 to prevent gaming

    # MoR minimum depth: prevents depth-0 collapse where model exits immediately.
    # 0 = allow immediate exit (dangerous for learning)
    # 1 = force at least one full recursion (recommended for MoE training)
    mor_min_depth: int = 1  # Default to 1 to prevent collapse

    # MoR runtime control: multiplier applied to the computed advantage loss.
    # This allows the trainer to temporarily damp/router gradients without
    # changing optimizer LR or recompiling.
    mor_advantage_loss_mult: float = 1.0

    # MoR auto-nudge: when routing collapses (>90% at any single depth),
    # automatically adjust the advantage loss to help recover.
    # 
    # For DEPTH-0 collapse (early exit): advantage is positive (hard tokens),
    # so we BOOST the advantage loss to encourage deeper routing.
    #
    # For MIN-DEPTH collapse (stuck at floor): advantage becomes negative
    # (easy tokens dominate), and negative advantage + low depth = penalty
    # for going deeper. We DAMPEN (set mult < 1.0) to reduce this conflict.
    mor_auto_nudge: bool = True
    mor_collapse_depth0_threshold: float = 0.90
    mor_advantage_nudge_mult: float = 2.0  # Boost factor for depth-0 collapse
    mor_advantage_nudge_damp: float = 0.1  # Dampen factor for min-depth collapse
    mor_advantage_nudge_duration_steps: int = 50
    mor_advantage_nudge_cooldown_steps: int = 500

    # ==========================================================================
    # Static Routing Mode (CUDA Graph Compatible)
    # ==========================================================================
    # When enabled, MoD/MoR compute ALL tokens through ALL layers, then apply
    # routing masks AFTER computation. This trades compute efficiency for CUDA
    # graph compatibility (5-15% speedup from reduced kernel launch overhead).
    #
    # Benefits:
    # - Full CUDA graph capture/replay works
    # - Router still learns (gradients flow through soft masks)
    # - Can switch to dynamic routing at inference for compute savings
    #
    # Tradeoffs:
    # - ~25-50% more FLOPs in MLP (MoD normally skips 50% of tokens)
    # - MoR runs all recursions (normally exits early for easy tokens)
    static_routing_mode: bool = False

    # ==========================================================================
    # Mixture of Experts (MoE) Configuration
    # ==========================================================================
    # MoE adds sparse FFN expert routing for higher model capacity at constant
    # compute cost. MoE layers are inserted between existing transformer blocks.
    #
    # When moe_enabled=False (default), the model is identical to baseline.
    # When enabled, MoE layers use top-k routing to select experts per token.
    moe_enabled: bool = False
    moe_num_experts: int = 4  # Number of expert FFN networks (auto-scaled if 0)
    moe_num_layers: int = 2   # Number of MoE layers to insert (auto-scaled if 0)
    moe_top_k: int = 1        # Experts per token (1=top-1 routing, 2=top-2)
    moe_aux_weight: float = 0.0001  # Load-balancing aux loss weight (very low to allow specialization)
    moe_warmup_steps: int = 1000  # Steps before full MoE contribution (for checkpoint cloning)
    moe_capacity_factor: float = float("inf")  # Expert capacity (inf=no dropping)
    moe_router_jitter: float = 0.15  # Noise for router exploration (helps break symmetry)
    moe_expert_diversity_noise: float = 0.05  # Additive weight noise to break expert symmetry
    moe_identity_init: bool = True  # Identity-preserving init for checkpoint cloning
    moe_track_divergence: bool = True  # Log expert divergence every N steps (CPU cost)
    moe_divergence_interval: int = 500  # Steps between divergence checks
    moe_forced_routing_steps: int = 0  # Steps to force position-based routing for diversification (0=disabled)

    # Optional domain teacher / domain forcing
    moe_domain_expert_map: str = ""  # e.g. "math:0,code:1,chat:2,pleias_synth:3"
    moe_teacher_weight: float = 0.0  # alpha in: loss += alpha * CE(router_logits, target_expert)
    moe_teacher_until_step: int = 0  # 0=forever, else apply teacher until this global step

    # Per-component LR scaling for MoE gradient stabilization
    # Use when upcycled experts explode while routers learn too slowly.
    # These defaults ALWAYS activate when MoE is enabled.
    moe_expert_lr_scale: float = 0.5  # LR multiplier for expert/MLP params (slower to prevent explosion)
    moe_router_lr_scale: float = 3.0  # LR multiplier for router/gate params (routers need faster learning)
    moe_lr_rewarmup_steps: int = 0    # Steps to re-warmup LR after mid-run optimizer reset (0=disabled)
    moe_expert_weight_decay_scale: float = 3.0  # WD multiplier for experts (prevents weight explosion)
    moe_reset_optimizer_state: bool = False  # Reset optimizer moments for MoE params on resume

    # Model config
    dim: int = 768
    n_macro_blocks: int = 3
    n_heads: int = 12
    n_kv_heads: int = 3

    # Model config - MOD+MOR
    model_size: str = "100M"
    mod_mor_dim: int = 1280
    n_mor_blocks: int = 8
    mor_recursions: int = 3
    mod_mor_n_heads: int = 20
    mod_mor_n_kv_heads: int = 5

    vocab_size: int = 50257
    max_seq_len: int = 2048

    # Step-based parameters (overwritten in __post_init__)
    max_steps: int = 5000
    warmup_steps: int = 150
    decay_start_step: int = 3500
    decay_steps: int = 1500
    seq_steps: Tuple[Tuple[int, int], ...] = ()
    save_interval: int = 1000

    # Training hyperparameters
    batch_size: int = 8
    grad_accum_steps: int = 2
    max_lr: float = 1e-4  # Conservative for 250M+; 3e-4 was causing instability
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    # NOTE: grad_clip=5.0 is calibrated for deep MoR models (8 blocks × 4 rec = 32 layers).
    # Pre-clip grad norms ~20-25 are normal; clip=1.0 throttled effective LR by 20x.
    grad_clip: float = 5.0

    # Dynamic gradient clipping: adapts clip threshold based on gradient norm EMA.
    # When enabled, clip = min(grad_clip_max, max(grad_clip_min, k * grad_norm_ema))
    # This allows the model to adapt to regime changes (e.g., MoE router activation).
    grad_clip_dynamic: bool = True  # Enable dynamic clipping (recommended for MoE)
    grad_clip_k: float = 2.0  # Multiplier on EMA (allow 2x "normal" gradient)
    grad_clip_min: float = 50.0  # Floor to prevent over-aggressive clipping
    grad_clip_max: float = 3000.0  # Ceiling to prevent runaway (allows MoE adaptation)
    grad_clip_ema_alpha: float = 0.05  # EMA decay (0.05 = ~20 step half-life)

    # WSD Scheduler
    lr_schedule: str = "wsd_adaptive"

    # Adaptive LR settings
    adaptive_lr: bool = True
    # Which signal drives adaptive cooldown triggering.
    # - 'train': uses training loss (more reactive; can overreact during curricula transitions)
    # - 'eval': uses eval loss at eval intervals (more stable; less frequent updates)
    adaptive_metric: str = "eval"
    adaptive_patience: int = 3
    adaptive_threshold: float = 0.05
    # Guardrail: do not allow adaptive cooldown to trigger before this fraction
    # of the run has completed (prevents early collapse to min_lr during the
    # volatile curriculum ramp phase).
    adaptive_min_trigger_pct: float = 0.50
    adaptive_cooldown_factor: float = 0.5
    adaptive_lr_reduction: float = 0.5

    # Resume LR behavior
    # By default we align the scheduled LR to the checkpoint optimizer LR at the
    # resume step (so restarts continue smoothly). If you saved at min_lr and
    # want to jump back to the schedule, set resume_ignore_ckpt_lr=True.
    resume_ignore_ckpt_lr: bool = False
    # Optional: force a specific LR at the resume step. If set (>0), it wins
    # over resume_ignore_ckpt_lr and checkpoint alignment.
    resume_lr_override: float = 0.0

    # Stochastic Weight Averaging
    use_swa: bool = False
    swa_start_pct: float = 0.75
    swa_lr_factor: float = 0.5

    # Batch filtering
    batch_filter: bool = False
    batch_filter_threshold: float = 2.5
    batch_filter_max_skip: float = 0.05

    # Dataset
    dataset_name: str = "pretrain_default"
    # Optional: explicit dataset preset for evaluation. If None, trainer picks a sensible
    # default based on dataset_name (e.g. uses a pretrain-like eval mix for pretrain_* runs).
    eval_dataset_name: Optional[str] = None
    tokenizer_name: str = "gpt2"

    # Optimization
    use_compile: bool = True
    compile_mode: str = "max-autotune-no-cudagraphs"
    use_triton_kernels: bool = True
    dtype: str = "bfloat16"

    # Memory optimization
    use_chunked_ce: bool = True
    chunked_ce_size: int = 4096
    use_liger_ce: bool = True
    gradient_checkpointing: bool = True

    # Logging & Observability
    # NOTE: log_interval/eval_interval/seed are defined later in the dedicated
    # Logging/Reproducibility sections to avoid duplicate dataclass fields.
    
    # Run identification - auto-generated if not provided
    # Used for naming diagnostics, logs, and other run-specific artifacts
    run_id: Optional[str] = None  # Auto-set to "{model_size}_{timestamp}" in __post_init__
    
    # Experiment Tracking
    use_wandb: bool = False
    wandb_project: str = "hydra-llm"
    wandb_entity: Optional[str] = None
    run_name: Optional[str] = None
    
    use_tensorboard: bool = False
    tensorboard_dir: str = "runs"
    log_dir: str = "logs"
    
    # Profiling
    use_profiler: bool = False
    profiler_dir: str = "profiler_traces"

    checkpoint_every_n: int = 2
    use_8bit_adam: bool = False
    use_adafactor: bool = False

    # Debug/stability
    halt_on_spike: bool = False
    ema_debug: bool = False  # Print EMA updates every 25 steps to trace loss->EMA flow
    eval_debug: bool = False  # Run eval sanity check on training batch at first eval

    # Logging
    log_interval: int = 50
    eval_interval: int = 500

    # Reproducibility
    seed: Optional[int] = None

    # Checkpoint management
    max_checkpoints: int = 3

    # Early stopping
    early_stop_patience: int = 3
    early_stop_threshold: float = 0.10
    chinchilla_multiplier: float = 20.0
    early_stop_min_progress: float = 0.50

    # Paths
    checkpoint_dir: str = "checkpoints"
    report_dir: str = "reports"

    def __post_init__(self) -> None:
        from datetime import datetime
        
        # Save user-provided save_interval before mode-based override
        _user_save_interval = self.save_interval if self.save_interval != 1000 else None
        
        # Auto-generate run_id if not provided
        if self.run_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_size = self.model_size.lower().replace(".", "_")
            self.run_id = f"{model_size}_{timestamp}"
        
        if self.mode == "testing":
            self.max_steps = 5000
            self.warmup_steps = 500  # Increased from 150 - large models need longer warmup
            self.decay_start_step = 3500
            self.decay_steps = 1500
            self.seq_steps = ()
            self.save_interval = 500
            self.max_seq_len = 512
        elif self.mode == "production":
            self.max_steps = 100000
            self.warmup_steps = 1000
            self.decay_start_step = 85000
            self.decay_steps = 15000
            self.seq_steps = ((50000, 512), (50000, 1024))
            self.save_interval = 1000
        elif self.mode == "chinchilla_third":
            self.max_steps = 90000
            self.warmup_steps = 500
            self.decay_start_step = 76500
            self.decay_steps = 13500
            self.seq_steps = ()
            self.max_seq_len = 512
            self.grad_accum_steps = 4
            self.save_interval = 1000
        else:
            raise ValueError(f"Unknown mode: {self.mode}. Use 'testing', 'production', or 'chinchilla_third'")
        
        # Restore user-provided save_interval if explicitly set (overrides mode default)
        if _user_save_interval is not None:
            self.save_interval = _user_save_interval

    def print_summary(self) -> None:
        _log.info("='" * 30)
        _log.info(f"TRAINING MODE: {self.mode.upper()}")
        _log.info("='" * 30)
        _log.info(f"  Max steps:      {self.max_steps:,}")
        _log.info(f"  Warmup steps:   {self.warmup_steps:,}")
        _log.info(f"  Decay starts:   {self.decay_start_step:,}")
        _log.info(f"  Decay steps:    {self.decay_steps:,}")
        _log.info(f"  Save interval:  {self.save_interval:,}")
        if self.seq_steps:
            _log.info(f"  Sequence steps: {self.seq_steps} -> {self.max_seq_len}")
            total_tokens = 0
            cumulative_steps = 0
            for steps, seq_len in self.seq_steps:
                phase_tokens = steps * self.batch_size * self.grad_accum_steps * seq_len
                total_tokens += phase_tokens
                cumulative_steps += steps
                _log.info(f"    Phase: {steps:,} steps @ {seq_len} seq -> {phase_tokens/1e6:.1f}M tokens")
            remaining = self.max_steps - cumulative_steps
            if remaining > 0:
                final_tokens = remaining * self.batch_size * self.grad_accum_steps * self.max_seq_len
                total_tokens += final_tokens
                _log.info(f"    Final: {remaining:,} steps @ {self.max_seq_len} seq -> {final_tokens/1e6:.1f}M tokens")
            _log.info(f"  Total tokens:   {total_tokens/1e9:.2f}B")
        else:
            total_tokens = self.max_steps * self.batch_size * self.grad_accum_steps * self.max_seq_len
            _log.info(f"  Sequence len:   {self.max_seq_len} (fixed)")
            _log.info(f"  Total tokens:   {total_tokens/1e6:.1f}M")

        if self.architecture == "mod_mor":
            effective_layers = self.n_mor_blocks * self.mor_recursions
            est_params = {"100M": 220, "500M": 600, "750M": 927, "900M": 860, "1B": 1160}.get(self.model_size, 220)
            _log.info("  Architecture:   MOD+MOR (HydraModel + LA3 attention)")
            _log.info(f"                  dim={self.mod_mor_dim}, {self.n_mor_blocks} MoR blocks × {self.mor_recursions} recursions = {effective_layers} effective layers")
            _log.info(f"                  ~{est_params}M params ({self.model_size} preset)")
            _log.info(f"                  MoD capacity: {self.mod_capacity:.0%} (~{(1-self.mod_capacity)*100:.0f}% compute savings)")
            
            # MoD Curriculum: now MoR-informed (waits for MoR early_exit_ratio to cross threshold)
            if self.mod_capacity >= 1.0:
                _log.info("  MoD Curriculum: OFF (capacity=1.0)")
            else:
                mor_threshold = getattr(self, "mod_enable_mor_early_exit_threshold", 0.38)
                loss_threshold = getattr(self, "mod_enable_loss_threshold", 4.5)
                _log.info(f"  MoD Curriculum: MoR-informed (waits for MoR early_exit > {mor_threshold:.0%})")
                if loss_threshold and loss_threshold > 0:
                    _log.info(f"                  AND loss < {loss_threshold:.1f}")

            # MoE info
            if self.moe_enabled:
                _log.info(f"  MoE:            ENABLED ({self.moe_num_experts} experts, {self.moe_num_layers} layers, top-{self.moe_top_k})")
                _log.info(f"                  aux_weight={self.moe_aux_weight}, jitter={self.moe_router_jitter}, warmup={self.moe_warmup_steps}")
            else:
                _log.info("  MoE:            OFF")

            # MoR banner: distinguish adaptive routing from fixed-depth-only operation.
            if not self.mor_adaptive:
                _log.info("  MoR Curriculum: OFF (mor_adaptive=false; fixed-depth only)")
            else:
                # Apply floor: MoR needs minimum steps to stabilize regardless of pct
                mor_enable_step = max(
                    self.mor_enable_min_steps,
                    int(self.max_steps * self.mor_enable_pct)
                )
                if self.mor_already_enabled:
                    _log.info("  MoR Curriculum: RESTART MODE (adaptive from step 0)")
                elif mor_enable_step >= self.max_steps:
                    _log.info("  MoR Curriculum: OFF (enable_step >= max_steps)")
                elif mor_enable_step > 0:
                    remaining_steps = self.max_steps - mor_enable_step
                    actual_rampup = min(min(self.mor_rampup_steps, 2 * mor_enable_step), remaining_steps)
                    actual_rampup = max(actual_rampup, min(100, int(self.max_steps * 0.1)))
                    _log.info(f"  MoR Curriculum: Fixed-depth until step {mor_enable_step:,} (min={self.mor_enable_min_steps}, pct={self.mor_enable_pct:.0%})")
                    mor_loss_thr = getattr(self, "mor_enable_loss_threshold", 5.0)
                    if mor_loss_thr and mor_loss_thr > 0:
                        _log.info(f"                  OR loss < {mor_loss_thr:.1f} (whichever first)")
                    _log.info(f"                  Then {actual_rampup:,} step rampup to full adaptive")
                else:
                    _log.info("  MoR Curriculum: Adaptive from start (no delay)")
        elif self.architecture == "vanilla":
            _log.info("  Architecture:   VANILLA (no MoD/MoR)")
            _log.info(f"                  dim={self.dim}, n_heads={self.n_heads}, n_kv_heads={self.n_kv_heads}")
        else:
            raise ValueError(f"Unknown architecture: {self.architecture}")
        
        # Run artifacts info
        _log.info(f"  Run ID:         {self.run_id}")
        _log.info(f"  Logs:           {self.log_dir}/training_{self.run_id}.log")
        _log.info(f"  Diagnostics:    {self.checkpoint_dir}/diagnostics_{self.run_id}.json")
        _log.info(f"  Reports:        {self.report_dir}/training_report_*.json")
        _log.info("=" * 60)

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.grad_accum_steps

    @property
    def tokens_per_step(self) -> int:
        return self.effective_batch_size * self.max_seq_len


MODEL_SIZE_CONFIGS: Dict[str, Dict[str, Any]] = {
    # Debug config: small model for fast iteration (2 blocks × 2 rec = 4 effective layers)
    # Expected grad_norm ~2-3, loss should drop to ~5.5 within 500 steps on web data.
    "debug": {
        "mod_mor_dim": 512,
        "n_mor_blocks": 2,
        "mor_recursions": 2,
        "mod_mor_n_heads": 8,
        "mod_mor_n_kv_heads": 2,
        "default_batch_size": 32,
        "default_grad_accum": 2,
        "grad_clip": 2.0,  # Lower depth = lower grad norms
        "aux_scale": 0.05,
    },
    # 50M config: DEEP model for MoD/MoR curriculum validation (8 blocks × 3 rec = 24 effective layers)
    # ~50M params. Narrower but deeper than 100M to test depth-based routing.
    "50M": {
        "mod_mor_dim": 512,  # Narrow
        "n_mor_blocks": 8,    # Deep: 8 blocks
        "mor_recursions": 3,  # 3 recursions per block
        "mod_mor_n_heads": 8,
        "mod_mor_n_kv_heads": 2,
        "default_batch_size": 24,
        "default_grad_accum": 2,
        "grad_clip": 4.0,  # 24 effective layers
        "aux_scale": 0.05,
    },
    "100M": {
        "mod_mor_dim": 768,
        "n_mor_blocks": 8,
        "mor_recursions": 4,
        "mod_mor_n_heads": 12,
        "mod_mor_n_kv_heads": 3,
        "default_batch_size": 16,
        "default_grad_accum": 4,
        "grad_clip": 5.0,  # 32 effective layers need higher clip
    },
    "debug_tall_skinny": {
        "mod_mor_dim": 384,
        "n_mor_blocks": 12,
        "mor_recursions": 2,
        "mod_mor_n_heads": 6,
        "mod_mor_n_kv_heads": 2,
        "default_batch_size": 32,
        "default_grad_accum": 1,
        "aux_scale": 0.01,
    },
    "DIAG": {
        "mod_mor_dim": 768,
        "n_mor_blocks": 8,
        "mor_recursions": 5,
        "mod_mor_n_heads": 12,
        "mod_mor_n_kv_heads": 3,
        "default_batch_size": 16,
        "default_grad_accum": 4,
        "aux_scale": 0.01,
    },
    "250M": {
        "mod_mor_dim": 1024,
        "n_mor_blocks": 10,
        "mor_recursions": 4,
        "mod_mor_n_heads": 16,
        "mod_mor_n_kv_heads": 4,
        "default_batch_size": 12,
        "default_grad_accum": 8,
        "aux_scale": 0.0173,
    },
    "300M": {
        "mod_mor_dim": 1152,
        "n_mor_blocks": 12,
        "mor_recursions": 4,
        "mod_mor_n_heads": 18,
        "mod_mor_n_kv_heads": 3,
        "default_batch_size": 10,
        "default_grad_accum": 8,
        "aux_scale": 0.02,
    },
    "500M": {
        "mod_mor_dim": 1792,
        "n_mor_blocks": 14,
        "mor_recursions": 4,
        "mod_mor_n_heads": 28,
        "mod_mor_n_kv_heads": 4,
        "default_batch_size": 4,  # Reduced for 32GB VRAM with 8bit_adam
        "default_grad_accum": 15,
        # Halve peak LR (μP auto would give ~4e-5, but 694M sees occasional o_proj spikes)
        "max_lr": 5.5e-5,
        "aux_scale": 0.0250,
    },
    "750M": {
        "mod_mor_dim": 1536,
        "n_mor_blocks": 18,
        "mor_recursions": 4,
        "mod_mor_n_heads": 24,
        "mod_mor_n_kv_heads": 4,
        "default_batch_size": 4,
        "default_grad_accum": 16,
        "aux_scale": 0.0382,
    },
    "1B": {
        "mod_mor_dim": 1792,
        "n_mor_blocks": 20,
        "mor_recursions": 4,
        "mod_mor_n_heads": 28,
        "mod_mor_n_kv_heads": 4,
        "default_batch_size": 2,
        "default_grad_accum": 30,
        "requires_8bit_adam": True,
        "aux_scale": 0.0490,
    },
    "1.5B": {
        "mod_mor_dim": 2048,
        "n_mor_blocks": 22,
        "mor_recursions": 4,
        "mod_mor_n_heads": 32,
        "mod_mor_n_kv_heads": 4,
        "default_batch_size": 1,
        "default_grad_accum": 60,
        "requires_8bit_adam": True,
        "aux_scale": 0.0685,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Config Building from CLI Args
# ─────────────────────────────────────────────────────────────────────────────

# Estimated parameter counts for μP scaling
_PARAM_ESTIMATES: Dict[str, int] = {
    "debug": 15,
    "50M": 50,
    "DIAG": 50,
    "100M": 220,
    "debug_tall_skinny": 100,
    "250M": 250,
    "300M": 300,
    "500M": 692,
    "750M": 750,
    "1B": 1000,
    "1.5B": 1500,
}


def build_config_from_args(
    args: argparse.Namespace,
    size_config: Dict[str, Any],
) -> TrainingConfig:
    """Construct TrainingConfig from parsed CLI arguments.

    Creates the base config with all settings from args and size_config.
    Does not apply LR/schedule overrides - use apply_lr_config() and
    apply_schedule_overrides() for those.

    Args:
        args: Parsed argparse namespace from build_argument_parser()
        size_config: Model size configuration dict from MODEL_SIZE_CONFIGS

    Returns:
        Base TrainingConfig (before LR/schedule/batch overrides)
    """
    return TrainingConfig(
        architecture=args.arch,
        attention_backend=args.attention,
        mode=args.mode,
        resume_from=args.resume,
        resume_ignore_ckpt_lr=args.resume_ignore_ckpt_lr,
        resume_lr_override=args.resume_lr_override,
        # MoR configuration
        mor_enable_pct=args.mor_enable_pct,
        mor_enable_min_steps=args.mor_enable_min_steps,
        mor_enable_loss_threshold=args.mor_enable_loss_threshold,
        mor_already_enabled=args.mor_already_enabled,
        mor_adaptive=(args.mor_adaptive.lower() == "true"),
        mor_advantage_loss_scale=args.mor_advantage_loss_scale,
        mor_min_depth=args.mor_min_depth,
        ponder_scale=args.ponder_scale,
        # MoD configuration
        mod_capacity=args.mod_capacity,
        mod_mlp_warmup_steps=args.mod_mlp_warmup_steps,
        mod_enable_mor_early_exit_threshold=args.mod_enable_mor_early_exit_threshold,
        mod_enable_loss_threshold=args.mod_enable_loss_threshold,
        mod_loss_aware_weight=args.mod_loss_aware_weight,
        aux_scale=0.0 if getattr(args, "mod_off", False) else size_config.get("aux_scale", args.aux_scale),
        # Model architecture from size_config
        model_size=args.model_size,
        mod_mor_dim=size_config["mod_mor_dim"],
        n_mor_blocks=size_config["n_mor_blocks"],
        mor_recursions=size_config["mor_recursions"],
        mod_mor_n_heads=size_config["mod_mor_n_heads"],
        mod_mor_n_kv_heads=size_config["mod_mor_n_kv_heads"],
        # Fixed architecture params
        dim=768,
        n_macro_blocks=3,
        n_heads=12,
        n_kv_heads=3,
        # Adaptive LR
        adaptive_lr=args.adaptive_lr,
        adaptive_metric=args.adaptive_metric,
        adaptive_min_trigger_pct=args.adaptive_min_trigger_pct,
        # SWA
        use_swa=args.use_swa,
        swa_start_pct=args.swa_start_pct,
        # Batch filtering
        batch_filter=args.batch_filter,
        batch_filter_threshold=args.batch_filter_threshold,
        # Optimization
        use_triton_kernels=args.triton_kernels,
        use_chunked_ce=args.chunked_ce,
        chunked_ce_size=args.chunked_ce_size,
        static_routing_mode=getattr(args, "static_routing_mode", False),
        use_compile=args.compile,
        dtype="bfloat16",
        gradient_checkpointing=args.gradient_checkpointing,
        checkpoint_every_n=args.checkpoint_every_n,
        use_8bit_adam=args.use_8bit_adam,
        use_adafactor=args.use_adafactor,
        # Dataset
        dataset_name=args.dataset,
        # Gradient clipping
        grad_clip=(
            float(args.grad_clip)
            if (args.grad_clip is not None and args.grad_clip > 0)
            else size_config.get("grad_clip", 5.0)
        ),
        grad_clip_dynamic=getattr(args, "grad_clip_dynamic", False),
        grad_clip_k=getattr(args, "grad_clip_k", 2.0),
        grad_clip_min=getattr(args, "grad_clip_min", 5.0),
        grad_clip_max=getattr(args, "grad_clip_max", 500.0),
        # Logging & observability
        log_interval=25,
        save_interval=args.save_interval,
        seed=args.seed,
        use_wandb=args.wandb,
        wandb_project=(args.wandb_project or "hydra-llm"),
        wandb_entity=args.wandb_entity,
        run_name=args.run_name,
        use_tensorboard=args.tensorboard,
        tensorboard_dir=(args.tensorboard_dir or "runs"),
        use_profiler=args.profiler,
        profiler_dir=(args.profiler_dir or "profiler_traces"),
        # Debug flags
        halt_on_spike=args.halt_on_spike,
        ema_debug=getattr(args, "ema_debug", False),
        eval_debug=getattr(args, "eval_debug", False),
        # MoE configuration
        moe_enabled=getattr(args, "moe_enabled", False),
        moe_num_experts=getattr(args, "moe_num_experts", 0),
        moe_num_layers=getattr(args, "moe_num_layers", 0),
        moe_top_k=getattr(args, "moe_top_k", 1),
        moe_aux_weight=getattr(args, "moe_aux_weight", 0.0001),
        moe_router_jitter=getattr(args, "moe_router_jitter", 0.15),
        moe_expert_diversity_noise=getattr(args, "moe_expert_diversity_noise", 0.05),
        moe_warmup_steps=getattr(args, "moe_warmup_steps", 1000),
        moe_identity_init=not getattr(args, "moe_no_identity_init", False),
        moe_track_divergence=getattr(args, "moe_track_divergence", False),
        moe_divergence_interval=getattr(args, "moe_divergence_interval", 100),
        moe_forced_routing_steps=getattr(args, "moe_forced_routing_steps", 0),
        moe_domain_expert_map=getattr(args, "moe_domain_expert_map", ""),
        moe_teacher_weight=getattr(args, "moe_teacher_weight", 0.0),
        moe_teacher_until_step=getattr(args, "moe_teacher_until_step", 0),
        # MoE gradient stabilization
        moe_expert_lr_scale=getattr(args, "moe_expert_lr_scale", 1.0),
        moe_router_lr_scale=getattr(args, "moe_router_lr_scale", 1.0),
        moe_lr_rewarmup_steps=getattr(args, "moe_lr_rewarmup_steps", 0),
        moe_expert_weight_decay_scale=getattr(args, "moe_expert_weight_decay_scale", 1.0),
        moe_reset_optimizer_state=getattr(args, "moe_reset_optimizer_state", False),
    )


def apply_lr_config(
    config: TrainingConfig,
    args: argparse.Namespace,
    size_config: Dict[str, Any],
) -> None:
    """Apply learning rate configuration with μP scaling.

    Mutates config in-place. Handles:
    - Auto-computed LR based on model size (μP scaling)
    - CLI --max_lr, --min_lr overrides
    - Size preset overrides

    Priority: CLI override > size_config > auto_lr

    Args:
        config: TrainingConfig to mutate
        args: Parsed argparse namespace
        size_config: Model size configuration dict
    """
    est_params = _PARAM_ESTIMATES.get(args.model_size, 220)
    auto_lr = compute_auto_lr(est_params)

    # Apply max_lr: CLI override > size_config > auto_lr
    if args.max_lr is not None and args.max_lr > 0:
        config.max_lr = float(args.max_lr)
        print(f"\n⚠️  OVERRIDE: max_lr={config.max_lr}")
    elif "max_lr" in size_config:
        config.max_lr = size_config["max_lr"]
        print(f"\n🔧 MODEL SIZE: max_lr={config.max_lr} (from {args.model_size} preset)")
    else:
        config.max_lr = auto_lr
        print(f"\n🔧 AUTO LR: max_lr={config.max_lr:.2e} (μP scaling for ~{est_params}M params)")

    # Apply min_lr: CLI override > 30% of max_lr
    if args.min_lr is not None and args.min_lr > 0:
        config.min_lr = float(args.min_lr)
        print(f"⚠️  OVERRIDE: min_lr={config.min_lr}")
    else:
        config.min_lr = config.max_lr * 0.3
        print(f"   min_lr={config.min_lr:.2e} (30% of max_lr)")


def apply_schedule_overrides(
    config: TrainingConfig,
    args: argparse.Namespace,
) -> None:
    """Apply --max_steps and short-run heuristics.

    Mutates config in-place. Handles:
    - Proportional schedule rescaling for custom max_steps
    - Save interval adjustment for short runs
    - Short-run (≤10K steps) MoR warmup adjustments

    Args:
        config: TrainingConfig to mutate
        args: Parsed argparse namespace
    """
    if args.max_steps is not None:
        config.max_steps = args.max_steps

        # Rescale step-based schedule to match mode proportions
        if config.mode == "testing":
            warmup_pct = 150 / 5000
            decay_start_pct = 3500 / 5000
        elif config.mode == "production":
            warmup_pct = 1000 / 100000
            decay_start_pct = 85000 / 100000
        elif config.mode == "chinchilla_third":
            warmup_pct = 500 / 90000
            decay_start_pct = 76500 / 90000
        else:
            warmup_pct = 0.01
            decay_start_pct = 0.85

        config.warmup_steps = max(1, int(round(config.max_steps * warmup_pct)))
        config.decay_start_step = max(0, int(round(config.max_steps * decay_start_pct)))
        config.decay_start_step = min(config.decay_start_step, max(0, config.max_steps - 1))
        config.decay_steps = max(1, config.max_steps - config.decay_start_step)

        # Adjust save_interval unless user explicitly set it
        user_set_save_interval = args.save_interval != 500
        if not user_set_save_interval:
            if config.max_steps >= 2000:
                config.save_interval = 500
            else:
                config.save_interval = min(config.save_interval, max(100, config.max_steps // 5))

        print(f"\n⚠️  OVERRIDE: max_steps={config.max_steps:,}, save_interval={config.save_interval:,}")

    # Short-run heuristics for ≤10K step runs
    if (
        config.max_steps is not None
        and config.max_steps <= 10_000
        and not args.no_short_run_override
    ):
        # Adjust MoR rampup for short runs
        if (
            args.mor_enable_pct == 0.10
            and not args.mor_already_enabled
            and args.mor_adaptive.lower() == "true"
        ):
            config.mor_rampup_steps = max(100, int(round(config.max_steps * 0.10)))
            print(
                f"⚙️  SHORT RUN: setting mor_rampup_steps={config.mor_rampup_steps}. "
                "MoD will trigger when MoR early_exit > 38%. Use --no_short_run_override to disable."
            )

    # Deprecation notice
    if args.recalc_lr:
        print("\n📈 NOTE: --recalc_lr is deprecated. LR schedule is now ALWAYS recalculated automatically on resume.")


def apply_batch_overrides(
    config: TrainingConfig,
    args: argparse.Namespace,
    size_config: Dict[str, Any],
) -> None:
    """Apply batch size, grad accum, and sequence length overrides.

    Mutates config in-place.

    Args:
        config: TrainingConfig to mutate
        args: Parsed argparse namespace
        size_config: Model size configuration dict
    """
    # Batch size
    if args.batch_size is None:
        config.batch_size = size_config.get("default_batch_size", 8)
    else:
        config.batch_size = args.batch_size
        print(f"\n⚠️  OVERRIDE: batch_size={config.batch_size}")

    # Gradient accumulation
    if args.grad_accum is None:
        config.grad_accum_steps = size_config.get("default_grad_accum", 2)
    else:
        config.grad_accum_steps = args.grad_accum
        print(f"\n⚠️  OVERRIDE: grad_accum_steps={config.grad_accum_steps}")

    # Sequence length
    if args.seq_len is not None:
        config.max_seq_len = args.seq_len
        config.seq_steps = ()
        print(f"\n⚠️  OVERRIDE: max_seq_len={config.max_seq_len} (fixed, no stepping)")
