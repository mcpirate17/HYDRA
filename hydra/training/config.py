from __future__ import annotations

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

    # MoR Curriculum (enables first, informs MoD)
    # MoR enables when: step >= max(mor_enable_min_steps, mor_enable_pct * max_steps) OR loss < mor_enable_loss_threshold
    mor_enable_pct: float = 0.10  # Enable MoR after 10% of training
    mor_enable_min_steps: int = 3000  # Minimum steps before MoR enables (floor for short runs)
    mor_enable_loss_threshold: float = 5.0  # OR enable when CE loss < 5.0
    mor_rampup_steps: int = 5000
    mor_already_enabled: bool = False

    # MoD Curriculum (MoR-informed: waits for MoR to indicate readiness)
    # MoD enables when ALL conditions are met:
    #   1. step >= mod_enable_min_step (routing stats need time to stabilize)
    #   2. MoR early_exit_ratio > threshold (model says tokens are easy)
    #   3. CE_EMA < loss_threshold (model is learning)
    mod_enable_min_step: int = 3000  # Minimum steps before MoD can trigger (routing stats noisy early)
    mod_enable_mor_early_exit_threshold: float = 0.38  # MoD enables when MoR says 38%+ tokens exit early
    mod_enable_loss_threshold: float = 4.5  # Safety floor: also require loss < 4.5

    # Loss-aware MoD supervision (teacher = top-k per-token CE loss)
    # 0.0 disables; typical starting point: 0.1 - 1.0
    mod_loss_aware_weight: float = 0.0

    # Auxiliary loss scales
    aux_scale: float = 0.1
    ponder_scale: float = 0.01

    # MoR: scales the loss-driven advantage signal that encourages deeper
    # recursions on hard tokens and shallower exits on easy tokens.
    mor_advantage_loss_scale: float = 0.1

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
    adaptive_cooldown_factor: float = 0.15
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
