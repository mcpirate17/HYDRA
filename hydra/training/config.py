from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


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
    mod_capacity: float = 0.5
    mor_adaptive: bool = True

    # MoR Curriculum
    mor_enable_pct: float = 0.30
    mor_rampup_steps: int = 5000
    mor_already_enabled: bool = False

    # MoD Curriculum
    mod_enable_pct: float = 0.10

    # Auxiliary loss scales
    aux_scale: float = 0.1
    ponder_scale: float = 0.01

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
    max_lr: float = 3e-4
    min_lr: float = 9e-5
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    # WSD Scheduler
    lr_schedule: str = "wsd_adaptive"

    # Adaptive LR settings
    adaptive_lr: bool = True
    adaptive_patience: int = 3
    adaptive_threshold: float = 0.05
    adaptive_cooldown_factor: float = 0.15
    adaptive_lr_reduction: float = 0.5

    # Stochastic Weight Averaging
    use_swa: bool = False
    swa_start_pct: float = 0.75
    swa_lr_factor: float = 0.5

    # Batch filtering
    batch_filter: bool = False
    batch_filter_threshold: float = 2.5
    batch_filter_max_skip: float = 0.05

    # Dataset
    dataset_name: str = "finefineweb"
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
    log_interval: int = 10
    seed: Optional[int] = 1337
    
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

    # Debug/stability
    halt_on_spike: bool = False

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
        if self.mode == "testing":
            self.max_steps = 5000
            self.warmup_steps = 150
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
        print(f"\n{'='*60}")
        print(f"TRAINING MODE: {self.mode.upper()}")
        print(f"{'='*60}")
        print(f"  Max steps:      {self.max_steps:,}")
        print(f"  Warmup steps:   {self.warmup_steps:,}")
        print(f"  Decay starts:   {self.decay_start_step:,}")
        print(f"  Decay steps:    {self.decay_steps:,}")
        print(f"  Save interval:  {self.save_interval:,}")
        if self.seq_steps:
            print(f"  Sequence steps: {self.seq_steps} -> {self.max_seq_len}")
            total_tokens = 0
            cumulative_steps = 0
            for steps, seq_len in self.seq_steps:
                phase_tokens = steps * self.batch_size * self.grad_accum_steps * seq_len
                total_tokens += phase_tokens
                cumulative_steps += steps
                print(f"    Phase: {steps:,} steps @ {seq_len} seq -> {phase_tokens/1e6:.1f}M tokens")
            remaining = self.max_steps - cumulative_steps
            if remaining > 0:
                final_tokens = remaining * self.batch_size * self.grad_accum_steps * self.max_seq_len
                total_tokens += final_tokens
                print(f"    Final: {remaining:,} steps @ {self.max_seq_len} seq -> {final_tokens/1e6:.1f}M tokens")
            print(f"  Total tokens:   {total_tokens/1e9:.2f}B")
        else:
            total_tokens = self.max_steps * self.batch_size * self.grad_accum_steps * self.max_seq_len
            print(f"  Sequence len:   {self.max_seq_len} (fixed)")
            print(f"  Total tokens:   {total_tokens/1e6:.1f}M")

        if self.architecture == "mod_mor":
            effective_layers = self.n_mor_blocks * self.mor_recursions
            est_params = {"100M": 220, "500M": 600, "750M": 927, "900M": 860, "1B": 1160}.get(self.model_size, 220)
            print("  Architecture:   MOD+MOR (CCGQAMoDMoRModel)")
            print(f"                  dim={self.mod_mor_dim}, {self.n_mor_blocks} MoR blocks × {self.mor_recursions} recursions = {effective_layers} effective layers")
            print(f"                  ~{est_params}M params ({self.model_size} preset)")
            print(f"                  MoD capacity: {self.mod_capacity:.0%} (~{(1-self.mod_capacity)*100:.0f}% compute savings)")
            mod_enable_step = int(self.max_steps * self.mod_enable_pct)
            if self.mod_capacity >= 1.0:
                print("  MoD Curriculum: OFF (capacity=1.0)")
            elif self.mod_enable_pct > 0:
                print(f"  MoD Curriculum: Disabled until step {mod_enable_step:,} ({self.mod_enable_pct:.0%})")
            else:
                print("  MoD Curriculum: Enabled from start (no delay)")
            mor_enable_step = int(self.max_steps * self.mor_enable_pct)
            remaining_steps = self.max_steps - mor_enable_step
            actual_rampup = min(min(self.mor_rampup_steps, 2 * mor_enable_step), remaining_steps)
            actual_rampup = max(actual_rampup, min(100, int(self.max_steps * 0.1)))
            if self.mor_already_enabled:
                print("  MoR Curriculum: RESTART MODE (adaptive from step 0)")
            elif self.mor_enable_pct > 0:
                print(f"  MoR Curriculum: Fixed-depth until step {mor_enable_step:,} ({self.mor_enable_pct:.0%})")
                print(f"                  Then {actual_rampup:,} step rampup to full adaptive")
            else:
                print("  MoR Curriculum: Adaptive from start (no delay)")
        else:
            raise ValueError(f"Unknown architecture: {self.architecture}")
        print(f"{'='*60}\n")

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.grad_accum_steps

    @property
    def tokens_per_step(self) -> int:
        return self.effective_batch_size * self.max_seq_len


MODEL_SIZE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "100M": {
        "mod_mor_dim": 768,
        "n_mor_blocks": 8,
        "mor_recursions": 4,
        "mod_mor_n_heads": 12,
        "mod_mor_n_kv_heads": 3,
        "default_batch_size": 16,
        "default_grad_accum": 4,
    },
    "debug_tall_skinny": {
        "mod_mor_dim": 384,
        "n_mor_blocks": 12,
        "mor_recursions": 2,
        "mod_mor_n_heads": 6,
        "mod_mor_n_kv_heads": 2,
        "default_batch_size": 32,
        "default_grad_accum": 1,
    },
    "DIAG": {
        "mod_mor_dim": 768,
        "n_mor_blocks": 8,
        "mor_recursions": 5,
        "mod_mor_n_heads": 12,
        "mod_mor_n_kv_heads": 3,
        "default_batch_size": 16,
        "default_grad_accum": 4,
    },
    "250M": {
        "mod_mor_dim": 1024,
        "n_mor_blocks": 10,
        "mor_recursions": 4,
        "mod_mor_n_heads": 16,
        "mod_mor_n_kv_heads": 4,
        "default_batch_size": 12,
        "default_grad_accum": 8,
    },
    "300M": {
        "mod_mor_dim": 1152,
        "n_mor_blocks": 12,
        "mor_recursions": 4,
        "mod_mor_n_heads": 18,
        "mod_mor_n_kv_heads": 3,
        "default_batch_size": 10,
        "default_grad_accum": 8,
    },
    "500M": {
        "mod_mor_dim": 1792,
        "n_mor_blocks": 14,
        "mor_recursions": 4,
        "mod_mor_n_heads": 28,
        "mod_mor_n_kv_heads": 4,
        "default_batch_size": 6,
        "default_grad_accum": 10,
    },
    "750M": {
        "mod_mor_dim": 1536,
        "n_mor_blocks": 18,
        "mor_recursions": 4,
        "mod_mor_n_heads": 24,
        "mod_mor_n_kv_heads": 4,
        "default_batch_size": 4,
        "default_grad_accum": 16,
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
    },
}
