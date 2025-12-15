#!/usr/bin/env python3
"""
Optimized Trainer for 100M HYDRA Variant

Performance Optimizations Applied (from python_strategy.md):
✓ torch.compile() with mode="max-autotune" for full graph optimization
✓ CUDA AMP (bfloat16) for speed and memory efficiency
✓ Fused AdamW optimizer (CUDA kernels)
✓ List comprehensions over loops
✓ Preallocated data structures
✓ Cached attribute lookups
✓ Generator expressions for memory efficiency
✓ Minimal attribute lookups in hot paths
✓ Windows console handler fix for stability

Training Features:
- FineFineWeb dataset (streaming, 4.9B samples)
- Cosine LR schedule with warmup
- Gradient clipping (max_norm=1.0)
- Detailed training report generation
- Checkpoint saving with metrics

Usage:
    python train_100m_optimized.py
"""

from __future__ import annotations

import signal
import sys
import os

# ============================================
# Windows Console Handler Fix
# Prevents spurious KeyboardInterrupt from Fortran/MKL runtime
# ============================================
# signal.signal(signal.SIGINT, signal.SIG_IGN)  # DISABLED - allows Ctrl+C

if sys.platform == "win32":
    import ctypes
    kernel32 = ctypes.windll.kernel32
    kernel32.SetConsoleCtrlHandler(None, True)

# Enable HF Transfer for faster downloads
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import math
import time
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

# ============================================
# Fix torch.compile recompilation storm for step counters
# Dynamo treats module integer attributes as static; this allows them to vary
# ============================================
torch._dynamo.config.allow_unspec_int_on_nn_module = True

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from hydra.model.hybrid_attention import HybridTransformer, HybridTransformerConfig
from hydra.model.ccgqa import CCGQAMoDMoRModel
from universal_data_loader import create_universal_loader
from data_filter import BatchFilter, FilterConfig
import torch._inductor.config as inductor_config
inductor_config.triton.cudagraphs = False

# ============================================
# Training Configuration (with __slots__ for memory efficiency)
# ============================================
@dataclass
class TrainingConfig:
    """Training configuration with optimized defaults for 100M variant."""
    
    # ============================================
    # TRAINING MODE: "testing", "production", "chinchilla_third"
    # ============================================
    mode: str = "testing"  # "testing" = 5K steps, "production" = 100K, "chinchilla_third" = 1/3 Chinchilla
    
    # Resume from checkpoint
    resume_from: Optional[str] = None  # Path to checkpoint to resume from
    start_step: int = 0  # Step to start from (set automatically when resuming)
    
    # ============================================
    # MODEL ARCHITECTURE: "vanilla" or "mod_mor" 
    # ============================================
    # vanilla: HybridTransformer (MQA+CCQA+MLA, no routing)
    # mod_mor: CCGQAMoDMoRModel (CCGQA + MoD + MoR, full HYDRA)
    architecture: str = "vanilla"  # "vanilla" or "mod_mor"
    mod_capacity: float = 0.5  # MoD: fraction of tokens to process (0.5 = 50% compute savings, 1.0 = MoD OFF)
    mor_adaptive: bool = True  # MoR adaptive routing (False = fixed-depth only, no routing)
    
    # MoR Curriculum: Delay adaptive routing to let base model learn first
    # MoD stays ON the whole time; only MoR adaptive depth is delayed
    mor_enable_pct: float = 0.30  # Enable MoR adaptive routing after this % of training (0.25-0.35 recommended)
    mor_rampup_steps: int = 1000  # Ramp up MoR over this many steps after enabling
    mor_already_enabled: bool = False  # Set True when resuming AFTER mor_enable_pct (restart flag)
    
    # Auxiliary loss scales (for routing regularization)
    aux_scale: float = 0.1      # MoD load balancing loss scale (default 0.1)
    ponder_scale: float = 0.01  # MoR ponder cost scale (default 0.01, use ~1e-4 for weak reg)
    
    # Model config - VANILLA (HybridTransformer)
    # 215.5M params: dim=768, 3 macro-blocks × 8 = 24 layers
    dim: int = 768
    n_macro_blocks: int = 3
    n_heads: int = 12
    n_kv_heads: int = 3
    
    # Model config - MOD+MOR (CCGQAMoDMoRModel)  
    # Model size configurations:
    #   100M: dim=1280, 8 blocks × 3 recursions = ~220M params
    #   500M: dim=2048, 10 blocks × 3 recursions = ~605M params
    # Note: these are defaults, can be overridden by model_size argument
    model_size: str = "100M"  # Model size preset: "100M" or "500M"
    mod_mor_dim: int = 1280
    n_mor_blocks: int = 8
    mor_recursions: int = 3
    mod_mor_n_heads: int = 20  # 1280/64 = 20
    mod_mor_n_kv_heads: int = 5  # 1280/256 = 5
    
    vocab_size: int = 50257
    max_seq_len: int = 2048  # Final max sequence length
    
    # All step-based parameters are computed from mode in __post_init__
    # These are defaults that get overwritten:
    max_steps: int = 5000
    warmup_steps: int = 150
    decay_start_step: int = 3500
    decay_steps: int = 1500
    seq_steps: tuple = ()  # Will be set in __post_init__
    save_interval: int = 5000
    
    # Training hyperparameters (fixed regardless of mode)
    batch_size: int = 8
    grad_accum_steps: int = 2  # Effective batch = 16
    max_lr: float = 5e-4
    min_lr: float = 1.5e-4  # 30% of max_lr
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    
    # WSD Scheduler (Warmup-Stable-Decay)
    lr_schedule: str = "wsd"  # "cosine", "wsd", or "wsd_adaptive"
    
    # Adaptive LR settings (for wsd_adaptive schedule)
    adaptive_lr: bool = False  # Enable loss-triggered cooldown
    adaptive_patience: int = 3  # Number of 100-step checks before triggering cooldown
    adaptive_threshold: float = 0.05  # Relative loss increase threshold (5%)
    adaptive_cooldown_factor: float = 0.15  # Decay phase = this fraction of remaining steps
    adaptive_lr_reduction: float = 0.5  # Reduce LR by this factor when triggered
    
    # Stochastic Weight Averaging (SWA)
    use_swa: bool = False  # Enable SWA for better generalization
    swa_start_pct: float = 0.75  # Start SWA at this % of training
    swa_lr_factor: float = 0.5  # SWA uses this fraction of current LR
    
    # Batch filtering (skip bad data during training)
    batch_filter: bool = False  # Enable loss-based batch filtering
    batch_filter_threshold: float = 2.5  # Skip if loss > threshold * EMA
    batch_filter_max_skip: float = 0.05  # Don't skip more than 5% of batches
    
    # Dataset
    dataset_name: str = "finefineweb"
    tokenizer_name: str = "gpt2"
    
    # Optimization
    use_compile: bool = True
    compile_mode: str = "max-autotune-no-cudagraphs"  # "default", "reduce-overhead", "max-autotune" (max-autotune has CUDA graph issues with grad accum)
    dtype: str = "bfloat16"
    
    # Logging
    log_interval: int = 50
    eval_interval: int = 500  # Evaluate every 500 steps (less frequent but more stable)
    # save_interval is computed in __post_init__
    
    # Checkpoint management
    max_checkpoints: int = 3  # Keep only last N periodic checkpoints
    
    # Early stopping (detect loss collapse/reversal)
    early_stop_patience: int = 3  # Stop if loss increases for N consecutive checkpoint intervals
    early_stop_threshold: float = 0.10  # 10% relative increase to trigger (e.g., 3.5 -> 3.85)
    chinchilla_multiplier: float = 20.0  # Tokens per param (Chinchilla optimal)
    early_stop_min_progress: float = 0.50  # Only check early stop after 50% of Chinchilla tokens
    
    # Paths
    checkpoint_dir: str = "checkpoints"
    report_dir: str = "reports"
    
    def __post_init__(self):
        """Configure all step-based parameters based on mode."""
        if self.mode == "testing":
            # Quick testing run: 5K steps, no sequence stepping
            self.max_steps = 5000
            self.warmup_steps = 150  # 3% warmup
            self.decay_start_step = 3500  # 70% stable
            self.decay_steps = 1500  # 30% decay
            self.seq_steps = ()  # No sequence stepping in testing
            self.save_interval = 500  # Frequent saves for testing
            # Tokens: 5000 * 16 * 2048 = 163.8M
            
        elif self.mode == "production":
            # Chinchilla-optimal: 100K steps with stepped sequence
            # Token calculation:
            #   Phase 1: 50K steps * 16 batch * 512 seq = 409.6M tokens
            #   Phase 2: 50K steps * 16 batch * 1024 seq = 819.2M tokens  
            #   Phase 3: (remaining) * 16 batch * 2048 seq = depends on total
            # Total with 100K steps: ~1.6B tokens (but we want Chinchilla ~4.3B)
            # For 215M model @ 20 tok/param = 4.3B tokens needed
            # At 16 batch * 1024 avg seq = 16K tok/step -> need ~268K steps
            # Compromise: 100K steps with stepped sequence = ~1.2B tokens
            self.max_steps = 100000
            self.warmup_steps = 1000  # 1% warmup
            self.decay_start_step = 85000  # 85% point
            self.decay_steps = 15000  # 15% decay
            # Stepped sequence: 512 -> 1024 -> 2048
            self.seq_steps = ((50000, 512), (50000, 1024))  # Then 2048 for remainder
            self.save_interval = 5000  # Every 5K steps
            # Phase 1: 50K * 16 * 512 = 409.6M
            # Phase 2: 50K * 16 * 1024 = 819.2M  
            # Total: 1.23B tokens
        
        elif self.mode == "chinchilla_third":
            # 1/3 Chinchilla optimal for 220M model
            # 220M * 20 tokens/param = 4.4B tokens for full Chinchilla
            # 1/3 = 1.47B tokens
            # At batch=8 * accum=4 * seq=512 = 16,384 tokens/step -> ~90K steps
            self.max_steps = 90000  # Total steps
            self.warmup_steps = 500  # Proper warmup from fresh start
            self.decay_start_step = 76500  # 85% point
            self.decay_steps = 13500  # 15% decay
            self.seq_steps = ()  # Fixed sequence length for speed comparison
            self.save_interval = 5000  # Every 5K steps
            self.max_seq_len = 512  # Shorter seq for speed
            self.grad_accum_steps = 4  # Higher accum to maintain effective batch size
            # Token calc: 90K * 8 * 4 * 512 = 1.47B tokens (exactly 1/3 Chinchilla)
        else:
            raise ValueError(f"Unknown mode: {self.mode}. Use 'testing', 'production', or 'chinchilla_third'")
        
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
        
        # Architecture info
        if self.architecture == "vanilla":
            print(f"  Architecture:   VANILLA (HybridTransformer)")
            print(f"                  dim={self.dim}, {self.n_macro_blocks} macro-blocks × 8 = {self.n_macro_blocks * 8} layers")
            print(f"                  ~215M params")
        elif self.architecture == "mod_mor":
            effective_layers = self.n_mor_blocks * self.mor_recursions
            # Estimate param count based on model_size preset
            est_params = {"100M": 220, "500M": 520}.get(self.model_size, 220)
            print(f"  Architecture:   MOD+MOR (CCGQAMoDMoRModel)")
            print(f"                  dim={self.mod_mor_dim}, {self.n_mor_blocks} MoR blocks × {self.mor_recursions} recursions = {effective_layers} effective layers")
            print(f"                  ~{est_params}M params ({self.model_size} preset)")
            print(f"                  Hybrid attention: MQA → CCQA → MLA pattern")
            print(f"                  MoD capacity: {self.mod_capacity:.0%} (~{(1-self.mod_capacity)*100:.0f}% compute savings)")
            # MoR Curriculum info
            mor_enable_step = int(self.max_steps * self.mor_enable_pct)
            if self.mor_already_enabled:
                print(f"  MoR Curriculum: RESTART MODE (adaptive from step 0)")
            elif self.mor_enable_pct > 0:
                print(f"  MoR Curriculum: Fixed-depth until step {mor_enable_step:,} ({self.mor_enable_pct:.0%})")
                print(f"                  Then {self.mor_rampup_steps:,} step rampup to full adaptive")
            else:
                print(f"  MoR Curriculum: Adaptive from start (no delay)")
        else:
            raise ValueError(f"Unknown architecture: {self.architecture}")
        print(f"{'='*60}\n")
    
    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.grad_accum_steps
    
    @property
    def tokens_per_step(self) -> int:
        return self.effective_batch_size * self.max_seq_len


# ============================================
# Training Metrics Tracker
# ============================================
@dataclass
class TrainingMetrics:
    """Track training metrics with preallocated storage."""
    
    # Preallocate lists with estimated capacity
    losses: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    grad_norms: List[float] = field(default_factory=list)
    tokens_per_sec: List[float] = field(default_factory=list)
    step_times: List[float] = field(default_factory=list)
    
    # Summary stats
    start_time: float = 0.0
    end_time: float = 0.0
    total_tokens: int = 0
    best_loss: float = float('inf')
    best_loss_step: int = 0
    initial_loss: float = 0.0
    final_loss: float = 0.0
    
    # EMA loss tracking
    ema_loss: float = 0.0
    ema_alpha: float = 0.05  # Smoothing factor (smaller = smoother)
    
    def update(self, step: int, loss: float, lr: float, grad_norm: float, 
               tps: float, step_time: float) -> None:
        """Update metrics (optimized - direct append)."""
        self.losses.append(loss)
        self.learning_rates.append(lr)
        self.grad_norms.append(grad_norm)
        self.tokens_per_sec.append(tps)
        self.step_times.append(step_time)
        
        # Update EMA loss
        if self.ema_loss == 0.0:
            self.ema_loss = loss  # Initialize with first loss
        else:
            self.ema_loss = self.ema_alpha * loss + (1 - self.ema_alpha) * self.ema_loss
        
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_loss_step = step
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "losses": self.losses,
            "learning_rates": self.learning_rates,
            "grad_norms": self.grad_norms,
            "tokens_per_sec": self.tokens_per_sec,
            "step_times": self.step_times,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_tokens": self.total_tokens,
            "best_loss": self.best_loss,
            "best_loss_step": self.best_loss_step,
            "initial_loss": self.initial_loss,
            "final_loss": self.final_loss,
            "training_time_seconds": self.end_time - self.start_time,
        }


# ============================================
# Learning Rate Schedules
# ============================================
def get_lr_cosine(step: int, warmup_steps: int, max_steps: int, 
                  max_lr: float, min_lr: float) -> float:
    """Cosine schedule with linear warmup."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (max_lr - min_lr) * cosine_decay


def get_lr_wsd(step: int, warmup_steps: int, decay_start_step: int,
               decay_steps: int, max_lr: float, min_lr: float) -> float:
    """
    WSD (Warmup-Stable-Decay) schedule with linear decay.
    
    Phases:
    1. Warmup: Linear ramp from 0 to max_lr
    2. Stable: Constant at max_lr (majority of training)
    3. Decay: Linear decay from max_lr to min_lr
    
    This schedule keeps high LR longer for maximum learning,
    only decaying at the end to refine the model.
    
    For infinite training: set decay_start_step = 1e12
    When ready to finish: update decay_start_step = current_step
    """
    # Phase 1: Warmup
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    
    # Phase 2: Stable (constant at max_lr)
    if step < decay_start_step:
        return max_lr
    
    # Phase 3: Linear Decay
    decay_progress = (step - decay_start_step) / max(1, decay_steps)
    decay_progress = min(1.0, decay_progress)  # Clamp to [0, 1]
    
    # Linear interpolation: max_lr -> min_lr
    return max_lr - (max_lr - min_lr) * decay_progress


class AdaptiveLRManager:
    """
    Adaptive Learning Rate Manager
    
    Combines WSD schedule with:
    1. Loss-triggered early cooldown - Starts decay when loss spikes
    2. Plateau detection - Reduces LR when stuck
    3. Optional SWA support - Better final model quality
    
    Based on NeurIPS 2024 research on compute-optimal training.
    """
    
    def __init__(self, config: 'TrainingConfig'):
        self.config = config
        self.base_max_lr = config.max_lr
        self.current_max_lr = config.max_lr
        
        # Adaptive cooldown state
        self.cooldown_triggered = False
        self.cooldown_start_step = config.decay_start_step
        self.cooldown_steps = config.decay_steps
        
        # Loss tracking for adaptive decisions
        self.loss_history: List[float] = []
        self.loss_ema = 0.0
        self.loss_ema_alpha = 0.1
        self.best_loss_window = float('inf')
        self.patience_counter = 0
        
        # SWA state
        self.swa_active = False
        self.swa_start_step = int(config.max_steps * config.swa_start_pct)
        self.swa_n = 0
        self.swa_state: Optional[Dict[str, torch.Tensor]] = None
        
        # Plateau detection
        self.plateau_counter = 0
        self.plateau_threshold = 10  # Steps without improvement
        
    def update(self, step: int, loss: float) -> None:
        """Update adaptive state with new loss observation."""
        # Update loss EMA
        if self.loss_ema == 0:
            self.loss_ema = loss
        else:
            self.loss_ema = self.loss_ema_alpha * loss + (1 - self.loss_ema_alpha) * self.loss_ema
        
        self.loss_history.append(loss)
        
        # Keep only recent history (last 100 observations)
        if len(self.loss_history) > 100:
            self.loss_history.pop(0)
        
        # Check for loss spike (adaptive cooldown trigger)
        if self.config.adaptive_lr and not self.cooldown_triggered:
            self._check_cooldown_trigger(step, loss)
        
        # Update best loss in window
        if loss < self.best_loss_window:
            self.best_loss_window = loss
            self.plateau_counter = 0
        else:
            self.plateau_counter += 1
    
    def _check_cooldown_trigger(self, step: int, loss: float) -> None:
        """Check if we should trigger early cooldown due to loss spike."""
        if len(self.loss_history) < 10:
            return
        
        # Calculate recent average (last 10) vs older average (10-20 back)
        recent_avg = sum(self.loss_history[-5:]) / 5
        older_avg = sum(self.loss_history[-15:-5]) / 10 if len(self.loss_history) >= 15 else recent_avg
        
        # Check if loss is increasing beyond threshold
        if older_avg > 0 and (recent_avg - older_avg) / older_avg > self.config.adaptive_threshold:
            self.patience_counter += 1
            
            if self.patience_counter >= self.config.adaptive_patience:
                self._trigger_cooldown(step)
        else:
            self.patience_counter = max(0, self.patience_counter - 1)
    
    def _trigger_cooldown(self, step: int) -> None:
        """Trigger early cooldown phase."""
        self.cooldown_triggered = True
        self.cooldown_start_step = step
        
        # Calculate remaining steps and set cooldown duration
        remaining_steps = self.config.max_steps - step
        self.cooldown_steps = int(remaining_steps * self.config.adaptive_cooldown_factor)
        
        # Optionally reduce max LR
        self.current_max_lr = self.base_max_lr * self.config.adaptive_lr_reduction
        
        print(f"\n" + "="*60)
        print(f"⚡ ADAPTIVE LR: Cooldown triggered at step {step}")
        print(f"   Loss trend: {sum(self.loss_history[-15:-5])/10:.4f} -> {sum(self.loss_history[-5:])/5:.4f}")
        print(f"   New decay phase: {self.cooldown_steps} steps")
        print(f"   Max LR reduced: {self.base_max_lr} -> {self.current_max_lr}")
        print("="*60 + "\n")
    
    def get_lr(self, step: int) -> float:
        """Get learning rate for current step."""
        config = self.config
        
        # Use adaptive cooldown parameters if triggered
        if self.cooldown_triggered:
            decay_start = self.cooldown_start_step
            decay_steps = self.cooldown_steps
            max_lr = self.current_max_lr
        else:
            decay_start = config.decay_start_step
            decay_steps = config.decay_steps
            max_lr = config.max_lr
        
        # WSD schedule
        lr = get_lr_wsd(
            step, config.warmup_steps, decay_start,
            decay_steps, max_lr, config.min_lr
        )
        
        # SWA LR adjustment
        if config.use_swa and step >= self.swa_start_step:
            lr = lr * config.swa_lr_factor
            if not self.swa_active:
                self.swa_active = True
                print(f"\n⚡ SWA activated at step {step}, LR factor: {config.swa_lr_factor}")
        
        return lr
    
    def update_swa(self, model: nn.Module) -> None:
        """Update SWA running average of model weights."""
        if not self.config.use_swa or not self.swa_active:
            return
        
        self.swa_n += 1
        
        with torch.no_grad():
            if self.swa_state is None:
                # Initialize SWA state with current model
                self.swa_state = {}
                for name, param in model.named_parameters():
                    self.swa_state[name] = param.data.clone()
            else:
                # Running average: swa = swa + (param - swa) / n
                for name, param in model.named_parameters():
                    self.swa_state[name].add_(
                        (param.data - self.swa_state[name]) / self.swa_n
                    )
    
    def apply_swa(self, model: nn.Module) -> None:
        """Apply SWA weights to model (call at end of training)."""
        if self.swa_state is None:
            return
        
        print(f"\nApplying SWA weights (averaged over {self.swa_n} updates)...")
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.swa_state:
                    param.data.copy_(self.swa_state[name])
    
    def get_state(self) -> Dict[str, Any]:
        """Get state for checkpointing."""
        return {
            'cooldown_triggered': self.cooldown_triggered,
            'cooldown_start_step': self.cooldown_start_step,
            'cooldown_steps': self.cooldown_steps,
            'current_max_lr': self.current_max_lr,
            'loss_ema': self.loss_ema,
            'best_loss_window': self.best_loss_window,
            'patience_counter': self.patience_counter,
            'swa_active': self.swa_active,
            'swa_n': self.swa_n,
        }
    
    def load_state(self, state: Dict[str, Any]) -> None:
        """Load state from checkpoint."""
        self.cooldown_triggered = state.get('cooldown_triggered', False)
        self.cooldown_start_step = state.get('cooldown_start_step', self.config.decay_start_step)
        self.cooldown_steps = state.get('cooldown_steps', self.config.decay_steps)
        self.current_max_lr = state.get('current_max_lr', self.config.max_lr)
        self.loss_ema = state.get('loss_ema', 0.0)
        self.best_loss_window = state.get('best_loss_window', float('inf'))
        self.patience_counter = state.get('patience_counter', 0)
        self.swa_active = state.get('swa_active', False)
        self.swa_n = state.get('swa_n', 0)


def get_lr(step: int, config: 'TrainingConfig') -> float:
    """Dispatch to appropriate LR schedule."""
    if config.lr_schedule == "wsd" or config.lr_schedule == "wsd_adaptive":
        return get_lr_wsd(
            step, config.warmup_steps, config.decay_start_step,
            config.decay_steps, config.max_lr, config.min_lr
        )
    else:  # cosine
        return get_lr_cosine(
            step, config.warmup_steps, config.max_steps,
            config.max_lr, config.min_lr
        )


# ============================================
# Optimized Trainer
# ============================================
class Trainer:
    """Optimized trainer with all performance best practices."""
    
    __slots__ = (
        'config', 'device', 'model', 'optimizer', 'scaler',
        'train_loader', 'metrics', 'dtype', '_cached_get_lr',
        '_param_groups', '_log_interval', '_tokens_per_step', '_use_scaler',
        '_checkpoint_history', '_checkpoint_losses', '_early_stop_counter',
        '_current_seq_len', '_use_mod_mor', '_start_step', '_mor_enable_step',
        '_diagnostics_file', '_diagnostics_data', '_last_ce_loss', '_last_aux_loss', '_last_ponder_loss',
        '_adaptive_lr', '_batch_filter', '_checkpoint_seq_len', '_checkpoint_config', '_checkpoint_lr'
    )
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.metrics = TrainingMetrics()
        
        # Cache frequently accessed values
        self._log_interval = config.log_interval
        self._tokens_per_step = config.tokens_per_step
        
        # Checkpoint management
        self._checkpoint_history: List[Path] = []  # Track periodic checkpoints for rotation
        self._checkpoint_losses: List[float] = []  # Track loss at each checkpoint for early stopping
        self._early_stop_counter: int = 0  # Count consecutive loss increases
        self._start_step: int = 0  # Step to start from
        
        # Diagnostics for layer-by-layer analysis
        self._diagnostics_file: Optional[Path] = None
        self._diagnostics_data: List[dict] = []
        self._last_ce_loss: float = 0.0
        self._last_aux_loss: float = 0.0
        self._last_ponder_loss: float = 0.0
        
        # Adaptive LR manager
        self._adaptive_lr: Optional[AdaptiveLRManager] = None
        if config.adaptive_lr or config.use_swa:
            self._adaptive_lr = AdaptiveLRManager(config)
        
        # Batch filter for skipping bad data
        self._batch_filter: Optional[BatchFilter] = None
        if config.batch_filter:
            filter_cfg = FilterConfig(
                loss_spike_threshold=config.batch_filter_threshold,
                max_skips_per_epoch=config.batch_filter_max_skip,
            )
            self._batch_filter = BatchFilter(filter_cfg)
        
        # Set dtype
        self.dtype = getattr(torch, config.dtype) if config.dtype != "float32" else torch.float32
        
        # Peek at checkpoint to get architecture params for model creation (if resuming)
        # This ensures model is created with MATCHING architecture before loading weights
        self._checkpoint_seq_len = None
        self._checkpoint_config = {}
        self._checkpoint_lr = None  # Will be set if resuming
        if config.resume_from:
            self._checkpoint_config = self._peek_checkpoint_config(config.resume_from)
            self._checkpoint_seq_len = self._checkpoint_config.get('max_seq_len', config.max_seq_len)
            
            # Override architecture from checkpoint (critical - model structure must match)
            if 'architecture' in self._checkpoint_config:
                ckpt_arch = self._checkpoint_config['architecture']
                if ckpt_arch != config.architecture:
                    print(f"Overriding architecture: {config.architecture} -> {ckpt_arch} (from checkpoint)")
                    config.architecture = ckpt_arch
            
            # Override architecture params from checkpoint to ensure match
            if 'mod_mor_dim' in self._checkpoint_config:
                config.mod_mor_dim = self._checkpoint_config['mod_mor_dim']
            if 'n_mor_blocks' in self._checkpoint_config:
                config.n_mor_blocks = self._checkpoint_config['n_mor_blocks']
            if 'mor_recursions' in self._checkpoint_config:
                config.mor_recursions = self._checkpoint_config['mor_recursions']
            if 'mod_mor_n_heads' in self._checkpoint_config:
                config.mod_mor_n_heads = self._checkpoint_config['mod_mor_n_heads']
            if 'mod_mor_n_kv_heads' in self._checkpoint_config:
                config.mod_mor_n_kv_heads = self._checkpoint_config['mod_mor_n_kv_heads']
            
            # Get learning rate from checkpoint optimizer state (for warm restart)
            if 'optimizer' in torch.load(config.resume_from, weights_only=False, map_location='cpu'):
                ckpt_full = torch.load(config.resume_from, weights_only=False, map_location='cpu')
                if 'optimizer' in ckpt_full and 'param_groups' in ckpt_full['optimizer']:
                    last_lr = ckpt_full['optimizer']['param_groups'][0].get('lr', config.max_lr)
                    print(f"Checkpoint final LR: {last_lr:.6f}")
                    # Store for potential use in LR warmup
                    self._checkpoint_lr = last_lr
        
        self._setup_model()
        self._setup_optimizer()
        self._setup_data()
        
        # Load checkpoint if resuming (after model is compiled)
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
        print(f"AMP dtype: {config.dtype}")
        
        # LR schedule info
        if config.lr_schedule == "wsd" or config.lr_schedule == "wsd_adaptive":
            stable_steps = config.decay_start_step - config.warmup_steps
            print(f"LR Schedule: WSD (Warmup-Stable-Decay)")
            print(f"  Warmup: {config.warmup_steps} steps ({config.warmup_steps/config.max_steps*100:.1f}%)")
            print(f"  Stable: {stable_steps} steps ({stable_steps/config.max_steps*100:.1f}%) at LR={config.max_lr}")
            print(f"  Decay:  {config.decay_steps} steps ({config.decay_steps/config.max_steps*100:.1f}%) -> LR={config.min_lr}")
            if config.adaptive_lr:
                print(f"  Adaptive: ENABLED (patience={config.adaptive_patience}, threshold={config.adaptive_threshold:.0%})")
            if config.use_swa:
                print(f"  SWA: ENABLED (starts at {config.swa_start_pct:.0%} of training)")
        else:
            print(f"LR Schedule: Cosine with warmup")
            print(f"  Warmup: {config.warmup_steps} steps, Max LR: {config.max_lr}, Min LR: {config.min_lr}")
        if config.batch_filter:
            print(f"Batch Filter: ENABLED (threshold={config.batch_filter_threshold}x, max_skip={config.batch_filter_max_skip:.0%})")
        print(f"{'='*70}\n")
    
    def _peek_checkpoint_config(self, checkpoint_path: str) -> dict:
        """Peek at checkpoint to get config for model creation.
        
        Returns dict with architecture params needed to recreate the model
        with matching shapes before loading the full checkpoint.
        """
        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location='cpu')
        ckpt_config = checkpoint.get('config', {})
        
        result = {}
        
        # Get seq_len from RoPE cache shape (most reliable)
        for key, tensor in checkpoint['model'].items():
            if 'cos_cached' in key:
                result['max_seq_len'] = tensor.shape[2]  # Shape is [1, 1, seq_len, head_dim/2]
                print(f"Checkpoint RoPE cache seq_len: {result['max_seq_len']}")
                break
        
        # Get architecture params from checkpoint config
        # These MUST match for weight loading to work
        arch_keys = [
            'architecture',  # Critical: vanilla vs mod_mor
            'mod_mor_dim', 'n_mor_blocks', 'mor_recursions', 
            'mod_mor_n_heads', 'mod_mor_n_kv_heads', 'mod_capacity',
            'vocab_size', 'mor_adaptive'
        ]
        for key in arch_keys:
            if key in ckpt_config:
                result[key] = ckpt_config[key]
        
        if result:
            print(f"Checkpoint architecture: dim={result.get('mod_mor_dim')}, "
                  f"blocks={result.get('n_mor_blocks')}, "
                  f"recursions={result.get('mor_recursions')}, "
                  f"heads={result.get('mod_mor_n_heads')}")
        
        return result
    
    def _peek_checkpoint_seq_len(self, checkpoint_path: str) -> int:
        """Peek at checkpoint to get seq_len from RoPE cache shape.
        
        This allows us to create/compile the model with matching shapes
        before loading the full checkpoint.
        """
        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location='cpu')
        
        # Find first cos_cached buffer to get seq_len
        for key, tensor in checkpoint['model'].items():
            if 'cos_cached' in key:
                seq_len = tensor.shape[2]  # Shape is [1, 1, seq_len, head_dim/2]
                print(f"Checkpoint RoPE cache seq_len: {seq_len}")
                return seq_len
        
        # Fallback to config's max_seq_len if no RoPE found
        return self.config.max_seq_len
    
    def _setup_model(self) -> None:
        """Initialize and optionally compile the model."""
        config = self.config
        
        # Calculate max seq_len needed across all training phases
        # This ensures RoPE cache is big enough and won't need runtime resizing
        max_needed_seq_len = config.max_seq_len
        for _, seq_len in config.seq_steps:
            max_needed_seq_len = max(max_needed_seq_len, seq_len)
        
        # If resuming, we create with max needed (checkpoint's RoPE will be expanded on load)
        model_seq_len = max_needed_seq_len
        print(f"Creating model with max_seq_len={model_seq_len} (covers all training phases)")
        
        if config.architecture == "vanilla":
            # HybridTransformer: MQA + CCQA + MLA, no routing
            model_config = HybridTransformerConfig(
                vocab_size=config.vocab_size,
                dim=config.dim,
                n_macro_blocks=config.n_macro_blocks,
                n_heads=config.n_heads,
                n_kv_heads=config.n_kv_heads,
                max_seq_len=model_seq_len,
                enable_mod=False,  # No MoD for vanilla
            )
            self.model = HybridTransformer(model_config).to(self.device)
            self._use_mod_mor = False
            
        elif config.architecture == "mod_mor":
            # CCGQAMoDMoRModel: CCGQA + MoD + MoR (full HYDRA)
            # Uses separate config for matched param count (~220M vs ~215M vanilla)
            self.model = CCGQAMoDMoRModel(
                vocab_size=config.vocab_size,
                dim=config.mod_mor_dim,  # 1280 for matched params
                n_mor_blocks=config.n_mor_blocks,  # 8 blocks
                recursions_per_block=config.mor_recursions,  # 3 recursions = 24 effective layers
                n_heads=config.mod_mor_n_heads,  # 20 (1280/64)
                n_kv_heads=config.mod_mor_n_kv_heads,  # 5 (1280/256)
                compression_factor=4,
                mlp_ratio=3.6,  # 3.6 * 1280 * 2 = 9216 hidden dim
                max_seq_len=model_seq_len,
                mod_capacity=config.mod_capacity,
                adaptive=config.mor_adaptive,  # Use config instead of hardcoded True
                tie_weights=True,
            ).to(self.device)
            self._use_mod_mor = True
            
            # Print MoD/MoR status
            mod_status = "OFF (capacity=1.0)" if config.mod_capacity >= 1.0 else f"{config.mod_capacity:.0%} capacity"
            mor_status = "adaptive" if config.mor_adaptive else "fixed-depth (no routing)"
            print(f"MoD: {mod_status}")
            print(f"MoR: {mor_status}, {config.mor_recursions} recursions/block")
            
            # Configure MoR curriculum: delay adaptive routing
            # Only relevant if mor_adaptive=True
            if config.mor_adaptive:
                mor_enable_step = int(config.max_steps * config.mor_enable_pct)
                
                # Handle restart flag: if resuming after MoR was already enabled, set enable_step=0
                if config.mor_already_enabled:
                    mor_enable_step = 0
                    print(f"MoR RESTART MODE: Adaptive routing enabled from start (resumed after enable point)")
                
                self.model.set_mor_curriculum(
                    enable_step=mor_enable_step,
                    rampup_steps=config.mor_rampup_steps
                )
                self._mor_enable_step = mor_enable_step
                
                if mor_enable_step > 0:
                    print(f"MoR CURRICULUM: Fixed-depth until step {mor_enable_step:,} ({config.mor_enable_pct:.0%}), then {config.mor_rampup_steps:,} step rampup")
            else:
                self._mor_enable_step = 0
                print(f"MoR CURRICULUM: Disabled (adaptive=False, running pure fixed-depth)")
        else:
            raise ValueError(f"Unknown architecture: {config.architecture}")
        
        # Apply torch.compile for maximum performance
        if config.use_compile and self.device == "cuda":
            mode = config.compile_mode
            if mode == "max-autotune":
                mode = "max-autotune-no-cudagraphs"  # prevents overwritten-CUDAGraph tensor failures

            print(f"Compiling model with mode='{mode}'...")
            self.model = torch.compile(
                self.model,
                mode=mode,
                fullgraph=False,   # set True only if it actually works for your model
                dynamic=False,     # keep shapes stable for best perf
            )
            print("Model compiled successfully!")
    
    def _setup_optimizer(self) -> None:
        """Setup fused AdamW optimizer with weight decay groups."""
        config = self.config
        
        # Separate parameters: decay vs no-decay
        # Cache parameter iteration to avoid repeated model traversal
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'weight' in name and 'norm' not in name and 'embed' not in name:
                    decay_params.append(param)
                else:
                    no_decay_params.append(param)
        
        # Fused optimizer (CUDA kernels, much faster)
        self.optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": config.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=config.max_lr,
            betas=(0.9, 0.95),
            fused=True,  # CUDA fused kernels
        )
        
        # Cache param groups reference
        self._param_groups = self.optimizer.param_groups
        
        # GradScaler for AMP - only needed for float16, not bfloat16
        # bfloat16 has enough dynamic range and doesn't need loss scaling
        use_scaler = config.dtype == "float16"
        self.scaler = GradScaler("cuda", enabled=use_scaler)
        self._use_scaler = use_scaler
    
    def _setup_data(self) -> None:
        """Initialize FineFineWeb data loader with stepped sequence support."""
        config = self.config
        
        # Start with first phase sequence length
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
            device="cpu",  # Load to CPU, move to GPU in training loop
            tokenizer_name=config.tokenizer_name,
        )
        self._tokens_per_step = config.batch_size * config.grad_accum_steps * initial_seq_len
        print(f"Tokens per step: {self._tokens_per_step:,}")
        print("Dataset ready!")
    
    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model, optimizer, and scaler state from checkpoint."""
        print(f"\n{'='*70}")
        print(f"RESUMING FROM CHECKPOINT: {checkpoint_path}")
        print(f"{'='*70}")
        
        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=self.device)
        
        # Load model state (model was created with matching seq_len from peek)
        model = self.model
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
        model.load_state_dict(checkpoint["model"])
        
        # Load optimizer and scaler state
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scaler.load_state_dict(checkpoint["scaler"])
        
        # Set start step
        self._start_step = checkpoint["step"]
        
        # Report what was loaded
        ckpt_metrics = checkpoint.get("metrics", {})
        print(f"  Loaded step: {self._start_step}")
        print(f"  Previous best loss: {ckpt_metrics.get('best_loss', 'N/A')}")
        print(f"  Previous total tokens: {ckpt_metrics.get('total_tokens', 'N/A'):,}")
        print(f"  Will continue to step: {self.config.max_steps}")
        print(f"  Remaining steps: {self.config.max_steps - self._start_step:,}")
        
        # Pre-populate metrics with checkpoint data
        if ckpt_metrics:
            self.metrics.best_loss = ckpt_metrics.get('best_loss', float('inf'))
            self.metrics.best_loss_step = ckpt_metrics.get('best_loss_step', 0)
            self.metrics.total_tokens = ckpt_metrics.get('total_tokens', 0)
        
        # Load adaptive LR manager state if present
        if "adaptive_lr_state" in checkpoint and self._adaptive_lr is not None:
            self._adaptive_lr.load_state(checkpoint["adaptive_lr_state"])
            print(f"  Loaded adaptive LR state: cooldown={self._adaptive_lr.cooldown_triggered}, "
                  f"patience={self._adaptive_lr.patience_counter}")
            # Load SWA model state if present
            if "swa_model" in checkpoint and hasattr(self._adaptive_lr, 'swa_state'):
                if self._adaptive_lr.swa_state is not None:
                    # SWA state is a dict of tensors, not a model
                    self._adaptive_lr.swa_state = checkpoint["swa_model"]
                    print(f"  Loaded SWA state")
        elif self._adaptive_lr is not None:
            print(f"  Adaptive LR enabled (fresh state - no prior adaptive state in checkpoint)")
        
        print(f"{'='*70}\n")
    
    def _count_params(self) -> int:
        """Count model parameters (optimized with generator)."""
        return sum(p.numel() for p in self.model.parameters())
    
    def _get_seq_len_for_step(self, step: int) -> int:
        """Get the sequence length for a given step based on stepped schedule."""
        config = self.config
        cumulative_steps = 0
        for phase_steps, seq_len in config.seq_steps:
            cumulative_steps += phase_steps
            if step < cumulative_steps:
                return seq_len
        return config.max_seq_len  # Final phase uses max_seq_len
    
    def _recreate_dataloader(self, seq_len: int) -> None:
        """Recreate dataloader with new sequence length and resize RoPE cache."""
        config = self.config
        if hasattr(self, 'train_loader') and self.train_loader:
            self.train_loader.close()
        
        print(f"\n{'='*70}")
        print(f"SEQUENCE LENGTH TRANSITION: {self._current_seq_len} -> {seq_len}")
        print(f"{'='*70}")
        
        # Resize RoPE cache in model to support new sequence length
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
        )
        self._current_seq_len = seq_len
        self._tokens_per_step = config.batch_size * config.grad_accum_steps * seq_len
        print(f"New tokens/step: {self._tokens_per_step:,}")
        print(f"{'='*70}\n")
    
    def _get_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get batch and move to device (optimized)."""
        batch = self.train_loader.get_batch()
        # Use non_blocking for async transfer
        return (
            batch["input_ids"].to(self.device, non_blocking=True),
            batch["labels"].to(self.device, non_blocking=True),
        )
    
    def train(self) -> TrainingMetrics:
        """Main training loop with all optimizations."""
        config = self.config
        model = self.model
        optimizer = self.optimizer
        scaler = self.scaler
        metrics = self.metrics
        
        # Cache values for hot loop
        grad_accum = config.grad_accum_steps
        grad_clip = config.grad_clip
        max_steps = config.max_steps
        log_interval = self._log_interval
        eval_interval = config.eval_interval
        save_interval = config.save_interval
        tokens_per_step = self._tokens_per_step
        dtype = self.dtype
        device = self.device
        param_groups = self._param_groups
        # Pre-compute denominator for loss scaling
        loss_scale = 1.0 / grad_accum
        
        # Resume from checkpoint if applicable
        start_step = self._start_step
        if start_step > 0:
            print(f"Resuming training from step {start_step}...")
        else:
            print("Starting training...")
        print("-" * 70)
        
        model.train()
        metrics.start_time = time.time()
        step = start_step  # Start from checkpoint step or 0
        accum_loss = 0.0
        use_scaler = self._use_scaler
        use_mod_mor = self._use_mod_mor
        
        eval_batches = 25  # More batches = more stable eval loss (was 8)
        eval_loader = create_universal_loader(
            dataset=config.dataset_name,
            batch_size=config.batch_size,
            seq_len=self._current_seq_len,   # keep it consistent with current phase
            vocab_size=config.vocab_size,
            device="cpu",                    # store eval batches on CPU
            tokenizer_name=config.tokenizer_name,
        )

        fixed_eval_batches = [eval_loader.get_batch() for _ in range(eval_batches)]

        # Check if there are any steps to run
        if start_step >= max_steps:
            print(f"⚠️  No steps to run: start_step={start_step} >= max_steps={max_steps}")
            print(f"   Increase --max_steps to continue training from this checkpoint.")
            metrics.end_time = time.time()
            return metrics
        
        while step < max_steps:
            step_start = time.time()
            optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()
            accum_loss = 0.0

            # Update global step for MoR warmup scheduling
            if use_mod_mor and hasattr(model, 'set_global_step'):
                # Handle compiled models that wrap the base model
                base_model = model._orig_mod if hasattr(model, '_orig_mod') else model
                if hasattr(base_model, 'set_global_step'):
                    base_model.set_global_step(step)
            
            # Gradient accumulation loop
            for micro_step in range(grad_accum):
                x, y = self._get_batch()
                
                if self.device == "cuda" and self.config.use_compile:
                    torch.compiler.cudagraph_mark_step_begin()

                with autocast(device, dtype=dtype):
                    if use_mod_mor:
                        # CCGQAMoDMoRModel returns (logits, losses_dict) when return_losses=True
                        logits, aux_losses = model(x, return_losses=True)
                        ce_loss = F.cross_entropy(
                            logits.view(-1, logits.size(-1)),
                            y.view(-1),
                            ignore_index=-100,
                        )
                        # Add auxiliary losses (MoD load balancing + MoR ponder cost)
                        # aux_loss pushes MoD capacity toward target (0.5)
                        # ponder_loss encourages efficient MoR depth usage
                        aux_loss = aux_losses.get("aux_loss", 0.0)
                        ponder_loss = aux_losses.get("ponder_loss", 0.0)
                        # Configurable loss scales from config
                        # aux_scale: 0.1 default (MoD load balancing)
                        # ponder_scale: 0.01 default, use ~1e-4 for weak MoR regularization
                        loss = ce_loss + self.config.aux_scale * aux_loss + self.config.ponder_scale * ponder_loss
                        
                        # Track individual losses for diagnostics
                        self._last_ce_loss = ce_loss.item() if hasattr(ce_loss, 'item') else float(ce_loss)
                        self._last_aux_loss = aux_loss.item() if hasattr(aux_loss, 'item') else float(aux_loss)
                        self._last_ponder_loss = ponder_loss.item() if hasattr(ponder_loss, 'item') else float(ponder_loss)
                    else:
                        logits = model(x)
                        loss = F.cross_entropy(
                            logits.view(-1, logits.size(-1)),
                            y.view(-1),
                            ignore_index=-100,
                        )
                
                # Batch filtering: check if this batch should be skipped
                # (loss spike from bad data)
                if self._batch_filter is not None:
                    should_skip, skip_reason = self._batch_filter.should_skip_batch(
                        loss.item(), step
                    )
                    if should_skip:
                        # Skip backward pass - just continue to next micro-batch
                        if step % 500 == 0:  # Log occasionally
                            stats = self._batch_filter.get_stats()
                            print(f"  [BatchFilter] Skipped batch: {skip_reason}, "
                                  f"total skipped: {stats['n_skipped']}/{stats['n_total']} "
                                  f"({stats['skip_ratio']*100:.1f}%)")
                        continue
                
                # Scale loss for accumulation
                scaled_loss = loss * loss_scale
                
                # bfloat16 doesn't need GradScaler
                if use_scaler:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                accum_loss += loss.item() * loss_scale
            
            # Gradient clipping
            if use_scaler:
                scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), grad_clip
            ).item()
            
            # Update LR (WSD or cosine based on config, with adaptive support)
            if self._adaptive_lr is not None:
                # Adaptive LR: update state with smoothed loss every 100 steps
                # Using per-step losses is too noisy - degradation only visible in smoothed values
                if step % 100 == 0:
                    self._adaptive_lr.update(step, accum_loss)
                lr = self._adaptive_lr.get_lr(step)
            else:
                lr = get_lr(step, config)
            for pg in param_groups:
                pg["lr"] = lr
            
            # Optimizer step
            if use_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            # SWA update (if enabled)
            if self._adaptive_lr is not None:
                self._adaptive_lr.update_swa(base_model)
            
            step += 1
            step_time = time.time() - step_start
            tps = tokens_per_step / step_time
            
            # Check for sequence length transition (stepped sequence training)
            target_seq_len = self._get_seq_len_for_step(step)
            if target_seq_len != self._current_seq_len:
                self._recreate_dataloader(target_seq_len)
                tokens_per_step = self._tokens_per_step  # Update cached value
            
            # Track initial loss (for both fresh training and resume)
            if step == start_step + 1:
                metrics.initial_loss = accum_loss
                if start_step == 0:
                    print(f"Initial loss: {accum_loss:.4f}")
                else:
                    print(f"Resume loss: {accum_loss:.4f} (previous best: {metrics.best_loss:.4f})")
            
            # Update metrics and check for new best
            prev_best = metrics.best_loss
            metrics.update(step, accum_loss, lr, grad_norm, tps, step_time)
            metrics.total_tokens += tokens_per_step
            
            # Best checkpoint logic:
            # - Check every 500 steps (not every step)
            # - Save if >20% improvement over previous best
            # - Otherwise save every 1000 steps regardless
            if step >= 1000 and step % 500 == 0:
                improvement = (prev_best - metrics.best_loss) / prev_best if prev_best < float('inf') else 0.0
                if improvement > 0.20:
                    # Significant improvement - save immediately
                    self._save_checkpoint(step, best=True)
                elif step % 1000 == 0 and metrics.best_loss < prev_best:
                    # Regular 1000-step checkpoint if any improvement
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


            # Logging - print every 25 steps, detailed diagnostics every 100 steps
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
                
                # Detailed MoD/MoR diagnostics every 100 steps
                if use_mod_mor and step % 100 == 0:
                    self._log_layer_diagnostics(step, accum_loss, lr, grad_norm)
            
            # Checkpoint + Early stopping check
            if step % save_interval == 0:
                self._save_checkpoint(step)
                
                # Early stopping: check if loss is increasing (only after 50% Chinchilla progress)
                if self._should_early_stop(
                    accum_loss if accum_loss > 0 else metrics.losses[-1],
                    step,
                    self._count_params()
                ):
                    print(f"\n⚠️  EARLY STOPPING at step {step}")
                    print(f"   Loss has increased for {config.early_stop_patience} consecutive checkpoints")
                    print(f"   Checkpoint losses: {self._checkpoint_losses[-4:]}")
                    break
            
            # Reset accumulator
            accum_loss = 0.0
        
        metrics.end_time = time.time()
        metrics.final_loss = metrics.losses[-1] if metrics.losses else 0.0
        
        print("-" * 70)
        print("Training complete!")
        
        # Apply SWA weights before final save (if enabled and collected samples)
        if self._adaptive_lr is not None and self._adaptive_lr.swa_n > 0:
            base_model = self.model
            if hasattr(base_model, "_orig_mod"):
                base_model = base_model._orig_mod
            self._adaptive_lr.apply_swa(base_model)
        
        # Save final checkpoint
        self._save_checkpoint(step, final=True)
        
        # Generate report
        self._generate_report()
        
        # Save final diagnostics
        self._save_diagnostics()
        
        return metrics
    
    def _log_layer_diagnostics(self, step: int, loss: float, lr: float, grad_norm: float) -> None:
        """Log detailed per-layer MoD/MoR diagnostics for debugging.
        
        This examines each layer to detect:
        - Router collapse (probs too low/high)
        - Depth collapse (MoR always using same depth)
        - Gradient flow issues
        
        MoD Verification: Logs k_selected, selected_frac, and routing mode to verify
        that MoD is truly skipping compute (hard mode) vs just weighting (soft mode).
        """
        model = self.model
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
        
        # Get routing stats from model
        if hasattr(model, "get_routing_stats"):
            stats = model.get_routing_stats()
        else:
            return
        
        # Build diagnostics record
        record = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "losses": {
                "total": loss,
                "ce": self._last_ce_loss,
                "aux": self._last_aux_loss,
                "ponder": self._last_ponder_loss,
            },
            "lr": lr,
            "grad_norm": grad_norm,
            "mod_layers": [],
            "mor_layers": [],
        }
        
        # Analyze MoD layers - verify k_selected, selected_frac, routing mode
        mod_stats = stats.get("mod_layers", [])
        mod_issues = []
        for layer_stat in mod_stats:
            layer_idx = layer_stat.get("layer", -1)
            probs_mean = layer_stat.get("probs_mean", 0.5)
            target = layer_stat.get("target_capacity", 0.5)
            
            # MoD verification stats: k_selected (tokens_processed), selected_frac (compute_ratio), routing_mode
            tokens_processed = layer_stat.get("tokens_processed", 0)
            tokens_total = layer_stat.get("tokens_total", 0)
            compute_ratio = layer_stat.get("compute_ratio", 1.0)
            compute_savings = layer_stat.get("compute_savings_pct", 0.0)
            routing_mode = layer_stat.get("routing_mode", "unknown")
            global_step = layer_stat.get("global_step", 0)
            warmup_steps = layer_stat.get("warmup_steps", 100)
            
            # Check for router collapse
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
            
            record["mod_layers"].append({
                "layer": layer_idx,
                "probs_mean": probs_mean,
                "probs_std": layer_stat.get("probs_std", 0),
                "target": target,
                "k_selected": tokens_processed,  # How many tokens were selected for MLP
                "k_total": tokens_total,         # Total tokens in batch
                "selected_frac": compute_ratio,  # Fraction of tokens processed
                "compute_savings_pct": compute_savings,  # % compute saved
                "routing_mode": routing_mode,    # "soft" (warmup) or "hard" (real skipping)
                "global_step": global_step,
                "warmup_steps": warmup_steps,
                "status": status,
            })
        
        # Analyze MoR layers
        mor_stats = stats.get("mor_layers", [])
        mor_issues = []
        for layer_stat in mor_stats:
            layer_idx = layer_stat.get("layer", -1)
            avg_depth = layer_stat.get("avg_depth", 0)
            expected_depth = layer_stat.get("expected_avg_depth", 1.5)
            router_probs = layer_stat.get("router_probs_mean", 0.5)
            depth_hist = layer_stat.get("depth_histogram", [])
            
            # Check for depth collapse
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
            
            record["mor_layers"].append({
                "layer": layer_idx,
                "avg_depth": avg_depth,
                "expected_depth": expected_depth,
                "router_probs_mean": router_probs,
                "depth_histogram": depth_hist,
                "status": status,
            })
        
        # Store record
        self._diagnostics_data.append(record)
        
        # Print compact summary (streamlined for long runs)
        # Get MoR phase
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
        
        # MoD summary
        mod_mode = "N/A"
        mod_savings = 0.0
        if mod_stats:
            mod_mode = mod_stats[0].get("routing_mode", "?").upper()
            mod_savings = sum(s.get("compute_savings_pct", 0) for s in mod_stats) / max(1, len(mod_stats))
        
        # MoR summary
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
        
        # Only print issues if any
        if mod_issues or mor_issues:
            print(f"  ⚠️  Issues: MoD={len(mod_issues)}, MoR={len(mor_issues)}")
        
        # Write to JSON file every 500 steps
        if step % 500 == 0:
            print(f"  [DIAG] MoD:{mod_mode} save={mod_savings:.0f}% | MoR:{mor_phase} d={mor_depth:.2f} [{depth_dist}%] | CE={self._last_ce_loss:.3f} aux={self._last_aux_loss:.4f} ponder={self._last_ponder_loss:.3f}")
            self._save_diagnostics()
    
    def _save_diagnostics(self) -> None:
        """Save diagnostics data to JSON file."""
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
        """Save model checkpoint."""
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
        
        # Get model state (handle compiled model)
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
        
        # Save adaptive LR manager state if enabled
        if self._adaptive_lr is not None:
            checkpoint["adaptive_lr_state"] = self._adaptive_lr.get_state()
            # Save SWA model if it exists
            if hasattr(self._adaptive_lr, '_swa_model') and self._adaptive_lr._swa_model is not None:
                checkpoint["swa_model"] = self._adaptive_lr._swa_model.state_dict()
        
        torch.save(checkpoint, ckpt_path)
        if best:
            print(f"🏆 New best! Loss: {self.metrics.best_loss:.4f} → {ckpt_path}")
        elif final:
            print(f"Checkpoint saved: {ckpt_path}")
        else:
            print(f"Checkpoint saved: {ckpt_path}")
            # Track periodic checkpoints for rotation (not best/final)
            self._checkpoint_history.append(ckpt_path)
            self._cleanup_old_checkpoints()
    
    def _cleanup_old_checkpoints(self) -> None:
        """Keep only the last N periodic checkpoints."""
        max_ckpts = self.config.max_checkpoints
        while len(self._checkpoint_history) > max_ckpts:
            old_ckpt = self._checkpoint_history.pop(0)
            if old_ckpt.exists():
                old_ckpt.unlink()
                print(f"   Removed old checkpoint: {old_ckpt.name}")
    
    def _should_early_stop(self, current_loss: float, current_step: int, total_params: int) -> bool:
        """Check if training should stop due to loss collapse/reversal.
        
        Only activates after 50% of Chinchilla-optimal training tokens.
        Chinchilla rule: ~20 tokens per parameter for compute-optimal training.
        """
        config = self.config
        
        self._checkpoint_losses.append(current_loss)
        
        # Calculate Chinchilla-optimal tokens and current progress
        chinchilla_tokens = total_params * config.chinchilla_multiplier
        tokens_so_far = current_step * config.tokens_per_step
        progress = tokens_so_far / chinchilla_tokens
        
        # Don't check early stopping until we've trained at least 50% of Chinchilla tokens
        if progress < config.early_stop_min_progress:
            return False
        
        # Need at least 2 checkpoints to compare
        if len(self._checkpoint_losses) < 2:
            return False
        
        prev_loss = self._checkpoint_losses[-2]
        
        # Calculate relative increase (e.g., 0.10 = 10% increase)
        relative_increase = (current_loss - prev_loss) / prev_loss if prev_loss > 0 else 0
        
        # Check if loss increased by more than threshold percentage
        if relative_increase > config.early_stop_threshold:
            self._early_stop_counter += 1
            print(f"   ⚠️  Loss increased: {prev_loss:.4f} → {current_loss:.4f} (+{relative_increase*100:.1f}%) [{self._early_stop_counter}/{config.early_stop_patience}]")
            print(f"       (Chinchilla progress: {progress*100:.1f}%, early stop active)")
        else:
            self._early_stop_counter = 0  # Reset if loss decreased or stayed flat
        
        return self._early_stop_counter >= config.early_stop_patience
    
    def _generate_report(self) -> None:
        """Generate comprehensive training report."""
        config = self.config
        metrics = self.metrics
        
        report_dir = Path(config.report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Calculate statistics
        training_time = metrics.end_time - metrics.start_time
        avg_tps = metrics.total_tokens / training_time if training_time > 0 else 0
        loss_reduction = (metrics.initial_loss - metrics.final_loss) / metrics.initial_loss * 100 if metrics.initial_loss > 0 else 0
        
        # Loss statistics
        losses = metrics.losses
        if losses:
            avg_loss = sum(losses) / len(losses)
            min_loss = min(losses)
            max_loss = max(losses)
            
            # Calculate loss at different points
            n = len(losses)
            loss_at_25 = losses[n // 4] if n > 4 else losses[-1]
            loss_at_50 = losses[n // 2] if n > 2 else losses[-1]
            loss_at_75 = losses[3 * n // 4] if n > 4 else losses[-1]
        else:
            avg_loss = min_loss = max_loss = 0
            loss_at_25 = loss_at_50 = loss_at_75 = 0
        
        # Throughput statistics
        tps_list = metrics.tokens_per_sec
        if tps_list:
            # Skip first few warmup steps for accurate stats
            warmup_skip = min(10, len(tps_list) // 10)
            steady_tps = tps_list[warmup_skip:] if len(tps_list) > warmup_skip else tps_list
            avg_tps_steady = sum(steady_tps) / len(steady_tps) if steady_tps else 0
            peak_tps = max(tps_list)
        else:
            avg_tps_steady = peak_tps = 0
        
        # Build report
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
                "average_step_time_ms": sum(metrics.step_times) / len(metrics.step_times) * 1000 if metrics.step_times else 0,
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
        
        # Save JSON report
        report_path = report_dir / f"training_report_{timestamp}.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        
        # Print summary
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
        """Assess model learning quality."""
        assessment = {}
        
        # Loss reduction assessment
        reduction = (metrics.initial_loss - metrics.final_loss) / metrics.initial_loss * 100 if metrics.initial_loss > 0 else 0
        
        if reduction >= 60:
            assessment["learning_quality"] = "Excellent - Strong convergence"
        elif reduction >= 40:
            assessment["learning_quality"] = "Good - Solid learning"
        elif reduction >= 20:
            assessment["learning_quality"] = "Fair - Moderate progress"
        else:
            assessment["learning_quality"] = "Poor - May need more steps or tuning"
        
        # Compare to random baseline (log(vocab_size) ≈ 10.82 for GPT-2)
        random_baseline = math.log(self.config.vocab_size)
        final_vs_random = (random_baseline - metrics.final_loss) / random_baseline * 100
        
        if metrics.final_loss < 5.0:
            assessment["vs_random_baseline"] = f"{final_vs_random:.1f}% better - Approaching usable"
        elif metrics.final_loss < 7.0:
            assessment["vs_random_baseline"] = f"{final_vs_random:.1f}% better - Learning patterns"
        else:
            assessment["vs_random_baseline"] = f"{final_vs_random:.1f}% better - Early training"
        
        # Convergence trend
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
        """Assess training stability and quality."""
        assessment = {}
        
        # Gradient stability
        if metrics.grad_norms:
            avg_grad = sum(metrics.grad_norms) / len(metrics.grad_norms)
            max_grad = max(metrics.grad_norms)
            
            if max_grad <= 1.0:
                assessment["gradient_stability"] = "Excellent - Well controlled"
            elif max_grad <= 5.0:
                assessment["gradient_stability"] = "Good - Occasional spikes"
            else:
                assessment["gradient_stability"] = f"Warning - Max grad {max_grad:.1f}"
        
        # Throughput assessment
        if metrics.tokens_per_sec:
            avg_tps = sum(metrics.tokens_per_sec) / len(metrics.tokens_per_sec)
            
            if avg_tps >= 30000:
                assessment["throughput"] = f"Excellent - {avg_tps/1000:.1f}K tok/s"
            elif avg_tps >= 15000:
                assessment["throughput"] = f"Good - {avg_tps/1000:.1f}K tok/s"
            else:
                assessment["throughput"] = f"Moderate - {avg_tps/1000:.1f}K tok/s"
        
        # Loss smoothness
        if len(metrics.losses) >= 20:
            # Calculate variance in loss changes
            loss_changes = [
                abs(metrics.losses[i] - metrics.losses[i-1]) 
                for i in range(1, len(metrics.losses))
            ]
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
        """Format seconds to human-readable string."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins}m {secs}s"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{hours}h {mins}m"
    
    def close(self) -> None:
        """Cleanup resources."""
        if hasattr(self, 'train_loader') and self.train_loader:
            self.train_loader.close()


# ============================================
# Main Entry Point
# ============================================
def main(
    architecture: str = "vanilla", 
    mode: str = "testing", 
    resume_from: Optional[str] = None, 
    max_steps_override: Optional[int] = None,
    mor_enable_pct: float = 0.30,
    mor_already_enabled: bool = False,
    mod_capacity: float = 0.5,
    mor_adaptive: bool = True,
    aux_scale: float = 0.1,
    ponder_scale: float = 0.01,
    recalc_lr_schedule: bool = False,
    adaptive_lr: bool = False,
    use_swa: bool = False,
    swa_start_pct: float = 0.75,
    batch_filter: bool = False,
    batch_filter_threshold: float = 2.5,
    batch_size: Optional[int] = None,
    grad_accum_steps: Optional[int] = None,
    dataset_name: str = "fineweb_edu",
    model_size: str = "100M",
):
    """Main training entry point.
    
    Args:
        architecture: "vanilla" or "mod_mor"
        mode: "testing" (5K), "production" (100K), or "chinchilla_third" (90K)
        resume_from: Path to checkpoint to resume from
        max_steps_override: Override max steps (for diagnostic runs, e.g., 500 or 1000)
        mor_enable_pct: MoR curriculum - enable adaptive after this % of training (0.0-1.0)
        mor_already_enabled: Restart flag - set True if resuming after MoR was enabled
        mod_capacity: MoD capacity ratio (0.5=50%, 1.0=all tokens=MoD OFF)
        mor_adaptive: MoR adaptive routing (True=on, False=fixed-depth only)
        aux_scale: MoD auxiliary loss scale (0.1 default)
        ponder_scale: MoR ponder loss scale (0.01 default, use ~1e-4 for weak reg)
        adaptive_lr: Enable loss-triggered early cooldown when degradation detected
        use_swa: Enable Stochastic Weight Averaging for better final model
        swa_start_pct: Start SWA after this % of training (default 75%)
        batch_filter: Enable loss-based batch filtering to skip bad data
        batch_filter_threshold: Skip batch if loss > threshold * running_avg
        batch_size: Micro batch size per GPU (None = use mode default)
        grad_accum_steps: Gradient accumulation steps (None = use mode default)
        model_size: Model size preset ("100M" or "500M")
    """
    
    # Model size configurations
    MODEL_SIZE_CONFIGS = {
        "100M": {
            "mod_mor_dim": 1280,
            "n_mor_blocks": 8,
            "mor_recursions": 3,
            "mod_mor_n_heads": 20,  # 1280/64
            "mod_mor_n_kv_heads": 5,  # 1280/256
            "default_batch_size": 32,  # Can fit ~32 on 32GB GPU
            "default_grad_accum": 4,
        },
        "500M": {
            "mod_mor_dim": 2048,
            "n_mor_blocks": 10,
            "mor_recursions": 3,
            "mod_mor_n_heads": 32,  # 2048/64
            "mod_mor_n_kv_heads": 8,  # 2048/256
            "default_batch_size": 16,  # Smaller batch for 500M
            "default_grad_accum": 8,  # Higher accum to maintain effective batch
        },
    }
    
    size_config = MODEL_SIZE_CONFIGS.get(model_size, MODEL_SIZE_CONFIGS["100M"])
    print(f"\n🔧 MODEL SIZE: {model_size}")
    print(f"   dim={size_config['mod_mor_dim']}, blocks={size_config['n_mor_blocks']}, "
          f"heads={size_config['mod_mor_n_heads']}")
    
    # Configuration for 100M variant testing with WSD scheduler
    config = TrainingConfig(
        # Architecture selection
        architecture=architecture,
        mode=mode,
        resume_from=resume_from,
        
        # MoR Curriculum settings
        mor_enable_pct=mor_enable_pct,
        mor_already_enabled=mor_already_enabled,
        
        # MoD/MoR override settings
        mod_capacity=mod_capacity,
        mor_adaptive=mor_adaptive,
        
        # Auxiliary loss scales
        aux_scale=aux_scale,
        ponder_scale=ponder_scale,
        
        # Adaptive LR settings
        adaptive_lr=adaptive_lr,
        use_swa=use_swa,
        swa_start_pct=swa_start_pct,
        
        # Batch filtering settings
        batch_filter=batch_filter,
        batch_filter_threshold=batch_filter_threshold,
        
        # Model size preset
        model_size=model_size,
        
        # MoD+MoR model dimensions (from size config)
        mod_mor_dim=size_config["mod_mor_dim"],
        n_mor_blocks=size_config["n_mor_blocks"],
        mor_recursions=size_config["mor_recursions"],
        mod_mor_n_heads=size_config["mod_mor_n_heads"],
        mod_mor_n_kv_heads=size_config["mod_mor_n_kv_heads"],
        
        # Vanilla model config (for fallback)
        dim=768,
        n_macro_blocks=3,
        n_heads=12,
        n_kv_heads=3,
        max_seq_len=512,  # Shorter for faster iteration
        
        # Training - batch size is critical for scaling!
        # Default batch sizes depend on model size to fit in GPU memory
        batch_size=batch_size if batch_size is not None else size_config["default_batch_size"],
        grad_accum_steps=grad_accum_steps if grad_accum_steps is not None else size_config["default_grad_accum"],
        max_steps=5000,
        
        # WSD Schedule: Warmup 3% -> Stable 67% -> Decay 30%
        # This keeps high LR much longer than cosine, preventing stagnation
        lr_schedule="wsd",
        warmup_steps=150,           # 3% warmup
        decay_start_step=3500,      # Start decay at 70% (after 67% stable)
        decay_steps=1500,           # 30% linear decay
        max_lr=5e-4 if model_size == "100M" else 3e-4,  # Lower LR for larger models
        min_lr=1.5e-4 if model_size == "100M" else 1e-4,
        
        # Dataset
        dataset_name=dataset_name,
        
        # Optimization
        use_compile=True,
        compile_mode="max-autotune-no-cudagraphs",  # "default" avoids CUDA graphs, works with grad accum
        dtype="bfloat16",
        
        # Logging - 25 steps for print, diagnostics every 100
        log_interval=25,
        save_interval=500,
    )
    
    # Apply max_steps override if provided
    if max_steps_override is not None:
        config.max_steps = max_steps_override
        # Adjust save interval for short runs - use 500 for longer diagnostic runs
        if max_steps_override >= 2000:
            config.save_interval = 500  # Every 500 steps for longer runs
        else:
            config.save_interval = min(config.save_interval, max(100, max_steps_override // 5))
        print(f"\n⚠️  DIAGNOSTIC RUN: max_steps={max_steps_override}, save_interval={config.save_interval}")
    
    # Recalculate LR schedule for extended/resume training
    # This ensures WSD decay happens at the right point relative to ACTUAL max_steps
    if recalc_lr_schedule:
        # WSD schedule: 85% stable, 15% decay (no re-warmup on resume)
        config.decay_start_step = int(config.max_steps * 0.85)
        config.decay_steps = config.max_steps - config.decay_start_step
        config.warmup_steps = 0 if resume_from else int(config.max_steps * 0.01)  # Skip warmup on resume
        print(f"\n📈 LR SCHEDULE RECALCULATED for max_steps={config.max_steps:,}:")
        print(f"   Warmup:      {config.warmup_steps:,} steps")
        print(f"   Stable:      steps {config.warmup_steps:,} - {config.decay_start_step:,} at LR={config.max_lr}")
        print(f"   Decay:       steps {config.decay_start_step:,} - {config.max_steps:,} (LR {config.max_lr} -> {config.min_lr})")
    
    # Create and run trainer
    trainer = Trainer(config)
    
    try:
        metrics = trainer.train()
        
        print("\n✅ Training completed successfully!")
        print(f"   Final loss: {metrics.final_loss:.4f}")
        print(f"   Best loss: {metrics.best_loss:.4f}")
        print(f"   Total tokens: {metrics.total_tokens:,}")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        raise
    
    finally:
        trainer.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="HYDRA Training")
    parser.add_argument("--arch", type=str, default="vanilla", choices=["vanilla", "mod_mor"],
                        help="Architecture: vanilla or mod_mor")
    parser.add_argument("--mode", type=str, default="testing", choices=["testing", "production", "chinchilla_third"],
                        help="Mode: testing (5K), production (100K), chinchilla_third (1/3 Chinchilla)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Override max steps (for diagnostic runs, e.g., 500 or 1000)")
    parser.add_argument("--mor_enable_pct", type=float, default=0.30,
                        help="MoR curriculum: enable adaptive routing after this %% of training (0.0-1.0, default 0.30)")
    parser.add_argument("--mor_already_enabled", action="store_true",
                        help="Restart flag: set if resuming AFTER MoR was already enabled")
    parser.add_argument("--mod_capacity", type=float, default=0.5,
                        help="MoD capacity ratio (0.5=50%% tokens, 1.0=all tokens=MoD OFF)")
    parser.add_argument("--mor_adaptive", type=str, default="true", choices=["true", "false"],
                        help="MoR adaptive routing (true=on, false=fixed-depth only)")
    parser.add_argument("--aux_scale", type=float, default=0.1,
                        help="MoD aux loss scale (0.1 default, 0.0=MoD loss OFF)")
    parser.add_argument("--ponder_scale", type=float, default=0.01,
                        help="MoR ponder loss scale (0.01 default, 1e-4=weak reg)")
    parser.add_argument("--recalc_lr", action="store_true",
                        help="Recalculate LR schedule based on max_steps (85%% stable, 15%% decay)")
    # Adaptive LR arguments
    parser.add_argument("--adaptive_lr", action="store_true",
                        help="Enable adaptive LR: auto-trigger cooldown when loss spikes")
    parser.add_argument("--use_swa", action="store_true",
                        help="Enable Stochastic Weight Averaging for better final model")
    parser.add_argument("--swa_start_pct", type=float, default=0.75,
                        help="Start SWA at this %% of training (default 0.75)")
    # Batch size arguments (critical for scaling)
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Micro batch size per GPU (default: 8 for 220M, scale down for larger models)")
    parser.add_argument("--grad_accum", type=int, default=None,
                        help="Gradient accumulation steps (effective_batch = batch_size * grad_accum)")
    # Batch filtering arguments
    parser.add_argument("--batch_filter", action="store_true",
                        help="Enable batch filtering: skip batches with loss spikes from bad data")
    parser.add_argument("--batch_filter_threshold", type=float, default=2.5,
                        help="Skip batch if loss > threshold * running_avg (default 2.5)")
    # Dataset argument
    parser.add_argument("--dataset", type=str, default="fineweb_edu",
                        help="Dataset name (default: fineweb_edu, alternatives: finefineweb, synthetic_mix)")
    # Model size argument
    parser.add_argument("--model_size", type=str, default="100M", choices=["100M", "500M"],
                        help="Model size preset: 100M (~220M params) or 500M (~605M params)")
    args = parser.parse_args()
    
    main(
        architecture=args.arch, 
        mode=args.mode, 
        resume_from=args.resume, 
        max_steps_override=args.max_steps,
        mor_enable_pct=args.mor_enable_pct,
        mor_already_enabled=args.mor_already_enabled,
        mod_capacity=args.mod_capacity,
        mor_adaptive=(args.mor_adaptive.lower() == "true"),
        aux_scale=args.aux_scale,
        ponder_scale=args.ponder_scale,
        recalc_lr_schedule=args.recalc_lr,
        adaptive_lr=args.adaptive_lr,
        use_swa=args.use_swa,
        swa_start_pct=args.swa_start_pct,
        batch_filter=args.batch_filter,
        batch_filter_threshold=args.batch_filter_threshold,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        dataset_name=args.dataset,
        model_size=args.model_size,
    )
