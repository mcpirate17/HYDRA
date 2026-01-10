"""HYDRA CLI argument parsing.

Handles command-line interface for trainer.py.
Separates argument definition from config construction.

Usage:
    from hydra.training.cli import (
        normalize_bool_flags,
        build_argument_parser,
        apply_convenience_flags,
    )

    sys.argv = normalize_bool_flags(sys.argv)
    parser = build_argument_parser()
    args = parser.parse_args()
    args = apply_convenience_flags(args)
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List

__all__ = [
    "normalize_bool_flags",
    "build_argument_parser",
    "apply_convenience_flags",
    "BOOL_FLAGS",
]

# ─────────────────────────────────────────────────────────────────────────────
# Bool Flag Normalization
# ─────────────────────────────────────────────────────────────────────────────

BOOL_FLAGS = [
    "--compile",
    "--triton_kernels",
    "--chunked_ce",
    "--wandb",
    "--tensorboard",
    "--profiler",
    "--grad_clip_dynamic",
]


def _normalize_single_bool_flag(argv: List[str], flag: str) -> List[str]:
    """Normalize a single '--flag true/false' to '--flag/--no-flag'.

    Allows backward-compatible usage like '--compile true' instead of
    argparse's BooleanOptionalAction format '--compile' or '--no-compile'.
    """
    out: List[str] = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok == flag and (i + 1) < len(argv):
            nxt = argv[i + 1].lower()
            if nxt in ("true", "1", "yes", "y"):
                out.append(flag)
                i += 2
                continue
            if nxt in ("false", "0", "no", "n"):
                out.append("--no-" + flag.lstrip("-"))
                i += 2
                continue
        out.append(tok)
        i += 1
    return out


def normalize_bool_flags(argv: List[str]) -> List[str]:
    """Normalize all boolean flags in argv for argparse compatibility.

    Converts '--flag true/false' syntax to '--flag/--no-flag' for all
    flags in BOOL_FLAGS.

    Args:
        argv: Command line arguments (typically sys.argv)

    Returns:
        Modified argv with normalized boolean flags
    """
    for flag in BOOL_FLAGS:
        argv = _normalize_single_bool_flag(argv, flag)
    return argv


# ─────────────────────────────────────────────────────────────────────────────
# Argument Parser Construction
# ─────────────────────────────────────────────────────────────────────────────


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the complete argument parser for HYDRA training.

    Returns:
        Configured ArgumentParser with all training options
    """
    parser = argparse.ArgumentParser(description="HYDRA Training")
    _add_core_args(parser)
    _add_routing_args(parser)
    _add_training_args(parser)
    _add_observability_args(parser)
    _add_moe_args(parser)
    _add_experimental_args(parser)
    return parser


def _add_core_args(parser: argparse.ArgumentParser) -> None:
    """Add architecture, mode, and resume arguments."""
    parser.add_argument(
        "--arch",
        type=str,
        default="mod_mor",
        choices=["vanilla", "mod_mor"],
        help="Architecture (note: backend currently supports mod_mor only)",
    )
    parser.add_argument(
        "--attention",
        type=str,
        default="ccgqa",
        choices=["ccgqa"],
        help="Attention backend: 'ccgqa' (Compressed Convolutional GQA, fast + low memory)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="testing",
        choices=["testing", "production", "chinchilla_third"],
        help="Mode: testing (5K), production (100K), chinchilla_third (1/3 Chinchilla)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--resume_ignore_ckpt_lr",
        action="store_true",
        help="On resume, ignore the checkpoint optimizer LR and use the scheduled LR.",
    )
    parser.add_argument(
        "--resume_lr_override",
        type=float,
        default=0.0,
        help="On resume, force LR at the resume step to this value (>0).",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Override max steps (for diagnostic runs, e.g., 500 or 1000)",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="100M",
        choices=["debug", "50M", "DIAG", "100M", "debug_tall_skinny", "250M", "300M", "500M", "750M", "1B", "1.5B"],
        help="Model size preset",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )


def _add_routing_args(parser: argparse.ArgumentParser) -> None:
    """Add MoD/MoR routing configuration arguments."""
    # MoR curriculum
    parser.add_argument(
        "--mor_enable_pct",
        type=float,
        default=0.10,
        help="MoR curriculum: enable adaptive routing after this %% of training (0.0-1.0)",
    )
    parser.add_argument(
        "--mor_enable_min_steps",
        type=int,
        default=3000,
        help="MoR curriculum: minimum steps before MoR can enable (hard floor)",
    )
    parser.add_argument(
        "--mor_enable_loss_threshold",
        type=float,
        default=5.0,
        help="MoR curriculum: enable adaptive routing when EMA CE loss drops below this",
    )
    parser.add_argument(
        "--mor_already_enabled",
        action="store_true",
        help="Restart flag: set if resuming AFTER MoR was already enabled",
    )
    parser.add_argument(
        "--mor_adaptive",
        type=str,
        default="true",
        choices=["true", "false"],
        help="MoR adaptive routing (true=on, false=fixed-depth only)",
    )
    parser.add_argument(
        "--mor_advantage_loss_scale",
        type=float,
        default=0.02,
        help="MoR routing: scale for loss-driven depth allocation. WARNING: >0.05 causes 'advantage gaming'.",
    )
    parser.add_argument(
        "--mor_min_depth",
        type=int,
        default=1,
        help="Minimum MoR recursion depth. 0=allow immediate exit, 1+=force iterations.",
    )

    # MoD configuration
    parser.add_argument(
        "--mod_capacity",
        type=float,
        default=0.5,
        help="MoD capacity ratio (0.5=50%% tokens, 1.0=all tokens=MoD OFF)",
    )
    parser.add_argument(
        "--mod_enable_mor_early_exit_threshold",
        type=float,
        default=0.38,
        help="MoD curriculum: enable when MoR early_exit_ratio exceeds this",
    )
    parser.add_argument(
        "--mod_enable_loss_threshold",
        type=float,
        default=4.5,
        help="MoD curriculum: safety floor - also require EMA CE < threshold",
    )
    parser.add_argument(
        "--mod_loss_aware_weight",
        type=float,
        default=0.0,
        help="Loss-aware MoD: supervise router to prioritize top-k hard tokens (0=off)",
    )
    parser.add_argument(
        "--mod_mlp_warmup_steps",
        type=int,
        default=1000,
        help="MoD warmup: steps to run dense MLP before enabling hard routing",
    )

    # Loss scales
    parser.add_argument(
        "--aux_scale",
        type=float,
        default=0.1,
        help="MoD aux loss scale (0.0=MoD loss OFF)",
    )
    parser.add_argument(
        "--ponder_scale",
        type=float,
        default=0.01,
        help="MoR ponder loss scale",
    )

    # Convenience flags
    parser.add_argument(
        "--mod_off",
        action="store_true",
        help="Disable MoD (forces dense compute and disables MoD-related losses/curriculum)",
    )
    parser.add_argument(
        "--mor_off",
        action="store_true",
        help="Disable MoR (forces fixed-depth only and disables MoR-related losses/curriculum)",
    )
    parser.add_argument(
        "--no_short_run_override",
        action="store_true",
        help="Disable SHORT RUN heuristics that delay MoD/MoR enable for runs <= 10K steps",
    )


def _add_training_args(parser: argparse.ArgumentParser) -> None:
    """Add training hyperparameter arguments."""
    # Learning rate
    parser.add_argument(
        "--max_lr",
        type=float,
        default=None,
        help="Override max learning rate",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=None,
        help="Override min learning rate",
    )
    parser.add_argument(
        "--adaptive_lr",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable adaptive LR: auto-trigger cooldown when loss spikes",
    )
    parser.add_argument(
        "--adaptive_metric",
        type=str,
        default="eval",
        choices=["train", "eval"],
        help="Adaptive LR trigger metric: 'eval' or 'train'",
    )
    parser.add_argument(
        "--adaptive_min_trigger_pct",
        type=float,
        default=0.50,
        help="Adaptive LR guardrail: block cooldown until this %% of run is complete",
    )
    parser.add_argument(
        "--recalc_lr",
        action="store_true",
        help="[DEPRECATED] LR schedule now auto-recalculates on resume. No-op.",
    )

    # SWA
    parser.add_argument(
        "--use_swa",
        action="store_true",
        help="Enable Stochastic Weight Averaging for better final model",
    )
    parser.add_argument(
        "--swa_start_pct",
        type=float,
        default=0.75,
        help="Start SWA at this %% of training",
    )

    # Batch configuration
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Micro batch size per GPU",
    )
    parser.add_argument(
        "--grad_accum",
        type=int,
        default=None,
        help="Gradient accumulation steps (effective_batch = batch_size * grad_accum)",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=None,
        help="Override max sequence length",
    )

    # Batch filtering
    parser.add_argument(
        "--batch_filter",
        action="store_true",
        help="Enable batch filtering: skip batches with loss spikes from bad data",
    )
    parser.add_argument(
        "--batch_filter_threshold",
        type=float,
        default=2.5,
        help="Skip batch if loss > threshold * running_avg",
    )

    # Dataset
    parser.add_argument(
        "--dataset",
        type=str,
        default="pretrain_default",
        help="Dataset name: pretrain_default, finefineweb-sequential, finefineweb-local, etc.",
    )
    parser.add_argument(
        "--finefineweb-local",
        dest="dataset",
        action="store_const",
        const="finefineweb-local",
        help="Alias for '--dataset finefineweb-local'",
    )

    # Optimization flags
    parser.add_argument(
        "--triton_kernels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Triton kernels (safe-by-default subset)",
    )
    parser.add_argument(
        "--chunked_ce",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable chunked cross-entropy (lower peak memory)",
    )
    parser.add_argument(
        "--chunked_ce_size",
        type=int,
        default=4096,
        help="Chunk size for chunked cross-entropy (tokens per chunk)",
    )
    parser.add_argument(
        "--static_routing_mode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable static routing for CUDA graph compatibility. "
        "Computes all tokens through all layers, applies routing masks after. "
        "Trades ~25-50%% more FLOPs for 5-15%% speedup from CUDA graphs.",
    )
    parser.add_argument(
        "--8bit_adam",
        action="store_true",
        dest="use_8bit_adam",
        help="Use 8-bit Adam (bitsandbytes) to save ~75%% optimizer memory",
    )
    parser.add_argument(
        "--adafactor",
        action="store_true",
        dest="use_adafactor",
        help="Use Adafactor optimizer (~25%% memory savings vs AdamW)",
    )
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=2,
        dest="checkpoint_every_n",
        help="Gradient checkpoint every N layers. 1=max memory savings, 2=balanced",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable gradient checkpointing to trade compute for memory",
    )
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable torch.compile for graph optimization",
    )

    # Gradient clipping
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=None,
        help="Override global grad clipping max-norm",
    )
    parser.add_argument(
        "--grad_clip_dynamic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable dynamic gradient clipping (adapts to gradient norm EMA)",
    )
    parser.add_argument(
        "--grad_clip_k",
        type=float,
        default=2.0,
        help="Dynamic clip multiplier on EMA",
    )
    parser.add_argument(
        "--grad_clip_min",
        type=float,
        default=50.0,
        help="Dynamic clip floor",
    )
    parser.add_argument(
        "--grad_clip_max",
        type=float,
        default=3000.0,
        help="Dynamic clip ceiling - SAFETY CAP",
    )

    # Checkpointing
    parser.add_argument(
        "--save_interval",
        type=int,
        default=500,
        help="Save checkpoint every N steps. 0=disable periodic saves.",
    )
    parser.add_argument(
        "--halt_on_spike",
        action="store_true",
        help="Debug: stop training immediately after first gradient spike",
    )

    # Debug flags
    parser.add_argument(
        "--ema_debug",
        action="store_true",
        help="Debug: print EMA updates every 25 steps",
    )
    parser.add_argument(
        "--eval_debug",
        action="store_true",
        help="Debug: run eval sanity check on training batch at first eval",
    )


def _add_observability_args(parser: argparse.ArgumentParser) -> None:
    """Add wandb, tensorboard, profiler arguments."""
    parser.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="W&B project name (default: hydra-llm)",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="W&B entity/team (optional)",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Experiment/run name for observability backends",
    )
    parser.add_argument(
        "--tensorboard",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable TensorBoard scalar logging",
    )
    parser.add_argument(
        "--tensorboard_dir",
        type=str,
        default=None,
        help="TensorBoard log dir (default: runs)",
    )
    parser.add_argument(
        "--profiler",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable torch.profiler traces",
    )
    parser.add_argument(
        "--profiler_dir",
        type=str,
        default=None,
        help="Profiler trace output dir (default: profiler_traces)",
    )


def _add_moe_args(parser: argparse.ArgumentParser) -> None:
    """Add Mixture-of-Experts arguments."""
    # Core MoE settings
    parser.add_argument(
        "--moe",
        action="store_true",
        dest="moe_enabled",
        help="Enable Mixture of Experts (sparse FFN routing)",
    )
    parser.add_argument(
        "--moe_num_experts",
        type=int,
        default=0,
        help="Number of expert FFNs (0=auto-scale by model size)",
    )
    parser.add_argument(
        "--moe_num_layers",
        type=int,
        default=0,
        help="Number of MoE layers to insert (0=auto-scale)",
    )
    parser.add_argument(
        "--moe_top_k",
        type=int,
        default=1,
        help="Number of experts per token (1=top-1, 2=top-2)",
    )
    parser.add_argument(
        "--moe_aux_weight",
        type=float,
        default=0.0001,
        help="MoE load-balancing auxiliary loss weight",
    )
    parser.add_argument(
        "--moe_router_jitter",
        type=float,
        default=0.15,
        help="Router jitter noise for exploration",
    )
    parser.add_argument(
        "--moe_expert_diversity_noise",
        type=float,
        default=0.05,
        help="Additive weight noise to break expert symmetry",
    )
    parser.add_argument(
        "--moe_warmup_steps",
        type=int,
        default=1000,
        help="MoE warmup steps (for checkpoint cloning)",
    )
    parser.add_argument(
        "--moe_no_identity_init",
        action="store_true",
        help="Disable identity-preserving init for MoE",
    )
    parser.add_argument(
        "--moe_track_divergence",
        action="store_true",
        help="Track expert weight divergence during training",
    )
    parser.add_argument(
        "--moe_divergence_interval",
        type=int,
        default=100,
        help="Steps between divergence checks",
    )
    parser.add_argument(
        "--moe_forced_routing_steps",
        type=int,
        default=0,
        help="Steps to force position-based routing for expert diversification",
    )

    # Domain teacher routing
    parser.add_argument(
        "--moe_domain_expert_map",
        type=str,
        default="",
        help="Comma-separated mapping from batch source_name to expert id",
    )
    parser.add_argument(
        "--moe_teacher_weight",
        type=float,
        default=0.0,
        help="Alpha for domain-teacher routing loss",
    )
    parser.add_argument(
        "--moe_teacher_until_step",
        type=int,
        default=0,
        help="Apply teacher loss until this global step (0=forever)",
    )

    # Per-component LR scaling
    parser.add_argument(
        "--moe_expert_lr_scale",
        type=float,
        default=1.0,
        help="LR multiplier for expert/MLP weights",
    )
    parser.add_argument(
        "--moe_router_lr_scale",
        type=float,
        default=1.0,
        help="LR multiplier for router/gate weights",
    )
    parser.add_argument(
        "--moe_lr_rewarmup_steps",
        type=int,
        default=0,
        help="Steps to linearly ramp LR from 0 after mid-run restart",
    )
    parser.add_argument(
        "--moe_expert_weight_decay_scale",
        type=float,
        default=1.0,
        help="Weight decay multiplier for MoE experts",
    )


def _add_experimental_args(parser: argparse.ArgumentParser) -> None:
    """Add experimental optimization arguments.

    These optimizations use SafeOptimizations wrapper with auto-fallback
    if anomalies (loss spikes, NaN grads, throughput drops) are detected.
    """
    # Flash Attention 3
    parser.add_argument(
        "--experimental_fa3",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Flash Attention 3 (Hopper/Blackwell). Auto-falls back to FA2 on failure.",
    )

    # CUDA Graphs
    parser.add_argument(
        "--experimental_cuda_graphs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable CUDA graph capture for reduced kernel launch overhead.",
    )
    parser.add_argument(
        "--cuda_graphs_warmup",
        type=int,
        default=50,
        help="Steps before CUDA graph capture (warmup period)",
    )

    # Blackwell-specific Triton tuning
    parser.add_argument(
        "--experimental_blackwell_tuning",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Blackwell-optimized Triton kernel configs (block sizes, warps).",
    )
    parser.add_argument(
        "--triton_block_q",
        type=int,
        default=128,
        help="Triton query block size (Blackwell: 128, others: 64)",
    )
    parser.add_argument(
        "--triton_block_kv",
        type=int,
        default=64,
        help="Triton KV block size",
    )
    parser.add_argument(
        "--triton_num_warps",
        type=int,
        default=8,
        help="Triton warp count (Blackwell: 8, others: 4)",
    )

    # Multi-threaded prefetch
    parser.add_argument(
        "--experimental_prefetch_threads",
        type=int,
        default=4,
        help="Number of data prefetch threads (0=disabled)",
    )
    parser.add_argument(
        "--prefetch_buffer_size",
        type=int,
        default=8,
        help="Number of batches to prefetch",
    )

    # FP8 (conservative default)
    parser.add_argument(
        "--experimental_fp8",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable FP8 compute (experimental, Hopper/Blackwell only)",
    )
    parser.add_argument(
        "--fp8_format",
        type=str,
        default="e4m3",
        choices=["e4m3", "e5m2"],
        help="FP8 format: e4m3 (more precision) or e5m2 (more range)",
    )

    # Safety monitoring
    parser.add_argument(
        "--experimental_safety_window",
        type=int,
        default=100,
        help="Steps to monitor after enabling experimental optimization",
    )
    parser.add_argument(
        "--experimental_loss_spike_threshold",
        type=float,
        default=2.0,
        help="Disable optimization if loss > threshold * EMA",
    )
    parser.add_argument(
        "--experimental_throughput_drop_threshold",
        type=float,
        default=0.5,
        help="Disable optimization if throughput < threshold * EMA",
    )

    # Pretest
    parser.add_argument(
        "--experimental_pretest_steps",
        type=int,
        default=10,
        help="Steps to run for optimization pretest",
    )
    parser.add_argument(
        "--experimental_skip_pretest",
        action="store_true",
        help="Skip pretests and enable all optimizations immediately (risky)",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Convenience Flag Application
# ─────────────────────────────────────────────────────────────────────────────


def apply_convenience_flags(args: argparse.Namespace) -> argparse.Namespace:
    """Apply convenience flags and resolve paths.

    Handles:
    - --mod_off: Disables MoD (sets capacity=1.0, disables losses)
    - --mor_off: Disables MoR (sets adaptive=false, disables losses)
    - Resume path resolution (tries checkpoints/ if not found)
    - --arch vanilla deprecation warning

    Args:
        args: Parsed argparse namespace

    Returns:
        Modified args namespace
    """
    # Apply --mod_off: disable MoD entirely
    if args.mod_off:
        args.mod_capacity = 1.0
        args.mod_enable_mor_early_exit_threshold = 1.0  # Never triggers
        args.mod_enable_loss_threshold = 0.0
        args.mod_loss_aware_weight = 0.0
        args.aux_scale = 0.0
        args.mod_mlp_warmup_steps = 10**9  # Effectively never

    # Apply --mor_off: disable MoR entirely
    if args.mor_off:
        args.mor_adaptive = "false"
        args.mor_enable_pct = 1.0
        args.mor_already_enabled = False
        args.ponder_scale = 0.0
        args.mor_advantage_loss_scale = 0.0

    # Resolve resume checkpoint path
    if args.resume is not None and not os.path.exists(args.resume):
        candidate = os.path.join("checkpoints", os.path.basename(args.resume))
        if os.path.exists(candidate):
            print(f"\n🔁 RESUME: '{args.resume}' -> '{candidate}'")
            args.resume = candidate
        else:
            raise FileNotFoundError(
                f"Resume checkpoint not found: '{args.resume}'. "
                f"Also tried: '{candidate}'."
            )

    # Backend compatibility: only mod_mor is implemented
    if args.arch == "vanilla":
        print("\n⚠️  NOTE: --arch vanilla is not implemented; using mod_mor instead.")
        args.arch = "mod_mor"

    return args
