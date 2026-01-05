from __future__ import annotations

import argparse
import os
import sys

from hydra.training import Trainer, TrainingConfig, MODEL_SIZE_CONFIGS


def main(config: TrainingConfig) -> None:
    """
    Main training entry point.
    
    Args:
        config: Fully populated TrainingConfig object.
    """
    print(f"\n🔧 MODEL SIZE: {config.model_size}")
    print(
        f"   dim={config.mod_mor_dim}, blocks={config.n_mor_blocks}, "
        f"heads={config.mod_mor_n_heads}"
    )

    config.print_summary()

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


def run_cli() -> None:
    # Back-compat / convenience: allow '--flag true/false' for bool flags.
    # argparse.BooleanOptionalAction expects '--flag' or '--no-flag' (no value).
    def _normalize_bool_flag(argv: list[str], flag: str) -> list[str]:
        out: list[str] = []
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
                    out.append("--no-" + flag.lstrip("-") )
                    i += 2
                    continue
            out.append(tok)
            i += 1
        return out

    sys.argv = _normalize_bool_flag(sys.argv, "--compile")
    sys.argv = _normalize_bool_flag(sys.argv, "--triton_kernels")
    sys.argv = _normalize_bool_flag(sys.argv, "--chunked_ce")
    sys.argv = _normalize_bool_flag(sys.argv, "--wandb")
    sys.argv = _normalize_bool_flag(sys.argv, "--tensorboard")
    sys.argv = _normalize_bool_flag(sys.argv, "--profiler")

    parser = argparse.ArgumentParser(description="HYDRA Training")
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
        help="Attention backend: ''ccgqa' (Compressed Convolutional GQA, fast + low memory)",
    )
    parser.add_argument("--mode", type=str, default="testing", choices=["testing", "production", "chinchilla_third"], help="Mode: testing (5K), production (100K), chinchilla_third (1/3 Chinchilla)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument(
        "--resume_ignore_ckpt_lr",
        action="store_true",
        help="On resume, ignore the checkpoint optimizer LR and use the scheduled LR (disables resume LR alignment).",
    )
    parser.add_argument(
        "--resume_lr_override",
        type=float,
        default=0.0,
        help="On resume, force LR at the resume step to this value (>0). Overrides checkpoint LR alignment.",
    )
    parser.add_argument("--max_steps", type=int, default=None, help="Override max steps (for diagnostic runs, e.g., 500 or 1000)")
    parser.add_argument("--mor_enable_pct", type=float, default=0.10, help="MoR curriculum: enable adaptive routing after this %% of training (0.0-1.0, default 0.10). Separate from LR warmup.")
    parser.add_argument("--mor_enable_min_steps", type=int, default=3000, help="MoR curriculum: minimum steps before MoR can enable (hard floor, default 3000). Uses ABSOLUTE step count from start of training.")
    parser.add_argument(
        "--mor_enable_loss_threshold",
        type=float,
        default=5.0,
        help="MoR curriculum: enable adaptive routing when EMA CE loss drops below this (default 5.0). Whichever triggers first (pct or loss) wins.",
    )
    parser.add_argument("--mor_already_enabled", action="store_true", help="Restart flag: set if resuming AFTER MoR was already enabled")
    parser.add_argument("--mod_capacity", type=float, default=0.5, help="MoD capacity ratio (0.5=50%% tokens, 1.0=all tokens=MoD OFF)")
    parser.add_argument(
        "--mod_enable_mor_early_exit_threshold",
        type=float,
        default=0.38,
        help="MoD curriculum: enable when MoR early_exit_ratio exceeds this (default 0.38 = 38%% tokens exit early). MoR-informed triggering.",
    )
    parser.add_argument(
        "--mod_enable_loss_threshold",
        type=float,
        default=4.5,
        help="MoD curriculum: safety floor - also require EMA CE < threshold (default 4.5). Set to 0 to disable.",
    )
    parser.add_argument(
        "--mod_loss_aware_weight",
        type=float,
        default=0.0,
        help="Loss-aware MoD: supervise router to prioritize top-k hard tokens by per-token CE loss (0=off).",
    )
    parser.add_argument(
        "--mod_mlp_warmup_steps",
        type=int,
        default=1000,
        help="MoD warmup: number of steps to run dense MLP before enabling hard routing (default 1000).",
    )
    parser.add_argument("--mor_adaptive", type=str, default="true", choices=["true", "false"], help="MoR adaptive routing (true=on, false=fixed-depth only)")
    parser.add_argument("--aux_scale", type=float, default=0.1, help="MoD aux loss scale (0.1 default, 0.0=MoD loss OFF)")
    parser.add_argument("--ponder_scale", type=float, default=0.01, help="MoR ponder loss scale (0.01 default, 1e-4=weak reg)")
    parser.add_argument(
        "--mor_advantage_loss_scale",
        type=float,
        default=0.02,
        help="MoR routing: scale for loss-driven depth allocation. WARNING: >0.05 causes 'advantage gaming'. Default: 0.02.",
    )
    parser.add_argument(
        "--mor_min_depth",
        type=int,
        default=1,
        help="Minimum MoR recursion depth. 0=allow immediate exit, 1+=force iterations. Default: 1 (prevents depth-0 collapse).",
    )

    # Convenience flags
    parser.add_argument(
        "--mod_off",
        action="store_true",
        help="Disable MoD (forces dense compute and disables MoD-related losses/curriculum).",
    )
    parser.add_argument(
        "--mor_off",
        action="store_true",
        help="Disable MoR (forces fixed-depth only and disables MoR-related losses/curriculum).",
    )
    parser.add_argument("--recalc_lr", action="store_true", help="[DEPRECATED] LR schedule now auto-recalculates on resume. This flag is a no-op.")
    parser.add_argument("--adaptive_lr", action=argparse.BooleanOptionalAction, default=True, help="Enable adaptive LR: auto-trigger cooldown when loss spikes (default: ON)")
    parser.add_argument(
        "--adaptive_metric",
        type=str,
        default="eval",
        choices=["train", "eval"],
        help="Adaptive LR trigger metric: 'eval' (default; uses eval_loss at eval intervals) or 'train' (more reactive)",
    )
    parser.add_argument(
        "--adaptive_min_trigger_pct",
        type=float,
        default=0.50,
        help="Adaptive LR guardrail: block adaptive cooldown until this %% of the run is complete (0.0-1.0, default 0.50).",
    )
    parser.add_argument("--use_swa", action="store_true", help="Enable Stochastic Weight Averaging for better final model")
    parser.add_argument("--swa_start_pct", type=float, default=0.75, help="Start SWA at this %% of training (default 0.75)")
    parser.add_argument("--batch_size", type=int, default=None, help="Micro batch size per GPU (default: 8 for 220M, scale down for larger models)")
    parser.add_argument("--grad_accum", type=int, default=None, help="Gradient accumulation steps (effective_batch = batch_size * grad_accum)")
    parser.add_argument("--max_lr", type=float, default=None, help="Override max learning rate (default: 3e-4)")
    parser.add_argument("--min_lr", type=float, default=None, help="Override min learning rate (default: 9e-5)")
    parser.add_argument("--batch_filter", action="store_true", help="Enable batch filtering: skip batches with loss spikes from bad data")
    parser.add_argument("--batch_filter_threshold", type=float, default=2.5, help="Skip batch if loss > threshold * running_avg (default 2.5)")
    parser.add_argument("--dataset", type=str, default="pretrain_default", help="Dataset name. Options: pretrain_default (80%% FFW + 12%% TinyStories + 5%% Pleias + 3%% chat), finefineweb-sequential, finefineweb-local, pretrain_mix, pretrain_web, pretrain_chat")
    parser.add_argument(
        "--finefineweb-local",
        dest="dataset",
        action="store_const",
        const="finefineweb-local",
        help="Alias for '--dataset finefineweb-local'",
    )
    parser.add_argument("--seq_len", type=int, default=None, help="Override max sequence length (default: 2048 for production, use 1024 for 900M on 32GB)")
    parser.add_argument("--model_size", type=str, default="100M", choices=["debug", "50M", "DIAG", "100M", "debug_tall_skinny", "250M", "300M", "500M", "750M", "1B", "1.5B"], help="Model size preset (debug=tiny for fast iteration, 50M for curriculum testing)")
    parser.add_argument("--no_short_run_override", action="store_true", help="Disable SHORT RUN heuristics that delay MoD/MoR enable for runs <= 10K steps")
    parser.add_argument("--triton_kernels", action=argparse.BooleanOptionalAction, default=True, help="Enable Triton kernels (safe-by-default subset; fused RoPE/RMSNorm remain opt-in)")
    parser.add_argument("--chunked_ce", action=argparse.BooleanOptionalAction, default=True, help="Enable chunked cross-entropy (lower peak memory; may reduce throughput)")
    parser.add_argument("--chunked_ce_size", type=int, default=4096, help="Chunk size for chunked cross-entropy (tokens per chunk)")
    parser.add_argument("--8bit_adam", action="store_true", dest="use_8bit_adam", help="Use 8-bit Adam (bitsandbytes) to save ~75%% optimizer memory. Recommended for 1B+ models.")
    parser.add_argument("--adafactor", action="store_true", dest="use_adafactor", help="Use Adafactor optimizer (~25%% memory savings vs AdamW, no momentum state). Best for memory-constrained training.")
    parser.add_argument("--checkpoint_every", type=int, default=2, dest="checkpoint_every_n", help="Gradient checkpoint every N layers. 1=max memory savings, 2=balanced (default)")
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True, help="Enable gradient checkpointing to trade compute for memory (default: ON)")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True, help="Enable torch.compile for graph optimization (default: ON)")
    parser.add_argument("--halt_on_spike", action="store_true", help="Debug: stop training immediately after first gradient spike and save a checkpoint")
    parser.add_argument("--save_interval", type=int, default=500, help="Save checkpoint every N steps. 0=disable periodic saves. Default: 500.")
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=None,
        help="Override global grad clipping max-norm (e.g. 10.0). If unset, uses the model-size preset.",
    )
    parser.add_argument(
        "--grad_clip_dynamic",
        action="store_true",
        help="Enable dynamic gradient clipping (adapts to gradient norm EMA).",
    )
    parser.add_argument(
        "--grad_clip_k",
        type=float,
        default=2.0,
        help="Dynamic clip multiplier on EMA (default: 2.0 = allow 2x normal gradient).",
    )
    parser.add_argument(
        "--grad_clip_min",
        type=float,
        default=5.0,
        help="Dynamic clip floor (default: 5.0).",
    )
    parser.add_argument(
        "--grad_clip_max",
        type=float,
        default=500.0,
        help="Dynamic clip ceiling - SAFETY CAP (default: 500.0).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility. If set, training is deterministic (same seed = same results).")
    parser.add_argument("--ema_debug", action="store_true", help="Debug: print EMA updates every 25 steps to trace loss->EMA flow")
    parser.add_argument("--eval_debug", action="store_true", help="Debug: run eval sanity check on training batch at first eval")

    # Observability
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=False, help="Enable Weights & Biases logging (requires wandb)")
    parser.add_argument("--wandb_project", type=str, default=None, help="W&B project name (default: hydra-llm)")
    parser.add_argument("--wandb_entity", type=str, default=None, help="W&B entity/team (optional)")
    parser.add_argument("--run_name", type=str, default=None, help="Experiment/run name for observability backends")
    parser.add_argument("--tensorboard", action=argparse.BooleanOptionalAction, default=False, help="Enable TensorBoard scalar logging (requires tensorboard)")
    parser.add_argument("--tensorboard_dir", type=str, default=None, help="TensorBoard log dir (default: runs)")
    parser.add_argument("--profiler", action=argparse.BooleanOptionalAction, default=False, help="Enable torch.profiler traces (viewable in TensorBoard)")
    parser.add_argument("--profiler_dir", type=str, default=None, help="Profiler trace output dir (default: profiler_traces)")

    # =========================================================================
    # Mixture of Experts (MoE) - Sparse FFN Routing
    # =========================================================================
    parser.add_argument(
        "--moe",
        action="store_true",
        dest="moe_enabled",
        help="Enable Mixture of Experts (sparse FFN routing). OFF by default.",
    )
    parser.add_argument(
        "--moe_num_experts",
        type=int,
        default=0,
        help="Number of expert FFNs (0=auto-scale by model size). Default: 0.",
    )
    parser.add_argument(
        "--moe_num_layers",
        type=int,
        default=0,
        help="Number of MoE layers to insert (0=auto-scale by model size). Default: 0.",
    )
    parser.add_argument(
        "--moe_top_k",
        type=int,
        default=1,
        help="Number of experts per token (1=top-1 routing, 2=top-2). Default: 1.",
    )
    parser.add_argument(
        "--moe_aux_weight",
        type=float,
        default=0.0001,
        help="MoE load-balancing auxiliary loss weight. Default: 0.0001 (very low to allow specialization).",
    )
    parser.add_argument(
        "--moe_router_jitter",
        type=float,
        default=0.15,
        help="Router jitter noise for exploration. Default: 0.15 (helps break symmetry).",
    )
    parser.add_argument(
        "--moe_expert_diversity_noise",
        type=float,
        default=0.05,
        help="Additive weight noise to break expert symmetry. Default: 0.05.",
    )
    parser.add_argument(
        "--moe_warmup_steps",
        type=int,
        default=1000,
        help="MoE warmup steps (for checkpoint cloning). Default: 1000.",
    )
    parser.add_argument(
        "--moe_no_identity_init",
        action="store_true",
        help="Disable identity-preserving init for MoE (use normal init).",
    )
    parser.add_argument(
        "--moe_track_divergence",
        action="store_true",
        help="Track expert weight divergence during training (adds CPU overhead).",
    )
    parser.add_argument(
        "--moe_divergence_interval",
        type=int,
        default=100,
        help="Steps between divergence checks. Default: 100.",
    )
    parser.add_argument(
        "--moe_forced_routing_steps",
        type=int,
        default=0,
        help="Steps to force position-based routing for expert diversification. 0=disabled. Try 500-1000 to force experts apart.",
    )
    parser.add_argument(
        "--moe_domain_expert_map",
        type=str,
        default="",
        help="Comma-separated mapping from batch source_name to expert id. Example: 'math:0,code:1,chat:2,pleias_synth:3'.",
    )
    parser.add_argument(
        "--moe_teacher_weight",
        type=float,
        default=0.0,
        help="Alpha for domain-teacher routing loss: loss += alpha * CE(router_logits, target_expert). Default: 0 (off).",
    )
    parser.add_argument(
        "--moe_teacher_until_step",
        type=int,
        default=0,
        help="Apply teacher loss until this global step (absolute). 0=forever. Useful for short diversification warmup.",
    )
    
    # =========================================================================
    # MoE Gradient Stabilization - Per-Component LR Scaling
    # =========================================================================
    parser.add_argument(
        "--moe_expert_lr_scale",
        type=float,
        default=1.0,
        help="LR multiplier for expert/MLP weights (e.g., 0.1 to stabilize upcycled experts). Default: 1.0.",
    )
    parser.add_argument(
        "--moe_router_lr_scale",
        type=float,
        default=1.0,
        help="LR multiplier for router/gate weights (e.g., 3.0 to accelerate router learning). Default: 1.0.",
    )
    parser.add_argument(
        "--moe_lr_rewarmup_steps",
        type=int,
        default=0,
        help="Steps to linearly ramp LR from 0 to target after mid-run restart (e.g., 100). Default: 0 (disabled).",
    )
    parser.add_argument(
        "--moe_expert_weight_decay_scale",
        type=float,
        default=1.0,
        help="Weight decay multiplier for MoE experts (e.g., 10.0 to shrink blown-up expert weights). Default: 1.0.",
    )

    args = parser.parse_args()

    # Apply convenience flags (override the individual knobs)
    if args.mod_off:
        args.mod_capacity = 1.0
        args.mod_enable_mor_early_exit_threshold = 1.0  # Never triggers
        args.mod_enable_loss_threshold = 0.0
        args.mod_loss_aware_weight = 0.0
        args.aux_scale = 0.0
        # Also prevent MoD from enabling via warmup.
        args.mod_mlp_warmup_steps = 10**9

    if args.mor_off:
        args.mor_adaptive = "false"
        args.mor_enable_pct = 1.0
        args.mor_already_enabled = False
        args.ponder_scale = 0.0
        args.mor_advantage_loss_scale = 0.0

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

    # Backend compatibility: only mod_mor is implemented.
    if args.arch == "vanilla":
        print("\n⚠️  NOTE: --arch vanilla is not implemented; using mod_mor instead.")
        args.arch = "mod_mor"

    # Logic moved from main() to run_cli()
    size_config = MODEL_SIZE_CONFIGS.get(args.model_size, MODEL_SIZE_CONFIGS["100M"])
    
    config = TrainingConfig(
        architecture=args.arch,
        attention_backend=args.attention,
        mode=args.mode,
        resume_from=args.resume,
        resume_ignore_ckpt_lr=args.resume_ignore_ckpt_lr,
        resume_lr_override=args.resume_lr_override,
        mor_enable_pct=args.mor_enable_pct,
        mor_enable_min_steps=args.mor_enable_min_steps,
        mor_enable_loss_threshold=args.mor_enable_loss_threshold,
        mor_already_enabled=args.mor_already_enabled,
        mod_capacity=args.mod_capacity,
        mod_mlp_warmup_steps=args.mod_mlp_warmup_steps,
        mod_enable_mor_early_exit_threshold=args.mod_enable_mor_early_exit_threshold,
        mod_enable_loss_threshold=args.mod_enable_loss_threshold,
        mod_loss_aware_weight=args.mod_loss_aware_weight,
        mor_adaptive=(args.mor_adaptive.lower() == "true"),
        aux_scale=0.0 if args.mod_off else size_config.get("aux_scale", args.aux_scale),
        ponder_scale=args.ponder_scale,
        mor_advantage_loss_scale=args.mor_advantage_loss_scale,
        mor_min_depth=args.mor_min_depth,
        adaptive_lr=args.adaptive_lr,
        adaptive_metric=args.adaptive_metric,
        adaptive_min_trigger_pct=args.adaptive_min_trigger_pct,
        use_swa=args.use_swa,
        swa_start_pct=args.swa_start_pct,
        batch_filter=args.batch_filter,
        batch_filter_threshold=args.batch_filter_threshold,
        model_size=args.model_size,
        mod_mor_dim=size_config["mod_mor_dim"],
        n_mor_blocks=size_config["n_mor_blocks"],
        mor_recursions=size_config["mor_recursions"],
        mod_mor_n_heads=size_config["mod_mor_n_heads"],
        mod_mor_n_kv_heads=size_config["mod_mor_n_kv_heads"],
        dim=768,
        n_macro_blocks=3,
        n_heads=12,
        n_kv_heads=3,
        use_triton_kernels=args.triton_kernels,
        use_chunked_ce=args.chunked_ce,
        chunked_ce_size=args.chunked_ce_size,
        dataset_name=args.dataset,
        use_compile=args.compile,
        # Use TrainingConfig default (currently max-autotune-no-cudagraphs) for stability.
        dtype="bfloat16",
        gradient_checkpointing=args.gradient_checkpointing,
        checkpoint_every_n=args.checkpoint_every_n,
        halt_on_spike=args.halt_on_spike,
        ema_debug=getattr(args, "ema_debug", False),
        eval_debug=getattr(args, "eval_debug", False),
        use_8bit_adam=args.use_8bit_adam,
        use_adafactor=args.use_adafactor,
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
        # Use model-size-specific grad_clip if provided, otherwise use default (5.0)
        grad_clip=(float(args.grad_clip) if (args.grad_clip is not None and args.grad_clip > 0) else size_config.get("grad_clip", 5.0)),
        # Dynamic gradient clipping
        grad_clip_dynamic=getattr(args, "grad_clip_dynamic", False),
        grad_clip_k=getattr(args, "grad_clip_k", 2.0),
        grad_clip_min=getattr(args, "grad_clip_min", 5.0),
        grad_clip_max=getattr(args, "grad_clip_max", 500.0),
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
        # Per-component LR scaling for MoE gradient stabilization
        moe_expert_lr_scale=getattr(args, "moe_expert_lr_scale", 1.0),
        moe_router_lr_scale=getattr(args, "moe_router_lr_scale", 1.0),
        moe_lr_rewarmup_steps=getattr(args, "moe_lr_rewarmup_steps", 0),
        moe_expert_weight_decay_scale=getattr(args, "moe_expert_weight_decay_scale", 1.0),
    )

    # Auto-compute LR based on estimated params using μP scaling
    # LR = base_lr / sqrt(params_M / 100)
    from hydra.training.config import compute_auto_lr
    est_params = {
        "debug": 15, "50M": 50, "DIAG": 50, "100M": 220, "debug_tall_skinny": 100,
        "250M": 250, "300M": 300, "500M": 692, "750M": 750, "1B": 1000, "1.5B": 1500
    }.get(args.model_size, 220)
    auto_lr = compute_auto_lr(est_params)
    
    # Priority: CLI override > size_config > auto_lr
    if args.max_lr is not None and args.max_lr > 0:
        config.max_lr = float(args.max_lr)
        print(f"\n⚠️  OVERRIDE: max_lr={config.max_lr}")
    elif "max_lr" in size_config:
        config.max_lr = size_config["max_lr"]
        print(f"\n🔧 MODEL SIZE: max_lr={config.max_lr} (from {args.model_size} preset)")
    else:
        config.max_lr = auto_lr
        print(f"\n🔧 AUTO LR: max_lr={config.max_lr:.2e} (μP scaling for ~{est_params}M params)")
    
    # Set min_lr to 30% of max_lr by default
    if args.min_lr is not None and args.min_lr > 0:
        config.min_lr = float(args.min_lr)
        print(f"⚠️  OVERRIDE: min_lr={config.min_lr}")
    else:
        config.min_lr = config.max_lr * 0.3
        print(f"   min_lr={config.min_lr:.2e} (30% of max_lr)")

    if args.max_steps is not None:
        config.max_steps = args.max_steps
        # Rescale step-based schedule to match the selected mode's proportions.
        # Otherwise, long runs can accidentally keep the TESTING (5K) schedule.
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

        # Only override save_interval if user didn't explicitly set it (default is 500 in argparse)
        user_set_save_interval = args.save_interval != 500
        if not user_set_save_interval:
            if config.max_steps >= 2000:
                config.save_interval = 500
            else:
                config.save_interval = min(config.save_interval, max(100, config.max_steps // 5))
        print(f"\n⚠️  OVERRIDE: max_steps={config.max_steps:,}, save_interval={config.save_interval:,}")

    # Short-run heuristic: for very short diagnostic runs, delay MoR to avoid
    # curriculum triggering too early. MoD is now MoR-informed so no override needed.
    if config.max_steps is not None and config.max_steps <= 10_000 and not args.no_short_run_override:
        # Delay MoR adaptive routing as well (keep fixed-depth for most of a short run).
        # Also shrink rampup to ~10% of the run so it can't exceed remaining steps.
        if (
            args.mor_enable_pct == 0.10
            and not args.mor_already_enabled
            and args.mor_adaptive.lower() == "true"
        ):
            # For short runs, don't delay MoR - it's needed for MoD to eventually trigger
            # Just shrink rampup to fit the run
            config.mor_rampup_steps = max(100, int(round(config.max_steps * 0.10)))
            print(
                f"⚙️  SHORT RUN: setting mor_rampup_steps={config.mor_rampup_steps}. "
                "MoD will trigger when MoR early_exit > 38%. Use --no_short_run_override to disable."
            )

    if args.batch_size is None:
        config.batch_size = size_config.get("default_batch_size", 8)
    else:
        config.batch_size = args.batch_size
        print(f"\n⚠️  OVERRIDE: batch_size={config.batch_size}")

    if args.grad_accum is None:
        config.grad_accum_steps = size_config.get("default_grad_accum", 2)
    else:
        config.grad_accum_steps = args.grad_accum
        print(f"\n⚠️  OVERRIDE: grad_accum_steps={config.grad_accum_steps}")

    if args.seq_len is not None:
        config.max_seq_len = args.seq_len
        config.seq_steps = ()
        print(f"\n⚠️  OVERRIDE: max_seq_len={config.max_seq_len} (fixed, no stepping)")

    if args.recalc_lr:
        print("\n📈 NOTE: --recalc_lr is deprecated. LR schedule is now ALWAYS recalculated automatically on resume.")

    main(config)


if __name__ == "__main__":
    # Optional workaround: some environments crash during interpreter finalization
    # after using HF streaming / pyarrow. Enable only when needed.
    _hard_exit = os.environ.get("HYDRA_HARD_EXIT", "0") == "1"
    try:
        run_cli()
    except Exception:
        # Never suppress errors (hard-exit would hide tracebacks).
        raise
    else:
        if _hard_exit:
            import sys

            try:
                sys.stdout.flush()
                sys.stderr.flush()
            finally:
                os._exit(0)
