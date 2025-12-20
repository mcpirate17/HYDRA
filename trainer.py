from __future__ import annotations

import argparse
import sys

from hydra.training import Trainer, TrainingConfig, MODEL_SIZE_CONFIGS


def main(
    architecture: str = "mod_mor",
    mode: str = "testing",
    resume_from: str | None = None,
    max_steps_override: int | None = None,
    mor_enable_pct: float = 0.30,
    mor_already_enabled: bool = False,
    mod_capacity: float = 0.5,
    mor_adaptive: bool = True,
    aux_scale: float = 0.1,
    ponder_scale: float = 0.01,
    recalc_lr_schedule: bool = False,
    adaptive_lr: bool = True,
    use_swa: bool = False,
    swa_start_pct: float = 0.75,
    batch_filter: bool = False,
    batch_filter_threshold: float = 2.5,
    batch_size: int | None = None,
    grad_accum_steps: int | None = None,
    dataset_name: str = "finefineweb",
    model_size: str = "100M",
    gradient_checkpointing: bool = True,
    checkpoint_every_n: int = 2,
    use_triton_kernels: bool = True,
    use_chunked_ce: bool = True,
    chunked_ce_size: int = 4096,
    halt_on_spike: bool = False,
    use_8bit_adam: bool = False,
    seq_len_override: int | None = None,
    seed: int | None = None,
    use_compile: bool = True,
) -> None:
    size_config = MODEL_SIZE_CONFIGS.get(model_size, MODEL_SIZE_CONFIGS["100M"])
    print(f"\n🔧 MODEL SIZE: {model_size}")
    print(
        f"   dim={size_config['mod_mor_dim']}, blocks={size_config['n_mor_blocks']}, "
        f"heads={size_config['mod_mor_n_heads']}"
    )

    config = TrainingConfig(
        architecture=architecture,
        mode=mode,
        resume_from=resume_from,
        mor_enable_pct=mor_enable_pct,
        mor_already_enabled=mor_already_enabled,
        mod_capacity=mod_capacity,
        mor_adaptive=mor_adaptive,
        aux_scale=aux_scale,
        ponder_scale=ponder_scale,
        adaptive_lr=adaptive_lr,
        use_swa=use_swa,
        swa_start_pct=swa_start_pct,
        batch_filter=batch_filter,
        batch_filter_threshold=batch_filter_threshold,
        model_size=model_size,
        mod_mor_dim=size_config["mod_mor_dim"],
        n_mor_blocks=size_config["n_mor_blocks"],
        mor_recursions=size_config["mor_recursions"],
        mod_mor_n_heads=size_config["mod_mor_n_heads"],
        mod_mor_n_kv_heads=size_config["mod_mor_n_kv_heads"],
        dim=768,
        n_macro_blocks=3,
        n_heads=12,
        n_kv_heads=3,
        use_triton_kernels=use_triton_kernels,
        use_chunked_ce=use_chunked_ce,
        chunked_ce_size=chunked_ce_size,
        dataset_name=dataset_name,
        use_compile=use_compile,
        compile_mode="max-autotune-no-cudagraphs",
        dtype="bfloat16",
        gradient_checkpointing=gradient_checkpointing,
        checkpoint_every_n=checkpoint_every_n,
        halt_on_spike=halt_on_spike,
        use_8bit_adam=use_8bit_adam,
        log_interval=25,
        save_interval=500,
        seed=seed,
    )

    if max_steps_override is not None:
        config.max_steps = max_steps_override
        if config.max_steps >= 2000:
            config.save_interval = 500
        else:
            config.save_interval = min(config.save_interval, max(100, config.max_steps // 5))
        print(f"\n⚠️  OVERRIDE: max_steps={config.max_steps:,}, save_interval={config.save_interval:,}")

    if batch_size is None:
        config.batch_size = size_config.get("default_batch_size", 8)
    else:
        config.batch_size = batch_size
        print(f"\n⚠️  OVERRIDE: batch_size={config.batch_size}")

    if grad_accum_steps is None:
        config.grad_accum_steps = size_config.get("default_grad_accum", 2)
    else:
        config.grad_accum_steps = grad_accum_steps
        print(f"\n⚠️  OVERRIDE: grad_accum_steps={config.grad_accum_steps}")

    if seq_len_override is not None:
        config.max_seq_len = seq_len_override
        config.seq_steps = ()
        print(f"\n⚠️  OVERRIDE: max_seq_len={config.max_seq_len} (fixed, no stepping)")

    if recalc_lr_schedule:
        print("\n📈 NOTE: --recalc_lr is deprecated. LR schedule is now ALWAYS recalculated automatically on resume.")

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
    parser = argparse.ArgumentParser(description="HYDRA Training")
    parser.add_argument("--arch", type=str, default="mod_mor", choices=["vanilla", "mod_mor"], help="Architecture: vanilla or mod_mor")
    parser.add_argument("--mode", type=str, default="testing", choices=["testing", "production", "chinchilla_third"], help="Mode: testing (5K), production (100K), chinchilla_third (1/3 Chinchilla)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--max_steps", type=int, default=None, help="Override max steps (for diagnostic runs, e.g., 500 or 1000)")
    parser.add_argument("--mor_enable_pct", type=float, default=0.30, help="MoR curriculum: enable adaptive routing after this %% of training (0.0-1.0, default 0.30)")
    parser.add_argument("--mor_already_enabled", action="store_true", help="Restart flag: set if resuming AFTER MoR was already enabled")
    parser.add_argument("--mod_capacity", type=float, default=0.5, help="MoD capacity ratio (0.5=50%% tokens, 1.0=all tokens=MoD OFF)")
    parser.add_argument("--mor_adaptive", type=str, default="true", choices=["true", "false"], help="MoR adaptive routing (true=on, false=fixed-depth only)")
    parser.add_argument("--aux_scale", type=float, default=0.1, help="MoD aux loss scale (0.1 default, 0.0=MoD loss OFF)")
    parser.add_argument("--ponder_scale", type=float, default=0.01, help="MoR ponder loss scale (0.01 default, 1e-4=weak reg)")
    parser.add_argument("--recalc_lr", action="store_true", help="[DEPRECATED] LR schedule now auto-recalculates on resume. This flag is a no-op.")
    parser.add_argument("--adaptive_lr", action=argparse.BooleanOptionalAction, default=True, help="Enable adaptive LR: auto-trigger cooldown when loss spikes (default: ON)")
    parser.add_argument("--use_swa", action="store_true", help="Enable Stochastic Weight Averaging for better final model")
    parser.add_argument("--swa_start_pct", type=float, default=0.75, help="Start SWA at this %% of training (default 0.75)")
    parser.add_argument("--batch_size", type=int, default=None, help="Micro batch size per GPU (default: 8 for 220M, scale down for larger models)")
    parser.add_argument("--grad_accum", type=int, default=None, help="Gradient accumulation steps (effective_batch = batch_size * grad_accum)")
    parser.add_argument("--batch_filter", action="store_true", help="Enable batch filtering: skip batches with loss spikes from bad data")
    parser.add_argument("--batch_filter_threshold", type=float, default=2.5, help="Skip batch if loss > threshold * running_avg (default 2.5)")
    parser.add_argument("--dataset", type=str, default="finefineweb-sequential", help="Dataset name. Options: finefineweb-sequential (default), finefineweb-local, pretrain_mix, pretrain_web, pretrain_chat")
    parser.add_argument("--seq_len", type=int, default=None, help="Override max sequence length (default: 2048 for production, use 1024 for 900M on 32GB)")
    parser.add_argument("--model_size", type=str, default="100M", choices=["DIAG", "100M", "250M", "300M", "500M", "750M", "1B", "1.5B"], help="Model size preset")
    parser.add_argument("--triton_kernels", action=argparse.BooleanOptionalAction, default=True, help="Enable Triton kernels (safe-by-default subset; fused RoPE/RMSNorm remain opt-in)")
    parser.add_argument("--chunked_ce", action=argparse.BooleanOptionalAction, default=True, help="Enable chunked cross-entropy (lower peak memory; may reduce throughput)")
    parser.add_argument("--chunked_ce_size", type=int, default=4096, help="Chunk size for chunked cross-entropy (tokens per chunk)")
    parser.add_argument("--8bit_adam", action="store_true", dest="use_8bit_adam", help="Use 8-bit Adam (bitsandbytes) to save ~75%% optimizer memory. Recommended for 1B+ models.")
    parser.add_argument("--checkpoint_every", type=int, default=2, dest="checkpoint_every_n", help="Gradient checkpoint every N layers. 1=max memory savings, 2=balanced (default)")
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True, help="Enable gradient checkpointing to trade compute for memory (default: ON)")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True, help="Enable torch.compile for graph optimization (default: ON)")
    parser.add_argument("--halt_on_spike", action="store_true", help="Debug: stop training immediately after first gradient spike and save a checkpoint")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility. If set, training is deterministic (same seed = same results).")
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
        seq_len_override=args.seq_len,
        use_triton_kernels=args.triton_kernels,
        use_chunked_ce=args.chunked_ce,
        chunked_ce_size=args.chunked_ce_size,
        halt_on_spike=args.halt_on_spike,
        use_8bit_adam=args.use_8bit_adam,
        gradient_checkpointing=args.gradient_checkpointing,
        checkpoint_every_n=args.checkpoint_every_n,
        seed=args.seed,
        use_compile=args.compile,
    )


if __name__ == "__main__":
    run_cli()
