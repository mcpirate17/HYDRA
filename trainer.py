"""HYDRA Training Entry Point.

Usage:
    python trainer.py --model_size 100M --mode testing --max_steps 1000

See `python trainer.py --help` for all options.
"""
from __future__ import annotations

import os
import sys

from hydra.training import (
    Trainer,
    TrainingConfig,
    MODEL_SIZE_CONFIGS,
    # CLI functions
    normalize_bool_flags,
    build_argument_parser,
    apply_convenience_flags,
    # Config building functions
    build_config_from_args,
    apply_lr_config,
    apply_schedule_overrides,
    apply_batch_overrides,
)


def main(config: TrainingConfig) -> None:
    """Main training entry point.

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
    """CLI entry point - parse args, build config, run training."""
    # Normalize bool flags for backward compatibility
    sys.argv = normalize_bool_flags(sys.argv)

    # Parse arguments
    parser = build_argument_parser()
    args = parser.parse_args()
    args = apply_convenience_flags(args)

    # Build config from args and size preset
    size_config = MODEL_SIZE_CONFIGS.get(args.model_size, MODEL_SIZE_CONFIGS["100M"])
    config = build_config_from_args(args, size_config)

    # Apply overrides (mutate config in-place)
    apply_lr_config(config, args, size_config)
    apply_schedule_overrides(config, args)
    apply_batch_overrides(config, args, size_config)

    # Run training
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
            try:
                sys.stdout.flush()
                sys.stderr.flush()
            finally:
                os._exit(0)
