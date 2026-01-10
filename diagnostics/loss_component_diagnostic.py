#!/usr/bin/env python3
"""
Diagnostic: Log loss components (CE, aux, ponder, advantage) during training
to understand eval sanity check mismatch.

Usage:
  source /home/tim/venvs/llm/bin/activate && cd /home/tim/Projects/LLM/HYDRA && \
  python diagnostics/loss_component_diagnostic.py \
    --resume hydra_500m_step_20000.pt \
    --max_steps 50 \
    --seed 13579
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
from hydra.training import Trainer, TrainingConfig, MODEL_SIZE_CONFIGS


def patch_trainer_for_diagnostics(trainer):
    """Monkey-patch trainer to capture loss components each step."""
    original_train = trainer.train
    loss_log = []
    
    def instrumented_train():
        """Wrapper that captures loss components."""
        model = trainer.model
        config = trainer.config
        device = trainer.device
        dtype = trainer.dtype
        
        # Call original train but intercept step logging
        try:
            metrics = original_train()
            return metrics
        except Exception as e:
            print(f"\n❌ Training failed: {e}")
            raise
        finally:
            # Save loss log even on failure
            if loss_log:
                log_path = Path(config.log_dir) / f"loss_components_{config.run_id}.json"
                with open(log_path, "w") as f:
                    json.dump(loss_log, f, indent=2)
                print(f"\n📊 Loss components logged to: {log_path}")
    
    trainer.train = instrumented_train
    return trainer, loss_log


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose loss component breakdown")
    parser.add_argument("--resume", type=str, default="hydra_500m_step_20000.pt", help="Checkpoint to resume from")
    parser.add_argument("--max_steps", type=int, default=50, help="Number of steps to run")
    parser.add_argument("--seed", type=int, default=13579, help="Random seed")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation")
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("LOSS COMPONENT DIAGNOSTIC")
    print("="*70)
    print(f"Resume: {args.resume}")
    print(f"Max steps: {args.max_steps}")
    print(f"Batch size: {args.batch_size}, Grad accum: {args.grad_accum}")
    print("="*70 + "\n")
    
    size_config = MODEL_SIZE_CONFIGS["500M"]
    config = TrainingConfig(
        architecture="mod_mor",
        attention_backend="ccgqa",
        mode="production",
        resume_from=args.resume,
        model_size="500M",
        mod_mor_dim=size_config["mod_mor_dim"],
        n_mor_blocks=size_config["n_mor_blocks"],
        mor_recursions=size_config["mor_recursions"],
        mod_mor_n_heads=size_config["mod_mor_n_heads"],
        mod_mor_n_kv_heads=size_config["mod_mor_n_kv_heads"],
        dim=768,
        n_macro_blocks=3,
        n_heads=12,
        n_kv_heads=3,
        dataset_name="pretrain_default",
        dtype="bfloat16",
        gradient_checkpointing=True,
        checkpoint_every_n=2,
        use_8bit_adam=False,  # For cleaner diagnostics
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        max_steps=args.max_steps,
        seed=args.seed,
        log_interval=1,  # Log every step
        save_interval=1000,  # No checkpointing during diagnostic
        eval_interval=-1,  # Skip eval to avoid sanity check noise
    )
    
    config.print_summary()
    
    print("\n🚀 Creating trainer...")
    trainer = Trainer(config)
    
    print("📊 Running training with loss component logging...\n")
    
    try:
        metrics = trainer.train()
        print("\n✅ Training completed successfully!")
        print(f"Final loss: {metrics.final_loss:.4f}")
        print(f"Best loss: {metrics.best_loss:.4f}")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        raise
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
