#!/usr/bin/env python3
"""
Diagnostic: Run 20–50 steps and log loss components to confirm 
eval sanity check mismatch is due to auxiliary terms.

Usage:
  source /home/tim/venvs/llm/bin/activate && cd /home/tim/Projects/LLM/HYDRA && \
  python -c "
import sys; sys.path.insert(0, '.')
from diagnostics.eval_sanity_diagnostic import run_diagnostic
run_diagnostic(resume='hydra_500m_step_20000.pt', max_steps=20)
"
"""

import json
from pathlib import Path
from typing import Optional
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from hydra.training import Trainer, TrainingConfig, MODEL_SIZE_CONFIGS


def run_diagnostic(resume: str = "hydra_500m_step_20000.pt", max_steps: int = 20):
    """Run short training with loss component logging."""
    
    print("\n" + "="*70)
    print("EVAL SANITY CHECK DIAGNOSTIC")
    print("="*70)
    print(f"Checkpoint: {resume}")
    print(f"Steps: {max_steps}")
    print("="*70 + "\n")
    
    size_config = MODEL_SIZE_CONFIGS["500M"]
    config = TrainingConfig(
        architecture="mod_mor",
        attention_backend="ccgqa",
        mode="production",
        resume_from=resume,
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
        use_8bit_adam=False,
        batch_size=4,
        grad_accum_steps=4,
        max_steps=max_steps,
        seed=13579,
        log_interval=1,
        save_interval=10000,  # No checkpointing
        eval_interval=-1,  # Skip eval (avoid sanity check)
        use_wandb=False,
        use_tensorboard=False,
    )
    
    print("Creating trainer...")
    trainer = Trainer(config)
    
    print("Running training...\n")
    print(f"{'Step':<6} {'CE':<10} {'Aux':<10} {'Ponder':<10} {'Advantage':<10} {'Total':<10} {'CE→Total Δ':<12}")
    print("-" * 90)
    
    # Patch trainer.train to capture loss components
    original_train = trainer.train
    loss_records = []
    
    def patched_train():
        """Run training and record loss components each step."""
        model = trainer.model
        base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        device = trainer.device
        dtype = trainer.dtype
        config = trainer.config
        
        # Call original train
        metrics = original_train()
        
        return metrics
    
    trainer.train = patched_train
    
    # Monkey-patch the step logging to capture loss components
    original_log_step = trainer.logger.info
    
    def capture_log_step(msg: str):
        """Intercept log messages to extract loss values."""
        original_log_step(msg)
        
        # Look for loss component logs: "CE=X.XXX aux=X.XXXX ponder=X.XXX"
        if "CE=" in msg:
            try:
                # Parse: "CE=3.6448 aux=0.0250 ponder=0.0005"
                parts = msg.split()
                ce_val = None
                aux_val = None
                ponder_val = None
                adv_val = None
                
                for part in parts:
                    if part.startswith("CE="):
                        ce_val = float(part.split("=")[1])
                    elif part.startswith("aux="):
                        aux_val = float(part.split("=")[1])
                    elif part.startswith("ponder="):
                        ponder_val = float(part.split("=")[1])
                    elif part.startswith("adv="):
                        adv_val = float(part.split("=")[1])
                
                if ce_val is not None:
                    aux_val = aux_val or 0.0
                    ponder_val = ponder_val or 0.0
                    adv_val = adv_val or 0.0
                    total = ce_val + aux_val + ponder_val + adv_val
                    delta = abs(total - ce_val)
                    
                    step = len(loss_records)
                    print(f"{step:<6} {ce_val:<10.4f} {aux_val:<10.4f} {ponder_val:<10.4f} {adv_val:<10.4f} {total:<10.4f} {delta:<12.4f}")
                    loss_records.append({
                        "step": step,
                        "ce": ce_val,
                        "aux": aux_val,
                        "ponder": ponder_val,
                        "advantage": adv_val,
                        "total": total,
                        "delta": delta,
                    })
            except Exception:
                pass  # Couldn't parse, skip
    
    trainer.logger.info = capture_log_step
    
    try:
        metrics = trainer.train()
        print("\n✅ Diagnostic completed successfully!\n")
        
        if loss_records:
            print("\nLOSS COMPONENT SUMMARY")
            print("-" * 70)
            avg_ce = sum(r["ce"] for r in loss_records) / len(loss_records)
            avg_aux = sum(r["aux"] for r in loss_records) / len(loss_records)
            avg_ponder = sum(r["ponder"] for r in loss_records) / len(loss_records)
            avg_adv = sum(r["advantage"] for r in loss_records) / len(loss_records)
            avg_delta = sum(r["delta"] for r in loss_records) / len(loss_records)
            
            print(f"Average CE loss:         {avg_ce:.4f}")
            print(f"Average Aux loss:        {avg_aux:.6f}")
            print(f"Average Ponder loss:     {avg_ponder:.6f}")
            print(f"Average Advantage loss:  {avg_adv:.6f}")
            print(f"Average CE→Total delta:  {avg_delta:.4f}")
            print("-" * 70)
            
            # Save records
            log_path = Path("diagnostics") / f"loss_diagnostic_{config.run_id}.json"
            with open(log_path, "w") as f:
                json.dump(loss_records, f, indent=2)
            print(f"\n📊 Full records saved to: {log_path}")
            
            if avg_adv < -0.1:
                print("\n⚠️  INSIGHT: Advantage loss is NEGATIVE (reduces training loss below CE).")
                print("   This explains why eval_on_train_batch (CE-only) > current_train_loss (CE + aux + ponder + advantage).")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        raise
    finally:
        trainer.close()


if __name__ == "__main__":
    run_diagnostic()
