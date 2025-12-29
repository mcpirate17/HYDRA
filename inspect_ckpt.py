import torch
import sys
import os

def inspect_checkpoint(ckpt_path):
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        return

    print(f"Inspecting checkpoint: {ckpt_path}")
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        
        print("\n--- Metadata ---")
        if "config" in ckpt:
            print("Config found in checkpoint.")
            # Print relevant config keys
            conf = ckpt["config"]
            keys_to_check = ["model_size", "mod_mor_dim", "n_mor_blocks", "mod_mor_n_heads", "dataset_name", "max_seq_len"]
            for k in keys_to_check:
                if hasattr(conf, k):
                    print(f"  {k}: {getattr(conf, k)}")
                elif isinstance(conf, dict) and k in conf:
                    print(f"  {k}: {conf[k]}")
        else:
            print("No 'config' key found in checkpoint.")

        print("\n--- Training State ---")
        if "step" in ckpt:
            print(f"  Step: {ckpt['step']}")
        if "epoch" in ckpt:
            print(f"  Epoch: {ckpt['epoch']}")
        
        print("\n--- Model State Keys (Sample) ---")
        state_dict = ckpt.get("model_state_dict", ckpt.get("model", {}))
        keys = list(state_dict.keys())
        print(f"  Total keys: {len(keys)}")
        print("  First 10 keys:")
        for k in keys[:10]:
            print(f"    {k}")
            
        # Check for MoD/MoR specific keys
        has_mod = any("router" in k and "mod" in k for k in keys)
        has_mor = any("router" in k and "mor" in k for k in keys)
        print(f"\n  Has MoD keys: {has_mod}")
        print(f"  Has MoR keys: {has_mor}")

    except Exception as e:
        print(f"Error loading checkpoint: {e}")

if __name__ == "__main__":
    inspect_checkpoint("checkpoints/hydra_100m_step_5500.pt")
