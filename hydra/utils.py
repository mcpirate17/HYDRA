"""
Utility functions for HYDRA models.

General purpose utilities that don't fit into specific modules.
"""

import os
import torch.nn as nn


def save_model_architecture(model: nn.Module, save_path: str) -> None:
    """Save the model architecture code to a file for verification.
    
    Args:
        model: PyTorch model to save architecture for
        save_path: Path to save the architecture text file
    """
    os.makedirs(
        os.path.dirname(save_path) if os.path.dirname(save_path) else ".",
        exist_ok=True
    )
    
    with open(save_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("Model Architecture\n")
        f.write("=" * 80 + "\n\n")
        f.write(str(model) + "\n\n")
        f.write("=" * 80 + "\n")
        f.write("Parameter Summary\n")
        f.write("=" * 80 + "\n\n")
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        f.write(f"Total parameters: {total_params:,}\n")
        f.write(f"Trainable parameters: {trainable:,}\n")
        f.write(f"Non-trainable parameters: {total_params - trainable:,}\n")
    
    print(f"Model architecture saved to: {save_path}")
