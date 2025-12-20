"""
NVIDIA Transformer Engine Integration for HYDRA

Transformer Engine provides FP8/BF16 hybrid training for maximum throughput
on Hopper+ GPUs (H100, H200, etc.).

Benefits:
- 2x higher TFLOPS vs BF16-only
- Automatic loss scaling for FP8 stability
- Drop-in replacements for Linear, LayerNorm

Requirements:
- CUDA 12.0+
- Hopper or newer GPU (sm_90+)
- pip install transformer-engine[pytorch]

Usage:
    from hydra.kernels.te_integration import (
        TE_AVAILABLE,
        te_linear,
        te_layernorm,
        enable_fp8_training,
    )
    
    if TE_AVAILABLE:
        enable_fp8_training()  # Enable FP8 context manager

References:
    https://github.com/NVIDIA/TransformerEngine
"""

import os
from typing import Optional, Tuple
from contextlib import contextmanager

import torch
import torch.nn as nn

# Feature flags
TE_AVAILABLE = False
FP8_AVAILABLE = False
TE_VERSION = "N/A"

try:
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import Format, DelayedScaling
    TE_AVAILABLE = True
    TE_VERSION = getattr(te, "__version__", "unknown")
    
    # Check if FP8 is supported on this GPU
    if torch.cuda.is_available():
        cc = torch.cuda.get_device_capability()
        FP8_AVAILABLE = cc[0] >= 9  # Hopper (sm_90) or newer
except ImportError:
    te = None
    Format = None
    DelayedScaling = None


def get_te_status() -> dict:
    """Get Transformer Engine status."""
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "gpu_name": torch.cuda.get_device_name(),
            "compute_capability": torch.cuda.get_device_capability(),
        }
    
    return {
        "te_available": TE_AVAILABLE,
        "te_version": TE_VERSION,
        "fp8_available": FP8_AVAILABLE,
        "reason": "Requires Hopper+ GPU (sm_90)" if TE_AVAILABLE and not FP8_AVAILABLE else None,
        **gpu_info,
    }


# FP8 Recipe Configuration
_FP8_RECIPE = None

def get_fp8_recipe():
    """Get the FP8 training recipe (delayed scaling for stability)."""
    global _FP8_RECIPE
    if _FP8_RECIPE is None and TE_AVAILABLE:
        _FP8_RECIPE = DelayedScaling(
            margin=0,
            fp8_format=Format.HYBRID,  # E4M3 forward, E5M2 backward
            amax_history_len=16,
            amax_compute_algo="most_recent",
        )
    return _FP8_RECIPE


@contextmanager
def fp8_autocast(enabled: bool = True):
    """Context manager for FP8 training.
    
    Usage:
        with fp8_autocast():
            output = model(input)
            loss = criterion(output, target)
    """
    if not TE_AVAILABLE or not FP8_AVAILABLE or not enabled:
        yield
        return
    
    recipe = get_fp8_recipe()
    with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
        yield


class TELinear(nn.Module):
    """Transformer Engine Linear layer with FP8 support.
    
    Drop-in replacement for nn.Linear with automatic FP8 compute.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        if TE_AVAILABLE:
            self.linear = te.Linear(
                in_features,
                out_features,
                bias=bias,
                device=device,
                params_dtype=dtype or torch.bfloat16,
            )
        else:
            self.linear = nn.Linear(in_features, out_features, bias=bias, device=device, dtype=dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class TELayerNorm(nn.Module):
    """Transformer Engine LayerNorm with fused operations."""
    
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        if TE_AVAILABLE:
            self.norm = te.LayerNorm(hidden_size, eps=eps)
        else:
            self.norm = nn.LayerNorm(hidden_size, eps=eps)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class TERMSNorm(nn.Module):
    """Transformer Engine RMSNorm with fused operations."""
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        if TE_AVAILABLE:
            self.norm = te.RMSNorm(hidden_size, eps=eps)
        else:
            # Fallback to PyTorch
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(hidden_size))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if TE_AVAILABLE:
            return self.norm(x)
        # PyTorch fallback
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class TETransformerLayer(nn.Module):
    """Full Transformer layer using Transformer Engine.
    
    This fuses:
    - LayerNorm
    - QKV projection
    - Attention
    - Output projection
    - MLP
    
    Into highly optimized FP8-capable kernels.
    """
    
    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
        num_attention_heads: int,
        num_gqa_groups: Optional[int] = None,
        bias: bool = False,
        layernorm_epsilon: float = 1e-5,
        attention_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
    ):
        super().__init__()
        
        if not TE_AVAILABLE:
            raise RuntimeError("Transformer Engine not available. Install with: pip install transformer-engine[pytorch]")
        
        self.layer = te.TransformerLayer(
            hidden_size=hidden_size,
            ffn_hidden_size=ffn_hidden_size,
            num_attention_heads=num_attention_heads,
            num_gqa_groups=num_gqa_groups,
            bias=bias,
            layernorm_epsilon=layernorm_epsilon,
            attention_dropout=attention_dropout,
            hidden_dropout=hidden_dropout,
            fuse_qkv_params=True,
            set_parallel_mode=False,
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.layer(hidden_states, attention_mask=attention_mask)


def patch_model_with_te(model: nn.Module, fp8_layers: bool = True) -> nn.Module:
    """Replace standard layers with Transformer Engine equivalents.
    
    Args:
        model: Model to patch
        fp8_layers: Whether to enable FP8 for Linear layers
    
    Returns:
        Patched model (modifies in-place and returns)
    """
    if not TE_AVAILABLE:
        print("⚠️  Transformer Engine not available")
        return model
    
    # Count replacements
    replaced = {"linear": 0, "layernorm": 0, "rmsnorm": 0}
    
    for name, module in model.named_modules():
        # Replace Linear layers
        if isinstance(module, nn.Linear) and fp8_layers:
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            parent = model.get_submodule(parent_name) if parent_name else model
            
            te_linear = te.Linear(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                params_dtype=module.weight.dtype,
            )
            # Copy weights
            te_linear.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                te_linear.bias.data.copy_(module.bias.data)
            
            setattr(parent, child_name, te_linear)
            replaced["linear"] += 1
    
    print(f"✅ Patched model with Transformer Engine:")
    print(f"   - Linear layers: {replaced['linear']}")
    return model


# Convenience function to check and report status
def print_te_status():
    """Print Transformer Engine availability and configuration."""
    status = get_te_status()
    print("\n" + "=" * 50)
    print("TRANSFORMER ENGINE STATUS")
    print("=" * 50)
    print(f"  Available: {status['te_available']}")
    print(f"  Version: {status['te_version']}")
    print(f"  FP8 Support: {status['fp8_available']}")
    if status.get('gpu_name'):
        print(f"  GPU: {status['gpu_name']}")
        print(f"  Compute Cap: {status['compute_capability']}")
    if status.get('reason'):
        print(f"  Note: {status['reason']}")
    print("=" * 50 + "\n")
