"""
Liger Kernel Integration for HYDRA

Liger provides highly optimized fused BF16 kernels that can:
- Reduce memory usage by 50-60%
- Speed up training by ~20%
- Drop-in replacements for RMSNorm, SwiGLU, CrossEntropy, RoPE

Installation:
    pip install liger-kernel

Usage:
    from hydra.kernels.liger_integration import patch_hydra_with_liger
    patch_hydra_with_liger()  # Call before model creation

References:
    https://github.com/linkedin/Liger-Kernel
"""

import os
from typing import Optional

# Feature flag for Liger
LIGER_AVAILABLE = False
LIGER_ENABLED = os.environ.get("HYDRA_USE_LIGER", "1") == "1"

try:
    import liger_kernel
    # Liger 0.5+ uses liger_kernel.transformers API
    from liger_kernel.transformers import (
        LigerRMSNorm,
        LigerSwiGLUMLP,
        LigerCrossEntropyLoss,
        LigerFusedLinearCrossEntropyLoss,
    )
    # Try to get the low-level SiLU function for manual use
    try:
        from liger_kernel.ops.swiglu import LigerSiLUMulFunction
    except ImportError:
        LigerSiLUMulFunction = None
    # Try to get RoPE
    try:
        from liger_kernel.ops.rope import liger_rotary_pos_emb
    except ImportError:
        liger_rotary_pos_emb = None
    LIGER_AVAILABLE = True
except ImportError:
    LigerRMSNorm = None
    LigerSwiGLUMLP = None
    LigerSiLUMulFunction = None
    LigerCrossEntropyLoss = None
    liger_rotary_pos_emb = None
    LigerFusedLinearCrossEntropyLoss = None


def get_liger_status() -> dict:
    """Get Liger kernel status."""
    return {
        "liger_available": LIGER_AVAILABLE,
        "liger_enabled": LIGER_ENABLED and LIGER_AVAILABLE,
        "version": getattr(liger_kernel, "__version__", "unknown") if LIGER_AVAILABLE else "N/A",
    }


def liger_rms_norm(dim: int, eps: float = 1e-6):
    """Get Liger RMSNorm if available, else return None.
    
    Liger RMSNorm fuses the normalization into a single kernel,
    reducing memory bandwidth and kernel launch overhead.
    
    Memory savings: ~30% for normalization layers
    Speed improvement: ~1.5-2x
    """
    if LIGER_AVAILABLE and LIGER_ENABLED and LigerRMSNorm is not None:
        return LigerRMSNorm(dim, eps=eps)
    return None


def liger_swiglu_forward(gate, up):
    """Liger fused SiLU * up operation.
    
    Fuses SiLU(gate) * up into a single kernel.
    Memory savings: Avoids materializing intermediate SiLU output.
    """
    if LIGER_AVAILABLE and LIGER_ENABLED and LigerSiLUMulFunction is not None:
        return LigerSiLUMulFunction.apply(gate, up)
    return None


def liger_cross_entropy_loss(ignore_index: int = -100, reduction: str = "mean"):
    """Get Liger CrossEntropy loss.
    
    Liger's cross-entropy fuses softmax + log + nll into one kernel,
    avoiding the massive logits tensor materialization.
    
    Memory savings: ~60% for loss computation
    Speed improvement: ~2x
    """
    if LIGER_AVAILABLE and LIGER_ENABLED and LigerCrossEntropyLoss is not None:
        return LigerCrossEntropyLoss(ignore_index=ignore_index, reduction=reduction)
    return None


def liger_fused_linear_cross_entropy(ignore_index: int = -100, reduction: str = "mean"):
    """Get Liger Fused Linear + CrossEntropy.
    
    THE ULTIMATE MEMORY OPTIMIZATION: Fuses:
    1. Linear projection (hidden -> vocab)
    2. Softmax
    3. Cross-entropy loss
    
    Never materializes the full [batch, seq, vocab_size] logits tensor!
    
    Memory savings: ~80% for the output layer
    Speed improvement: ~2-3x
    
    Usage:
        loss_fn = liger_fused_linear_cross_entropy()
        loss = loss_fn(hidden_states, lm_head.weight, targets)
    """
    if LIGER_AVAILABLE and LIGER_ENABLED and LigerFusedLinearCrossEntropyLoss is not None:
        return LigerFusedLinearCrossEntropyLoss(ignore_index=ignore_index, reduction=reduction)
    return None


def liger_rope(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Apply Liger RoPE.
    
    Liger's RoPE is fused and optimized for memory efficiency.
    """
    if LIGER_AVAILABLE and LIGER_ENABLED and liger_rotary_pos_emb is not None:
        q_embed, k_embed = liger_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim)
        return q_embed, k_embed
    return None, None


def patch_hydra_with_liger():
    """Monkey-patch HYDRA modules to use Liger kernels where possible.
    
    Call this BEFORE creating your model.
    
    This patches:
    - RMSNorm -> LigerRMSNorm
    - SwiGLU activation -> LigerSiLUMulFunction
    """
    if not LIGER_AVAILABLE or not LIGER_ENABLED:
        print("⚠️  Liger kernels not available. Install with: pip install liger-kernel")
        return False
    
    import hydra.layers.common as common_module
    
    # Patch RMSNorm
    original_rmsnorm = common_module.RMSNorm
    common_module.RMSNorm = LigerRMSNorm
    common_module._ORIGINAL_RMSNORM = original_rmsnorm  # Keep reference
    
    # Patch SwiGLU in fused MLP (only if LigerSiLUMulFunction is available)
    if LigerSiLUMulFunction is not None:
        original_swiglu_fused = common_module.SwiGLUMLPFused
        
        class LigerSwiGLUMLPFused(original_swiglu_fused):
            """SwiGLU MLP using Liger's fused kernel."""
            def forward(self, x):
                gate_up = self.gate_up(x)
                gate, up = gate_up.chunk(2, dim=-1)
                return self.down(LigerSiLUMulFunction.apply(gate, up))
        
        common_module.SwiGLUMLPFused = LigerSwiGLUMLPFused
        common_module.SwiGLUMLP = LigerSwiGLUMLPFused
        common_module._ORIGINAL_SWIGLU = original_swiglu_fused
        swiglu_patched = True
    else:
        swiglu_patched = False
    
    print("✅ HYDRA patched with Liger kernels:")
    print("   - RMSNorm -> LigerRMSNorm (fused, ~30% memory savings)")
    if swiglu_patched:
        print("   - SwiGLU -> LigerSiLUMulFunction (fused activation)")
    print("   - Use liger_cross_entropy_loss() for ~60% CE memory savings")
    print("   - Use liger_fused_linear_cross_entropy() for ~80% output layer savings")
    return True


def apply_liger_kernel_to_model(model, use_fused_ce: bool = True):
    """Apply Liger optimizations to an existing model.
    
    This replaces modules in-place after model creation.
    More targeted than patch_hydra_with_liger().
    
    Args:
        model: The model to optimize
        use_fused_ce: Whether to return a fused cross-entropy loss function
        
    Returns:
        Tuple of (model, loss_fn) where loss_fn is Liger's fused CE if available
    """
    if not LIGER_AVAILABLE or not LIGER_ENABLED:
        return model, None
    
    import torch.nn as nn
    
    # Replace RMSNorm modules
    replaced_count = 0
    for name, module in model.named_modules():
        if module.__class__.__name__ == 'RMSNorm' and not isinstance(module, LigerRMSNorm):
            # Get parent module
            parts = name.rsplit('.', 1)
            if len(parts) == 2:
                parent_name, child_name = parts
                parent = model.get_submodule(parent_name)
            else:
                parent = model
                child_name = name
            
            # Create Liger replacement
            new_norm = LigerRMSNorm(module.weight.shape[0], eps=module.eps)
            new_norm.weight.data.copy_(module.weight.data)
            setattr(parent, child_name, new_norm)
            replaced_count += 1
    
    if replaced_count > 0:
        print(f"✅ Replaced {replaced_count} RMSNorm layers with LigerRMSNorm")
    
    # Return fused CE loss if requested
    loss_fn = None
    if use_fused_ce and LigerFusedLinearCrossEntropyLoss is not None:
        loss_fn = LigerFusedLinearCrossEntropyLoss(ignore_index=-100)
        print("✅ Using Liger Fused Linear Cross-Entropy (80% memory savings on output)")
    elif use_fused_ce and LigerCrossEntropyLoss is not None:
        loss_fn = LigerCrossEntropyLoss(ignore_index=-100)
        print("✅ Using Liger Cross-Entropy (60% memory savings)")
    
    return model, loss_fn


# Auto-patch on import if HYDRA_AUTO_LIGER=1
if os.environ.get("HYDRA_AUTO_LIGER", "0") == "1":
    patch_hydra_with_liger()
