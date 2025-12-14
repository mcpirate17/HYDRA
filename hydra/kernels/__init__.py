"""
HYDRA Custom Triton Kernels

Fused GPU kernels for maximum performance on attention operations.
"""

from .fused_ops import (
    fused_rope,
    fused_qk_norm,
    fused_swiglu,
    fused_rms_norm,
)

__all__ = [
    "fused_rope",
    "fused_qk_norm", 
    "fused_swiglu",
    "fused_rms_norm",
]
