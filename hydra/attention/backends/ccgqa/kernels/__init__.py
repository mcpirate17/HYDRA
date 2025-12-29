"""
Triton kernels for CCGQA attention.

Phase 1: Fused attention kernel to replace F.scaled_dot_product_attention
"""

from .fused_attention import ccgqa_attention_fused

__all__ = ["ccgqa_attention_fused"]
