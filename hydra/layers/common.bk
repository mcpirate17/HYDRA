"""
HYDRA Common Layers

Shared components:
- RMSNorm: Root Mean Square Layer Normalization
- SwiGLUMLP: Gated Linear Unit MLP
- RotaryEmbedding: Shared RoPE cache (reduces memory 24x for 24-layer models)
- Flexible attention backend: Flash Attention 2, xFormers, or PyTorch SDPA

Usage:
    from hydra.layers import RMSNorm, SwiGLUMLP, RotaryEmbedding
    
    # Shared RoPE across all attention layers
    rope = RotaryEmbedding(head_dim=64, max_seq_len=8192)
    
    # In attention:
    q = rope(q, seq_len)
    k = rope(k, seq_len)
"""

import math
from typing import Optional, Tuple, Literal
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# External Library Detection
# =============================================================================

# Fused Triton kernels
try:
    from hydra.kernels import (
        fused_rope,
        fused_qk_norm,
        fused_swiglu,
        fused_rms_norm,
        TRITON_AVAILABLE,
    )
    FUSED_KERNELS_AVAILABLE = TRITON_AVAILABLE
except ImportError:
    FUSED_KERNELS_AVAILABLE = False
    fused_rope = None
    fused_qk_norm = None
    fused_swiglu = None
    fused_rms_norm = None

# Flash Attention 2
try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import unpad_input, pad_input
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    flash_attn_func = None

# xFormers memory-efficient attention
try:
    import xformers.ops as xops
    XFORMERS_AVAILABLE = True
except ImportError:
    XFORMERS_AVAILABLE = False
    xops = None


# =============================================================================
# Configuration
# =============================================================================

# Global setting for attention backend
# Options: "auto", "flash", "xformers", "sdpa", "naive"
_ATTENTION_BACKEND: str = "auto"


def set_attention_backend(backend: Literal["auto", "flash", "xformers", "sdpa", "naive"]):
    """Set the global attention backend.
    
    Args:
        backend: One of:
            - "auto": Automatically select best available (flash > xformers > sdpa)
            - "flash": Force Flash Attention 2 (raises if unavailable)
            - "xformers": Force xFormers (raises if unavailable)
            - "sdpa": Force PyTorch scaled_dot_product_attention
            - "naive": Force naive O(n²) attention (for debugging)
    """
    global _ATTENTION_BACKEND
    
    if backend == "flash" and not FLASH_ATTN_AVAILABLE:
        raise ImportError("Flash Attention 2 not available. Install: pip install flash-attn --no-build-isolation")
    if backend == "xformers" and not XFORMERS_AVAILABLE:
        raise ImportError("xFormers not available. Install: pip install xformers")
    
    _ATTENTION_BACKEND = backend


def get_attention_backend() -> str:
    """Get the effective attention backend."""
    if _ATTENTION_BACKEND == "auto":
        if FLASH_ATTN_AVAILABLE:
            return "flash"
        elif XFORMERS_AVAILABLE:
            return "xformers"
        else:
            return "sdpa"
    return _ATTENTION_BACKEND


# =============================================================================
# RMSNorm
# =============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.
    
    More efficient than LayerNorm as it skips mean centering.
    Supports fused Triton kernel for additional speedup.
    
    Args:
        dim: Feature dimension
        eps: Epsilon for numerical stability
        elementwise_affine: Whether to learn per-element scale (default True)
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        elementwise_affine: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use fused kernel if available and we have learnable weights
        if FUSED_KERNELS_AVAILABLE and self.weight is not None:
            return fused_rms_norm(x, self.weight, self.eps)
        
        # PyTorch implementation
        dtype = x.dtype
        x = x.float()
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = x * rms
        
        if self.weight is not None:
            # Cast weight to input dtype for mixed precision
            x = x.to(dtype) * self.weight.to(dtype)
        else:
            x = x.to(dtype)
        
        return x

    def extra_repr(self) -> str:
        return f"{self.dim}, eps={self.eps}, elementwise_affine={self.elementwise_affine}"


# =============================================================================
# SwiGLU MLP
# =============================================================================

class SwiGLUMLP(nn.Module):
    """SwiGLU MLP with fused gate/up projection.
    
    SwiGLU: out = (W_gate @ x * silu(W_gate @ x)) @ W_down
    
    Fuses W_gate and W_up into single projection for efficiency.
    
    Args:
        dim: Input/output dimension
        hidden_dim: Hidden dimension (typically 2.67x-4x of dim)
        dropout: Dropout rate (default 0.0)
        bias: Whether to use bias in linear layers
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        
        # Fused gate + up projection
        self.gate_up = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        self.down = nn.Linear(hidden_dim, dim, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up(x)
        gate, up = gate_up.chunk(2, dim=-1)
        
        # Use fused kernel if available
        if FUSED_KERNELS_AVAILABLE:
            hidden = fused_swiglu(gate, up)
        else:
            hidden = F.silu(gate) * up
        
        return self.dropout(self.down(hidden))


def get_activation(name: str) -> nn.Module:
    """Get activation function by name."""
    activations = {
        "gelu": nn.GELU(),
        "relu": nn.ReLU(),
        "silu": nn.SiLU(),
        "swish": nn.SiLU(),  # Swish = SiLU
        "tanh": nn.Tanh(),
    }
    if name not in activations:
        raise ValueError(f"Unknown activation: {name}. Choose from {list(activations.keys())}")
    return activations[name]


# =============================================================================
# Rotary Position Embeddings (Shared)
# =============================================================================

class RotaryEmbedding(nn.Module):
    """Rotary Position Embeddings with shared cache.
    
    Creates RoPE embeddings once and reuses across all attention layers.
    For a 24-layer model, this reduces memory usage by 24x.
    
    Supports:
    - Standard RoPE (GPT-NeoX style)
    - Fused Triton kernel when available
    - Dynamic sequence length extension
    
    Args:
        head_dim: Dimension per attention head (must be even)
        max_seq_len: Maximum sequence length to precompute
        theta: Base for frequency computation (default 10000.0)
        scaling_factor: For NTK-aware scaling (default 1.0)
    """

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int = 8192,
        theta: float = 10000.0,
        scaling_factor: float = 1.0,
    ):
        super().__init__()
        
        assert head_dim % 2 == 0, f"head_dim must be even, got {head_dim}"
        
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.scaling_factor = scaling_factor
        
        # Precompute frequencies
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len: int):
        """Build or extend the RoPE cache."""
        # Inverse frequencies
        inv_freq = 1.0 / (
            self.theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim)
        )
        
        # Apply NTK scaling if specified
        if self.scaling_factor != 1.0:
            inv_freq = inv_freq / self.scaling_factor
        
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Position indices
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)  # [seq_len, head_dim/2]
        
        # Cache cos and sin with broadcast dimensions [1, 1, seq, head_dim/2]
        cos_cached = freqs.cos().unsqueeze(0).unsqueeze(0)
        sin_cached = freqs.sin().unsqueeze(0).unsqueeze(0)
        
        self.register_buffer("cos_cached", cos_cached, persistent=False)
        self.register_buffer("sin_cached", sin_cached, persistent=False)
        self.max_seq_len = seq_len
    
    def _extend_cache(self, seq_len: int):
        """Extend cache if needed for longer sequences."""
        if seq_len > self.max_seq_len:
            # Double the size for amortized extension
            new_len = max(seq_len, self.max_seq_len * 2)
            self._build_cache(new_len)
    
    def forward(self, x: torch.Tensor, seq_len: int = None) -> torch.Tensor:
        """Apply rotary embeddings to input tensor.
        
        Args:
            x: Input tensor [batch, n_heads, seq_len, head_dim]
            seq_len: Actual sequence length (for slicing cache)
            
        Returns:
            Rotated tensor with same shape as input
        """
        if seq_len is None:
            seq_len = x.shape[2]
        
        # Extend cache if needed
        if seq_len > self.max_seq_len:
            self._extend_cache(seq_len)
        
        # Get cached values for current sequence length
        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]
        
        # Use fused kernel if available
        if FUSED_KERNELS_AVAILABLE and fused_rope is not None:
            return fused_rope(x, cos, sin)
        
        # PyTorch implementation
        return self._apply_rope_pytorch(x, cos, sin)
    
    @staticmethod
    def _apply_rope_pytorch(
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """PyTorch implementation of RoPE."""
        # Split into pairs
        x1, x2 = x[..., ::2], x[..., 1::2]
        
        # Apply rotation
        rotated = torch.stack(
            [x1 * cos - x2 * sin, x1 * sin + x2 * cos],
            dim=-1
        ).flatten(-2)
        
        return rotated

    def get_cos_sin(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get cos/sin for external use (e.g., in KV cache)."""
        if seq_len > self.max_seq_len:
            self._extend_cache(seq_len)
        return self.cos_cached[:, :, :seq_len, :], self.sin_cached[:, :, :seq_len, :]


# =============================================================================
# Flexible Attention Function
# =============================================================================

def flexible_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = True,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Flexible attention with automatic backend selection.
    
    Automatically uses best available backend:
    1. Flash Attention 2 (fastest, memory efficient)
    2. xFormers (good memory efficiency)
    3. PyTorch SDPA (good performance, always available)
    
    Args:
        q: Query tensor [batch, n_heads, seq_len, head_dim]
        k: Key tensor [batch, n_kv_heads, seq_len, head_dim]
        v: Value tensor [batch, n_kv_heads, seq_len, head_dim]
        attn_mask: Optional attention mask
        dropout_p: Dropout probability
        is_causal: Whether to use causal masking
        scale: Optional scale factor (default: 1/sqrt(head_dim))
        
    Returns:
        Output tensor [batch, n_heads, seq_len, head_dim]
    """
    backend = get_attention_backend()
    
    B, H, S, D = q.shape
    _, KV_H, _, _ = k.shape
    
    if scale is None:
        scale = D ** -0.5
    
    # Handle GQA expansion if needed
    if H != KV_H:
        n_groups = H // KV_H
        k = k.unsqueeze(2).expand(-1, -1, n_groups, -1, -1).reshape(B, H, S, D)
        v = v.unsqueeze(2).expand(-1, -1, n_groups, -1, -1).reshape(B, H, S, D)
    
    if backend == "flash" and FLASH_ATTN_AVAILABLE:
        # Flash Attention expects [batch, seq, n_heads, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        out = flash_attn_func(
            q, k, v,
            dropout_p=dropout_p if q.requires_grad else 0.0,
            softmax_scale=scale,
            causal=is_causal,
        )
        return out.transpose(1, 2)
    
    elif backend == "xformers" and XFORMERS_AVAILABLE:
        # xFormers expects [batch, seq, n_heads, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        attn_bias = xops.LowerTriangularMask() if is_causal else attn_mask
        
        out = xops.memory_efficient_attention(
            q, k, v,
            attn_bias=attn_bias,
            scale=scale,
            p=dropout_p if q.requires_grad else 0.0,
        )
        return out.transpose(1, 2)
    
    else:
        # PyTorch SDPA
        return F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=dropout_p if q.requires_grad else 0.0,
            is_causal=is_causal if attn_mask is None else False,
            scale=scale,
        )


# =============================================================================
# Gradient Checkpointing Utilities
# =============================================================================

def checkpoint_sequential(
    functions: list,
    x: torch.Tensor,
    use_checkpoint: bool = True,
    preserve_rng_state: bool = True,
) -> torch.Tensor:
    """Apply gradient checkpointing to a sequence of functions.
    
    Args:
        functions: List of callables (e.g., transformer blocks)
        x: Input tensor
        use_checkpoint: Whether to use checkpointing (disable for inference)
        preserve_rng_state: Whether to preserve RNG state (important for dropout)
        
    Returns:
        Output tensor after applying all functions
    """
    if not use_checkpoint or not x.requires_grad:
        for fn in functions:
            x = fn(x)
        return x
    
    from torch.utils.checkpoint import checkpoint
    
    for fn in functions:
        x = checkpoint(
            fn,
            x,
            use_reentrant=False,
            preserve_rng_state=preserve_rng_state,
        )
    
    return x


class GradientCheckpointMixin:
    """Mixin to add gradient checkpointing support to models.
    
    Usage:
        class MyModel(nn.Module, GradientCheckpointMixin):
            def __init__(self):
                self._gradient_checkpointing = False
                ...
            
            def forward(self, x):
                if self._gradient_checkpointing:
                    x = checkpoint(self._forward_impl, x)
                else:
                    x = self._forward_impl(x)
                return x
    """
    
    _gradient_checkpointing: bool = False
    
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency."""
        self._gradient_checkpointing = True
        
        # Propagate to submodules
        for module in self.modules():
            if hasattr(module, "_gradient_checkpointing"):
                module._gradient_checkpointing = True
    
    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing."""
        self._gradient_checkpointing = False
        
        for module in self.modules():
            if hasattr(module, "_gradient_checkpointing"):
                module._gradient_checkpointing = False
    
    @property
    def is_gradient_checkpointing(self) -> bool:
        """Check if gradient checkpointing is enabled."""
        return getattr(self, "_gradient_checkpointing", False)


# =============================================================================
# Weight Initialization Utilities
# =============================================================================

def init_weights_normal(module: nn.Module, std: float = 0.02):
    """Initialize weights with normal distribution.
    
    Args:
        module: Module to initialize
        std: Standard deviation for normal distribution
    """
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=std)
    elif isinstance(module, nn.Conv1d):
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def scale_residual_weights(module: nn.Module, n_layers: int, scale_factor: float = None):
    """Scale residual projection weights for deep networks.
    
    Following GPT-2 and other deep transformer practices.
    
    Args:
        module: Module containing residual projections
        n_layers: Number of transformer layers
        scale_factor: Optional explicit scale (default: 1/sqrt(2*n_layers))
    """
    if scale_factor is None:
        scale_factor = 1.0 / math.sqrt(2 * n_layers)
    
    for name, param in module.named_parameters():
        # Scale output projections and down projections
        if "o_proj" in name or "down" in name:
            param.data *= scale_factor


# =============================================================================
# Utility Functions
# =============================================================================

@lru_cache(maxsize=32)
def get_alibi_slopes(n_heads: int) -> torch.Tensor:
    """Get ALiBi slopes for attention (cached).
    
    Args:
        n_heads: Number of attention heads
        
    Returns:
        Tensor of slopes [n_heads]
    """
    def _get_slopes_power_of_2(n: int):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio**i for i in range(n)]
    
    if math.log2(n_heads).is_integer():
        return torch.tensor(_get_slopes_power_of_2(n_heads))
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(n_heads))
        slopes_a = _get_slopes_power_of_2(closest_power_of_2)
        slopes_b = _get_slopes_power_of_2(2 * closest_power_of_2)
        slopes_b = slopes_b[0::2][:n_heads - closest_power_of_2]
        return torch.tensor(slopes_a + slopes_b)


def compute_memory_footprint(model: nn.Module, batch_size: int = 1, seq_len: int = 512) -> dict:
    """Estimate memory footprint of a model.
    
    Args:
        model: PyTorch model
        batch_size: Batch size for activation memory estimate
        seq_len: Sequence length for activation memory estimate
        
    Returns:
        Dict with memory estimates in MB
    """
    param_mem = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
    buffer_mem = sum(b.numel() * b.element_size() for b in model.buffers()) / 1e6
    
    # Rough activation estimate (varies by architecture)
    # Assume ~4 bytes per activation, ~10x params for activations at training
    activation_mem = param_mem * 10 * batch_size * (seq_len / 512)
    
    return {
        "parameters_mb": param_mem,
        "buffers_mb": buffer_mem,
        "total_static_mb": param_mem + buffer_mem,
        "estimated_activation_mb": activation_mem,
        "estimated_training_mb": param_mem * 4 + activation_mem,  # Optimizer states
    }
