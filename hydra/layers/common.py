"""
HYDRA Shared Layers

Deduplicated common components used across attention variants.
All implementations are torch.compile compatible.

Key design decisions for compile compatibility:
1. Use register_buffer for step counters (not Python ints)
2. Cache boolean decisions outside forward pass
3. Use @torch.compiler.disable for stat-gathering functions
4. Triton kernels wrapped with compiler.disable
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Callable

# =============================================================================
# FEATURE FLAGS: Detect available backends at import time
# =============================================================================

FUSED_KERNELS_AVAILABLE = False
try:
    from hydra.kernels import fused_rms_norm, fused_swiglu, fused_rope, fused_qk_norm
    FUSED_KERNELS_AVAILABLE = True
except ImportError:
    fused_rms_norm = None
    fused_swiglu = None
    fused_rope = None
    fused_qk_norm = None

FLASH_ATTN_AVAILABLE = False
try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    flash_attn_func = None

XFORMERS_AVAILABLE = False
try:
    import xformers.ops as xops
    XFORMERS_AVAILABLE = True
except ImportError:
    xops = None


# =============================================================================
# ACTIVATION FUNCTIONS
# =============================================================================

def get_activation(name: str) -> Callable:
    """Get activation function by name."""
    activations = {
        "relu": F.relu,
        "gelu": F.gelu,
        "silu": F.silu,
        "swish": F.silu,
        "tanh": torch.tanh,
        "sigmoid": torch.sigmoid,
    }
    if name.lower() not in activations:
        raise ValueError(f"Unknown activation: {name}. Available: {list(activations.keys())}")
    return activations[name.lower()]


# =============================================================================
# RMSNorm: Root Mean Square Layer Normalization
# =============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.
    
    More efficient than LayerNorm as it doesn't compute mean.
    Formula: x * rsqrt(mean(x^2) + eps) * weight
    
    Args:
        dim: Hidden dimension
        eps: Small constant for numerical stability
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use fused kernel if available (already wrapped with @compiler.disable in fused_ops.py)
        if FUSED_KERNELS_AVAILABLE and fused_rms_norm is not None:
            return fused_rms_norm(x, self.weight, self.eps)
        
        # PyTorch implementation
        dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(dtype)


# =============================================================================
# SwiGLU MLP: Gated Linear Unit with Swish activation
# =============================================================================

class SwiGLUMLP(nn.Module):
    """SwiGLU MLP block.
    
    SwiGLU = Swish(x * W1) * (x * W2) then project down
    More expressive than standard MLP with similar param count.
    
    Args:
        dim: Input/output dimension
        hidden_dim: Hidden dimension (typically 2.67x dim for param parity with 4x GELU MLP)
        bias: Whether to use bias in linear layers
    """

    def __init__(self, dim: int, hidden_dim: int, bias: bool = False):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)  # Gate projection
        self.w2 = nn.Linear(dim, hidden_dim, bias=bias)  # Up projection
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)  # Down projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use fused kernel if available
        if FUSED_KERNELS_AVAILABLE and fused_swiglu is not None:
            return fused_swiglu(x, self.w1.weight, self.w2.weight, self.w3.weight,
                               self.w1.bias, self.w2.bias, self.w3.bias)
        
        # PyTorch implementation
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


# =============================================================================
# Rotary Position Embeddings (RoPE)
# =============================================================================

class RotaryEmbedding(nn.Module):
    """Rotary Position Embeddings.
    
    Encodes position by rotating query/key vectors. Enables extrapolation
    beyond training length and has nice properties for relative positions.
    
    Args:
        head_dim: Dimension per attention head (must be even)
        max_seq_len: Maximum sequence length to cache
        base: Base for frequency computation (10000 in original paper)
        scaling_factor: Optional scaling for extended context
    """

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int = 8192,
        base: float = 10000.0,
        scaling_factor: float = 1.0,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.scaling_factor = scaling_factor
        
        self._init_cache(max_seq_len)

    def _init_cache(self, seq_len: int):
        """Initialize or extend the cos/sin cache."""
        head_dim = self.head_dim
        
        # Compute frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        if self.scaling_factor != 1.0:
            inv_freq = inv_freq / self.scaling_factor
        
        # Compute position encodings
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        
        # Cache as buffers (will move to correct device automatically)
        # Shape: [1, 1, seq_len, head_dim//2]
        self.register_buffer("cos_cached", freqs.cos().unsqueeze(0).unsqueeze(0), persistent=False)
        self.register_buffer("sin_cached", freqs.sin().unsqueeze(0).unsqueeze(0), persistent=False)
        self.max_seq_len = seq_len

    def extend_cache(self, seq_len: int):
        """Extend cache if needed for longer sequences."""
        if seq_len > self.max_seq_len:
            self._init_cache(seq_len)
            # Move to same device as existing buffers
            device = self.cos_cached.device
            self.cos_cached = self.cos_cached.to(device)
            self.sin_cached = self.sin_cached.to(device)

    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> torch.Tensor:
        """Apply rotary embeddings to input tensor.
        
        Args:
            x: Input tensor of shape [B, n_heads, seq_len, head_dim]
            seq_len: Optional sequence length (defaults to x.shape[2])
        
        Returns:
            Rotated tensor of same shape
        """
        if seq_len is None:
            seq_len = x.shape[2]
        
        # Extend cache if needed
        if seq_len > self.max_seq_len:
            self.extend_cache(seq_len)
        
        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]
        
        # Use fused kernel if available
        if FUSED_KERNELS_AVAILABLE and fused_rope is not None:
            return fused_rope(x, cos, sin)
        
        # PyTorch implementation: split into pairs, rotate, recombine
        x1, x2 = x[..., ::2], x[..., 1::2]
        rotated = torch.stack(
            [x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1
        ).flatten(-2)
        return rotated

    def get_cos_sin(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get cos/sin tensors for manual application."""
        if seq_len > self.max_seq_len:
            self.extend_cache(seq_len)
        return self.cos_cached[:, :, :seq_len, :], self.sin_cached[:, :, :seq_len, :]


# =============================================================================
# FLEXIBLE ATTENTION: Backend selection for scaled dot-product attention
# =============================================================================

_ATTENTION_BACKEND = "auto"  # "auto", "flash", "xformers", "sdpa", "naive"

def set_attention_backend(backend: str):
    """Set the attention backend to use.
    
    Args:
        backend: One of "auto", "flash", "xformers", "sdpa", "naive"
    """
    global _ATTENTION_BACKEND
    valid = {"auto", "flash", "xformers", "sdpa", "naive"}
    if backend not in valid:
        raise ValueError(f"Invalid backend: {backend}. Valid: {valid}")
    _ATTENTION_BACKEND = backend

def get_attention_backend() -> str:
    """Get current attention backend."""
    return _ATTENTION_BACKEND

def flexible_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = True,
    scale: Optional[float] = None,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """Flexible attention with automatic backend selection.
    
    Tries backends in order: Flash Attention 2 > xFormers > PyTorch SDPA > naive
    
    Args:
        q: Query tensor [B, n_heads, seq_len, head_dim]
        k: Key tensor [B, n_kv_heads, seq_len, head_dim]
        v: Value tensor [B, n_kv_heads, seq_len, head_dim]
        is_causal: Whether to use causal masking
        scale: Optional attention scale (defaults to 1/sqrt(head_dim))
        dropout_p: Dropout probability (only used in training)
    
    Returns:
        Output tensor [B, n_heads, seq_len, head_dim]
    """
    backend = _ATTENTION_BACKEND
    B, n_heads, seq_len, head_dim = q.shape
    n_kv_heads = k.shape[1]
    
    # GQA expansion if needed
    if n_kv_heads != n_heads:
        n_groups = n_heads // n_kv_heads
        k = k.unsqueeze(2).expand(-1, -1, n_groups, -1, -1).reshape(B, n_heads, seq_len, head_dim)
        v = v.unsqueeze(2).expand(-1, -1, n_groups, -1, -1).reshape(B, n_heads, seq_len, head_dim)
    
    if scale is None:
        scale = head_dim ** -0.5
    
    # Try Flash Attention 2
    if backend in ("auto", "flash") and FLASH_ATTN_AVAILABLE and flash_attn_func is not None:
        # Flash attention expects [B, seq_len, n_heads, head_dim]
        q_fa = q.transpose(1, 2).contiguous()
        k_fa = k.transpose(1, 2).contiguous()
        v_fa = v.transpose(1, 2).contiguous()
        out = flash_attn_func(q_fa, k_fa, v_fa, causal=is_causal, softmax_scale=scale)
        return out.transpose(1, 2)
    
    # Try xFormers
    if backend in ("auto", "xformers") and XFORMERS_AVAILABLE and xops is not None:
        # xFormers expects [B, seq_len, n_heads, head_dim]
        q_xf = q.transpose(1, 2).contiguous()
        k_xf = k.transpose(1, 2).contiguous()
        v_xf = v.transpose(1, 2).contiguous()
        attn_bias = xops.LowerTriangularMask() if is_causal else None
        out = xops.memory_efficient_attention(q_xf, k_xf, v_xf, attn_bias=attn_bias, scale=scale)
        return out.transpose(1, 2)
    
    # PyTorch SDPA (default)
    if backend in ("auto", "sdpa"):
        return F.scaled_dot_product_attention(
            q, k, v,
            is_causal=is_causal,
            scale=scale,
            dropout_p=dropout_p if q.requires_grad else 0.0,
        )
    
    # Naive implementation (for debugging)
    attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
    if is_causal:
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool), diagonal=1)
        attn_weights.masked_fill_(causal_mask, float('-inf'))
    attn_weights = F.softmax(attn_weights, dim=-1)
    if dropout_p > 0 and q.requires_grad:
        attn_weights = F.dropout(attn_weights, p=dropout_p)
    return torch.matmul(attn_weights, v)


# =============================================================================
# GRADIENT CHECKPOINTING MIXIN
# =============================================================================

class GradientCheckpointMixin:
    """Mixin to add gradient checkpointing support to any model.
    
    Gradient checkpointing trades compute for memory by not storing
    intermediate activations. Instead, they're recomputed during backward.
    
    Usage:
        class MyModel(nn.Module, GradientCheckpointMixin):
            def __init__(self):
                super().__init__()
                self.enable_gradient_checkpointing()
    """
    
    _gradient_checkpointing_enabled: bool = False
    
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for this module."""
        self._gradient_checkpointing_enabled = True
    
    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing for this module."""
        self._gradient_checkpointing_enabled = False
    
    @property
    def gradient_checkpointing_enabled(self) -> bool:
        return getattr(self, '_gradient_checkpointing_enabled', False)


def checkpoint_sequential(
    functions: list,
    input: torch.Tensor,
    use_reentrant: bool = False,
) -> torch.Tensor:
    """Apply gradient checkpointing to a sequence of functions.
    
    Args:
        functions: List of nn.Module or callable
        input: Input tensor
        use_reentrant: Whether to use reentrant checkpointing (False recommended)
    
    Returns:
        Output tensor after applying all functions
    """
    from torch.utils.checkpoint import checkpoint
    
    def run_function(start, end, functions):
        def forward(input):
            for j in range(start, end):
                input = functions[j](input)
            return input
        return forward
    
    # Checkpoint every function
    for i, func in enumerate(functions):
        if isinstance(func, nn.Module):
            input = checkpoint(func, input, use_reentrant=use_reentrant)
        else:
            input = checkpoint(run_function(i, i+1, functions), input, use_reentrant=use_reentrant)
    
    return input


# =============================================================================
# WEIGHT INITIALIZATION UTILITIES
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


def scale_residual_weights(model: nn.Module, n_layers: int, pattern: str = "output"):
    """Scale residual connection weights for stable deep training.
    
    Following GPT-2 and related work, scale output projections by 1/sqrt(2*n_layers).
    
    Args:
        model: Model to scale
        n_layers: Number of layers (for scaling factor)
        pattern: String pattern to match in parameter names ("output", "o_proj", "down")
    """
    scale = 1.0 / math.sqrt(2 * n_layers)
    for name, param in model.named_parameters():
        if pattern in name:
            param.data *= scale


# =============================================================================
# ALIBI SLOPES (for alternative position encoding)
# =============================================================================

def get_alibi_slopes(n_heads: int) -> torch.Tensor:
    """Compute ALiBi slopes for attention bias.
    
    ALiBi adds a linear bias based on position distance: bias = -slope * |i - j|
    
    Args:
        n_heads: Number of attention heads
    
    Returns:
        Tensor of slopes, shape [n_heads]
    """
    def get_slopes_power_of_2(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * (ratio ** i) for i in range(n)]
    
    if math.log2(n_heads).is_integer():
        return torch.tensor(get_slopes_power_of_2(n_heads))
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(n_heads))
        slopes = get_slopes_power_of_2(closest_power_of_2)
        extra_slopes = get_slopes_power_of_2(2 * closest_power_of_2)[0::2][:n_heads - closest_power_of_2]
        return torch.tensor(slopes + extra_slopes)


# =============================================================================
# MEMORY UTILITIES
# =============================================================================

@torch.compiler.disable
def compute_memory_footprint(model: nn.Module, input_shape: Tuple[int, ...], dtype: torch.dtype = torch.float32) -> dict:
    """Compute memory footprint of a model.
    
    Args:
        model: Model to analyze
        input_shape: Shape of input tensor (B, seq_len) or (B, seq_len, dim)
        dtype: Data type for computation
    
    Returns:
        Dict with memory breakdown in MB
    """
    # Parameter memory
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
    
    # Gradient memory (same as params during training)
    grad_memory = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)
    
    # Buffer memory
    buffer_memory = sum(b.numel() * b.element_size() for b in model.buffers())
    
    # Estimate activation memory (rough, depends on actual forward pass)
    # This is a heuristic based on typical transformer architectures
    if len(input_shape) == 2:
        B, seq_len = input_shape
        dim = getattr(model, 'dim', 1024)
    else:
        B, seq_len, dim = input_shape
    
    n_layers = getattr(model, 'n_layers', getattr(model, 'n_mor_blocks', 12))
    bytes_per_elem = torch.tensor([], dtype=dtype).element_size()
    
    # Rough estimate: 4x model dim per layer for activations
    activation_memory = B * seq_len * dim * 4 * n_layers * bytes_per_elem
    
    MB = 1024 * 1024
    return {
        "parameters_mb": param_memory / MB,
        "gradients_mb": grad_memory / MB,
        "buffers_mb": buffer_memory / MB,
        "activations_mb_estimate": activation_memory / MB,
        "total_training_mb_estimate": (param_memory + grad_memory + buffer_memory + activation_memory) / MB,
    }


__all__ = [
    # Core layers
    "RMSNorm",
    "SwiGLUMLP", 
    "RotaryEmbedding",
    "get_activation",
    # Attention
    "flexible_attention",
    "set_attention_backend",
    "get_attention_backend",
    # Gradient checkpointing
    "GradientCheckpointMixin",
    "checkpoint_sequential",
    # Weight initialization
    "init_weights_normal",
    "scale_residual_weights",
    # Utilities
    "get_alibi_slopes",
    "compute_memory_footprint",
    # Feature flags
    "FUSED_KERNELS_AVAILABLE",
    "FLASH_ATTN_AVAILABLE",
    "XFORMERS_AVAILABLE",
]
