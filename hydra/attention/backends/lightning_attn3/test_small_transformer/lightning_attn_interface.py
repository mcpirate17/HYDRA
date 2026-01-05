import math
import logging

import torch
import torch.nn.functional as F

from .triton import lightning_attn3, lightning_attn3_no_decay, lightning_attn3_parallel
from .triton import lightning_attn3_no_decay_chunked

logger = logging.getLogger(__name__)

# Cache for Blackwell detection
_IS_BLACKWELL_CACHE: dict[int, bool] = {}

def _is_blackwell(device_idx: int = 0) -> bool:
    """Check if device is Blackwell (SM 12.x) architecture."""
    if device_idx not in _IS_BLACKWELL_CACHE:
        props = torch.cuda.get_device_properties(device_idx)
        _IS_BLACKWELL_CACHE[device_idx] = props.major >= 12
    return _IS_BLACKWELL_CACHE[device_idx]


# ============================================================================
# SELF-TEST FOR LIGHTNING ATTENTION-3
# ============================================================================

class LightningAttn3ValidationError(RuntimeError):
    """Raised when Lightning Attention-3 kernel validation fails."""
    pass


def validate_lightning_attn3(
    n_heads: int,
    head_dim: int,
    seq_len: int,
    batch_size: int = 1,
    dtype: torch.dtype = torch.float16,
    device: str | torch.device = "cuda",
    use_decay: bool = False,
    raise_on_failure: bool = True,
) -> tuple[bool, str | None]:
    """
    Validate Lightning Attention-3 forward and backward for a given configuration.
    
    This runs a lightweight test to catch Triton OutOfResources errors (shared
    memory OOM) before they occur during actual training.
    
    Args:
        n_heads: Number of attention heads
        head_dim: Dimension per head (d)
        seq_len: Sequence length (n)
        batch_size: Batch size for test (default 1)
        dtype: Data type (default float16)
        device: CUDA device
        use_decay: Whether to test decay variant
        raise_on_failure: Raise exception on failure (default True)
        
    Returns:
        (success: bool, error_message: str | None)
        
    Raises:
        LightningAttn3ValidationError: If validation fails and raise_on_failure=True
        
    Example:
        >>> validate_lightning_attn3(n_heads=32, head_dim=128, seq_len=2048)
        (True, None)
    """
    if not torch.cuda.is_available():
        msg = "CUDA not available for Lightning Attention-3 validation"
        if raise_on_failure:
            raise LightningAttn3ValidationError(msg)
        return False, msg
    
    # Get device properties for diagnostic
    if isinstance(device, str):
        device = torch.device(device)
    device_idx = device.index if device.index is not None else torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device_idx)
    gpu_name = props.name
    sm_version = f"{props.major}.{props.minor}"
    max_sram = props.shared_memory_per_block_optin
    
    config_str = (
        f"Lightning Attention-3 config: "
        f"n_heads={n_heads}, head_dim={head_dim}, seq_len={seq_len}, "
        f"dtype={dtype}, decay={use_decay}"
    )
    device_str = f"GPU: {gpu_name} (SM {sm_version}, max_sram={max_sram} bytes)"
    
    logger.info(f"Validating {config_str}")
    logger.debug(device_str)
    
    try:
        # Create minimal test tensors
        q = torch.randn(
            batch_size, n_heads, seq_len, head_dim,
            device=device, dtype=dtype, requires_grad=True
        )
        k = torch.randn_like(q, requires_grad=True)
        v = torch.randn_like(q, requires_grad=True)
        
        # Optional decay parameter
        s = None
        if use_decay:
            s = torch.randn(n_heads, device=device, dtype=dtype).abs() * 0.1
        
        # Forward pass
        out = lightning_attn_func(q, k, v, s=s)
        
        # Backward pass
        loss = out.sum()
        loss.backward()
        
        # Verify gradients exist
        if q.grad is None or k.grad is None or v.grad is None:
            msg = f"Gradient computation failed - gradients are None\n{config_str}\n{device_str}"
            if raise_on_failure:
                raise LightningAttn3ValidationError(msg)
            return False, msg
        
        # Check for NaN/Inf
        if torch.isnan(out).any() or torch.isinf(out).any():
            msg = f"Forward pass produced NaN/Inf\n{config_str}\n{device_str}"
            if raise_on_failure:
                raise LightningAttn3ValidationError(msg)
            return False, msg
        
        if torch.isnan(q.grad).any() or torch.isnan(k.grad).any() or torch.isnan(v.grad).any():
            msg = f"Backward pass produced NaN gradients\n{config_str}\n{device_str}"
            if raise_on_failure:
                raise LightningAttn3ValidationError(msg)
            return False, msg
        
        # Clean up
        del q, k, v, out, loss
        torch.cuda.empty_cache()
        
        logger.info(f"✓ Lightning Attention-3 validation passed")
        return True, None
        
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        
        # Detect Triton OutOfResources (shared memory OOM)
        is_oom = "OutOfResources" in error_type or "out of resource" in error_msg.lower()
        is_sram_oom = "shared memory" in error_msg.lower() or "smem" in error_msg.lower()
        
        if is_oom or is_sram_oom:
            msg = (
                f"Lightning Attention-3 SHARED MEMORY OOM\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"{config_str}\n"
                f"{device_str}\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"Error: {error_type}: {error_msg}\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"DIAGNOSIS:\n"
                f"  The kernel requires more shared memory than available.\n"
                f"  SM {sm_version} has {max_sram} bytes opt-in shared memory.\n"
                f"\n"
                f"SUGGESTED FIXES:\n"
                f"  1. Reduce head_dim to ≤128\n"
                f"  2. If on Blackwell (SM 12.x), ensure chunked backward is used\n"
                f"  3. Try smaller sequence lengths\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            )
        else:
            msg = (
                f"Lightning Attention-3 validation FAILED\n"
                f"{config_str}\n"
                f"{device_str}\n"
                f"Error: {error_type}: {error_msg}"
            )
        
        logger.error(msg)
        
        if raise_on_failure:
            raise LightningAttn3ValidationError(msg) from e
        return False, msg


def validate_at_init(
    n_heads: int,
    head_dim: int,
    seq_len: int = 128,
    device: str | torch.device = "cuda",
) -> None:
    """
    Quick validation to call at model __init__ time.
    
    Uses minimal seq_len to reduce overhead while still testing the kernel.
    
    Args:
        n_heads: Number of attention heads
        head_dim: Dimension per head
        seq_len: Sequence length for test (default 128 for speed)
        device: CUDA device
        
    Raises:
        LightningAttn3ValidationError: If kernel fails
        
    Example:
        class MyModel(nn.Module):
            def __init__(self, n_heads, head_dim, ...):
                super().__init__()
                validate_at_init(n_heads, head_dim)
                ...
    """
    validate_lightning_attn3(
        n_heads=n_heads,
        head_dim=head_dim,
        seq_len=seq_len,
        batch_size=1,
        device=device,
        raise_on_failure=True,
    )


def is_support(dim):
    return 16 % dim


def next_power_of_2(n):
    return 2 ** (int(math.ceil(math.log(n, 2))))


def _cpu_linear_attention_reference(q, k, v):
    """
    CPU-compatible reference implementation for linear attention (no decay).
    
    Used for testing only - NOT for training (too slow).
    Implements causal linear attention: O = cumsum(K^T @ V) @ Q^T
    """
    b, h, n, d = q.shape
    e = v.shape[-1]
    
    # Causal linear attention via cumulative sum formulation
    # This is O(n²) but works on CPU for testing
    # Note: scaling is applied in lightning_attn_func, so not needed here
    
    # Compute attention scores (causal mask applied via cumulative sum)
    # For linear attention: output[i] = sum_{j<=i} q[i] @ k[j] * v[j]
    # This is equivalent to: output = cumsum(k^T @ v) @ q^T
    out = torch.zeros(b, h, n, e, device=q.device, dtype=q.dtype)
    
    for i in range(n):
        # Cumulative sum up to i
        k_cum = k[:, :, :i+1, :]  # [b, h, i+1, d]
        v_cum = v[:, :, :i+1, :]  # [b, h, i+1, e]
        
        # kv_cum = sum_{j=0 to i} k[j]^T @ v[j]  [d, e]
        kv_cum = torch.einsum('bhjd,bhje->bhde', k_cum, v_cum)
        
        # output[i] = q[i] @ kv_cum  [b, h, d] @ [b, h, d, e] -> [b, h, e]
        q_i = q[:, :, i, :]  # [b, h, d]
        out[:, :, i, :] = torch.einsum('bhd,bhde->bhe', q_i, kv_cum)
    
    return out


def lightning_attn_func(q, k, v, s=None, variant="chunk_loop"):
    # CPU fallback for testing (Triton kernels require CUDA)
    if not q.is_cuda:
        logger.warning("lightning_attn_func: CPU input detected, using reference implementation (slow, for testing only)")
        return _cpu_linear_attention_reference(q, k, v)
    
    # Apply scaling for numerical stability (prevents exploding gradients)
    b, h, n, d = q.shape
    scale = 1.0 / math.sqrt(d)
    q_scaled = q * scale
    
    if s is None:
        # Use chunked backward on Blackwell (SM 12.x) due to shared memory limits
        device_idx = q.device.index if q.device.index is not None else 0
        if _is_blackwell(device_idx):
            fn = lightning_attn3_no_decay_chunked
            # Keep tests/diagnostics stable: populate the no_decay cache with the
            # effective Blackwell selection.
            from .triton.lightning_attn3_no_decay import _BWD_KERNEL_CACHE
            _BWD_KERNEL_CACHE[device_idx] = ("chunked", 16, 0)
        else:
            fn = lightning_attn3_no_decay
    else:
        if variant == "parallel":
            fn = lightning_attn3_parallel
        elif variant == "chunk_loop":
            fn = lightning_attn3
    b, h, n, d = q.shape
    e = v.shape[-1]
    assert is_support(d) and is_support(e)

    # pad v's feature dim to power of 2
    e_pad = next_power_of_2(e)
    need_pad = e_pad != e
    if need_pad:
        v = F.pad(v, (0, e_pad - e))

    if d > 128:
        # split over head
        if d % 64 == 0:
            m = 64
        elif d % 32 == 0:
            m = 32
        elif d % 16 == 0:
            m = 16
        arr = [m * i for i in range(d // m + 1)]
        if arr[-1] != d:
            arr.append(d)
        n = len(arr)
        o = 0
        for i in range(n - 1):
            start = arr[i]
            end = arr[i + 1]
            q1 = q_scaled[..., start:end]
            k1 = k[..., start:end]
            if s != None:
                o += fn(q1, k1, v, s)
            else:
                o += fn(q1, k1, v)
    else:
        if s != None:
            o = fn(q_scaled, k, v, s)
        else:
            o = fn(q_scaled, k, v)

    if need_pad:
        o = o[:, :, :, :e]

    return o
