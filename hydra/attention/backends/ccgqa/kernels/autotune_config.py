"""
Block size autotuning and caching for CCGQA attention kernels.
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, Tuple, Optional

import torch


# Cache file location
CACHE_DIR = Path(__file__).parent / ".autotune_cache"
CACHE_FILE = CACHE_DIR / "block_configs.json"


def get_cache_key(N: int, D: int, dtype: str, device: str) -> str:
    """Generate cache key for configuration."""
    return f"N{N}_D{D}_{dtype}_{device}"


class BlockSizeCache:
    """Cache for optimal block sizes."""
    
    def __init__(self):
        self.cache: Dict[str, Tuple[int, int, int]] = {}
        self.load()
    
    def load(self):
        """Load cache from disk."""
        if CACHE_FILE.exists():
            try:
                with open(CACHE_FILE, 'r') as f:
                    data = json.load(f)
                    # Convert string keys to tuples
                    self.cache = {k: tuple(v) for k, v in data.items()}
            except Exception as e:
                print(f"Warning: Failed to load autotune cache: {e}")
                self.cache = {}
    
    def save(self):
        """Save cache to disk."""
        CACHE_DIR.mkdir(exist_ok=True)
        try:
            with open(CACHE_FILE, 'w') as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save autotune cache: {e}")
    
    def get(self, N: int, D: int, dtype: torch.dtype, device: str) -> Optional[Tuple[int, int, int]]:
        """Get cached block size configuration."""
        key = get_cache_key(N, D, str(dtype), device)
        return self.cache.get(key)
    
    def set(self, N: int, D: int, dtype: torch.dtype, device: str, 
            block_m: int, block_n: int, block_d: int):
        """Set block size configuration."""
        key = get_cache_key(N, D, str(dtype), device)
        self.cache[key] = (block_m, block_n, block_d)
        self.save()


# Global cache instance
_CACHE = BlockSizeCache()


def get_block_sizes(N: int, D: int, dtype: torch.dtype, device: str) -> Tuple[int, int, int]:
    """
    Get optimal block sizes for given configuration.
    Returns (BLOCK_M, BLOCK_N, BLOCK_D).
    Falls back to heuristics if not cached.
    """
    # Check cache first
    cached = _CACHE.get(N, D, dtype, device)
    if cached is not None:
        return cached
    
    # Fallback to heuristics (current implementation)
    return get_heuristic_block_sizes(N, D)


def get_heuristic_block_sizes(N: int, D: int) -> Tuple[int, int, int]:
    """
    Heuristic block size selection (existing logic).
    """
    if N >= 2048:
        if D <= 32:
            BLOCK_M, BLOCK_N = 128, 128
        elif D <= 48:
            BLOCK_M, BLOCK_N = 64, 128
        else:  # D=64
            BLOCK_M, BLOCK_N = 64, 64
    elif N >= 1024:
        if D <= 32:
            BLOCK_M, BLOCK_N = 64, 128
        elif D <= 48:
            BLOCK_M, BLOCK_N = 64, 64
        else:  # D=64
            BLOCK_M, BLOCK_N = 64, 64
    else:  # N < 1024
        if D <= 64:
            BLOCK_M, BLOCK_N = 64, 64
        else:
            BLOCK_M, BLOCK_N = 32, 64
    
    BLOCK_D = D  # Always use full head dimension
    return BLOCK_M, BLOCK_N, BLOCK_D


def benchmark_config(
    q: torch.Tensor,
    k: torch.Tensor, 
    v: torch.Tensor,
    block_m: int,
    block_n: int,
    block_d: int,
    warmup: int = 10,
    reps: int = 100,
) -> float:
    """
    Benchmark a specific block size configuration.
    Returns average time in milliseconds.
    """
    # Import kernels here to avoid circular dependency
    from .fused_attention import _fwd_kernel
    import triton
    
    B, H, N, D = q.shape
    
    # Allocate outputs
    out = torch.empty_like(q)
    lse = torch.empty((B, H, N), device=q.device, dtype=torch.float32)
    
    # Grid configuration (matches fused_attention.py)
    grid = (B, H, triton.cdiv(N, block_m))
    
    # Warmup
    for _ in range(warmup):
        _fwd_kernel[grid](
            q, k, v, out, lse,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            B=B, H=H, N=N, D=D,
            IS_CAUSAL=True,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_D=block_d,
            SAVE_LSE=True,
        )
    
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.time()
    for _ in range(reps):
        _fwd_kernel[grid](
            q, k, v, out, lse,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            B=B, H=H, N=N, D=D,
            IS_CAUSAL=True,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_D=block_d,
            SAVE_LSE=True,
        )
    torch.cuda.synchronize()
    
    elapsed = (time.time() - start) / reps * 1000
    return elapsed


def autotune_block_sizes(
    N: int,
    D: int,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    B: int = 4,
    H: int = 8,
) -> Tuple[int, int, int]:
    """
    Autotune block sizes for given sequence length and head dimension.
    """
    print(f"\nAutotuning block sizes for N={N}, D={D}, dtype={dtype}...")
    
    # Create test tensors
    q = torch.randn(B, H, N, D, dtype=dtype, device=device)
    k = torch.randn(B, H, N, D, dtype=dtype, device=device)
    v = torch.randn(B, H, N, D, dtype=dtype, device=device)
    
    # Candidate block sizes (must fit in shared memory)
    # Shared memory usage ~= BLOCK_M * BLOCK_N * 4 bytes (float32)
    # RTX 5090 has 101KB = ~103,000 bytes per SM
    candidates = []
    
    if D <= 32:
        # More room for larger blocks
        for block_m in [32, 64, 128]:
            for block_n in [32, 64, 128]:
                # Rough shared memory estimate: (block_m * D + block_n * D + block_m * block_n) * 4
                smem = (block_m * D + block_n * D + block_m * block_n) * 4
                if smem < 90000:  # Leave some margin
                    candidates.append((block_m, block_n, D))
    elif D <= 48:
        for block_m in [32, 64, 128]:
            for block_n in [32, 64, 128]:
                smem = (block_m * D + block_n * D + block_m * block_n) * 4
                if smem < 90000:
                    candidates.append((block_m, block_n, D))
    else:  # D=64
        # Limited by shared memory
        for block_m in [32, 64]:
            for block_n in [32, 64]:
                smem = (block_m * D + block_n * D + block_m * block_n) * 4
                if smem < 90000:
                    candidates.append((block_m, block_n, D))
    
    if not candidates:
        print(f"  Warning: No valid candidates for D={D}, using heuristic")
        return get_heuristic_block_sizes(N, D)
    
    print(f"  Testing {len(candidates)} configurations...")
    
    best_time = float('inf')
    best_config = None
    
    for block_m, block_n, block_d in candidates:
        try:
            elapsed = benchmark_config(q, k, v, block_m, block_n, block_d, warmup=5, reps=50)
            print(f"    BLOCK_M={block_m:3d}, BLOCK_N={block_n:3d}: {elapsed:.4f}ms")
            
            if elapsed < best_time:
                best_time = elapsed
                best_config = (block_m, block_n, block_d)
        except Exception as e:
            print(f"    BLOCK_M={block_m:3d}, BLOCK_N={block_n:3d}: Failed ({e})")
            continue
    
    if best_config is None:
        print(f"  All configs failed, using heuristic")
        return get_heuristic_block_sizes(N, D)
    
    print(f"  ✓ Best: BLOCK_M={best_config[0]}, BLOCK_N={best_config[1]} ({best_time:.4f}ms)")
    
    # Cache the result
    _CACHE.set(N, D, dtype, device, *best_config)
    
    return best_config
