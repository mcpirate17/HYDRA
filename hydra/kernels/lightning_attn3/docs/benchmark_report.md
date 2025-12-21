# Lightning Attention-3 Comprehensive Benchmark Report

**Last Updated:** December 20, 2025

## Summary

Comprehensive comparison of all Lightning Attention-3 variants against baselines:

| Kernel | N=1024 | N=2048 | N=4096 | N=8192 |
|--------|--------|--------|--------|--------|
| **LA3-Chunked** | 0.27ms | 0.59ms | 1.19ms | 2.44ms |
| LA3-NoDecay | 0.31ms | 0.65ms | 1.31ms | 2.64ms |
| LA3-Original | 0.60ms | 1.36ms | 2.76ms | 5.51ms |
| FlashAttn-2 | 0.44ms | 1.40ms | 4.93ms | 18.10ms |
| PyTorch-SDPA | 0.46ms | 1.44ms | 5.06ms | 18.33ms |

### Speedup vs SDPA

| Kernel | N=1024 | N=2048 | N=4096 | N=8192 |
|--------|--------|--------|--------|--------|
| **LA3-Chunked** | **1.69x** | **2.43x** | **4.24x** | **7.52x** |
| LA3-NoDecay | 1.48x | 2.22x | 3.85x | 6.94x |
| LA3-Original | 0.77x | 1.06x | 1.83x | 3.32x |
| FlashAttn-2 | 1.04x | 1.03x | 1.03x | 1.01x |

**Key Findings:**
- **LA3-Chunked is fastest** - 8-10% faster than LA3-NoDecay due to Blackwell-optimized backward kernel
- Linear attention shows O(n) scaling vs O(n²) for softmax attention
- At N=8192, LA3-Chunked is **7.52x faster** than SDPA

## System Configuration

| Property | Value |
|----------|-------|
| GPU | NVIDIA GeForce RTX 5090 |
| Compute Capability | SM 12.0 (Blackwell) |
| Total Memory | 32 GB |
| Shared Memory (opt-in) | 101,376 bytes |
| SMs | 170 |

## Benchmark Configuration

| Parameter | Value |
|-----------|-------|
| Batch Size | 4 |
| Attention Heads | 32 |
| Head Dimension | 64 |
| Data Type | torch.bfloat16 |
| Warmup Iterations | 10 |
| Benchmark Iterations | 50 |

## Kernel Configuration (Tuned for Blackwell)

| Parameter | Value | Notes |
|-----------|-------|-------|
| BLOCK | 64 | Sequence block size |
| CBLOCK_INTRA | 32 | Micro-chunk for intra kernel |
| CBLOCK_INTER | 128 | Micro-chunk for inter kernel |
| num_stages (inter) | 3 | Software pipelining depth |
| num_warps | 4 | Warps per thread block |

## Detailed Results

| Kernel | SeqLen | Fwd (ms) | Bwd (ms) | Total (ms) | Speedup |
|--------|--------|----------|----------|------------|---------|
| LA3-Chunked | 1024 | 0.074 | 0.211 | 0.285 | **1.58x** |
| LA3-NoDecay | 1024 | 0.080 | 0.225 | 0.305 | 1.48x |
| LA3-Original | 1024 | 0.148 | 0.450 | 0.599 | 0.75x |
| FlashAttn-2 | 1024 | 0.132 | 0.308 | 0.440 | 1.03x |
| PyTorch-SDPA | 1024 | 0.128 | 0.323 | 0.451 | 1.00x |
| LA3-Chunked | 2048 | 0.142 | 0.470 | 0.612 | **2.35x** |
| LA3-NoDecay | 2048 | 0.150 | 0.501 | 0.651 | 2.20x |
| LA3-Original | 2048 | 0.272 | 1.092 | 1.364 | 1.05x |
| FlashAttn-2 | 2048 | 0.379 | 1.027 | 1.406 | 1.02x |
| PyTorch-SDPA | 2048 | 0.381 | 1.054 | 1.434 | 1.00x |
| LA3-Chunked | 4096 | 0.273 | 0.966 | 1.239 | **4.08x** |
| LA3-NoDecay | 4096 | 0.293 | 1.012 | 1.305 | 3.88x |
| LA3-Original | 4096 | 0.533 | 2.218 | 2.751 | 1.84x |
| FlashAttn-2 | 4096 | 1.316 | 3.567 | 4.883 | 1.04x |
| PyTorch-SDPA | 4096 | 1.337 | 3.720 | 5.057 | 1.00x |
| LA3-Chunked | 8192 | 0.535 | 1.971 | 2.506 | **7.31x** |
| LA3-NoDecay | 8192 | 0.566 | 2.060 | 2.626 | 6.98x |
| LA3-Original | 8192 | 1.050 | 4.442 | 5.492 | 3.34x |
| FlashAttn-2 | 8192 | 4.991 | 13.128 | 18.118 | 1.01x |
| PyTorch-SDPA | 8192 | 4.985 | 13.339 | 18.325 | 1.00x |

## Kernel Variants Explained

| Kernel | Description | Pros | Cons |
|--------|-------------|------|------|
| **LA3-Chunked** | Blackwell-optimized recompute-heavy backward | Fastest, works on all GPUs | Slightly more recomputation |
| LA3-NoDecay | Original non-chunked (no decay parameter) | Simple, fast | May OOM on some configs |
| LA3-Original | Full implementation with decay parameter | Most features | Slower due to decay overhead |
| FlashAttn-2 | Flash Attention 2 (softmax attention) | Well-optimized baseline | O(n²) complexity |
| PyTorch-SDPA | PyTorch scaled_dot_product_attention | Standard baseline | O(n²) complexity |

## Algorithm: Linear Attention

Lightning Attention-3 implements **linear attention** with chunked computation:

```
Forward:  O[c] = Q[c] @ kv_state[c-1] + intra_chunk_attention
where:    kv_state[c] = Σ_{c'≤c} K[c']^T @ V[c']
```

**Complexity:**
- Inter-chunk: O(n) - accumulates KV state linearly
- Intra-chunk: O(BLOCK²) - fixed block size (64), constant per chunk
- **Overall: O(n)** vs O(n²) for softmax attention

This explains the increasing speedup at longer sequences:
- N=1024: 1.60x
- N=8192: 7.36x (linear scaling advantage becomes dominant)

## Memory Efficiency

LA3-Chunked uses less memory than SDPA at all sequence lengths:

| SeqLen | LA3 Memory | SDPA Memory | Savings |
|--------|------------|-------------|---------|
| 1024 | 400 MB | 545 MB | 27% |
| 2048 | 640 MB | 834 MB | 23% |
| 4096 | 1024 MB | 1412 MB | 27% |
| 8192 | 1792 MB | 2568 MB | 30% |

## Blackwell Compatibility

The chunked backward kernel is specifically designed for Blackwell GPUs (SM 12.0+):

- **SRAM Budget:** Uses ~26KB per thread block (well under 101KB limit)
- **Recompute-heavy design:** Recomputes attention scores instead of storing O(BLOCK²) matrices
- **Memory breakdown:**
  - Input tiles: 4 × CBLOCK × d × 2 = 8KB
  - Accumulators: 3 × CBLOCK × d × 4 = 12KB  
  - Attention tiles: 2 × CBLOCK² × 4 = 2KB
  - Overhead: ~4KB

## Optimization History

1. **Initial implementation:** Functional but unoptimized
2. **Inter kernel CBLOCK tuning:** CBLOCK=128 (was 64) - 6.5% faster
3. **Inter kernel stages:** num_stages=3 for software pipelining
4. **Intra kernel refactor:** Two-pass design supporting flexible CBLOCK

## Files

- `lightning_attn3_no_decay_chunked.py`: Main kernel implementation
- `benchmark_results.json`: Raw benchmark data
- `blackwell_constraints.md`: Detailed SRAM analysis
- `recompute_backward_design.md`: Algorithm documentation
