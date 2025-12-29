# CCGQA Attention Documentation

**Compressed Convolutional Grouped Query Attention**

## Quick Links

- **[Baseline Benchmark](./baseline_benchmark.md)** - Performance baseline vs FlashAttn-2, PyTorch SDPA, GQA
- **[Benchmark Script](../benchmark.py)** - Comprehensive GPU/CPU benchmark with charting
- **[Implementation](../attention.py)** - Main CCGQAAttention module

## Overview

CCGQA is a novel attention mechanism that combines:
1. **Compression**: 2-8x reduction in attention matrix size
2. **Convolution**: Local context via Conv1d on Q and K
3. **Grouped Query Attention**: Efficient KV cache sharing
4. **QK Coupling**: Mean-coupling between queries and keys
5. **Value Shift**: Layer-wise value transformation

## Current Status

**Baseline Established** (December 22, 2024)

| Metric | Value | Comparison |
|--------|-------|------------|
| **Speed (N=512)** | 2.84ms | **9.2x slower** than FlashAttn-2 |
| **Speed (N=1024)** | 1.46ms | **4.4x slower** than FlashAttn-2 |
| **Memory (N=512)** | 31.3MB | **3.1x less** than reference GQA |
| **Memory (N=1024)** | 58.3MB | **5.4x less** than reference GQA |

**Key Finding**: CCGQA trades speed for memory efficiency. The backward pass is the primary bottleneck (64-82% of total time).

## Optimization Roadmap

### Phase 1: Kernel Fusion (Target: Q1 2025)
- [ ] Fused forward pass (Triton/CUDA)
- [ ] Custom backward kernel
- [ ] **Goal**: 3-5x speedup

### Phase 2: Convolution Optimization (Target: Q1 2025)
- [ ] Depthwise-separable convolutions
- [ ] Inline into attention kernel
- [ ] **Goal**: 1.5-2x additional speedup

### Phase 3: Compiler Integration (Target: Q2 2025)
- [ ] torch.compile compatibility
- [ ] Graph optimization
- [ ] **Goal**: 1.3-1.8x additional speedup

### Final Target
- **Speed**: ≤2x slower than FlashAttention-2
- **Memory**: Maintain 3-5x advantage

## Benchmark Quick Start

```bash
# Quick GPU benchmark (recommended)
python hydra/attention/backends/ccgqa/benchmark.py --quick --device cuda --plot --save

# Full GPU benchmark (longer, more sequence lengths)
python hydra/attention/backends/ccgqa/benchmark.py --device cuda --plot --save

# CPU benchmark
python hydra/attention/backends/ccgqa/benchmark.py --device cpu --save
```

**Outputs**:
- JSON results in `docs/benchmark_results_*.json`
- Charts: `benchmark_comparison.png`, `fwd_bwd_comparison.png`, `memory_comparison.png`

## Architecture

```
Input (B, N, D)
    ↓
[Q, K, V Projections]
    ↓
[Conv1d on Q, K] ← Local context
    ↓
[Compression] ← 2-8x reduction
    ↓
[QK-Mean Coupling] ← Stability
    ↓
[Attention + Value-Shift]
    ↓
[Output Projection]
    ↓
Output (B, N, D)
```

## Key Features

1. **Memory Efficient**: 3-5x less memory than standard attention
2. **Long Context**: Compression enables longer sequences
3. **Local + Global**: Conv + attention combines patterns
4. **Stable**: QK normalization, mean coupling

## Performance Characteristics

**Strengths**:
- Low memory footprint (ideal for long contexts)
- Scales better than quadratic attention
- Combines local (conv) and global (attention) context

**Current Limitations** (to be addressed):
- Slower than optimized kernels (FlashAttn-2, PyTorch SDPA)
- Backward pass bottleneck (82% of time @ N=512)
- No kernel fusion yet

## Files

```
ccgqa/
├── attention.py          # Main CCGQAAttention implementation
├── benchmark.py          # Comprehensive benchmark script ← NEW
├── benchmarks/
│   ├── benchmark_ccqa_cpu.py       # Simple CPU benchmark
│   └── benchmark_ccqa_report.py    # JSON report generator
└── docs/
    ├── README.md                    # This file
    ├── baseline_benchmark.md        # Performance baseline ← UPDATED
    ├── benchmark_results_*.json     # Raw benchmark data
    ├── benchmark_comparison.png     # Total time chart
    ├── fwd_bwd_comparison.png       # Forward/backward breakdown
    └── memory_comparison.png        # Memory usage chart
```

## Interface Contract

The CCGQA module is a drop-in attention implementation used inside the HYDRA model stack:

- **Backend name**: `ccgqa`
- **Implementation**: `hydra.attention.backends.ccgqa.attention.CCGQAAttention`
- **Public API**: Import via `from hydra.attention.backends.ccgqa.attention import CCGQAAttention`
- **CPU-safe**: Can import and run without CUDA dependencies

## Next Steps

1. **Profile** - Use `torch.profiler` to identify hotspots
2. **Fuse** - Implement Triton kernel for forward pass
3. **Measure** - Re-run benchmark to quantify improvement
4. **Iterate** - Continue optimizing until target achieved

See [baseline_benchmark.md](./baseline_benchmark.md) for detailed analysis and optimization plan.

---

**Last Updated**: December 22, 2024  
**Status**: Baseline established, optimization in progress
