# Shared Task Notes

## Current Status

Completed inline profiling of HYDRA 100M training. Key bottlenecks and optimization opportunities identified.

## Profiling Results (100M Model, batch=4, seq=1024)

### Time Breakdown
- **Forward pass**: ~35% of step time (18ms)
- **Backward pass**: ~62% of step time (33ms)
- **Optimizer step**: ~3% (2ms)
- **Total step time**: ~53ms
- **Throughput**: ~76K tokens/sec (synthetic), ~31K tokens/sec (real data)

### Memory Breakdown
- Model weights: ~135MB
- Per-layer activations: ~640-650MB each (8 layers)
- Loss computation: +1.2GB (logits materialized)
- Peak during backward: ~5.4GB
- Final after optimizer: ~1.2GB

### Top GPU Bottlenecks (by time)
1. **Matrix multiply (aten::mm)**: 36% - cuBLAS optimized, no action needed
2. **aten::copy_**: 19% - Memory copies, possible layout inefficiency
3. **aten::add_/fill_**: 13% each - Elementwise ops, fusion candidates
4. **FusedSwiGLU**: 2.7% forward, 2.7% backward - Already fused
5. **LigerCrossEntropy**: 0.7% forward - Efficient chunked CE working

### Key Findings
1. **Dynamic routing saves 21% compute** - MoD+MoR gives significant speedup over static routing
2. **Layer 0 attention dominates** (25% of forward) - First block runs 40x longer than others (warm-up effect)
3. **Memory copies (12K+ calls)** - Excessive `aten::copy_` and `aten::to` operations
4. **Log softmax backward** is 27% of backward pass - From cross-entropy

### Optimization Opportunities
1. **Memory**: Gradient checkpointing already enabled (every 2 layers)
2. **Memory**: Chunked CE already active - working correctly
3. **Speed**: CUDA graphs fail due to dynamic routing - need `--static_routing_mode`
4. **Speed**: torch.compile active with `max-autotune-no-cudagraphs`
5. **Investigate**: High copy_ overhead - check for unnecessary tensor copies

## Profiling Scripts Added
- `diagnostics/profile_100m_training.py` - Basic profiling
- `diagnostics/profile_100m_detailed.py` - Module-level breakdown with hooks
- `diagnostics/profile_100m_trace.json` - Chrome trace (load in chrome://tracing)
- `diagnostics/profile_100m_real_training.json` - Actual training trace

## Known Issue
Coverage reporting (`--cov`) causes torch reimport error. Tests run fine without coverage flags.

## Next Steps
1. Investigate `aten::copy_` hotspot - may indicate tensor layout issues
2. Consider attention mechanism optimizations (layer 0 is slowest)
3. Profile at 500M/1B to see if patterns scale
