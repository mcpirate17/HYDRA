# Shared Task Notes

## Current Status

Inline profiling of HYDRA 100M training complete. Copy overhead (`aten::copy_`) investigated and found to be well-handled by existing code.

## Key Profiling Results (100M Model)

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

### Top GPU Bottlenecks
1. **Matrix multiply (aten::mm)**: 36% - cuBLAS optimized, no action needed
2. **Dynamic routing** saves 21% compute vs static routing
3. **SDPA/Flash Attention** handles GQA efficiently

### Copy Overhead Investigation (RESOLVED)
The 19% `aten::copy_` time was investigated:
- **Non-contiguous Q/K slices**: Handled efficiently by `OptimizedConvSequence` (5.6% overhead for explicit pre-contiguous, so leaving as-is is faster)
- **Value shift operations**: Only 0.02ms per attention call (negligible)
- **Main sources**: Optimizer state updates and necessary backward pass copies

**Conclusion**: Copy overhead is well-optimized; no changes needed.

## Optimization Opportunities Already Implemented
- Gradient checkpointing (every 2 layers)
- Chunked cross-entropy (Liger fused CE)
- Fused QKV projection
- torch.compile with `max-autotune-no-cudagraphs`
- Flash Attention via SDPA

## Profiling Scripts
- `diagnostics/profile_100m_training.py` - Basic profiling
- `diagnostics/profile_100m_detailed.py` - Module-level breakdown
- `diagnostics/profile_copy_operations.py` - Copy hotspot investigation
- `diagnostics/benchmark_contiguity.py` - Contiguity impact
- `diagnostics/profile_value_shift.py` - Value shift alternatives
- `diagnostics/profile_100m_trace.json` - Chrome trace

## Next Steps
1. **Profile at larger scales** (500M/1B) to see if patterns hold
2. **Static routing mode** could enable CUDA graphs for further speedup
3. **Attention layer 0** runs 40x longer than others on first call (JIT warmup) - not a real bottleneck
