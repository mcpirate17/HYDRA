# Shared Task Notes

## Current Status

**100M profiling COMPLETE.** All bottlenecks identified and investigated. No actionable optimizations remaining at this scale.

## Summary of Findings

| Metric | Value |
|--------|-------|
| Step time | ~53ms (per microbatch) |
| Forward | 35% (18ms) |
| Backward | 62% (33ms) |
| Optimizer | 3% (2ms) |
| Peak VRAM | ~5.4GB |
| Throughput | ~35K tok/s |

**Key insight**: Matrix multiply dominates (36%), copy overhead is necessary (optimizer/backward), dynamic routing saves 21% compute. All existing optimizations are effective.

## Data Loading Investigation (CLOSED)

**Finding**: Data loading is NOT a bottleneck. The "76K vs 31K tok/s gap" was investigated and found to be non-existent:

| Dataset | Avg Throughput | Peak Throughput |
|---------|----------------|-----------------|
| Synthetic | 35.7K tok/s | 39.7K tok/s |
| finefineweb-local | 35.3K tok/s | 39.8K tok/s |

**Details**:
- Data loading accounts for only 0.2% of step time (~0.4ms per batch)
- Background prefetch thread keeps buffer well-stocked (~40K-90K tokens)
- Tokenizer uses batched encoding with Rust-based fast tokenizer (52K texts/s)
- No measurable difference between synthetic and real data throughput

## Profiling Artifacts
- `diagnostics/profile_100m_training.py` - Basic profiling
- `diagnostics/profile_100m_detailed.py` - Module-level breakdown
- `diagnostics/profile_100m_trace.json` - Chrome trace (load in chrome://tracing)
- `diagnostics/profile_copy_operations.py` - Copy investigation (closed)
- `diagnostics/profile_data_loading.py` - Data loading profiler
- `diagnostics/profile_100m_dataflow.py` - Training dataflow profiler

## What's Next (if continuing)

The 100M profiling goal is complete. Potential follow-up work:
1. **500M/1B profiling** - Verify patterns hold at larger scales
2. **Static routing + CUDA graphs** - Could reduce kernel launch overhead
