# Shared Task Notes

## Current Status

**100M profiling COMPLETE.** All bottlenecks identified and investigated. No actionable optimizations remaining at this scale.

## Summary of Findings

| Metric | Value |
|--------|-------|
| Step time | ~53ms |
| Forward | 35% (18ms) |
| Backward | 62% (33ms) |
| Optimizer | 3% (2ms) |
| Peak VRAM | ~5.4GB |
| Throughput | 31K tok/s (real data) |

**Key insight**: Matrix multiply dominates (36%), copy overhead is necessary (optimizer/backward), dynamic routing saves 21% compute. All existing optimizations are effective.

## Profiling Artifacts
- `diagnostics/profile_100m_training.py` - Basic profiling
- `diagnostics/profile_100m_detailed.py` - Module-level breakdown
- `diagnostics/profile_100m_trace.json` - Chrome trace (load in chrome://tracing)
- `diagnostics/profile_copy_operations.py` - Copy investigation (closed)

## What's Next (if continuing)

The 100M profiling goal is complete. Potential follow-up work:
1. **500M/1B profiling** - Verify patterns hold at larger scales
2. **Static routing + CUDA graphs** - Could reduce kernel launch overhead
3. **Real data pipeline** - 31K vs 76K tok/s gap suggests data loading worth investigating
