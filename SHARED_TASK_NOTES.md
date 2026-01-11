# Shared Task Notes

## Current Status

**100M profiling COMPLETE.** Data loading investigation RE-CONFIRMED on 2026-01-11. No throughput gap exists.

## Latest Verified Results (2026-01-11)

| Metric | Synthetic | Real Data |
|--------|-----------|-----------|
| Throughput | 41.3K tok/s | 41.7K tok/s |
| Step time | 198ms | 197ms |
| Data fetch | 0.08ms (0.0%) | 0.45ms (0.2%) |
| Forward | 72.5ms (36.5%) | 71.8ms (36.5%) |
| Backward | 125.6ms (63.3%) | 124.2ms (63.2%) |

**Conclusion**: The "76K vs 31K tok/s gap" does NOT exist. Real data performs identically to synthetic.

## What's Next

The 100M profiling goal is fully complete. Suggested follow-up:
1. **500M/1B profiling** - Verify patterns hold at larger scales
2. **Static routing + CUDA graphs** - Could reduce kernel launch overhead
3. **Production training runs** - Start actual training on finefineweb
