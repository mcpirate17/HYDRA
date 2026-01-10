# HYDRA Experimental Optimizations

This document describes the experimental optimizations available in HYDRA training, their expected performance impact, known failure modes, and validation status across model sizes.

## Overview

HYDRA uses a **SafeOptimizations** wrapper that automatically:
- Runs pretests before training to validate each optimization
- Monitors for anomalies (loss spikes, NaN gradients, throughput drops)
- Auto-disables failing optimizations with fallback to safe defaults
- Logs results for tracking which optimizations work on which configurations

## Quick Reference

| Optimization | CLI Flag | Default | Speedup | Status |
|-------------|----------|---------|---------|--------|
| Flash Attention 3 | `--experimental_fa3` | on | 15-25% | Requires FA3 package |
| CUDA Graphs | `--experimental_cuda_graphs` | on | 5-15% | Incompatible with MoD/MoR |
| Static Routing Mode | `--static_routing_mode` | off | Enables CUDA graphs | Stable |
| Blackwell Tuning | `--experimental_blackwell_tuning` | on | 10-15% | RTX 50xx only |
| Prefetch Threads | `--experimental_prefetch_threads` | 4 | 5-10% | Stable |
| FP8 Compute | `--experimental_fp8` | off | 20-40% | Experimental |
| Fused Backward Kernels | `--triton_kernels` | on | 10-20% | Stable |

---

## Flash Attention 3 (FA3)

### What It Does

Flash Attention 3 is the next-generation fused attention kernel optimized for Hopper (H100) and Blackwell (RTX 50xx) architectures. It provides:
- Reduced memory bandwidth through kernel fusion
- Better SM utilization with warp specialization
- Native FP8 support on Blackwell

### Expected Speedup

| GPU Architecture | Speedup vs FA2 | Notes |
|-----------------|----------------|-------|
| Blackwell (SM 12.0) | 20-25% | Best performance |
| Hopper (SM 9.0) | 15-20% | Well optimized |
| Ada (SM 8.9) | 5-10% | Limited benefit |
| Ampere (SM 8.0) | N/A | Not supported |

### CLI Flags

```bash
--experimental_fa3          # Enable FA3 (default: on)
--no-experimental_fa3       # Disable FA3
```

### Known Failure Modes

1. **Package Not Installed**: FA3 requires `flash-attn>=3.0`. Install with:
   ```bash
   pip install flash-attn --no-build-isolation
   ```

2. **Incompatible GPU**: FA3 requires Hopper+ architecture. Falls back to FA2 on older GPUs.

3. **Sequence Length Limits**: Very long sequences (>32K) may hit shared memory limits.

### Validation Status

| Model Size | 100M | 250M | 500M | 1B |
|-----------|------|------|------|-----|
| Blackwell | N/A* | N/A* | N/A* | N/A* |
| Hopper | N/A* | N/A* | N/A* | N/A* |

*FA3 package not installed in current environment. Install to enable.

---

## CUDA Graphs

### What It Does

CUDA Graphs capture a sequence of CUDA operations into a single graph that can be launched with minimal CPU overhead. Benefits:
- Reduced kernel launch latency
- Better GPU utilization
- Lower CPU overhead per step

### Expected Speedup

| Batch Size | Seq Length | Speedup | Notes |
|-----------|------------|---------|-------|
| Small (2-4) | Short (512) | 10-15% | Best improvement |
| Medium (8-16) | Medium (2K) | 5-10% | Good improvement |
| Large (32+) | Long (8K+) | 2-5% | GPU-bound anyway |

### CLI Flags

```bash
--experimental_cuda_graphs      # Enable CUDA graphs (default: on)
--no-experimental_cuda_graphs   # Disable CUDA graphs
--cuda_graphs_warmup 50         # Steps before graph capture
```

### Known Failure Modes

1. **Dynamic Control Flow**: CUDA graphs require static computation graphs. **HYDRA's MoD (Mixture-of-Depths) and MoR (Mixture-of-Recursions) use dynamic routing that is fundamentally incompatible with CUDA graphs.**

2. **Memory Allocation During Capture**: Any dynamic allocation during graph capture will fail.

3. **Shape Changes**: Changing input shapes requires graph recapture.

### Validation Status

| Model Size | 100M | 250M | 500M | 1B |
|-----------|------|------|------|-----|
| Blackwell | FAIL | FAIL | FAIL | FAIL |
| Hopper | FAIL | FAIL | FAIL | FAIL |
| Ada | FAIL | FAIL | FAIL | FAIL |

**Note**: CUDA graphs consistently fail on all HYDRA models due to MoD/MoR dynamic routing. This optimization is automatically disabled by the SafeOptimizations pretest system.

---

## Static Routing Mode

### What It Does

Static Routing Mode makes MoD and MoR routing deterministic by using soft routing weights instead of dynamic token selection. This enables:
- **CUDA Graph Compatibility**: Fixed computation graph allows graph capture
- **Consistent Profiling**: Same operations every step for accurate profiling
- **Deployment Simplicity**: No dynamic control flow in production

### How It Works

| Component | Dynamic Mode (default) | Static Mode |
|-----------|----------------------|-------------|
| **MoD Router** | Hard top-k token selection | Soft weighted sum over all tokens |
| **MoR Router** | Variable recursion depth | Fixed max depth with soft weights |
| **Compute** | Sparse (75% tokens) | Dense (100% tokens) |
| **Memory** | Lower | Higher |

### CLI Flags

```bash
--static_routing_mode       # Enable static routing
--no-static_routing_mode    # Disable (default)
```

### Expected Impact

| Metric | Dynamic | Static | Notes |
|--------|---------|--------|-------|
| Memory | Lower | +15-25% | All tokens processed |
| Throughput | Higher | Slightly lower | No sparse ops |
| CUDA Graphs | ❌ | ✅ | Main benefit |
| Training Quality | Baseline | Equivalent | Both converge well |

### When to Use

**Use Dynamic Mode (default)** for:
- Training efficiency (sparse compute savings)
- Memory-constrained environments
- Maximum throughput

**Use Static Mode** for:
- Deploying with CUDA graphs for inference
- Profiling with consistent operation counts
- Integration with systems requiring static computation graphs
- Debugging routing behavior

### Combining with CUDA Graphs

When static routing is enabled, CUDA graphs can be used:

```bash
python trainer.py \
    --model_size 500M \
    --static_routing_mode \
    --experimental_cuda_graphs \
    --cuda_graphs_warmup 50
```

### Validation Status

| Model Size | 100M | 250M | 500M | 1B |
|-----------|------|------|------|-----|
| Static Mode | PASS | PASS | PASS | PASS |
| + CUDA Graphs | PASS | PASS | PASS | PASS |

---

## Fused Backward Kernels

### What They Do

Custom Triton kernels that fuse multiple backward pass operations into single kernel launches:

| Kernel | Operations Fused | Kernel Reduction |
|--------|-----------------|------------------|
| `fused_swiglu_backward` | SiLU grad, mul grad, elementwise | ~12 → 1 |
| `fused_rms_norm_backward` | Norm grad, scale grad, bias grad | ~6 → 1 |
| `fused_qk_norm_backward` | L2 norm grad, scaling grad | ~8 → 1 |

### Expected Speedup

| Operation | Forward | Backward | Overall |
|-----------|---------|----------|---------|
| SwiGLU MLP | 1.3× | 2.0× | 1.5× |
| RMSNorm | 1.5× | 1.8× | 1.6× |
| QK-Norm | 1.5× | 1.7× | 1.5× |

### CLI Flags

Enabled automatically with `--triton_kernels`. Individual control via environment variables:

```bash
# Disable specific backward kernels
export HYDRA_DISABLE_FUSED_SWIGLU_BWD=1
export HYDRA_DISABLE_FUSED_RMS_NORM_BWD=1
export HYDRA_DISABLE_FUSED_QK_NORM_BWD=1
```

### Validation Status

| Model Size | 100M | 250M | 500M | 1B |
|-----------|------|------|------|-----|
| All GPUs | PASS | PASS | PASS | PASS |

All fused backward kernels are autograd-compatible and numerically validated.

---

## Blackwell Triton Tuning

### What It Does

Optimizes Triton kernel configurations specifically for NVIDIA Blackwell architecture (RTX 5090, etc.):
- Larger block sizes to utilize increased SM resources
- More warps per block for higher occupancy
- Optimized pipeline stages for Blackwell's memory subsystem

### Configuration

| Parameter | Blackwell | Default | Description |
|-----------|-----------|---------|-------------|
| `BLOCK_Q` | 128 | 64 | Query block size |
| `BLOCK_KV` | 64 | 64 | Key/Value block size |
| `num_warps` | 8 | 4 | Warps per thread block |
| `num_stages` | 3 | 2 | Software pipeline stages |

### Expected Speedup

| Operation | Speedup | Notes |
|-----------|---------|-------|
| Attention | 10-15% | Main benefit |
| SwiGLU MLP | 5-10% | Moderate benefit |
| LayerNorm | 2-5% | Minor benefit |

### CLI Flags

```bash
--experimental_blackwell_tuning     # Enable Blackwell tuning (default: on)
--no-experimental_blackwell_tuning  # Disable
--triton_block_q 128                # Query block size
--triton_block_kv 64                # KV block size
--triton_num_warps 8                # Warp count
```

### Known Failure Modes

1. **Non-Blackwell GPU**: Automatically disabled on non-Blackwell GPUs (no failure, just uses defaults).

2. **Shared Memory Limits**: Very large block sizes may exceed shared memory on some configurations.

### Validation Status

| Model Size | 100M | 250M | 500M | 1B |
|-----------|------|------|------|-----|
| Blackwell | PASS | PASS | PASS | PASS |
| Hopper | N/A | N/A | N/A | N/A |
| Ada | N/A | N/A | N/A | N/A |

---

## Prefetch Threads

### What It Does

Uses multiple Python threads to prefetch data batches while GPU is computing:
- Overlaps data loading with computation
- Reduces idle time waiting for batches
- Configurable number of threads and buffer size

### Expected Speedup

| Dataset | Threads | Speedup | Notes |
|---------|---------|---------|-------|
| Local SSD | 2-4 | 5-10% | Good overlap |
| NFS/Network | 4-8 | 10-20% | Hides latency |
| HuggingFace Hub | 4-8 | 15-25% | Network bound |

### CLI Flags

```bash
--experimental_prefetch_threads 4   # Number of threads (0=disabled)
--prefetch_buffer_size 8            # Batches to prefetch
```

### Known Failure Modes

1. **Memory Pressure**: Large buffer sizes with large batches can cause OOM.

2. **Thread Contention**: Too many threads on systems with few CPU cores can cause slowdown.

3. **GIL Contention**: Python GIL may limit benefit with many threads.

### Validation Status

| Model Size | 100M | 250M | 500M | 1B |
|-----------|------|------|------|-----|
| All GPUs | PASS | PASS | PASS | PASS |

This optimization is pure Python threading and works on all configurations.

---

## FP8 Compute

### What It Does

Uses 8-bit floating point for compute-intensive operations:
- Native FP8 tensor cores on Hopper/Blackwell
- 2x throughput vs FP16/BF16 tensor cores
- Two formats: E4M3 (more precision) and E5M2 (more range)

### Expected Speedup

| Operation | E4M3 Speedup | E5M2 Speedup | Notes |
|-----------|--------------|--------------|-------|
| Attention | 30-40% | 25-35% | Best benefit |
| MLP | 25-35% | 20-30% | Good benefit |
| Overall | 20-30% | 15-25% | With overhead |

### CLI Flags

```bash
--experimental_fp8              # Enable FP8 (default: off)
--no-experimental_fp8           # Disable FP8
--fp8_format e4m3               # Format: e4m3 or e5m2
```

### Known Failure Modes

1. **Quantization Error**: FP8 has limited precision. May cause:
   - Loss spikes during training
   - Gradient underflow/overflow
   - Convergence issues

2. **Not All Operations Supported**: Some operations fall back to higher precision.

3. **Scaling Required**: Proper loss scaling is critical for FP8 training.

### Validation Status

| Model Size | 100M | 250M | 500M | 1B |
|-----------|------|------|------|-----|
| Blackwell | EXPERIMENTAL | EXPERIMENTAL | EXPERIMENTAL | EXPERIMENTAL |
| Hopper | EXPERIMENTAL | EXPERIMENTAL | EXPERIMENTAL | EXPERIMENTAL |

**Warning**: FP8 is experimental and disabled by default. Enable only if you understand the trade-offs and are prepared for potential training instability.

---

## SafeOptimizations Monitoring

### Anomaly Detection

The SafeOptimizations wrapper monitors for:

| Anomaly Type | Threshold | Action |
|-------------|-----------|--------|
| Loss Spike | >2x rolling EMA | Log warning, count toward disable |
| NaN Gradient | Any NaN detected | Immediate disable candidate |
| Inf Gradient | Any Inf detected | Immediate disable candidate |
| Throughput Drop | <50% rolling EMA | Log warning, count toward disable |

### Safety Window

After enabling an optimization, it's monitored for 100 steps (configurable):
- 3 anomalies within the window triggers auto-disable
- After 100 clean steps, optimization graduates to "stable"

### CLI Flags

```bash
--experimental_safety_window 100            # Steps to monitor
--experimental_loss_spike_threshold 2.0     # Loss spike multiplier
--experimental_throughput_drop_threshold 0.5 # Throughput drop threshold
--experimental_skip_pretest                 # Skip pretests (risky)
```

---

## Pretest Hook System

### Automatic Pretests

The `PretestHook` automatically runs pretests when:
- Training config changes
- A new checkpoint is loaded
- Model size changes

### Log Files

Pretest results are logged to:
```
checkpoints/pretest_logs/
├── pretest_history.json          # Full history
├── pretest_500M_2026-01-10.json  # Per-run logs
└── pretest_1B_2026-01-10.json
```

### Querying History

```python
from hydra.training import PretestHook

hook = PretestHook(log_dir="checkpoints/pretest_logs")

# Get history for a model size
history = hook.get_history(model_size="500M")

# Get summary
summary = hook.get_summary()

# Print formatted summary
hook.print_summary()
```

---

## Recommended Configurations

### RTX 5090 (Blackwell) - Training

```bash
python trainer.py \
    --model_size 500M \
    --experimental_fa3 \              # Enable if FA3 installed
    --no-experimental_cuda_graphs \   # Disable (incompatible with dynamic MoD/MoR)
    --experimental_blackwell_tuning \ # Enable Blackwell optimizations
    --experimental_prefetch_threads 4 \
    --triton_kernels \                # Enable fused kernels
    --no-experimental_fp8             # Keep off unless testing
```

### RTX 5090 (Blackwell) - With CUDA Graphs

```bash
python trainer.py \
    --model_size 500M \
    --static_routing_mode \           # Required for CUDA graphs
    --experimental_cuda_graphs \      # Now compatible with static routing
    --cuda_graphs_warmup 50 \
    --experimental_blackwell_tuning \
    --triton_kernels
```

### H100 (Hopper)

```bash
python trainer.py \
    --model_size 1B \
    --experimental_fa3 \              # Enable if FA3 installed
    --no-experimental_cuda_graphs \   # Disable (incompatible with MoD/MoR)
    --no-experimental_blackwell_tuning \ # Not Blackwell
    --experimental_prefetch_threads 4 \
    --triton_kernels
```

### RTX 4090 (Ada)

```bash
python trainer.py \
    --model_size 500M \
    --no-experimental_fa3 \           # FA3 not beneficial on Ada
    --no-experimental_cuda_graphs \   # Disable (incompatible with MoD/MoR)
    --no-experimental_blackwell_tuning \
    --experimental_prefetch_threads 4 \
    --triton_kernels
```

### Inference Deployment (Any GPU)

```bash
# For maximum inference throughput with CUDA graphs
python trainer.py \
    --model_size 500M \
    --static_routing_mode \
    --experimental_cuda_graphs \
    --compile
```

---

## Troubleshooting

### Optimization Not Enabling

1. Check pretest logs:
   ```bash
   cat checkpoints/pretest_logs/pretest_history.json | jq '.[-1]'
   ```

2. Run pretests manually:
   ```python
   from hydra.training import run_pretests_for_checkpoint
   record = run_pretests_for_checkpoint("checkpoints/hydra_500m_final.pt", "500M")
   ```

### Training Slower Than Expected

1. Check which optimizations are active:
   ```python
   # In training logs, look for:
   # SafeOptimizations status: {'fa3': 'DISABLED', 'blackwell_tuning': 'ENABLED', ...}
   ```

2. Verify GPU utilization:
   ```bash
   nvidia-smi dmon -s u
   ```

### Anomalies Detected

1. Check SafeOptimizations anomaly log:
   ```python
   # Look for warnings like:
   # [SafeOpts] cuda_graphs anomaly: loss_spike (value=5.2345)
   ```

2. Disable the problematic optimization and retry:
   ```bash
   python trainer.py --no-experimental_cuda_graphs ...
   ```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-01-10 | Initial SafeOptimizations system |
| | | - FA3, CUDA Graphs, Blackwell Tuning, Prefetch, FP8 |
| | | - PretestHook with JSON logging |
| | | - W&B and TensorBoard integration |
