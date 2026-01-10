# hydra.kernels

This package contains low-level performance primitives ("kernels") and optional integrations.

## Taxonomy

- **Kernel**: a low-level fused primitive with a strict I/O contract, typically CUDA/Triton.
  Examples: fused RoPE, fused RMSNorm, fused SwiGLU, chunked cross-entropy.
- **Integration**: optional external kernel providers or libraries (Liger, Transformer Engine).
- **Backend**: higher-level modules (e.g. attention backends) that may call kernels.
  Backends live under `hydra/attention/`.

Rule of thumb:
- If it is *selected* based on capability (device/dtype/seq_len), it belongs in a backend/registry.
- If it is a *fused primitive* called by a backend, it belongs in `hydra.kernels`.

## Contents

- `losses/`: taxonomy wrappers for CE variants
- `mlp/`: taxonomy wrappers for MLP kernels
- `norms/`: taxonomy wrappers for norm kernels
- `rope/`: taxonomy wrappers for RoPE kernels
- `fused_ops.py`: HYDRA Triton kernels (guarded for CPU-only environments)
- `liger_integration.py`: LinkedIn Liger kernel wrappers (optional)
- `te_integration.py`: NVIDIA Transformer Engine FP8 integration (optional)
- `perf_config.py`: hardware detection and perf toggles

## Triton Kernels (`fused_ops.py`)

### Forward Kernels
| Kernel | Description | Speedup |
|--------|-------------|---------|
| `fused_rope` | Rotary Position Embedding | 2-3× |
| `fused_qk_norm` | L2 normalization + scaling for Q/K | 1.5-2× |
| `fused_swiglu` | SiLU(gate) * up activation | 1.3× |
| `fused_rms_norm` | RMS normalization | 1.5× |

### Fused Backward Kernels (New)
The backward kernels fuse multiple PyTorch ops into single Triton kernels, dramatically reducing kernel launch overhead:

| Kernel | Description | Kernel Reduction |
|--------|-------------|------------------|
| `fused_swiglu_backward` | SwiGLU gradient computation | ~12 → 1 kernels |
| `fused_rms_norm_backward` | RMSNorm gradient computation | ~6 → 1 kernels |
| `fused_qk_norm_backward` | QK-Norm gradient computation | ~8 → 1 kernels |

All backward kernels:
- Are autograd-compatible via `torch.autograd.Function`
- Support bf16 and fp32
- Are enabled by default when `--triton_kernels` is set

### Cross-Entropy Kernels
| Kernel | Description | Memory Savings |
|--------|-------------|----------------|
| `fused_chunked_cross_entropy` | Chunked CE to avoid full logits | ~60% |
| Liger FusedLinearCrossEntropy | Fused linear + chunked CE | ~80% |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HYDRA_DISABLE_TRITON` | `0` | Disable all Triton kernels |
| `HYDRA_ENABLE_FUSED_ROPE` | `1` | Enable fused RoPE |
| `HYDRA_ENABLE_FUSED_RMS_NORM` | `1` | Enable fused RMSNorm forward |
| `HYDRA_ENABLE_FUSED_RMS_NORM_BWD` | `1` | Enable fused RMSNorm backward |
| `HYDRA_ENABLE_FUSED_SWIGLU_BWD` | `1` | Enable fused SwiGLU backward |
| `HYDRA_ENABLE_FUSED_QK_NORM_BWD` | `1` | Enable fused QK-Norm backward |
| `HYDRA_ENABLE_LIGER_CE` | `1` | Enable Liger cross-entropy |

Use `HYDRA_DISABLE_*` variants to force-disable specific kernels.

## Import Safety

`hydra.kernels` is designed to be importable on CPU-only machines:
- Triton-backed imports are wrapped in `try/except`
- Call sites should treat fused ops as optional and provide pure-torch fallbacks
- Feature flags (`USE_FUSED_*`) are `False` when Triton is unavailable

## Call-Site Map

- `hydra/layers/common.py` optionally uses fused norm/MLP/RoPE kernels
- `hydra/attention/ccqa.py` optionally uses `fused_rope`
- `hydra/training/*` uses `fused_chunked_cross_entropy` when enabled
- `hydra/model/hybrid_attention_variants.py` calls into `hydra.attention.backends.lightning_attn3` for LLA3

## Benchmarking

Use the diagnostic scripts to benchmark kernels:
```bash
# Benchmark all backward kernels
python diagnostics/benchmark_backward_kernels.py

# Profile specific kernels
python diagnostics/profile_kernels.py

# Measure Python overhead
python diagnostics/profile_python_overhead.py
```
