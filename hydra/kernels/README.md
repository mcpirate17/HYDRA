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
  - `fused_rope`, `fused_qk_norm`, `fused_rms_norm`, `fused_swiglu`
  - `fused_chunked_cross_entropy`
- `liger_integration.py`: LinkedIn Liger kernel wrappers (optional)
- `te_integration.py`: NVIDIA Transformer Engine FP8 integration (optional)
- `perf_config.py`: hardware detection and perf toggles

## Import safety

`hydra.kernels` is designed to be importable on CPU-only machines:
- Triton-backed imports are wrapped in `try/except`.
- Call sites should treat fused ops as optional and provide pure-torch fallbacks.

## Call-site map (partial)

- `hydra/layers/common.py` optionally uses fused norm/MLP/RoPE kernels.
- `hydra/attention/ccqa.py` optionally uses `fused_rope`.
- `hydra/training/*` uses `fused_chunked_cross_entropy` when enabled.
- `hydra/model/hybrid_attention_variants.py` calls into `hydra.attention.backends.lightning_attn3` for LLA3.
