# HYDRA Diagnostics

Overview of diagnostic and benchmarking tools in `diagnostics/`.

## Categories

### Quick CLI Entry
- `__main__.py` — top-level runner for common diagnostics

### Benchmarks / Throughput
- `benchmark_hydra_models.py` — full model GPU benchmark. REQUIRES CUDA.
- `benchmark_performance.py` — comprehensive performance benchmarking suite. REQUIRES CUDA.
- `tall_skinny_bench.py` — tall/skinny matrix throughput analysis. REQUIRES CUDA.
- `throughput_sweep.py` — batch size and sequence length sweeps. REQUIRES CUDA.
- `seq_len_vram_sweep.py` — VRAM usage vs sequence length. REQUIRES CUDA.

### Kernel Benchmarks (New)
- `benchmark_backward_kernels.py` — benchmark fused backward kernels (SwiGLU, RMSNorm, QK-Norm). REQUIRES CUDA.
- `benchmark_swiglu_backward.py` — detailed SwiGLU backward kernel profiling. REQUIRES CUDA.
- `profile_kernels.py` — Triton kernel profiling with torch.profiler. REQUIRES CUDA.
- `profile_python_overhead.py` — measure Python-level overhead in training loop. REQUIRES CUDA.

### Model / Training Diagnostics
- `deep_diagnosis.py` — gradient flow and layer-level checks. GPU recommended.
- `model_health_check.py` — CPU-friendly model sanity checks.
- `eval_checkpoint.py` — load/evaluate a checkpoint (default device: CUDA).
- `eval_sanity_diagnostic.py` — short-run sanity checks.
- `loss_component_diagnostic.py` — analyze CE, aux, ponder loss components.
- `diagnose_learning.py` — learning rate and loss diagnostics.
- `routing_healthcheck.py` — MoD/MoR routing behavior checks.

### Scaling Analysis
- `scaling_analysis.py` — multi-scale model analysis with curve fitting and 4B predictions.
- `sweep_training_knobs.py` — hyperparameter sweep utilities.

### MoE-Specific Tools
- `moe_specialization.py` — expert specialization analysis (inference mode uses CUDA; weight-only mode runs on CPU).
- `moe_watchdog.py` — runtime monitor for checkpoints/logs (runs on CPU but benefits from GPU checkpoints).
- `moe_weight_balance.py` — checkpoint weight-norm analysis (CPU).

### Data Converters / Helpers
- `convert_nemotron_jsonl_to_pt.py` — convert Nemotron JSONL to .pt shards. Requires `transformers`.
- `convert_small_chat_to_pt.py` — convert small chat dataset to .pt shards. Requires `transformers`.
- `collect_pt_by_seq.py` — organize .pt shards by sequence length.

### External Kernel Checks
- `lightning_attn_healthcheck.py` — tests for lightning-attention kernels. REQUIRES CUDA and `lightning_attn`.

## Usage

### Quick Start
```bash
# Run model health check (CPU-friendly)
python -m diagnostics model_health_check

# Run benchmark (requires CUDA)
python diagnostics/benchmark_hydra_models.py --model_size 100M

# Profile backward kernels
python diagnostics/benchmark_backward_kernels.py

# Run scaling analysis
python diagnostics/scaling_analysis.py --variants 100M 250M 500M --steps 30
```

### Programmatic API
```python
import diagnostics

# List available tools
diagnostics.list_tools()

# Run a specific tool
diagnostics.run_tool("model_health_check")
```

## Notes
- Many diagnostics are heavy and intended to run on dedicated GPU nodes.
- Read the header at the top of each script for required dependencies.
- New kernel benchmarks help identify performance bottlenecks in fused operations.
