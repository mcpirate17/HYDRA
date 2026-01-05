# HYDRA Diagnostics

Diagnostic and benchmarking tools for the HYDRA transformer.

## Categories

### Model Benchmarks
- `benchmark_hydra_models.py` — Benchmark full models (HydraBaseModel, HydraModel), memory usage. **REQUIRES CUDA** for full results.

### Training Diagnostics
- `deep_diagnosis.py` — Gradient flow and layer-level checks. GPU recommended.
- `model_health_check.py` — CPU-friendly model sanity checks.
- `eval_checkpoint.py` — Load/evaluate a checkpoint (default: CUDA).
- `eval_sanity_diagnostic.py`, `loss_component_diagnostic.py` — Short-run diagnostics for loss components.
- `diagnose_learning.py` — Parse training logs and detect issues.

### Throughput & Scaling
- `throughput_sweep.py`, `seq_len_vram_sweep.py` — Throughput/VRAM sweeps. **REQUIRES CUDA**.
- `tall_skinny_bench.py` — Tall/skinny matrix benchmarks.
- `scaling_analysis.py` — Model scaling analysis.
- `sweep_training_knobs.py` — Hyperparameter sweep utility.

### MoE-Specific Tools
- `moe_specialization.py` — Expert specialization analysis.
- `moe_watchdog.py` — Runtime monitor for checkpoints/logs.
- `moe_weight_balance.py` — Checkpoint weight-norm analysis (CPU).

### Routing Healthchecks
- `routing_healthcheck.py` — MoD/MoR routing validation.

### Data Converters
- `convert_nemotron_jsonl_to_pt.py`, `convert_small_chat_to_pt.py`, `collect_pt_by_seq.py` — Dataset converters. Requires `transformers`.

### External Kernel Checks
- `lightning_attn_healthcheck.py` — Tests for lightning-attention kernels. **REQUIRES CUDA** + `lightning_attn`.

## Attention-Level Benchmarks

For **attention kernel benchmarks** (CCGQA vs GQA vs Flash vs SDPA), see:

```bash
python -m hydra.attention.backends.ccgqa.benchmarks.benchmark_attention --save --plot
```

## Usage

### CLI
```bash
# Model benchmarks
python -m diagnostics.benchmark_hydra_models

# Run any diagnostic
python -m diagnostics.<tool_name>
```

### Python API
```python
from diagnostics import list_tools, run_tool

# List available tools
print(list_tools())

# Run a specific tool
run_tool("scaling_analysis")
```

## Output Directory

Diagnostic outputs (JSON files from training runs) are stored in `diagnostics/output/`.
This directory is gitignored.
