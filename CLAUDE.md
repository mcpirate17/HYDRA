# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HYDRA (Hybrid Dynamic Routing Architecture) is a transformer architecture combining:
- **CCGQA**: Compressed Convolutional Grouped Query Attention (arXiv:2510.04476)
- **MoD**: Mixture-of-Depths for token-level routing (arXiv:2404.02258)
- **MoR**: Mixture-of-Recursions for layer-level depth adaptation (arXiv:2507.10524)
- **MoE**: Optional Mixture-of-Experts for increased model capacity (sparse FFN routing)

## Environment Setup

**Required**: Always activate the venv before running any Python/pip commands:
```bash
source /home/tim/venvs/llm/bin/activate && <command>
```

Install in development mode:
```bash
source /home/tim/venvs/llm/bin/activate && pip install -e .
```

## Common Commands

### Training
```bash
# Quick test run (100M model)
source /home/tim/venvs/llm/bin/activate && python trainer.py --model_size 100M --mode testing --max_steps 1000

# Production training (1B model with all optimizations)
source /home/tim/venvs/llm/bin/activate && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python trainer.py \
    --model_size 1B --mode production --8bit_adam --checkpoint_every 1 \
    --triton_kernels --chunked_ce --dataset finefineweb-sequential

# With MoE (Mixture of Experts)
source /home/tim/venvs/llm/bin/activate && python trainer.py --model_size 500M --moe --moe_num_experts 8

# With static routing (enables CUDA graphs)
source /home/tim/venvs/llm/bin/activate && python trainer.py --model_size 500M --static_routing_mode --experimental_cuda_graphs
```

### Testing
```bash
# Run all tests
source /home/tim/venvs/llm/bin/activate && pytest

# Run specific test file
source /home/tim/venvs/llm/bin/activate && pytest tests/test_paper_compliance.py -v

# Run fast tests only (skip slow)
source /home/tim/venvs/llm/bin/activate && pytest -m "not slow"

# Run single test
source /home/tim/venvs/llm/bin/activate && pytest tests/test_paper_compliance.py::test_function_name -v
```

### Diagnostics
```bash
# Scaling analysis
source /home/tim/venvs/llm/bin/activate && python diagnostics/scaling_analysis.py --variants 100M 250M 500M

# MoD/MoR routing healthcheck
source /home/tim/venvs/llm/bin/activate && python diagnostics/routing_healthcheck.py

# Benchmark fused backward kernels
source /home/tim/venvs/llm/bin/activate && python diagnostics/benchmark_backward_kernels.py

# Profile kernel performance
source /home/tim/venvs/llm/bin/activate && python diagnostics/profile_kernels.py

# MoE expert specialization analysis
source /home/tim/venvs/llm/bin/activate && python diagnostics/moe_specialization.py --checkpoint checkpoints/hydra_500m_final.pt
```

## Architecture Overview

### Code Structure
- `hydra/` - Main package (all importable code)
  - `model/framework/` - Model wiring (MoD/MoR + factories)
  - `attention/backends/ccgqa/` - CCGQA attention implementation
  - `routing/` - MoD, MoR, and MoE routing modules
  - `layers/` - Shared layers (RMSNorm, SwiGLU, RoPE)
  - `kernels/` - Triton/CUDA kernels (fused forward + backward)
  - `training/` - Trainer, config, checkpointing, metrics, SafeOptimizations
  - `data/` - Data loading utilities
  - `optim/` - Optimizers and schedulers
- `trainer.py` - Training entrypoint (CLI)
- `tests/` - Test suite
- `diagnostics/` - Diagnostic and benchmark scripts (not discovered by pytest)
- `scripts/` - Utility scripts (query_training_db.py, build_training_db.py, etc.)
- `checkpoints/` - Training checkpoints (`hydra_{model_size}_*.pt`) and `training.db`

### Key Components

**Model Hierarchy**:
- `CCGQAMoDMoRModel` → Top-level model with embedding + MoD routing
- `CCGQAMoDBlockWrapper` → MoD wrapper around MoR blocks
- `CCGQAMoRBlock` → MoR block with recursion routing
- `CCGQABlock` → Base block with CCGQA attention + SwiGLU MLP
- `CCGQAAttention` → Core attention with 4x compression, convolutions, GQA

**Routing Flow**:
1. MoD router decides which tokens (75% capacity) get full computation
2. MoR router decides recursion depth per token (Gaussian soft routing)
3. CCGQA performs attention in 4x compressed latent space
4. (Optional) MoE routes tokens to specialized expert FFNs

**Static Routing Mode** (`--static_routing_mode`):
- Uses soft routing weights instead of dynamic token selection
- Enables CUDA graph compatibility
- Dense compute (all tokens processed with weighted contributions)

**Public API** (from `hydra/__init__.py`):
```python
from hydra import create_ccgqa_mod_mor_model, CCGQAAttention, CCGQABlock
```

## Development Rules

### torch.compile Constraints
Inside `forward()` or compiled regions:
- No `.item()`, `.tolist()`, `.cpu()`, `.numpy()`, or host sync
- No printing or logging
- No Python control flow based on tensor values
- Vectorize instead of Python loops creating tensors

### Performance Priorities
- Performance and memory efficiency over readability
- Prefer vectorized tensor operations over Python loops
- Prefer library primitives (Flash-Attn, xFormers, Triton) over custom logic
- Use in-place ops and `out=` arguments where safe
- Use `.view()`, `.expand()` over `.reshape()`, `.repeat()` to avoid copies

### Structural Rules
- New importable code goes in `hydra/` only
- Back-compat shims go in `hydra/model/ccgqa/` (keep thin)
- No new root-level shims
- Use canonical import paths; don't reach into backend internals

## Model Sizes

| Variant | Params | Dim | Blocks × Rec | Peak VRAM |
|---------|--------|-----|--------------|-----------|
| debug | ~33M | 512 | 2 × 2 | ~4GB |
| 100M | ~104M | 768 | 8 × 4 | ~14GB |
| 250M | ~198M | 1024 | 10 × 4 | ~18GB |
| 500M | ~426M | 1280 | 16 × 4 | ~22GB |
| 1B | ~973M | 1792 | 20 × 4 | ~29GB |

For 750M+ models: `--8bit_adam --checkpoint_every 1` are essential.

## Key Environment Variables

| Variable | Description |
|----------|-------------|
| `HYDRA_DATA_ROOT` | Parent directory for dataset shards |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | Reduce VRAM fragmentation |

**Triton Kernel Controls** (enabled by default with `--triton_kernels`):

| Variable | Description |
|----------|-------------|
| `HYDRA_DISABLE_TRITON` | Disable all Triton kernels globally |
| `HYDRA_ENABLE_FUSED_ROPE` | Enable fused RoPE kernel (default: 1) |
| `HYDRA_ENABLE_FUSED_RMS_NORM` | Enable fused RMSNorm forward (default: 1) |
| `HYDRA_ENABLE_FUSED_RMS_NORM_BWD` | Enable fused RMSNorm backward (default: 1) |
| `HYDRA_ENABLE_FUSED_SWIGLU_BWD` | Enable fused SwiGLU backward (default: 1) |
| `HYDRA_ENABLE_FUSED_QK_NORM_BWD` | Enable fused QK-Norm backward (default: 1) |
| `HYDRA_ENABLE_LIGER_CE` | Enable Liger fused cross-entropy (default: 1) |

Use `HYDRA_DISABLE_*` variants to force-disable specific kernels for debugging.

## Key CLI Flags

| Flag | Description |
|------|-------------|
| `--triton_kernels` | Enable all fused Triton kernels |
| `--chunked_ce` | Enable chunked cross-entropy (memory efficient) |
| `--static_routing_mode` | Use soft routing for CUDA graph compatibility |
| `--moe` | Enable Mixture of Experts |
| `--moe_num_experts N` | Number of expert FFNs (default: 4) |
| `--8bit_adam` | Use 8-bit Adam optimizer (essential for 750M+) |
| `--experimental_cuda_graphs` | Enable CUDA graphs (requires static routing) |

## Training Metrics Database

Training metrics are stored in SQLite (`checkpoints/training.db`) for cross-run analysis.

### Data Flow
1. **During training**: Metrics collected every 100 steps → `_diagnostics_data` list
2. **Periodically**: Saved to JSON (`checkpoints/diagnostics_{run_id}.json`)
3. **Training end**: `_update_training_db()` loads JSON into SQLite via `TrainingDB.load_diagnostics_json()`

### Database Tables
| Table | Purpose |
|-------|---------|
| `models` | Model metadata (id, params, architecture) |
| `runs` | Run summaries (start/end step, best loss, config) |
| `steps` | Per-step metrics with multi-scale EMA |
| `routing_mod` | MoD stats per layer (selected_frac, compute_savings) |
| `routing_mor` | MoR stats per layer (avg_depth, expected_depth) |
| `routing_moe` | MoE metrics (entropy, divergence, utilization) |
| `adaptive_lr` | LR scheduler state |

### Key Files
- `hydra/training/db.py` - `TrainingDB` class with schema and query methods
- `hydra/training/trainer.py:_log_layer_diagnostics()` - Collects routing stats every 100 steps
- `hydra/training/trainer.py:_update_training_db()` - Loads JSON to DB at training end
- `scripts/build_training_db.py` - Backfill DB from existing JSON files
- `scripts/query_training_db.py` - Query DB for analysis

### Query Commands
```bash
# Model stats
source /home/tim/venvs/llm/bin/activate && python scripts/query_training_db.py --model 500m --stats

# Loss milestones
source /home/tim/venvs/llm/bin/activate && python scripts/query_training_db.py --model 500m --milestones

# Multi-scale EMA
source /home/tim/venvs/llm/bin/activate && python scripts/query_training_db.py --model 500m --ema --start 100000

# Rebuild DB from JSON
source /home/tim/venvs/llm/bin/activate && python scripts/build_training_db.py --model-id 500m
```

### Multi-Scale EMA
| EMA | Alpha | Window | Use |
|-----|-------|--------|-----|
| `ema_short` | 0.99 | ~100 steps | Spike detection |
| `ema_medium` | 0.999 | ~1K steps | Session progress |
| `ema_long` | 0.9999 | ~10K steps | Cross-run trends |
