# HYDRA: Hybrid Dynamic Routing Architecture

> **A scalable transformer architecture combining Compressed Convolutional Grouped Query Attention (CCGQA), Mixture-of-Depths (MoD), and Mixture-of-Recursions (MoR) for efficient and adaptive language modeling.**

---

## 🎯 Overview

HYDRA is a modern transformer architecture that achieves **state-of-the-art efficiency** through three synergistic innovations:

| Component | Paper | Key Innovation |
|-----------|-------|----------------|
| **CCGQA** | [arXiv:2510.04476](https://arxiv.org/abs/2510.04476) | Attention in compressed latent space with convolutions |
| **MoD** | [arXiv:2404.02258](https://arxiv.org/abs/2404.02258) | Token-level dynamic computation routing |
| **MoR** | [arXiv:2507.10524](https://arxiv.org/abs/2507.10524) | Layer-level adaptive depth with recursion |

### Why "HYDRA"?

Like the mythical multi-headed Hydra, this architecture features **multiple routing heads** that dynamically adapt computation:
- **MoD heads** decide which tokens need full processing
- **MoR heads** decide how many recursive layers each position needs
- **CCGQA heads** perform efficient compressed attention

---

## 🏗️ Architecture

### High-Level Structure

```
Input Tokens
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│                    Token Embedding                          │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│              MoD Router (Token Selection)                   │
│     "Which tokens need full computation this layer?"        │
│     - Soft routing during training (all tokens, weighted)   │
│     - Hard top-k routing during inference (75% capacity)    │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│                   MoR Block (Recursive)                     │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              MoR Router (Depth Selection)             │  │
│  │  "How many recursive iterations for this position?"   │  │
│  │  - Gaussian soft routing during training              │  │
│  │  - Layer-aware: early layers 40%, late layers 80%     │  │
│  └───────────────────────────────────────────────────────┘  │
│                          │                                   │
│                          ▼                                   │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              CCGQA Attention Block                    │  │
│  │                                                       │  │
│  │  Input ──► Compress (4x) ──► Q,K,V Projections       │  │
│  │                │                                      │  │
│  │                ▼                                      │  │
│  │  ┌──────────────────────────────────────────────┐    │  │
│  │  │ Sequence Conv ──► Channel Conv ──► QK Mean   │    │  │
│  │  │ (causal, k=3)    (pointwise)     (coupling)  │    │  │
│  │  └──────────────────────────────────────────────┘    │  │
│  │                │                                      │  │
│  │                ▼                                      │  │
│  │  QK L2 Norm + Temperature ──► Attention ──► Value    │  │
│  │                │                                      │  │
│  │                ▼                                      │  │
│  │  Expand (4x) ──► Residual Add ──► FFN ──► Output     │  │
│  └───────────────────────────────────────────────────────┘  │
│                          │                                   │
│                    (Repeat × r recursions)                   │
└─────────────────────────────────────────────────────────────┘
     │
     ▼ (Repeat × n_blocks with MoD routing)
     │
┌─────────────────────────────────────────────────────────────┐
│                    Final LayerNorm                          │
│                    LM Head → Logits                         │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 Model Variants

HYDRA supports multiple scales optimized for different GPU memory budgets:

| Variant | Parameters | Dim | MoR Blocks × Rec | Eff Layers | GPU Memory | Status |
|---------|------------|-----|------------------|------------|------------|--------|
| **debug** | ~33M | 512 | 2 × 2 | 4 | ~4GB | ✅ Fast iteration |
| **50M** | ~50M | 512 | 8 × 3 | 24 | ~8GB | ✅ Deep & narrow |
| **100M** | ~104M | 768 | 8 × 4 | 32 | ~14GB | ✅ Validated |
| **250M** | ~198M | 1024 | 10 × 4 | 40 | ~18GB | ✅ Validated |
| **500M** | ~426M | 1280 | 16 × 4 | 64 | ~22GB | ✅ Validated |
| **750M** | ~665M | 1536 | 18 × 4 | 72 | ~26GB | ✅ Validated |
| **1B** | ~973M | 1792 | 20 × 4 | 80 | ~29GB | ✅ Validated |
| **1.5B** | ~1,369M | 2048 | 22 × 4 | 88 | ~36GB | ⚠️ 48GB+ GPU |

> **Note:** GPU memory is peak usage during training with 8-bit Adam + gradient checkpointing on RTX 5090 32GB.
>
> **50M "deep" config:** Designed for MoD/MoR curriculum validation. Narrow (dim=512) but deep (24 effective layers) to test dynamic routing effectiveness.

---

## 🔬 Attention Backend Comparison

HYDRA supports two attention backends:

| Backend | Type | Complexity | Best For |
|---------|------|------------|----------|
| **LA3** | Linear Attention | O(n) | Long sequences, memory efficiency |
| **CCGQA** | Compressed GQA | O(n²) compressed | Better convergence, shorter sequences |

### Benchmark Results (December 2024)

Training comparison on `debug` model (500 steps) and `50M` model (1000 steps):

| Model | Backend | Initial Loss | Final Loss | Best Loss | Reduction |
|-------|---------|-------------|------------|-----------|-----------|
| debug | LA3 | 10.94 | 6.37 | 6.22 | 41.8% |
| debug | **CCGQA** | 11.29 | **5.88** | **5.69** | **47.9%** |
| 50M | LA3 | 11.01 | 6.30 | 6.18 | 42.8% |
| 50M | **CCGQA** | 12.69 | **5.12** | **4.95** | **59.7%** |

**Key Finding:** CCGQA consistently outperforms LA3 in convergence speed and final loss quality. The 50M deep model with CCGQA achieved a 1.18-point lower final loss (5.12 vs 6.30).

```bash
# Run with CCGQA attention (recommended for < 4K sequence length)
python trainer.py --model_size 50M --attention ccgqa --max_steps 5000

# Run with LA3 attention (for long sequences or memory constraints)  
python trainer.py --model_size 50M --attention la3 --max_steps 5000
```

### Block Architecture

Each **MoR Block** contains the following layers:

```
┌─────────────────────────────────────────────────────────────────┐
│  MoR Block (repeated n_mor_blocks times)                        │
├─────────────────────────────────────────────────────────────────┤
│  1. CCGQA Attention                                             │
│     ├── RMSNorm (pre-norm)                                      │
│     ├── Q/K/V Linear projections (dim → n_heads × head_dim)    │
│     ├── RoPE positional embeddings                              │
│     ├── Grouped Query Attention (4:1 to 8:1 GQA ratio)         │
│     ├── Context Compression (for long sequences)                │
│     └── Output Linear projection                                │
│                                                                 │
│  2. SwiGLU MLP                                                  │
│     ├── RMSNorm (pre-norm)                                      │
│     ├── Gate Linear (dim → hidden_dim)                         │
│     ├── Up Linear (dim → hidden_dim)                           │
│     ├── SiLU activation × gate                                  │
│     └── Down Linear (hidden_dim → dim)                         │
│                                                                 │
│  3. MoD Router (Mixture of Depths)                              │
│     └── Token-level routing (75% capacity, skip unimportant)    │
│                                                                 │
│  4. MoR Router (Mixture of Recursions)                          │
│     ├── Recursion embedding (one per recursion depth)           │
│     └── Decides which tokens need more processing               │
└─────────────────────────────────────────────────────────────────┘
```

**Effective Layers** = `n_mor_blocks × recursions` (weights are shared across recursions within each block)

---

## 🚀 Training on RTX 5090 (32GB)

### Memory Requirements by Model Size

Benchmarked on RTX 5090 32GB with 8-bit Adam + gradient checkpointing (every layer):

| Model | Actual Params | Dim | Blocks × Rec | Eff Layers | Batch | Accum | Peak Mem | Throughput |
|-------|---------------|-----|--------------|------------|-------|-------|----------|------------|
| **100M** | ~104M | 768 | 8 × 4 | 32 | 32 | 4 | ~14GB | ~30K tok/s |
| **250M** | ~198M | 1024 | 10 × 4 | 40 | 24 | 5 | ~18GB | ~20K tok/s |
| **500M** | ~426M | 1280 | 16 × 4 | 64 | 8 | 8 | ~22GB | ~12K tok/s |
| **750M** | ~665M | 1536 | 18 × 4 | 72 | 4 | 16 | ~26GB | ~8K tok/s |
| **1B** | ~973M | 1792 | 20 × 4 | 80 | 2 | 30 | ~29GB | ~5K tok/s |
| **1.5B** | ~1,369M | 2048 | 22 × 4 | 88 | 1 | 60 | ~36GB | ⚠️ 48GB+ |

> ⚠️ **1B Model Warning:** `batch_size=3` peaks at ~32GB (borderline on 32GB GPU), `batch_size=4+` will OOM!
> 
> ⚠️ **1.5B Model:** Requires 48GB+ VRAM (A6000, RTX 6000, or multi-GPU setup)

### Required Flags for Large Models (750M+)

```bash
--8bit_adam              # Essential - saves ~75% optimizer memory
--checkpoint_every 1     # Gradient checkpointing on every layer
```

### Training Commands

**100M Model (quick testing):**
```bash
python trainer.py \
  --model_size 100M \
  --mode testing \
  --max_steps 1000
```

**1B Model (production):**
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python trainer.py \
    --model_size 1B \
    --mode production \
    --8bit_adam \
    --checkpoint_every 1 \
    --adaptive_lr \
    --triton_kernels \
    --chunked_ce \
    --dataset finefineweb-sequential \
    --seed 42
```

> **Note:** Batch size and gradient accumulation are automatically set based on `--model_size`. Override with `--batch_size` and `--grad_accum` if needed.

---

## 🗄️ Local Dataset Mounts (recommended)

If you store converted `.pt` shards on an external drive, avoid relying on GUI automount paths like `/media/<user>/<Drive Name>/...` (they can change, or be unavailable in non-GUI sessions).

HYDRA will use these environment variables when present:
- `HYDRA_DATA_ROOT`: A stable parent directory that contains `hydra_small_chat_pt/`, `hydra_nemotron_pt/`, and optionally `hf_finefineweb/`.
- `HYDRA_SMALL_CHAT_PT_DIR`: Explicit path to `hydra_small_chat_pt`.
- `HYDRA_NEMOTRON_PT_DIR`: Explicit path to `hydra_nemotron_pt`.

### Option A: Mount the drive to a stable path (best)

1) Find the drive UUID + filesystem type:
```bash
lsblk -f
```

2) Create a mount point (example):
```bash
sudo mkdir -p /mnt/hydra_data
```

3) Add an `/etc/fstab` entry using the UUID (edit with `sudo nano /etc/fstab`). Examples:

- **ext4**:
```text
UUID=<YOUR_UUID>  /mnt/hydra_data  ext4  defaults,nofail  0  2
```

- **exFAT** (common for portable SSDs):
```text
UUID=<YOUR_UUID>  /mnt/hydra_data  exfat  defaults,nofail,uid=1000,gid=1000,umask=022  0  0
```

4) Mount it:
```bash
sudo mount -a
```

5) Point HYDRA at the stable location:
```bash
export HYDRA_DATA_ROOT=/mnt/hydra_data
```

You can put the `export` into `~/.bashrc` or `~/.profile` to make it permanent.

### Option B: Symlink (quick, but less robust)

If you don’t want to edit `fstab`, you can symlink the expected default paths to your current mount:
```bash
sudo mkdir -p /mnt/nvme0
sudo ln -s "/media/<user>/<Drive Name>/hydra_small_chat_pt" /mnt/nvme0/hydra_small_chat_pt
sudo ln -s "/media/<user>/<Drive Name>/hydra_nemotron_pt" /mnt/nvme0/hydra_nemotron_pt
```

---

## ⚡ Optional FP8 (Transformer Engine)

HYDRA includes optional integration with NVIDIA Transformer Engine (TE) to run **FP8** for *Linear projections* when available.

- Default is **OFF** to avoid surprising numeric changes and because TE requires extra dependencies and Hopper+ GPUs.
- When enabled and supported, HYDRA will use TE’s `fp8_autocast` + `TELinear` for the LA3 adapter’s `q/k/v/o` projections.

Requirements:
- Hopper+ GPU (sm_90+) and CUDA 12+
- `pip install transformer-engine[pytorch]`

Enable for the LA3 adapter (opt-in):
- Set `te_fp8_projections=True` via the attention kwargs path (see [hydra/model/hybrid_attention_variants.py](hydra/model/hybrid_attention_variants.py)).

Note:
- This only affects projection layers; the LA3 Triton attention kernels still run in fp16/bf16.

## 🧭 Hybrid Attention Routing

The MoD+MoR model uses a fixed layerwise attention pattern by default:
- 3× `lla3` blocks then 1× `ccqa` block (repeat every 4 blocks)

You can disable the hybrid routing and force all blocks to use compressed attention:
- Set `hybrid_attention=False` when constructing `CCGQAMoDMoRModel` (then all blocks use `ccqa`).

### Stepped Sequence Training (Advanced)

For 1B model with longer context, use stepped sequence scheduling:

| Phase | Seq Len | Batch | Accum | Memory | Tokens/Step |
|-------|---------|-------|-------|--------|-------------|
| **1** | 512 | 2 | 30 | 28.5GB | 30,720 |
| **2** | 1024 | 1 | 32 | ~25GB | 32,768 |
| **3** | 2048 | 1 | 32 | ~28GB | 65,536 |

---

## 🔬 Paper Compliance

### CCGQA (arXiv:2510.04476)

| Requirement | Implementation | Status |
|-------------|----------------|--------|
| 4× compression factor | `compression_factor=4` | ✅ |
| Sequence convolutions | Causal 1D conv, kernel=3 | ✅ |
| Channel convolutions | Pointwise 1×1 conv | ✅ |
| QK-mean coupling | Mean shared before/after conv | ✅ |
| QK L2 normalization | With learnable temperature | ✅ |
| GQA head sharing | `n_kv_heads < n_heads` | ✅ |
| Value shift | Half heads see previous token | ✅ |

### MoD (arXiv:2404.02258)

| Requirement | Implementation | Status |
|-------------|----------------|--------|
| Router architecture | Linear projection + sigmoid | ✅ |
| Soft routing (training) | Weighted sum by router probs | ✅ |
| Hard routing (inference) | Top-k selection, k=capacity | ✅ |
| 75% capacity target | `capacity_ratio=0.75` | ✅ |
| Auxiliary loss | BCE to maintain capacity | ✅ |
| Auto-scaling aux weight | `0.01 * (L/32) * √(d/768)` | ✅ |

### MoR (arXiv:2507.10524)

| Requirement | Implementation | Status |
|-------------|----------------|--------|
| Gaussian soft routing | `N(μ, σ)` weighting over depths | ✅ |
| Layer-aware capacity | Early: 40%, Late: 80% | ✅ |
| Recursion embeddings | Additive depth embeddings | ✅ |
| Ponder loss | Encourages early stopping | ✅ |
| Depth histogram | Per-layer depth distribution | ✅ |

---

## 📁 Project Structure

```
HYDRA/
├── hydra/                    # Main package
│   ├── __init__.py          # Package exports
│   ├── logging.py           # Logging utilities
│   ├── utils.py             # Common utilities
│   ├── attention/           # Attention backends
│   │   ├── backends/        
│   │   │   ├── ccgqa/       # Compressed Convolutional GQA
│   │   │   └── lightning_attn3/  # LA3 linear attention
│   │   └── factory.py       # Attention factory
│   ├── data/                # Data loading utilities
│   ├── kernels/             # Triton/CUDA kernels
│   ├── layers/              # Core layer implementations
│   ├── model/               # Model components
│   │   ├── framework/       # Model wiring (MoD/MoR + factories)
│   │   └── ccgqa/           # Back-compat shims
│   ├── optim/               # Optimizers and schedulers
│   ├── routing/             # Routing modules (MoD, MoR)
│   └── training/            # Training infrastructure
│       ├── trainer.py       # Main trainer class
│       ├── config.py        # Configuration dataclasses
│       ├── checkpointing.py # Checkpoint management
│       └── metrics.py       # Training metrics
├── trainer.py               # Training entrypoint (CLI)
├── scripts/                 # Utility scripts
│   ├── compare_mod_mor_effectiveness.py  # MoD/MoR comparison
│   └── ...
├── tests/                   # Test suite (305 tests)
│   └── test_paper_compliance.py  # Paper compliance tests
├── diagnostics/             # Diagnostic and benchmarking tools
│   ├── mod_mor_routing_healthcheck.py  # Routing health checks
│   ├── scaling_analysis.py  # Multi-scale analysis
│   └── ...
├── configs/                 # Model configurations
│   └── variants.yaml        # Model variant definitions
├── reports/                 # Generated analysis reports
├── checkpoints/             # Training checkpoints (hydra_{model_size}_*.pt)
├── README.md                # This file
├── pytest.ini               # Test configuration
└── requirements.txt         # Dependencies
```

---

## 🎯 MoD/MoR Curriculum Training

HYDRA uses a **curriculum approach** for MoD and MoR to ensure stable training:

### MoD (Mixture of Depths) Curriculum

| Phase | Step Range | Behavior |
|-------|-----------|----------|
| **Warmup** | 0 → 10% | MoD **disabled** (dense MLP, all tokens processed) |
| **Loss Gate** | 10% → force% | MoD enables when CE loss EMA < 5.0 |
| **Force Enable** | 15-20% | MoD **forced on** regardless of loss |
| **Active** | 20% → 100% | MoD active, ~50% compute savings |

### MoR (Mixture of Recursions) Curriculum

| Phase | Step Range | Behavior |
|-------|-----------|----------|
| **Fixed Depth** | 0 → 20-30% | All tokens use maximum recursion depth |
| **Ramp Up** | 20% → 30% | Gradually enable adaptive depth routing |
| **Full Adaptive** | 30% → 100% | MoR decides recursion depth per-token |

### Running Curriculum Experiments

```bash
# Standard curriculum (MoD@10%, MoR@30%)
python trainer.py --model_size 50M --attention ccgqa --max_steps 5000

# Override curriculum timing (for short experiments)
python trainer.py --model_size 50M --attention ccgqa --max_steps 1000 \
    --no_short_run_override \
    --mod_enable_pct 0.10 --mod_force_enable_pct 0.15 \
    --mor_enable_pct 0.20

# Disable MoD/MoR (vanilla baseline)
python trainer.py --model_size 50M --attention ccgqa --max_steps 5000 \
    --mod_off --mor_off
```

### MoD/MoR Effectiveness Comparison

Run the comprehensive comparison script to evaluate routing effectiveness:

```bash
# Full comparison: vanilla vs MoD-only vs MoR-only vs full routing
python scripts/compare_mod_mor_effectiveness.py --model_size 50M --max_steps 5000

# Quick test (1000 steps)
python scripts/compare_mod_mor_effectiveness.py --model_size 50M --max_steps 1000 --quick
```

This generates:
- **Loss curves** comparing all configurations
- **MoD compute savings** per layer
- **MoR depth histograms** (token distribution across recursion levels)
- **Summary report** with key findings

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourname/hydra.git
cd hydra

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Basic Usage

```python
from hydra import create_ccgqa_mod_mor_model

# Create a 100M model
model = create_ccgqa_mod_mor_model(
    vocab_size=32000,
    dim=768,
    n_mor_blocks=8,
    recursions=4,
    n_heads=12,
    n_kv_heads=3,
    compression_factor=4,
    capacity_ratio=0.75,
)

# Forward pass
input_ids = torch.randint(0, 32000, (1, 512))
outputs = model(input_ids)

logits = outputs["logits"]           # [batch, seq, vocab]
aux_loss = outputs["aux_loss"]       # MoD capacity loss
ponder_loss = outputs["ponder_loss"] # MoR depth loss

# Total loss for training
total_loss = ce_loss + 0.01 * aux_loss + 0.01 * ponder_loss
```

### Run Tests

```bash
# Run full compliance test suite (64 tests)
pytest tests/test_paper_compliance.py -v

# Run fast tests only
pytest tests/test_paper_compliance.py -v -m "not slow"
```

### Run Diagnostics

```bash
# Run scaling analysis across all variants
python diagnostics/scaling_analysis.py \
    --variants 100M 250M 500M 750M 900M 1B 1.5B \
    --steps 30 \
    --plot \
    --predict-4b \
    --output reports/scaling_analysis_results.json
```

### Attention Architecture (MoR blocks)

HYDRA uses **Lightning-Attention 3** (lla3) for efficient O(n) linear attention combined with **CCGQA** (Compressed Convolutional Grouped Query Attention).

**Performance**: 7.52x faster than PyTorch SDPA at N=8192, 27% less memory. See [hydra/attention/backends/lightning_attn3/README.md](hydra/attention/backends/lightning_attn3/README.md) for benchmarks.

**Default pattern**: 3× Lightning-Attention blocks + 1× CCGQA block per MoR macro-block

```bash
# This is the default (no env var needed):
python diagnostics/tall_skinny_bench.py --device cuda --preset 100m --steps 1

# Explicitly set the named pattern:
HYDRA_MOR_ATTENTION_PATTERN_NAME='lla3x3+ccqa' python diagnostics/tall_skinny_bench.py --device cuda --preset 100m --steps 1

# Or define as literal token sequence:
HYDRA_MOR_ATTENTION_PATTERN='lla3,lla3,lla3,ccqa' python diagnostics/tall_skinny_bench.py --device cuda --preset 100m --steps 1
```

**Requirements**:
- `lla3` requires CUDA (the Triton kernels are CUDA-based, optimized for Blackwell/SM12).
- HYDRA_MOR_ATTENTION_OVERRIDE still exists and overrides all blocks if set.

---

## 📈 Scaling Analysis

The architecture has been validated across 7 model scales with curve fitting to predict 4B behavior:

### Curve Fitting Results

| Metric | Best Fit | R² | 4B Prediction |
|--------|----------|-----|---------------|
| `aux_loss_weight` | Polynomial (deg 2) | 0.990 | ~0.102 |
| `mod_prob` | Polynomial (deg 2) | 0.901 | Stable ~0.75 |
| `mor_depth` | Constant | 1.000 | 1.0 |
| Compute time | Polynomial (deg 2) | 0.951 | ~47s/step |

### Auto-Scaling Formula

For large models, the auxiliary loss weight automatically scales:

```python
aux_loss_weight = 0.01 * (effective_layers / 32) * sqrt(dim / 768)
```

This ensures MoD capacity remains at 75% even as model size increases.

---

## 🧪 Testing Philosophy

HYDRA follows a rigorous testing philosophy:

1. **Paper Compliance**: Every architectural claim is validated against the source papers
2. **Scale Invariance**: Tests run at multiple scales (100M → 1.5B)
3. **Repeatability**: All tests use fixed seeds and are deterministic
4. **Regression Prevention**: Scaling analysis detects drift in hyperparameters

---

## ⚙️ Performance Optimizations

### Kernel-Level Optimizations

**Liger Kernels (BF16 fused operations, auto-enabled)**
- **LigerRMSNorm**: ~30% memory savings, 1.5-2× faster
- **LigerSwiGLU**: ~1.3× faster, avoids intermediate materialization
- **LigerCrossEntropy**: ~60% memory savings, 2× faster
- **LigerFusedLinearCrossEntropy**: ~80% output layer savings, never materializes full logits

**Triton Custom Kernels (opt-in via `--triton_kernels`)**
- **fused_qk_norm**: Fused L2 normalization for Q/K (1.5-2× faster, autograd-compatible)
- **fused_swiglu**: Fused SiLU activation (1.3× faster, autograd-compatible)
- **fused_rope**: Fused RoPE (2-3× faster, opt-in via `HYDRA_ENABLE_FUSED_ROPE=1`)
- **fused_rms_norm**: Fused RMSNorm (opt-in via `HYDRA_ENABLE_FUSED_RMS_NORM=1`)

**Flash Attention**
- Flash Attention 2/3 auto-detected and enabled
- Memory-efficient attention (no QK^T materialization)
- FP8 support on Flash Attention 3

### Training Infrastructure

**torch.compile**: Graph optimization with `max-autotune-no-cudagraphs` mode
**Mixed Precision**: BF16 forward/backward with FP32 master weights
**Memory Optimization**:
- Gradient checkpointing (every N layers, default N=2)
- 8-bit Adam (~75% optimizer memory savings, essential for 750M+)
- Chunked cross-entropy (4096 tokens per chunk)

**Data Loading**:
- Multi-worker parallel loading (4-8× faster)
- Background prefetching (2× prefetch factor)
- Rust-based fast tokenizers (3-10× faster)
- HF Transfer protocol (5-10× faster downloads)

**Learning Rate**:
- WSD (Warmup-Stable-Decay) scheduler with adaptive LR
- Auto-trigger cooldown on loss spikes
- Stochastic Weight Averaging (last 25% of training)
- Batch filtering (skip corrupted/noisy batches)

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HYDRA_ENABLE_FUSED_ROPE` | `0` | Enable fused RoPE kernel (opt-in due to GPU compatibility) |
| `HYDRA_ENABLE_FUSED_RMS_NORM` | `0` | Enable fused RMSNorm kernel (opt-in due to gradient concerns) |
| `HF_HUB_ENABLE_HF_TRANSFER` | `1` | Enable fast HuggingFace transfers (auto-enabled) |
| `HYDRA_MOR_ATTENTION_PATTERN_NAME` | `lla3x3+ccqa` | Attention pattern for MoR blocks (CUDA only) |

```bash
# Enable all fused kernels (experimental)
export HYDRA_ENABLE_FUSED_ROPE=1
export HYDRA_ENABLE_FUSED_RMS_NORM=1
python trainer.py --triton_kernels ...
```

---

## 📚 References

```bibtex
@article{ccgqa2024,
  title={Compressed Convolutional Attention: Efficient Attention in a Compressed Latent Space},
  author={...},
  journal={arXiv preprint arXiv:2510.04476},
  year={2024}
}

@article{mod2024,
  title={Mixture-of-Depths: Dynamically allocating compute in transformer-based language models},
  author={Raposo, David and others},
  journal={arXiv preprint arXiv:2404.02258},
  year={2024}
}

@article{mor2025,
  title={Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation},
  author={...},
  journal={arXiv preprint arXiv:2507.10524},
  year={2025}
}
```

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

---

## 🤝 Contributing

Contributions welcome!

---

<p align="center">
  <strong>HYDRA</strong> - Multi-headed efficiency for modern transformers 🐉
</p>
