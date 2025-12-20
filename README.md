# HYDRA: Hybrid Dynamic Routing Architecture

<p align="center">
  <img src="docs/hydra_architecture.png" alt="HYDRA Architecture" width="600">
</p>

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
| **100M** | ~104M | 768 | 8 × 4 | 32 | ~14GB | ✅ Validated |
| **250M** | ~198M | 1024 | 10 × 4 | 40 | ~18GB | ✅ Validated |
| **500M** | ~426M | 1280 | 16 × 4 | 64 | ~22GB | ✅ Validated |
| **750M** | ~665M | 1536 | 18 × 4 | 72 | ~26GB | ✅ Validated |
| **1B** | ~973M | 1792 | 20 × 4 | 80 | ~29GB | ✅ Validated |
| **1.5B** | ~1,369M | 2048 | 22 × 4 | 88 | ~36GB | ⚠️ 48GB+ GPU |

> **Note:** GPU memory is peak usage during training with 8-bit Adam + gradient checkpointing on RTX 5090 32GB.

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
│   ├── model/               # Core model components
│   │   ├── __init__.py
│   │   └── ccgqa.py         # CCGQA + MoD + MoR implementation
│   └── routing/             # Routing modules (MoD, MoR)
│       └── __init__.py
├── tests/                   # Test suite
│   ├── __init__.py
│   └── test_paper_compliance.py  # 64 compliance tests
├── diagnostics/             # Scaling and compliance tools
│   ├── __init__.py
│   ├── scaling_analysis.py  # Multi-scale curve fitting
│   ├── run_variant_diagnostics.py
│   └── deep_diagnosis.py
├── configs/                 # Model configurations
│   └── variants.yaml        # Model variant definitions
├── reports/                 # Generated analysis reports
│   ├── scaling_analysis.png
│   ├── scaling_summary_table.png
│   └── scaling_analysis_results.json
├── docs/                    # Documentation
│   └── ARCHITECTURE.md      # This file
├── README.md               # Project overview
├── pytest.ini              # Test configuration
└── requirements.txt        # Dependencies
```

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

HYDRA uses **Lightning-Attention 2** (lla2) for efficient scaled-dot-product attention combined with **CCGQA** (Compressed Convolutional Grouped Query Attention).

**Default pattern**: 3× Lightning-Attention blocks + 1× CCGQA block per MoR macro-block

```bash
# This is the default (no env var needed):
python diagnostics/tall_skinny_bench.py --device cuda --preset 100m --steps 1

# Explicitly set the named pattern:
HYDRA_MOR_ATTENTION_PATTERN_NAME='lla2x3+ccqa' python diagnostics/tall_skinny_bench.py --device cuda --preset 100m --steps 1

# Or define as literal token sequence:
HYDRA_MOR_ATTENTION_PATTERN='lla2,lla2,lla2,ccqa' python diagnostics/tall_skinny_bench.py --device cuda --preset 100m --steps 1
```

**Requirements**:
- `lla2` requires CUDA (the external lightning-attention kernels are Triton/CUDA based).
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
| `HYDRA_MOR_ATTENTION_PATTERN_NAME` | `lla2x3+ccqa` | Attention pattern for MoR blocks (CUDA only) |

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

Contributions welcome! Please read [CONTRIBUTING.md](docs/CONTRIBUTING.md) first.

---

<p align="center">
  <strong>HYDRA</strong> - Multi-headed efficiency for modern transformers 🐉
</p>
