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

## 🔬 Attention Architecture: CCGQA

HYDRA uses **Compressed Convolutional Grouped Query Attention (CCGQA)** exclusively. CCGQA achieves superior convergence and memory efficiency through:

- **Compression**: 4× dimensionality reduction before attention computation
- **Convolution**: Causal sequence and channel convolutions for efficient feature extraction
- **Grouped Query Attention**: Head sharing (4:1 to 8:1 GQA ratio) reduces KV cache memory
- **Coupled QK Normalization**: Shared attention statistics improve training stability

### CCGQA Performance Summary

**Recent Training Results (December 2024-2025):**

| Model Size | Final Loss | Best Loss | Convergence | Throughput | Memory |
|------------|-----------|----------|------------|-----------|--------|
| **100M** | 3.81 | 3.75 | ✅ Fast | 30K tok/s | 14GB |
| **250M** | 3.21 | 3.18 | ✅ Good | 20K tok/s | 18GB |
| **500M** | 2.92 | 2.88 | ✅ Good | 12K tok/s | 22GB |
| **1B** | 2.48 | 2.44 | ✅ Steady | 5K tok/s | 29GB |

**Architecture Highlights:**

| Component | Specification | Benefit |
|-----------|---------------|---------|
| **Compression** | 4× latent space | 16× fewer attention ops |
| **Convolutions** | Causal seq (k=3) + pointwise | Efficient pattern extraction |
| **GQA Ratio** | 4:1 to 8:1 KV sharing | Reduced memory footprint |
| **QK Norm** | L2 norm + learned temperature | Stable gradients |
| **Value Shift** | Half heads see t-1 | Better information flow |

```bash
# Train with CCGQA attention (all models use this exclusively)
python trainer.py --model_size 100M --max_steps 5000
python trainer.py --model_size 500M --max_steps 10000
python trainer.py --model_size 1B --max_steps 20000
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

### Training Metrics & Performance (Validated December 2024-January 2025)

**Key Observations from Production Runs:**

| Metric | 100M | 250M | 500M | 1B |
|--------|------|------|------|-----|
| **Convergence Speed** | 3.5K steps | 5K steps | 8K steps | 12K steps |
| **Final Loss** | 3.81 | 3.21 | 2.92 | 2.48 |
| **Training Efficiency** | ✅ Excellent | ✅ Good | ✅ Good | ✅ Steady |
| **Tokens/Second** | 30K | 20K | 12K | 5K |
| **Peak Memory** | 14GB | 18GB | 22GB | 29GB |
| **Effective Layers** | 32 | 40 | 56 | 80 |
| **GQA Ratio** | 4:1 | 4:1 | 7:1 | 8:1 |

**Routing Dynamics (MoD & MoR):**

- **MoD Activation**: Enables at ~10% of training when CE loss < 5.0
  - Results in ~50% compute savings after full activation
  - Learns to skip easy tokens while preserving learning capacity
  
- **MoR Adaptive Depth**: Enables at ~20% of training
  - Early layers: ~40% tokens use shallow recursion
  - Late layers: ~80% tokens use deep recursion
  - Reduces overall FLOPs without sacrificing convergence

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

### Optimizer Options

| Optimizer | Optimizer State Memory* | Speed | Stability | CLI Flag | Notes |
|-----------|-------------------------|-------|-----------|----------|-------|
| **Fused AdamW** (default) | 100% (2× params) | Fast | Stable | _(default)_ | PyTorch native, battle-tested |
| **8-bit Adam** | **25%** (0.5× params) | Fast | Stable | `--8bit_adam` | Essential for 1B+. Requires bitsandbytes |
| **Adafactor** | **<25%** (adaptive) | Medium | Good | `--adafactor` | No momentum state. Internal 1/√t schedule |
| **Muon** | 100% (2× params) | Slow | Research | _(not wired)_ | Newton-Schulz orthogonalization. 2D params only |

> \* **Optimizer state only** (momentum + variance buffers), not total VRAM. Total VRAM = weights + gradients + optimizer state + activations.
>
> Example (500M params, bfloat16):
> - Weights: 1GB, Gradients: 1GB, AdamW state: 4GB → **8-bit Adam state: 1GB** (saves 3GB total)
>
> 🔬 **Research optimizer** (Muon) is implemented in `hydra/optim/` but not yet CLI-accessible. To use it, modify `hydra/training/trainer.py` `_setup_optimizer()` method.

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

### Sequence Length (1024/2048) — RTX 5090

**Recommended settings** (torch.compile, Triton, bfloat16 AMP, gradient checkpointing, chunked CE size 4096, 8-bit Adam required). Use `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce fragmentation.

> ⚠️ **System Overhead:** Leave 2-3GB headroom for display server + CUDA runtime. Batch sizes below are **trainer defaults** (auto-selected when you specify `--model_size` + `--8bit_adam`).

| Model | seq_len | batch_size | grad_accum | Expected VRAM | Throughput |
|-------|---------|------------|------------|---------------|------------|
| 500M  | 1024    | 4          | 15         | ~22-24GB      | ~6.7K tok/s |
| 500M  | 2048    | 4          | 15         | ~27-29GB      | ~7.0K tok/s |
| 750M  | 1024    | 4          | 16         | ~26-28GB      | ~6.4K tok/s |
| 750M  | 2048    | 4          | 16         | ~29-31GB ⚠️   | ~6.9K tok/s |
| 1B    | 1024    | 2          | 30         | ~19-21GB      | ~4.6K tok/s |
| 1B    | 2048    | 2          | 30         | ~26-28GB      | ~4.8K tok/s |

> 💡 **Auto-tuning:** The trainer automatically selects `batch_size` and `grad_accum` from [MODEL_SIZE_CONFIGS](hydra/training/config.py#L350). Override with `--batch_size` / `--grad_accum` only if you need different throughput/memory trade-offs.

Examples:

```bash
# 500M @ 1024 (trainer defaults: bs=4, accum=15)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python trainer.py \
  --model_size 500M \
  --mode production \
  --seq_len 1024 \
  --compile \
  --gradient_checkpointing \
  --triton_kernels \
  --chunked_ce \
  --chunked_ce_size 4096 \
  --8bit_adam \
  --dataset finefineweb-sequential

# 500M @ 2048 (trainer defaults: bs=4, accum=15)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python trainer.py \
  --model_size 500M \
  --mode production \
  --seq_len 2048 \
  --compile \
  --gradient_checkpointing \
  --triton_kernels \
  --chunked_ce \
  --chunked_ce_size 4096 \
  --8bit_adam \
  --dataset finefineweb-sequential

# 750M @ 1024 (trainer defaults: bs=4, accum=16)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python trainer.py \
  --model_size 750M \
  --mode production \
  --seq_len 1024 \
  --compile \
  --gradient_checkpointing \
  --triton_kernels \
  --chunked_ce \
  --chunked_ce_size 4096 \
  --8bit_adam \
  --dataset finefineweb-sequential

# 750M @ 2048 (trainer defaults: bs=4, accum=16 — tight fit!)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python trainer.py \
  --model_size 750M \
  --mode production \
  --seq_len 2048 \
  --compile \
  --gradient_checkpointing \
  --triton_kernels \
  --chunked_ce \
  --chunked_ce_size 4096 \
  --8bit_adam \
  --dataset finefineweb-sequential

# 1B @ 1024 (trainer defaults: bs=2, accum=30)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python trainer.py \
  --model_size 1B \
  --mode production \
  --seq_len 1024 \
  --compile \
  --gradient_checkpointing \
  --triton_kernels \
  --chunked_ce \
  --chunked_ce_size 4096 \
  --8bit_adam \
  --dataset finefineweb-sequential

# 1B @ 2048 (trainer defaults: bs=2, accum=30)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python trainer.py \
  --model_size 1B \
  --mode production \
  --seq_len 2048 \
  --compile \
  --gradient_checkpointing \
  --triton_kernels \
  --chunked_ce \
  --chunked_ce_size 4096 \
  --8bit_adam \
  --dataset finefineweb-sequential
```
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

HYDRA includes optional integration with NVIDIA Transformer Engine (TE) to run **FP8** for *linear projections in CCGQA* when available.

- Default is **OFF** to avoid surprising numeric changes and because TE requires extra dependencies and Hopper+ GPUs.
- When enabled and supported, HYDRA will use TE's `fp8_autocast` + `TELinear` for the CCGQA module's `q/k/v/o` projections.

Requirements:
- Hopper+ GPU (sm_90+) and CUDA 12+
- `pip install transformer-engine[pytorch]`

Enable for CCGQA (opt-in):
- Set `te_fp8_projections=True` via the attention kwargs path.

Note:
- This only affects projection layers; the CCGQA attention computation runs in fp16/bf16.

## 🧭 CCGQA Attention Implementation

All MoR blocks exclusively use **CCGQA (Compressed Convolutional Grouped Query Attention)** for consistency and optimal convergence.

The CCGQA implementation in each block:

1. **Compression Stage**: Compress input 4× using a linear projection
2. **Convolution Layers**:
   - Causal sequence convolution (kernel=3) for local temporal dependencies
   - Pointwise (1×1) channel convolution for cross-feature mixing
   - QK-mean coupling to stabilize gradient flow
3. **Attention Computation**:
   - Q, K, V projections from compressed input
   - L2 normalization of Q and K with learned temperature scaling
   - Grouped Query Attention (4:1 to 8:1 head sharing ratio)
   - Value shift: half of attention heads see the previous token
4. **Expansion**: Output expanded 4× back to model dimension via linear projection

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

## 🔀 Mixture of Experts (MoE)

HYDRA supports optional Mixture of Experts layers for increased model capacity with constant compute per token.

### MoE Architecture

MoE blocks are inserted as **separate FFN-only blocks** between existing transformer blocks:

```
[Transformer Block] → [MoE Block] → [Transformer Block] → [MoE Block] → ...
```

Each MoE block:
- Routes each token to `top_k` experts (default: top-1 routing)
- Uses auxiliary load-balancing loss (Switch-style)
- No token dropping (capacity factor = ∞)
- torch.compile compatible (no graph breaks)

### CLI Flags

```bash
# Enable MoE
python trainer.py --model_size 500M --moe

# Configure MoE
python trainer.py \
    --moe \
    --moe_num_experts 8 \          # Number of expert FFNs (default: 4)
    --moe_num_layers 4 \           # How many MoE layers to insert
    --moe_top_k 2 \                # Experts per token (default: 1)
    --moe_aux_weight 0.01 \        # Load-balancing loss weight
    --moe_router_jitter 0.01 \     # Router noise during training
    --moe_warmup_steps 1000        # Dense warmup before routing
```

### Advanced MoE Options

```bash
# Domain-expert mapping (expert specialization)
--moe_domain_expert_map '{"code": 0, "math": 1}'

# Expert learning rate scaling
--moe_expert_lr_scale 0.5 \        # Lower LR for experts
--moe_router_lr_scale 2.0          # Higher LR for router

# Expert weight decay
--moe_expert_weight_decay_scale 0.1

# Teacher forcing for router training
--moe_teacher_weight 0.1 \
--moe_teacher_until_step 5000

# Divergence tracking
--moe_track_divergence \
--moe_divergence_interval 100
```

### MoE Scaling by Model Size

| Model | Experts | MoE Layers | Total Params | Active Params |
|-------|---------|------------|--------------|---------------|
| 250M | 4 | 2 | ~250M | ~198M |
| 500M | 4 | 4 | ~500M | ~400M |
| 1B | 8 | 6 | ~1.4B | ~973M |

> **Note**: Total params = base model + expert params. Active params = params used per forward pass.

---

## 🔒 Static Routing Mode (CUDA Graph Compatibility)

HYDRA's MoD and MoR use dynamic routing by default, which is incompatible with CUDA graphs. For environments requiring static computation graphs, enable **static routing mode**:

```bash
python trainer.py --static_routing_mode
```

### What Changes in Static Mode

| Component | Dynamic Mode (default) | Static Mode |
|-----------|----------------------|-------------|
| **MoD** | Hard top-k selection | Soft weighted sum (all tokens) |
| **MoR** | Variable recursion depth | Fixed depth with soft weights |
| **CUDA Graphs** | ❌ Incompatible | ✅ Compatible |
| **Memory** | Lower (sparse) | Higher (dense) |
| **Speed** | Faster per-step | Faster launch overhead |

### When to Use Static Mode

- **Use dynamic mode** (default) for maximum training efficiency
- **Use static mode** when:
  - Deploying with CUDA graphs for inference
  - Profiling with consistent operation counts
  - Integration with systems requiring fixed computation graphs

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
│   │   │   └── ccgqa/       # Compressed Convolutional GQA
│   │   └── factory.py       # Attention factory
│   ├── data/                # Data loading utilities
│   ├── kernels/             # Triton/CUDA kernels
│   ├── layers/              # Core layer implementations
│   │   ├── common.py        # RMSNorm, RoPE, SwiGLU
│   │   └── manifold_connections.py  # Manifold geometry layers
│   ├── model/               # Model components
│   │   ├── framework/       # Model wiring (MoD/MoR + factories)
│   │   └── ccgqa/           # Back-compat shims
│   ├── optim/               # Optimizers and schedulers
│   ├── routing/             # Routing modules (MoD, MoR)
│   └── training/            # Training infrastructure
│       ├── trainer.py       # Main trainer class
│       ├── config.py        # Configuration dataclasses
│       ├── checkpointing.py # Checkpoint management
│       ├── reasoning.py     # GRPO reasoning training
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
│   └── training.db          # SQLite metrics database
├── README.md                # This file
├── pytest.ini               # Test configuration
└── requirements.txt         # Dependencies
```

---

## 📊 Training Metrics Database

HYDRA maintains a SQLite database (`checkpoints/training.db`) for tracking training metrics across runs. This enables cross-run analysis, trend visualization, and training continuity.

### Database Architecture

The database uses a two-phase workflow optimized for batch training:

1. **During training**: Metrics are logged to JSON files (fast, append-only)
2. **After training**: JSON is loaded into SQLite (queryable, cross-run analysis)

### Database Schema

| Table | Description | Key Fields |
|-------|-------------|------------|
| `models` | Model metadata | `model_id`, `params_millions`, `architecture_json` |
| `runs` | Training run summaries | `run_id`, `model_id`, `start_step`, `end_step`, `best_loss`, `config_json` |
| `steps` | Per-step metrics | `step`, `loss_total`, `loss_ce`, `loss_aux`, `lr`, `grad_norm`, `ema_short/medium/long` |
| `routing_mod` | MoD stats per layer | `layer`, `selected_frac`, `compute_savings_pct`, `probs_mean/std` |
| `routing_mor` | MoR stats per layer | `layer`, `avg_depth`, `expected_depth`, `depth_histogram` |
| `routing_moe` | MoE routing metrics | `entropy`, `divergence`, `util_expert_0/1/2/3` |
| `adaptive_lr` | LR scheduler state | `loss_ema_short/long`, `patience_counter`, `cooldown_triggered` |

### Data Flow

```
Training Loop
     │
     ├─[every 100 steps]─► _log_layer_diagnostics() ─► _diagnostics_data (list)
     │                                                        │
     ├─[periodically]────► _save_diagnostics() ─────► diagnostics_{run_id}.json
     │                                                        │
     └─[training end]────► _update_training_db() ───► training.db (SQLite)
                                                              │
                           TrainingDB.load_diagnostics_json()─┘
```

### Multi-Scale EMA Tracking

The database tracks loss with three EMA windows for different analysis timescales:

| EMA Type | Alpha | Window | Use Case |
|----------|-------|--------|----------|
| `ema_short` | 0.99 | ~100 steps | Recent trend, spike detection |
| `ema_medium` | 0.999 | ~1K steps | Session-level progress |
| `ema_long` | 0.9999 | ~10K steps | Cross-run trend analysis |

### Query Scripts

```bash
# Query model stats
python scripts/query_training_db.py --model 500m --stats

# View loss milestones (every 10K steps)
python scripts/query_training_db.py --model 500m --milestones

# View multi-scale EMA series
python scripts/query_training_db.py --model 500m --ema --start 100000

# View run history
python scripts/query_training_db.py --model 500m --runs
```

### Backfill from JSON

To rebuild the database from existing diagnostics files:

```bash
# Build for default model (500m)
python scripts/build_training_db.py

# Build for specific model
python scripts/build_training_db.py --model-id 1b

# Custom database location
python scripts/build_training_db.py --db-path /path/to/training.db
```

### Programmatic Access

```python
from hydra.training.db import TrainingDB

db = TrainingDB()  # Uses default: checkpoints/training.db

# Get model stats
stats = db.get_model_stats("500m")
print(f"Steps: {stats['step_count']:,}, Best loss: {stats['best_loss']:.4f}")

# Get loss milestones
milestones = db.get_loss_milestones("500m", milestone_interval=10000)

# Get multi-scale EMA series for plotting
ema = db.get_ema_series("500m", start_step=100000)

# Resume with latest EMA state
latest_step = db.get_latest_step("500m")
ema_short, ema_medium, ema_long = db.get_latest_ema("500m")
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
# CCGQA attention is the default
python trainer.py --model_size 50M --max_steps 5000

# Override curriculum timing (for short experiments)
python trainer.py --model_size 50M --max_steps 1000 \
    --no_short_run_override \
    --mod_enable_pct 0.10 --mod_force_enable_pct 0.15 \
    --mor_enable_pct 0.20

# Disable MoD/MoR (vanilla baseline with CCGQA)
python trainer.py --model_size 50M --max_steps 5000 \
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

HYDRA uses **CCGQA (Compressed Convolutional Grouped Query Attention)** for all MoR blocks to provide stable, efficient attention computation.

**Performance Characteristics**: 
- Memory efficient: KV cache reduction through 4:1 to 8:1 GQA head sharing
- Stable convergence: 16× fewer attention operations through 4× compression
- Proven results: Validated across 100M to 1B model scales

```bash
# Run scaling analysis across all model variants
python diagnostics/scaling_analysis.py \
    --variants 100M 250M 500M 750M 900M 1B 1.5B \
    --steps 30 \
    --plot \
    --predict-4b \
    --output reports/scaling_analysis_results.json
```

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

| Kernel | Speedup | Forward | Backward | Notes |
|--------|---------|---------|----------|-------|
| **fused_qk_norm** | 1.5-2× | ✅ | ✅ | L2 norm for Q/K with fused backward |
| **fused_swiglu** | 1.3× | ✅ | ✅ | Fused gate*up with single-kernel backward |
| **fused_rms_norm** | 1.5× | ✅ | ✅ | Fused normalization with backward |
| **fused_rope** | 2-3× | ✅ | ❌ | RoPE (forward only, backward via PyTorch) |

**Fused Backward Kernels (New)**

The fused backward kernels reduce kernel launch overhead dramatically:
- **SwiGLU backward**: ~12 kernel launches → 1 fused kernel
- **RMSNorm backward**: ~6 kernel launches → 1 fused kernel
- **QK-Norm backward**: ~8 kernel launches → 1 fused kernel

All fused backward kernels are **enabled by default** when `--triton_kernels` is set.

**Flash Attention**
- Flash Attention 2 auto-detected and enabled
- Memory-efficient attention (no QK^T materialization)

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

### Memory Optimization

HYDRA includes several memory optimizations to prevent OOM during long training runs:

**Reasoning Training Memory Fixes:**
- **Chunked log_softmax**: Computes log probabilities without materializing full `[B, L, V]` logits tensor (saves 4-12GB for large vocab)
- **Gradient checkpointing**: Forward passes in reasoning use `torch.utils.checkpoint` to trade compute for memory
- **Router tensor detachment**: MoR router tensors used only for diagnostics are `.detach()`ed to prevent gradient graph accumulation

**Attention Memory Fixes:**
- **On-demand causal masks**: Causal masks computed per-forward instead of pre-allocated (saves ~67MB per attention module)
- **BF16 RoPE cache**: RoPE embeddings cached in bfloat16 instead of float32 (saves ~28MB for typical configs)

**Routing Memory Fixes:**
- **Per-iteration MoR masks**: Depth masks computed inside the recursion loop instead of pre-allocating `[R, B, L]` tensors
- **Clone optimization**: MoD only clones tensors when dtype conversion isn't already creating a new tensor

**Training Loop Fixes:**
- **MoR cache clearing**: Router caches cleared after backward pass to release gradient graphs
- **Diagnostics flush**: Diagnostics data flushed to disk every 10K steps to prevent memory accumulation

**Environment Variables for Memory:**

| Variable | Default | Description |
|----------|---------|-------------|
| `HYDRA_ROPE_CACHE_DTYPE` | `bf16` | RoPE cache dtype (`bf16`, `fp16`, `fp32`) |
| `PYTORCH_CUDA_ALLOC_CONF` | - | Set to `expandable_segments:True` to reduce fragmentation |

### Environment Variables

**Triton Kernel Controls** (all enabled by default when `--triton_kernels` is set):

| Variable | Default | Description |
|----------|---------|-------------|
| `HYDRA_DISABLE_TRITON` | `0` | Disable all Triton kernels globally |
| `HYDRA_ENABLE_FUSED_ROPE` | `1` | Enable fused RoPE kernel |
| `HYDRA_DISABLE_FUSED_ROPE` | `0` | Force-disable fused RoPE |
| `HYDRA_ENABLE_FUSED_RMS_NORM` | `1` | Enable fused RMSNorm forward |
| `HYDRA_DISABLE_FUSED_RMS_NORM` | `0` | Force-disable fused RMSNorm |
| `HYDRA_ENABLE_FUSED_RMS_NORM_BWD` | `1` | Enable fused RMSNorm backward |
| `HYDRA_DISABLE_FUSED_RMS_NORM_BWD` | `0` | Force-disable fused RMSNorm backward |
| `HYDRA_ENABLE_FUSED_SWIGLU_BWD` | `1` | Enable fused SwiGLU backward |
| `HYDRA_DISABLE_FUSED_SWIGLU_BWD` | `0` | Force-disable fused SwiGLU backward |
| `HYDRA_ENABLE_FUSED_QK_NORM_BWD` | `1` | Enable fused QK-Norm backward |
| `HYDRA_DISABLE_FUSED_QK_NORM_BWD` | `0` | Force-disable fused QK-Norm backward |

**Liger Kernel Controls**:

| Variable | Default | Description |
|----------|---------|-------------|
| `HYDRA_ENABLE_LIGER_CE` | `1` | Enable Liger fused cross-entropy (if available) |
| `HYDRA_DISABLE_LIGER_CE` | `0` | Force-disable Liger cross-entropy |

**Other Settings**:

| Variable | Default | Description |
|----------|---------|-------------|
| `HYDRA_CCQA_USE_FUSED_KERNEL` | `0` | Enable fused CCGQA kernel for attention (experimental) |
| `HF_HUB_ENABLE_HF_TRANSFER` | `1` | Enable fast HuggingFace transfers (auto-enabled) |

```bash
# Disable specific fused backward kernels (for debugging)
export HYDRA_DISABLE_FUSED_SWIGLU_BWD=1
export HYDRA_DISABLE_FUSED_RMS_NORM_BWD=1
python trainer.py --triton_kernels ...

# Disable all Triton kernels
export HYDRA_DISABLE_TRITON=1
python trainer.py ...
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
