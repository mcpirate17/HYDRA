# Memory Optimization Guide: Batch Size, Gradients, and Optimizers

**Hardware:** RTX 5090 (31.36 GB VRAM)  
**Date:** December 2025  
**Models:** 500M, 750M, 1B HYDRA (MoD+MoR)

---

## 📊 Memory Budget Breakdown

For a typical training step, VRAM is allocated as:

```
Total VRAM = Weights + Activations + Gradients + Optimizer State + CUDA Overhead
```

**Example: 500M params @ bfloat16, batch_size=4, seq_len=1024**

| Component | Size (GB) | % of Total | Notes |
|-----------|-----------|------------|-------|
| **Weights** | 1.0 | 4% | Model parameters (500M × 2 bytes) |
| **Activations** | 12-16 | 48-64% | Cached for backward pass. Reduced by gradient checkpointing |
| **Gradients** | 1.0 | 4% | ∇L for each param |
| **Optimizer State** | 4.0 (AdamW) | 16% | Momentum + variance (2× params in fp32) |
| | 1.0 (8-bit Adam) | 4% | Quantized states (0.5× params) |
| | 0.5 (Adafactor) | 2% | Adaptive, no momentum |
| | 2.0 (Lion) | 8% | Momentum only (1× params in fp32) |
| **CUDA/System** | 1-2 | 4-8% | Fragmentation, kernels, display server |
| **Total (AdamW)** | ~19-24 GB | 100% | |
| **Total (8-bit Adam)** | ~16-21 GB | 100% | **Saves 3GB** |

> 💡 **Key insight:** Activations dominate memory at small batch sizes. Optimizer state dominates at large batch sizes.

---

## 🎛️ Control Knobs: What Each Parameter Does

### 1. Batch Size (`--batch_size`)

**What it is:** Number of examples processed per GPU per gradient accumulation step (microbatch size).

**Memory impact:**
- **Linear with activations**: `batch_size=8` uses ~2× memory of `batch_size=4`
- **No impact on weights/gradients/optimizer state**

**Performance impact:**
- **Higher = faster throughput** (better GPU utilization, more parallelism)
- **Diminishing returns** after batch_size ~8-16 for attention models
- **Too high = OOM** ⚠️

**When to increase:** GPU is underutilized (<80% VRAM), throughput is low
**When to decrease:** OOM errors, or you need lower effective batch for training dynamics

---

### 2. Gradient Accumulation (`--grad_accum`)

**What it is:** Accumulate gradients over N microbatches before optimizer step.

**Effective batch size:** `batch_size × grad_accum`

**Memory impact:**
- **Zero impact on VRAM** (gradients are accumulated in-place)
- Allows large effective batch sizes on limited VRAM

**Performance impact:**
- **Slower tok/s** (more forward/backward passes per optimizer step)
- **Same convergence** as large batch_size (if effective batch matches)
- **Lower overhead** than distributed data parallelism

**When to increase:** Need larger effective batch for stability/convergence, but VRAM-constrained
**When to decrease:** Throughput is too low, want faster iteration

**Trade-off:**
```
batch_size=8, grad_accum=4  → 32 effective, faster tok/s
batch_size=2, grad_accum=16 → 32 effective, slower tok/s, less VRAM
```

---

### 3. Sequence Length (`--seq_len`)

**What it is:** Maximum context length (tokens per example).

**Memory impact:**
- **Quadratic for attention** without optimizations: O(seq_len²) for full attention
- **Linear for CCGQA** (HYDRA's compressed attention): O(seq_len)
- **Linear for activations** in feedforward layers
- Overall: ~1.5-2× memory when doubling seq_len

**Performance impact:**
- **Longer = slower per token** (more computation per step)
- **Model quality improves** with longer context (up to a point)

**Recommendations:**
- **512-1024**: Fast iteration, lower quality on long-context tasks
- **2048-4096**: Balanced, standard for most LLMs
- **8192+**: Memory-hungry, only if task requires it

---

### 4. Gradient Checkpointing (`--gradient_checkpointing`, `--checkpoint_every`)

**What it is:** Recompute activations during backward pass instead of storing them.

**Memory impact:**
- **Saves 35-60% activation memory** depending on `checkpoint_every`
- `--checkpoint_every 1`: Checkpoint every layer (~60% savings, slowest)
- `--checkpoint_every 2`: Checkpoint every 2 layers (~35% savings, balanced) ← **default**

**Performance impact:**
- **Adds ~15-25% compute overhead** (recomputation during backward)
- Essential for large models/long sequences

**When to use:**
- Always enable for 750M+ models
- Always enable for seq_len ≥ 1024
- Disable only for tiny models (<100M) on huge GPUs

---

### 5. Chunked Cross-Entropy (`--chunked_ce`, `--chunked_ce_size`)

**What it is:** Split CE loss computation into chunks to reduce peak memory.

**Memory impact:**
- **Saves 2-4GB** on large vocab (50K tokens) with long sequences
- Chunk size 4096 (default) is balanced

**Performance impact:**
- **Slight slowdown** (~5-10%) due to kernel launch overhead
- **Negligible with torch.compile** (kernels are fused)

**Recommendation:** Always enable (default ON).

---

## 🔀 Optimizer Memory Comparison

| Optimizer | State Memory | When to Use | Expected Benefit |
|-----------|--------------|-------------|------------------|
| **AdamW (fused)** | 2× params (fp32) | Default, stable, proven | Baseline |
| **8-bit Adam** | 0.5× params | 1B+ models, VRAM-constrained | **3-4GB savings** on 500M |
| **Adafactor** | <0.5× params | Extreme VRAM constraints | **3-4GB savings**, different dynamics |
| **Lion** | 1× params | Speed experiments, research | **1-2GB savings**, needs LR tuning |
| **C-Lion** | 1× params | Stable alternative to Lion | **1-2GB savings**, better early training |

**State size (500M params):**
- AdamW: 4GB (2× params × 4 bytes)
- 8-bit Adam: 1GB (2× params × 1 byte)
- Adafactor: 0.5GB (adaptive factorization)
- Lion: 2GB (1× params × 4 bytes, momentum only)

---

## 📖 Decision Recipe

### Step 1: Choose Optimizer Based on Model Size

```
< 500M params:  AdamW (default) — plenty of VRAM headroom
500M - 750M:    8-bit Adam       — safe, proven, essential
1B+:            8-bit Adam       — mandatory for 32GB VRAM
Extreme OOM:    Adafactor        — last resort, different training dynamics
```

### Step 2: Set Base Batch Size (Use Trainer Defaults)

The trainer auto-selects based on `--model_size`:

```python
MODEL_SIZE_CONFIGS = {
    "500M": {"default_batch_size": 4, "default_grad_accum": 15},
    "750M": {"default_batch_size": 4, "default_grad_accum": 16},
    "1B":   {"default_batch_size": 2, "default_grad_accum": 30},
}
```

**Override only if:**
- You measured actual VRAM usage and have headroom
- You need different throughput/convergence trade-offs

### Step 3: Tune for Sequence Length

| seq_len | Recommendation |
|---------|----------------|
| 512 | Can use higher batch_size (2× default) |
| 1024 | Use defaults |
| 2048 | May need to halve batch_size |
| 4096+ | Definitely halve batch_size, consider Adafactor |

### Step 4: Enable Memory Savers

**Always enable (default ON):**
```bash
--compile                     # torch.compile for kernel fusion
--gradient_checkpointing      # 35% activation savings
--chunked_ce                  # 2-4GB savings
--triton_kernels              # Fused ops (safe subset)
```

**For large models (750M+):**
```bash
--8bit_adam                   # 3-4GB optimizer savings
--checkpoint_every 1          # 60% activation savings (slower)
```

**Environment:**
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # Reduce fragmentation
```

### Step 5: Monitor and Adjust

**Signs you have headroom:**
- Peak VRAM < 25GB on RTX 5090
- `nvidia-smi` shows <80% utilization during training

**Action:** Increase `batch_size` by 1-2, measure again.

**Signs you're at the limit:**
- Peak VRAM > 29GB
- Occasional OOM spikes

**Action:** Decrease `batch_size` by 1, or increase `grad_accum` to maintain effective batch.

---

## 🎯 Recommended Configs (RTX 5090, 31GB)

### 500M Model

**Training command:**
```bash
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
```

**Memory breakdown (batch_size=4, grad_accum=15, seq_len=1024):**
- Weights: 1GB
- Activations: 12GB (with checkpointing)
- Gradients: 1GB
- 8-bit Adam state: 1GB
- CUDA overhead: 2GB
- **Total: ~17-20GB** ✅ Safe

**Throughput:** ~6.7K tok/s

**To squeeze more performance:**
- Try `batch_size=6` (expect +1K tok/s, +4GB VRAM)
- Or try `seq_len=2048` with `batch_size=4` (~7K tok/s, 27-29GB VRAM)

---

### 750M Model

**Training command:**
```bash
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
```

**Memory breakdown (batch_size=4, grad_accum=16, seq_len=1024):**
- Weights: 1.5GB
- Activations: 15GB
- Gradients: 1.5GB
- 8-bit Adam state: 1.5GB
- CUDA overhead: 2GB
- **Total: ~21-24GB** ✅ Safe

**Throughput:** ~6.4K tok/s

**For seq_len=2048:**
- Use `batch_size=4` (28-30GB) ⚠️ **Tight fit**
- Or drop to `batch_size=3` (25-27GB) safer

---

### 1B Model

**Training command:**
```bash
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
```

**Memory breakdown (batch_size=2, grad_accum=30, seq_len=1024):**
- Weights: 2GB
- Activations: 12GB
- Gradients: 2GB
- 8-bit Adam state: 2GB
- CUDA overhead: 2GB
- **Total: ~20-22GB** ✅ Safe

**Throughput:** ~4.6K tok/s

**For seq_len=2048:**
- Use `batch_size=2` (26-28GB) ✅ Safe
- `batch_size=3` will OOM

---

## 🔬 Advanced: Optimizer State Math

### AdamW (Standard)

Stores for each parameter:
- **Momentum** (first moment): 1× params in fp32 = 4 bytes/param
- **Variance** (second moment): 1× params in fp32 = 4 bytes/param
- **Total:** 8 bytes/param

**Example:** 500M params → 500M × 8 = 4GB

### 8-bit Adam (Quantized)

Stores:
- **Momentum**: 1× params quantized to int8 = 1 byte/param
- **Variance**: 1× params quantized to int8 = 1 byte/param
- **Quantization metadata**: Small overhead
- **Total:** ~2 bytes/param

**Example:** 500M params → 500M × 2 = 1GB  
**Savings:** 3GB (75% reduction in optimizer state)

### Adafactor (Factorized)

Stores:
- **Row factor**: sqrt(params) in fp32
- **Column factor**: sqrt(params) in fp32
- **Total:** ~2 × sqrt(params) × 4 bytes

**Example:** 500M params → 2 × 22K × 4 = 0.18GB  
**Savings:** 3.82GB (95% reduction, but different optimizer dynamics)

### Lion (Momentum-only)

Stores:
- **Momentum** (interpolated): 1× params in fp32 = 4 bytes/param
- **No variance** (uses sign-based updates)
- **Total:** 4 bytes/param

**Example:** 500M params → 500M × 4 = 2GB  
**Savings:** 2GB (50% reduction)

---

## 📈 Performance vs Memory Trade-offs

### Batch Size vs Throughput (500M, seq_len=1024)

| batch_size | grad_accum | Effective | VRAM | Throughput | Efficiency |
|------------|------------|-----------|------|------------|------------|
| 1 | 60 | 60 | 15GB | 5.7K tok/s | 38% |
| 2 | 30 | 60 | 17GB | 6.5K tok/s | 43% |
| 4 | 15 | 60 | 22GB | 6.7K tok/s | 45% |
| 8 | 7-8 | 56-64 | 28GB | 7.4K tok/s | 49% |
| 16 | 4 | 64 | OOM | — | — |

**Takeaway:** Throughput increases ~30% from batch_size=1 to 8, but VRAM doubles.

### Sequence Length vs Memory (500M, batch_size=4)

| seq_len | VRAM | Throughput | Quality |
|---------|------|------------|---------|
| 512 | 16GB | 10K tok/s | Basic |
| 1024 | 22GB | 6.7K tok/s | Good |
| 2048 | 29GB | 7.0K tok/s | Better |
| 4096 | OOM | — | Best |

**Takeaway:** 2× seq_len → ~1.4× VRAM, but longer context improves model quality.

### Gradient Checkpointing Impact (500M, batch_size=4, seq_len=1024)

| checkpoint_every | VRAM Saved | Slowdown | Recommendation |
|------------------|------------|----------|----------------|
| OFF | 0GB | 0% | Only for <100M models |
| 4 | 3GB | ~8% | Good for 500M |
| 2 | 5GB | ~15% | **Default, balanced** |
| 1 | 8GB | ~25% | Essential for 1B+ |

---

## 🚨 Troubleshooting OOM

### Checklist (in order of impact):

1. ✅ **Enable 8-bit Adam** (`--8bit_adam`) → Saves 3-4GB
2. ✅ **Reduce batch_size by 1** → Saves 3-6GB per step
3. ✅ **Enable gradient checkpointing** (`--gradient_checkpointing`) → Saves 5-8GB
4. ✅ **Reduce seq_len** (e.g., 2048 → 1024) → Saves 4-6GB
5. ✅ **Set `checkpoint_every 1`** → Additional 3-4GB savings, ~10% slower
6. ⚠️ **Try Adafactor** (`--adafactor`) → Saves additional 0.5-1GB, different dynamics
7. ⚠️ **Disable compile** (`--no-compile`) → May save 1-2GB, much slower (not recommended)

### When Nothing Works:

**Multi-GPU training** (not yet implemented in HYDRA):
- 2× RTX 5090 = 62GB VRAM
- Enables 1.5B+ models or longer sequences

**Reduce model size:**
- 750M → 500M preserves most capabilities, fits easily

---

## 🎓 Key Principles

1. **Activations dominate at small batch sizes** → Use gradient checkpointing
2. **Optimizer state dominates at large batch sizes** → Use 8-bit Adam
3. **Throughput scales sublinearly with batch_size** → Don't over-optimize
4. **Effective batch matters for convergence** → Maintain it via grad_accum
5. **Leave 2-3GB VRAM headroom** → System overhead + fragmentation
6. **Use trainer defaults first** → They're tuned for stability

---

## 📚 References

- **8-bit Adam paper:** https://arxiv.org/abs/2110.02861
- **Adafactor paper:** https://arxiv.org/abs/1804.04235
- **Lion paper:** https://arxiv.org/abs/2302.06675
- **Gradient checkpointing:** https://arxiv.org/abs/1604.06174
- **PyTorch Memory Management:** https://pytorch.org/docs/stable/notes/cuda.html

---

**Last updated:** December 29, 2025  
**Hardware:** NVIDIA RTX 5090 (31.36 GB)  
**Framework:** PyTorch 2.x + torch.compile + Triton
