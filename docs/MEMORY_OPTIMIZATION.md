# HYDRA Memory Optimization Guide

Based on deep diagnostic analysis of a 2B parameter model (500M base + MoE) on RTX 5090 32GB.

## Current Memory Profile

### Model: 500M + MoE (6 layers × 4 experts) = 2004M params

| Stage | Memory | Notes |
|-------|--------|-------|
| Model on GPU | 5.94 GB | bf16 weights |
| After forward | 8.41 GB | +2.5GB activations |
| **After backward** | **24.27 GB** | **+16GB gradients + activations** |
| After optimizer | 27.27 GB | +3GB optimizer state |
| **Peak** | **28.53 GB** | Barely fits in 32GB |

### Parameter Breakdown

| Component | Params | Size (bf16) |
|-----------|--------|-------------|
| MLP layers | 930.4M | 1.86 GB |
| MoE experts | 858.8M | 1.72 GB |
| Embeddings | 90.2M | 0.18 GB |
| Attention | 25.7M | 0.05 GB |
| Other | 99.2M | 0.20 GB |
| **Total** | **2004.3M** | **4.01 GB** |

## Why OOM at Step 50?

1. **Gradients**: Same size as model = 4 GB
2. **Optimizer state (8-bit)**: ~3 GB (would be 8 GB with 32-bit Adam)
3. **Activations during backward**: ~16 GB (even with gradient checkpointing)
4. **PyTorch reserved overhead**: ~2 GB

**Total: ~29 GB** → OOM with 32 GB GPU when variance adds 3-4 GB peak

## Memory Reduction Strategies

### 1. Reduce Batch Size (Immediate Fix)

Current: `batch_size=4, seq_len=1024` → 4K tokens
Try: `batch_size=2, seq_len=1024` → 2K tokens

Saves: ~6-8 GB activation memory

```bash
--batch_size 2 --grad_accum 8  # Maintain effective batch
```

### 2. Enable Aggressive Gradient Checkpointing

Current: `checkpoint_every=1` (every layer)
Already optimal, but verify it's enabled:

```bash
--gradient_checkpointing --checkpoint_every 1
```

### 3. Use Chunked Cross-Entropy

Avoids materializing full logits (50257 × seq_len × batch):

```bash
--chunked_ce --chunked_ce_size 2048
```

Saves: ~1-2 GB for 2B vocab model

### 4. Reduce MoE Memory During Training

**Option A: Freeze MoE experts temporarily**
```python
# In training loop, after 10K steps
for moe_layer in model.moe_layers:
    for expert in moe_layer.experts:
        for param in expert.parameters():
            param.requires_grad = False
```

Saves: 858.8M × 2 bytes = 1.7 GB gradients

**Option B: Use MoE gradient checkpointing**
Each expert forward is checkpointed, recomputed during backward.

### 5. Mixed Precision Optimizations

**Current**: bf16 AMP is enabled
**Additional**: Ensure optimizer states are in bf16/8-bit

```bash
--8bit_adam  # Already using this
```

### 6. Reduce Sequence Length

If acceptable for training phase:
```bash
--seq_len 512  # Half the activations
```

Saves: ~8 GB activation memory

### 7. Memory-Efficient Attention

CCGQA already uses 4× compression. Additional savings:
- Flash Attention 2: Already auto-detected
- Sage Attention: For A100/H100, uses INT8

### 8. Gradient Accumulation Without Holding Graphs

Current implementation clears MoR caches after backward. Verify:
```python
if use_mod_mor:
    clear_mor_caches(model)
```

## Compile Warmup (NEW)

For large models that OOM during torch.compile graph capture, use compile warmup:

```bash
--compile_warmup_steps 10 \           # Conservative settings for first 10 steps
--compile_warmup_batch_size 1 \       # Reduced batch during warmup
--compile_warmup_seq_len 512 \        # Reduced seq len during warmup
--compile_warmup_checkpoint_every 1   # Aggressive checkpointing during warmup
```

This dramatically reduces memory during the torch.compile graph capture phase:
- **Step 0-9**: batch=1, seq=512, checkpoint every layer → ~20GB peak
- **Step 10+**: batch=4, seq=1024, normal checkpointing → ~28GB peak

The transition happens automatically after warmup steps complete.

### How It Works

1. During warmup, the trainer uses conservative settings:
   - Smaller batch size (default: 1)
   - Shorter sequences (default: 512)
   - Checkpoint every layer (default: every 1)

2. torch.compile captures the graph with these small tensors

3. After warmup completes, trainer transitions to target settings:
   - Restores original batch size
   - Restores original sequence length
   - Restores original checkpointing interval
   - Clears CUDA cache to release warmup allocations

### Example for 2B Model

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python trainer.py \
    --model_size 500M \
    --moe --moe_num_layers 6 \
    --resume checkpoints/reasoning/reasoning_checkpoint.pt \
    --compile_warmup_steps 10 \
    --compile_warmup_batch_size 1 \
    --compile_warmup_seq_len 512 \
    --8bit_adam \
    --gradient_checkpointing
```

---

## Recommended Configuration for 32GB GPU

For 2B model (500M + MoE):

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python trainer.py \
    --model_size 500M \
    --moe --moe_num_layers 6 \
    --batch_size 2 \          # Reduced from 4
    --grad_accum 8 \          # Increased to compensate
    --seq_len 1024 \
    --gradient_checkpointing \
    --checkpoint_every 1 \
    --8bit_adam \
    --chunked_ce \
    --chunked_ce_size 2048 \
    --triton_kernels
```

**Expected memory**: ~26-27 GB (safe margin for 32GB)

## Memory Monitoring

Add to training command for debugging:
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
```

For detailed profiling:
```bash
python diagnostics/deep_memory_diagnostic.py \
    --checkpoint your_checkpoint.pt \
    --snapshot  # Creates visualization file
```

## References

- [PyTorch Memory Management](https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
- [Understanding CUDA Memory Usage](https://pytorch.org/docs/stable/torch_cuda_memory.html)
- [Memory Optimization Overview - torchtune](https://pytorch.org/torchtune/stable/tutorials/memory_optimizations.html)
- [Gradient Checkpointing in PyTorch](https://www.codegenes.net/blog/gradient-checkpointing-pytorch/)
- [PyTorch Memory Visualization](https://pytorch.org/memory_viz)

## Diagnostic Tools

```bash
# Deep memory diagnostic
python diagnostics/deep_memory_diagnostic.py --checkpoint path/to/ckpt.pt

# GRPO-specific memory profiling
python diagnostics/profile_grpo_memory.py

# Real-time monitoring during training
watch -n 1 nvidia-smi
```
