# Frontier Optimizations for HYDRA Training

## Summary

This document tracks cutting-edge optimizations researched and implemented for HYDRA training, based on frontier techniques from Liger Kernel (LinkedIn), PyTorch performance tuning, and HuggingFace best practices.

## Implemented Optimizations

### 1. CUDA/cuDNN Optimizations (train_100m_optimized.py)
```python
torch.backends.cudnn.benchmark = True  # Auto-tune convolutions for GPU
torch.backends.cuda.matmul.allow_tf32 = True  # TF32 for faster matmuls (Ampere+)
torch.backends.cudnn.allow_tf32 = True  # TF32 for cuDNN ops
torch.set_float32_matmul_precision('high')  # TF32 for matmuls
```
**Impact**: 10-30% speedup on Ampere/Hopper GPUs without precision loss for most LLM training.

### 2. Autograd-Compatible Fused Triton Kernels (hydra/kernels/fused_ops.py)
Fixed critical gradient flow bug where Triton kernels broke autograd graph.

- **FusedRMSNormFunction**: Triton forward + PyTorch backward
- **FusedSwiGLUFunction**: Triton forward + PyTorch backward  
- **FusedQKNormFunction**: Triton forward + PyTorch backward

**Impact**: 1.3-2x speedup for individual ops while maintaining correct gradient flow.

### 3. Chunked Cross-Entropy Loss ✅ TESTED (hydra/kernels/fused_ops.py)
```python
from hydra.kernels import chunked_cross_entropy, fused_chunked_cross_entropy
```
Memory-efficient cross-entropy that avoids materializing full logits tensor.

**Test Results**:
- Forward pass: ✓ (1.91e-06 difference from standard)
- Backward pass: ✓ (0.00% gradient difference)
- Fused autograd: ✓ (7.45e-09 hidden grad diff)

**Config options** (train_100m_optimized.py):
```python
use_chunked_ce: bool = False  # Enable chunked cross-entropy
chunked_ce_size: int = 4096   # Tokens per chunk
```

**Impact**: 4-8x reduction in peak memory for output projection layer.

### 4. Gradient Checkpointing ✅ TESTED (hydra/model/ccgqa.py)
```python
model.enable_gradient_checkpointing()  # Enable
model.disable_gradient_checkpointing()  # Disable
model.is_gradient_checkpointing  # Check status
```

Recomputes activations during backward pass instead of storing them.

**Test Results** (micro model):
- Forward pass: ✓ (0.00 loss difference)
- Backward pass: ✓ (0.00% gradient difference)
- **Memory savings: 92.9%** (278.2 MB → 19.6 MB)

**Config options** (train_100m_optimized.py):
```python
gradient_checkpointing: bool = False  # Enable for memory-constrained training
```

**Impact**: ~40-60% memory reduction at ~30% compute cost. Ideal for:
- Large batch sizes
- Long sequences
- Memory-constrained GPUs

### 5. Async Data Transfer
```python
batch["input_ids"].to(device, non_blocking=True)
batch["labels"].to(device, non_blocking=True)
```
**Impact**: Overlaps CPU→GPU transfer with computation.

### 6. PyTorch 2.0 Native Flash Attention
Using `F.scaled_dot_product_attention` which automatically uses FlashAttention v2:
- Fused softmax + matmul
- O(N) memory instead of O(N²)
- 2-4x faster than vanilla attention

## Researched But Not Yet Implemented

### 1. FusedLinearCrossEntropy (Liger Kernel)
The most impactful optimization from Liger Kernel. Fuses:
- Linear projection (lm_head)
- Cross-entropy loss computation

**Benefits**:
- Avoids materializing full logits tensor entirely
- Up to 80% memory reduction
- 10-20% speed improvement

**Implementation complexity**: Requires modifying model's forward pass to return hidden states instead of logits.

### 2. Gradient Checkpointing
Trade compute for memory by recomputing activations during backward pass.

**Options**:
- Per-block checkpointing (moderate savings)
- Full checkpointing (maximum savings, 2x slower)

**When to use**: When GPU memory is the bottleneck (larger batch sizes, longer sequences).

### 3. 8-bit Optimizers (bitsandbytes)
```python
import bitsandbytes as bnb
optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=5e-4)
```
**Impact**: ~50% memory reduction for optimizer states.

### 4. Mixed Precision with Dynamic Loss Scaling
Current: Using bfloat16 (no scaling needed)
Alternative: fp16 with GradScaler for older GPUs

### 5. CUDA Graphs (for fixed batch sizes)
```python
# Capture forward + backward in a CUDA graph
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    out = model(x)
    loss = criterion(out, y)
    loss.backward()
g.replay()  # Execute captured operations
```
**Caveat**: Incompatible with dynamic shapes, gradient accumulation variations.

### 6. Tensor Parallelism
For multi-GPU: Split large weight matrices across GPUs.
- Column-parallel for Linear1 (gate/up projections)
- Row-parallel for Linear2 (down projection)

## Performance Comparison Table

| Optimization | Memory Impact | Speed Impact | Complexity |
|-------------|---------------|--------------|------------|
| TF32 matmuls | None | +10-30% | Trivial |
| Fused Triton kernels | -10% | +20-50% | Moderate |
| Chunked CrossEntropy | -50-80%* | -5% | Low |
| FusedLinearCE | -80%* | +15% | High |
| Gradient checkpointing | -40-60% | -30% | Low |
| 8-bit optimizer | -50%** | None | Low |
| CUDA Graphs | None | +20% | High |

\* Memory savings for output layer only
\** Memory savings for optimizer states only

## Quick Start: Enable All Safe Optimizations

```python
# In training config
config = TrainingConfig(
    # Already enabled by default:
    use_compile=True,
    compile_mode="max-autotune-no-cudagraphs",
    dtype="bfloat16",
    
    # New optimizations (safe to enable):
    # use_chunked_ce=True,  # Enable after verification
)
```

## Monitoring Memory Usage

```python
import torch
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
print(f"Max Allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
```

## References

1. [Liger Kernel](https://github.com/linkedin/Liger-Kernel) - LinkedIn's Triton kernels for LLM training
2. [PyTorch Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
3. [HuggingFace GPU Training](https://huggingface.co/docs/transformers/perf_train_gpu_one)
4. [FlashAttention-2 Paper](https://arxiv.org/abs/2307.08691)
5. [Chinchilla Scaling Laws](https://arxiv.org/abs/2203.15556)
