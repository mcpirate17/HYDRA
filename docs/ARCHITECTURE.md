# HYDRA Architecture Deep Dive

## Table of Contents
1. [Theoretical Foundation](#theoretical-foundation)
2. [CCGQA: Compressed Convolutional Grouped Query Attention](#ccgqa)
3. [MoD: Mixture-of-Depths](#mod)
4. [MoR: Mixture-of-Recursions](#mor)
5. [Integration: The Complete HYDRA Stack](#integration)
6. [Scaling Laws](#scaling-laws)
7. [Implementation Details](#implementation-details)

---

## Theoretical Foundation

HYDRA addresses three fundamental inefficiencies in standard transformers:

| Problem | Standard Transformer | HYDRA Solution |
|---------|---------------------|----------------|
| Quadratic attention | O(n²d) | CCGQA: O(n²d/C) with C=4 |
| Uniform compute | All tokens same cost | MoD: 75% tokens processed |
| Fixed depth | All layers always run | MoR: Adaptive 1-5 recursions |

### Efficiency Gains

```
Standard Transformer:  FLOPs = n_layers × (attn_flops + ffn_flops)
HYDRA Transformer:     FLOPs = n_layers × 0.75 × ((attn_flops/4) + ffn_flops) × avg_depth
                            ≈ 0.75 × 0.5 × FLOPs_standard
                            ≈ 37.5% of baseline FLOPs
```

---

## CCGQA: Compressed Convolutional Grouped Query Attention

**Paper**: [arXiv:2510.04476](https://arxiv.org/abs/2510.04476)

### Key Insight

Standard attention computes Q, K, V in full dimension, then computes O(n²d) attention. CCGQA projects to a compressed latent space first, reducing both memory and compute.

### Architecture

```
Input x: [batch, seq, dim]
           │
           ▼
    ┌──────────────────┐
    │  Q_down, K_down  │  Linear: dim → dim/C (C=4)
    │     V_down       │
    └──────────────────┘
           │
           ▼
    ┌──────────────────┐
    │  Sequence Conv   │  Causal 1D conv, kernel=3
    │  (on Q and K)    │  Adds local context
    └──────────────────┘
           │
           ▼
    ┌──────────────────┐
    │  Channel Conv    │  1×1 pointwise conv
    │  (on Q and K)    │  Cross-channel mixing
    └──────────────────┘
           │
           ▼
    ┌──────────────────┐
    │   QK-Mean        │  Q = Q + mean(K)
    │   Coupling       │  K = K + mean(Q)
    └──────────────────┘
           │
           ▼
    ┌──────────────────┐
    │  QK L2 Norm      │  Q = Q / ||Q||₂
    │  + Temperature   │  K = K / ||K||₂
    │                  │  attn = (Q @ K.T) * τ
    └──────────────────┘
           │
           ▼
    ┌──────────────────┐
    │  GQA Attention   │  n_heads Q, n_kv_heads K/V
    │  (Compressed)    │  K,V shared across groups
    └──────────────────┘
           │
           ▼
    ┌──────────────────┐
    │  Value Shift     │  Half heads see V[t-1]
    │  (optional)      │  Temporal inductive bias
    └──────────────────┘
           │
           ▼
    ┌──────────────────┐
    │    Up-project    │  Linear: dim/C → dim
    └──────────────────┘
           │
           ▼
Output: [batch, seq, dim]
```

### Why It Works

1. **Compression preserves information**: The 4× compression is learned, not fixed
2. **Convolutions add locality**: Standard attention is position-agnostic; convs add local bias
3. **QK-mean coupling**: Prevents Q and K from drifting apart during training
4. **L2 norm stabilizes**: Prevents attention logits from exploding

---

## MoD: Mixture-of-Depths

**Paper**: [arXiv:2404.02258](https://arxiv.org/abs/2404.02258)

### Key Insight

Not all tokens need equal compute. In a sentence like "The quick brown fox jumps", the word "the" is predictable and needs less processing than "jumps".

### Architecture

```
Input x: [batch, seq, dim]
           │
           ▼
    ┌──────────────────┐
    │  Router          │  Linear: dim → 1
    │  r = σ(Wx)       │  Sigmoid probability
    └──────────────────┘
           │
           ▼
    ┌─────────┴─────────┐
    │                   │
    ▼ (Training)        ▼ (Inference)
┌────────────┐    ┌────────────────┐
│ Soft Route │    │   Hard Route   │
│ y = r·f(x) │    │ Select top-k   │
│ + (1-r)·x  │    │ k = 0.75 × seq │
└────────────┘    └────────────────┘
    │                   │
    └─────────┬─────────┘
              ▼
    ┌──────────────────┐
    │  Auxiliary Loss  │  L_aux = BCE(r, 0.75)
    │  (Training only) │  Keeps capacity at 75%
    └──────────────────┘
```

### Training vs Inference

| Aspect | Training | Inference |
|--------|----------|-----------|
| Routing | Soft (differentiable) | Hard (top-k) |
| All tokens | Processed (weighted) | Only 75% processed |
| Gradients | Flow through all | N/A |
| Speed | Slower (full compute) | Faster (25% skip) |

### Auxiliary Loss Scaling

For large models, the auxiliary loss must scale to remain effective:

```python
aux_loss_weight = 0.01 * (effective_layers / 32) * sqrt(dim / 768)
```

Without this, large models see aux_loss drowned out by CE loss, causing capacity collapse.

---

## MoR: Mixture-of-Recursions

**Paper**: [arXiv:2507.10524](https://arxiv.org/abs/2507.10524)

### Key Insight

Different positions in a sequence need different "thinking depth". MoR allows tokens to exit early or iterate more.

### Architecture

```
Input x: [batch, seq, dim]
           │
           ▼
┌──────────────────────────────────────────┐
│  For r in range(recursions):             │
│                                          │
│    ┌──────────────────┐                  │
│    │  Recursion Embed │  depth_emb[r]    │
│    │  x = x + emb[r]  │  Additive        │
│    └──────────────────┘                  │
│              │                           │
│              ▼                           │
│    ┌──────────────────┐                  │
│    │  MoR Router      │  μ = Linear(x)   │
│    │  Gaussian soft   │  p(r) ∝ N(r|μ,σ) │
│    └──────────────────┘                  │
│              │                           │
│              ▼                           │
│    ┌──────────────────┐                  │
│    │  CCGQA Block     │  Attention + FFN │
│    │  (weighted by    │                  │
│    │   Gaussian prob) │                  │
│    └──────────────────┘                  │
│              │                           │
│    Accumulate: out += p(r) × block(x)    │
│                                          │
└──────────────────────────────────────────┘
           │
           ▼
    ┌──────────────────┐
    │  Ponder Loss     │  Encourage early exit
    │  L = Σ r × p(r)  │  Penalize deep compute
    └──────────────────┘
```

### Layer-Aware Capacity

The MoR router adapts based on layer position:

```python
# Early layers: explore more, 40% base capacity
# Late layers: more decisive, 80% base capacity
layer_scale = 0.4 + 0.4 * (layer_idx / (n_layers - 1))
```

This reflects that early layers do feature extraction (needs exploration) while late layers do prediction (needs commitment).

### Depth Histogram

Each layer produces a histogram showing how computation is distributed:

```
Layer 0:  depth 1: ████████████████ 60%
          depth 2: ████████ 25%
          depth 3: ████ 10%
          depth 4: ██ 5%

Layer 23: depth 1: ████ 10%
          depth 2: ████████ 25%
          depth 3: ████████████ 35%
          depth 4: ████████████████ 30%
```

---

## Integration: The Complete HYDRA Stack

### Data Flow

```
Tokens → Embedding → [MoD Router → [MoR Router → CCGQA Block]×r ]×n → LM Head → Logits
```

### Loss Function

```python
total_loss = (
    ce_loss                           # Language modeling
    + aux_loss_weight * aux_loss      # MoD capacity control
    + ponder_weight * ponder_loss     # MoR depth regularization
)
```

### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `compression_factor` | 4 | Paper recommendation, good tradeoff |
| `capacity_ratio` | 0.75 | 25% compute savings, minimal quality loss |
| `recursions` | 4-5 | Matches effective depth of baseline |
| `n_kv_heads` | n_heads / 4 | 4× KV cache reduction |
| `aux_loss_weight` | Auto-scaled | See formula above |
| `ponder_weight` | 0.01 | Small, just to encourage efficiency |

---

## Scaling Laws

### Validated Scales

| Scale | Actual Params | aux_weight | MoD prob | Status |
|-------|---------------|------------|----------|--------|
| 100M | 100.5M | 0.0100 | 0.808 | ✅ |
| 250M | 216.3M | 0.0173 | 0.787 | ✅ |
| 500M | 569.6M | 0.0283 | 0.759 | ✅ |
| 750M | 926.8M | 0.0382 | 0.779 | ✅ |
| 900M | 1194.8M | 0.0408 | 0.847 | ✅ |
| 1B | 1420.4M | 0.0490 | 0.798 | ✅ |
| 1.5B | 2182.3M | 0.0685 | 0.982 | ✅ |

### 4B Predictions

Curve fitting (polynomial degree 2, R² > 0.95) predicts:

```python
# 4B model configuration
dim = 4096
n_mor_blocks = 40
recursions = 4
effective_layers = 160
aux_loss_weight = 0.115  # Predicted from scaling
```

---

## Implementation Details

### Memory Optimization

1. **Gradient checkpointing**: Recompute activations during backward
2. **KV cache compression**: CCGQA reduces cache by 4×, GQA by another 4×
3. **Sparse attention patterns**: MoD skips 25% of tokens

### Training Stability

1. **QK normalization**: Prevents attention explosion
2. **Aux loss scaling**: Maintains MoD capacity at scale
3. **Recursion embeddings**: Helps router distinguish depths

### Inference Optimization

1. **Hard MoD routing**: Skip 25% of tokens entirely
2. **Early exit (future)**: MoR could enable early layer exit
3. **KV cache sharing**: GQA reduces cache memory

---

## Future Directions

1. **Speculative decoding**: Use MoD router to predict which tokens to speculate
2. **Adaptive batch sizing**: Process "easy" tokens in larger batches
3. **Layer dropping**: Extend MoD to skip entire layers
4. **Distillation**: Train smaller models from HYDRA teacher

---

*Last updated: December 2025*
