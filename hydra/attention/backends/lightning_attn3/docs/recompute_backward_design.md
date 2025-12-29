# Lightning Attention-3: Recompute-Heavy Backward Algorithm

## Design Goals

1. **SRAM-bounded**: Shared memory scales with `O(d²)` where `d = head_dim`, NOT `O(N²)` or `O(BLOCK²)`
2. **Blackwell-safe**: Total shared memory < 101KB for SM 12.x
3. **Gradient-correct**: Mathematically equivalent to standard backward
4. **Minimal HBM traffic**: Recompute intermediates rather than reload from global memory

---

## Background: Lightning Attention Forward

Lightning Attention computes linear attention without decay:
```
O[i] = Σ_{j≤i} Q[i] · K[j]ᵀ · V[j]
```

Rewritten in chunked form with cumulative KV state:
```
For each chunk c:
    kv_state += K[c]ᵀ @ V[c]        # [d, d] accumulator
    O[c] = Q[c] @ kv_state          # [BLOCK, d] output (inter-chunk)
    O[c] += tril(Q[c] @ K[c]ᵀ) @ V[c]  # [BLOCK, d] intra-chunk causal
```

The problematic term is **intra-chunk**: computing `tril(Q @ Kᵀ)` naively creates a `[BLOCK, BLOCK]` matrix.

---

## Current Backward: Memory-Heavy

The existing `_bwd_intra_kernel` stores:
| Buffer | Shape | Size (BLOCK=64, d=64, fp32) |
|--------|-------|------------------------------|
| `qk_trans` | [BLOCK, BLOCK] | 16KB |
| `dqk` | [BLOCK, BLOCK] | 16KB |
| `index` | [BLOCK, BLOCK] | 16KB |
| `q, k, v` | [BLOCK, d] each | 12KB |
| `dq, dk, dv` | [BLOCK, d] each | 12KB |
| `do` | [BLOCK, d] | 4KB |
| **Total** | | ~76KB per block |

With CBLOCK sub-chunking overhead, this exceeds 101KB.

---

## Recompute-Heavy Design: Micro-Chunked Backward

### Core Insight

Instead of materializing `[BLOCK, BLOCK]` attention matrices, process in **micro-chunks** of size `CBLOCK` (e.g., 16) and **recompute attention scores on-the-fly**.

### Memory Budget (Target: <50KB)

| Buffer | Shape | Size (CBLOCK=16, d=64, fp32) |
|--------|-------|-------------------------------|
| `q_tile` | [CBLOCK, d] | 4KB |
| `k_tile` | [CBLOCK, d] | 4KB |
| `v_tile` | [CBLOCK, d] | 4KB |
| `do_tile` | [CBLOCK, d] | 4KB |
| `dq_acc` | [CBLOCK, d] | 4KB |
| `dk_acc` | [CBLOCK, d] | 4KB |
| `dv_acc` | [CBLOCK, d] | 4KB |
| `attn_tile` | [CBLOCK, CBLOCK] | 1KB |
| `dattn_tile` | [CBLOCK, CBLOCK] | 1KB |
| **Total** | | ~30KB |

Scales as `O(CBLOCK × d + CBLOCK²)` ≈ `O(d²)` when `CBLOCK ≤ d`.

---

## Algorithm: Recompute Intra-Chunk Backward

### Notation
- `c`: chunk index (0 to NUM_CHUNKS-1)
- `i, j`: micro-chunk indices within a chunk (0 to NUM_CBLOCKS-1)
- `Q[c,i]`: micro-chunk `i` of queries in chunk `c`, shape [CBLOCK, d]
- Causal mask: position `(c,i)` can attend to `(c,j)` iff `j ≤ i`

### Forward Recomputation (for gradient derivation)

The intra-chunk output for query micro-chunk `i`:
```
O_intra[c,i] = Σ_{j≤i} tril_micro(Q[c,i] @ K[c,j]ᵀ) @ V[c,j]
```

Where `tril_micro` applies causal masking:
- If `j < i`: full attention (no masking needed)
- If `j == i`: lower-triangular mask within micro-chunk

### Backward Pass Algorithm

#### Phase 1: Compute dV (easiest, no recomputation needed for attention)

For each micro-chunk `j`:
```
dV[c,j] = 0
for i in range(j, NUM_CBLOCKS):  # All queries that attend to this K,V
    # Recompute attention: A[i,j] = Q[c,i] @ K[c,j]ᵀ
    A_ij = Q[c,i] @ K[c,j]ᵀ  # [CBLOCK, CBLOCK]
    if i == j:
        A_ij = tril(A_ij)  # Apply causal within micro-chunk
    
    # dV[j] += Aᵀ[i,j] @ dO[i]
    dV[c,j] += A_ij.T @ dO[c,i]
```

#### Phase 2: Compute dQ and dK (requires recomputing attention)

For each query micro-chunk `i`:
```
dQ[c,i] = 0
for j in range(0, i+1):  # All K,V that this query attends to
    # Recompute attention
    A_ij = Q[c,i] @ K[c,j]ᵀ  # [CBLOCK, CBLOCK]
    if i == j:
        A_ij = tril(A_ij)
    
    # Compute dA from dO and V
    dA_ij = dO[c,i] @ V[c,j].T  # [CBLOCK, CBLOCK]
    if i == j:
        dA_ij = tril(dA_ij)  # Mask gradient too
    
    # dQ[i] += dA[i,j] @ K[j]
    dQ[c,i] += dA_ij @ K[c,j]
    
    # dK[j] += dAᵀ[i,j] @ Q[i]  (accumulated across all i)
    dK[c,j] += dA_ij.T @ Q[c,i]
```

### Gradient Accumulation Safety

**Problem**: dK[j] is updated by multiple query micro-chunks (all i ≥ j).

**Solution**: Two-phase approach:
1. **Phase 1 (dV)**: Single writer per j, no conflicts
2. **Phase 2 (dQ, dK)**: 
   - dQ: Single writer per i, no conflicts
   - dK: Use atomic adds OR sequential accumulation with explicit sync

**Recommended**: Sequential inner loop for dK accumulation (avoids atomics):
```python
# Process in order: for each K micro-chunk j
for j in range(NUM_CBLOCKS):
    dK_j_acc = zeros([CBLOCK, d])
    for i in range(j, NUM_CBLOCKS):  # All queries attending to j
        # Recompute A_ij, dA_ij
        ...
        dK_j_acc += dA_ij.T @ Q[c,i]
    dK[c,j] = dK_j_acc  # Single write
```

---

## Inter-Chunk Backward (KV State)

The inter-chunk component uses cumulative KV state:
```
O_inter[c] = Q[c] @ kv_state[c-1]
```

Where `kv_state[c] = Σ_{c'≤c} K[c']ᵀ @ V[c']`.

### Backward for Inter-Chunk

```
# Forward saved: kv_state at each chunk boundary
# Backward:
dkv_state = 0
for c in reversed(range(NUM_CHUNKS)):
    # dQ_inter[c] = dO[c] @ kv_state[c-1].T
    dQ[c] += dO[c] @ kv_state[c-1].T
    
    # dkv_state += Q[c].T @ dO[c]
    dkv_state += Q[c].T @ dO[c]
    
    # dK[c], dV[c] from dkv_state
    # dkv = K.T @ V, so dK += dkv @ V.T, dV += K @ dkv
    dK[c] += dkv_state @ V[c].T  # Note: this is accumulated
    dV[c] += K[c] @ dkv_state
```

**Memory**: Only need `dkv_state` [d, d] = 16KB for d=64.

---

## Triton Kernel Structure

```python
@triton.jit
def _bwd_intra_recompute_kernel(
    Q, K, V, dO,  # Inputs [B, H, N, d]
    dQ, dK, dV,    # Outputs [B, H, N, d]
    stride_qb, stride_qh, stride_qn, stride_qd,
    N: tl.constexpr,
    d: tl.constexpr,
    CBLOCK: tl.constexpr,  # Micro-chunk size (e.g., 16)
    NUM_CBLOCKS: tl.constexpr,
):
    # Grid: (B*H, NUM_CHUNKS)
    off_bh = tl.program_id(0)
    off_chunk = tl.program_id(1)
    
    # Pointers to this chunk's data
    chunk_start = off_chunk * BLOCK
    
    # --- Phase 1: Compute dV ---
    for j in range(NUM_CBLOCKS):
        dv_acc = tl.zeros([CBLOCK, d], dtype=tl.float32)
        k_j = load_tile(K, chunk_start + j * CBLOCK)  # [CBLOCK, d]
        v_j = load_tile(V, chunk_start + j * CBLOCK)
        
        for i in range(j, NUM_CBLOCKS):
            q_i = load_tile(Q, chunk_start + i * CBLOCK)
            do_i = load_tile(dO, chunk_start + i * CBLOCK)
            
            # Recompute attention
            a_ij = tl.dot(q_i, tl.trans(k_j))  # [CBLOCK, CBLOCK]
            if i == j:
                a_ij = apply_causal_mask(a_ij)
            
            # dV += A.T @ dO
            dv_acc += tl.dot(tl.trans(a_ij), do_i)
        
        store_tile(dV, chunk_start + j * CBLOCK, dv_acc)
    
    # --- Phase 2: Compute dQ and dK ---
    for j in range(NUM_CBLOCKS):
        dk_acc = tl.zeros([CBLOCK, d], dtype=tl.float32)
        k_j = load_tile(K, chunk_start + j * CBLOCK)
        v_j = load_tile(V, chunk_start + j * CBLOCK)
        
        for i in range(j, NUM_CBLOCKS):
            q_i = load_tile(Q, chunk_start + i * CBLOCK)
            do_i = load_tile(dO, chunk_start + i * CBLOCK)
            
            # Recompute attention
            a_ij = tl.dot(q_i, tl.trans(k_j))
            
            # Compute dA = dO @ V.T
            da_ij = tl.dot(do_i, tl.trans(v_j))
            
            if i == j:
                a_ij = apply_causal_mask(a_ij)
                da_ij = apply_causal_mask(da_ij)
            
            # dK[j] += dA.T @ Q[i]
            dk_acc += tl.dot(tl.trans(da_ij), q_i)
            
            # dQ[i] += dA @ K[j] (need separate accumulator per i)
            if i == j:  # Only compute dQ when processing diagonal
                dq_acc = tl.zeros([CBLOCK, d], dtype=tl.float32)
            dq_acc += tl.dot(da_ij, k_j)
            if i == NUM_CBLOCKS - 1 or True:  # Store dQ for this i
                # Need to handle dQ accumulation across j...
                pass
        
        store_tile(dK, chunk_start + j * CBLOCK, dk_acc)
```

---

## Complexity Analysis

### Compute
- Original: `O(N² × d)` per chunk
- Recompute: `O(N² × d × 2)` per chunk (2x for recomputing attention twice)

### Memory
- Original: `O(BLOCK² + BLOCK × d)` shared memory
- Recompute: `O(CBLOCK² + CBLOCK × d)` shared memory

### HBM Traffic
- Original: Read Q,K,V,dO once, write dQ,dK,dV once
- Recompute: Same (no extra global memory access)

---

## Implementation Phases

### Phase 1: Basic Correctness
1. Implement naive double-loop kernel
2. Verify gradient correctness with finite differences
3. Benchmark against original

### Phase 2: Optimization
1. Fuse Phase 1 and Phase 2 where possible
2. Optimize register usage
3. Tune CBLOCK for Blackwell

### Phase 3: Integration
1. Replace `_bwd_intra_kernel` in `lightning_attn3_no_decay.py`
2. Add runtime selection based on GPU architecture
3. Full test suite validation

---

## Key Invariants

1. **Causal Mask**: Only apply within diagonal micro-chunks (i == j)
2. **Accumulation Order**: dK requires summing over all attending queries
3. **Numerical Precision**: Accumulate in fp32, convert at store
4. **Shared Memory**: Never allocate [BLOCK, BLOCK] matrices

---

## Open Questions

1. **Optimal CBLOCK**: Trade-off between parallelism and recomputation
2. **Warp Specialization**: Can we pipeline dV and dQ/dK phases?
3. **Register Pressure**: How many tiles can we hold simultaneously?
4. **Fusion with Inter-Chunk**: Can we merge inter/intra backward kernels?
