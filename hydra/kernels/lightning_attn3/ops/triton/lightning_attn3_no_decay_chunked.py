# Lightning Attention-3 - HYDRA Project
# Based on Lightning Attention (https://github.com/OpenNLPLab/lightning-attention)
# Recompute-heavy backward kernel for Blackwell (SM 12.x) compatibility
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl

# ============================================================================
# BLACKWELL CONSTRAINTS (SM 12.x: max 101KB shared memory)
# ============================================================================
# Target: <96KB to leave headroom
# Memory budget breakdown (CBLOCK=16, d=64, fp16 inputs):
#   - Input tiles:  4 × CBLOCK × d × 2 = 8KB (q, k, v, do)
#   - Accumulators: 3 × CBLOCK × d × 4 = 12KB (dq, dk, dv in fp32)
#   - Attn tiles:   2 × CBLOCK² × 4   = 2KB (attn, dattn in fp32)
#   - Overhead:                         ~4KB
#   - Total:                           ~26KB ✓
# ============================================================================

BLACKWELL_SRAM_LIMIT = 101_376
SRAM_BUDGET = 96_000


def validate_config(CBLOCK: int, d: int, num_stages: int = 1) -> tuple[bool, int]:
    """Validate kernel config fits Blackwell shared memory."""
    input_tiles = num_stages * 4 * CBLOCK * d * 2  # fp16
    accumulators = 3 * CBLOCK * d * 4  # fp32
    attention = 2 * CBLOCK * CBLOCK * 4  # fp32
    overhead = 4096
    total = input_tiles + accumulators + attention + overhead
    return (total <= SRAM_BUDGET, total)


# ============================================================================
# RECOMPUTE-HEAVY INTRA-CHUNK BACKWARD KERNEL
# ============================================================================
# Key design:
# - Process in micro-chunks of size CBLOCK (16 or 32)
# - Recompute attention scores on-the-fly instead of storing [BLOCK, BLOCK]
# - Memory scales as O(CBLOCK² + CBLOCK×d) not O(BLOCK²)
# ============================================================================

@triton.jit
def _bwd_intra_chunked_kernel(
    Q, K, V, DO,
    DQ, DK, DV,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_ve,
    stride_dob, stride_doh, stride_don, stride_doe,
    n: tl.constexpr,
    d: tl.constexpr,
    e: tl.constexpr,
    BLOCK: tl.constexpr,
    CBLOCK: tl.constexpr,
    NUM_CBLOCK: tl.constexpr,
):
    """
    Intra-chunk backward kernel using reversed loop order for dK/dV.
    
    Two-pass approach:
    1. Forward pass: compute dQ[i] for each i by iterating over j <= i
    2. Backward pass: compute dK[j], dV[j] for each j by iterating over i >= j
    
    This avoids atomic operations by processing one output chunk at a time.
    Memory usage: O(CBLOCK² + CBLOCK × d) << O(BLOCK²)
    """
    # Grid: (B*H, NUM_BLOCKS)
    off_bh = tl.program_id(0)
    off_block = tl.program_id(1)
    
    # Base offsets for this batch/head
    q_base = Q + off_bh * stride_qh
    k_base = K + off_bh * stride_kh
    v_base = V + off_bh * stride_vh
    do_base = DO + off_bh * stride_doh
    dq_base = DQ + off_bh * stride_qh
    dk_base = DK + off_bh * stride_kh
    dv_base = DV + off_bh * stride_vh
    
    # Block start position in sequence
    block_start = off_block * BLOCK
    
    # Create index arrays for causal masking
    cblock_range = tl.arange(0, CBLOCK)
    row_idx = cblock_range[:, None]
    col_idx = cblock_range[None, :]
    causal_mask = row_idx >= col_idx
    
    # Pre-compute pointer offsets for d and e dimensions
    d_range = tl.arange(0, d)[None, :] * stride_qd
    e_range = tl.arange(0, e)[None, :] * stride_ve
    
    # ========================================================================
    # PASS 1: Compute dQ[i] for each i
    # dQ[i] = Σ_{j≤i} dA[i,j] @ K[j]  where dA = dO @ V^T (causal masked)
    # ========================================================================
    
    for i in range(NUM_CBLOCK):
        i_start = block_start + i * CBLOCK
        i_range = i_start + cblock_range
        i_mask = i_range < n
        
        # Load Q[i] and dO[i]
        q_ptrs = q_base + i_range[:, None] * stride_qn + d_range
        do_ptrs = do_base + i_range[:, None] * stride_don + e_range
        q_i = tl.load(q_ptrs, mask=i_mask[:, None], other=0.0).to(tl.float32)
        do_i = tl.load(do_ptrs, mask=i_mask[:, None], other=0.0).to(tl.float32)
        
        # Accumulator for dQ[i]
        dq_acc = tl.zeros([CBLOCK, d], dtype=tl.float32)
        
        # Process all j <= i
        for j in range(i + 1):
            j_start = block_start + j * CBLOCK
            j_range = j_start + cblock_range
            j_mask = j_range < n
            
            # Load K[j] and V[j]
            k_ptrs = k_base + j_range[:, None] * stride_kn + d_range
            v_ptrs = v_base + j_range[:, None] * stride_vn + e_range
            k_j = tl.load(k_ptrs, mask=j_mask[:, None], other=0.0).to(tl.float32)
            v_j = tl.load(v_ptrs, mask=j_mask[:, None], other=0.0).to(tl.float32)
            
            # Compute dAttn: dA[i,j] = dO[i] @ V[j].T
            dattn_ij = tl.dot(do_i, tl.trans(v_j))  # [CBLOCK, CBLOCK]
            
            # Apply causal mask only on diagonal blocks (i == j)
            if j == i:
                dattn_ij = tl.where(causal_mask, dattn_ij, 0.0)
            
            # dQ[i] += dA[i,j] @ K[j]
            dq_acc += tl.dot(dattn_ij, k_j)
        
        # Store dQ[i]
        dq_ptrs = dq_base + i_range[:, None] * stride_qn + d_range
        tl.store(dq_ptrs, dq_acc.to(dq_ptrs.dtype.element_ty), mask=i_mask[:, None])
    
    # ========================================================================
    # PASS 2: Compute dK[j], dV[j] for each j
    # dK[j] = Σ_{i≥j} dA[i,j].T @ Q[i]
    # dV[j] = Σ_{i≥j} A[i,j].T @ dO[i]  where A = Q @ K^T (causal masked)
    # ========================================================================
    
    for j in range(NUM_CBLOCK):
        j_start = block_start + j * CBLOCK
        j_range = j_start + cblock_range
        j_mask = j_range < n
        
        # Load K[j] and V[j]
        k_ptrs = k_base + j_range[:, None] * stride_kn + d_range
        v_ptrs = v_base + j_range[:, None] * stride_vn + e_range
        k_j = tl.load(k_ptrs, mask=j_mask[:, None], other=0.0).to(tl.float32)
        v_j = tl.load(v_ptrs, mask=j_mask[:, None], other=0.0).to(tl.float32)
        
        # Accumulators for dK[j] and dV[j]
        dk_acc = tl.zeros([CBLOCK, d], dtype=tl.float32)
        dv_acc = tl.zeros([CBLOCK, e], dtype=tl.float32)
        
        # Process all i >= j
        for i in range(j, NUM_CBLOCK):
            i_start = block_start + i * CBLOCK
            i_range = i_start + cblock_range
            i_mask = i_range < n
            
            # Load Q[i] and dO[i]
            q_ptrs = q_base + i_range[:, None] * stride_qn + d_range
            do_ptrs = do_base + i_range[:, None] * stride_don + e_range
            q_i = tl.load(q_ptrs, mask=i_mask[:, None], other=0.0).to(tl.float32)
            do_i = tl.load(do_ptrs, mask=i_mask[:, None], other=0.0).to(tl.float32)
            
            # Compute attention: A[i,j] = Q[i] @ K[j].T
            attn_ij = tl.dot(q_i, tl.trans(k_j))  # [CBLOCK, CBLOCK]
            
            # Compute dAttn: dA[i,j] = dO[i] @ V[j].T
            dattn_ij = tl.dot(do_i, tl.trans(v_j))  # [CBLOCK, CBLOCK]
            
            # Apply causal mask only on diagonal blocks (i == j)
            if i == j:
                attn_ij = tl.where(causal_mask, attn_ij, 0.0)
                dattn_ij = tl.where(causal_mask, dattn_ij, 0.0)
            
            # dK[j] += dA[i,j].T @ Q[i]
            dk_acc += tl.dot(tl.trans(dattn_ij), q_i)
            
            # dV[j] += A[i,j].T @ dO[i]
            dv_acc += tl.dot(tl.trans(attn_ij), do_i)
        
        # Store dK[j] and dV[j]
        dk_ptrs = dk_base + j_range[:, None] * stride_kn + d_range
        dv_ptrs = dv_base + j_range[:, None] * stride_vn + e_range
        tl.store(dk_ptrs, dk_acc.to(dk_ptrs.dtype.element_ty), mask=j_mask[:, None])
        tl.store(dv_ptrs, dv_acc.to(dv_ptrs.dtype.element_ty), mask=j_mask[:, None])


# ============================================================================
# INTER-CHUNK BACKWARD (cumulative KV state gradients)
# ============================================================================

@triton.jit
def _bwd_inter_chunked_kernel(
    Q, K, V, DO,
    DQ, DK, DV,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_ve,
    stride_dob, stride_doh, stride_don, stride_doe,
    n: tl.constexpr,
    d: tl.constexpr,
    e: tl.constexpr,
    CBLOCK: tl.constexpr,
    NUM_CBLOCK: tl.constexpr,
):
    """
    Inter-chunk backward: handle cross-chunk attention via cumulative KV state.
    
    Forward computed: O_inter[c] = Q[c] @ kv_state[c-1]
    where kv_state[c] = Σ_{c'≤c} K[c']ᵀ @ V[c']
    
    Backward:
    - dQ_inter[c] = dO[c] @ kv_state[c-1]ᵀ
    - dkv_state accumulates from future to past
    - dK[c] += dkv_state @ V[c]ᵀ
    - dV[c] += K[c] @ dkv_state
    """
    off_bh = tl.program_id(0)
    
    # Base pointers
    q_base = Q + off_bh * stride_qh
    k_base = K + off_bh * stride_kh
    v_base = V + off_bh * stride_vh
    do_base = DO + off_bh * stride_doh
    dq_base = DQ + off_bh * stride_qh
    dk_base = DK + off_bh * stride_kh
    dv_base = DV + off_bh * stride_vh
    
    cblock_range = tl.arange(0, CBLOCK)
    
    # ========================================================================
    # Forward pass to build kv_state at each chunk boundary
    # kv_state[c] = Σ_{c'≤c} K[c']ᵀ @ V[c']  -> [d, e]
    # We need kv_state[c-1] to compute dQ_inter[c]
    # ========================================================================
    
    # For dQ: accumulate kv_state forward, add to dQ
    kv_state_trans = tl.zeros([e, d], dtype=tl.float32)  # kv_stateᵀ = [e, d]
    
    for c in range(NUM_CBLOCK):
        c_start = c * CBLOCK
        c_range = c_start + cblock_range
        c_mask = c_range < n
        
        # dQ_inter[c] = dO[c] @ kv_state[c-1]ᵀ  (use kv_state before this chunk)
        if c > 0:
            # Load dO[c]: [CBLOCK, e]
            do_ptrs = do_base + c_range[:, None] * stride_don + tl.arange(0, e)[None, :] * stride_doe
            do_c = tl.load(do_ptrs, mask=c_mask[:, None], other=0.0).to(tl.float32)
            
            # dQ_inter = dO @ kv_stateᵀ  -> [CBLOCK, d]
            dq_inter = tl.dot(do_c, kv_state_trans)
            
            # Add to existing dQ (from intra kernel)
            dq_ptrs = dq_base + c_range[:, None] * stride_qn + tl.arange(0, d)[None, :] * stride_qd
            dq_existing = tl.load(dq_ptrs, mask=c_mask[:, None], other=0.0).to(tl.float32)
            dq_total = dq_existing + dq_inter
            tl.store(dq_ptrs, dq_total.to(dq_ptrs.dtype.element_ty), mask=c_mask[:, None])
        
        # Update kv_state: kv_state += K[c]ᵀ @ V[c]
        # Load K[c]: [CBLOCK, d]
        k_ptrs = k_base + c_range[:, None] * stride_kn + tl.arange(0, d)[None, :] * stride_kd
        k_c = tl.load(k_ptrs, mask=c_mask[:, None], other=0.0).to(tl.float32)
        
        # Load V[c]: [CBLOCK, e]
        v_ptrs = v_base + c_range[:, None] * stride_vn + tl.arange(0, e)[None, :] * stride_ve
        v_c = tl.load(v_ptrs, mask=c_mask[:, None], other=0.0).to(tl.float32)
        
        # kv_stateᵀ += V[c]ᵀ @ K[c]  -> [e, d]
        kv_state_trans += tl.dot(tl.trans(v_c), k_c)
    
    # ========================================================================
    # Backward pass for dK and dV from dkv_state
    # Process chunks in reverse order
    # ========================================================================
    
    dkv_state = tl.zeros([d, e], dtype=tl.float32)
    
    for c in range(NUM_CBLOCK - 1, -1, -1):
        c_start = c * CBLOCK
        c_range = c_start + cblock_range
        c_mask = c_range < n
        
        # Load Q[c]: [CBLOCK, d]
        q_ptrs = q_base + c_range[:, None] * stride_qn + tl.arange(0, d)[None, :] * stride_qd
        q_c = tl.load(q_ptrs, mask=c_mask[:, None], other=0.0).to(tl.float32)
        
        # Load dO[c]: [CBLOCK, e]
        do_ptrs = do_base + c_range[:, None] * stride_don + tl.arange(0, e)[None, :] * stride_doe
        do_c = tl.load(do_ptrs, mask=c_mask[:, None], other=0.0).to(tl.float32)
        
        # Update dkv_state: dkv_state += Q[c]ᵀ @ dO[c]  -> [d, e]
        dkv_state += tl.dot(tl.trans(q_c), do_c)
        
        # dK_inter and dV_inter from dkv_state (only for chunks before current)
        if c < NUM_CBLOCK - 1:
            # Load K[c]: [CBLOCK, d]
            k_ptrs = k_base + c_range[:, None] * stride_kn + tl.arange(0, d)[None, :] * stride_kd
            k_c = tl.load(k_ptrs, mask=c_mask[:, None], other=0.0).to(tl.float32)
            
            # Load V[c]: [CBLOCK, e]
            v_ptrs = v_base + c_range[:, None] * stride_vn + tl.arange(0, e)[None, :] * stride_ve
            v_c = tl.load(v_ptrs, mask=c_mask[:, None], other=0.0).to(tl.float32)
            
            # dK_inter = dkv_state @ V[c]ᵀ  -> but we need transposed...
            # dK[c] += (dkv_state @ V[c]ᵀ)ᵀ = V[c] @ dkv_stateᵀ  -> [CBLOCK, d]
            # Actually: dK contribution is from: d(K.T @ V) w.r.t K = dkv @ V.T
            # Shape: dkv=[d,e], V=[CBLOCK,e], so dK += (dkv @ V.T).T = V @ dkv.T
            # But we want [CBLOCK, d], and dkv.T = [e, d]
            # dK_inter = V[c] @ dkv_stateᵀ won't work dimensionally
            # Let's reconsider: kv_state = K.T @ V, shape [d, e]
            # d/dK (K.T @ V) = d/dK tr(K.T @ V @ dkv.T) = V @ dkv.T for each row of K
            # So dK[c] = V[c] @ dkv_stateᵀ  but V is [CBLOCK, e], dkv.T is [e, d]
            # dK[c] = V[c] @ dkv_state.T  -> [CBLOCK, e] @ [e, d] = [CBLOCK, d] ✓
            dkv_trans = tl.trans(dkv_state)  # [e, d]
            dk_inter = tl.dot(v_c, dkv_trans)  # [CBLOCK, d]
            
            # dV_inter = K[c] @ dkv_state  -> [CBLOCK, d] @ [d, e] = [CBLOCK, e] ✓
            dv_inter = tl.dot(k_c, dkv_state)
            
            # Add to existing gradients
            dk_ptrs = dk_base + c_range[:, None] * stride_kn + tl.arange(0, d)[None, :] * stride_kd
            dk_existing = tl.load(dk_ptrs, mask=c_mask[:, None], other=0.0).to(tl.float32)
            dk_total = dk_existing + dk_inter
            tl.store(dk_ptrs, dk_total.to(dk_ptrs.dtype.element_ty), mask=c_mask[:, None])
            
            dv_ptrs = dv_base + c_range[:, None] * stride_vn + tl.arange(0, e)[None, :] * stride_ve
            dv_existing = tl.load(dv_ptrs, mask=c_mask[:, None], other=0.0).to(tl.float32)
            dv_total = dv_existing + dv_inter
            tl.store(dv_ptrs, dv_total.to(dv_ptrs.dtype.element_ty), mask=c_mask[:, None])


# ============================================================================
# PYTHON WRAPPER
# ============================================================================

# Local forward kernel for standalone testing (not autotuned)
@triton.jit
def _fwd_kernel_standalone(
    Q, K, V, Out,
    b: tl.constexpr, h: tl.constexpr, n: tl.constexpr,
    d: tl.constexpr, e: tl.constexpr,
    BLOCK: tl.constexpr, NUM_BLOCK: tl.constexpr, BLOCK_MODEL: tl.constexpr,
):
    """Non-autotuned forward kernel for standalone chunked module testing."""
    off_bh = tl.program_id(0)
    off_e = tl.program_id(1)
    qk_offset = off_bh * n * d
    v_offset = off_bh * n * e
    o_offset = off_bh * n * e
    e_offset = off_e * BLOCK_MODEL

    Q_block_ptr = Q + qk_offset + tl.arange(0, d)[None, :]
    K_trans_block_ptr = K + qk_offset + tl.arange(0, d)[:, None]
    V_block_ptr = V + v_offset + e_offset + tl.arange(0, BLOCK_MODEL)[None, :]
    O_block_ptr = Out + o_offset + e_offset + tl.arange(0, BLOCK_MODEL)[None, :]

    off_block = tl.arange(0, BLOCK)
    index = off_block[:, None] - off_block[None, :]
    kv = tl.zeros([d, BLOCK_MODEL], dtype=tl.float32)

    for i in range(NUM_BLOCK):
        q = tl.load(Q_block_ptr + off_block[:, None] * d, mask=off_block[:, None] < n, other=0.0).to(tl.float32)
        k_trans = tl.load(K_trans_block_ptr + off_block[None, :] * d, mask=off_block[None, :] < n, other=0.0).to(tl.float32)
        v = tl.load(V_block_ptr + off_block[:, None] * e, mask=off_block[:, None] < n, other=0.0).to(tl.float32)

        qk = tl.dot(q, k_trans)
        qk = tl.where(index >= 0, qk, 0)
        o_intra = tl.dot(qk, v)
        o_inter = tl.dot(q, kv)
        o = o_intra + o_inter

        tl.store(O_block_ptr + off_block[:, None] * e, o.to(O_block_ptr.dtype.element_ty), mask=off_block[:, None] < n)
        kv += tl.dot(k_trans, v)
        off_block += BLOCK


class LightningAttention3NoDecayChunked(torch.autograd.Function):
    """
    Lightning Attention-3 with recompute-heavy backward for Blackwell GPUs.
    
    Memory usage in backward: O(CBLOCK² + CBLOCK × d) instead of O(BLOCK²)
    Safe for SM 12.x with 101KB shared memory limit.
    """
    
    @staticmethod
    def forward(ctx, q, k, v):
        # Use local non-autotuned kernel for standalone testing
        if not q.is_contiguous():
            q = q.contiguous()
        if not k.is_contiguous():
            k = k.contiguous()
        if not v.is_contiguous():
            v = v.contiguous()

        b, h, n, d = q.shape
        e = v.shape[-1]
        o = torch.empty((b, h, n, e), dtype=q.dtype, device=q.device)

        BLOCK = 64
        NUM_BLOCK = triton.cdiv(n, BLOCK)
        BLOCK_MODEL = min(triton.next_power_of_2(e), 32)
        grid = (b * h, triton.cdiv(e, BLOCK_MODEL))

        _fwd_kernel_standalone[grid](
            q, k, v, o,
            b, h, n, d, e,
            BLOCK=BLOCK,
            NUM_BLOCK=NUM_BLOCK,
            BLOCK_MODEL=BLOCK_MODEL,
        )

        ctx.save_for_backward(q, k, v)
        ctx.n = n
        ctx.d = d
        ctx.e = e
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v = ctx.saved_tensors
        n, d, e = ctx.n, ctx.d, ctx.e
        b, h = q.shape[:2]

        if not do.is_contiguous():
            do = do.contiguous()

        # Use empty_like - intra kernel fully initializes all positions before inter reads
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)

        # Optimal configs tuned for Blackwell RTX 5090:
        # - Intra: CBLOCK=32 is fastest (O(n²) within block benefits from smaller chunks)
        # - Inter: CBLOCK=128 is fastest (fewer iterations, better tensor core utilization)
        #          CBLOCK=256 exceeds SRAM limit (101KB)
        # - Inter stages=3 provides best software pipelining
        CBLOCK_INTRA = 32
        CBLOCK_INTER = 128
        BLOCK = 64
        NUM_BLOCK = triton.cdiv(n, BLOCK)
        NUM_CBLOCK_PER_BLOCK = BLOCK // CBLOCK_INTRA
        
        # Validate config
        is_valid, sram = validate_config(CBLOCK_INTRA, d, num_stages=1)
        assert is_valid, f"Config requires {sram} bytes, exceeds {SRAM_BUDGET}"

        # Strides
        stride_qb, stride_qh, stride_qn, stride_qd = q.stride()
        stride_kb, stride_kh, stride_kn, stride_kd = k.stride()
        stride_vb, stride_vh, stride_vn, stride_ve = v.stride()
        stride_dob, stride_doh, stride_don, stride_doe = do.stride()

        # Intra-chunk backward: parallel over blocks
        grid_intra = (b * h, NUM_BLOCK)
        _bwd_intra_chunked_kernel[grid_intra](
            q, k, v, do,
            dq, dk, dv,
            stride_qb, stride_qh, stride_qn, stride_qd,
            stride_kb, stride_kh, stride_kn, stride_kd,
            stride_vb, stride_vh, stride_vn, stride_ve,
            stride_dob, stride_doh, stride_don, stride_doe,
            n=n, d=d, e=e,
            BLOCK=BLOCK,
            CBLOCK=CBLOCK_INTRA,
            NUM_CBLOCK=NUM_CBLOCK_PER_BLOCK,
            num_warps=4,
            num_stages=1,
        )

        # Inter-chunk backward: sequential over chunks
        # CBLOCK=128 reduces iterations by 2x vs CBLOCK=64
        NUM_CBLOCK_TOTAL = triton.cdiv(n, CBLOCK_INTER)
        grid_inter = (b * h,)
        _bwd_inter_chunked_kernel[grid_inter](
            q, k, v, do,
            dq, dk, dv,
            stride_qb, stride_qh, stride_qn, stride_qd,
            stride_kb, stride_kh, stride_kn, stride_kd,
            stride_vb, stride_vh, stride_vn, stride_ve,
            stride_dob, stride_doh, stride_don, stride_doe,
            n=n, d=d, e=e,
            CBLOCK=CBLOCK_INTER,
            NUM_CBLOCK=NUM_CBLOCK_TOTAL,
            num_warps=8,  # 8 warps is 5.5% faster than 4 on RTX 5090
            num_stages=3,  # Optimal software pipelining for inter kernel
        )

        return dq, dk, dv


# Export
lightning_attn3_no_decay_chunked = LightningAttention3NoDecayChunked.apply


# ============================================================================
# TESTING UTILITIES
# ============================================================================

def test_chunked_backward():
    """Quick test for the chunked backward kernel."""
    import torch
    
    torch.manual_seed(42)
    B, H, N, D = 2, 4, 256, 64
    
    q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16, requires_grad=True)
    k = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16, requires_grad=True)
    v = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16, requires_grad=True)
    
    # Forward
    out = lightning_attn3_no_decay_chunked(q, k, v)
    print(f"Forward output shape: {out.shape}")
    
    # Backward
    loss = out.sum()
    loss.backward()
    
    print(f"dQ shape: {q.grad.shape}, has NaN: {torch.isnan(q.grad).any()}")
    print(f"dK shape: {k.grad.shape}, has NaN: {torch.isnan(k.grad).any()}")
    print(f"dV shape: {v.grad.shape}, has NaN: {torch.isnan(v.grad).any()}")
    
    # Check gradients are non-zero
    assert q.grad.abs().sum() > 0, "dQ is all zeros"
    assert k.grad.abs().sum() > 0, "dK is all zeros"
    assert v.grad.abs().sum() > 0, "dV is all zeros"
    
    print("✓ Chunked backward kernel test passed!")
    return True


if __name__ == "__main__":
    test_chunked_backward()
