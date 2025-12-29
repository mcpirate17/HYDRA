"""
Memory-efficient backward kernel for CCGQA attention.
Single-pass algorithm that recomputes forward instead of loading O.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _bwd_efficient_kernel(
    Q, K, V, LSE, dO, dQ, dK, dV,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_lb, stride_lh, stride_ln,
    B: tl.constexpr, H: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    """
    Memory-efficient backward pass.
    Single kernel that computes dQ, dK, dV together by iterating strategically.
    Recomputes O on-the-fly instead of loading, saving memory bandwidth.
    """
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    block_m = tl.program_id(2)  # Q block
    
    # Offsets
    qo_offset = batch_idx * stride_qb + head_idx * stride_qh
    kv_offset = batch_idx * stride_kb + head_idx * stride_kh
    l_offset = batch_idx * stride_lb + head_idx * stride_lh
    
    # Block indices for this Q chunk
    offs_m = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    mask_m = offs_m < N
    
    # Load Q block
    q_ptrs = Q + qo_offset + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)
    
    # Load LSE and dO for this Q block
    lse_ptrs = LSE + l_offset + offs_m * stride_ln
    lse = tl.load(lse_ptrs, mask=mask_m, other=0.0).to(tl.float32)
    
    do_ptrs = dO + qo_offset + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd
    do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)
    
    # Initialize dQ accumulator
    dq_acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    
    # Compute D = rowsum(dO * O) efficiently by recomputing O on-the-fly
    # D is needed for softmax backward
    D_acc = tl.zeros([BLOCK_M], dtype=tl.float32)
    
    scale = 1.0 / tl.sqrt(tl.cast(D, tl.float32))
    
    # Loop over K/V blocks
    num_blocks_n = tl.cdiv(N, BLOCK_N)
    
    # First pass: compute D by recomputing attention output
    for block_n in range(0, num_blocks_n):
        offs_n_iter = block_n * BLOCK_N + offs_n
        mask_n = offs_n_iter < N
        
        # Check causal masking
        skip_block = tl.constexpr(False)
        if IS_CAUSAL:
            skip_block = (block_m * BLOCK_M + BLOCK_M - 1) < (block_n * BLOCK_N)
        
        if not skip_block:
            # Load K, V - Match split-K layout exactly
            k_ptrs = K + kv_offset + offs_n_iter[None, :] * stride_kn + offs_d[:, None] * stride_kd
            k = tl.load(k_ptrs, mask=mask_n[None, :], other=0.0).to(tl.float32)
            
            v_ptrs = V + kv_offset + offs_n_iter[:, None] * stride_vn + offs_d[None, :] * stride_vd
            v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)
            
            # Recompute QK
            qk = tl.dot(q, k) * scale  # [BLOCK_M, BLOCK_N]
            
            if IS_CAUSAL:
                causal_mask = offs_m[:, None] >= offs_n_iter[None, :]
                qk = tl.where(causal_mask, qk, float("-inf"))
            
            # Recompute P
            p = tl.exp(qk - lse[:, None])
            
            # Recompute O chunk and compute D contribution
            o_chunk = tl.dot(p, v)
            D_acc += tl.sum(do * o_chunk, axis=1)
    
    # Second pass: compute gradients
    for block_n in range(0, num_blocks_n):
        offs_n_iter = block_n * BLOCK_N + offs_n
        mask_n = offs_n_iter < N
        
        skip_block = tl.constexpr(False)
        if IS_CAUSAL:
            skip_block = (block_m * BLOCK_M + BLOCK_M - 1) < (block_n * BLOCK_N)
        
        if not skip_block:
            # Load K, V again - Match split-K layout
            k_ptrs = K + kv_offset + offs_n_iter[None, :] * stride_kn + offs_d[:, None] * stride_kd
            k = tl.load(k_ptrs, mask=mask_n[None, :], other=0.0).to(tl.float32)
            
            v_ptrs = V + kv_offset + offs_n_iter[:, None] * stride_vn + offs_d[None, :] * stride_vd
            v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)
            
            # Recompute attention scores
            qk = tl.dot(q, k) * scale
            
            if IS_CAUSAL:
                causal_mask = offs_m[:, None] >= offs_n_iter[None, :]
                qk = tl.where(causal_mask, qk, float("-inf"))
            
            p = tl.exp(qk - lse[:, None])
            
            # Compute dP = dO @ V^T
            dp = tl.dot(do, v.trans())
            
            # Softmax backward: dS = P * (dP - D)
            ds = p * (dp - D_acc[:, None])
            ds = ds * scale
            
            # Accumulate dQ = dS @ K^T
            dq_acc += tl.dot(ds, k.trans())
            
            # For dK and dV, we need to use atomics since multiple Q blocks contribute
            # Compute dK = Q^T @ dS
            dk = tl.dot(q.trans(), ds)
            
            # Atomic add for dK
            dk_ptrs = dK + kv_offset + offs_n_iter[None, :] * stride_kn + offs_d[:, None] * stride_kd
            tl.atomic_add(dk_ptrs, dk, mask=mask_n[None, :])
            
            # Compute dV = P^T @ dO  
            dv = tl.dot(p.trans(), do)
            
            # Atomic add for dV
            dv_ptrs = dV + kv_offset + offs_n_iter[:, None] * stride_vn + offs_d[None, :] * stride_vd
            tl.atomic_add(dv_ptrs, dv, mask=mask_n[:, None])
    
    # Store dQ (no atomics needed)
    dq_ptrs = dQ + qo_offset + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd
    tl.store(dq_ptrs, dq_acc, mask=mask_m[:, None])

