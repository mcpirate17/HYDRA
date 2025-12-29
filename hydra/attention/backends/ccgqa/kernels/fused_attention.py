"""
Triton-accelerated CCGQA attention kernels.
Phase 2: Forward + Backward fused kernels with split-K backward.
Phase 3: Sequence-length-aware block size selection for optimal performance.
"""

import torch
import triton
import triton.language as tl

from .autotune_config import get_block_sizes


@triton.jit
def _fwd_gqa_kernel(
    Q, K, V, Out, L,  # Pointers
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    stride_lb, stride_lh, stride_ln,
    B: tl.constexpr,
    H_Q: tl.constexpr,
    H_KV: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    SAVE_LSE: tl.constexpr,
):
    """GQA forward kernel: one KV head program computes GROUP_SIZE Q heads, reusing K/V loads."""
    batch_idx = tl.program_id(0)
    head_kv = tl.program_id(1)
    block_m = tl.program_id(2)

    offs_m = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    mask_m = offs_m < N

    q_base = Q + batch_idx * stride_qb
    k_offset = batch_idx * stride_kb + head_kv * stride_kh
    v_offset = batch_idx * stride_vb + head_kv * stride_vh
    o_base = Out + batch_idx * stride_ob
    l_base = L + batch_idx * stride_lb

    heads = head_kv * GROUP_SIZE + tl.arange(0, GROUP_SIZE)

    # Load Q for all grouped heads into a (G, BM, BD) tensor.
    q_ptrs = (
        q_base
        + heads[:, None, None] * stride_qh
        + offs_m[None, :, None] * stride_qn
        + offs_d[None, None, :] * stride_qd
    )
    q = tl.load(q_ptrs, mask=mask_m[None, :, None], other=0.0).to(tl.float32)

    m_i = tl.zeros([GROUP_SIZE, BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([GROUP_SIZE, BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([GROUP_SIZE, BLOCK_M, BLOCK_D], dtype=tl.float32)

    scale = 1.0 / tl.sqrt(tl.cast(D, tl.float32))
    num_blocks = tl.cdiv(N, BLOCK_N)
    for block_n in range(0, num_blocks):
        offs_n_iter = block_n * BLOCK_N + offs_n
        mask_n = offs_n_iter < N

        k_ptrs = K + k_offset + offs_n_iter[None, :] * stride_kn + offs_d[:, None] * stride_kd
        k = tl.load(k_ptrs, mask=mask_n[None, :], other=0.0).to(tl.float32)

        v_ptrs = V + v_offset + offs_n_iter[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)

        # Compute qk for all heads at once: (G*BM, BD) @ (BD, BN) -> (G*BM, BN).
        q_flat = tl.reshape(q, (GROUP_SIZE * BLOCK_M, BLOCK_D))
        qk_flat = tl.dot(q_flat, k) * scale
        qk = tl.reshape(qk_flat, (GROUP_SIZE, BLOCK_M, BLOCK_N))

        if IS_CAUSAL:
            causal_mask = offs_m[:, None] >= offs_n_iter[None, :]
            qk = tl.where(causal_mask[None, :, :], qk, float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, axis=2))
        alpha = tl.exp(m_i - m_ij)
        p = tl.exp(qk - m_ij[:, :, None])

        acc = acc * alpha[:, :, None]
        p_flat = tl.reshape(p, (GROUP_SIZE * BLOCK_M, BLOCK_N))
        out_flat = tl.dot(p_flat.to(v.dtype), v)
        acc = acc + tl.reshape(out_flat, (GROUP_SIZE, BLOCK_M, BLOCK_D))
        l_i = l_i * alpha + tl.sum(p, axis=2)
        m_i = m_ij

    acc = acc / l_i[:, :, None]

    out_ptrs = (
        o_base
        + heads[:, None, None] * stride_oh
        + offs_m[None, :, None] * stride_on
        + offs_d[None, None, :] * stride_od
    )
    tl.store(out_ptrs, acc.to(Out.dtype.element_ty), mask=mask_m[None, :, None])

    if SAVE_LSE:
        lse = m_i + tl.log(l_i)
        lse_ptrs = l_base + heads[:, None] * stride_lh + offs_m[None, :] * stride_ln
        tl.store(lse_ptrs, lse, mask=mask_m[None, :])


@triton.jit
def _bwd_gqa_kv_kernel(
    Q, K, V, Out, LSE, dO, dQ, dK, dV,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    stride_lb, stride_lh, stride_ln,
    B: tl.constexpr,
    H_Q: tl.constexpr,
    H_KV: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    """GQA backward kernel for dK/dV (split-K style), reusing K/V across GROUP_SIZE heads."""
    batch_idx = tl.program_id(0)
    head_kv = tl.program_id(1)
    block_n = tl.program_id(2)

    q_base = Q + batch_idx * stride_qb
    kv_offset = batch_idx * stride_kb + head_kv * stride_kh
    v_offset = batch_idx * stride_vb + head_kv * stride_vh
    o_base = Out + batch_idx * stride_ob
    l_base = LSE + batch_idx * stride_lb
    do_base = dO + batch_idx * stride_ob

    offs_n = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    mask_n = offs_n < N

    k_ptrs = K + kv_offset + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd
    k = tl.load(k_ptrs, mask=mask_n[None, :], other=0.0).to(tl.float32)

    v_ptrs = V + v_offset + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
    v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)

    dk_acc = tl.zeros([BLOCK_D, BLOCK_N], dtype=tl.float32)
    dv_acc = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
    scale = 1.0 / tl.sqrt(tl.cast(D, tl.float32))

    num_blocks_m = tl.cdiv(N, BLOCK_M)
    for block_m in range(0, num_blocks_m):
        offs_m_iter = block_m * BLOCK_M + offs_m
        mask_m = offs_m_iter < N

        skip_block = tl.constexpr(False)
        if IS_CAUSAL:
            skip_block = (block_m * BLOCK_M + BLOCK_M - 1) < (block_n * BLOCK_N)

        if not skip_block:
            heads = head_kv * GROUP_SIZE + tl.arange(0, GROUP_SIZE)

            q_ptrs = (
                q_base
                + heads[:, None, None] * stride_qh
                + offs_m_iter[None, :, None] * stride_qn
                + offs_d[None, None, :] * stride_qd
            )
            q = tl.load(q_ptrs, mask=mask_m[None, :, None], other=0.0).to(tl.float32)

            do_ptrs = (
                do_base
                + heads[:, None, None] * stride_oh
                + offs_m_iter[None, :, None] * stride_on
                + offs_d[None, None, :] * stride_od
            )
            do = tl.load(do_ptrs, mask=mask_m[None, :, None], other=0.0).to(tl.float32)

            out_ptrs = (
                o_base
                + heads[:, None, None] * stride_oh
                + offs_m_iter[None, :, None] * stride_on
                + offs_d[None, None, :] * stride_od
            )
            out = tl.load(out_ptrs, mask=mask_m[None, :, None], other=0.0).to(tl.float32)

            lse_ptrs = l_base + heads[:, None] * stride_lh + offs_m_iter[None, :] * stride_ln
            lse = tl.load(lse_ptrs, mask=mask_m[None, :], other=0.0).to(tl.float32)

            q_flat = tl.reshape(q, (GROUP_SIZE * BLOCK_M, BLOCK_D))
            qk_flat = tl.dot(q_flat, k) * scale
            qk = tl.reshape(qk_flat, (GROUP_SIZE, BLOCK_M, BLOCK_N))
            if IS_CAUSAL:
                causal_mask = offs_m_iter[:, None] >= offs_n[None, :]
                qk = tl.where(causal_mask[None, :, :], qk, float("-inf"))

            p = tl.exp(qk - lse[:, :, None])
            D_row = tl.sum(do * out, axis=2)

            # dV: sum_g (P[g]^T @ dO[g]) => (BN, BD)
            p_flat = tl.reshape(p, (GROUP_SIZE * BLOCK_M, BLOCK_N))
            do_flat = tl.reshape(do, (GROUP_SIZE * BLOCK_M, BLOCK_D))
            dv_acc += tl.dot(p_flat.trans(), do_flat)

            # dK: sum_g (Q[g]^T @ dS[g]) => (BD, BN)
            dp_flat = tl.dot(do_flat, v.trans())
            dp = tl.reshape(dp_flat, (GROUP_SIZE, BLOCK_M, BLOCK_N))
            ds = p * (dp - D_row[:, :, None])
            ds = ds * scale
            ds_flat = tl.reshape(ds, (GROUP_SIZE * BLOCK_M, BLOCK_N))
            dk_acc += tl.dot(q_flat.trans(), ds_flat)

    dk_ptrs = dK + kv_offset + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd
    tl.store(dk_ptrs, dk_acc, mask=mask_n[None, :])

    dv_ptrs = dV + v_offset + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
    tl.store(dv_ptrs, dv_acc, mask=mask_n[:, None])


@triton.jit
def _bwd_gqa_dq_kernel(
    Q, K, V, Out, LSE, dO, dQ,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    stride_lb, stride_lh, stride_ln,
    B: tl.constexpr,
    H_Q: tl.constexpr,
    H_KV: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    """GQA backward kernel for dQ: one KV head program computes GROUP_SIZE dQ blocks (reuses K/V)."""
    batch_idx = tl.program_id(0)
    head_kv = tl.program_id(1)
    block_m = tl.program_id(2)

    offs_m = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    mask_m = offs_m < N

    q_base = Q + batch_idx * stride_qb
    kv_offset = batch_idx * stride_kb + head_kv * stride_kh
    v_offset = batch_idx * stride_vb + head_kv * stride_vh
    o_base = Out + batch_idx * stride_ob
    l_base = LSE + batch_idx * stride_lb
    do_base = dO + batch_idx * stride_ob
    dq_base = dQ + batch_idx * stride_qb

    heads = head_kv * GROUP_SIZE + tl.arange(0, GROUP_SIZE)

    q_ptrs = (
        q_base
        + heads[:, None, None] * stride_qh
        + offs_m[None, :, None] * stride_qn
        + offs_d[None, None, :] * stride_qd
    )
    q = tl.load(q_ptrs, mask=mask_m[None, :, None], other=0.0).to(tl.float32)

    lse_ptrs = l_base + heads[:, None] * stride_lh + offs_m[None, :] * stride_ln
    lse = tl.load(lse_ptrs, mask=mask_m[None, :], other=0.0).to(tl.float32)

    do_ptrs = (
        do_base
        + heads[:, None, None] * stride_oh
        + offs_m[None, :, None] * stride_on
        + offs_d[None, None, :] * stride_od
    )
    do = tl.load(do_ptrs, mask=mask_m[None, :, None], other=0.0).to(tl.float32)

    out_ptrs = (
        o_base
        + heads[:, None, None] * stride_oh
        + offs_m[None, :, None] * stride_on
        + offs_d[None, None, :] * stride_od
    )
    out = tl.load(out_ptrs, mask=mask_m[None, :, None], other=0.0).to(tl.float32)

    D_row = tl.sum(do * out, axis=2)
    dq_acc = tl.zeros([GROUP_SIZE, BLOCK_M, BLOCK_D], dtype=tl.float32)

    scale = 1.0 / tl.sqrt(tl.cast(D, tl.float32))
    num_blocks_n = tl.cdiv(N, BLOCK_N)
    for block_n in range(0, num_blocks_n):
        offs_n_iter = block_n * BLOCK_N + offs_n
        mask_n = offs_n_iter < N

        skip_block = tl.constexpr(False)
        if IS_CAUSAL:
            skip_block = (block_n * BLOCK_N) > (block_m * BLOCK_M + BLOCK_M - 1)

        if not skip_block:
            k_ptrs = K + kv_offset + offs_n_iter[None, :] * stride_kn + offs_d[:, None] * stride_kd
            k = tl.load(k_ptrs, mask=mask_n[None, :], other=0.0).to(tl.float32)

            v_ptrs = V + v_offset + offs_n_iter[:, None] * stride_vn + offs_d[None, :] * stride_vd
            v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)

            q_flat = tl.reshape(q, (GROUP_SIZE * BLOCK_M, BLOCK_D))
            qk_flat = tl.dot(q_flat, k) * scale
            qk = tl.reshape(qk_flat, (GROUP_SIZE, BLOCK_M, BLOCK_N))
            if IS_CAUSAL:
                causal_mask = offs_m[:, None] >= offs_n_iter[None, :]
                qk = tl.where(causal_mask[None, :, :], qk, float("-inf"))

            p = tl.exp(qk - lse[:, :, None])
            do_flat = tl.reshape(do, (GROUP_SIZE * BLOCK_M, BLOCK_D))
            dp_flat = tl.dot(do_flat, v.trans())
            dp = tl.reshape(dp_flat, (GROUP_SIZE, BLOCK_M, BLOCK_N))
            ds = p * (dp - D_row[:, :, None])
            ds = ds * scale

            ds_flat = tl.reshape(ds, (GROUP_SIZE * BLOCK_M, BLOCK_N))
            dq_flat = tl.dot(ds_flat, k.trans())
            dq_acc += tl.reshape(dq_flat, (GROUP_SIZE, BLOCK_M, BLOCK_D))

    dq_ptrs = (
        dq_base
        + heads[:, None, None] * stride_qh
        + offs_m[None, :, None] * stride_qn
        + offs_d[None, None, :] * stride_qd
    )
    tl.store(dq_ptrs, dq_acc, mask=mask_m[None, :, None])


@triton.jit
def _fwd_kernel(
    Q, K, V, Out, L,  # Pointers  
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    stride_lb, stride_lh, stride_ln,
    B: tl.constexpr, H: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    SAVE_LSE: tl.constexpr,
):
    """Forward kernel for attention: O = softmax(QK^T / sqrt(D)) @ V"""
    # Get batch/head/block indices
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    block_m = tl.program_id(2)
    
    # Compute offsets
    qo_offset = batch_idx * stride_qb + head_idx * stride_qh
    kv_offset = batch_idx * stride_kb + head_idx * stride_kh
    l_offset = batch_idx * stride_lb + head_idx * stride_lh
    
    # Block indices
    offs_m = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    
    # Initialize accumulators
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    
    # Load Q for this block
    q_ptrs = Q + qo_offset + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd
    q_mask = offs_m[:, None] < N
    q = tl.load(q_ptrs, mask=q_mask, other=0.0).to(tl.float32)
    
    # Compute scale
    scale = 1.0 / tl.sqrt(tl.cast(D, tl.float32))
    
    # Iterate over K/V blocks
    num_blocks = tl.cdiv(N, BLOCK_N)
    for block_n in range(0, num_blocks):
        # Load K block
        k_ptrs = K + kv_offset + (block_n * BLOCK_N + offs_n[None, :]) * stride_kn + offs_d[:, None] * stride_kd
        k_mask = (block_n * BLOCK_N + offs_n[None, :]) < N
        k = tl.load(k_ptrs, mask=k_mask, other=0.0).to(tl.float32)
        
        # Compute QK^T
        qk = tl.dot(q, k)
        qk = qk * scale
        
        # Apply causal mask if needed
        if IS_CAUSAL:
            causal_mask = (offs_m[:, None]) >= (block_n * BLOCK_N + offs_n[None, :])
            qk = tl.where(causal_mask, qk, float("-inf"))
        
        # Online softmax: compute max and update
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_ij)
        p = tl.exp(qk - m_ij[:, None])
        
        # Load V block
        v_ptrs = V + kv_offset + (block_n * BLOCK_N + offs_n[:, None]) * stride_vn + offs_d[None, :] * stride_vd
        v_mask = (block_n * BLOCK_N + offs_n[:, None]) < N
        v = tl.load(v_ptrs, mask=v_mask, other=0.0).to(tl.float32)
        
        # Update accumulated output
        acc = acc * alpha[:, None]
        acc = acc + tl.dot(p.to(v.dtype), v)
        
        # Update softmax statistics
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_ij
    
    # Final normalization
    acc = acc / l_i[:, None]
    
    # Store output
    out_ptrs = Out + qo_offset + offs_m[:, None] * stride_on + offs_d[None, :] * stride_od
    tl.store(out_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < N)
    
    # Save LSE for backward
    if SAVE_LSE:
        lse = m_i + tl.log(l_i)
        lse_ptrs = L + l_offset + offs_m * stride_ln
        tl.store(lse_ptrs, lse, mask=offs_m < N)


@triton.jit
def _bwd_kernel(
    Q, K, V, Out, LSE, dO, dQ, dK, dV,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    stride_lb, stride_lh, stride_ln,
    B: tl.constexpr, H: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    """
    Backward kernel - iterates over K/V blocks to avoid atomics.
    Each block computes gradients for a chunk of K and V.
    """
    # Get indices - iterate over K/V blocks instead of Q blocks
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    block_n = tl.program_id(2)  # K/V block index
    
    # Offsets
    qo_offset = batch_idx * stride_qb + head_idx * stride_qh
    kv_offset = batch_idx * stride_kb + head_idx * stride_kh
    l_offset = batch_idx * stride_lb + head_idx * stride_lh
    
    # Block indices
    offs_n = block_n * BLOCK_N + tl.arange(0, BLOCK_N)  # K/V indices for this block
    offs_m = tl.arange(0, BLOCK_M)  # Q indices (will loop over)
    offs_d = tl.arange(0, BLOCK_D)
    mask_n = offs_n < N
    
    # Load K, V for this block
    k_ptrs = K + kv_offset + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd
    k = tl.load(k_ptrs, mask=mask_n[None, :], other=0.0).to(tl.float32)
    
    v_ptrs = V + kv_offset + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
    v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)
    
    # Initialize accumulators for dK, dV
    dk_acc = tl.zeros([BLOCK_D, BLOCK_N], dtype=tl.float32)
    dv_acc = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
    
    # Compute scale
    scale = 1.0 / tl.sqrt(tl.cast(D, tl.float32))
    
    # Loop over Q blocks
    num_blocks_m = tl.cdiv(N, BLOCK_M)
    for block_m in range(0, num_blocks_m):
        offs_m_iter = block_m * BLOCK_M + offs_m
        mask_m = offs_m_iter < N
        
        # Apply causal masking - skip computation if this Q block is entirely masked
        skip_block = tl.constexpr(False)
        if IS_CAUSAL:
            # For causal: Q[i] can only attend to K[j] where j <= i
            # Skip if all queries in this block are before all keys in current block
            skip_block = (block_m * BLOCK_M + BLOCK_M - 1) < (block_n * BLOCK_N)
        
        if not skip_block:
            # Load Q, LSE, dO, Out for this Q block
            q_ptrs = Q + qo_offset + offs_m_iter[:, None] * stride_qn + offs_d[None, :] * stride_qd
            q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)
            
            lse_ptrs = LSE + l_offset + offs_m_iter * stride_ln
            lse = tl.load(lse_ptrs, mask=mask_m, other=0.0).to(tl.float32)
            
            do_ptrs = dO + qo_offset + offs_m_iter[:, None] * stride_on + offs_d[None, :] * stride_od
            do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)
            
            out_ptrs = Out + qo_offset + offs_m_iter[:, None] * stride_on + offs_d[None, :] * stride_od
            out = tl.load(out_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)
            
            # Recompute attention: QK^T / sqrt(D)
            qk = tl.dot(q, k)  # [BLOCK_M, BLOCK_N]
            qk = qk * scale
            
            # Apply causal mask
            if IS_CAUSAL:
                causal_mask = offs_m_iter[:, None] >= offs_n[None, :]
                qk = tl.where(causal_mask, qk, float("-inf"))
            
            # Recompute softmax: P = exp(QK - LSE)
            p = tl.exp(qk - lse[:, None])  # [BLOCK_M, BLOCK_N]
            
            # Compute D = rowsum(dO * O) for softmax backward
            D_row = tl.sum(do * out, axis=1)  # [BLOCK_M]
            
            # Compute dV: accumulate P^T @ dO
            dv_acc += tl.dot(p.trans(), do)  # [BLOCK_N, BLOCK_D]
            
            # Compute dP = dO @ V^T
            dp = tl.dot(do, v.trans())  # [BLOCK_M, BLOCK_N]
            
            # Softmax backward: dS = P * (dP - D)
            ds = p * (dp - D_row[:, None])  # [BLOCK_M, BLOCK_N]
            ds = ds * scale
            
            # Accumulate dK: Q^T @ dS
            dk_acc += tl.dot(q.trans(), ds)  # [BLOCK_D, BLOCK_N]
    
    # Store dK and dV (no atomics needed!)
    dk_ptrs = dK + kv_offset + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd
    tl.store(dk_ptrs, dk_acc, mask=mask_n[None, :])
    
    dv_ptrs = dV + kv_offset + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
    tl.store(dv_ptrs, dv_acc, mask=mask_n[:, None])


@triton.jit
def _bwd_dq_kernel(
    Q, K, V, Out, LSE, dO, dQ,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    stride_lb, stride_lh, stride_ln,
    B: tl.constexpr, H: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    """Separate kernel for computing dQ (iterates over Q blocks)"""
    # Get indices
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    block_m = tl.program_id(2)
    
    # Offsets
    qo_offset = batch_idx * stride_qb + head_idx * stride_qh
    kv_offset = batch_idx * stride_kb + head_idx * stride_kh
    l_offset = batch_idx * stride_lb + head_idx * stride_lh
    
    # Block indices
    offs_m = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    mask_m = offs_m < N
    
    # Load Q, LSE, dO, Out for this block
    q_ptrs = Q + qo_offset + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)
    
    lse_ptrs = LSE + l_offset + offs_m * stride_ln
    lse = tl.load(lse_ptrs, mask=mask_m, other=0.0).to(tl.float32)
    
    do_ptrs = dO + qo_offset + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd
    do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)
    
    out_ptrs = Out + qo_offset + offs_m[:, None] * stride_on + offs_d[None, :] * stride_od
    out = tl.load(out_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)
    
    # Compute D = rowsum(dO * Out)
    D_row = tl.sum(do * out, axis=1)
    
    # Initialize dQ accumulator
    dq_acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    
    # Compute scale
    scale = 1.0 / tl.sqrt(tl.cast(D, tl.float32))
    
    # Loop over K/V blocks
    num_blocks_n = tl.cdiv(N, BLOCK_N)
    for block_n in range(0, num_blocks_n):
        offs_n_iter = block_n * BLOCK_N + offs_n
        mask_n = offs_n_iter < N
        
        # Apply causal masking - skip if this K block is entirely after all queries
        skip_block = tl.constexpr(False)
        if IS_CAUSAL:
            # Skip if this K block is entirely after all queries in current block
            skip_block = (block_n * BLOCK_N) > (block_m * BLOCK_M + BLOCK_M - 1)
        
        if not skip_block:
            # Load K, V
            k_ptrs = K + kv_offset + offs_n_iter[None, :] * stride_kn + offs_d[:, None] * stride_kd
            k = tl.load(k_ptrs, mask=mask_n[None, :], other=0.0).to(tl.float32)
            
            v_ptrs = V + kv_offset + offs_n_iter[:, None] * stride_vn + offs_d[None, :] * stride_vd
            v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)
            
            # Recompute attention
            qk = tl.dot(q, k)
            qk = qk * scale
            
            if IS_CAUSAL:
                causal_mask = offs_m[:, None] >= offs_n_iter[None, :]
                qk = tl.where(causal_mask, qk, float("-inf"))
            
            # Recompute P
            p = tl.exp(qk - lse[:, None])
            
            # Compute dP = dO @ V^T
            dp = tl.dot(do, v.trans())
            
            # Softmax backward: dS = P * (dP - D)
            ds = p * (dp - D_row[:, None])
            ds = ds * scale
            
            # Accumulate dQ = dS @ K
            dq_acc += tl.dot(ds, k.trans())
    
    # Store dQ
    dq_ptrs = dQ + qo_offset + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd
    tl.store(dq_ptrs, dq_acc, mask=mask_m[:, None])


class CCGQAAttentionFused(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal=False, use_fused_backward=True):
        """Fused attention forward pass"""
        B, Hq, N, D = q.shape
        Hkv = k.shape[1]
        
        # Allocate output and LSE
        out = torch.empty_like(q)
        lse = torch.empty((B, Hq, N), device=q.device, dtype=torch.float32)
        
        # Get optimal block sizes (cached or heuristic)
        BLOCK_M, BLOCK_N, BLOCK_D = get_block_sizes(N, D, q.dtype, str(q.device))

        if Hkv == Hq:
            grid = (B, Hq, triton.cdiv(N, BLOCK_M))
            _fwd_kernel[grid](
                q, k, v, out, lse,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                lse.stride(0), lse.stride(1), lse.stride(2),
                B=B, H=Hq, N=N, D=D,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
                IS_CAUSAL=is_causal,
                SAVE_LSE=use_fused_backward,
            )
            ctx.is_gqa = False
            ctx.h_kv = Hq
            ctx.group_size = 1
        else:
            # GQA KV-sharing path: K/V have fewer heads than Q.
            group_size = Hq // Hkv

            # GQA kernels materialize per-group tensors (Q/dO/Out) which can spill on large blocks.
            # Clamp blocks to stay under shared-memory limits on newer GPUs (e.g. SM 12.0).
            if group_size > 1:
                BLOCK_M = min(BLOCK_M, 32)
                BLOCK_N = min(BLOCK_N, 64)

            grid = (B, Hkv, triton.cdiv(N, BLOCK_M))
            _fwd_gqa_kernel[grid](
                q, k, v, out, lse,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                lse.stride(0), lse.stride(1), lse.stride(2),
                B=B, H_Q=Hq, H_KV=Hkv, N=N, D=D,
                GROUP_SIZE=group_size,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
                IS_CAUSAL=is_causal,
                SAVE_LSE=use_fused_backward,
                num_warps=4,
                num_stages=1,
            )
            ctx.is_gqa = True
            ctx.h_kv = Hkv
            ctx.group_size = group_size
        
        # Save for backward
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.is_causal = is_causal
        ctx.use_fused_backward = use_fused_backward
        
        return out
    
    @staticmethod
    def backward(ctx, grad_out):
        """Attention backward pass"""
        q, k, v, out, lse = ctx.saved_tensors
        is_causal = ctx.is_causal
        use_fused = ctx.use_fused_backward
        
        B, Hq, N, D = q.shape
        Hkv = getattr(ctx, "h_kv", k.shape[1])
        group_size = getattr(ctx, "group_size", 1)
        is_gqa = getattr(ctx, "is_gqa", False)
        
        # Allocate gradients. Kernels accumulate in fp32 and store to these buffers.
        # Using input dtype reduces global memory bandwidth vs fp32 buffers.
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        
        if use_fused:
            # Split-K backward (2-pass, no atomics)
            BLOCK_M, BLOCK_N, BLOCK_D = get_block_sizes(N, D, q.dtype, str(q.device))

            if is_gqa and group_size > 1:
                BLOCK_M = min(BLOCK_M, 32)
                BLOCK_N = min(BLOCK_N, 64)

            if not is_gqa:
                grid_kv = (B, Hq, triton.cdiv(N, BLOCK_N))
                _bwd_kernel[grid_kv](
                    q, k, v, out, lse, grad_out, dq, dk, dv,
                    q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                    k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                    v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                    out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                    lse.stride(0), lse.stride(1), lse.stride(2),
                    B=B, H=Hq, N=N, D=D,
                    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
                    IS_CAUSAL=is_causal,
                )

                grid_q = (B, Hq, triton.cdiv(N, BLOCK_M))
                _bwd_dq_kernel[grid_q](
                    q, k, v, out, lse, grad_out, dq,
                    q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                    k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                    v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                    out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                    lse.stride(0), lse.stride(1), lse.stride(2),
                    B=B, H=Hq, N=N, D=D,
                    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
                    IS_CAUSAL=is_causal,
                )
            else:
                grid_kv = (B, Hkv, triton.cdiv(N, BLOCK_N))
                _bwd_gqa_kv_kernel[grid_kv](
                    q, k, v, out, lse, grad_out, dq, dk, dv,
                    q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                    k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                    v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                    out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                    lse.stride(0), lse.stride(1), lse.stride(2),
                    B=B, H_Q=Hq, H_KV=Hkv, N=N, D=D,
                    GROUP_SIZE=group_size,
                    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
                    IS_CAUSAL=is_causal,
                    num_warps=4,
                    num_stages=1,
                )

                grid_q = (B, Hkv, triton.cdiv(N, BLOCK_M))
                _bwd_gqa_dq_kernel[grid_q](
                    q, k, v, out, lse, grad_out, dq,
                    q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                    k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                    v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                    out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                    lse.stride(0), lse.stride(1), lse.stride(2),
                    B=B, H_Q=Hq, H_KV=Hkv, N=N, D=D,
                    GROUP_SIZE=group_size,
                    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
                    IS_CAUSAL=is_causal,
                    num_warps=4,
                    num_stages=1,
                )

            # Already in the right dtype.
            pass
        else:
            # Phase 1 fallback: Manual gradient computation in float32
            q_fp32 = q.to(torch.float32)
            k_fp32 = k.to(torch.float32)
            v_fp32 = v.to(torch.float32)
            do_fp32 = grad_out.to(torch.float32)
            
            scale = 1.0 / (D ** 0.5)
            
            # Recompute forward
            attn = torch.matmul(q_fp32, k_fp32.transpose(-2, -1)) * scale
            if is_causal:
                mask = torch.triu(torch.ones(N, N, device=q.device, dtype=torch.bool), diagonal=1)
                attn = attn.masked_fill(mask, float('-inf'))
            p = torch.softmax(attn, dim=-1)
            
            # Backward: dV = P^T @ dO
            dv = torch.matmul(p.transpose(-2, -1), do_fp32).to(v.dtype)
            
            # dP = dO @ V^T
            dp = torch.matmul(do_fp32, v_fp32.transpose(-2, -1))
            
            # Softmax backward: dS = P * (dP - rowsum(dP * P))
            ds = p * (dp - (dp * p).sum(dim=-1, keepdim=True))
            ds = ds * scale
            
            # dQ = dS @ K, dK = dS^T @ Q
            dq = torch.matmul(ds, k_fp32).to(q.dtype)
            dk = torch.matmul(ds.transpose(-2, -1), q_fp32).to(k.dtype)
        
        return dq, dk, dv, None, None


def ccgqa_attention_fused(q, k, v, is_causal=False, use_fused_backward=True):
    """
    Public API for fused CCGQA attention.
    
    Args:
        q: Query [B, H, N, D]
        k: Key [B, H, N, D]
        v: Value [B, H, N, D]
        is_causal: Apply causal masking
        use_fused_backward: Use split-K fused backward (default True, fastest)
    
    Returns:
        out: Attention output [B, H, N, D]
    """
    return CCGQAAttentionFused.apply(q, k, v, is_causal, use_fused_backward)
