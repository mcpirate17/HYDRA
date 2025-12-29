"""
Fused attention kernel for CCGQA.

This kernel fuses the attention compute (Q @ K.T, softmax, @ V) into a single Triton kernel
to reduce memory bandwidth and improve performance.

Based on FlashAttention-2 architecture with modifications for CCGQA.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _bwd_kernel(
    Q, K, V, Out, dOut,
    dQ, dK, dV,
    L,  # Softmax LSE from forward
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    stride_lb, stride_lh, stride_ln,
    B: tl.constexpr,
    H: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    """
    Backward pass of fused attention.
    
    Computes gradients dQ, dK, dV given dOut and saved activations.
    Uses FlashAttention-2 backward algorithm with recomputation.
    """
    # Program IDs
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)
    
    # Offsets
    qo_offset = pid_b * stride_qb + pid_h * stride_qh
    kv_offset = pid_b * stride_kb + pid_h * stride_kh
    l_offset = pid_b * stride_lb + pid_h * stride_lh
    
    # Block ranges
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    
    # Load Q, Out, dOut, L for this block
    q_ptrs = Q + qo_offset + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd
    o_ptrs = Out + qo_offset + offs_m[:, None] * stride_on + offs_d[None, :] * stride_od
    do_ptrs = dOut + qo_offset + offs_m[:, None] * stride_on + offs_d[None, :] * stride_od
    l_ptrs = L + l_offset + offs_m * stride_ln
    
    mask_m = offs_m[:, None] < N
    q = tl.load(q_ptrs, mask=mask_m, other=0.0).to(tl.float32)
    o = tl.load(o_ptrs, mask=mask_m, other=0.0).to(tl.float32)
    do = tl.load(do_ptrs, mask=mask_m, other=0.0).to(tl.float32)
    l = tl.load(l_ptrs, mask=offs_m < N, other=0.0).to(tl.float32)
    
    # Compute D = rowsum(dOut * Out)
    Di = tl.sum(do * o, axis=1)
    
    # Scale
    scale = 1.0 / tl.sqrt(tl.cast(D, tl.float32))
    
    # Initialize dQ accumulator
    dq_acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    
    # Iterate over K/V blocks
    num_blocks = tl.cdiv(N, BLOCK_N)
    for block_n in range(0, num_blocks):
        # Load K, V
        k_ptrs = K + kv_offset + (block_n * BLOCK_N + offs_n[None, :]) * stride_kn + offs_d[:, None] * stride_kd
        v_ptrs = V + kv_offset + (block_n * BLOCK_N + offs_n[:, None]) * stride_vn + offs_d[None, :] * stride_vd
        
        mask_n = (block_n * BLOCK_N + offs_n) < N
        k = tl.load(k_ptrs, mask=mask_n[None, :], other=0.0).to(tl.float32)
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)
        
        # Recompute attention scores
        qk = tl.dot(q, k) * scale
        
        # Apply causal mask
        if IS_CAUSAL:
            causal_mask = (offs_m[:, None]) >= (block_n * BLOCK_N + offs_n[None, :])
            qk = tl.where(causal_mask, qk, float("-inf"))
        
        # Recompute softmax (P = softmax(QK))
        p = tl.exp(qk - l[:, None])
        
        # Compute dV: dV += P^T @ dO
        dv = tl.dot(p.to(do.dtype).trans(), do)
        dv_ptrs = dV + kv_offset + (block_n * BLOCK_N + offs_n[:, None]) * stride_vn + offs_d[None, :] * stride_vd
        tl.atomic_add(dv_ptrs, dv, mask=mask_n[:, None])
        
        # Compute dP: dP = dO @ V^T
        dp = tl.dot(do, v.trans())
        
        # Compute dS: dS = P * (dP - D)
        ds = p * (dp - Di[:, None])
        ds = ds * scale
        
        # Compute dQ: dQ += dS @ K
        dq_acc += tl.dot(ds.to(k.dtype), k.trans())
        
        # Compute dK: dK += dS^T @ Q
        dk = tl.dot(ds.to(q.dtype).trans(), q)
        dk_ptrs = dK + kv_offset + (block_n * BLOCK_N + offs_n[None, :]) * stride_kn + offs_d[:, None] * stride_kd
        tl.atomic_add(dk_ptrs, dk, mask=mask_n[None, :])
    
    # Store dQ
    dq_ptrs = dQ + qo_offset + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd
    tl.store(dq_ptrs, dq_acc.to(dQ.dtype.element_ty), mask=mask_m)


class CCGQAAttentionFused(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal=False, use_fused_backward=True):
        """
        Fused attention forward pass.
        
        Args:
            q: Query tensor [B, H, N, D]
            k: Key tensor [B, H, N, D]
            v: Value tensor [B, H, N, D]
            is_causal: Whether to apply causal masking
            use_fused_backward: Use Triton backward kernel (Phase 2)
        
        Returns:
            out: Output tensor [B, H, N, D]
        """
        # Get shapes
        B, H, N, D = q.shape
        
        # Allocate output and LSE (log-sum-exp for backward)
        out = torch.empty_like(q)
        lse = torch.empty((B, H, N), device=q.device, dtype=torch.float32)
        
        # Launch configuration
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_D = D
        
        grid = (B, H, triton.cdiv(N, BLOCK_M))
        
        # Launch kernel
        _fwd_kernel[grid](
            q, k, v, out, lse,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            B=B, H=H, N=N, D=D,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
            IS_CAUSAL=is_causal,
            SAVE_LSE=use_fused_backward,
        )
        
        # Save for backward
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.is_causal = is_causal
        ctx.use_fused_backward = use_fused_backward)
    
    # Initialize accumulators
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")  # Max values for softmax
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)  # Sum of exp for softmax
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)  # Accumulated output
    
    # Load Q for this block
    q_ptrs = Q + qo_offset + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd
    q_mask = offs_m[:, None] < N
    q = tl.load(q_ptrs, mask=q_mask, other=0.0).to(tl.float32)
    
    # Compute scale (D is constexpr, cast to float32 directly)
    scale = 1.0 / tl.sqrt(tl.cast(D, tl.float32))
    
    # Iterate over K/V blocks
    num_blocks = tl.cdiv(N, BLOCK_N)
    for block_n in range(0, num_blocks):
        # Load K block
        k_ptrs = K + kv_offset + (block_n * BLOCK_N + offs_n[None, :]) * stride_kn + offs_d[:, None] * stride_kd
        k_mask = (block_n * BLOCK_N + offs_n[None, :]) < N
        k = tl.load(k_ptrs, mask=k_mask, other=0.0).to(tl.float32)
        
        # Compute QK^T
        qk = tl.dot(q, k)  # [BLOCK_M, BLOCK_N]
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
    o_ptrs = Out + qo_offset + offs_m[:, None] * stride_on + offs_d[None, :] * stride_od
    o_mask = offs_m[:, None] < N
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=o_mask)
    
    # Store softmax statistics for backward (log-sum-exp)
    if SAVE_LSE:
        lse = m_i + tl.log(l_i)
        l_ptrs = L + pid_b * stride_lb + pid_h * stride_lh + offs_m * stride_ln
        l_mask = offs_m < N
        tl.store(l_ptrs, lse, mask=l_mask)


class CCGQAAttentionFused(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal=False):
        """
        Fused attention forward pass.
        
        Args:
            q: Query tensor [B, H, N, D]
            k: Key tensor [B, H, N, D]
            v: Value tensor [B, H, N, D]
            is_causal: Whether to apply causal masking
        
        Returns:
            out: Output tensor [B, H, N, D]
        """
        # Get shapes
        B, H, N, D = q.shape
        
        # Allocate output
        out = torch.empty_like(q)
        
        # Launch configuration
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_D = D
        
        grid = (B, H, triton.cdiv(N, BLOCK_M))
        
        # Launch kernel
        _fwd_kernel[grid](
            q, k, v, out,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            B=B, H=H, N=N, D=D,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
            IS_CAUSAL=is_causal,
        )
        
        # Save for backward
        ctx.save_for_backward(q, k, v, out)
        ctx.is_causal = is_causal
        
        return outfused Triton kernel for gradients.
        
        Phase 2: Fully fused backward kernel for 2-3x speedup.
        Falls back to PyTorch if fused backward disabled.
        """
        q, k, v, out, lse = ctx.saved_tensors
        is_causal = ctx.is_causal
        use_fused_backward = ctx.use_fused_backward
        
        if not use_fused_backward:
            # Fallback: recompute with PyTorch SDPA
            q_bwd = q.detach().requires_grad_(True)
            k_bwd = k.detach().requires_grad_(True)
            v_bwd = v.detach().requires_grad_(True)
            
            with torch.enable_grad():
                out_recompute = torch.nn.functional.scaled_dot_product_attention(
                    q_bwd, k_bwd, v_bwd, is_causal=is_causal
                )
                grads = torch.autograd.grad(
                    outputs=out_recompute,
                    inputs=(q_bwd, k_bwd, v_bwd),
                    grad_outputs=grad_out,
                    retain_graph=False,
                )
            return grads[0], grads[1], grads[2], None, None
        
        # Fused backward path
        B, H, N, D = q.shape
        
        # Allocate gradient tensors
        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        , use_fused_backward=True):
    """
    Fused attention compute for CCGQA.
    
    This is a drop-in replacement for F.scaled_dot_product_attention
    with improved memory efficiency.
    
    Args:
        q: Query tensor [B, H, N, D]
        k: Key tensor [B, H, N, D]
        v: Value tensor [B, H, N, D]
        is_causal: Whether to apply causal masking
        use_fused_backward: Use fused backward kernel (Phase 2)
    
    Returns:
        out: Output tensor [B, H, N, D]
    """
    return CCGQAAttentionFused.apply(q, k, v, is_causal, use_fused_backward,
            B=B, H=H, N=N, D=D,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
            IS_CAUSAL=is_causal,
        )
        
        return dq, dk, dv, None),
                grad_outputs=grad_out,
                retain_graph=False,
            )
        
        return grads[0], grads[1], grads[2], None


def ccgqa_attention_fused(q, k, v, is_causal=False):
    """
    Fused attention compute for CCGQA.
    
    This is a drop-in replacement for F.scaled_dot_product_attention
    with improved memory efficiency.
    
    Args:
        q: Query tensor [B, H, N, D]
        k: Key tensor [B, H, N, D]
        v: Value tensor [B, H, N, D]
        is_causal: Whether to apply causal masking
    
    Returns:
        out: Output tensor [B, H, N, D]
    """
    return CCGQAAttentionFused.apply(q, k, v, is_causal)
