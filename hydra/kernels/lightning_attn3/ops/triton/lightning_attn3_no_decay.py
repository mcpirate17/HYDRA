# Lightning Attention-3 - HYDRA Project
# Based on Lightning Attention (https://github.com/OpenNLPLab/lightning-attention)
# Modified for Blackwell (SM 12.x) compatibility with hardware-aware kernel selection
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl

# Cache for backward kernel selection per device
# Values: ("original", BLOCK, CBLOCK) or ("chunked", CBLOCK)
_BWD_KERNEL_CACHE: dict[int, tuple[str, int, int]] = {}

# SRAM limits by architecture
_BLACKWELL_SRAM_LIMIT = 101_376  # bytes
_PRE_BLACKWELL_SRAM_LIMIT = 163_840  # bytes (most Ampere/Hopper)


# ============================================================================
# AUTOTUNE CONFIGURATIONS
# ============================================================================

_FWD_CONFIGS = [
    triton.Config({'BLOCK': 64, 'BLOCK_MODEL': 32}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK': 64, 'BLOCK_MODEL': 64}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK': 128, 'BLOCK_MODEL': 32}, num_warps=8, num_stages=2),
    triton.Config({'BLOCK': 128, 'BLOCK_MODEL': 64}, num_warps=8, num_stages=1),
]

_BWD_INTRA_CONFIGS = [
    triton.Config({'BLOCK': 64, 'CBLOCK': 32}, num_warps=4, num_stages=1),
    triton.Config({'BLOCK': 64, 'CBLOCK': 16}, num_warps=4, num_stages=1),
    triton.Config({'BLOCK': 128, 'CBLOCK': 32}, num_warps=8, num_stages=1),
]

_BWD_INTER_CONFIGS = [
    triton.Config({'CBLOCK': 32}, num_warps=4, num_stages=1),
    triton.Config({'CBLOCK': 64}, num_warps=4, num_stages=1),
]


def _estimate_original_bwd_sram(block: int, cblock: int, d: int, e: int) -> int:
    """Estimate shared memory for original backward kernels."""
    # Intra kernel: Q[BLOCK,d], K[BLOCK,d], V[BLOCK,e], DO[BLOCK,e], attention[BLOCK,BLOCK]
    # All in fp32 during compute
    intra = (2 * block * d + 2 * block * e + block * block) * 4
    # Inter kernel: similar plus kv_state[d,e]
    inter = intra + d * e * 4
    return max(intra, inter)


def _select_backward_kernel(device: torch.device, d: int, e: int) -> tuple[str, int, int]:
    """
    Select the appropriate backward kernel based on GPU architecture.
    
    Returns:
        ("original", BLOCK, CBLOCK) for pre-Blackwell GPUs
        ("chunked", CBLOCK, 0) for Blackwell GPUs
        
    Raises:
        RuntimeError: If no safe kernel configuration exists
    """
    props = torch.cuda.get_device_properties(device)
    cc_major = props.major
    cc_minor = props.minor
    # Use shared_memory_per_block_optin (opt-in extended shared memory)
    max_sram = props.shared_memory_per_block_optin
    
    if cc_major >= 12:
        # Blackwell (SM 12.x): Use chunked recompute-heavy kernel
        # Try larger CBLOCK first for better performance
        from .lightning_attn3_no_decay_chunked import validate_config
        
        for cblock in [32, 16]:  # Prefer larger tiles for better occupancy
            is_valid, sram = validate_config(cblock, d, num_stages=1)
            if is_valid and sram < max_sram:
                return ("chunked", cblock, 0)
        
        raise RuntimeError(
            f"Lightning Attention-3 backward: No safe kernel config for Blackwell "
            f"(SM {cc_major}.{cc_minor}, max_sram={max_sram}, d={d}, e={e}). "
            f"Try reducing head_dim to ≤128."
        )
    else:
        # Pre-Blackwell: Use original kernel with appropriate tile sizes
        for block, cblock in [(64, 32), (32, 16)]:
            sram = _estimate_original_bwd_sram(block, cblock, d, e)
            if sram < max_sram:
                return ("original", block, cblock)
        
        raise RuntimeError(
            f"Lightning Attention-3 backward: No safe kernel config for "
            f"SM {cc_major}.{cc_minor} (max_sram={max_sram}, d={d}, e={e}). "
            f"Try reducing head_dim."
        )


@triton.autotune(
    configs=_FWD_CONFIGS,
    key=['n', 'd', 'e'],
)
@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    Out,
    b: tl.constexpr,
    h: tl.constexpr,
    n: tl.constexpr,
    d: tl.constexpr,
    e: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_BLOCK: tl.constexpr,
    BLOCK_MODEL: tl.constexpr,
):
    ##### get offset
    off_bh = tl.program_id(0)
    off_bh % h
    off_e = tl.program_id(1)
    qk_offset = off_bh * n * d
    v_offset = off_bh * n * e
    o_offset = off_bh * n * e
    # channel offset
    e_offset = off_e * BLOCK_MODEL

    ##### get block ptr
    Q_block_ptr = Q + qk_offset + tl.arange(0, d)[None, :]
    K_trans_block_ptr = K + qk_offset + tl.arange(0, d)[:, None]
    V_block_ptr = V + v_offset + e_offset + tl.arange(0, BLOCK_MODEL)[None, :]
    O_block_ptr = Out + o_offset + e_offset + tl.arange(0, BLOCK_MODEL)[None, :]

    ##### init diag decay(Lambda); q, k decay; kv
    # q, k decay
    off_block = tl.arange(
        0, BLOCK
    )  # Not bug, this is a bit different from algorithm 1, but is mathematically equivalent
    # diag decay
    index = off_block[:, None] - off_block[None, :]
    kv = tl.zeros([d, BLOCK_MODEL], dtype=tl.float32)

    ##### compute
    for i in range(NUM_BLOCK):
        # load
        q = tl.load(
            Q_block_ptr + off_block[:, None] * d, mask=off_block[:, None] < n, other=0.0
        ).to(tl.float32)
        k_trans = tl.load(
            K_trans_block_ptr + off_block[None, :] * d,
            mask=off_block[None, :] < n,
            other=0.0,
        ).to(tl.float32)
        v = tl.load(
            V_block_ptr + off_block[:, None] * e, mask=off_block[:, None] < n, other=0.0
        ).to(tl.float32)

        # compute
        qk = tl.dot(q, k_trans)
        qk = tl.where(index >= 0, qk, 0)
        o_intra = tl.dot(qk, v)
        o_inter = tl.dot(q, kv)
        o = o_intra + o_inter

        # save and update
        tl.store(
            O_block_ptr + off_block[:, None] * e,
            o.to(O_block_ptr.dtype.element_ty),
            mask=off_block[:, None] < n,
        )
        kv += tl.dot(k_trans, v)
        off_block += BLOCK


@triton.jit
def _bwd_intra_kernel(
    Q,
    K,
    V,
    DO,
    DQ,
    DK,
    DV,
    b: tl.constexpr,
    h: tl.constexpr,
    n: tl.constexpr,
    d: tl.constexpr,
    e: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_BLOCK: tl.constexpr,
    CBLOCK: tl.constexpr,
    NUM_CBLOCK: tl.constexpr,
):
    ##### get offset
    off_bh = tl.program_id(0)
    off_block = tl.program_id(1)
    # Removed: off_bh % h (was unused)
    qk_offset = off_bh * n * d
    v_offset = off_bh * n * e
    o_offset = off_bh * n * e
    block_offset = off_block * BLOCK + tl.arange(0, BLOCK)

    ##### get block ptr
    Q_trans_block_ptr = (
        Q + qk_offset + block_offset[None, :] * d + tl.arange(0, d)[:, None]
    )
    K_block_ptr = K + qk_offset + block_offset[:, None] * d + tl.arange(0, d)[None, :]
    V_trans_block_ptr = (
        V + v_offset + block_offset[None, :] * e + tl.arange(0, e)[:, None]
    )

    DQ_block_ptr = DQ + qk_offset + block_offset[:, None] * d + tl.arange(0, d)[None, :]
    DK_trans_block_ptr = (
        DK + qk_offset + block_offset[None, :] * d + tl.arange(0, d)[:, None]
    )
    DV_block_ptr = DV + v_offset + block_offset[:, None] * e + tl.arange(0, e)[None, :]
    DO_block_ptr = DO + o_offset + block_offset[:, None] * e + tl.arange(0, e)[None, :]

    ##### init diag decay(Lambda)
    array = tl.arange(0, BLOCK).to(tl.float32)
    # diag
    index = array[:, None] - array[None, :]

    ##### load block
    k = tl.load(K_block_ptr, mask=block_offset[:, None] < n, other=0.0).to(tl.float32)
    v_trans = tl.load(V_trans_block_ptr, mask=block_offset[None, :] < n, other=0.0).to(
        tl.float32
    )
    do = tl.load(DO_block_ptr, mask=block_offset[:, None] < n, other=0.0).to(tl.float32)
    q_trans = tl.load(Q_trans_block_ptr, mask=block_offset[None, :] < n, other=0.0).to(
        tl.float32
    )

    ##### compute
    dqk = tl.dot(do, v_trans)
    dqk = tl.where(index >= 0, dqk, 0)
    dq_intra = tl.dot(dqk, k)

    dk_intra_trans = tl.dot(q_trans, dqk)

    qk_trans = tl.dot(k, q_trans)
    qk_trans = tl.where(index <= 0, qk_trans, 0)
    dv_intra = tl.dot(qk_trans, do)

    dq = dq_intra
    dk_trans = dk_intra_trans
    dv = dv_intra

    # save
    tl.store(
        DQ_block_ptr,
        dq.to(DQ_block_ptr.dtype.element_ty),
        mask=block_offset[:, None] < n,
    )
    tl.store(
        DK_trans_block_ptr,
        dk_trans.to(DK_trans_block_ptr.dtype.element_ty),
        mask=block_offset[None, :] < n,
    )
    tl.store(
        DV_block_ptr,
        dv.to(DV_block_ptr.dtype.element_ty),
        mask=block_offset[:, None] < n,
    )


@triton.jit
def _bwd_inter_kernel(
    Q,
    K,
    V,
    DO,
    DQ,
    DK,
    DV,
    b: tl.constexpr,
    h: tl.constexpr,
    n: tl.constexpr,
    d: tl.constexpr,
    e: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_BLOCK: tl.constexpr,
    CBLOCK: tl.constexpr,
    NUM_CBLOCK: tl.constexpr,
):
    ##### get offset
    off_bh = tl.program_id(0)
    # Removed: off_bh % h (was unused)

    qk_offset = off_bh * n * d
    v_offset = off_bh * n * e
    o_offset = off_bh * n * e

    ##### get block ptr
    DQ_block_ptr = (
        DQ + qk_offset + tl.arange(0, CBLOCK)[:, None] * d + tl.arange(0, d)[None, :]
    )
    K_block_ptr = (
        K + qk_offset + tl.arange(0, CBLOCK)[:, None] * d + tl.arange(0, d)[None, :]
    )
    V_trans_block_ptr = (
        V + v_offset + tl.arange(0, CBLOCK)[None, :] * e + tl.arange(0, e)[:, None]
    )
    DO_block_ptr = (
        DO + o_offset + tl.arange(0, CBLOCK)[:, None] * e + tl.arange(0, e)[None, :]
    )
    # mask
    off_block1 = tl.arange(0, CBLOCK)
    off_block2 = tl.arange(0, CBLOCK)

    ##### init lambda; kv
    kv_trans = tl.zeros([e, d], dtype=tl.float32)

    ##### compute dq inter
    for i in range(NUM_BLOCK):
        # compute in subblock
        for j in range(NUM_CBLOCK):
            if i > 0:  # if not add this, may have bug
                do = tl.load(DO_block_ptr, mask=off_block1[:, None] < n, other=0.0).to(
                    tl.float32
                )
                dq_inter = tl.dot(do, kv_trans)
                dq = dq_inter + tl.load(
                    DQ_block_ptr, mask=off_block1[:, None] < n, other=0.0
                )
                tl.store(
                    DQ_block_ptr,
                    dq.to(DQ_block_ptr.dtype.element_ty),
                    mask=off_block1[:, None] < n,
                )

            DQ_block_ptr += CBLOCK * d
            DO_block_ptr += CBLOCK * e
            off_block1 += CBLOCK

        # update kv in subblock
        kv_trans_current = tl.zeros([e, d], dtype=tl.float32)
        for j in range(NUM_CBLOCK):
            v_trans = tl.load(
                V_trans_block_ptr, mask=off_block2[None, :] < n, other=0.0
            ).to(tl.float32)
            k = tl.load(K_block_ptr, mask=off_block2[:, None] < n, other=0.0).to(
                tl.float32
            )
            kv_trans_current += tl.dot(v_trans, k)

            K_block_ptr += CBLOCK * d
            V_trans_block_ptr += CBLOCK * e
            off_block2 += CBLOCK

        kv_trans += kv_trans_current

    ##### get block ptr
    m = NUM_BLOCK * BLOCK
    off_block1 = m + tl.arange(0, CBLOCK)
    off_block2 = m + tl.arange(0, CBLOCK)

    Q_trans_block_ptr = (
        Q
        + qk_offset
        + m * d
        + tl.arange(0, CBLOCK)[None, :] * d
        + tl.arange(0, d)[:, None]
    )
    K_block_ptr = (
        K
        + qk_offset
        + m * d
        + tl.arange(0, CBLOCK)[:, None] * d
        + tl.arange(0, d)[None, :]
    )
    V_trans_block_ptr = (
        V
        + v_offset
        + m * e
        + tl.arange(0, CBLOCK)[None, :] * e
        + tl.arange(0, e)[:, None]
    )

    DK_trans_block_ptr = (
        DK
        + qk_offset
        + m * d
        + tl.arange(0, CBLOCK)[None, :] * d
        + tl.arange(0, d)[:, None]
    )
    DV_block_ptr = (
        DV
        + v_offset
        + m * e
        + tl.arange(0, CBLOCK)[:, None] * e
        + tl.arange(0, e)[None, :]
    )
    DO_block_ptr = (
        DO
        + o_offset
        + m * e
        + tl.arange(0, CBLOCK)[:, None] * e
        + tl.arange(0, e)[None, :]
    )

    ##### init dkv
    dkv = tl.zeros([d, e], dtype=tl.float32)

    ##### compute dk, dv inter
    for i in range(NUM_BLOCK - 1, -1, -1):
        # compute in subblock
        for j in range(NUM_CBLOCK - 1, -1, -1):
            K_block_ptr -= CBLOCK * d
            V_trans_block_ptr -= CBLOCK * e
            DK_trans_block_ptr -= CBLOCK * d
            DV_block_ptr -= CBLOCK * e
            off_block1 -= CBLOCK

            if i < NUM_BLOCK - 1:  # if not add this, may have bug
                k = tl.load(K_block_ptr, mask=off_block1[:, None] < n, other=0.0).to(
                    tl.float32
                )
                v_trans = tl.load(
                    V_trans_block_ptr, mask=off_block1[None, :] < n, other=0.0
                ).to(tl.float32)

                dk_inter_trans = tl.dot(dkv, v_trans)
                dv_inter = tl.dot(k, dkv)

                dk_trans = dk_inter_trans + tl.load(
                    DK_trans_block_ptr, mask=off_block1[None, :] < n, other=0.0
                )
                dv = dv_inter + tl.load(
                    DV_block_ptr, mask=off_block1[:, None] < n, other=0.0
                )

                tl.store(
                    DK_trans_block_ptr,
                    dk_trans.to(DK_trans_block_ptr.dtype.element_ty),
                    mask=off_block1[None, :] < n,
                )
                tl.store(
                    DV_block_ptr,
                    dv.to(DV_block_ptr.dtype.element_ty),
                    mask=off_block1[:, None] < n,
                )

        # update dkv in subblock
        dkv_current = tl.zeros([d, e], dtype=tl.float32)
        for j in range(NUM_CBLOCK - 1, -1, -1):
            DO_block_ptr -= CBLOCK * e
            Q_trans_block_ptr -= CBLOCK * d
            off_block2 -= CBLOCK

            do = tl.load(DO_block_ptr, mask=off_block2[:, None] < n, other=0.0).to(
                tl.float32
            )
            q_trans = tl.load(
                Q_trans_block_ptr, mask=off_block2[None, :] < n, other=0.0
            ).to(tl.float32)
            dkv_current += tl.dot(q_trans, do)

        dkv += dkv_current


class LightningAttention3NoDecay(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v):
        # Skip contiguous() if already contiguous
        if not q.is_contiguous():
            q = q.contiguous()
        if not k.is_contiguous():
            k = k.contiguous()
        if not v.is_contiguous():
            v = v.contiguous()

        b, h, n, d = q.shape
        e = v.shape[-1]
        o = torch.empty((b, h, n, e), dtype=q.dtype, device=q.device)

        # Grid size for autotune - use max possible BLOCK_MODEL
        BLOCK_MODEL_MAX = min(triton.next_power_of_2(e), 64)
        grid = lambda meta: (b * h, triton.cdiv(e, meta['BLOCK_MODEL']))

        _fwd_kernel[grid](
            q, k, v, o,
            b, h, n, d, e,
            NUM_BLOCK=triton.cdiv(n, 64),  # Hint for autotune
        )

        # Save for backward
        ctx.save_for_backward(q, k, v)
        ctx.n = n
        ctx.d = d
        ctx.e = e

        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v = ctx.saved_tensors
        
        # Use cached dimensions from forward
        n, d, e = ctx.n, ctx.d, ctx.e
        b, h = q.shape[:2]

        # Skip contiguous() if already contiguous (common case)
        if not do.is_contiguous():
            do = do.contiguous()

        # Pre-allocate output gradients
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)

        # --- Hardware-aware backward kernel selection ---
        device_idx = q.device.index if q.device.index is not None else 0
        if device_idx not in _BWD_KERNEL_CACHE:
            _BWD_KERNEL_CACHE[device_idx] = _select_backward_kernel(q.device, d, e)

        kernel_type, block_or_cblock, cblock_or_zero = _BWD_KERNEL_CACHE[device_idx]

        if kernel_type == "chunked":
            # Blackwell (SM 12.x): Use recompute-heavy chunked kernels
            from .lightning_attn3_no_decay_chunked import (
                _bwd_intra_chunked_kernel,
                _bwd_inter_chunked_kernel,
            )
            
            # Optimal configs tuned for Blackwell:
            # - Intra: CBLOCK=32 is fastest (O(n²) within block benefits from smaller chunks)
            # - Inter: CBLOCK=64 is fastest (fewer iterations, better tensor core utilization)
            CBLOCK_INTRA = block_or_cblock  # Usually 32
            CBLOCK_INTER = 64  # Fixed at 64 for better performance
            BLOCK = 64
            NUM_BLOCK = triton.cdiv(n, BLOCK)
            NUM_CBLOCK_PER_BLOCK = BLOCK // CBLOCK_INTRA

            # Intra-chunk backward: parallel over blocks
            grid_intra = (b * h, NUM_BLOCK)
            _bwd_intra_chunked_kernel[grid_intra](
                q, k, v, do,
                dq, dk, dv,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                do.stride(0), do.stride(1), do.stride(2), do.stride(3),
                n=n, d=d, e=e,
                BLOCK=BLOCK,
                CBLOCK=CBLOCK_INTRA,
                NUM_CBLOCK=NUM_CBLOCK_PER_BLOCK,
                num_warps=4,
                num_stages=1,
            )

            # Inter-chunk backward: sequential over chunks (CBLOCK=64 for fewer iterations)
            NUM_CBLOCK_TOTAL = triton.cdiv(n, CBLOCK_INTER)
            grid_inter = (b * h,)
            _bwd_inter_chunked_kernel[grid_inter](
                q, k, v, do,
                dq, dk, dv,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                do.stride(0), do.stride(1), do.stride(2), do.stride(3),
                n=n, d=d, e=e,
                CBLOCK=CBLOCK_INTER,
                NUM_CBLOCK=NUM_CBLOCK_TOTAL,
                num_warps=4,
                num_stages=2,  # Software pipelining for inter kernel
            )
        else:
            # Pre-Blackwell: Use original LA3 kernels
            BLOCK = block_or_cblock
            CBLOCK = cblock_or_zero
            NUM_BLOCK = triton.cdiv(n, BLOCK)
            NUM_CBLOCK = BLOCK // CBLOCK

            # Intra part: compute in parallel
            grid = (b * h, NUM_BLOCK)
            _bwd_intra_kernel[grid](
                q, k, v, do,
                dq, dk, dv,
                b, h, n, d, e,
                BLOCK=BLOCK,
                NUM_BLOCK=NUM_BLOCK,
                CBLOCK=CBLOCK,
                NUM_CBLOCK=NUM_CBLOCK,
                num_warps=4,
                num_stages=1,
            )

            # Inter part: compute sequentially
            grid = (b * h,)
            _bwd_inter_kernel[grid](
                q, k, v, do,
                dq, dk, dv,
                b, h, n, d, e,
                BLOCK=BLOCK,
                NUM_BLOCK=NUM_BLOCK,
                CBLOCK=CBLOCK,
                NUM_CBLOCK=NUM_CBLOCK,
                num_warps=4,
                num_stages=1,
            )

        return dq, dk, dv


lightning_attn3_no_decay = LightningAttention3NoDecay.apply
