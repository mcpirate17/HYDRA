# Lightning Attention-3 - HYDRA Project
# Based on Lightning Attention (https://github.com/OpenNLPLab/lightning-attention)
# Parallel variant for improved throughput
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl
import os

from .tuning_cache import get as _tcache_get
from .tuning_cache import set as _tcache_set
from .tuning_cache import make_key as _tcache_key


# Debug/benchmark visibility: populated during forward/backward
# Format: (BLOCK, CBLOCK, num_stages, num_warps)
_LAST_PARALLEL_CONFIG: tuple[int, int, int, int] | None = None

# Cache best measured parallel configs per shape/device.
# Key: (device_idx, bh, n, d, e, dtype)
_PARALLEL_CONFIG_CACHE: dict[
    tuple[int, int, int, int, int, torch.dtype],
    tuple[int, int, int, int],
] = {}

# Internal escape hatch used by the autotuner to avoid recursion.
_FORCE_PARALLEL_CONFIG: tuple[int, int, int, int] | None = None


def _autotune_parallel_config(
    *,
    device_idx: int,
    b: int,
    h: int,
    n: int,
    d: int,
    e: int,
    dtype: torch.dtype,
) -> tuple[int, int, int, int]:
    """Pick (BLOCK, CBLOCK, num_stages, num_warps) by timing a small grid once.

    Notes:
      - The parallel kernel currently requires `n % BLOCK == 0`.
      - We keep the grid intentionally small to limit first-run Triton compile cost.
      - Disable with `HYDRA_LA3_AUTOTUNE_PARALLEL=0`.
    """
    if os.getenv('HYDRA_LA3_AUTOTUNE_PARALLEL', '1') in ('0', '', 'false', 'False'):
        return (256, 64, 2, 4)

    bh = b * h
    key = (device_idx, bh, n, d, e, dtype)
    cached = _PARALLEL_CONFIG_CACHE.get(key)
    if cached is not None:
        return cached

    persisted = _tcache_get('parallel', _tcache_key(device_idx, bh, n, d, e, str(dtype)))
    if (
        isinstance(persisted, list)
        and len(persisted) == 4
        and all(isinstance(x, int) for x in persisted)
    ):
        cfg = (int(persisted[0]), int(persisted[1]), int(persisted[2]), int(persisted[3]))
        _PARALLEL_CONFIG_CACHE[key] = cfg
        return cfg

    # Candidate grid (small, safe defaults). Include smaller blocks so tests and
    # short sequence lengths (e.g. n=64) are supported.
    block_candidates = tuple(bc for bc in (256, 128, 64, 32) if bc <= n)
    cblock_candidates = (64, 32, 16)
    stage_candidates = (2, 1)
    # Note: on RTX 5090, micro-sweeps show warps=2 can win at N=2048.
    warp_candidates = (8, 4, 2) if n <= 2048 else (8, 4)

    candidates: list[tuple[int, int, int, int]] = []
    for block in block_candidates:
        if n % block != 0:
            continue
        for cblock in cblock_candidates:
            if cblock > block or block % cblock != 0:
                continue
            for stages in stage_candidates:
                for warps in warp_candidates:
                    # Avoid 8-warps for tiny tiles.
                    if warps == 8 and cblock < 32:
                        continue
                    candidates.append((block, cblock, stages, warps))

    if not candidates:
        fallback = (256, 64, 2, 4)
        _PARALLEL_CONFIG_CACHE[key] = fallback
        return fallback

    # Representative tensors for timing.
    q = torch.randn((b, h, n, d), device='cuda', dtype=dtype, requires_grad=True)
    k = torch.randn((b, h, n, d), device='cuda', dtype=dtype, requires_grad=True)
    v = torch.randn((b, h, n, e), device='cuda', dtype=dtype, requires_grad=True)
    s = torch.ones((h,), device='cuda', dtype=dtype)

    best_cfg: tuple[int, int, int, int] | None = None
    best_ms: float = float('inf')

    global _FORCE_PARALLEL_CONFIG
    try:
        # NOTE: this autotuner may be invoked from inside the custom autograd
        # forward (which runs under torch.no_grad()). Enable grad explicitly.
        with torch.enable_grad():
            # Compile + warm each candidate once.
            for cfg in candidates:
                _FORCE_PARALLEL_CONFIG = cfg
                q.grad = k.grad = v.grad = None
                out = LightningAttention3Parallel.apply(q, k, v, s)
                out.sum().backward()
            torch.cuda.synchronize()

            for cfg in candidates:
                _FORCE_PARALLEL_CONFIG = cfg

                def _bwd():
                    q.grad = k.grad = v.grad = None
                    out = LightningAttention3Parallel.apply(q, k, v, s)
                    out.sum().backward()

                ms = triton.testing.do_bench(_bwd, warmup=1, rep=15)
                if ms < best_ms:
                    best_ms = float(ms)
                    best_cfg = cfg
    finally:
        _FORCE_PARALLEL_CONFIG = None

    assert best_cfg is not None
    _PARALLEL_CONFIG_CACHE[key] = best_cfg
    _tcache_set('parallel', _tcache_key(device_idx, bh, n, d, e, str(dtype)), [best_cfg[0], best_cfg[1], best_cfg[2], best_cfg[3]])
    return best_cfg


@triton.jit
def _fwd_diag_kernel(
    Q,
    K,
    V,
    Out,
    S,
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
    off = tl.program_id(0)
    off_bh = off // NUM_BLOCK
    off_block = off % NUM_BLOCK
    off_cblock = tl.program_id(1)

    off_h = off_bh % h

    qk_offset = off_bh * n * d
    v_offset = off_bh * n * e
    o_offset = off_bh * n * e

    block_offset = off_block * BLOCK
    qk_block_offset = block_offset * d
    v_block_offset = block_offset * e
    o_block_offset = block_offset * e

    cblock_offset = off_cblock * CBLOCK
    q_cblock_offset = cblock_offset * d
    o_cblock_offset = cblock_offset * e

    Q_block_ptr = (
        Q
        + qk_offset
        + qk_block_offset
        + q_cblock_offset
        + tl.arange(0, CBLOCK)[:, None] * d
        + tl.arange(0, d)[None, :]
    )
    K_trans_block_ptr = (
        K
        + qk_offset
        + qk_block_offset
        + tl.arange(0, CBLOCK)[None, :] * d
        + tl.arange(0, d)[:, None]
    )
    V_block_ptr = (
        V
        + v_offset
        + v_block_offset
        + tl.arange(0, CBLOCK)[:, None] * e
        + tl.arange(0, e)[None, :]
    )
    O_block_ptr = (
        Out
        + o_offset
        + o_block_offset
        + o_cblock_offset
        + tl.arange(0, CBLOCK)[:, None] * e
        + tl.arange(0, e)[None, :]
    )

    S_block_ptr = S + off_h
    s = tl.load(S_block_ptr)

    i = off_cblock
    q_index = tl.arange(0, CBLOCK) + i * CBLOCK

    q = tl.load(Q_block_ptr, mask=q_index[:, None] < n, other=0.0).to(tl.float32)

    qkv = tl.zeros([CBLOCK, e], dtype=tl.float32)
    # none diag

    for j in range(i + 1):
        kv_index = tl.arange(0, CBLOCK) + j * CBLOCK
        diff = q_index[:, None] - kv_index[None, :]
        s_index = s.to(tl.float32) * diff.to(tl.float32)
        s_index = tl.where(diff >= 0, -s_index, float("-inf"))
        decay = tl.exp(s_index)

        k_trans = tl.load(K_trans_block_ptr, mask=kv_index[None, :] < n, other=0.0).to(
            tl.float32
        )
        v = tl.load(V_block_ptr, mask=kv_index[:, None] < n, other=0.0).to(tl.float32)

        qk = tl.dot(q, k_trans) * decay

        qkv += tl.dot(qk, v)

        K_trans_block_ptr += CBLOCK * d
        V_block_ptr += CBLOCK * e

    tl.store(
        O_block_ptr, qkv.to(O_block_ptr.dtype.element_ty), mask=q_index[:, None] < n
    )


@triton.jit
def _fwd_kv_parallel(
    K,
    V,
    S,
    KV,
    b: tl.constexpr,
    h: tl.constexpr,
    n: tl.constexpr,
    d: tl.constexpr,
    e: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_BLOCK: tl.constexpr,
    D_FBLOCK: tl.constexpr,
    E_FBLOCK: tl.constexpr,
    NUM_FBLOCK: tl.constexpr,
    CBLOCK: tl.constexpr,
    NUM_CBLOCK: tl.constexpr,
):
    off_bh = tl.program_id(0)
    off_block = tl.program_id(1)
    off_de = tl.program_id(2)

    off_h = off_bh % h
    off_d = off_de // NUM_FBLOCK
    off_e = off_de % NUM_FBLOCK

    block_offset = off_block * BLOCK

    k_block_offset = block_offset * d
    v_block_offset = block_offset * e
    kv_block_offset = off_block * d * e

    k_offset = off_bh * n * d
    v_offset = off_bh * n * e
    kv_offset = off_bh * NUM_BLOCK * d * e
    d_offset = off_d * D_FBLOCK
    e_offset = off_e * E_FBLOCK

    # (CBLOCK, FBLOCK)
    K_trans_block_ptr = (
        K
        + k_offset
        + k_block_offset
        + d_offset
        + tl.arange(0, CBLOCK)[None, :] * d
        + tl.arange(0, D_FBLOCK)[:, None]
    )
    V_block_ptr = (
        V
        + v_offset
        + v_block_offset
        + e_offset
        + tl.arange(0, CBLOCK)[:, None] * e
        + tl.arange(0, E_FBLOCK)[None, :]
    )
    KV_block_ptr = (
        KV
        + kv_offset
        + kv_block_offset
        + d_offset * e
        + e_offset
        + tl.arange(0, D_FBLOCK)[:, None] * e
        + tl.arange(0, E_FBLOCK)[None, :]
    )

    s_ptrs = S + off_h
    s = tl.load(s_ptrs)

    # compute block array
    c_array = tl.arange(0, CBLOCK)

    kv = tl.zeros([D_FBLOCK, E_FBLOCK], dtype=tl.float32)
    for j in range(NUM_CBLOCK):
        k_trans = tl.load(K_trans_block_ptr).to(tl.float32)
        v = tl.load(V_block_ptr).to(tl.float32)
        k_decay = tl.exp(-s.to(tl.float32) * (BLOCK - (j * CBLOCK + c_array[None, :])))

        kv += tl.dot(k_trans * k_decay, v)

        K_trans_block_ptr += CBLOCK * d
        V_block_ptr += CBLOCK * e

    tl.store(KV_block_ptr, kv.to(KV_block_ptr.dtype.element_ty))


@triton.jit
def _fwd_kv_reduce(
    K,
    V,
    S,
    KV,
    b: tl.constexpr,
    h: tl.constexpr,
    n: tl.constexpr,
    d: tl.constexpr,
    e: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_BLOCK: tl.constexpr,
    D_FBLOCK: tl.constexpr,
    E_FBLOCK: tl.constexpr,
    NUM_FBLOCK: tl.constexpr,
    CBLOCK: tl.constexpr,
    NUM_CBLOCK: tl.constexpr,
):
    off_bh = tl.program_id(0)
    off_h = off_bh % h
    off_d = tl.program_id(1)
    off_e = tl.program_id(2)

    kv_offset = off_bh * NUM_BLOCK * d * e
    d_offset = off_d * D_FBLOCK
    e_offset = off_e * E_FBLOCK

    # (CBLOCK, FBLOCK)
    KV_block_ptr = (
        KV
        + kv_offset
        + d_offset * e
        + e_offset
        + tl.arange(0, D_FBLOCK)[:, None] * e
        + tl.arange(0, E_FBLOCK)[None, :]
    )

    s_ptrs = S + off_h
    s = tl.load(s_ptrs)

    block_decay = tl.exp(-s.to(tl.float32) * BLOCK)

    # compute block array

    kv = tl.zeros([D_FBLOCK, E_FBLOCK], dtype=tl.float32)
    for i in range(NUM_BLOCK):
        kv_current = tl.load(KV_block_ptr).to(tl.float32)
        tl.store(KV_block_ptr, kv.to(KV_block_ptr.dtype.element_ty))

        kv = block_decay * kv + kv_current
        KV_block_ptr += d * e


@triton.jit
def _fwd_none_diag_kernel(
    Q,
    K,
    V,
    Out,
    S,
    KV,
    b: tl.constexpr,
    h: tl.constexpr,
    n: tl.constexpr,
    d: tl.constexpr,
    e: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_BLOCK: tl.constexpr,
    D_FBLOCK: tl.constexpr,
    E_FBLOCK: tl.constexpr,
    NUM_FBLOCK: tl.constexpr,
):
    off_bh = tl.program_id(0)
    off_h = off_bh % h

    off_n = tl.program_id(1)
    off_e = tl.program_id(2)

    n_offset = off_n * BLOCK
    e_offset = off_e * E_FBLOCK

    q_offset = off_bh * n * d + n_offset * d
    o_offset = off_bh * n * e + n_offset * e + e_offset

    kv_offset = off_bh * NUM_BLOCK * d * e + off_n * d * e + e_offset

    Q_block_ptr = (
        Q + q_offset + tl.arange(0, BLOCK)[:, None] * d + tl.arange(0, d)[None, :]
    )
    O_block_ptr = (
        Out
        + o_offset
        + tl.arange(0, BLOCK)[:, None] * e
        + tl.arange(0, E_FBLOCK)[None, :]
    )
    KV_block_ptr = (
        KV + kv_offset + tl.arange(0, d)[:, None] * e + tl.arange(0, E_FBLOCK)[None, :]
    )
    S_block_ptr = S + off_h
    s = tl.load(S_block_ptr)

    array = tl.arange(0, BLOCK)

    q = tl.load(Q_block_ptr).to(tl.float32)
    kv = tl.load(KV_block_ptr).to(tl.float32)
    q_decay = tl.exp(-s.to(tl.float32) * array[:, None])
    qkv_none_diag = tl.dot(q, kv) * q_decay
    qkv_diag = tl.load(O_block_ptr).to(tl.float32)

    qkv = qkv_diag + qkv_none_diag

    tl.store(O_block_ptr, qkv.to(O_block_ptr.dtype.element_ty))


##### total parallel
@triton.jit
def _fwd_none_diag_kernel(
    Q,
    K,
    V,
    Out,
    S,
    KV,
    b: tl.constexpr,
    h: tl.constexpr,
    n: tl.constexpr,
    d: tl.constexpr,
    e: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_BLOCK: tl.constexpr,
    D_FBLOCK: tl.constexpr,
    E_FBLOCK: tl.constexpr,
    NUM_FBLOCK: tl.constexpr,
    CBLOCK: tl.constexpr,
    NUM_CBLOCK: tl.constexpr,
):
    off_bh = tl.program_id(0)
    off_h = off_bh % h

    off_nc = tl.program_id(1)
    off_n = off_nc // NUM_CBLOCK
    off_c = off_nc % NUM_CBLOCK
    off_e = tl.program_id(2)

    n_offset = off_n * BLOCK
    c_offset = off_c * CBLOCK
    e_offset = off_e * E_FBLOCK

    q_offset = off_bh * n * d + (n_offset + c_offset) * d
    o_offset = off_bh * n * e + (n_offset + c_offset) * e + e_offset

    kv_offset = off_bh * NUM_BLOCK * d * e + off_n * d * e + e_offset

    Q_block_ptr = (
        Q + q_offset + tl.arange(0, CBLOCK)[:, None] * d + tl.arange(0, d)[None, :]
    )
    O_block_ptr = (
        Out
        + o_offset
        + tl.arange(0, CBLOCK)[:, None] * e
        + tl.arange(0, E_FBLOCK)[None, :]
    )
    KV_block_ptr = (
        KV + kv_offset + tl.arange(0, d)[:, None] * e + tl.arange(0, E_FBLOCK)[None, :]
    )
    S_block_ptr = S + off_h
    s = tl.load(S_block_ptr)

    c_array = tl.arange(0, CBLOCK)

    kv = tl.load(KV_block_ptr).to(tl.float32)
    q = tl.load(Q_block_ptr).to(tl.float32)
    q_decay = tl.exp(-s.to(tl.float32) * (off_c * CBLOCK + c_array[:, None]))
    qkv_none_diag = tl.dot(q, kv) * q_decay
    qkv_diag = tl.load(O_block_ptr).to(tl.float32)

    qkv = qkv_diag + qkv_none_diag

    tl.store(O_block_ptr, qkv.to(O_block_ptr.dtype.element_ty))


###################### bwd
@triton.jit
def _bwd_diag_kernel(
    Q,
    K,
    V,
    S,
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
    off = tl.program_id(0)
    off_bh = off // NUM_BLOCK
    off_block = off % NUM_BLOCK
    off_cblock = tl.program_id(1)

    off_h = off_bh % h

    #####
    qk_offset = off_bh * n * d
    v_offset = off_bh * n * e
    o_offset = off_bh * n * e

    block_offset = off_block * BLOCK
    qk_block_offset = block_offset * d
    v_block_offset = block_offset * e
    o_block_offset = block_offset * e

    cblock_offset = off_cblock * CBLOCK
    qk_cblock_offset = cblock_offset * d
    v_cblock_offset = cblock_offset * e
    o_cblock_offset = cblock_offset * e

    S_block_ptr = S + off_h
    s = tl.load(S_block_ptr)

    # dq
    DO_block_ptr = (
        DO
        + o_offset
        + o_block_offset
        + o_cblock_offset
        + tl.arange(0, CBLOCK)[:, None] * e
        + tl.arange(0, e)[None, :]
    )
    DQ_block_ptr = (
        DQ
        + qk_offset
        + qk_block_offset
        + qk_cblock_offset
        + tl.arange(0, CBLOCK)[:, None] * d
        + tl.arange(0, d)[None, :]
    )
    K_block_ptr = (
        K
        + qk_offset
        + qk_block_offset
        + tl.arange(0, CBLOCK)[:, None] * d
        + tl.arange(0, d)[None, :]
    )
    V_trans_block_ptr = (
        V
        + v_offset
        + v_block_offset
        + tl.arange(0, CBLOCK)[None, :] * e
        + tl.arange(0, e)[:, None]
    )

    do = tl.load(DO_block_ptr).to(tl.float32)
    dq = tl.zeros([CBLOCK, d], dtype=tl.float32)

    i = off_cblock
    do_index = tl.arange(0, CBLOCK) + i * CBLOCK
    for j in range(i + 1):
        k = tl.load(K_block_ptr).to(tl.float32)
        v_trans = tl.load(V_trans_block_ptr).to(tl.float32)

        # compute
        v_index = tl.arange(0, CBLOCK) + j * CBLOCK
        diff = do_index[:, None] - v_index[None, :]
        s_index = s.to(tl.float32) * diff.to(tl.float32)
        s_index = tl.where(diff >= 0, -s_index, float("-inf"))
        diag_decay = tl.exp(s_index)

        dqk = tl.dot(do, v_trans) * diag_decay
        dq += tl.dot(dqk, k)

        K_block_ptr += CBLOCK * d
        V_trans_block_ptr += CBLOCK * e

    tl.store(DQ_block_ptr, dq.to(DQ_block_ptr.dtype.element_ty))

    # dk
    V_trans_block_ptr = (
        V
        + v_offset
        + v_block_offset
        + v_cblock_offset
        + tl.arange(0, CBLOCK)[None, :] * e
        + tl.arange(0, e)[:, None]
    )
    DO_block_ptr = (
        DO
        + o_offset
        + o_block_offset
        + o_cblock_offset
        + tl.arange(0, CBLOCK)[:, None] * e
        + tl.arange(0, e)[None, :]
    )
    Q_trans_block_ptr = (
        Q
        + qk_offset
        + qk_block_offset
        + qk_cblock_offset
        + tl.arange(0, CBLOCK)[None, :] * d
        + tl.arange(0, d)[:, None]
    )
    DK_trans_block_ptr = (
        DK
        + qk_offset
        + qk_block_offset
        + qk_cblock_offset
        + tl.arange(0, CBLOCK)[None, :] * d
        + tl.arange(0, d)[:, None]
    )

    v_trans = tl.load(V_trans_block_ptr).to(tl.float32)
    v_index = tl.arange(0, CBLOCK) + i * CBLOCK
    dk_trans = tl.zeros([d, CBLOCK], dtype=tl.float32)

    # add
    K_block_ptr = (
        K
        + qk_offset
        + qk_block_offset
        + qk_cblock_offset
        + tl.arange(0, CBLOCK)[:, None] * d
        + tl.arange(0, d)[None, :]
    )
    DV_block_ptr = (
        DV
        + v_offset
        + v_block_offset
        + v_cblock_offset
        + tl.arange(0, CBLOCK)[:, None] * e
        + tl.arange(0, e)[None, :]
    )

    dv = tl.zeros([CBLOCK, e], dtype=tl.float32)
    k = tl.load(K_block_ptr).to(tl.float32)
    for j in range(i, NUM_CBLOCK):
        q_trans = tl.load(Q_trans_block_ptr).to(tl.float32)
        do = tl.load(DO_block_ptr).to(tl.float32)

        do_index = tl.arange(0, CBLOCK) + j * CBLOCK
        diff = do_index[:, None] - v_index[None, :]
        s_index = s.to(tl.float32) * diff.to(tl.float32)
        s_index = tl.where(diff >= 0, -s_index, float("-inf"))
        diag_decay = tl.exp(s_index)

        dqk = tl.dot(do, v_trans) * diag_decay
        dk_trans += tl.dot(q_trans, dqk)

        Q_trans_block_ptr += CBLOCK * d
        DO_block_ptr += CBLOCK * e

        # add
        diag_decay_trans = tl.trans(diag_decay)
        qk_trans = tl.dot(k, q_trans) * diag_decay_trans
        dv += tl.dot(qk_trans, do)

    tl.store(DK_trans_block_ptr, dk_trans.to(DK_trans_block_ptr.dtype.element_ty))
    tl.store(DV_block_ptr, dv.to(DV_block_ptr.dtype.element_ty))


@triton.jit
def _bwd_dkv_parallel(
    Q,
    DO,
    S,
    DKV,
    b: tl.constexpr,
    h: tl.constexpr,
    n: tl.constexpr,
    d: tl.constexpr,
    e: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_BLOCK: tl.constexpr,
    D_FBLOCK: tl.constexpr,
    E_FBLOCK: tl.constexpr,
    NUM_FBLOCK: tl.constexpr,
    CBLOCK: tl.constexpr,
    NUM_CBLOCK: tl.constexpr,
):
    off_bh = tl.program_id(0)
    off_block = tl.program_id(1)
    off_de = tl.program_id(2)

    off_h = off_bh % h
    off_d = off_de // NUM_FBLOCK
    off_e = off_de % NUM_FBLOCK

    block_offset = off_block * BLOCK
    qk_block_offset = block_offset * d
    o_block_offset = block_offset * e
    kv_block_offset = off_block * d * e

    qk_offset = off_bh * n * d
    o_offset = off_bh * n * e
    kv_offset = off_bh * NUM_BLOCK * d * e
    d_offset = off_d * D_FBLOCK
    e_offset = off_e * E_FBLOCK

    # (CBLOCK, FBLOCK)
    DKV_block_ptr = (
        DKV
        + kv_offset
        + kv_block_offset
        + d_offset * e
        + e_offset
        + tl.arange(0, D_FBLOCK)[:, None] * e
        + tl.arange(0, E_FBLOCK)[None, :]
    )

    Q_trans_block_ptr = (
        Q
        + qk_offset
        + qk_block_offset
        + d_offset
        + tl.arange(0, CBLOCK)[None, :] * d
        + tl.arange(0, D_FBLOCK)[:, None]
    )
    DO_block_ptr = (
        DO
        + o_offset
        + o_block_offset
        + e_offset
        + tl.arange(0, CBLOCK)[:, None] * e
        + tl.arange(0, E_FBLOCK)[None, :]
    )

    s_ptrs = S + off_h
    s = tl.load(s_ptrs)

    c_array = tl.arange(0, CBLOCK)

    dkv = tl.zeros([D_FBLOCK, E_FBLOCK], dtype=tl.float32)

    for j in range(NUM_CBLOCK):
        do = tl.load(DO_block_ptr).to(tl.float32)
        q_trans = tl.load(Q_trans_block_ptr).to(tl.float32)
        q_decay_trans = tl.exp(-s.to(tl.float32) * (j * CBLOCK + c_array[None, :]))
        dkv += tl.dot(q_trans * q_decay_trans, do)

        DO_block_ptr += CBLOCK * e
        Q_trans_block_ptr += CBLOCK * d

    tl.store(DKV_block_ptr, dkv.to(DKV_block_ptr.dtype.element_ty))


@triton.jit
def _bwd_dkv_reduce(
    Q,
    DO,
    S,
    DKV,
    b: tl.constexpr,
    h: tl.constexpr,
    n: tl.constexpr,
    d: tl.constexpr,
    e: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_BLOCK: tl.constexpr,
    D_FBLOCK: tl.constexpr,
    E_FBLOCK: tl.constexpr,
    NUM_FBLOCK: tl.constexpr,
    CBLOCK: tl.constexpr,
    NUM_CBLOCK: tl.constexpr,
):
    off_bh = tl.program_id(0)
    off_h = off_bh % h
    off_d = tl.program_id(1)
    off_e = tl.program_id(2)

    kv_offset = off_bh * NUM_BLOCK * d * e
    d_offset = off_d * D_FBLOCK
    e_offset = off_e * E_FBLOCK

    DKV_block_ptr = (
        DKV
        + kv_offset
        + d_offset * e
        + e_offset
        + NUM_BLOCK * d * e
        + tl.arange(0, D_FBLOCK)[:, None] * e
        + tl.arange(0, E_FBLOCK)[None, :]
    )

    s_ptrs = S + off_h
    s = tl.load(s_ptrs)

    block_decay = tl.exp(-s.to(tl.float32) * BLOCK)

    # compute block array

    dkv = tl.zeros([D_FBLOCK, E_FBLOCK], dtype=tl.float32)
    for i in range(NUM_BLOCK - 1, -1, -1):
        DKV_block_ptr -= d * e
        dkv_current = tl.load(DKV_block_ptr).to(tl.float32)
        tl.store(DKV_block_ptr, dkv.to(DKV_block_ptr.dtype.element_ty))

        dkv = block_decay * dkv + dkv_current


@triton.jit
def _bwd_none_diag_kernel(
    Q,
    K,
    V,
    S,
    DO,
    DQ,
    DK,
    DV,
    KV,
    DKV,
    b: tl.constexpr,
    h: tl.constexpr,
    n: tl.constexpr,
    d: tl.constexpr,
    e: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_BLOCK: tl.constexpr,
    D_FBLOCK: tl.constexpr,
    E_FBLOCK: tl.constexpr,
    NUM_FBLOCK: tl.constexpr,
    CBLOCK: tl.constexpr,
    NUM_CBLOCK: tl.constexpr,
):
    off_bh = tl.program_id(0)
    off_h = off_bh % h

    off_nc = tl.program_id(1)
    off_n = off_nc // NUM_CBLOCK
    off_c = off_nc % NUM_CBLOCK
    off_de = tl.program_id(2)

    n_offset = off_n * BLOCK
    c_offset = off_c * CBLOCK
    d_offset = off_de * D_FBLOCK
    e_offset = off_de * E_FBLOCK

    qk_offset = off_bh * n * d + (n_offset + c_offset) * d
    v_offset = off_bh * n * e + (n_offset + c_offset) * e
    o_offset = off_bh * n * e + (n_offset + c_offset) * e

    kv_offset = off_bh * NUM_BLOCK * d * e + off_n * d * e
    kv_trans_offset = off_bh * NUM_BLOCK * d * e + off_n * d * e

    S_block_ptr = S + off_h
    s = tl.load(S_block_ptr)

    # dq
    DO_block_ptr = (
        DO + o_offset + tl.arange(0, CBLOCK)[:, None] * e + tl.arange(0, e)[None, :]
    )
    KV_trans_block_ptr = (
        KV
        + kv_trans_offset
        + d_offset * e
        + tl.arange(0, D_FBLOCK)[None, :] * e
        + tl.arange(0, e)[:, None]
    )
    DQ_block_ptr = (
        DQ
        + qk_offset
        + d_offset
        + tl.arange(0, CBLOCK)[:, None] * d
        + tl.arange(0, D_FBLOCK)[None, :]
    )

    c_array = tl.arange(0, CBLOCK)
    kv_trans = tl.load(KV_trans_block_ptr).to(tl.float32)
    q_decay = tl.exp(-s.to(tl.float32) * (off_c * CBLOCK + c_array[:, None]))
    do = tl.load(DO_block_ptr).to(tl.float32)
    dq_none_diag = tl.dot(do, kv_trans) * q_decay
    dq = dq_none_diag + tl.load(DQ_block_ptr)
    tl.store(DQ_block_ptr, dq.to(DQ_block_ptr.dtype.element_ty))

    # dk
    DK_trans_block_ptr = (
        DK
        + qk_offset
        + d_offset
        + tl.arange(0, CBLOCK)[None, :] * d
        + tl.arange(0, D_FBLOCK)[:, None]
    )
    DKV_block_ptr = (
        DKV
        + kv_offset
        + d_offset * e
        + tl.arange(0, D_FBLOCK)[:, None] * e
        + tl.arange(0, e)[None, :]
    )
    V_trans_block_ptr = (
        V + v_offset + tl.arange(0, CBLOCK)[None, :] * e + tl.arange(0, e)[:, None]
    )

    v_trans = tl.load(V_trans_block_ptr).to(tl.float32)
    dkv = tl.load(DKV_block_ptr).to(tl.float32)
    k_decay_trans = tl.exp(
        -s.to(tl.float32) * (BLOCK - (off_c * CBLOCK + c_array[None, :]))
    )

    dk_none_diag_trans = tl.dot(dkv, v_trans) * k_decay_trans
    dk_trans = dk_none_diag_trans + tl.load(DK_trans_block_ptr)
    tl.store(DK_trans_block_ptr, dk_trans.to(DK_trans_block_ptr.dtype.element_ty))

    # dv
    DKV_block_ptr_ = (
        DKV
        + kv_offset
        + e_offset
        + tl.arange(0, d)[:, None] * e
        + tl.arange(0, E_FBLOCK)[None, :]
    )
    K_block_ptr = (
        K + qk_offset + tl.arange(0, CBLOCK)[:, None] * d + tl.arange(0, d)[None, :]
    )
    DV_block_ptr = (
        DV
        + v_offset
        + e_offset
        + tl.arange(0, CBLOCK)[:, None] * e
        + tl.arange(0, E_FBLOCK)[None, :]
    )

    k_decay = tl.exp(-s.to(tl.float32) * (BLOCK - (off_c * CBLOCK + c_array[:, None])))
    k = tl.load(K_block_ptr).to(tl.float32)
    dkv_ = tl.load(DKV_block_ptr_).to(tl.float32)
    dv_none_diag = tl.dot(k, dkv_) * k_decay
    dv = dv_none_diag + tl.load(DV_block_ptr)
    tl.store(DV_block_ptr, dv.to(DV_block_ptr.dtype.element_ty))


class LightningAttention3Parallel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, s):
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        s = s.contiguous()

        # shape constraints
        b, h, n, d = q.shape
        e = v.shape[-1]
        # right
        o = torch.empty((b, h, n, e), dtype=q.dtype, device=q.device)

        device_idx = torch.cuda.current_device() if q.is_cuda else -1

        global _LAST_PARALLEL_CONFIG
        cfg = _FORCE_PARALLEL_CONFIG
        if cfg is None:
            cfg = _autotune_parallel_config(
                device_idx=device_idx,
                b=b,
                h=h,
                n=n,
                d=d,
                e=e,
                dtype=q.dtype,
            )

        BLOCK, CBLOCK, num_stages, num_warps = cfg

        # Ensure divisibility for short sequences / odd shapes.
        if n % BLOCK != 0:
            for bc in (256, 128, 64, 32, 16):
                if bc <= n and n % bc == 0:
                    BLOCK = bc
                    break
            else:
                BLOCK = n

        if CBLOCK > BLOCK or (BLOCK % CBLOCK) != 0:
            for cc in (64, 32, 16):
                if cc <= BLOCK and (BLOCK % cc) == 0:
                    CBLOCK = cc
                    break
            else:
                CBLOCK = BLOCK

        cfg = (BLOCK, CBLOCK, num_stages, num_warps)
        _LAST_PARALLEL_CONFIG = cfg
        ctx.PARALLEL_CFG = cfg

        NUM_BLOCK = n // BLOCK
        NUM_CBLOCK = BLOCK // CBLOCK

        grid = (b * h * NUM_BLOCK, NUM_CBLOCK)
        _fwd_diag_kernel[grid](
            q,
            k,
            v,
            o,
            s,
            b,
            h,
            n,
            d,
            e,
            BLOCK=BLOCK,
            NUM_BLOCK=NUM_BLOCK,
            CBLOCK=CBLOCK,
            NUM_CBLOCK=NUM_CBLOCK,
            num_warps=num_warps,
            num_stages=num_stages,
        )

        NUM_FBLOCK = 1
        D_FBLOCK = d // NUM_FBLOCK
        E_FBLOCK = e // NUM_FBLOCK
        assert d % NUM_FBLOCK == 0
        assert e % NUM_FBLOCK == 0
        grid = (b * h, NUM_FBLOCK, NUM_FBLOCK)

        kv = torch.empty((b, h, NUM_BLOCK, d, e), dtype=torch.float32, device=q.device)
        grid = (b * h, NUM_BLOCK, NUM_FBLOCK * NUM_FBLOCK)
        _fwd_kv_parallel[grid](
            k,
            v,
            s,
            kv,
            b,
            h,
            n,
            d,
            e,
            BLOCK=BLOCK,
            NUM_BLOCK=NUM_BLOCK,
            D_FBLOCK=D_FBLOCK,
            E_FBLOCK=E_FBLOCK,
            NUM_FBLOCK=NUM_FBLOCK,
            CBLOCK=CBLOCK,
            NUM_CBLOCK=NUM_CBLOCK,
            num_warps=num_warps,
            num_stages=num_stages,
        )

        grid = (b * h, NUM_FBLOCK, NUM_FBLOCK)
        _fwd_kv_reduce[grid](
            k,
            v,
            s,
            kv,
            b,
            h,
            n,
            d,
            e,
            BLOCK=BLOCK,
            NUM_BLOCK=NUM_BLOCK,
            D_FBLOCK=D_FBLOCK,
            E_FBLOCK=E_FBLOCK,
            NUM_FBLOCK=NUM_FBLOCK,
            CBLOCK=CBLOCK,
            NUM_CBLOCK=NUM_CBLOCK,
            num_warps=num_warps,
            num_stages=num_stages,
        )

        grid = (b * h, NUM_BLOCK * NUM_CBLOCK, NUM_FBLOCK)
        _fwd_none_diag_kernel[grid](
            q,
            k,
            v,
            o,
            s,
            kv,
            b,
            h,
            n,
            d,
            e,
            BLOCK=BLOCK,
            NUM_BLOCK=NUM_BLOCK,
            D_FBLOCK=D_FBLOCK,
            E_FBLOCK=E_FBLOCK,
            NUM_FBLOCK=NUM_FBLOCK,
            CBLOCK=CBLOCK,
            NUM_CBLOCK=NUM_CBLOCK,
            num_warps=num_warps,
            num_stages=num_stages,
        )

        ctx.save_for_backward(q, k, v, s, kv)

        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, s, kv = ctx.saved_tensors

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        s = s.contiguous()

        do = do.contiguous()
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)

        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        grid = (q.shape[0] * q.shape[1], 1)

        b, h, n, d = q.shape
        e = v.shape[-1]

        global _LAST_PARALLEL_CONFIG
        cfg = getattr(ctx, 'PARALLEL_CFG', (256, 64, 2, 4))
        BLOCK, CBLOCK, num_stages, num_warps = cfg

        # Mirror forward safety in case a config was injected externally.
        if n % BLOCK != 0:
            for bc in (256, 128, 64, 32, 16):
                if bc <= n and n % bc == 0:
                    BLOCK = bc
                    break
            else:
                BLOCK = n

        if CBLOCK > BLOCK or (BLOCK % CBLOCK) != 0:
            for cc in (64, 32, 16):
                if cc <= BLOCK and (BLOCK % cc) == 0:
                    CBLOCK = cc
                    break
            else:
                CBLOCK = BLOCK

        cfg = (BLOCK, CBLOCK, num_stages, num_warps)
        _LAST_PARALLEL_CONFIG = cfg

        NUM_BLOCK = n // BLOCK
        NUM_CBLOCK = BLOCK // CBLOCK

        grid = (b * h * NUM_BLOCK, NUM_CBLOCK)
        _bwd_diag_kernel[grid](
            q,
            k,
            v,
            s,
            do,
            dq,
            dk,
            dv,
            b,
            h,
            n,
            d,
            e,
            BLOCK=BLOCK,
            NUM_BLOCK=NUM_BLOCK,
            CBLOCK=CBLOCK,
            NUM_CBLOCK=NUM_CBLOCK,
            num_warps=num_warps,
            num_stages=num_stages,
        )

        dkv = torch.empty((b, h, NUM_BLOCK, d, e), dtype=torch.float32, device=q.device)
        NUM_FBLOCK = 1
        D_FBLOCK = d // NUM_FBLOCK
        E_FBLOCK = e // NUM_FBLOCK
        assert d % NUM_FBLOCK == 0
        assert e % NUM_FBLOCK == 0

        grid = (b * h, NUM_BLOCK, NUM_FBLOCK * NUM_FBLOCK)
        _bwd_dkv_parallel[grid](
            q,
            do,
            s,
            dkv,
            b,
            h,
            n,
            d,
            e,
            BLOCK=BLOCK,
            NUM_BLOCK=NUM_BLOCK,
            D_FBLOCK=D_FBLOCK,
            E_FBLOCK=E_FBLOCK,
            NUM_FBLOCK=NUM_FBLOCK,
            CBLOCK=CBLOCK,
            NUM_CBLOCK=NUM_CBLOCK,
            num_warps=num_warps,
            num_stages=num_stages,
        )

        grid = (b * h, NUM_FBLOCK, NUM_FBLOCK)
        _bwd_dkv_reduce[grid](
            q,
            do,
            s,
            dkv,
            b,
            h,
            n,
            d,
            e,
            BLOCK=BLOCK,
            NUM_BLOCK=NUM_BLOCK,
            D_FBLOCK=D_FBLOCK,
            E_FBLOCK=E_FBLOCK,
            NUM_FBLOCK=NUM_FBLOCK,
            CBLOCK=CBLOCK,
            NUM_CBLOCK=NUM_CBLOCK,
            num_warps=num_warps,
            num_stages=num_stages,
        )

        grid = (b * h, NUM_BLOCK * NUM_CBLOCK, NUM_FBLOCK)
        _bwd_none_diag_kernel[grid](
            q,
            k,
            v,
            s,
            do,
            dq,
            dk,
            dv,
            kv,
            dkv,
            b,
            h,
            n,
            d,
            e,
            BLOCK=BLOCK,
            NUM_BLOCK=NUM_BLOCK,
            D_FBLOCK=D_FBLOCK,
            E_FBLOCK=E_FBLOCK,
            NUM_FBLOCK=NUM_FBLOCK,
            CBLOCK=CBLOCK,
            NUM_CBLOCK=NUM_CBLOCK,
            num_warps=num_warps,
            num_stages=num_stages,
        )

        return dq, dk, dv, None


lightning_attn3_parallel = LightningAttention3Parallel.apply
