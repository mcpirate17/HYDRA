"""CPU-only correctness tests for recent CCGQA attention refactors.

These tests lock in functional equivalence for:
- Value-shift implementation (prealloc vs pad+cat)
- QK-mean coupling implementation for GQA (head-space vs expand+reshape)
- RoPE fallback math (when fused kernels are unavailable)

They intentionally avoid CUDA and run entirely on CPU.
"""

import torch
import torch.nn.functional as F

import hydra.attention.backends.ccgqa as ccgqa_mod
from hydra.attention.backends.ccgqa.attention import CCGQAAttention


def _forward_reference(
    attn: CCGQAAttention,
    x: torch.Tensor,
    *,
    value_shift_impl: str = "prealloc",
    qk_mean_impl: str = "headspace",
) -> torch.Tensor:
    """Reference forward that can switch between equivalent implementations."""
    B, S, _ = x.shape

    qkv = attn.qkv_proj(x)
    q = qkv[..., :attn.latent_dim]
    k = qkv[..., attn.latent_dim:attn.latent_dim + attn.kv_dim]

    if attn.use_value_shift:
        v_curr = qkv[..., attn.latent_dim + attn.kv_dim:attn.latent_dim + attn.kv_dim + attn.kv_dim // 2]
        v_prev = qkv[..., attn.latent_dim + attn.kv_dim + attn.kv_dim // 2:]

        if value_shift_impl == "padcat":
            v_prev_shifted = F.pad(v_prev[:, :-1, :], (0, 0, 1, 0), value=0)
            v = torch.cat([v_curr, v_prev_shifted], dim=-1)
        elif value_shift_impl == "prealloc":
            half_kv_dim = attn.kv_dim // 2
            v = torch.empty((B, S, attn.kv_dim), device=v_curr.device, dtype=v_curr.dtype)
            v[..., :half_kv_dim] = v_curr
            v[..., half_kv_dim:] = 0
            v[:, 1:, half_kv_dim:] = v_prev[:, :-1, :]
        else:
            raise ValueError(f"Unknown value_shift_impl={value_shift_impl}")
    else:
        v = qkv[..., attn.latent_dim + attn.kv_dim:]

    if attn.use_qk_mean:
        q_pre = q
        k_pre = k

    if attn.use_convs:
        q = attn.q_conv(q)
        k = attn.k_conv(k)

    if attn.use_qk_mean and attn.n_groups == 1:
        qk_mean = 0.5 * (q_pre + k_pre)
        q = q + qk_mean
        k = k + qk_mean
    elif attn.use_qk_mean:
        q_mean = q_pre.view(B, S, attn.n_heads, attn.head_dim).mean(dim=2)
        k_mean = k_pre.view(B, S, attn.n_kv_heads, attn.head_dim).mean(dim=2)

        if qk_mean_impl == "old":
            q = q + (0.5 * k_mean.unsqueeze(2).expand(-1, -1, attn.n_heads, -1).reshape(B, S, attn.latent_dim))
            k = k + (0.5 * q_mean.unsqueeze(2).expand(-1, -1, attn.n_kv_heads, -1).reshape(B, S, attn.kv_dim))
        elif qk_mean_impl == "headspace":
            q = (
                q.view(B, S, attn.n_heads, attn.head_dim)
                + (0.5 * k_mean.unsqueeze(2))
            ).reshape(B, S, attn.latent_dim)
            k = (
                k.view(B, S, attn.n_kv_heads, attn.head_dim)
                + (0.5 * q_mean.unsqueeze(2))
            ).reshape(B, S, attn.kv_dim)
        else:
            raise ValueError(f"Unknown qk_mean_impl={qk_mean_impl}")

    q = q.view(B, S, attn.n_heads, attn.head_dim).transpose(1, 2)
    k = k.view(B, S, attn.n_kv_heads, attn.head_dim).transpose(1, 2)
    v = v.view(B, S, attn.n_kv_heads, attn.head_dim).transpose(1, 2)

    if attn.use_qk_norm:
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1) * attn.key_temperature

    if attn.use_rope:
        q = attn._apply_rope(q, S)
        k = attn._apply_rope(k, S)

    k = k.unsqueeze(2).expand(-1, -1, attn.n_groups, -1, -1).reshape(B, attn.n_heads, S, attn.head_dim)
    v = v.unsqueeze(2).expand(-1, -1, attn.n_groups, -1, -1).reshape(B, attn.n_heads, S, attn.head_dim)

    out = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=None,
        is_causal=True,
        scale=None if attn.use_qk_norm else attn.scale,
    )

    out = out.transpose(1, 2).contiguous().view(B, S, attn.latent_dim)
    return attn.o_proj(out) * attn.output_scale


def test_value_shift_prealloc_matches_padcat_cpu():
    torch.manual_seed(0)

    attn = CCGQAAttention(
        dim=64,
        n_heads=4,
        n_kv_heads=2,
        compression_factor=2,
        max_seq_len=128,
        use_rope=False,
        use_qk_norm=False,
        use_convs=False,
        use_qk_mean=False,
        use_value_shift=True,
    ).cpu()
    attn.eval()

    x = torch.randn(2, 32, 64, dtype=torch.float32)

    with torch.no_grad():
        out_current = attn(x)
        out_ref = _forward_reference(attn, x, value_shift_impl="padcat", qk_mean_impl="headspace")

    torch.testing.assert_close(out_current, out_ref, rtol=0, atol=0)


def test_qk_mean_headspace_matches_old_expand_cpu():
    torch.manual_seed(0)

    attn = CCGQAAttention(
        dim=64,
        n_heads=4,
        n_kv_heads=2,  # groups=2
        compression_factor=2,
        max_seq_len=128,
        use_rope=False,
        use_qk_norm=False,
        use_convs=False,
        use_qk_mean=True,
        use_value_shift=False,
    ).cpu()
    attn.eval()

    x = torch.randn(2, 32, 64, dtype=torch.float32)

    with torch.no_grad():
        out_current = attn(x)
        out_ref = _forward_reference(attn, x, value_shift_impl="prealloc", qk_mean_impl="old")

    torch.testing.assert_close(out_current, out_ref, rtol=0, atol=0)


def test_rope_fallback_matches_reference_math_cpu(monkeypatch):
    # Force fallback path irrespective of whether fused kernels exist.
    monkeypatch.setattr(ccgqa_mod.attention, "FUSED_KERNELS_AVAILABLE", False)
    monkeypatch.setattr(ccgqa_mod.attention, "fused_rope", None)

    torch.manual_seed(0)

    attn = CCGQAAttention(
        dim=64,
        n_heads=4,
        n_kv_heads=2,
        compression_factor=2,
        max_seq_len=128,
        use_rope=True,
        use_qk_norm=False,
        use_convs=False,
        use_qk_mean=False,
        use_value_shift=False,
    ).cpu()
    attn.eval()

    B, H, S, D = 2, attn.n_heads, 32, attn.head_dim
    x = torch.randn(B, H, S, D, dtype=torch.float32)

    cos = attn.cos_cached[:, :, :S, :]
    sin = attn.sin_cached[:, :, :S, :]

    with torch.no_grad():
        out = attn._apply_rope(x, S)

        x1, x2 = x[..., ::2], x[..., 1::2]
        ref = torch.empty_like(x)
        ref[..., ::2] = x1 * cos - x2 * sin
        ref[..., 1::2] = x1 * sin + x2 * cos

    torch.testing.assert_close(out, ref, rtol=0, atol=0)
