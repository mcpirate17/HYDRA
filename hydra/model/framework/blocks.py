"""Block-level components for HYDRA models.

Contains:
- HydraBlock: Simple transformer block (baseline)
- HydraMoRBlock: Block with Mixture-of-Recursions
- MoDMLPWrapper: Mixture-of-Depths wrapper for MLP

Legacy names (CCGQABlock, CCGQAMoRBlock) are aliased for backward compat.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from hydra.attention.backends.ccgqa.attention import CCGQAAttention
from hydra.attention.factory import build_hybrid_attention_module
from hydra.layers import RMSNorm, SwiGLUMLPFused as SwiGLUMLP
from hydra.routing.loss_tracker import MovingAverageBaseline
from hydra.routing.mixture_of_depths import MoDRouter
from hydra.routing.mixture_of_recursions import (
    MoRConfig,
    MoRExecutor,
    MoRRouter,
    dim_to_depth_scale,
)


class HydraBlock(nn.Module):
    """Simple transformer block with configurable attention (pre-norm).
    
    Used in HydraBaseModel for baseline comparisons.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int = 16,
        n_kv_heads: int = 4,
        compression_factor: int = 4,
        mlp_ratio: float = 2.67,
        max_seq_len: int = 8192,
        norm_eps: float = 1e-6,
        **attention_kwargs,
    ):
        super().__init__()

        self.attention = CCGQAAttention(
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            compression_factor=compression_factor,
            max_seq_len=max_seq_len,
            **attention_kwargs,
        )

        self.norm1 = RMSNorm(dim, eps=norm_eps)
        self.norm2 = RMSNorm(dim, eps=norm_eps)

        hidden_dim = int(dim * mlp_ratio)
        hidden_dim = ((hidden_dim + 255) // 256) * 256
        self.mlp = SwiGLUMLP(dim, hidden_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = x + self.attention(self.norm1(x), mask=mask)
        return h + self.mlp(self.norm2(h))


class MoDMLPWrapper(nn.Module):
    """Mixture-of-Depths wrapper applied to the MLP sublayer only."""

    def __init__(
        self,
        mlp: nn.Module,
        dim: int,
        capacity_ratio: float = 0.5,
        aux_loss_weight: float = 0.01,
        warmup_steps: int = 100,
        force_enable_step: Optional[int] = None,
        max_seq_len: int = 2048,
        enable_loss_threshold: Optional[float] = None,
        loss_aware_weight: float = 0.0,
    ):
        super().__init__()
        self.mlp = mlp
        self.capacity_ratio = capacity_ratio
        self.aux_loss_weight = aux_loss_weight
        self.warmup_steps = warmup_steps
        self.force_enable_step = (
            int(force_enable_step)
            if (force_enable_step is not None and int(force_enable_step) > 0)
            else None
        )
        self.max_seq_len = max_seq_len
        self.loss_aware_weight = float(loss_aware_weight)

        self.enable_loss_threshold = (
            float(enable_loss_threshold)
            if (enable_loss_threshold is not None and enable_loss_threshold > 0)
            else None
        )
        self._loss_unlocked: bool = self.enable_loss_threshold is None

        self.register_buffer("_global_step", torch.zeros((), dtype=torch.int64), persistent=False)
        self.register_buffer("_zero_scalar", torch.tensor(0.0), persistent=False)
        self.register_buffer("_one_scalar", torch.tensor(1.0), persistent=False)
        self.register_buffer("_zero_int64", torch.zeros((), dtype=torch.int64), persistent=False)

        self.mod_router = MoDRouter(
            dim=dim,
            capacity_ratio=capacity_ratio,
            aux_loss_weight=aux_loss_weight,
            max_seq_len=max_seq_len,
        )
        assert isinstance(self.mod_router, MoDRouter), f"mod_router must be MoDRouter, got {type(self.mod_router)}"

        self._aux_loss: torch.Tensor = self._zero_scalar
        self._last_probs_mean_t: torch.Tensor = self._zero_scalar
        self._last_probs_std_t: torch.Tensor = self._zero_scalar
        self._routing_mode: str = "soft"
        self._tokens_processed: int = 0
        self._tokens_total: int = 0
        self._last_scores: Optional[torch.Tensor] = None

    def set_global_step(self, step: int):
        self._global_step.fill_(step)
        if self.force_enable_step is not None and step >= self.force_enable_step:
            self._loss_unlocked = True
        self._mod_enabled = (step >= self.warmup_steps) and self._loss_unlocked

    @torch.compiler.disable
    def update_loss_ema(self, loss_ema: float) -> None:
        if self._loss_unlocked:
            return
        if self.enable_loss_threshold is None:
            self._loss_unlocked = True
            return
        try:
            loss_val = float(loss_ema)
        except Exception:
            return
        if loss_val < self.enable_loss_threshold:
            self._loss_unlocked = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not getattr(self, "_mod_enabled", False):
            self._routing_mode = "disabled"
            B, L, _ = x.shape
            self._tokens_total = L
            self._tokens_processed = L
            scores = self.mod_router.forward_logits(x)
            self._last_scores = scores
            probs = torch.sigmoid(scores.clamp(-10.0, 10.0))
            self._last_probs_mean_t = probs.mean().detach()
            self._last_probs_std_t = probs.std().detach()

            if self.training and self.aux_loss_weight > 0:
                mean_prob = probs.mean()
                target_prob = self.capacity_ratio
                capacity_loss = (mean_prob - target_prob).pow(2)
                prob_variance = probs.var()
                expected_var = target_prob * (1 - target_prob) * 0.5
                collapse_loss = torch.exp(-prob_variance / max(expected_var, 0.01) * 5.0)
                self._aux_loss = self.aux_loss_weight * (capacity_loss + 0.5 * collapse_loss)
            else:
                self._aux_loss = self._zero_scalar
            return self.mlp(x)

        if not self.training:
            return self._forward_hard(x)

        return self._forward_hard_with_ste(x)

    def _forward_hard(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape

        self._tokens_total = L
        self._routing_mode = "hard"

        _, indices, scores = self.mod_router(x, return_scores=True)
        self._last_scores = scores
        k = indices.shape[1]

        probs = torch.sigmoid(scores.clamp(-10.0, 10.0))
        self._last_probs_mean_t = probs.mean().detach()
        self._last_probs_std_t = probs.std().detach()
        self._tokens_processed = k

        if self.aux_loss_weight > 0:
            self._aux_loss = self.mod_router.get_aux_loss()

        indices, _ = torch.sort(indices, dim=1)

        indices_exp = indices.unsqueeze(-1).expand(-1, -1, D)
        x_selected = torch.gather(x, 1, indices_exp)
        mlp_out_selected = self.mlp(x_selected)

        output = torch.zeros_like(x)
        output.scatter_(1, indices_exp, mlp_out_selected.to(output.dtype))
        return output

    def _forward_hard_with_ste(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        self._tokens_total = L
        self._routing_mode = "hard"

        _, indices, scores = self.mod_router(x, return_scores=True)
        self._last_scores = scores
        k = indices.shape[1]

        probs = torch.sigmoid(scores.clamp(-10.0, 10.0))
        self._last_probs_mean_t = probs.mean().detach()
        self._last_probs_std_t = probs.std().detach()
        self._tokens_processed = k

        indices, _ = torch.sort(indices, dim=1)
        indices_exp = indices.unsqueeze(-1).expand(-1, -1, D)
        x_selected = torch.gather(x, 1, indices_exp)
        assert x_selected.shape[1] == k, f"gather failed: expected {k} tokens, got {x_selected.shape[1]}"

        mlp_out_selected = self.mlp(x_selected)
        output = torch.zeros(B, L, D, dtype=mlp_out_selected.dtype, device=x.device)
        output.scatter_(1, indices_exp, mlp_out_selected)

        selected_probs = torch.gather(probs, 1, indices)
        ste_grad_path = (selected_probs.sum() - selected_probs.sum().detach()) * 0.0
        output = output + ste_grad_path

        if self.aux_loss_weight > 0:
            self._aux_loss = self.mod_router.get_aux_loss()
        return output

    def get_aux_loss(self) -> torch.Tensor:
        return self._aux_loss

    def compute_loss_aware_loss(self, token_losses: torch.Tensor) -> torch.Tensor:
        if self.loss_aware_weight <= 0:
            return self._zero_scalar
        scores = getattr(self, "_last_scores", None)
        if scores is None:
            return self._zero_scalar

        B, L = token_losses.shape
        k = int(max(1, min(L, int(L * float(self.capacity_ratio)))))

        valid_mask = torch.isfinite(token_losses)
        if not valid_mask.any():
            return self._zero_scalar

        with torch.no_grad():
            _, hard_idx = torch.topk(token_losses, k, dim=1)
            teacher = torch.zeros((B, L), device=token_losses.device, dtype=torch.float32)
            teacher.scatter_(1, hard_idx, 1.0)

            pos_weight_val = float(max(1.0, (L - k) / max(1, k)))
            pos_weight = torch.tensor(pos_weight_val, device=token_losses.device, dtype=torch.float32)

        scores_f = scores.float().clamp(-10.0, 10.0)
        per_tok = F.binary_cross_entropy_with_logits(
            scores_f,
            teacher,
            pos_weight=pos_weight,
            reduction="none",
        )
        loss = (per_tok * valid_mask.float()).sum() / valid_mask.float().sum().clamp_min(1.0)
        return loss * self.loss_aware_weight

    @torch.compiler.disable
    def get_routing_stats(self) -> dict:
        tokens_processed = getattr(self, "_tokens_processed", 0)
        tokens_total = getattr(self, "_tokens_total", 1)
        compute_ratio = tokens_processed / max(1, tokens_total)

        mean_t = getattr(self, "_last_probs_mean_t", None)
        std_t = getattr(self, "_last_probs_std_t", None)
        return {
            "probs_mean": float(mean_t.item()) if mean_t is not None else 0.0,
            "probs_std": float(std_t.item()) if std_t is not None else 0.0,
            "target_capacity": self.capacity_ratio,
            "tokens_processed": tokens_processed,
            "tokens_total": tokens_total,
            "compute_ratio": compute_ratio,
            "compute_savings_pct": (1.0 - compute_ratio) * 100.0,
            "routing_mode": getattr(self, "_routing_mode", "unknown"),
            "global_step": int(self._global_step),
            "warmup_steps": self.warmup_steps,
            "force_enable_step": self.force_enable_step,
            "enable_loss_threshold": self.enable_loss_threshold,
            "loss_unlocked": bool(getattr(self, "_loss_unlocked", True)),
        }


class HydraBlockWithMoDMLP(nn.Module):
    """HYDRA block with MoD applied only to the MLP sublayer."""

    def __init__(
        self,
        dim: int,
        n_heads: int = 16,
        n_kv_heads: int = 4,
        compression_factor: int = 4,
        mlp_ratio: float = 2.67,
        max_seq_len: int = 8192,
        norm_eps: float = 1e-6,
        mod_capacity_ratio: float = 0.5,
        mod_aux_loss_weight: float = 0.01,
        mod_warmup_steps: int = 100,
        mod_loss_aware_weight: float = 0.0,
        **attention_kwargs,
    ):
        super().__init__()

        self.attention = CCGQAAttention(
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            compression_factor=compression_factor,
            max_seq_len=max_seq_len,
            **attention_kwargs,
        )

        self.norm1 = RMSNorm(dim, eps=norm_eps)
        self.norm2 = RMSNorm(dim, eps=norm_eps)

        hidden_dim = int(dim * mlp_ratio)
        hidden_dim = ((hidden_dim + 255) // 256) * 256
        base_mlp = SwiGLUMLP(dim, hidden_dim)
        self.mod_mlp = MoDMLPWrapper(
            mlp=base_mlp,
            dim=dim,
            capacity_ratio=mod_capacity_ratio,
            aux_loss_weight=mod_aux_loss_weight,
            warmup_steps=mod_warmup_steps,
            max_seq_len=max_seq_len,
            loss_aware_weight=mod_loss_aware_weight,
        )

    def set_global_step(self, step: int):
        self.mod_mlp.set_global_step(step)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = x + self.attention(self.norm1(x), mask=mask)
        return h + self.mod_mlp(self.norm2(h))

    def forward_with_losses(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, dict]:
        output = self.forward(x, mask=mask)
        losses = {}
        if self.training:
            losses["aux_loss"] = self.mod_mlp.get_aux_loss()
        return output, losses

    def get_aux_loss(self) -> torch.Tensor:
        return self.mod_mlp.get_aux_loss()

    @torch.compiler.disable
    def get_routing_stats(self) -> dict:
        return self.mod_mlp.get_routing_stats()


class HydraMoRBlock(nn.Module):
    """HYDRA block with Mixture-of-Recursions (MoR) on MLP.
    
    Architecture:
    - Attention (LA3 or fallback) runs once per forward
    - MLP can run multiple recursions per token (MoR)
    - Optional MoD on MLP sublayer (skips easy tokens)
    """

    def __init__(
        self,
        dim: int,
        n_heads: int = 16,
        n_kv_heads: int = 4,
        compression_factor: int = 4,
        mlp_ratio: float = 2.67,
        max_seq_len: int = 8192,
        max_recursions: int = 6,
        adaptive: bool = True,
        halt_threshold: float = 0.9,
        ponder_loss_weight: float = 0.01,
        layer_idx: int = 0,
        total_layers: int = 1,
        attention_type: str = "lla3",
        dim_ref: int = 768,
        depth_alpha: float = 0.0,
        depth_scale_max: float = 2.0,
        **attention_kwargs,
    ):
        super().__init__()

        self.max_recursions = max_recursions
        self.adaptive = adaptive
        self.layer_idx = layer_idx
        self.total_layers = total_layers
        self.dim = dim
        self.attention_type = attention_type

        self._depth_scale = dim_to_depth_scale(dim, dim_ref, depth_alpha, depth_scale_max)

        mod_mlp_capacity = attention_kwargs.pop("mod_mlp_capacity", None)
        mod_mlp_aux_weight = attention_kwargs.pop("mod_mlp_aux_weight", 0.01)
        mod_mlp_warmup = attention_kwargs.pop("mod_mlp_warmup", 100)
        mod_force_enable_step = attention_kwargs.pop("mod_force_enable_step", None)
        mod_enable_loss_threshold = attention_kwargs.pop("mod_enable_loss_threshold", None)
        mod_loss_aware_weight = attention_kwargs.pop("mod_loss_aware_weight", 0.0)
        mor_warmup = attention_kwargs.pop("mor_warmup", 1000)
        self.use_mod_mlp = mod_mlp_capacity is not None and mod_mlp_capacity > 0

        self.attention = build_hybrid_attention_module(
            attention_type,
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            max_seq_len=max_seq_len,
            compression_factor=compression_factor,
            attention_kwargs=attention_kwargs,
        )
        self.residual_scale = 0.5

        use_ada_norm = os.environ.get("HYDRA_USE_ADA_RMSNORM", "1") == "1"
        if use_ada_norm:
            from hydra.layers import AdaRMSNorm as _Norm
        else:
            from hydra.layers import RMSNorm as _Norm

        self.norm1 = _Norm(dim)

        hidden_dim = int(dim * mlp_ratio)
        hidden_dim = ((hidden_dim + 255) // 256) * 256
        self.mlp = SwiGLUMLP(dim, hidden_dim)
        self.norm2 = _Norm(dim)

        if self.use_mod_mlp:
            self.mod_mlp_wrapper = MoDMLPWrapper(
                mlp=self.mlp,
                dim=dim,
                capacity_ratio=mod_mlp_capacity,
                aux_loss_weight=mod_mlp_aux_weight,
                warmup_steps=mod_mlp_warmup,
                force_enable_step=mod_force_enable_step,
                max_seq_len=max_seq_len,
                enable_loss_threshold=mod_enable_loss_threshold,
                loss_aware_weight=mod_loss_aware_weight,
            )
        else:
            self.mod_mlp_wrapper = None

        self.block = None

        mor_config = MoRConfig(
            dim=dim,
            n_recursions=max_recursions,
            ponder_loss_weight=ponder_loss_weight,
            warmup_steps=mor_warmup,
            layer_idx=layer_idx,
            total_layers=total_layers,
            dim_ref=dim_ref,
            depth_alpha=depth_alpha,
            depth_scale_max=depth_scale_max,
            advantage_loss_scale=attention_kwargs.pop("mor_advantage_loss_scale", 0.1),
        )
        self.mor_router = MoRRouter(mor_config)
        self.mor_executor = MoRExecutor(mor_config)

        self._mor_config = mor_config
        self.ponder_loss_weight = ponder_loss_weight

        self.register_buffer("_global_step", torch.zeros((), dtype=torch.int64), persistent=False)
        self.register_buffer("_zero_scalar", torch.tensor(0.0), persistent=False)
        self.register_buffer("_one_scalar", torch.tensor(1.0), persistent=False)
        self.register_buffer("_zero_int64", torch.zeros((), dtype=torch.int64), persistent=False)

        # Optional diagnostics: attention output RMS (scalar tensor).
        # Enabled via env to avoid overhead in normal training.
        self._enable_attn_out_rms = os.environ.get("HYDRA_ENABLE_ATTN_OUT_RMS", "0") == "1"
        self._last_attn_out_rms_t: torch.Tensor = self._zero_scalar
        self._last_attn_out_finite_frac_t: torch.Tensor = self._one_scalar
        self._last_attn_out_nan_ct_t: torch.Tensor = self._zero_int64
        self._last_attn_out_inf_ct_t: torch.Tensor = self._zero_int64
        self._last_attn_out_dtype: Optional[str] = None
        self._last_attn_out_shape: Optional[Tuple[int, ...]] = None

        self._warmup_steps = 2500
        self._mor_enable_step = 0
        self._mor_rampup_steps = 0

        self.final_norm = RMSNorm(dim)
        self._ponder_loss: torch.Tensor = self._zero_scalar
        self._avg_ponder_time: torch.Tensor = self._zero_scalar

        self._last_target_depths: Optional[torch.Tensor] = None
        self._last_router_probs_mean: float = 0.0
        self._last_router_probs_std: float = 0.0

    @torch.compiler.disable
    def _update_attn_out_diagnostics(self, attn_out: torch.Tensor) -> None:
        a = attn_out.detach()
        a_f = a.float()
        self._last_attn_out_rms_t = a_f.pow(2).mean().sqrt().detach()

        numel = a.numel()
        if numel == 0:
            self._last_attn_out_finite_frac_t = self._one_scalar.to(device=a.device)
            self._last_attn_out_nan_ct_t = self._zero_int64.to(device=a.device)
            self._last_attn_out_inf_ct_t = self._zero_int64.to(device=a.device)
        else:
            finite = torch.isfinite(a)
            finite_ct = finite.sum()
            self._last_attn_out_finite_frac_t = (finite_ct.float() / float(numel)).detach()
            self._last_attn_out_nan_ct_t = torch.isnan(a).sum().to(dtype=torch.int64).detach()
            self._last_attn_out_inf_ct_t = torch.isinf(a).sum().to(dtype=torch.int64).detach()

        self._last_attn_out_dtype = str(a.dtype)
        self._last_attn_out_shape = tuple(a.shape)

    def _forward_fixed(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.residual_scale * self.attention(self.norm1(x))
        if self._enable_attn_out_rms:
            self._update_attn_out_diagnostics(attn_out)
        h = x + attn_out
        for i in range(self.max_recursions):
            rec_bias = self.mor_executor.recursion_bias[i].squeeze()
            rec_embed = self.mor_executor.recursion_embed(self.mor_executor._recursion_indices[i : i + 1]).squeeze()
            h_with_rec = h + rec_bias + rec_embed
            if self.mod_mlp_wrapper is not None:
                h = h + self.mod_mlp_wrapper(self.norm2(h_with_rec))
            else:
                h = h + self.mlp(self.norm2(h_with_rec))
        return self.final_norm(h)

    def _forward_mor(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_out = self.residual_scale * self.attention(self.norm1(x))
        if self._enable_attn_out_rms:
            self._update_attn_out_diagnostics(attn_out)
        h = x + attn_out
        depths, probs, logits = self.mor_router(h)

        self._last_target_depths = depths.detach().float()
        self._last_router_probs_tensor = probs.detach()
        self._last_router_logits = logits
        self._last_depths = depths
        self._last_probs = probs

        mlp = self.mod_mlp_wrapper if self.mod_mlp_wrapper is not None else self.mlp
        output = self.mor_executor(h, depths, probs, mlp, self.norm2)

        ponder_loss = self.mor_router.compute_ponder_loss(depths, probs, logits, token_losses=None, baseline=None)

        if self.training:
            n_rec = self.max_recursions
            depth_continuous = probs * (n_rec - 1)
            self._ponder_loss = ponder_loss.detach()
            self._last_avg_depth = depth_continuous.mean().detach()

        return self.final_norm(output), ponder_loss

    def compute_advantage_loss(self, token_losses: torch.Tensor, baseline: "MovingAverageBaseline") -> torch.Tensor:
        if not hasattr(self, "_last_probs") or self._last_probs is None:
            return self._zero_scalar

        probs = self._last_probs
        n_rec = self.max_recursions
        depth_continuous = probs * (n_rec - 1)
        advantage = baseline.compute_advantage(token_losses)
        scale = self._mor_config.advantage_loss_scale
        return -(advantage * depth_continuous).mean() * scale

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.adaptive:
            rampup_scale = self.get_mor_rampup_scale()
            if rampup_scale <= 0.0:
                return self._forward_fixed(x)

            adaptive_output, _ = self._forward_mor(x)
            if rampup_scale >= 1.0:
                return adaptive_output

            fixed_output = self._forward_fixed(x)
            return rampup_scale * adaptive_output + (1.0 - rampup_scale) * fixed_output
        return self._forward_fixed(x)

    def forward_with_losses(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, dict]:
        if self.adaptive:
            rampup_scale = self.get_mor_rampup_scale()
            if rampup_scale <= 0.0:
                output = self._forward_fixed(x)
                aux_loss = self.mod_mlp_wrapper.get_aux_loss() if self.mod_mlp_wrapper is not None else self._zero_scalar
                zero_loss = self._zero_scalar
                if self.training:
                    self._ponder_loss = zero_loss.detach()
                return output, {"ponder_loss": zero_loss, "aux_loss": aux_loss}

            adaptive_output, ponder_loss = self._forward_mor(x)
            ponder_loss = ponder_loss * rampup_scale
            aux_loss = self.mod_mlp_wrapper.get_aux_loss() if self.mod_mlp_wrapper is not None else self._zero_scalar

            if rampup_scale >= 1.0:
                if self.training:
                    self._ponder_loss = ponder_loss.detach()
                return adaptive_output, {"ponder_loss": ponder_loss, "aux_loss": aux_loss}

            fixed_output = self._forward_fixed(x)
            output = rampup_scale * adaptive_output + (1.0 - rampup_scale) * fixed_output
            if self.training:
                self._ponder_loss = ponder_loss.detach()
            return output, {"ponder_loss": ponder_loss, "aux_loss": aux_loss}

        output = self._forward_fixed(x)
        aux_loss = self.mod_mlp_wrapper.get_aux_loss() if self.mod_mlp_wrapper is not None else self._zero_scalar
        zero_loss = self._zero_scalar
        if self.training:
            self._ponder_loss = zero_loss.detach()
        return output, {"ponder_loss": zero_loss, "aux_loss": aux_loss}

    def get_ponder_loss(self) -> torch.Tensor:
        return self._ponder_loss

    @torch.compiler.disable
    def get_routing_stats(self) -> dict:
        router_probs_tensor = getattr(self, "_last_router_probs_tensor", None)
        if router_probs_tensor is not None:
            self._last_router_probs_mean = router_probs_tensor.mean().item()
            self._last_router_probs_std = router_probs_tensor.std().item()

        stats = {
            "router_probs_mean": getattr(self, "_last_router_probs_mean", 0.0),
            "router_probs_std": getattr(self, "_last_router_probs_std", 0.0),
            "layer_idx": self.layer_idx,
            "total_layers": self.total_layers,
            "target_depth_ratio": getattr(self, "target_depth_ratio", 0.6),
            "expected_avg_depth": getattr(self, "target_depth_ratio", 0.6) * (self.max_recursions - 1),
        }

        if self._last_target_depths is not None:
            depths = self._last_target_depths.flatten().float()
            stats["avg_depth"] = depths.mean().item()
            stats["depth_std"] = depths.std().item()
            hist = torch.histc(depths, bins=self.max_recursions, min=0, max=self.max_recursions - 1)
            stats["depth_histogram"] = hist.tolist()

        if hasattr(self, "_recursion_tokens_processed"):
            tokens_per_recursion = self._recursion_tokens_processed
            total_tokens = sum(int(t) if hasattr(t, "item") else t for t in tokens_per_recursion)
            max_possible = len(tokens_per_recursion) * (
                self._last_target_depths.numel() if self._last_target_depths is not None else 0
            )
            if max_possible > 0:
                stats["compute_ratio"] = total_tokens / max_possible
                stats["compute_savings_pct"] = (1.0 - total_tokens / max_possible) * 100
            stats["tokens_per_recursion"] = [int(t) if hasattr(t, "item") else t for t in tokens_per_recursion]

        return stats

    def set_global_step(self, step: int):
        self._global_step.fill_(step)
        self._cached_global_step = step
        self._mor_adaptive_cached = step >= self._mor_enable_step
        if step < self._mor_enable_step:
            self._mor_rampup_scale_cached = 0.0
        elif self._mor_rampup_steps <= 0:
            self._mor_rampup_scale_cached = 1.0
        else:
            steps_since_enable = step - self._mor_enable_step
            raw_scale = min(1.0, steps_since_enable / self._mor_rampup_steps)
            self._mor_rampup_scale_cached = round(raw_scale * 10) / 10
        if self.mod_mlp_wrapper is not None:
            self.mod_mlp_wrapper.set_global_step(step)

    def set_mor_enable_step(self, enable_step: int, rampup_steps: int = 1000):
        self._mor_enable_step = enable_step
        self._mor_rampup_steps = rampup_steps

    def is_mor_adaptive_enabled(self) -> bool:
        return getattr(self, "_mor_adaptive_cached", False)

    def get_mor_rampup_scale(self) -> float:
        return getattr(self, "_mor_rampup_scale_cached", 1.0)


class HydraMoDBlockWrapper(nn.Module):
    """Add MoD gather/scatter routing around an arbitrary block."""

    def __init__(
        self,
        block: nn.Module,
        dim: int,
        capacity_ratio: float = 0.5,
        aux_loss_weight: float = 0.01,
        warmup_steps: int = 100,
    ):
        super().__init__()
        self.block = block
        self.capacity_ratio = capacity_ratio
        self.aux_loss_weight = aux_loss_weight
        self.warmup_steps = warmup_steps
        self.register_buffer("_global_step", torch.zeros((), dtype=torch.int64), persistent=False)
        self.register_buffer("_zero_scalar", torch.tensor(0.0), persistent=False)
        self._last_probs_mean_t: torch.Tensor = self._zero_scalar
        self._last_probs_std_t: torch.Tensor = self._zero_scalar

        self.mod_router = MoDRouter(dim=dim, capacity_ratio=capacity_ratio, aux_loss_weight=aux_loss_weight)
        assert isinstance(self.mod_router, MoDRouter), f"mod_router must be MoDRouter, got {type(self.mod_router)}"

        self._aux_loss: torch.Tensor = self._zero_scalar
        self._last_probs_mean: float = 0.0
        self._last_probs_std: float = 0.0
        self._routing_mode: str = "soft"

    def set_global_step(self, step: int):
        self._global_step.fill_(step)
        self._use_hard_routing = step >= self.warmup_steps

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.training:
            output, _ = self.forward_with_losses(x, **kwargs)
            return output

        B, L, D = x.shape
        _, indices, _ = self.mod_router(x)
        k = indices.shape[1]
        if k == L:
            return self.block(x, **kwargs)
        if k == 0:
            return torch.zeros_like(x)

        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, D)
        x_selected = torch.gather(x, 1, indices_expanded)
        out_selected = self.block(x_selected, **kwargs)
        output = torch.zeros_like(x)
        output.scatter_(1, indices_expanded, out_selected.to(output.dtype))
        return output

    def forward_with_losses(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, dict]:
        B, L, D = x.shape
        _, indices, scores = self.mod_router(x, return_scores=True)
        k = indices.shape[1]
        probs = torch.sigmoid(scores)

        use_hard_routing = getattr(self, "_use_hard_routing", False)
        self._routing_mode = "hard" if use_hard_routing else "soft"
        self._tokens_total = L

        if k >= L:
            if hasattr(self.block, "forward_with_losses"):
                block_out, inner_losses = self.block.forward_with_losses(x, **kwargs)
            else:
                block_out = self.block(x, **kwargs)
                inner_losses = {}
            output = block_out
            self._tokens_processed = L
        elif use_hard_routing:
            indices_expanded = indices.unsqueeze(-1).expand(-1, -1, D)
            x_selected = torch.gather(x, 1, indices_expanded)
            assert x_selected.shape[1] == k, f"gather failed: expected {k} tokens, got {x_selected.shape[1]}"
            assert x_selected.shape[1] < L, f"No compute savings: k={k} >= L={L}"

            if hasattr(self.block, "forward_with_losses"):
                block_out_selected, inner_losses = self.block.forward_with_losses(x_selected, **kwargs)
            else:
                block_out_selected = self.block(x_selected, **kwargs)
                inner_losses = {}

            output = torch.zeros_like(x)
            output.scatter_(1, indices_expanded, block_out_selected.to(output.dtype))

            selected_probs = torch.gather(probs, 1, indices)
            ste_grad_path = (selected_probs.sum() - selected_probs.sum().detach()) * 0.0
            output = output + ste_grad_path
            self._tokens_processed = k
        else:
            if hasattr(self.block, "forward_with_losses"):
                block_out, inner_losses = self.block.forward_with_losses(x, **kwargs)
            else:
                block_out = self.block(x, **kwargs)
                inner_losses = {}
            gate = probs.unsqueeze(-1)
            output = gate * block_out
            self._tokens_processed = L

        if self.training and self.aux_loss_weight > 0:
            aux_loss = self.mod_router.get_aux_loss()
            self._aux_loss = aux_loss.detach()
            self._last_probs_mean_t = probs.mean().detach()
            self._last_probs_std_t = probs.std().detach()
        else:
            aux_loss = self._zero_scalar

        inner_losses["aux_loss"] = aux_loss
        return output, inner_losses

    def get_aux_loss(self) -> torch.Tensor:
        return self.mod_router.get_aux_loss()

    @torch.compiler.disable
    def get_routing_stats(self) -> dict:
        tokens_processed = getattr(self, "_tokens_processed", 0)
        tokens_total = getattr(self, "_tokens_total", 1)
        mean_t = getattr(self, "_last_probs_mean_t", None)
        std_t = getattr(self, "_last_probs_std_t", None)
        compute_ratio = tokens_processed / max(1, tokens_total)
        return {
            "probs_mean": float(mean_t.item()) if mean_t is not None else 0.0,
            "probs_std": float(std_t.item()) if std_t is not None else 0.0,
            "target_capacity": self.capacity_ratio,
            "tokens_processed": tokens_processed,
            "tokens_total": tokens_total,
            "compute_ratio": compute_ratio,
            "compute_savings_pct": (1.0 - compute_ratio) * 100,
            "routing_mode": getattr(self, "_routing_mode", "unknown"),
            "global_step": int(self._global_step),
            "warmup_steps": self.warmup_steps,
        }


# Backward compatibility aliases
CCGQABlock = HydraBlock
CCGQAMoRBlock = HydraMoRBlock
CCGQABlockWithMoDMLP = HydraBlockWithMoDMLP
CCGQAMoDBlockWrapper = HydraMoDBlockWrapper


__all__ = [
    # New names
    "HydraBlock",
    "HydraBlockWithMoDMLP",
    "HydraMoRBlock",
    "HydraMoDBlockWrapper",
    "MoDMLPWrapper",
    # Backward compat
    "CCGQABlock",
    "CCGQABlockWithMoDMLP",
    "CCGQAMoRBlock",
    "CCGQAMoDBlockWrapper",
]
