"""HYDRA Model - Lightning Attention 3 with MoD/MoR efficiency stack.

Main model classes:
- HydraModel: Full efficiency stack (LA3 attention + MoD + MoR)
- HydraBaseModel: Simple baseline (no routing, for comparison)

Legacy names (CCGQAMoDMoRModel, CCGQAModel) are aliased for backward compat.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as gradient_checkpoint

from hydra.layers import RMSNorm
from hydra.routing.loss_tracker import MovingAverageBaseline

_log = logging.getLogger(__name__)

from .blocks import HydraBlock, HydraMoRBlock

# Backward compat aliases
CCGQABlock = HydraBlock
CCGQAMoRBlock = HydraMoRBlock


class HydraBaseModel(nn.Module):
	"""Simple transformer model without routing (baseline for comparison)."""

	def __init__(
		self,
		vocab_size: int = 50257,
		dim: int = 1344,
		n_layers: int = 24,
		n_heads: int = 21,
		n_kv_heads: int = 3,
		compression_factor: int = 4,
		mlp_ratio: float = 2.67,
		max_seq_len: int = 8192,
		tie_weights: bool = True,
		**kwargs,
	):
		super().__init__()

		self.dim = dim
		self.vocab_size = vocab_size
		self.n_layers = n_layers

		self.tok_emb = nn.Embedding(vocab_size, dim)

		self.layers = nn.ModuleList(
			[
				HydraBlock(
					dim=dim,
					n_heads=n_heads,
					n_kv_heads=n_kv_heads,
					compression_factor=compression_factor,
					mlp_ratio=mlp_ratio,
					max_seq_len=max_seq_len,
				)
				for _ in range(n_layers)
			]
		)

		self.norm = RMSNorm(dim)
		self.output = nn.Linear(dim, vocab_size, bias=False)

		self._init_weights()
		if tie_weights:
			self.output.weight = self.tok_emb.weight

	def _init_weights(self):
		residual_scale = 1.0 / math.sqrt(2 * self.n_layers)

		for name, module in self.named_modules():
			if isinstance(module, nn.Linear):
				nn.init.normal_(module.weight, mean=0.0, std=0.02)
				if module.bias is not None:
					nn.init.zeros_(module.bias)
				if "o_proj" in name or "down" in name:
					module.weight.data *= residual_scale
			elif isinstance(module, nn.Embedding):
				# GPT-2 style: std = 0.02
				nn.init.normal_(module.weight, mean=0.0, std=0.02)
			elif isinstance(module, nn.Conv1d):
				nn.init.normal_(module.weight, mean=0.0, std=0.02)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# Token embedding with sqrt(dim) scaling (LLaMA style)
		h = self.tok_emb(x) * math.sqrt(self.dim)
		for layer in self.layers:
			h = layer(h)
		h = self.norm(h)
		return self.output(h)


# Backward compat alias
CCGQAModel = HydraBaseModel


class HydraModel(nn.Module):
	"""HYDRA Transformer with MoD + MoR.
	
	Architecture:
	- Attention backend: LA3 (default, O(n) linear) or CCGQA (compressed latent)
	- Mixture-of-Depths (MoD): Skip MLP for easy tokens
	- Mixture-of-Recursions (MoR): Variable-depth MLP processing
	
	This is the main production model for HYDRA.
	"""

	def __init__(
		self,
		vocab_size: int = 50257,
		dim: int = 1536,
		n_mor_blocks: int = 4,
		recursions_per_block: int = 6,
		n_heads: int = 24,
		n_kv_heads: int = 4,
		compression_factor: int = 4,
		mlp_ratio: float = 2.67,
		max_seq_len: int = 8192,
		mod_capacity: float = 0.5,
		aux_loss_weight: float = None,
		adaptive: bool = True,
		tie_weights: bool = True,
		hybrid_attention: bool = True,
		mod_loss_aware_weight: float = 0.0,
		dim_ref: int = 768,
		depth_alpha: float = 0.0,
		depth_scale_max: float = 2.0,
		attention_backend: str = "ccgqa",  # Only CCGQA is supported
		**kwargs,
	):
		super().__init__()

		self.dim = dim
		self.vocab_size = vocab_size
		self.n_mor_blocks = n_mor_blocks
		self.recursions_per_block = recursions_per_block
		self.effective_layers = n_mor_blocks * recursions_per_block
		self.n_heads = n_heads
		self.n_kv_heads = n_kv_heads
		self.compression_factor = compression_factor
		self.mlp_ratio = mlp_ratio
		self.max_seq_len = max_seq_len
		self.mod_capacity = mod_capacity
		self.mod_loss_aware_weight = float(mod_loss_aware_weight)
		self.adaptive = adaptive
		self.hybrid_attention = hybrid_attention
		self.attention_backend = attention_backend.lower() if attention_backend else "ccgqa"

		self.dim_ref = dim_ref
		self.depth_alpha = depth_alpha
		self.depth_scale_max = depth_scale_max

		if aux_loss_weight is None:
			depth_scale = max(1.0, self.effective_layers / 32)
			dim_scale = max(1.0, (dim / 768) ** 0.5)
			aux_loss_weight = 0.5 * depth_scale * dim_scale
		self.aux_loss_weight = aux_loss_weight

		self.register_buffer("_global_step", torch.zeros((), dtype=torch.int64), persistent=False)
		self.register_buffer("_zero_scalar", torch.tensor(0.0), persistent=False)

		self.tok_emb = nn.Embedding(vocab_size, dim)

		def _get_attention_type(block_idx: int) -> str:
			# CCGQA is the only supported attention backend for HydraModel
			# LA3 backend files are kept for reference but not used in training
			return "ccqa"

		attention_pattern = [_get_attention_type(i) for i in range(n_mor_blocks)]
		self._attention_pattern = attention_pattern

		if not hasattr(self, "_logged_attention_pattern"):
			import logging
			logger = logging.getLogger("HYDRA")
			logger.info(f"Attention: CCGQA ({n_mor_blocks} blocks)")
			self._logged_attention_pattern = True

		self.layers = nn.ModuleList()
		for i in range(n_mor_blocks):
			use_mod_mlp = (0 < i < n_mor_blocks - 1)
			mor_block = HydraMoRBlock(
				dim=dim,
				n_heads=n_heads,
				n_kv_heads=n_kv_heads,
				compression_factor=compression_factor,
				mlp_ratio=mlp_ratio,
				max_seq_len=max_seq_len,
				max_recursions=recursions_per_block,
				adaptive=adaptive,
				layer_idx=i,
				total_layers=n_mor_blocks,
				attention_type=attention_pattern[i],
				mod_mlp_capacity=mod_capacity if use_mod_mlp else None,
				mod_mlp_aux_weight=self.aux_loss_weight if use_mod_mlp else 0.0,
				mod_mlp_warmup=kwargs.get("mod_mlp_warmup", 100),
				mod_force_enable_step=kwargs.get("mod_force_enable_step", None),
				mod_enable_loss_threshold=kwargs.get("mod_enable_loss_threshold", None),
				mod_loss_aware_weight=self.mod_loss_aware_weight,
				mor_warmup=kwargs.get("mor_warmup", 1000),
				dim_ref=dim_ref,
				depth_alpha=depth_alpha,
				depth_scale_max=depth_scale_max,
			)
			self.layers.append(mor_block)

		self.norm = RMSNorm(dim)
		self.output = nn.Linear(dim, vocab_size, bias=False)

		self.loss_baseline = MovingAverageBaseline(decay=0.99, warmup_steps=1000)

		self._init_weights()
		if tie_weights:
			self.output.weight = self.tok_emb.weight

	def _init_weights(self):
		residual_scale = 1.0 / math.sqrt(2 * self.effective_layers)

		for name, module in self.named_modules():
			is_te_linear = (
				module.__class__.__module__.startswith("transformer_engine")
				and hasattr(module, "weight")
				and isinstance(getattr(module, "weight", None), torch.nn.Parameter)
				and getattr(module, "weight").ndim == 2
			)

			if isinstance(module, nn.Linear) or is_te_linear:
				is_router = "router" in name
				nn.init.normal_(module.weight, mean=0.0, std=0.02)
				bias = getattr(module, "bias", None)
				if bias is not None and not is_router:
					nn.init.zeros_(bias)
				if "o_proj" in name or "down" in name:
					module.weight.data *= residual_scale
			elif isinstance(module, nn.Embedding):
				# GPT-2 style: std = 0.02
				nn.init.normal_(module.weight, mean=0.0, std=0.02)
			elif isinstance(module, nn.Conv1d):
				nn.init.normal_(module.weight, mean=0.0, std=0.02)

	def resize_rope_cache(self, new_max_seq_len: int) -> None:
		resized_count = 0
		for layer in self.layers:
			if hasattr(layer, "attention"):
				attn = layer.attention
				if hasattr(attn, "_init_rope") and hasattr(attn, "cos_cached"):
					current_len = attn.cos_cached.shape[2]
					if current_len < new_max_seq_len:
						attn._init_rope(new_max_seq_len)
						device = next(attn.parameters()).device
						attn.cos_cached = attn.cos_cached.to(device)
						attn.sin_cached = attn.sin_cached.to(device)
						resized_count += 1

		if resized_count > 0:
			_log.debug(f"Resized RoPE cache to {new_max_seq_len} in {resized_count} attention modules")
		self.max_seq_len = new_max_seq_len

	_gradient_checkpointing: bool = False
	_checkpoint_every_n: int = 1

	def enable_gradient_checkpointing(self, every_n: int = 1) -> None:
		self._gradient_checkpointing = True
		self._checkpoint_every_n = max(1, every_n)

	def disable_gradient_checkpointing(self) -> None:
		self._gradient_checkpointing = False

	@property
	def is_gradient_checkpointing(self) -> bool:
		return self._gradient_checkpointing

	def forward(self, x: torch.Tensor, return_losses: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
		# Token embedding with sqrt(dim) scaling (LLaMA style)
		h = self.tok_emb(x) * math.sqrt(self.dim)

		if return_losses:
			if self._gradient_checkpointing and self.training:
				layer_results = []
				for i, layer in enumerate(self.layers):
					if i % self._checkpoint_every_n == 0:
						h, layer_losses = gradient_checkpoint(layer.forward_with_losses, h, use_reentrant=False)
					else:
						h, layer_losses = layer.forward_with_losses(h)
					layer_results.append(layer_losses)
			else:
				layer_results = []
				for layer in self.layers:
					h, layer_losses = layer.forward_with_losses(h)
					layer_results.append(layer_losses)

			aux_losses = [losses["aux_loss"] for losses in layer_results if "aux_loss" in losses]
			ponder_losses = [losses["ponder_loss"] for losses in layer_results if "ponder_loss" in losses]

			h = self.norm(h)
			logits = self.output(h)

			aux_loss = sum(aux_losses) if aux_losses else self._zero_scalar
			ponder_loss = sum(ponder_losses) if ponder_losses else self._zero_scalar
			return logits, {"aux_loss": aux_loss, "ponder_loss": ponder_loss}

		if self._gradient_checkpointing and self.training:
			for i, layer in enumerate(self.layers):
				if i % self._checkpoint_every_n == 0:
					h = gradient_checkpoint(layer, h, use_reentrant=False)
				else:
					h = layer(h)
		else:
			for i, layer in enumerate(self.layers):
				h = layer(h)

		h = self.norm(h)
		return self.output(h)

	def forward_hidden(self, x: torch.Tensor) -> torch.Tensor:
		# Token embedding with sqrt(dim) scaling (LLaMA style)
		h = self.tok_emb(x) * math.sqrt(self.dim)

		if self._gradient_checkpointing and self.training:
			for i, layer in enumerate(self.layers):
				if i % self._checkpoint_every_n == 0:
					h = gradient_checkpoint(layer, h, use_reentrant=False)
				else:
					h = layer(h)
		else:
			for layer in self.layers:
				h = layer(h)

		return self.norm(h)

	def forward_hidden_with_losses(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, dict]:
		"""Return post-norm hidden states + auxiliary losses (no logits).

		Memory-efficient alternative to forward(return_losses=True) that avoids
		materializing the full logits tensor. Use with chunked cross-entropy.

		Args:
			x: Input token ids [batch, seq_len]
			mask: Optional attention mask [batch, seq_len] (1=valid, 0=pad)

		Returns:
			Tuple of:
				- hidden: Hidden states after final norm [batch, seq_len, dim]
				- losses: Dict with 'aux_loss' and 'ponder_loss'
		"""
		# Token embedding with sqrt(dim) scaling (LLaMA style)
		h = self.tok_emb(x) * math.sqrt(self.dim)

		if self._gradient_checkpointing and self.training:
			layer_results = []
			for i, layer in enumerate(self.layers):
				if i % self._checkpoint_every_n == 0:
					h, layer_losses = gradient_checkpoint(layer.forward_with_losses, h, mask, use_reentrant=False)
				else:
					h, layer_losses = layer.forward_with_losses(h, mask)
				layer_results.append(layer_losses)
		else:
			layer_results = []
			for layer in self.layers:
				h, layer_losses = layer.forward_with_losses(h, mask)
				layer_results.append(layer_losses)

		aux_losses = [losses["aux_loss"] for losses in layer_results if "aux_loss" in losses]
		ponder_losses = [losses["ponder_loss"] for losses in layer_results if "ponder_loss" in losses]

		aux_loss = sum(aux_losses) if aux_losses else self._zero_scalar
		ponder_loss = sum(ponder_losses) if ponder_losses else self._zero_scalar
		return self.norm(h), {"aux_loss": aux_loss, "ponder_loss": ponder_loss}

	def get_aux_losses(self) -> dict:
		mod_aux_loss = self._zero_scalar
		mor_ponder_loss = self._zero_scalar
		for layer in self.layers:
			if hasattr(layer, "get_ponder_loss"):
				mor_ponder_loss = mor_ponder_loss + layer.get_ponder_loss()
			if hasattr(layer, "mod_mlp_wrapper") and layer.mod_mlp_wrapper is not None:
				mod_aux_loss = mod_aux_loss + layer.mod_mlp_wrapper.get_aux_loss()
		return {"mod_aux_loss": mod_aux_loss, "mor_ponder_loss": mor_ponder_loss, "total": mod_aux_loss + mor_ponder_loss}

	def update_loss_baseline(self, logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
		B, L, V = logits.shape
		logits_flat = logits.view(-1, V)
		targets_flat = targets.view(-1)

		with torch.no_grad():
			token_losses = F.cross_entropy(logits_flat, targets_flat, ignore_index=ignore_index, reduction="none").view(B, L)
			valid_mask = targets != ignore_index
			token_losses = token_losses * valid_mask.float()

		self.loss_baseline.update(token_losses[valid_mask])
		advantage = self.loss_baseline.compute_advantage(token_losses)
		self._last_advantage = advantage.detach()
		self._last_baseline = self.loss_baseline.baseline
		return advantage.mean()

	def compute_advantage_loss(self, logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
		B, L, V = logits.shape
		logits_flat = logits.view(-1, V)
		targets_flat = targets.view(-1)

		with torch.no_grad():
			token_losses = F.cross_entropy(logits_flat, targets_flat, ignore_index=ignore_index, reduction="none").view(B, L)
			valid_mask = targets != ignore_index
			token_losses = token_losses * valid_mask.float()
			token_losses_teacher = token_losses.masked_fill(~valid_mask, float("-inf"))

		self.loss_baseline.update(token_losses[valid_mask])

		total_advantage_loss = self._zero_scalar
		for layer in self.layers:
			if hasattr(layer, "compute_advantage_loss"):
				total_advantage_loss = total_advantage_loss + layer.compute_advantage_loss(token_losses, self.loss_baseline)

			mod_wrap = getattr(layer, "mod_mlp_wrapper", None)
			if mod_wrap is not None and hasattr(mod_wrap, "compute_loss_aware_loss"):
				total_advantage_loss = total_advantage_loss + mod_wrap.compute_loss_aware_loss(token_losses_teacher)

		self._last_token_losses = token_losses.detach()
		self._last_baseline_value = self.loss_baseline.baseline
		self._last_advantage_loss = total_advantage_loss.detach()
		return total_advantage_loss

	def compute_advantage_loss_from_token_losses(self, token_losses: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
		valid_mask = targets != ignore_index
		token_losses = token_losses * valid_mask.float()
		token_losses_teacher = token_losses.masked_fill(~valid_mask, float("-inf"))

		self.loss_baseline.update(token_losses[valid_mask])
		total_loss = self._zero_scalar
		for layer in self.layers:
			if hasattr(layer, "compute_advantage_loss"):
				total_loss = total_loss + layer.compute_advantage_loss(token_losses, self.loss_baseline)

			mod_wrap = getattr(layer, "mod_mlp_wrapper", None)
			if mod_wrap is not None and hasattr(mod_wrap, "compute_loss_aware_loss"):
				total_loss = total_loss + mod_wrap.compute_loss_aware_loss(token_losses_teacher)

		self._last_token_losses = token_losses.detach()
		self._last_baseline_value = self.loss_baseline.baseline
		self._last_advantage_loss = total_loss.detach()
		return total_loss

	def get_efficiency_losses(self) -> dict:
		aux = self.get_aux_losses()
		return {"ponder_loss": aux["mor_ponder_loss"], "aux_loss": aux["mod_aux_loss"]}

	def set_global_step(self, step: int):
		self._global_step.fill_(step)
		self._cached_global_step = step
		self._mor_adaptive_cached = step >= getattr(self, "_mor_enable_step", 0)
		for layer in self.layers:
			if hasattr(layer, "set_global_step"):
				layer.set_global_step(step)

	@torch.compiler.disable
	def update_mod_loss_ema(self, loss_ema: float) -> None:
		for layer in self.layers:
			mod_wrap = getattr(layer, "mod_mlp_wrapper", None)
			if mod_wrap is None:
				continue
			if hasattr(mod_wrap, "update_loss_ema"):
				mod_wrap.update_loss_ema(loss_ema)

	def set_mor_curriculum(self, enable_step: int, rampup_steps: int = 1000):
		self._mor_enable_step = enable_step
		self._mor_rampup_steps = rampup_steps
		for layer in self.layers:
			if hasattr(layer, "set_mor_enable_step"):
				layer.set_mor_enable_step(enable_step, rampup_steps)

	def trigger_mor_early(self, current_step: int, rampup_steps: int = 1000):
		"""Dynamically enable MoR at current step (called when loss threshold is met)."""
		self._mor_enable_step = current_step
		self._mor_rampup_steps = rampup_steps
		for layer in self.layers:
			if hasattr(layer, "set_mor_enable_step"):
				layer.set_mor_enable_step(current_step, rampup_steps)

	def trigger_mod_from_mor(self, current_step: int) -> None:
		"""Dynamically enable MoD at current step (called when MoR early_exit crosses threshold).
		
		This is the MoR-informed MoD triggering: MoD activates when MoR indicates
		a sufficient fraction of tokens are exiting early (low-complexity).
		"""
		for layer in self.layers:
			mod_wrap = getattr(layer, "mod_mlp_wrapper", None)
			if mod_wrap is not None:
				# Set warmup_steps to current_step so MoD enables immediately
				mod_wrap.warmup_steps = current_step
				mod_wrap._loss_unlocked = True  # Bypass loss gate
				# Force the enabled flag
				mod_wrap._mod_enabled = True

	def is_mor_adaptive_enabled(self) -> bool:
		return getattr(self, "_mor_adaptive_cached", False)

	def get_mor_status(self) -> dict:
		global_step = getattr(self, "_cached_global_step", 0)
		enable_step = getattr(self, "_mor_enable_step", 0)
		rampup_steps = getattr(self, "_mor_rampup_steps", 1000)

		if global_step < enable_step:
			phase = "fixed-depth"
			progress = global_step / max(1, enable_step)
		elif global_step < enable_step + rampup_steps:
			phase = "rampup"
			progress = (global_step - enable_step) / max(1, rampup_steps)
		else:
			phase = "full-adaptive"
			progress = 1.0

		return {
			"phase": phase,
			"global_step": global_step,
			"enable_step": enable_step,
			"rampup_steps": rampup_steps,
			"rampup_progress": progress,
			"mor_enabled": global_step >= enable_step,
		}

	@torch.compiler.disable
	def get_routing_stats(self) -> dict:
		mod_stats = []
		for i, layer in enumerate(self.layers):
			if isinstance(layer, HydraMoRBlock) and layer.mod_mlp_wrapper is not None:
				mod_stats.append({"layer": i, **layer.mod_mlp_wrapper.get_routing_stats()})

		mor_stats = []
		for i, layer in enumerate(self.layers):
			if isinstance(layer, HydraMoRBlock):
				mor_stats.append({"layer": i, **layer.get_routing_stats()})

		summary = {}
		if mod_stats:
			probs = [s.get("probs_mean", 0) for s in mod_stats]
			summary["mod_probs_mean"] = sum(probs) / len(probs) if probs else 0
		if mor_stats:
			avg_depths = [s.get("avg_depth", 0) for s in mor_stats if "avg_depth" in s]
			if avg_depths:
				summary["mor_avg_depth"] = sum(avg_depths) / len(avg_depths)

		return {"mod_layers": mod_stats, "mor_layers": mor_stats, "summary": summary}


# Backward compatibility alias
CCGQAMoDMoRModel = HydraModel


__all__ = [
	"HydraModel",
	"HydraBaseModel",
	# Backward compat
	"CCGQAModel",
	"CCGQAMoDMoRModel",
]
