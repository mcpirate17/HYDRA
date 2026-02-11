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

# Use HYDRA logger for consistent output with trainer
_log = logging.getLogger("HYDRA")

# Track if we've already warned about OOB tokens (avoid log spam)
_OOB_TOKEN_WARNED = False


@torch.compiler.disable
def _log_oob_warning(x: torch.Tensor, vocab_size: int) -> None:
	"""Log OOB token warning (outside compiled region).
	
	Skips entirely during CUDA graph capture to avoid stream invalidation.
	"""
	global _OOB_TOKEN_WARNED
	if _OOB_TOKEN_WARNED:
		return
	
	# Check if we're inside CUDA graph capture - skip host sync operations
	if torch.cuda.is_current_stream_capturing():
		return
	
	oob_mask = (x < 0) | (x >= vocab_size)
	if oob_mask.any():
		n_oob = int(oob_mask.sum().item())
		min_val = int(x.min().item())
		max_val = int(x.max().item())
		_log.warning(
			f"OOB tokens detected in embedding input! "
			f"Found {n_oob} tokens outside [0, {vocab_size}). "
			f"Range: [{min_val}, {max_val}]. Clamping to valid range."
		)
		_OOB_TOKEN_WARNED = True


def _safe_embedding_lookup(embedding: nn.Embedding, x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
	"""Embedding lookup with bounds validation to prevent Xid 13 GPU crashes.
	
	torch.compile compatible - clamping is always applied (no-op for valid tokens).
	Warning logging is done outside the compiled region.
	
	Args:
		embedding: nn.Embedding module
		x: Token IDs tensor [batch, seq_len]
		scale: Optional scaling factor (e.g., sqrt(dim) for LLaMA-style)
	
	Returns:
		Embedded tensor [batch, seq_len, dim]
	"""
	vocab_size = embedding.num_embeddings
	
	# Log warning if OOB tokens detected (outside compiled region)
	_log_oob_warning(x, vocab_size)
	
	# Always clamp - this is a no-op for valid tokens but prevents GPU crash for OOB
	x = x.clamp(0, vocab_size - 1)
	
	return embedding(x) * scale

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
		self._sqrt_dim = math.sqrt(dim)
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

	def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
		"""Forward pass.
		
		Args:
			x: Input token ids [batch, seq_len]
			mask: Optional attention mask [batch, seq_len] (unused, for API compat)
		
		Returns:
			Logits tensor [batch, seq_len, vocab_size]
		"""
		# Token embedding with sqrt(dim) scaling (LLaMA style)
		# Uses safe lookup to prevent Xid 13 GPU crashes from OOB tokens
		h = _safe_embedding_lookup(self.tok_emb, x, scale=self._sqrt_dim)
		for layer in self.layers:
			h = layer(h)
		h = self.norm(h)
		return self.output(h)


# Backward compat alias
CCGQAModel = HydraBaseModel


class HydraModel(nn.Module):
	"""HYDRA Transformer with MoD + MoR + optional MoE.
	
	Architecture:
	- Attention backend: LA3 (default, O(n) linear) or CCGQA (compressed latent)
	- Mixture-of-Depths (MoD): Skip MLP for easy tokens
	- Mixture-of-Recursions (MoR): Variable-depth MLP processing
	- Mixture-of-Experts (MoE): Optional sparse FFN routing (additive)
	
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
		# MoR configuration
		mor_min_depth: int = 0,  # Minimum recursion depth (0=allow immediate exit, 1+=force iterations)
		# Static routing mode for CUDA graph compatibility
		static_routing_mode: bool = False,
		# MoE configuration
		moe_enabled: bool = False,
		moe_num_experts: int = 4,
		moe_num_layers: int = 2,
		moe_top_k: int = 1,
		moe_aux_weight: float = 0.01,
		moe_warmup_steps: int = 1000,
		moe_capacity_factor: float = float("inf"),
		moe_router_jitter: float = 0.0,
		moe_expert_diversity_noise: float = 0.0,
		moe_identity_init: bool = True,
		moe_forced_routing_steps: int = 0,  # Steps to force position-based routing
		moe_teacher_until_step: int = 0,
		# Manifold connections configuration
		manifold_enabled: bool = False,
		manifold_type: str = "sphere",
		manifold_n_components: int = 8,
		manifold_warmup_steps: int = 1000,
		manifold_placement_interval: int = 2,
		manifold_curvature: float = 1.0,
		**kwargs,
	):
		super().__init__()

		self.dim = dim
		self._sqrt_dim = math.sqrt(dim)
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

		# MoE configuration
		self.moe_enabled = moe_enabled
		self.moe_num_experts = moe_num_experts
		self.moe_num_layers = moe_num_layers
		self.moe_top_k = moe_top_k
		self.moe_aux_weight = moe_aux_weight
		self.moe_warmup_steps = moe_warmup_steps
		self.moe_capacity_factor = moe_capacity_factor
		self.moe_router_jitter = moe_router_jitter
		self.moe_expert_diversity_noise = moe_expert_diversity_noise
		self.moe_identity_init = moe_identity_init
		self.moe_forced_routing_steps = moe_forced_routing_steps
		self.moe_teacher_until_step = moe_teacher_until_step
		self.mor_min_depth = mor_min_depth
		self.static_routing_mode = static_routing_mode

		# Manifold connections configuration
		self.manifold_enabled = manifold_enabled
		self.manifold_type = manifold_type
		self.manifold_n_components = manifold_n_components
		self.manifold_warmup_steps = manifold_warmup_steps
		self.manifold_placement_interval = manifold_placement_interval
		self.manifold_curvature = manifold_curvature

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
				mor_min_depth=mor_min_depth,
				static_routing_mode=static_routing_mode,
			)
			self.layers.append(mor_block)

		# MoE layers (optional, additive)
		self.moe_layers = nn.ModuleList()
		self._moe_placement = ()
		if self.moe_enabled and self.moe_num_layers > 0:
			from hydra.routing import MoEFFNBlock, compute_moe_placement
			
			# Compute deterministic MoE placement
			self._moe_placement = compute_moe_placement(n_mor_blocks, self.moe_num_layers)
			
			_log.info(f"MoE: {self.moe_num_experts} experts, {len(self._moe_placement)} layers at positions {self._moe_placement}")
			
			# Create MoE blocks
			for moe_idx in range(len(self._moe_placement)):
				moe_block = MoEFFNBlock(
					dim=dim,
					hidden_dim=None,  # Auto-compute from mlp_ratio
					num_experts=self.moe_num_experts,
					top_k=self.moe_top_k,
					aux_loss_weight=self.moe_aux_weight,
					router_jitter=self.moe_router_jitter,
					expert_diversity_noise=self.moe_expert_diversity_noise,
					capacity_factor=self.moe_capacity_factor,
					residual_scale=1.0,
					identity_init=self.moe_identity_init,
					warmup_steps=self.moe_warmup_steps,
					mlp_ratio=mlp_ratio,
					forced_routing_steps=self.moe_forced_routing_steps,
					teacher_until_step=self.moe_teacher_until_step,
				)
				self.moe_layers.append(moe_block)

		# Manifold connections (optional, for gradient stabilization)
		self.manifold_connections = nn.ModuleList()
		self._manifold_placement = ()
		if self.manifold_enabled:
			from hydra.layers import ManifoldConstrainedHyperConnection

			# Compute placement: every N blocks
			interval = max(1, self.manifold_placement_interval)
			self._manifold_placement = tuple(range(0, n_mor_blocks, interval))

			_log.info(f"Manifold: {len(self._manifold_placement)} connections at positions {self._manifold_placement}")

			for _ in self._manifold_placement:
				mc = ManifoldConstrainedHyperConnection(
					dim=dim,
					n_components=self.manifold_n_components,
					manifold_type=self.manifold_type,
					warmup_steps=self.manifold_warmup_steps,
					curvature=self.manifold_curvature,
				)
				self.manifold_connections.append(mc)

		self.norm = RMSNorm(dim)
		self.output = nn.Linear(dim, vocab_size, bias=False)

		# Cache placement lookups — these are constant after __init__ and were
		# being rebuilt as dicts on every forward() call.
		self._moe_after_block: dict = {}
		if self.moe_enabled and self.moe_layers:
			for moe_idx, block_idx in enumerate(self._moe_placement):
				self._moe_after_block[block_idx] = moe_idx

		self._manifold_after_block: dict = {}
		if self.manifold_connections:
			for mc_idx, block_idx in enumerate(self._manifold_placement):
				self._manifold_after_block[block_idx] = mc_idx

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
	_skip_moe: bool = False  # Skip MoE layers during forward (for memory-efficient GRPO)

	def enable_gradient_checkpointing(self, every_n: int = 1) -> None:
		self._gradient_checkpointing = True
		self._checkpoint_every_n = max(1, every_n)
		# Disable MoR sparse packing — variable-size tensors break checkpoint
		# recomputation (GPU float non-determinism changes n_active by ±1).
		self._set_mor_force_dense(True)

	def disable_gradient_checkpointing(self) -> None:
		self._gradient_checkpointing = False
		self._set_mor_force_dense(False)

	def _set_mor_force_dense(self, force: bool) -> None:
		"""Set _force_dense on all MoRExecutor modules."""
		from hydra.routing.mixture_of_recursions import MoRExecutor
		for m in self.modules():
			if isinstance(m, MoRExecutor):
				m._force_dense = force

	@property
	def is_gradient_checkpointing(self) -> bool:
		return self._gradient_checkpointing

	def set_skip_moe(self, skip: bool) -> None:
		"""Enable/disable MoE layer skipping for memory-efficient GRPO.

		When skip=True, MoE layers are bypassed during forward pass.
		This dramatically reduces memory for reasoning training while
		still training the base transformer layers.

		Args:
			skip: If True, skip all MoE layers during forward pass.
		"""
		self._skip_moe = skip
		if skip:
			_log.info("MoE layers DISABLED (skip_moe=True)")
		else:
			_log.info("MoE layers ENABLED (skip_moe=False)")

	def forward(
		self,
		x: torch.Tensor,
		mask: Optional[torch.Tensor] = None,
		return_losses: bool = False,
	) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
		"""Forward pass.
		
		Args:
			x: Input token ids [batch, seq_len]
			mask: Optional attention mask [batch, seq_len] (unused, for API compat)
			return_losses: If True, return (logits, losses_dict) with aux/ponder losses
		
		Returns:
			If return_losses=False: logits [batch, seq_len, vocab_size]
			If return_losses=True: (logits, {"aux_loss": ..., "ponder_loss": ..., "moe_aux_loss": ...})
		"""
		# Token embedding with sqrt(dim) scaling (LLaMA style)
		# Uses safe lookup to prevent Xid 13 GPU crashes from OOB tokens
		h = _safe_embedding_lookup(self.tok_emb, x, scale=self._sqrt_dim)

		# Use cached placement lookups (built once in __init__)
		# Skip MoE if _skip_moe flag is set (for memory-efficient GRPO)
		moe_after_block = {} if self._skip_moe else self._moe_after_block
		manifold_after_block = self._manifold_after_block

		if return_losses:
			if self._gradient_checkpointing and self.training:
				layer_results = []
				moe_results = []
				for i, layer in enumerate(self.layers):
					if i % self._checkpoint_every_n == 0:
						h, layer_losses = gradient_checkpoint(layer.forward_with_losses, h, use_reentrant=False)
					else:
						h, layer_losses = layer.forward_with_losses(h)
					layer_results.append(layer_losses)
					
					# Apply MoE after this block if scheduled
					if i in moe_after_block:
						moe_idx = moe_after_block[i]
						moe_block = self.moe_layers[moe_idx]
						if i % self._checkpoint_every_n == 0:
							h, moe_losses = gradient_checkpoint(moe_block.forward_with_losses, h, use_reentrant=False)
						else:
							h, moe_losses = moe_block.forward_with_losses(h)
						moe_results.append(moe_losses)

					# Apply manifold connection after this block if scheduled
					if i in manifold_after_block:
						mc_idx = manifold_after_block[i]
						mc = self.manifold_connections[mc_idx]
						if i % self._checkpoint_every_n == 0:
							h = gradient_checkpoint(mc, h, use_reentrant=False)
						else:
							h = mc(h)
			else:
				layer_results = []
				moe_results = []
				for i, layer in enumerate(self.layers):
					h, layer_losses = layer.forward_with_losses(h)
					layer_results.append(layer_losses)

					# Apply MoE after this block if scheduled
					if i in moe_after_block:
						moe_idx = moe_after_block[i]
						moe_block = self.moe_layers[moe_idx]
						h, moe_losses = moe_block.forward_with_losses(h)
						moe_results.append(moe_losses)

					# Apply manifold connection after this block if scheduled
					if i in manifold_after_block:
						mc_idx = manifold_after_block[i]
						h = self.manifold_connections[mc_idx](h)

			aux_losses = [losses["aux_loss"] for losses in layer_results if "aux_loss" in losses]
			ponder_losses = [losses["ponder_loss"] for losses in layer_results if "ponder_loss" in losses]
			moe_aux_losses = [losses.get("moe_aux_loss", self._zero_scalar) for losses in moe_results]
			moe_teacher_losses = [losses.get("moe_teacher_loss", self._zero_scalar) for losses in moe_results]

			h = self.norm(h)
			logits = self.output(h)

			aux_loss = torch.stack(aux_losses).sum() if aux_losses else self._zero_scalar
			ponder_loss = torch.stack(ponder_losses).sum() if ponder_losses else self._zero_scalar
			moe_aux_loss = torch.stack(moe_aux_losses).sum() if moe_aux_losses else self._zero_scalar
			moe_teacher_loss = torch.stack(moe_teacher_losses).sum() if moe_teacher_losses else self._zero_scalar
			return logits, {"aux_loss": aux_loss, "ponder_loss": ponder_loss, "moe_aux_loss": moe_aux_loss, "moe_teacher_loss": moe_teacher_loss}

		if self._gradient_checkpointing and self.training:
			for i, layer in enumerate(self.layers):
				if i % self._checkpoint_every_n == 0:
					h = gradient_checkpoint(layer, h, use_reentrant=False)
				else:
					h = layer(h)
				# Apply MoE after this block if scheduled
				if i in moe_after_block:
					moe_idx = moe_after_block[i]
					moe_block = self.moe_layers[moe_idx]
					if i % self._checkpoint_every_n == 0:
						h = gradient_checkpoint(moe_block, h, use_reentrant=False)
					else:
						h = moe_block(h)
				# Apply manifold connection after this block if scheduled
				if i in manifold_after_block:
					mc_idx = manifold_after_block[i]
					mc = self.manifold_connections[mc_idx]
					if i % self._checkpoint_every_n == 0:
						h = gradient_checkpoint(mc, h, use_reentrant=False)
					else:
						h = mc(h)
		else:
			for i, layer in enumerate(self.layers):
				h = layer(h)
				# Apply MoE after this block if scheduled
				if i in moe_after_block:
					moe_idx = moe_after_block[i]
					h = self.moe_layers[moe_idx](h)
				# Apply manifold connection after this block if scheduled
				if i in manifold_after_block:
					mc_idx = manifold_after_block[i]
					h = self.manifold_connections[mc_idx](h)

		h = self.norm(h)
		return self.output(h)

	def forward_with_cache(
		self,
		x: torch.Tensor,
		past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
		start_pos: int = 0,
	) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
		"""Forward pass with KV cache for incremental generation.

		On prefill (past_key_values=None), processes full sequence and returns
		cached K,V per layer. On decode (past_key_values provided), processes
		only new tokens and concatenates with cache.

		Skips MoE and manifold connections for generation simplicity.

		Args:
			x: Token ids [B, S] where S=full seq for prefill, S=1 for decode
			past_key_values: List of (K_cache, V_cache) per layer, or None
			start_pos: Position offset for RoPE

		Returns:
			(logits, new_past_key_values) where logits is [B, S, vocab_size]
		"""
		h = _safe_embedding_lookup(self.tok_emb, x, scale=self._sqrt_dim)

		n_layers = len(self.layers)
		new_past_key_values = []

		for i, layer in enumerate(self.layers):
			layer_past_kv = past_key_values[i] if past_key_values is not None else None
			h, new_kv = layer.forward_with_cache(
				h, past_kv=layer_past_kv, start_pos=start_pos,
			)
			new_past_key_values.append(new_kv)

		h = self.norm(h)
		logits = self.output(h)
		return logits, new_past_key_values

	def forward_hidden(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
		"""Return post-norm hidden states (pre-logits).

		Used for memory-efficient loss functions (e.g., chunked CE) that avoid
		materializing the full logits tensor.

		Args:
			x: Input token ids [batch, seq_len]
			mask: Optional attention mask [batch, seq_len] (1=valid, 0=pad)

		Returns:
			Hidden states after final norm, shape [batch, seq_len, dim]
		"""
		# Token embedding with sqrt(dim) scaling (LLaMA style)
		# Uses safe lookup to prevent Xid 13 GPU crashes from OOB tokens
		h = _safe_embedding_lookup(self.tok_emb, x, scale=self._sqrt_dim)

		# Use cached placement lookups (built once in __init__)
		# Skip MoE if _skip_moe flag is set (for memory-efficient GRPO)
		moe_after_block = {} if self._skip_moe else self._moe_after_block
		manifold_after_block = self._manifold_after_block

		if self._gradient_checkpointing and self.training:
			for i, layer in enumerate(self.layers):
				if i % self._checkpoint_every_n == 0:
					h = gradient_checkpoint(layer, h, use_reentrant=False)
				else:
					h = layer(h)
				# Apply MoE after this block if scheduled
				if i in moe_after_block:
					moe_idx = moe_after_block[i]
					moe_block = self.moe_layers[moe_idx]
					if i % self._checkpoint_every_n == 0:
						h = gradient_checkpoint(moe_block, h, use_reentrant=False)
					else:
						h = moe_block(h)
				# Apply manifold connection after this block if scheduled
				if i in manifold_after_block:
					mc_idx = manifold_after_block[i]
					mc = self.manifold_connections[mc_idx]
					if i % self._checkpoint_every_n == 0:
						h = gradient_checkpoint(mc, h, use_reentrant=False)
					else:
						h = mc(h)
		else:
			for i, layer in enumerate(self.layers):
				h = layer(h)
				if i in moe_after_block:
					h = self.moe_layers[moe_after_block[i]](h)
				if i in manifold_after_block:
					h = self.manifold_connections[manifold_after_block[i]](h)

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
				- losses: Dict with 'aux_loss', 'ponder_loss', 'moe_aux_loss'
		"""
		# Token embedding with sqrt(dim) scaling (LLaMA style)
		# Uses safe lookup to prevent Xid 13 GPU crashes from OOB tokens
		h = _safe_embedding_lookup(self.tok_emb, x, scale=self._sqrt_dim)

		# Use cached placement lookups (built once in __init__)
		# Skip MoE if _skip_moe flag is set (for memory-efficient GRPO)
		moe_after_block = {} if self._skip_moe else self._moe_after_block
		manifold_after_block = self._manifold_after_block

		if self._gradient_checkpointing and self.training:
			layer_results = []
			moe_results = []
			for i, layer in enumerate(self.layers):
				if i % self._checkpoint_every_n == 0:
					h, layer_losses = gradient_checkpoint(layer.forward_with_losses, h, mask, use_reentrant=False)
				else:
					h, layer_losses = layer.forward_with_losses(h, mask)
				layer_results.append(layer_losses)
				# Apply MoE after this block if scheduled
				if i in moe_after_block:
					moe_idx = moe_after_block[i]
					moe_block = self.moe_layers[moe_idx]
					if i % self._checkpoint_every_n == 0:
						h, moe_losses = gradient_checkpoint(moe_block.forward_with_losses, h, use_reentrant=False)
					else:
						h, moe_losses = moe_block.forward_with_losses(h)
					moe_results.append(moe_losses)
				# Apply manifold connection after this block if scheduled
				if i in manifold_after_block:
					mc_idx = manifold_after_block[i]
					mc = self.manifold_connections[mc_idx]
					if i % self._checkpoint_every_n == 0:
						h = gradient_checkpoint(mc, h, use_reentrant=False)
					else:
						h = mc(h)
		else:
			layer_results = []
			moe_results = []
			for i, layer in enumerate(self.layers):
				h, layer_losses = layer.forward_with_losses(h, mask)
				layer_results.append(layer_losses)
				if i in moe_after_block:
					moe_idx = moe_after_block[i]
					h, moe_losses = self.moe_layers[moe_idx].forward_with_losses(h)
					moe_results.append(moe_losses)
				if i in manifold_after_block:
					h = self.manifold_connections[manifold_after_block[i]](h)

		aux_losses = [losses["aux_loss"] for losses in layer_results if "aux_loss" in losses]
		ponder_losses = [losses["ponder_loss"] for losses in layer_results if "ponder_loss" in losses]
		moe_aux_losses = [losses.get("moe_aux_loss", self._zero_scalar) for losses in moe_results]
		moe_teacher_losses = [losses.get("moe_teacher_loss", self._zero_scalar) for losses in moe_results]

		aux_loss = torch.stack(aux_losses).sum() if aux_losses else self._zero_scalar
		ponder_loss = torch.stack(ponder_losses).sum() if ponder_losses else self._zero_scalar
		moe_aux_loss = torch.stack(moe_aux_losses).sum() if moe_aux_losses else self._zero_scalar
		moe_teacher_loss = torch.stack(moe_teacher_losses).sum() if moe_teacher_losses else self._zero_scalar
		return self.norm(h), {"aux_loss": aux_loss, "ponder_loss": ponder_loss, "moe_aux_loss": moe_aux_loss, "moe_teacher_loss": moe_teacher_loss}

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
		# Update MoE layers
		for moe_layer in self.moe_layers:
			if hasattr(moe_layer, "set_global_step"):
				moe_layer.set_global_step(step)
		# Update manifold connections
		for mc in self.manifold_connections:
			mc.set_global_step(step)
	
	def freeze_manifold_grads(self) -> None:
		"""Freeze all manifold connection gradients.
		
		Call this when resuming a checkpoint that doesn't have manifold params,
		to prevent catastrophic forgetting while the model adapts.
		"""
		for mc in self.manifold_connections:
			mc.freeze_grads()
	
	def unfreeze_manifold_grads(self) -> None:
		"""Unfreeze manifold connection gradients to resume learning."""
		for mc in self.manifold_connections:
			mc.unfreeze_grads()
	
	def manifold_grads_frozen(self) -> bool:
		"""Return True if manifold gradients are frozen."""
		if not self.manifold_connections:
			return False
		return self.manifold_connections[0].grads_frozen
	
	def set_manifold_warmup_start(self, step: int) -> None:
		"""Set the warmup start step for all manifold connections.
		
		Call this when adding manifold to a trained checkpoint to ensure
		warmup is relative to when manifold was added.
		"""
		for mc in self.manifold_connections:
			mc.set_warmup_start_step(step)

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

		# MoE stats
		moe_stats = []
		for i, moe_layer in enumerate(self.moe_layers):
			if hasattr(moe_layer, "get_routing_stats"):
				block_idx = self._moe_placement[i] if i < len(self._moe_placement) else -1
				moe_stats.append({"moe_idx": i, "after_block": block_idx, **moe_layer.get_routing_stats()})

		summary = {}
		if mod_stats:
			probs = [s.get("probs_mean", 0) for s in mod_stats]
			summary["mod_probs_mean"] = sum(probs) / len(probs) if probs else 0
		if mor_stats:
			avg_depths = [s.get("avg_depth", 0) for s in mor_stats if "avg_depth" in s]
			if avg_depths:
				summary["mor_avg_depth"] = sum(avg_depths) / len(avg_depths)
		if moe_stats:
			expert_utils = [s.get("expert_utilization", []) for s in moe_stats]
			aux_losses = [s.get("aux_loss", 0) for s in moe_stats]
			summary["moe_enabled"] = True
			summary["moe_num_layers"] = len(moe_stats)
			summary["moe_avg_aux_loss"] = sum(aux_losses) / len(aux_losses) if aux_losses else 0
		else:
			summary["moe_enabled"] = False

		# Manifold connection stats
		manifold_stats = []
		for i, mc in enumerate(self.manifold_connections):
			block_idx = self._manifold_placement[i] if i < len(self._manifold_placement) else -1
			manifold_stats.append({"mc_idx": i, "after_block": block_idx, **mc.get_stats()})
		if manifold_stats:
			alphas = [s.get("residual_alpha", 0) for s in manifold_stats]
			summary["manifold_enabled"] = True
			summary["manifold_num_connections"] = len(manifold_stats)
			summary["manifold_avg_alpha"] = sum(alphas) / len(alphas) if alphas else 0
		else:
			summary["manifold_enabled"] = False

		return {"mod_layers": mod_stats, "mor_layers": mor_stats, "moe_layers": moe_stats, "manifold_connections": manifold_stats, "summary": summary}


# Backward compatibility alias
CCGQAMoDMoRModel = HydraModel


__all__ = [
	"HydraModel",
	"HydraBaseModel",
	# Backward compat
	"CCGQAModel",
	"CCGQAMoDMoRModel",
]
