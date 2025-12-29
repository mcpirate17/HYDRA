"""Factory functions for constructing HYDRA model variants.

HYDRA uses Lightning Attention 3 (LA3) - an O(n) linear attention mechanism.
"""

from __future__ import annotations

from .model import HydraModel, HydraBaseModel


def create_hydra_model(
	vocab_size: int = 50257,
	dim: int = 2048,
	n_mor_blocks: int = 8,
	recursions_per_block: int = 4,
	n_heads: int = 32,
	n_kv_heads: int = 4,
	compression_factor: int = 4,
	mlp_ratio: float = 2.67,
	max_seq_len: int = 8192,
	mod_capacity: float = 0.5,
	aux_loss_weight: float = None,
	adaptive: bool = True,
	hybrid_attention: bool = True,
	mod_mlp_warmup: int = 100,
	mor_warmup: int = 1000,
	dim_ref: int = 768,
	depth_alpha: float = 0.0,
	depth_scale_max: float = 2.0,
) -> HydraModel:
	"""Create a HYDRA model with LA3 attention + MoD + MoR."""
	return HydraModel(
		vocab_size=vocab_size,
		dim=dim,
		n_mor_blocks=n_mor_blocks,
		recursions_per_block=recursions_per_block,
		n_heads=n_heads,
		n_kv_heads=n_kv_heads,
		compression_factor=compression_factor,
		mlp_ratio=mlp_ratio,
		max_seq_len=max_seq_len,
		mod_capacity=mod_capacity,
		aux_loss_weight=aux_loss_weight,
		adaptive=adaptive,
		hybrid_attention=hybrid_attention,
		mod_mlp_warmup=mod_mlp_warmup,
		mor_warmup=mor_warmup,
		dim_ref=dim_ref,
		depth_alpha=depth_alpha,
		depth_scale_max=depth_scale_max,
	)


def create_base_model(spec) -> HydraBaseModel:
	"""Create a simple baseline model (no routing)."""
	return HydraBaseModel(
		vocab_size=getattr(spec, "vocab_size", 50257),
		dim=getattr(spec, "dim", 1344),
		n_layers=getattr(spec, "n_layers", 24),
		n_heads=getattr(spec, "n_heads", 21),
		n_kv_heads=getattr(spec, "n_kv_heads", 3),
		compression_factor=getattr(spec, "compression_factor", 4),
		mlp_ratio=getattr(spec, "mlp_ratio", 2.67),
		max_seq_len=getattr(spec, "max_seq_len", 8192),
		tie_weights=getattr(spec, "tie_weights", True),
	)


# Backward compatibility aliases
create_ccgqa_model = create_base_model
create_ccgqa_mod_mor_model = create_hydra_model


__all__ = [
	"create_hydra_model",
	"create_base_model",
	# Backward compat
	"create_ccgqa_model",
	"create_ccgqa_mod_mor_model",
]
