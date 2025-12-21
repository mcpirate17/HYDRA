from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from hydra.model.ccgqa import CCGQAMoDMoRModel
from hydra.training.config import MODEL_SIZE_CONFIGS


@dataclass(frozen=True, slots=True)
class HealthSummary:
    mod_routing_mode_counts: Dict[str, int]
    mod_probs_mean: float
    mod_compute_ratio: float
    mor_avg_depth: float
    mor_collapsed_layers: int


def _build_model(
    *,
    model_size: str,
    seq_len: int,
    max_steps: int,
    mod_enable_pct: float,
    mor_enable_pct: float,
    mod_capacity: float,
    mor_adaptive: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[CCGQAMoDMoRModel, int, int]:
    size_cfg = MODEL_SIZE_CONFIGS[model_size]

    mod_enable_step = int(max_steps * mod_enable_pct)
    mor_enable_step = int(max_steps * mor_enable_pct)

    model = CCGQAMoDMoRModel(
        vocab_size=50257,
        dim=size_cfg["mod_mor_dim"],
        n_mor_blocks=size_cfg["n_mor_blocks"],
        recursions_per_block=size_cfg["mor_recursions"],
        n_heads=size_cfg["mod_mor_n_heads"],
        n_kv_heads=size_cfg["mod_mor_n_kv_heads"],
        compression_factor=4,
        mlp_ratio=3.6,
        max_seq_len=seq_len,
        mod_capacity=mod_capacity,
        adaptive=mor_adaptive,
        tie_weights=True,
        mod_mlp_warmup=mod_enable_step,
        mor_warmup=mor_enable_step,
    ).to(device=device, dtype=dtype)

    if mor_adaptive:
        # Mirror training behavior: fixed-depth until enable_step, then ramp.
        rampup_steps = max(100, int(max_steps * 0.10))
        model.set_mor_curriculum(enable_step=mor_enable_step, rampup_steps=rampup_steps)

    return model, mod_enable_step, mor_enable_step


def _summarize(model: CCGQAMoDMoRModel) -> HealthSummary:
    mod_probs: List[float] = []
    mod_ratios: List[float] = []
    mod_modes: Dict[str, int] = {}

    mor_depths: List[float] = []
    mor_collapsed = 0

    for layer in model.layers:
        if hasattr(layer, "mod_mlp_wrapper") and layer.mod_mlp_wrapper is not None:
            stats = layer.mod_mlp_wrapper.get_routing_stats()
            mod_probs.append(float(stats.get("probs_mean", 0.0)))
            mod_ratios.append(float(stats.get("compute_ratio", 0.0)))
            mode = str(stats.get("routing_mode", "unknown"))
            mod_modes[mode] = mod_modes.get(mode, 0) + 1

        if hasattr(layer, "get_routing_stats"):
            stats = layer.get_routing_stats()
            if "avg_depth" in stats:
                mor_depths.append(float(stats["avg_depth"]))
            hist = stats.get("depth_histogram")
            if isinstance(hist, list) and len(hist) > 0:
                total = float(sum(hist))
                if total > 0:
                    peak = max(hist) / total
                    if peak > 0.95:
                        mor_collapsed += 1

    mod_probs_mean = sum(mod_probs) / max(1, len(mod_probs))
    mod_ratio_mean = sum(mod_ratios) / max(1, len(mod_ratios))
    mor_depth_mean = sum(mor_depths) / max(1, len(mor_depths))

    return HealthSummary(
        mod_routing_mode_counts=mod_modes,
        mod_probs_mean=mod_probs_mean,
        mod_compute_ratio=mod_ratio_mean,
        mor_avg_depth=mor_depth_mean,
        mor_collapsed_layers=mor_collapsed,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="HYDRA routing healthcheck (MoD/MoR)")
    p.add_argument("--model_size", type=str, default="100M", choices=sorted(MODEL_SIZE_CONFIGS.keys()))
    p.add_argument("--seq_len", type=int, default=256)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--max_steps", type=int, default=15000, help="Used only to compute enable steps from enable_pct")
    p.add_argument("--mod_enable_pct", type=float, default=0.10)
    p.add_argument("--mor_enable_pct", type=float, default=0.30)
    p.add_argument("--mod_capacity", type=float, default=0.5)
    p.add_argument("--mor_adaptive", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32

    model, mod_enable_step, mor_enable_step = _build_model(
        model_size=args.model_size,
        seq_len=args.seq_len,
        max_steps=args.max_steps,
        mod_enable_pct=args.mod_enable_pct,
        mor_enable_pct=args.mor_enable_pct,
        mod_capacity=args.mod_capacity,
        mor_adaptive=args.mor_adaptive,
        device=device,
        dtype=dtype,
    )

    model.train()

    for step in range(1, args.steps + 1):
        model.set_global_step(step)
        x = torch.randint(0, 50257, (2, args.seq_len), device=device)
        logits, losses = model(x, return_losses=True)
        _ = (logits, losses)

        if step in (1, mod_enable_step, mor_enable_step, args.steps):
            s = _summarize(model)
            print(
                f"step={step:>5}  "
                f"MoD(mode={s.mod_routing_mode_counts}, probs_mean={s.mod_probs_mean:.3f}, compute_ratio={s.mod_compute_ratio:.3f})  "
                f"MoR(avg_depth={s.mor_avg_depth:.3f}, collapsed_layers={s.mor_collapsed_layers})"
            )

    print("--")
    print(f"device={device.type} dtype={dtype} model_size={args.model_size}")
    print(f"mod_enable_step={mod_enable_step} ({args.mod_enable_pct:.0%} of max_steps={args.max_steps})")
    print(f"mor_enable_step={mor_enable_step} ({args.mor_enable_pct:.0%} of max_steps={args.max_steps})")


if __name__ == "__main__":
    main()
