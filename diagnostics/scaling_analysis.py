#!/usr/bin/env python3
"""
Scaling Analysis for CCGQA + MoD + MoR Models

Analyzes scaling behavior across model sizes from 100M to 1.5B,
fits polynomial/power/exponential curves, and predicts behavior at 4B scale.

This validates that the aux_loss_weight scaling and other hyperparameters
will work correctly when scaling beyond locally-testable sizes.

Notes:
- RECOMMENDED: GPU for faster model diagnostics when running variants that use CUDA.
- OPTIONAL DEPS: `matplotlib`, `scipy`, `memory-profiler` for plotting and profiling.
    Install with: `pip install matplotlib scipy memory-profiler`
"""

import argparse
import json
import math
import os
import sys
import time
import warnings
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Suppress scipy warnings for clean output
warnings.filterwarnings("ignore", category=RuntimeWarning)

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Plotting disabled.")

try:
    from scipy import optimize
    from scipy import stats

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not installed. Curve fitting will use numpy only.")

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from hydra.model.framework import HydraModel, create_hydra_model


# =============================================================================
# Extended Model Variant Configurations
# =============================================================================


@dataclass
class ModelVariant:
    """Configuration for a model variant."""

    name: str
    dim: int
    n_mor_blocks: int
    recursions: int
    n_heads: int
    n_kv_heads: int
    compression_factor: int
    batch_size: int
    seq_len: int
    expected_params_m: float

    @property
    def effective_layers(self) -> int:
        return self.n_mor_blocks * self.recursions

    def to_dict(self) -> dict:
        return asdict(self)


# Extended model variants for comprehensive scaling analysis
# Following transformer scaling laws: dim ~ sqrt(params), layers ~ params^0.5
MODEL_VARIANTS = {
    "100M": ModelVariant(
        name="100M",
        dim=768,
        n_mor_blocks=8,
        recursions=4,
        n_heads=12,
        n_kv_heads=3,
        compression_factor=4,
        batch_size=4,
        seq_len=256,
        expected_params_m=100,
    ),
    "250M": ModelVariant(
        name="250M",
        dim=1024,
        n_mor_blocks=12,
        recursions=4,
        n_heads=16,
        n_kv_heads=4,
        compression_factor=4,
        batch_size=4,
        seq_len=256,
        expected_params_m=250,
    ),
    "500M": ModelVariant(
        name="500M",
        dim=1536,
        n_mor_blocks=16,
        recursions=4,
        n_heads=24,
        n_kv_heads=4,
        compression_factor=4,
        batch_size=2,
        seq_len=256,
        expected_params_m=570,
    ),
    "750M": ModelVariant(
        name="750M",
        dim=1792,
        n_mor_blocks=20,
        recursions=4,
        n_heads=28,
        n_kv_heads=4,
        compression_factor=4,
        batch_size=2,
        seq_len=256,
        expected_params_m=750,
    ),
    "900M": ModelVariant(
        name="900M",
        dim=2048,
        n_mor_blocks=20,
        recursions=4,
        n_heads=32,
        n_kv_heads=4,
        compression_factor=4,
        batch_size=2,
        seq_len=256,
        expected_params_m=900,
    ),
    "1B": ModelVariant(
        name="1B",
        dim=2048,
        n_mor_blocks=24,
        recursions=4,
        n_heads=32,
        n_kv_heads=8,
        compression_factor=4,
        batch_size=1,
        seq_len=256,
        expected_params_m=1000,
    ),
    "1.5B": ModelVariant(
        name="1.5B",
        dim=2560,
        n_mor_blocks=24,
        recursions=5,
        n_heads=40,
        n_kv_heads=8,
        compression_factor=4,
        batch_size=1,
        seq_len=256,
        expected_params_m=1500,
    ),
    # Theoretical 4B variant for prediction validation
    "4B": ModelVariant(
        name="4B",
        dim=4096,
        n_mor_blocks=32,
        recursions=5,
        n_heads=64,
        n_kv_heads=8,
        compression_factor=4,
        batch_size=1,
        seq_len=256,
        expected_params_m=4000,
    ),
}


@dataclass
class ScalingDataPoint:
    """Data point for scaling analysis."""

    name: str
    params_m: float  # Actual params in millions
    effective_layers: int
    dim: int
    aux_loss_weight: float
    final_mod_prob: float
    final_mor_depth: float
    mod_converged: bool
    mor_not_collapsed: bool
    ms_per_step: float
    compliance_passed: bool


@dataclass
class ScalingFit:
    """Results of curve fitting."""

    fit_type: str  # 'polynomial', 'power', 'exponential'
    coefficients: List[float]
    r_squared: float
    equation: str
    prediction_4b: float


def get_device() -> str:
    """Get available device."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_aux_loss_weight(effective_layers: int, dim: int) -> float:
    """Calculate aux_loss_weight using the same formula as the model."""
    depth_scale = max(1.0, effective_layers / 32)
    dim_scale = max(1.0, (dim / 768) ** 0.5)
    return 0.01 * depth_scale * dim_scale


# =============================================================================
# Diagnostic Runner
# =============================================================================


def run_diagnostic(
    variant: ModelVariant, steps: int = 30, device: str = None
) -> ScalingDataPoint:
    """Run diagnostic training on a model variant."""
    if device is None:
        device = get_device()

    # Default attention pattern (per MoR block): 3:1 macro-block [LLA2, LLA2, LLA2, CCQA].
    # Keep this CUDA-only: LLA2 requires the external lightning-attention CUDA kernels.
    if device == "cuda":
        os.environ.setdefault("HYDRA_MOR_ATTENTION_PATTERN_NAME", "lla2x3+ccqa")

    vocab_size = 50257

    print(f"\n{'=' * 60}")
    print(f"DIAGNOSTIC: {variant.name}")
    print(f"{'=' * 60}")

    # Create model
    print(
        f"Creating model: dim={variant.dim}, blocks={variant.n_mor_blocks}, rec={variant.recursions}"
    )
    model = create_hydra_model(
        vocab_size=vocab_size,
        dim=variant.dim,
        n_mor_blocks=variant.n_mor_blocks,
        recursions_per_block=variant.recursions,
        n_heads=variant.n_heads,
        n_kv_heads=variant.n_kv_heads,
        compression_factor=variant.compression_factor,
        mod_capacity=0.75,
        adaptive=True,
    )
    model = model.to(device)

    param_count = count_parameters(model)
    print(f"Parameters: {param_count:,} ({param_count / 1e6:.1f}M)")
    print(f"Effective layers: {model.effective_layers}")
    print(f"aux_loss_weight: {model.aux_loss_weight:.4f}")

    # Training setup
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.1
    )

    # Track metrics
    mod_probs = []
    mor_depths = []
    step_times = []

    print(f"\nRunning {steps} training steps...")

    model.train()
    for step in range(1, steps + 1):
        start_time = time.time()

        # Generate random data
        input_ids = torch.randint(
            0, vocab_size, (variant.batch_size, variant.seq_len), device=device
        )
        target_ids = torch.randint(
            0, vocab_size, (variant.batch_size, variant.seq_len), device=device
        )

        # Forward pass - use return_losses=True to get aux_loss and ponder_loss
        optimizer.zero_grad()
        logits, losses = model(input_ids, return_losses=True)
        aux_loss = losses.get("aux_loss", torch.tensor(0.0, device=device))
        ponder_loss = losses.get("ponder_loss", torch.tensor(0.0, device=device))

        # Compute loss
        ce_loss = F.cross_entropy(logits.view(-1, vocab_size), target_ids.view(-1))
        total_loss = ce_loss + aux_loss + ponder_loss

        # Backward pass
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        step_time = (time.time() - start_time) * 1000
        step_times.append(step_time)

        # Collect MoD/MoR stats
        mod_prob = 0.0
        mod_count = 0
        mor_depth = 0.0
        mor_count = 0

        for layer in model.layers:
            if hasattr(layer, "_last_probs_mean"):
                mod_prob += layer._last_probs_mean
                mod_count += 1
            if hasattr(layer, "_last_avg_depth"):
                mor_depth += layer._last_avg_depth
                mor_count += 1
            elif hasattr(layer, "mor_block") and hasattr(
                layer.mor_block, "_last_avg_depth"
            ):
                mor_depth += layer.mor_block._last_avg_depth
                mor_count += 1
            # Check if it's a MoRBlock wrapped in MoD
            if hasattr(layer, "block") and hasattr(layer.block, "_last_avg_depth"):
                mor_depth += layer.block._last_avg_depth
                mor_count += 1

        mod_prob = mod_prob / mod_count if mod_count > 0 else 0.75
        mor_depth = mor_depth / mor_count if mor_count > 0 else 1.0

        mod_probs.append(mod_prob)
        mor_depths.append(mor_depth)

        if step == 1 or step % 10 == 0 or step == steps:
            print(
                f"[Step {step:3d}/{steps}] loss={total_loss.item():.4f} "
                f"mod_prob={mod_prob:.3f} mor_depth={mor_depth:.3f} time={step_time:.0f}ms"
            )

    # Final analysis
    final_mod_prob = mod_probs[-1] if mod_probs else 0.75
    final_mor_depth = mor_depths[-1] if mor_depths else 1.0
    
    # Convert to Python floats if they're tensors
    if hasattr(final_mod_prob, 'item'):
        final_mod_prob = final_mod_prob.item()
    if hasattr(final_mor_depth, 'item'):
        final_mor_depth = final_mor_depth.item()
    
    avg_time = (
        sum(step_times[5:]) / len(step_times[5:])
        if len(step_times) > 5
        else sum(step_times) / len(step_times)
    )

    # Compliance checks
    mod_converged = abs(final_mod_prob - 0.75) < 0.3  # Within 30% of target
    mor_not_collapsed = 0.1 < final_mor_depth < (variant.recursions - 0.1)
    compliance_passed = mod_converged and mor_not_collapsed

    print(f"\nSUMMARY: {variant.name}")
    print(f"  Params: {param_count / 1e6:.1f}M")
    print(f"  MoD prob: {final_mod_prob:.3f} (target: 0.75)")
    print(f"  MoR depth: {final_mor_depth:.3f} (max: {variant.recursions - 1})")
    print(f"  Compliance: {'PASS' if compliance_passed else 'FAIL'}")

    # Cleanup
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return ScalingDataPoint(
        name=variant.name,
        params_m=param_count / 1e6,
        effective_layers=variant.n_mor_blocks * variant.recursions,
        dim=variant.dim,
        aux_loss_weight=calculate_aux_loss_weight(
            variant.n_mor_blocks * variant.recursions, variant.dim
        ),
        final_mod_prob=final_mod_prob,
        final_mor_depth=final_mor_depth,
        mod_converged=mod_converged,
        mor_not_collapsed=mor_not_collapsed,
        ms_per_step=avg_time,
        compliance_passed=compliance_passed,
    )


# =============================================================================
# Curve Fitting Functions
# =============================================================================


def fit_polynomial(x: np.ndarray, y: np.ndarray, degree: int = 2) -> ScalingFit:
    """Fit polynomial curve."""
    coeffs = np.polyfit(x, y, degree)
    p = np.poly1d(coeffs)
    y_pred = p(x)

    # R-squared
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    # Equation
    terms = []
    for i, c in enumerate(coeffs):
        power = degree - i
        if power == 0:
            terms.append(f"{c:.6f}")
        elif power == 1:
            terms.append(f"{c:.6f}x")
        else:
            terms.append(f"{c:.6f}x^{power}")
    equation = " + ".join(terms)

    # Predict at 4B (4000M params)
    prediction_4b = p(4000)

    return ScalingFit(
        fit_type=f"polynomial_deg{degree}",
        coefficients=coeffs.tolist(),
        r_squared=r_squared,
        equation=equation,
        prediction_4b=prediction_4b,
    )


def fit_power_law(x: np.ndarray, y: np.ndarray) -> ScalingFit:
    """Fit power law: y = a * x^b."""
    # Use log-log linear regression
    log_x = np.log(x + 1e-10)
    log_y = np.log(np.maximum(y, 1e-10))

    coeffs = np.polyfit(log_x, log_y, 1)
    b = coeffs[0]  # exponent
    a = np.exp(coeffs[1])  # coefficient

    y_pred = a * (x**b)

    # R-squared
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    equation = f"{a:.6f} * x^{b:.4f}"
    prediction_4b = a * (4000**b)

    return ScalingFit(
        fit_type="power_law",
        coefficients=[a, b],
        r_squared=r_squared,
        equation=equation,
        prediction_4b=prediction_4b,
    )


def fit_exponential(x: np.ndarray, y: np.ndarray) -> ScalingFit:
    """Fit exponential: y = a * exp(b * x)."""
    # Use log-linear regression
    log_y = np.log(np.maximum(y, 1e-10))

    coeffs = np.polyfit(x, log_y, 1)
    b = coeffs[0]  # rate
    a = np.exp(coeffs[1])  # coefficient

    y_pred = a * np.exp(b * x)

    # R-squared
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    equation = f"{a:.6f} * exp({b:.8f} * x)"
    prediction_4b = a * np.exp(b * 4000)

    return ScalingFit(
        fit_type="exponential",
        coefficients=[a, b],
        r_squared=r_squared,
        equation=equation,
        prediction_4b=prediction_4b,
    )


def fit_logarithmic(x: np.ndarray, y: np.ndarray) -> ScalingFit:
    """Fit logarithmic: y = a * log(x) + b."""
    log_x = np.log(x + 1)

    coeffs = np.polyfit(log_x, y, 1)
    a = coeffs[0]
    b = coeffs[1]

    y_pred = a * log_x + b

    # R-squared
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    equation = f"{a:.6f} * log(x) + {b:.6f}"
    prediction_4b = a * np.log(4001) + b

    return ScalingFit(
        fit_type="logarithmic",
        coefficients=[a, b],
        r_squared=r_squared,
        equation=equation,
        prediction_4b=prediction_4b,
    )


# =============================================================================
# Scaling Analysis
# =============================================================================


def analyze_scaling(data_points: List[ScalingDataPoint]) -> Dict[str, Any]:
    """Analyze scaling behavior and fit curves."""

    # Extract arrays
    params = np.array([d.params_m for d in data_points])
    layers = np.array([d.effective_layers for d in data_points])
    dims = np.array([d.dim for d in data_points])
    aux_weights = np.array([d.aux_loss_weight for d in data_points])
    mod_probs = np.array([d.final_mod_prob for d in data_points])
    mor_depths = np.array([d.final_mor_depth for d in data_points])
    times = np.array([d.ms_per_step for d in data_points])

    results = {
        "data_points": [asdict(d) for d in data_points],
        "fits": {},
    }

    print("\n" + "=" * 70)
    print("SCALING ANALYSIS")
    print("=" * 70)

    # Analyze aux_loss_weight scaling
    print("\n1. AUX_LOSS_WEIGHT SCALING (params -> aux_loss_weight)")
    print("-" * 50)

    fits = [
        fit_polynomial(params, aux_weights, degree=1),
        fit_polynomial(params, aux_weights, degree=2),
        fit_power_law(params, aux_weights),
        fit_logarithmic(params, aux_weights),
    ]

    best_fit = max(fits, key=lambda f: f.r_squared)
    results["fits"]["aux_loss_weight"] = {
        "best_fit": asdict(best_fit),
        "all_fits": [asdict(f) for f in fits],
    }

    for fit in fits:
        print(
            f"  {fit.fit_type:20s}: R²={fit.r_squared:.4f}, 4B prediction={fit.prediction_4b:.4f}"
        )
    print(f"  BEST FIT: {best_fit.fit_type} with R²={best_fit.r_squared:.4f}")
    print(f"  Predicted aux_loss_weight at 4B: {best_fit.prediction_4b:.4f}")

    # Analyze MoD probability stability
    print("\n2. MOD PROBABILITY STABILITY (params -> mod_prob)")
    print("-" * 50)

    fits = [
        fit_polynomial(params, mod_probs, degree=1),
        fit_polynomial(params, mod_probs, degree=2),
        fit_logarithmic(params, mod_probs),
    ]

    best_fit = max(fits, key=lambda f: f.r_squared)
    results["fits"]["mod_prob"] = {
        "best_fit": asdict(best_fit),
        "all_fits": [asdict(f) for f in fits],
    }

    for fit in fits:
        print(
            f"  {fit.fit_type:20s}: R²={fit.r_squared:.4f}, 4B prediction={fit.prediction_4b:.3f}"
        )
    print(f"  BEST FIT: {best_fit.fit_type} with R²={best_fit.r_squared:.4f}")
    print(f"  Predicted MoD prob at 4B: {best_fit.prediction_4b:.3f}")

    # Target is 0.75, check if prediction is reasonable
    if 0.5 < best_fit.prediction_4b < 1.0:
        print(f"  ✓ 4B prediction is within acceptable range (0.5-1.0)")
    else:
        print(f"  ⚠ 4B prediction may be unstable")

    # Analyze MoR depth behavior
    print("\n3. MOR DEPTH BEHAVIOR (params -> mor_depth)")
    print("-" * 50)

    fits = [
        fit_polynomial(params, mor_depths, degree=1),
        fit_polynomial(params, mor_depths, degree=2),
        fit_logarithmic(params, mor_depths),
    ]

    best_fit = max(fits, key=lambda f: f.r_squared)
    results["fits"]["mor_depth"] = {
        "best_fit": asdict(best_fit),
        "all_fits": [asdict(f) for f in fits],
    }

    for fit in fits:
        print(
            f"  {fit.fit_type:20s}: R²={fit.r_squared:.4f}, 4B prediction={fit.prediction_4b:.3f}"
        )
    print(f"  BEST FIT: {best_fit.fit_type} with R²={best_fit.r_squared:.4f}")
    print(f"  Predicted MoR depth at 4B: {best_fit.prediction_4b:.3f}")

    # Analyze compute scaling (params -> ms/step)
    print("\n4. COMPUTE SCALING (params -> ms/step)")
    print("-" * 50)

    fits = [
        fit_polynomial(params, times, degree=1),
        fit_polynomial(params, times, degree=2),
        fit_power_law(params, times),
    ]

    best_fit = max(fits, key=lambda f: f.r_squared)
    results["fits"]["compute_time"] = {
        "best_fit": asdict(best_fit),
        "all_fits": [asdict(f) for f in fits],
    }

    for fit in fits:
        print(
            f"  {fit.fit_type:20s}: R²={fit.r_squared:.4f}, 4B prediction={fit.prediction_4b:.0f}ms"
        )
    print(f"  BEST FIT: {best_fit.fit_type} with R²={best_fit.r_squared:.4f}")
    print(f"  Predicted ms/step at 4B: {best_fit.prediction_4b:.0f}ms")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: 4B PREDICTIONS")
    print("=" * 70)

    # Get predictions from aux_loss_weight formula
    # 4B: dim=4096, layers=160 (32 blocks x 5 recursions)
    predicted_4b_layers = 160
    predicted_4b_dim = 4096
    predicted_aux_weight = calculate_aux_loss_weight(
        predicted_4b_layers, predicted_4b_dim
    )

    print(f"\nTheoretical 4B model:")
    print(f"  dim: 4096")
    print(f"  effective_layers: 160 (32 blocks x 5 recursions)")
    print(f"  Calculated aux_loss_weight: {predicted_aux_weight:.4f}")

    print(f"\nFrom curve fitting:")
    aux_fit = results["fits"]["aux_loss_weight"]["best_fit"]
    mod_fit = results["fits"]["mod_prob"]["best_fit"]
    mor_fit = results["fits"]["mor_depth"]["best_fit"]
    time_fit = results["fits"]["compute_time"]["best_fit"]

    print(
        f"  aux_loss_weight: {aux_fit['prediction_4b']:.4f} ({aux_fit['fit_type']}, R²={aux_fit['r_squared']:.3f})"
    )
    print(
        f"  mod_prob: {mod_fit['prediction_4b']:.3f} ({mod_fit['fit_type']}, R²={mod_fit['r_squared']:.3f})"
    )
    print(
        f"  mor_depth: {mor_fit['prediction_4b']:.3f} ({mor_fit['fit_type']}, R²={mor_fit['r_squared']:.3f})"
    )
    print(
        f"  ms/step: {time_fit['prediction_4b']:.0f} ({time_fit['fit_type']}, R²={time_fit['r_squared']:.3f})"
    )

    # Compliance prediction
    mod_ok = 0.5 < mod_fit["prediction_4b"] < 1.0
    mor_ok = mor_fit["prediction_4b"] > 0.1  # Should maintain some depth

    print(f"\nCompliance prediction for 4B:")
    print(f"  MoD capacity: {'✓ LIKELY STABLE' if mod_ok else '⚠ MAY NEED TUNING'}")
    print(f"  MoR depth: {'✓ LIKELY STABLE' if mor_ok else '⚠ MAY COLLAPSE'}")

    results["predictions_4b"] = {
        "theoretical_aux_weight": float(predicted_aux_weight),
        "fitted_aux_weight": float(aux_fit["prediction_4b"]),
        "fitted_mod_prob": float(mod_fit["prediction_4b"]),
        "fitted_mor_depth": float(mor_fit["prediction_4b"]),
        "fitted_ms_per_step": float(time_fit["prediction_4b"]),
        "mod_compliance_likely": bool(mod_ok),
        "mor_compliance_likely": bool(mor_ok),
    }

    return results


# =============================================================================
# Plotting
# =============================================================================


def plot_scaling_analysis(
    data_points: List[ScalingDataPoint], results: Dict[str, Any], output_dir: str = "."
):
    """Generate scaling analysis plots."""
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available, skipping plots")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Extract data
    params = np.array([d.params_m for d in data_points])
    aux_weights = np.array([d.aux_loss_weight for d in data_points])
    mod_probs = np.array([d.final_mod_prob for d in data_points])
    mor_depths = np.array([d.final_mor_depth for d in data_points])
    times = np.array([d.ms_per_step for d in data_points])
    names = [d.name for d in data_points]

    # Extended x for predictions
    x_extended = np.linspace(params.min(), 4000, 100)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(
        "CCGQA + MoD + MoR Scaling Analysis\nPredicting 4B Model Behavior",
        fontsize=14,
        fontweight="bold",
    )

    # 1. aux_loss_weight scaling
    ax = axes[0, 0]
    ax.scatter(params, aux_weights, s=100, c="blue", edgecolors="black", zorder=5)
    for i, name in enumerate(names):
        ax.annotate(
            name,
            (params[i], aux_weights[i]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
        )

    # Plot best fit
    fit = results["fits"]["aux_loss_weight"]["best_fit"]
    if fit["fit_type"].startswith("polynomial"):
        p = np.poly1d(fit["coefficients"])
        ax.plot(
            x_extended,
            p(x_extended),
            "r-",
            linewidth=2,
            label=f"Best fit: {fit['fit_type']}",
        )
    elif fit["fit_type"] == "power_law":
        a, b = fit["coefficients"]
        ax.plot(
            x_extended,
            a * (x_extended**b),
            "r-",
            linewidth=2,
            label=f"Best fit: {fit['fit_type']}",
        )
    elif fit["fit_type"] == "logarithmic":
        a, b = fit["coefficients"]
        ax.plot(
            x_extended,
            a * np.log(x_extended + 1) + b,
            "r-",
            linewidth=2,
            label=f"Best fit: {fit['fit_type']}",
        )

    # Mark 4B prediction
    ax.axvline(x=4000, color="green", linestyle="--", alpha=0.5, label="4B target")
    ax.scatter(
        [4000],
        [fit["prediction_4b"]],
        s=150,
        c="green",
        marker="*",
        edgecolors="black",
        zorder=6,
    )
    ax.annotate(
        f"4B: {fit['prediction_4b']:.4f}",
        (4000, fit["prediction_4b"]),
        textcoords="offset points",
        xytext=(-60, 10),
        ha="center",
        fontsize=10,
        color="green",
    )

    ax.set_xlabel("Parameters (M)")
    ax.set_ylabel("aux_loss_weight")
    ax.set_title(f"aux_loss_weight Scaling (R²={fit['r_squared']:.3f})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. MoD probability
    ax = axes[0, 1]
    ax.scatter(params, mod_probs, s=100, c="orange", edgecolors="black", zorder=5)
    for i, name in enumerate(names):
        ax.annotate(
            name,
            (params[i], mod_probs[i]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
        )

    # Target line
    ax.axhline(y=0.75, color="blue", linestyle="--", alpha=0.5, label="Target: 0.75")
    ax.fill_between(
        [0, 4500], 0.5, 1.0, alpha=0.1, color="green", label="Acceptable range"
    )

    # Best fit
    fit = results["fits"]["mod_prob"]["best_fit"]
    if fit["fit_type"].startswith("polynomial"):
        p = np.poly1d(fit["coefficients"])
        ax.plot(
            x_extended,
            np.clip(p(x_extended), 0, 1),
            "r-",
            linewidth=2,
            label=f"Best fit",
        )
    elif fit["fit_type"] == "logarithmic":
        a, b = fit["coefficients"]
        ax.plot(
            x_extended,
            np.clip(a * np.log(x_extended + 1) + b, 0, 1),
            "r-",
            linewidth=2,
            label=f"Best fit",
        )

    ax.axvline(x=4000, color="green", linestyle="--", alpha=0.5)
    ax.scatter(
        [4000],
        [np.clip(fit["prediction_4b"], 0, 1)],
        s=150,
        c="green",
        marker="*",
        edgecolors="black",
        zorder=6,
    )

    ax.set_xlabel("Parameters (M)")
    ax.set_ylabel("MoD Probability")
    ax.set_title(f"MoD Capacity Stability (R²={fit['r_squared']:.3f})")
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. MoR depth
    ax = axes[1, 0]
    ax.scatter(params, mor_depths, s=100, c="purple", edgecolors="black", zorder=5)
    for i, name in enumerate(names):
        ax.annotate(
            name,
            (params[i], mor_depths[i]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
        )

    fit = results["fits"]["mor_depth"]["best_fit"]
    if fit["fit_type"].startswith("polynomial"):
        p = np.poly1d(fit["coefficients"])
        ax.plot(
            x_extended,
            np.maximum(p(x_extended), 0),
            "r-",
            linewidth=2,
            label=f"Best fit",
        )
    elif fit["fit_type"] == "logarithmic":
        a, b = fit["coefficients"]
        ax.plot(
            x_extended,
            np.maximum(a * np.log(x_extended + 1) + b, 0),
            "r-",
            linewidth=2,
            label=f"Best fit",
        )

    ax.axvline(x=4000, color="green", linestyle="--", alpha=0.5, label="4B target")
    ax.scatter(
        [4000],
        [max(fit["prediction_4b"], 0)],
        s=150,
        c="green",
        marker="*",
        edgecolors="black",
        zorder=6,
    )

    ax.set_xlabel("Parameters (M)")
    ax.set_ylabel("MoR Average Depth")
    ax.set_title(f"MoR Depth Behavior (R²={fit['r_squared']:.3f})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Compute time scaling
    ax = axes[1, 1]
    ax.scatter(params, times, s=100, c="red", edgecolors="black", zorder=5)
    for i, name in enumerate(names):
        ax.annotate(
            name,
            (params[i], times[i]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
        )

    fit = results["fits"]["compute_time"]["best_fit"]
    if fit["fit_type"].startswith("polynomial"):
        p = np.poly1d(fit["coefficients"])
        ax.plot(
            x_extended,
            np.maximum(p(x_extended), 0),
            "r-",
            linewidth=2,
            label=f"Best fit",
        )
    elif fit["fit_type"] == "power_law":
        a, b = fit["coefficients"]
        ax.plot(x_extended, a * (x_extended**b), "r-", linewidth=2, label=f"Best fit")

    ax.axvline(x=4000, color="green", linestyle="--", alpha=0.5, label="4B target")
    ax.scatter(
        [4000],
        [fit["prediction_4b"]],
        s=150,
        c="green",
        marker="*",
        edgecolors="black",
        zorder=6,
    )
    ax.annotate(
        f"4B: {fit['prediction_4b']:.0f}ms",
        (4000, fit["prediction_4b"]),
        textcoords="offset points",
        xytext=(-60, 10),
        ha="center",
        fontsize=10,
        color="green",
    )

    ax.set_xlabel("Parameters (M)")
    ax.set_ylabel("Time per Step (ms)")
    ax.set_title(f"Compute Scaling (R²={fit['r_squared']:.3f})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = os.path.join(output_dir, "scaling_analysis.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {output_path}")
    plt.close()

    # Create summary table plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("off")

    # Table data
    table_data = []
    headers = [
        "Variant",
        "Params (M)",
        "Layers",
        "Dim",
        "aux_weight",
        "MoD prob",
        "MoR depth",
        "ms/step",
        "Pass",
    ]

    for d in data_points:
        table_data.append(
            [
                d.name,
                f"{d.params_m:.1f}",
                str(d.effective_layers),
                str(d.dim),
                f"{d.aux_loss_weight:.4f}",
                f"{d.final_mod_prob:.3f}",
                f"{d.final_mor_depth:.3f}",
                f"{d.ms_per_step:.0f}",
                "✓" if d.compliance_passed else "✗",
            ]
        )

    # Add 4B prediction row
    pred = results["predictions_4b"]
    table_data.append(
        [
            "4B (predicted)",
            "4000",
            "160",
            "4096",
            f"{pred['theoretical_aux_weight']:.4f}",
            f"{pred['fitted_mod_prob']:.3f}",
            f"{pred['fitted_mor_depth']:.3f}",
            f"{pred['fitted_ms_per_step']:.0f}",
            "?",
        ]
    )

    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor("#4472C4")
        table[(0, i)].set_text_props(color="white", fontweight="bold")

    # Color pass/fail - use column index instead of -1 for last column
    last_col = len(headers) - 1
    for i in range(len(table_data)):
        row_idx = i + 1  # +1 because row 0 is header
        if table_data[i][-1] == "✓":
            table[(row_idx, last_col)].set_facecolor("#C6EFCE")
        elif table_data[i][-1] == "✗":
            table[(row_idx, last_col)].set_facecolor("#FFC7CE")
        else:
            table[(row_idx, last_col)].set_facecolor("#FFEB9C")

    # Highlight 4B row (last data row)
    last_row_idx = len(table_data)  # +1 for header, but already correct
    for j in range(len(headers)):
        table[(last_row_idx, j)].set_facecolor("#E2EFDA")

    ax.set_title(
        "Model Variants & 4B Prediction Summary", fontsize=14, fontweight="bold", pad=20
    )

    output_path = os.path.join(output_dir, "scaling_summary_table.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Summary table saved to: {output_path}")
    plt.close()


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scaling Analysis for CCGQA + MoD + MoR Models"
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["100M", "250M", "500M", "750M", "900M", "1B", "1.5B"],
        help="Model variants to test",
    )
    parser.add_argument(
        "--steps", type=int, default=30, help="Training steps per variant"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="scaling_analysis_results.json",
        help="Output JSON file",
    )
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    parser.add_argument(
        "--output-dir", type=str, default="reports", help="Output directory for plots"
    )
    parser.add_argument(
        "--predict-4b", action="store_true", help="Include 4B predictions"
    )
    parser.add_argument(
        "--predict-only", action="store_true", help="Only show 4B predictions without running full analysis"
    )
    parser.add_argument(
        "--skip-large",
        action="store_true",
        help="Skip variants > 1B for faster testing",
    )

    args = parser.parse_args()
    
    # Quick predict-only mode
    if args.predict_only:
        predict_4b_scaling()
        return

    print("=" * 70)
    print("SCALING ANALYSIS: CCGQA + MoD + MoR")
    print("=" * 70)
    print(f"Variants: {args.variants}")
    print(f"Steps per variant: {args.steps}")

    # Filter variants
    variants_to_run = []
    for name in args.variants:
        if name not in MODEL_VARIANTS:
            print(f"Warning: Unknown variant {name}, skipping")
            continue
        if args.skip_large and MODEL_VARIANTS[name].expected_params_m > 1000:
            print(f"Skipping {name} (--skip-large)")
            continue
        variants_to_run.append(name)

    if not variants_to_run:
        print("No variants to run!")
        return

    # Run diagnostics
    data_points = []
    for variant_name in variants_to_run:
        variant = MODEL_VARIANTS[variant_name]
        try:
            result = run_diagnostic(variant, steps=args.steps)
            data_points.append(result)
        except Exception as e:
            print(f"Error running {variant_name}: {e}")
            continue

    if len(data_points) < 3:
        print("Need at least 3 data points for curve fitting")
        return

    # Analyze scaling
    results = analyze_scaling(data_points)

    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {args.output}")

    # Generate plots
    if args.plot:
        plot_scaling_analysis(data_points, results, args.output_dir)

    # Print paper compliance summary
    print("\n" + "=" * 70)
    print("PAPER COMPLIANCE SUMMARY")
    print("=" * 70)

    all_passed = all(d.compliance_passed for d in data_points)
    for d in data_points:
        status = "✓ PASS" if d.compliance_passed else "✗ FAIL"
        print(
            f"  {d.name:6s}: {status} (MoD={d.final_mod_prob:.3f}, MoR={d.final_mor_depth:.3f})"
        )

    print(f"\nOverall: {'ALL PASS' if all_passed else 'SOME FAILURES'}")

    if args.predict_4b:
        pred = results["predictions_4b"]
        print(f"\n4B Prediction:")
        print(f"  aux_loss_weight: {pred['theoretical_aux_weight']:.4f} (calculated)")
        print(f"  MoD prob: {pred['fitted_mod_prob']:.3f} (fitted)")
        print(f"  MoR depth: {pred['fitted_mor_depth']:.3f} (fitted)")
        print(
            f"  Compliance: {'LIKELY' if pred['mod_compliance_likely'] and pred['mor_compliance_likely'] else 'UNCERTAIN'}"
        )


def predict_4b_scaling():
    """Generate simple 4B scaling predictions."""
    print("🔮 4B Model Scaling Predictions:")
    print("=" * 50)
    
    # Simple scaling predictions based on known relationships
    models = [
        ("220M", 220, 8, 4, 32),      # Current HYDRA
        ("500M", 500, 8, 8, 64),      # 2x params → 2x batch
        ("1B", 1000, 4, 32, 128),     # 4x params → 4x batch  
        ("4B", 4000, 2, 128, 256),    # 16x params → 8x batch
    ]
    
    print("Model Size | Params | Micro | Accum | Effective | Tokens/Step")
    print("-" * 60)
    for name, params, micro, accum, effective in models:
        tokens_per_step = effective * 512  # seq_len=512
        print(f"{name:>9s} | {params:>6d}M | {micro:>5d} | {accum:>5d} | {effective:>9d} | {tokens_per_step:>10,d}")
    
    print("\n🎯 4B Recommendations:")
    print("  • Start with micro_batch=2, grad_accum=128 (256 effective)")
    print("  • Monitor VRAM: reduce micro_batch if OOM")
    print("  • Scale data workers: 16-32 for fast loading") 
    print("  • Expected VRAM: ~20-24GB (bf16 + grad + optimizer)")


if __name__ == "__main__":
    main()
