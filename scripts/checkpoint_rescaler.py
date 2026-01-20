#!/usr/bin/env python3
"""
Checkpoint Rescaling Utility for HYDRA.

Diagnoses and fixes gradient-related issues in checkpoints that can cause
training instability on resume. Uses abstract base classes and composition
for extensible rescaling operations.

Usage:
    # Diagnose issues
    python scripts/checkpoint_rescaler.py checkpoints/hydra_500m_step_235000.pt --diagnose

    # Apply recommended fixes (creates backup automatically)
    python scripts/checkpoint_rescaler.py checkpoints/hydra_500m_step_235000.pt --fix

    # Custom rescaling
    python scripts/checkpoint_rescaler.py checkpoints/hydra_500m_step_235000.pt \
        --grad-ema-target 5.0 --momentum-scale 0.5

Author: HYDRA Team
"""

from __future__ import annotations

import argparse
import logging
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class DiagnosticResult:
    """Result of diagnosing a specific checkpoint component."""

    component: str
    status: str  # "healthy", "warning", "critical"
    current_value: Any
    recommended_value: Any | None = None
    message: str = ""


@dataclass
class RescaleResult:
    """Result of applying a rescaling operation."""

    rescaler_name: str
    success: bool
    changes_made: dict[str, Any] = field(default_factory=dict)
    message: str = ""


# =============================================================================
# Abstract Base Class
# =============================================================================


class CheckpointRescaler(ABC):
    """
    Abstract base class for checkpoint rescaling operations.

    Subclasses implement specific rescaling logic for different checkpoint
    components (optimizer state, trainer state, grad scaler, etc.).
    """

    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"rescaler.{name}")

    @abstractmethod
    def diagnose(self, checkpoint: dict[str, Any]) -> DiagnosticResult:
        """
        Diagnose issues in the checkpoint related to this rescaler's domain.

        Args:
            checkpoint: Loaded checkpoint dictionary

        Returns:
            DiagnosticResult with status and recommendations
        """
        pass

    @abstractmethod
    def rescale(
        self, checkpoint: dict[str, Any], **kwargs: Any
    ) -> RescaleResult:
        """
        Apply rescaling to the checkpoint.

        Args:
            checkpoint: Loaded checkpoint dictionary (modified in-place)
            **kwargs: Rescaler-specific parameters

        Returns:
            RescaleResult with details of changes made
        """
        pass

    def should_apply(self, diagnostic: DiagnosticResult) -> bool:
        """Determine if rescaling should be applied based on diagnosis."""
        return diagnostic.status in ("warning", "critical")


# =============================================================================
# Concrete Rescalers
# =============================================================================


class GradientEMARescaler(CheckpointRescaler):
    """
    Rescales the gradient norm EMA in trainer_state.

    High grad_norm_ema causes dynamic clipping to be too permissive,
    allowing large gradients that destabilize training.
    """

    # Thresholds for diagnosis
    HEALTHY_MAX = 50.0
    WARNING_MAX = 200.0
    DEFAULT_TARGET = 5.0

    def __init__(self):
        super().__init__("gradient_ema")

    def diagnose(self, checkpoint: dict[str, Any]) -> DiagnosticResult:
        trainer_state = checkpoint.get("trainer_state", {})
        grad_ema = trainer_state.get("grad_norm_ema", 0.0)

        if grad_ema <= self.HEALTHY_MAX:
            return DiagnosticResult(
                component="grad_norm_ema",
                status="healthy",
                current_value=grad_ema,
                message=f"Gradient EMA {grad_ema:.2f} is within healthy range",
            )
        elif grad_ema <= self.WARNING_MAX:
            return DiagnosticResult(
                component="grad_norm_ema",
                status="warning",
                current_value=grad_ema,
                recommended_value=self.DEFAULT_TARGET,
                message=f"Gradient EMA {grad_ema:.2f} is elevated (threshold: {self.HEALTHY_MAX})",
            )
        else:
            return DiagnosticResult(
                component="grad_norm_ema",
                status="critical",
                current_value=grad_ema,
                recommended_value=self.DEFAULT_TARGET,
                message=f"Gradient EMA {grad_ema:.2f} is critically high - will cause instability",
            )

    def rescale(
        self, checkpoint: dict[str, Any], target: float | None = None, **kwargs: Any
    ) -> RescaleResult:
        target = target if target is not None else self.DEFAULT_TARGET
        trainer_state = checkpoint.get("trainer_state", {})
        old_value = trainer_state.get("grad_norm_ema", 0.0)

        if "trainer_state" not in checkpoint:
            checkpoint["trainer_state"] = {}

        checkpoint["trainer_state"]["grad_norm_ema"] = target

        return RescaleResult(
            rescaler_name=self.name,
            success=True,
            changes_made={"grad_norm_ema": {"old": old_value, "new": target}},
            message=f"Reset grad_norm_ema: {old_value:.2f} -> {target:.2f}",
        )


class OptimizerMomentumRescaler(CheckpointRescaler):
    """
    Rescales optimizer momentum states (exp_avg for standard Adam,
    absmax for 8-bit Adam).

    Inflated momentum can perpetuate gradient instability even after
    the underlying cause is fixed.
    """

    # Thresholds for 8-bit Adam absmax values
    ABSMAX1_HEALTHY = 0.01  # exp_avg scale
    ABSMAX2_HEALTHY = 0.001  # exp_avg_sq scale
    ABSMAX1_WARNING = 0.1
    ABSMAX2_WARNING = 0.01

    def __init__(self):
        super().__init__("optimizer_momentum")

    def _is_8bit_adam(self, optimizer_state: dict[str, Any]) -> bool:
        """Check if optimizer uses 8-bit Adam format."""
        state = optimizer_state.get("state", {})
        if not state:
            return False
        first_param = next(iter(state.values()), {})
        return "absmax1" in first_param

    def _analyze_8bit_state(
        self, optimizer_state: dict[str, Any]
    ) -> tuple[float, float, int, int]:
        """Analyze 8-bit Adam state, return (max_absmax1, max_absmax2, count1, count2)."""
        state = optimizer_state.get("state", {})
        max_am1, max_am2 = 0.0, 0.0
        high_am1_count, high_am2_count = 0, 0

        for param_state in state.values():
            if "absmax1" in param_state:
                am1 = param_state["absmax1"]
                am1_max = float(am1.max().item()) if am1.numel() > 0 else 0.0
                max_am1 = max(max_am1, am1_max)
                if am1_max > self.ABSMAX1_WARNING:
                    high_am1_count += 1

            if "absmax2" in param_state:
                am2 = param_state["absmax2"]
                am2_max = float(am2.max().item()) if am2.numel() > 0 else 0.0
                max_am2 = max(max_am2, am2_max)
                if am2_max > self.ABSMAX2_WARNING:
                    high_am2_count += 1

        return max_am1, max_am2, high_am1_count, high_am2_count

    def diagnose(self, checkpoint: dict[str, Any]) -> DiagnosticResult:
        optimizer = checkpoint.get("optimizer", {})
        if not optimizer:
            return DiagnosticResult(
                component="optimizer_momentum",
                status="healthy",
                current_value=None,
                message="No optimizer state found",
            )

        if self._is_8bit_adam(optimizer):
            max_am1, max_am2, high1, high2 = self._analyze_8bit_state(optimizer)
            current = {"max_absmax1": max_am1, "max_absmax2": max_am2}

            if max_am1 <= self.ABSMAX1_HEALTHY and max_am2 <= self.ABSMAX2_HEALTHY:
                return DiagnosticResult(
                    component="optimizer_momentum",
                    status="healthy",
                    current_value=current,
                    message=f"8-bit Adam state healthy (absmax1={max_am1:.6f}, absmax2={max_am2:.6f})",
                )
            elif max_am1 <= self.ABSMAX1_WARNING and max_am2 <= self.ABSMAX2_WARNING:
                return DiagnosticResult(
                    component="optimizer_momentum",
                    status="warning",
                    current_value=current,
                    recommended_value=0.5,
                    message=f"8-bit Adam momentum slightly elevated ({high1} params > threshold)",
                )
            else:
                return DiagnosticResult(
                    component="optimizer_momentum",
                    status="critical",
                    current_value=current,
                    recommended_value=0.1,
                    message=f"8-bit Adam momentum critically high (absmax1={max_am1:.4f})",
                )
        else:
            # Standard Adam - check exp_avg directly
            return DiagnosticResult(
                component="optimizer_momentum",
                status="healthy",
                current_value=None,
                message="Standard Adam optimizer (momentum check not implemented)",
            )

    def rescale(
        self, checkpoint: dict[str, Any], scale: float = 0.5, **kwargs: Any
    ) -> RescaleResult:
        optimizer = checkpoint.get("optimizer", {})
        if not optimizer or not self._is_8bit_adam(optimizer):
            return RescaleResult(
                rescaler_name=self.name,
                success=False,
                message="No 8-bit Adam state to rescale",
            )

        state = optimizer.get("state", {})
        scaled_count = 0

        for param_state in state.values():
            if "absmax1" in param_state:
                param_state["absmax1"] = param_state["absmax1"] * scale
                scaled_count += 1
            if "absmax2" in param_state:
                param_state["absmax2"] = param_state["absmax2"] * scale

        return RescaleResult(
            rescaler_name=self.name,
            success=True,
            changes_made={"params_scaled": scaled_count, "scale_factor": scale},
            message=f"Scaled {scaled_count} param momentum states by {scale}",
        )


class GradScalerRescaler(CheckpointRescaler):
    """
    Resets or adjusts the GradScaler state for mixed-precision training.

    Corrupted scaler state can cause NaN gradients or improper scaling.
    """

    DEFAULT_SCALE = 65536.0  # 2^16, standard initial scale

    def __init__(self):
        super().__init__("grad_scaler")

    def diagnose(self, checkpoint: dict[str, Any]) -> DiagnosticResult:
        scaler = checkpoint.get("scaler", {})

        if not scaler:
            return DiagnosticResult(
                component="grad_scaler",
                status="healthy",
                current_value=None,
                message="No GradScaler state (likely disabled or empty)",
            )

        scale = scaler.get("scale", self.DEFAULT_SCALE)
        growth_tracker = scaler.get("_growth_tracker", 0)

        # Check for problematic states
        if scale < 1.0 or scale > 2**24:
            return DiagnosticResult(
                component="grad_scaler",
                status="critical",
                current_value={"scale": scale, "growth_tracker": growth_tracker},
                recommended_value=self.DEFAULT_SCALE,
                message=f"GradScaler scale {scale} is out of range",
            )

        return DiagnosticResult(
            component="grad_scaler",
            status="healthy",
            current_value={"scale": scale, "growth_tracker": growth_tracker},
            message=f"GradScaler healthy (scale={scale:.0f})",
        )

    def rescale(
        self, checkpoint: dict[str, Any], scale: float | None = None, **kwargs: Any
    ) -> RescaleResult:
        scale = scale if scale is not None else self.DEFAULT_SCALE
        old_scaler = checkpoint.get("scaler", {})

        # Reset to fresh state
        checkpoint["scaler"] = {
            "scale": scale,
            "_growth_tracker": 0,
            "_backoff_factor": 0.5,
            "_growth_factor": 2.0,
            "_growth_interval": 2000,
        }

        return RescaleResult(
            rescaler_name=self.name,
            success=True,
            changes_made={"old_scale": old_scaler.get("scale"), "new_scale": scale},
            message=f"Reset GradScaler to scale={scale:.0f}",
        )


class CEEMARescaler(CheckpointRescaler):
    """
    Rescales the cross-entropy EMA used for loss spike detection.

    Inflated CE EMA can mask actual loss spikes or cause false alarms.
    """

    HEALTHY_MAX = 10.0
    DEFAULT_TARGET = 3.0

    def __init__(self):
        super().__init__("ce_ema")

    def diagnose(self, checkpoint: dict[str, Any]) -> DiagnosticResult:
        trainer_state = checkpoint.get("trainer_state", {})
        ce_ema = trainer_state.get("ce_ema", 0.0)

        if ce_ema <= self.HEALTHY_MAX:
            return DiagnosticResult(
                component="ce_ema",
                status="healthy",
                current_value=ce_ema,
                message=f"CE EMA {ce_ema:.4f} is within normal range",
            )
        else:
            return DiagnosticResult(
                component="ce_ema",
                status="warning",
                current_value=ce_ema,
                recommended_value=self.DEFAULT_TARGET,
                message=f"CE EMA {ce_ema:.4f} is elevated",
            )

    def rescale(
        self, checkpoint: dict[str, Any], target: float | None = None, **kwargs: Any
    ) -> RescaleResult:
        target = target if target is not None else self.DEFAULT_TARGET
        trainer_state = checkpoint.get("trainer_state", {})
        old_value = trainer_state.get("ce_ema", 0.0)

        if "trainer_state" not in checkpoint:
            checkpoint["trainer_state"] = {}

        checkpoint["trainer_state"]["ce_ema"] = target

        return RescaleResult(
            rescaler_name=self.name,
            success=True,
            changes_made={"ce_ema": {"old": old_value, "new": target}},
            message=f"Reset ce_ema: {old_value:.4f} -> {target:.4f}",
        )


# =============================================================================
# Composite Rescaler
# =============================================================================


class CompositeRescaler(CheckpointRescaler):
    """
    Chains multiple rescalers together for comprehensive checkpoint repair.
    """

    def __init__(self, rescalers: list[CheckpointRescaler] | None = None):
        super().__init__("composite")
        self.rescalers = rescalers or []

    def add_rescaler(self, rescaler: CheckpointRescaler) -> "CompositeRescaler":
        """Add a rescaler to the chain."""
        self.rescalers.append(rescaler)
        return self

    def diagnose(self, checkpoint: dict[str, Any]) -> DiagnosticResult:
        """Run all rescaler diagnostics and return aggregate result."""
        results = [r.diagnose(checkpoint) for r in self.rescalers]

        critical = [r for r in results if r.status == "critical"]
        warnings = [r for r in results if r.status == "warning"]

        if critical:
            return DiagnosticResult(
                component="composite",
                status="critical",
                current_value=results,
                message=f"{len(critical)} critical issues, {len(warnings)} warnings",
            )
        elif warnings:
            return DiagnosticResult(
                component="composite",
                status="warning",
                current_value=results,
                message=f"{len(warnings)} warnings found",
            )
        else:
            return DiagnosticResult(
                component="composite",
                status="healthy",
                current_value=results,
                message="All components healthy",
            )

    def rescale(
        self, checkpoint: dict[str, Any], **kwargs: Any
    ) -> RescaleResult:
        """Apply all rescalers in sequence."""
        all_changes: dict[str, Any] = {}
        messages = []

        for rescaler in self.rescalers:
            result = rescaler.rescale(checkpoint, **kwargs)
            if result.success:
                all_changes[rescaler.name] = result.changes_made
                messages.append(result.message)

        return RescaleResult(
            rescaler_name=self.name,
            success=True,
            changes_made=all_changes,
            message="; ".join(messages),
        )


# =============================================================================
# Checkpoint Doctor
# =============================================================================


class CheckpointDoctor:
    """
    High-level utility for diagnosing and fixing checkpoint issues.

    Automatically selects appropriate rescalers based on diagnosis.
    """

    def __init__(self, checkpoint_path: str | Path):
        self.checkpoint_path = Path(checkpoint_path)
        self.checkpoint: dict[str, Any] | None = None
        self.diagnostics: list[DiagnosticResult] = []

        # Initialize all available rescalers
        self.available_rescalers = [
            GradientEMARescaler(),
            OptimizerMomentumRescaler(),
            GradScalerRescaler(),
            CEEMARescaler(),
        ]

    def load(self) -> "CheckpointDoctor":
        """Load the checkpoint."""
        logger.info(f"Loading checkpoint: {self.checkpoint_path}")
        self.checkpoint = torch.load(
            self.checkpoint_path, map_location="cpu", weights_only=False
        )
        logger.info(f"  Step: {self.checkpoint.get('step', 'unknown')}")
        return self

    def diagnose(self) -> list[DiagnosticResult]:
        """Run all diagnostics and return results."""
        if self.checkpoint is None:
            raise ValueError("Checkpoint not loaded. Call load() first.")

        self.diagnostics = []
        logger.info("\nRunning diagnostics...")
        logger.info("-" * 60)

        for rescaler in self.available_rescalers:
            result = rescaler.diagnose(self.checkpoint)
            self.diagnostics.append(result)

            status_icon = {
                "healthy": "\u2714",
                "warning": "\u26a0\ufe0f",
                "critical": "\u274c",
            }.get(result.status, "?")

            logger.info(f"  {status_icon} {result.component}: {result.message}")
            if result.recommended_value is not None:
                logger.info(f"      Recommended: {result.recommended_value}")

        return self.diagnostics

    def create_backup(self) -> Path:
        """Create a backup of the original checkpoint."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.checkpoint_path.with_suffix(f".backup_{timestamp}.pt")
        logger.info(f"\nCreating backup: {backup_path}")
        shutil.copy2(self.checkpoint_path, backup_path)
        return backup_path

    def apply_fixes(
        self,
        grad_ema_target: float | None = None,
        momentum_scale: float | None = None,
        reset_scaler: bool = False,
        ce_ema_target: float | None = None,
        auto: bool = False,
    ) -> list[RescaleResult]:
        """
        Apply specified fixes to the checkpoint.

        Args:
            grad_ema_target: Target value for grad_norm_ema
            momentum_scale: Scale factor for optimizer momentum
            reset_scaler: Whether to reset GradScaler
            ce_ema_target: Target value for ce_ema
            auto: Automatically fix based on diagnosis
        """
        if self.checkpoint is None:
            raise ValueError("Checkpoint not loaded. Call load() first.")

        results = []

        if auto:
            # Apply fixes based on diagnostics
            for diag in self.diagnostics:
                if diag.status in ("warning", "critical"):
                    if diag.component == "grad_norm_ema":
                        grad_ema_target = grad_ema_target or diag.recommended_value
                    elif diag.component == "optimizer_momentum":
                        momentum_scale = momentum_scale or diag.recommended_value
                    elif diag.component == "grad_scaler":
                        reset_scaler = True
                    elif diag.component == "ce_ema":
                        ce_ema_target = ce_ema_target or diag.recommended_value

        # Apply gradient EMA fix
        if grad_ema_target is not None:
            rescaler = GradientEMARescaler()
            result = rescaler.rescale(self.checkpoint, target=grad_ema_target)
            results.append(result)
            logger.info(f"  {result.message}")

        # Apply momentum scaling
        if momentum_scale is not None:
            rescaler = OptimizerMomentumRescaler()
            result = rescaler.rescale(self.checkpoint, scale=momentum_scale)
            results.append(result)
            logger.info(f"  {result.message}")

        # Reset GradScaler
        if reset_scaler:
            rescaler = GradScalerRescaler()
            result = rescaler.rescale(self.checkpoint)
            results.append(result)
            logger.info(f"  {result.message}")

        # Apply CE EMA fix
        if ce_ema_target is not None:
            rescaler = CEEMARescaler()
            result = rescaler.rescale(self.checkpoint, target=ce_ema_target)
            results.append(result)
            logger.info(f"  {result.message}")

        return results

    def save(self, output_path: str | Path | None = None) -> Path:
        """Save the modified checkpoint."""
        if self.checkpoint is None:
            raise ValueError("Checkpoint not loaded. Call load() first.")

        output_path = Path(output_path) if output_path else self.checkpoint_path
        logger.info(f"\nSaving checkpoint: {output_path}")
        torch.save(self.checkpoint, output_path)
        return output_path

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of checkpoint state."""
        if self.checkpoint is None:
            raise ValueError("Checkpoint not loaded. Call load() first.")

        trainer_state = self.checkpoint.get("trainer_state", {})
        metrics = self.checkpoint.get("metrics", {})

        return {
            "step": self.checkpoint.get("step"),
            "grad_norm_ema": trainer_state.get("grad_norm_ema"),
            "ce_ema": trainer_state.get("ce_ema"),
            "best_loss": metrics.get("best_loss"),
            "current_loss": metrics.get("current_loss"),
            "has_optimizer": "optimizer" in self.checkpoint,
            "has_scaler": bool(self.checkpoint.get("scaler")),
        }


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Checkpoint Rescaling Utility for HYDRA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Diagnose checkpoint issues
  python scripts/checkpoint_rescaler.py checkpoints/model.pt --diagnose

  # Apply automatic fixes based on diagnosis
  python scripts/checkpoint_rescaler.py checkpoints/model.pt --fix

  # Custom gradient EMA reset
  python scripts/checkpoint_rescaler.py checkpoints/model.pt --grad-ema-target 5.0

  # Full reset (gradient EMA + momentum scaling)
  python scripts/checkpoint_rescaler.py checkpoints/model.pt \\
      --grad-ema-target 5.0 --momentum-scale 0.5 --reset-scaler
        """,
    )

    parser.add_argument("checkpoint", type=str, help="Path to checkpoint file")
    parser.add_argument(
        "--diagnose", action="store_true", help="Only diagnose, don't modify"
    )
    parser.add_argument(
        "--fix", action="store_true", help="Apply automatic fixes based on diagnosis"
    )
    parser.add_argument(
        "--grad-ema-target",
        type=float,
        default=None,
        help="Target value for grad_norm_ema (default: 5.0 if fixing)",
    )
    parser.add_argument(
        "--momentum-scale",
        type=float,
        default=None,
        help="Scale factor for optimizer momentum (e.g., 0.5 to halve)",
    )
    parser.add_argument(
        "--reset-scaler", action="store_true", help="Reset GradScaler to default state"
    )
    parser.add_argument(
        "--ce-ema-target",
        type=float,
        default=None,
        help="Target value for ce_ema",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output path (default: overwrite original after backup)",
    )
    parser.add_argument(
        "--no-backup", action="store_true", help="Skip creating backup (dangerous!)"
    )

    args = parser.parse_args()

    # Create doctor and load checkpoint
    doctor = CheckpointDoctor(args.checkpoint)
    doctor.load()

    # Show summary
    summary = doctor.get_summary()
    print("\n" + "=" * 60)
    print("CHECKPOINT SUMMARY")
    print("=" * 60)
    for key, value in summary.items():
        print(f"  {key}: {value}")

    # Run diagnostics
    doctor.diagnose()

    # If only diagnosing, stop here
    if args.diagnose:
        return

    # Check if any fixes are requested
    has_fixes = (
        args.fix
        or args.grad_ema_target is not None
        or args.momentum_scale is not None
        or args.reset_scaler
        or args.ce_ema_target is not None
    )

    if not has_fixes:
        print("\nNo fixes requested. Use --fix for automatic fixes or specify targets.")
        return

    # Create backup
    if not args.no_backup:
        backup_path = doctor.create_backup()
        print(f"  Backup created: {backup_path}")

    # Apply fixes
    print("\n" + "=" * 60)
    print("APPLYING FIXES")
    print("=" * 60)

    doctor.apply_fixes(
        grad_ema_target=args.grad_ema_target,
        momentum_scale=args.momentum_scale,
        reset_scaler=args.reset_scaler,
        ce_ema_target=args.ce_ema_target,
        auto=args.fix,
    )

    # Save
    output_path = doctor.save(args.output)

    # Verify
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    doctor.load()  # Reload to verify
    doctor.diagnose()

    print("\n" + "=" * 60)
    print(f"Done! Modified checkpoint saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
