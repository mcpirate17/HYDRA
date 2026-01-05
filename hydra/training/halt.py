"""
HaltController: owns halt-policy env vars, EMA window tracking, and halt decision logic.

The trainer delegates all halt decisions to this controller.
"""
from __future__ import annotations

import math
import os
from collections import deque
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class HaltConfig:
    """Configuration for halt thresholds, parsed from environment."""
    ema_window: int
    ema_delta: float
    ema_abs: float
    spike_window: int
    spike_count: int

    @classmethod
    def from_env(cls) -> "HaltConfig":
        """Parse halt configuration from HYDRA_HALT_* environment variables."""
        return cls(
            ema_window=int(os.environ.get("HYDRA_HALT_EMA_WINDOW", "200") or 200),
            ema_delta=float(os.environ.get("HYDRA_HALT_EMA_DELTA", "0.5") or 0.5),
            ema_abs=float(os.environ.get("HYDRA_HALT_EMA_ABS", "8.0") or 8.0),
            spike_window=int(os.environ.get("HYDRA_HALT_SPIKE_WINDOW", "200") or 200),
            spike_count=int(os.environ.get("HYDRA_HALT_SPIKE_COUNT", "10") or 10),
        )


class HaltController:
    """
    Tracks EMA loss history and decides when to halt training.

    Halt conditions:
    - NaN/Inf loss (immediate halt)
    - Sustained EMA degradation: EMA increased by more than `ema_delta` over 
      `ema_window` steps AND current EMA is above `ema_abs` threshold
    
    Usage:
        controller = HaltController.from_env()
        # In training loop:
        halt, reason = controller.update(step, ema_loss, spikes_in_window)
        if halt:
            save_checkpoint_and_exit(reason)
    """

    def __init__(self, config: HaltConfig, debug: bool = False):
        self.config = config
        self.debug = debug
        # maxlen = window + 1 so we can compute delta over exactly `window` steps
        self._ema_hist: deque[float] = deque(maxlen=max(2, config.ema_window + 1))

    @classmethod
    def from_env(cls, debug: bool = False) -> "HaltController":
        """Create HaltController with config parsed from environment."""
        return cls(HaltConfig.from_env(), debug=debug)

    def update(
        self,
        step: int,
        ema_loss: float,
        spikes_in_window: int = 0,
        *,
        logger: Optional[object] = None,
    ) -> Tuple[bool, str]:
        """
        Update EMA history and check halt conditions.

        Args:
            step: Current training step
            ema_loss: Current EMA loss value
            spikes_in_window: Number of gradient spikes in recent window (from SpikeTracker)
            logger: Optional logger for debug output

        Returns:
            (halt, reason): halt=True means training should stop, reason explains why
        """
        self._ema_hist.append(float(ema_loss))
        
        # Debug logging
        if self.debug and logger is not None and step % 25 == 0:
            if len(self._ema_hist) >= 2:
                oldest = self._ema_hist[0]
                newest = self._ema_hist[-1]
                delta = newest - oldest
                ema_is_high = newest > self.config.ema_abs
                logger.info(
                    f"  [EMA_DEBUG] halt_check: window_len={len(self._ema_hist)} "
                    f"oldest={oldest:.4f} newest={newest:.4f} "
                    f"delta={delta:.4f} (thr={self.config.ema_delta}) "
                    f"ema_high={ema_is_high} (thr={self.config.ema_abs})"
                )

        # Check for EMA degradation (only when window is full)
        if len(self._ema_hist) >= self._ema_hist.maxlen:
            ema_delta = float(self._ema_hist[-1]) - float(self._ema_hist[0])
            current_ema = float(self._ema_hist[-1])

            # Only halt if:
            # 1. EMA increased by more than threshold over window, AND
            # 2. Current EMA is above the absolute threshold (model is actually struggling)
            # This prevents false halts during normal variance when loss is still reasonable
            delta_exceeded = math.isfinite(ema_delta) and (ema_delta > self.config.ema_delta)
            ema_is_high = current_ema > self.config.ema_abs

            if delta_exceeded and ema_is_high:
                # Include spike info in halt reason if spikes contributed
                if spikes_in_window >= self.config.spike_count:
                    reason = (
                        f"spike-rate explosion ({spikes_in_window}/{self.config.spike_window}) + "
                        f"EMA blow-up (Δ={ema_delta:.3f} over {self.config.ema_window})"
                    )
                else:
                    reason = (
                        f"sustained EMA degradation "
                        f"(Δ={ema_delta:.3f} over {self.config.ema_window} steps)"
                    )
                return True, reason

        return False, ""

    def check_loss_finite(self, loss: float) -> Tuple[bool, str]:
        """
        Check if loss is finite.

        Args:
            loss: Current loss value

        Returns:
            (halt, reason): halt=True if loss is NaN/Inf
        """
        try:
            if not math.isfinite(float(loss)):
                return True, "NaN/Inf loss"
        except Exception:
            return True, "NaN/Inf loss"
        return False, ""

    @property
    def spike_window(self) -> int:
        """Return spike window size (for SpikeTracker initialization)."""
        return self.config.spike_window

    @property
    def spike_count_threshold(self) -> int:
        """Return spike count threshold for halt messages."""
        return self.config.spike_count
