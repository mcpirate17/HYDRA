"""Safe Optimizations Wrapper for HYDRA Training.

Provides runtime monitoring and automatic fallback for experimental optimizations.
Monitors for anomalies (loss spikes, NaN/Inf gradients, throughput drops) and
auto-disables failing optimizations within a safety window.

Usage:
    from hydra.training.safe_optimizations import SafeOptimizations, OptimizationConfig

    opt_config = OptimizationConfig(
        enable_fa3=True,
        enable_cuda_graphs=True,
        enable_blackwell_tuning=True,
    )
    safe_opts = SafeOptimizations(opt_config)

    # In training loop:
    if safe_opts.should_use_cuda_graphs():
        # Use CUDA graphs
        pass

    # After each step:
    safe_opts.record_step(loss=loss, grad_norm=grad_norm, step_time=elapsed)
"""
from __future__ import annotations

import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Tuple

import torch

_log = logging.getLogger(__name__)


class OptimizationStatus(Enum):
    """Status of an optimization."""
    DISABLED = auto()        # User disabled or not supported
    PRETESTING = auto()      # Running initial validation
    ENABLED = auto()         # Active and healthy
    MONITORING = auto()      # Enabled but in safety window
    FAILED = auto()          # Auto-disabled due to anomaly
    FALLBACK = auto()        # Using fallback implementation


@dataclass
class OptimizationConfig:
    """Configuration for experimental optimizations.

    All optimizations default to True (enabled) but will auto-disable
    if they cause anomalies during the safety window.
    """
    # Flash Attention 3 (Blackwell/Hopper optimized)
    enable_fa3: bool = True
    fa3_fallback_to_fa2: bool = True  # Fall back to FA2 if FA3 fails

    # CUDA Graphs for reduced launch overhead
    enable_cuda_graphs: bool = True
    cuda_graphs_warmup_steps: int = 50  # Steps before graph capture
    cuda_graphs_capture_steps: int = 10  # Steps to use for capture

    # Blackwell-specific Triton tuning
    enable_blackwell_tuning: bool = True
    triton_block_size_q: int = 128  # Query block size
    triton_block_size_kv: int = 64  # KV block size
    triton_num_warps: int = 8  # Warp count for Blackwell
    triton_num_stages: int = 3  # Pipeline stages

    # Multi-threaded data prefetch
    enable_prefetch_threads: int = 4  # 0 = disabled
    prefetch_buffer_size: int = 8  # Batches to prefetch

    # FP8 support (Blackwell native)
    enable_fp8: bool = False  # Conservative default - experimental
    fp8_format: str = "e4m3"  # e4m3 or e5m2

    # Safety monitoring
    safety_window_steps: int = 100  # Steps to monitor after enabling
    loss_spike_threshold: float = 2.0  # Disable if loss > threshold * EMA
    throughput_drop_threshold: float = 0.5  # Disable if throughput < threshold * EMA

    # Pretest settings
    pretest_steps: int = 10  # Steps to run for pretest
    pretest_batch_size: int = 2  # Small batch for quick pretest


@dataclass
class OptimizationState:
    """Runtime state for a single optimization."""
    name: str
    status: OptimizationStatus = OptimizationStatus.DISABLED
    enabled_at_step: int = 0
    disabled_at_step: int = 0
    disable_reason: str = ""
    anomaly_count: int = 0
    pretest_passed: bool = False

    def is_active(self) -> bool:
        """Check if optimization is currently active (enabled or monitoring)."""
        return self.status in (OptimizationStatus.ENABLED, OptimizationStatus.MONITORING)


@dataclass
class AnomalyEvent:
    """Record of a detected anomaly."""
    step: int
    optimization: str
    anomaly_type: str  # "loss_spike", "nan_grad", "inf_grad", "throughput_drop"
    value: float
    threshold: float
    timestamp: float = field(default_factory=time.time)


class SafeOptimizations:
    """Safe wrapper for experimental optimizations with auto-fallback.

    Monitors training metrics and automatically disables optimizations
    that cause anomalies within the safety window.
    """

    def __init__(
        self,
        config: OptimizationConfig,
        device: str = "cuda",
        logger: Optional[logging.Logger] = None,
    ):
        self.config = config
        self.device = device
        self.logger = logger or _log

        # Initialize optimization states
        self._states: Dict[str, OptimizationState] = {
            "fa3": OptimizationState("fa3"),
            "cuda_graphs": OptimizationState("cuda_graphs"),
            "blackwell_tuning": OptimizationState("blackwell_tuning"),
            "prefetch_threads": OptimizationState("prefetch_threads"),
            "fp8": OptimizationState("fp8"),
        }

        # Monitoring state
        self._current_step = 0
        self._loss_history: deque = deque(maxlen=100)
        self._throughput_history: deque = deque(maxlen=100)
        self._loss_ema = 0.0
        self._throughput_ema = 0.0
        self._ema_alpha = 0.1  # EMA smoothing factor

        # Anomaly tracking
        self._anomalies: List[AnomalyEvent] = []
        self._max_anomalies_per_opt = 3  # Disable after this many anomalies

        # Hardware detection
        self._gpu_arch = self._detect_gpu_arch()
        self._has_fa3 = self._check_fa3_available()
        self._has_cuda_graphs = self._check_cuda_graphs_available()

        # Initialize based on config and hardware
        self._initialize_optimizations()

    def _detect_gpu_arch(self) -> str:
        """Detect GPU architecture for optimization selection."""
        if not torch.cuda.is_available():
            return "cpu"

        capability = torch.cuda.get_device_capability(0)
        major, minor = capability

        if major >= 10:
            return "blackwell"  # SM100+
        elif major >= 9:
            return "hopper"  # SM90
        elif major >= 8:
            if minor >= 9:
                return "ada"  # SM89 (RTX 40xx)
            return "ampere"  # SM80-86
        return "older"

    def _check_fa3_available(self) -> bool:
        """Check if Flash Attention 3 is available."""
        try:
            # FA3 requires Hopper+ and specific package
            if self._gpu_arch not in ("blackwell", "hopper"):
                return False

            # Try importing flash_attn with FA3 support
            try:
                from flash_attn import flash_attn_func
                # Check version for FA3 support
                import flash_attn
                version = getattr(flash_attn, "__version__", "0.0.0")
                major = int(version.split(".")[0])
                return major >= 3
            except ImportError:
                return False
        except Exception:
            return False

    def _check_cuda_graphs_available(self) -> bool:
        """Check if CUDA graphs are supported."""
        if not torch.cuda.is_available():
            return False
        # CUDA graphs available on all modern GPUs
        return True

    def _initialize_optimizations(self) -> None:
        """Initialize optimization states based on config and hardware."""
        cfg = self.config

        # FA3: Only on Hopper+ with package available
        if cfg.enable_fa3 and self._has_fa3:
            self._states["fa3"].status = OptimizationStatus.PRETESTING
            self.logger.info(f"FA3: Enabled for pretest (GPU arch: {self._gpu_arch})")
        else:
            reason = "not available" if not self._has_fa3 else "disabled by config"
            self._states["fa3"].status = OptimizationStatus.DISABLED
            self._states["fa3"].disable_reason = reason
            self.logger.info(f"FA3: Disabled ({reason})")

        # CUDA Graphs
        if cfg.enable_cuda_graphs and self._has_cuda_graphs:
            self._states["cuda_graphs"].status = OptimizationStatus.PRETESTING
            self.logger.info("CUDA Graphs: Enabled for pretest")
        else:
            self._states["cuda_graphs"].status = OptimizationStatus.DISABLED

        # Blackwell tuning: Only on Blackwell
        if cfg.enable_blackwell_tuning and self._gpu_arch == "blackwell":
            self._states["blackwell_tuning"].status = OptimizationStatus.PRETESTING
            self.logger.info("Blackwell Triton tuning: Enabled for pretest")
        else:
            self._states["blackwell_tuning"].status = OptimizationStatus.DISABLED
            if cfg.enable_blackwell_tuning and self._gpu_arch != "blackwell":
                self.logger.info(f"Blackwell tuning: Disabled (GPU arch: {self._gpu_arch})")

        # Prefetch threads
        if cfg.enable_prefetch_threads > 0:
            self._states["prefetch_threads"].status = OptimizationStatus.ENABLED
            self._states["prefetch_threads"].pretest_passed = True  # No pretest needed
            self.logger.info(f"Prefetch threads: {cfg.enable_prefetch_threads} threads")

        # FP8: Conservative, requires explicit enable
        if cfg.enable_fp8 and self._gpu_arch in ("blackwell", "hopper", "ada"):
            self._states["fp8"].status = OptimizationStatus.PRETESTING
            self.logger.info(f"FP8 ({cfg.fp8_format}): Enabled for pretest")
        else:
            self._states["fp8"].status = OptimizationStatus.DISABLED

    def run_pretests(
        self,
        model: torch.nn.Module,
        sample_batch: torch.Tensor,
    ) -> Dict[str, bool]:
        """Run pretests for all pending optimizations.

        Args:
            model: The model to test with
            sample_batch: A sample input batch for testing

        Returns:
            Dict mapping optimization name to pretest success
        """
        results = {}

        for name, state in self._states.items():
            if state.status != OptimizationStatus.PRETESTING:
                continue

            self.logger.info(f"Running pretest for {name}...")
            try:
                passed = self._run_single_pretest(name, model, sample_batch)
                results[name] = passed

                if passed:
                    state.status = OptimizationStatus.MONITORING
                    state.pretest_passed = True
                    state.enabled_at_step = self._current_step
                    self.logger.info(f"  {name}: PASSED - enabling with monitoring")
                else:
                    state.status = OptimizationStatus.FAILED
                    state.disable_reason = "pretest failed"
                    self.logger.warning(f"  {name}: FAILED - disabling")

            except Exception as e:
                results[name] = False
                state.status = OptimizationStatus.FAILED
                state.disable_reason = f"pretest exception: {str(e)[:100]}"
                self.logger.error(f"  {name}: EXCEPTION - {e}")

        return results

    def _run_single_pretest(
        self,
        name: str,
        model: torch.nn.Module,
        sample_batch: torch.Tensor,
    ) -> bool:
        """Run pretest for a single optimization."""
        cfg = self.config

        if name == "fa3":
            return self._pretest_fa3(model, sample_batch)
        elif name == "cuda_graphs":
            return self._pretest_cuda_graphs(model, sample_batch)
        elif name == "blackwell_tuning":
            return self._pretest_blackwell_tuning(model, sample_batch)
        elif name == "fp8":
            return self._pretest_fp8(model, sample_batch)

        return True  # Default pass for unknown optimizations

    def _pretest_fa3(
        self,
        model: torch.nn.Module,
        sample_batch: torch.Tensor,
    ) -> bool:
        """Pretest Flash Attention 3."""
        try:
            # Run a few forward passes and check for NaN/Inf
            model.eval()
            with torch.no_grad():
                for _ in range(self.config.pretest_steps):
                    output = model(sample_batch)
                    if torch.isnan(output).any() or torch.isinf(output).any():
                        return False
            model.train()
            return True
        except Exception as e:
            self.logger.warning(f"FA3 pretest exception: {e}")
            return False

    def _pretest_cuda_graphs(
        self,
        model: torch.nn.Module,
        sample_batch: torch.Tensor,
    ) -> bool:
        """Pretest CUDA graph capture."""
        try:
            # Test graph capture
            model.eval()

            # Warmup
            with torch.no_grad():
                _ = model(sample_batch)

            # Try capturing a graph
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                with torch.no_grad():
                    _ = model(sample_batch)

            # Replay test
            g.replay()
            torch.cuda.synchronize()

            model.train()
            return True
        except Exception as e:
            self.logger.warning(f"CUDA graphs pretest exception: {e}")
            return False

    def _pretest_blackwell_tuning(
        self,
        model: torch.nn.Module,
        sample_batch: torch.Tensor,
    ) -> bool:
        """Pretest Blackwell-specific Triton configs."""
        # Blackwell tuning is config-based, just verify kernels compile
        try:
            model.eval()
            with torch.no_grad():
                output = model(sample_batch)
                if torch.isnan(output).any() or torch.isinf(output).any():
                    return False
            model.train()
            return True
        except Exception as e:
            self.logger.warning(f"Blackwell tuning pretest exception: {e}")
            return False

    def _pretest_fp8(
        self,
        model: torch.nn.Module,
        sample_batch: torch.Tensor,
    ) -> bool:
        """Pretest FP8 computation."""
        try:
            # FP8 requires special handling
            if not hasattr(torch, "float8_e4m3fn"):
                return False

            # Basic FP8 cast test
            test_tensor = torch.randn(64, 64, device=self.device, dtype=torch.bfloat16)
            fp8_tensor = test_tensor.to(torch.float8_e4m3fn)
            back_tensor = fp8_tensor.to(torch.bfloat16)

            # Check roundtrip error is reasonable
            error = (test_tensor - back_tensor).abs().mean().item()
            return error < 0.1  # Reasonable FP8 quantization error
        except Exception as e:
            self.logger.warning(f"FP8 pretest exception: {e}")
            return False

    def record_step(
        self,
        step: int,
        loss: float,
        grad_norm: Optional[float] = None,
        step_time: Optional[float] = None,
        tokens_per_sec: Optional[float] = None,
    ) -> List[AnomalyEvent]:
        """Record metrics for a training step and check for anomalies.

        Args:
            step: Current training step
            loss: Loss value for this step
            grad_norm: Gradient norm (optional)
            step_time: Time for this step in seconds (optional)
            tokens_per_sec: Throughput (optional)

        Returns:
            List of any anomalies detected
        """
        self._current_step = step
        detected_anomalies = []

        # Update loss tracking
        self._loss_history.append(loss)
        if self._loss_ema == 0.0:
            self._loss_ema = loss
        else:
            self._loss_ema = (1 - self._ema_alpha) * self._loss_ema + self._ema_alpha * loss

        # Update throughput tracking
        if tokens_per_sec is not None:
            self._throughput_history.append(tokens_per_sec)
            if self._throughput_ema == 0.0:
                self._throughput_ema = tokens_per_sec
            else:
                self._throughput_ema = (
                    (1 - self._ema_alpha) * self._throughput_ema +
                    self._ema_alpha * tokens_per_sec
                )

        # Check for anomalies on active optimizations in safety window
        for name, state in self._states.items():
            if not state.is_active():
                continue

            # Check if in safety window
            steps_since_enable = step - state.enabled_at_step
            in_safety_window = steps_since_enable < self.config.safety_window_steps

            if not in_safety_window:
                # Graduated from monitoring to stable
                if state.status == OptimizationStatus.MONITORING:
                    state.status = OptimizationStatus.ENABLED
                    self.logger.info(f"{name}: Graduated from monitoring to stable")
                continue

            # Loss spike check
            if self._loss_ema > 0 and loss > self.config.loss_spike_threshold * self._loss_ema:
                anomaly = AnomalyEvent(
                    step=step,
                    optimization=name,
                    anomaly_type="loss_spike",
                    value=loss,
                    threshold=self.config.loss_spike_threshold * self._loss_ema,
                )
                detected_anomalies.append(anomaly)
                self._handle_anomaly(name, anomaly)

            # NaN/Inf gradient check
            if grad_norm is not None:
                if not torch.isfinite(torch.tensor(grad_norm)):
                    anomaly_type = "nan_grad" if grad_norm != grad_norm else "inf_grad"
                    anomaly = AnomalyEvent(
                        step=step,
                        optimization=name,
                        anomaly_type=anomaly_type,
                        value=grad_norm,
                        threshold=0.0,
                    )
                    detected_anomalies.append(anomaly)
                    self._handle_anomaly(name, anomaly)

            # Throughput drop check
            if tokens_per_sec is not None and self._throughput_ema > 0:
                if tokens_per_sec < self.config.throughput_drop_threshold * self._throughput_ema:
                    anomaly = AnomalyEvent(
                        step=step,
                        optimization=name,
                        anomaly_type="throughput_drop",
                        value=tokens_per_sec,
                        threshold=self.config.throughput_drop_threshold * self._throughput_ema,
                    )
                    detected_anomalies.append(anomaly)
                    self._handle_anomaly(name, anomaly)

        self._anomalies.extend(detected_anomalies)
        return detected_anomalies

    def _handle_anomaly(self, name: str, anomaly: AnomalyEvent) -> None:
        """Handle a detected anomaly."""
        state = self._states[name]
        state.anomaly_count += 1

        self.logger.warning(
            f"ANOMALY DETECTED for {name}: {anomaly.anomaly_type} "
            f"(value={anomaly.value:.4f}, threshold={anomaly.threshold:.4f}, "
            f"step={anomaly.step})"
        )

        if state.anomaly_count >= self._max_anomalies_per_opt:
            self._disable_optimization(name, f"too many anomalies ({state.anomaly_count})")

    def _disable_optimization(self, name: str, reason: str) -> None:
        """Disable an optimization and fall back."""
        state = self._states[name]
        state.status = OptimizationStatus.FALLBACK
        state.disabled_at_step = self._current_step
        state.disable_reason = reason

        self.logger.warning(
            f"DISABLING {name}: {reason} (was enabled at step {state.enabled_at_step})"
        )

        # Apply fallback logic
        if name == "fa3" and self.config.fa3_fallback_to_fa2:
            self.logger.info("  -> Falling back to Flash Attention 2")

    # ─────────────────────────────────────────────────────────────────────────
    # Query Methods for Training Code
    # ─────────────────────────────────────────────────────────────────────────

    def should_use_fa3(self) -> bool:
        """Check if FA3 should be used."""
        return self._states["fa3"].is_active()

    def should_use_cuda_graphs(self) -> bool:
        """Check if CUDA graphs should be used."""
        state = self._states["cuda_graphs"]
        if not state.is_active():
            return False
        # Wait for warmup period
        return self._current_step >= self.config.cuda_graphs_warmup_steps

    def should_use_blackwell_tuning(self) -> bool:
        """Check if Blackwell tuning should be used."""
        return self._states["blackwell_tuning"].is_active()

    def should_use_fp8(self) -> bool:
        """Check if FP8 should be used."""
        return self._states["fp8"].is_active()

    def get_prefetch_threads(self) -> int:
        """Get number of prefetch threads to use."""
        if self._states["prefetch_threads"].is_active():
            return self.config.enable_prefetch_threads
        return 0

    def get_triton_config(self) -> Dict[str, int]:
        """Get Triton kernel configuration."""
        if self.should_use_blackwell_tuning():
            return {
                "BLOCK_Q": self.config.triton_block_size_q,
                "BLOCK_KV": self.config.triton_block_size_kv,
                "num_warps": self.config.triton_num_warps,
                "num_stages": self.config.triton_num_stages,
            }
        # Default config for other GPUs
        return {
            "BLOCK_Q": 64,
            "BLOCK_KV": 64,
            "num_warps": 4,
            "num_stages": 2,
        }

    def get_status_summary(self) -> Dict[str, str]:
        """Get a summary of all optimization statuses."""
        return {
            name: state.status.name
            for name, state in self._states.items()
        }

    def get_anomaly_summary(self) -> Dict[str, int]:
        """Get count of anomalies by type."""
        summary: Dict[str, int] = {}
        for anomaly in self._anomalies:
            key = f"{anomaly.optimization}:{anomaly.anomaly_type}"
            summary[key] = summary.get(key, 0) + 1
        return summary

    def log_status(self) -> None:
        """Log current status of all optimizations."""
        self.logger.info("=" * 60)
        self.logger.info("SAFE OPTIMIZATIONS STATUS")
        self.logger.info("=" * 60)
        self.logger.info(f"GPU Architecture: {self._gpu_arch}")
        self.logger.info(f"Current Step: {self._current_step}")
        self.logger.info("-" * 60)

        for name, state in self._states.items():
            status_str = state.status.name
            extra = ""
            if state.is_active():
                steps_active = self._current_step - state.enabled_at_step
                extra = f" (active for {steps_active} steps)"
            elif state.status == OptimizationStatus.FALLBACK:
                extra = f" ({state.disable_reason})"
            elif state.status == OptimizationStatus.DISABLED:
                extra = f" ({state.disable_reason})" if state.disable_reason else ""

            self.logger.info(f"  {name:20s}: {status_str:12s}{extra}")

        if self._anomalies:
            self.logger.info("-" * 60)
            self.logger.info(f"Total anomalies: {len(self._anomalies)}")
            for key, count in self.get_anomaly_summary().items():
                self.logger.info(f"  {key}: {count}")

        self.logger.info("=" * 60)


def create_safe_optimizations_from_args(args) -> SafeOptimizations:
    """Create SafeOptimizations from CLI args.

    Args:
        args: Parsed argparse namespace with experimental flags

    Returns:
        Configured SafeOptimizations instance
    """
    config = OptimizationConfig(
        enable_fa3=getattr(args, "experimental_fa3", True),
        enable_cuda_graphs=getattr(args, "experimental_cuda_graphs", True),
        enable_blackwell_tuning=getattr(args, "experimental_blackwell_tuning", True),
        enable_prefetch_threads=getattr(args, "experimental_prefetch_threads", 4),
        enable_fp8=getattr(args, "experimental_fp8", False),
        safety_window_steps=getattr(args, "experimental_safety_window", 100),
    )

    return SafeOptimizations(config)
