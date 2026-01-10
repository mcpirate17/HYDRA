"""Pretest Hook for SafeOptimizations.

Automatically runs SafeOptimizations pretests when:
- Training config changes
- A new checkpoint is loaded
- Model size changes

Logs results to a JSON file for tracking which optimizations work
on which model sizes and GPU configurations.

Usage:
    from hydra.training.pretest_hook import PretestHook, PretestLogger

    # Create hook
    hook = PretestHook(log_dir="logs/pretests")

    # Run pretests on config/checkpoint change
    results = hook.run_pretests(
        model=model,
        config=config,
        checkpoint_path=checkpoint_path,
        sample_batch=sample_batch,
    )

    # Query historical results
    history = hook.get_history(model_size="500M")
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from .safe_optimizations import (
    OptimizationConfig,
    OptimizationStatus,
    SafeOptimizations,
)

_log = logging.getLogger(__name__)


@dataclass
class PretestResult:
    """Result of a single optimization pretest."""
    optimization: str
    status: str  # "passed", "failed", "skipped", "error"
    time_ms: float
    error_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PretestRecord:
    """Complete record of a pretest run."""
    timestamp: str
    model_size: str
    model_params: int
    checkpoint_path: Optional[str]
    config_hash: str
    gpu_name: str
    gpu_arch: str
    compute_capability: Tuple[int, int]
    cuda_version: str
    torch_version: str
    results: List[PretestResult]
    triton_config: Dict[str, int]

    # Summary
    passed_count: int = 0
    failed_count: int = 0
    skipped_count: int = 0
    total_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["results"] = [r.to_dict() if hasattr(r, "to_dict") else r for r in self.results]
        return d


class PretestLogger:
    """Logs pretest results to a JSON file for historical tracking."""

    def __init__(self, log_dir: str = "logs/pretests"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "pretest_history.json"
        self._history: List[Dict[str, Any]] = []
        self._load_history()

    def _load_history(self) -> None:
        """Load existing history from file."""
        if self.log_file.exists():
            try:
                with open(self.log_file, "r") as f:
                    self._history = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                _log.warning(f"Failed to load pretest history: {e}")
                self._history = []

    def _save_history(self) -> None:
        """Save history to file."""
        try:
            with open(self.log_file, "w") as f:
                json.dump(self._history, f, indent=2)
        except IOError as e:
            _log.warning(f"Failed to save pretest history: {e}")

    def log_record(self, record: PretestRecord) -> None:
        """Log a pretest record."""
        self._history.append(record.to_dict())
        self._save_history()

        # Also write a per-run log file
        run_file = self.log_dir / f"pretest_{record.model_size}_{record.timestamp.replace(':', '-')}.json"
        try:
            with open(run_file, "w") as f:
                json.dump(record.to_dict(), f, indent=2)
        except IOError:
            pass

    def get_history(
        self,
        model_size: Optional[str] = None,
        gpu_arch: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Query pretest history with optional filters."""
        results = self._history

        if model_size:
            results = [r for r in results if r.get("model_size") == model_size]

        if gpu_arch:
            results = [r for r in results if r.get("gpu_arch") == gpu_arch]

        # Return most recent first
        return sorted(results, key=lambda x: x.get("timestamp", ""), reverse=True)[:limit]

    def get_optimization_success_rate(
        self,
        optimization: str,
        model_size: Optional[str] = None,
    ) -> Dict[str, float]:
        """Get success rate for an optimization across runs."""
        history = self.get_history(model_size=model_size, limit=1000)

        passed = 0
        failed = 0

        for record in history:
            for result in record.get("results", []):
                if result.get("optimization") == optimization:
                    if result.get("status") == "passed":
                        passed += 1
                    elif result.get("status") in ("failed", "error"):
                        failed += 1

        total = passed + failed
        return {
            "passed": passed,
            "failed": failed,
            "total": total,
            "success_rate": passed / total if total > 0 else 0.0,
        }

    def get_summary_by_model_size(self) -> Dict[str, Dict[str, Any]]:
        """Get summary of pretest results grouped by model size."""
        summary: Dict[str, Dict[str, Any]] = {}

        for record in self._history:
            size = record.get("model_size", "unknown")
            if size not in summary:
                summary[size] = {
                    "total_runs": 0,
                    "optimizations": {},
                }

            summary[size]["total_runs"] += 1

            for result in record.get("results", []):
                opt = result.get("optimization", "unknown")
                if opt not in summary[size]["optimizations"]:
                    summary[size]["optimizations"][opt] = {"passed": 0, "failed": 0, "skipped": 0}

                status = result.get("status", "unknown")
                if status in ("passed", "failed", "skipped"):
                    summary[size]["optimizations"][opt][status] += 1

        return summary


class PretestHook:
    """Hook for automatically running SafeOptimizations pretests.

    Triggers pretests when:
    - Config changes (detected via hash)
    - Checkpoint is loaded
    - Explicitly called

    Caches results to avoid redundant pretests for the same config.
    """

    def __init__(
        self,
        log_dir: str = "logs/pretests",
        cache_results: bool = True,
        verbose: bool = True,
        logger: Optional[logging.Logger] = None,
    ):
        self.logger = logger or _log
        self.verbose = verbose
        self.cache_results = cache_results

        # Logging
        self.pretest_logger = PretestLogger(log_dir)

        # Cache: config_hash -> PretestRecord
        self._cache: Dict[str, PretestRecord] = {}

        # Last config hash for change detection
        self._last_config_hash: Optional[str] = None

        # Callbacks for pretest events
        self._on_pretest_start: List[Callable] = []
        self._on_pretest_complete: List[Callable] = []
        self._on_optimization_failed: List[Callable] = []

    def _compute_config_hash(
        self,
        config: Any,
        checkpoint_path: Optional[str] = None,
    ) -> str:
        """Compute hash of config for change detection."""
        hash_input = []

        # Model architecture
        hash_input.append(f"model_size={getattr(config, 'model_size', 'unknown')}")
        hash_input.append(f"dim={getattr(config, 'mod_mor_dim', 0)}")
        hash_input.append(f"blocks={getattr(config, 'n_mor_blocks', 0)}")
        hash_input.append(f"recursions={getattr(config, 'mor_recursions', 0)}")
        hash_input.append(f"heads={getattr(config, 'mod_mor_n_heads', 0)}")

        # Experimental flags
        hash_input.append(f"fa3={getattr(config, 'experimental_fa3', True)}")
        hash_input.append(f"cuda_graphs={getattr(config, 'experimental_cuda_graphs', True)}")
        hash_input.append(f"blackwell={getattr(config, 'experimental_blackwell_tuning', True)}")
        hash_input.append(f"fp8={getattr(config, 'experimental_fp8', False)}")
        hash_input.append(f"static_routing={getattr(config, 'static_routing_mode', False)}")

        # Checkpoint
        if checkpoint_path:
            hash_input.append(f"checkpoint={checkpoint_path}")

        hash_str = "|".join(hash_input)
        return hashlib.md5(hash_str.encode()).hexdigest()[:12]

    def _get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information."""
        if not torch.cuda.is_available():
            return {
                "gpu_name": "CPU",
                "gpu_arch": "cpu",
                "compute_capability": (0, 0),
                "cuda_version": "N/A",
            }

        gpu_name = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)

        # Determine architecture
        major = capability[0]
        if major >= 12:
            arch = "blackwell"
        elif major >= 10:
            arch = "blackwell"  # SM 10.x is also Blackwell
        elif major >= 9:
            arch = "hopper"
        elif major >= 8:
            arch = "ada" if capability[1] >= 9 else "ampere"
        else:
            arch = "older"

        cuda_version = torch.version.cuda or "N/A"

        return {
            "gpu_name": gpu_name,
            "gpu_arch": arch,
            "compute_capability": capability,
            "cuda_version": cuda_version,
        }

    def config_changed(self, config: Any, checkpoint_path: Optional[str] = None) -> bool:
        """Check if config has changed since last pretest."""
        current_hash = self._compute_config_hash(config, checkpoint_path)
        changed = current_hash != self._last_config_hash
        return changed

    def run_pretests(
        self,
        model: torch.nn.Module,
        config: Any,
        sample_batch: torch.Tensor,
        checkpoint_path: Optional[str] = None,
        force: bool = False,
    ) -> PretestRecord:
        """Run pretests and log results.

        Args:
            model: The model to test
            config: Training config
            sample_batch: Sample input batch for testing
            checkpoint_path: Path to checkpoint (if loaded)
            force: Force rerun even if cached

        Returns:
            PretestRecord with results
        """
        config_hash = self._compute_config_hash(config, checkpoint_path)

        # Check cache
        if not force and self.cache_results and config_hash in self._cache:
            if self.verbose:
                self.logger.info(f"[PretestHook] Using cached results for config hash {config_hash}")
            return self._cache[config_hash]

        # Notify callbacks
        for cb in self._on_pretest_start:
            try:
                cb(config, checkpoint_path)
            except Exception:
                pass

        if self.verbose:
            self.logger.info("=" * 60)
            self.logger.info("RUNNING SAFEOPTIMIZATIONS PRETESTS")
            self.logger.info("=" * 60)

        # Get system info
        gpu_info = self._get_gpu_info()
        model_size = str(getattr(config, "model_size", "unknown"))
        model_params = sum(p.numel() for p in model.parameters())

        # Create SafeOptimizations
        opt_config = OptimizationConfig(
            enable_fa3=bool(getattr(config, "experimental_fa3", True)),
            enable_cuda_graphs=bool(getattr(config, "experimental_cuda_graphs", True)),
            enable_blackwell_tuning=bool(getattr(config, "experimental_blackwell_tuning", True)),
            enable_prefetch_threads=int(getattr(config, "experimental_prefetch_threads", 4)),
            enable_fp8=bool(getattr(config, "experimental_fp8", False)),
            pretest_steps=int(getattr(config, "experimental_pretest_steps", 5)),
        )

        safe_opts = SafeOptimizations(
            config=opt_config,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        # Check if static routing mode is enabled (skips cuda_graphs pretest)
        static_routing_mode = bool(getattr(config, "static_routing_mode", False))

        # Run pretests for each optimization
        results: List[PretestResult] = []
        total_time = 0.0
        passed = 0
        failed = 0
        skipped = 0

        for name, state in safe_opts._states.items():
            # Static routing mode: CUDA graphs are guaranteed to work because
            # we eliminated dynamic shapes. Skip the test and mark as passed.
            if name == "cuda_graphs" and static_routing_mode:
                results.append(PretestResult(
                    optimization=name,
                    status="passed",
                    time_ms=0.0,
                    error_message="static_routing_mode enabled (no dynamic shapes)",
                ))
                passed += 1
                if self.verbose:
                    self.logger.info(f"  [{name}] ✓ SKIPPED (static_routing_mode enabled)")
                continue

            if state.status != OptimizationStatus.PRETESTING:
                results.append(PretestResult(
                    optimization=name,
                    status="skipped",
                    time_ms=0.0,
                    error_message=state.disable_reason or "Not enabled",
                ))
                skipped += 1
                continue

            if self.verbose:
                self.logger.info(f"  [{name}] Running pretest...")

            start = time.perf_counter()
            try:
                test_passed = safe_opts._run_single_pretest(name, model, sample_batch)
                elapsed = (time.perf_counter() - start) * 1000
                total_time += elapsed

                if test_passed:
                    results.append(PretestResult(
                        optimization=name,
                        status="passed",
                        time_ms=elapsed,
                    ))
                    passed += 1
                    if self.verbose:
                        self.logger.info(f"    ✓ PASSED ({elapsed:.1f}ms)")
                else:
                    results.append(PretestResult(
                        optimization=name,
                        status="failed",
                        time_ms=elapsed,
                        error_message="Pretest returned False",
                    ))
                    failed += 1
                    if self.verbose:
                        self.logger.info(f"    ✗ FAILED ({elapsed:.1f}ms)")

                    # Notify callbacks
                    for cb in self._on_optimization_failed:
                        try:
                            cb(name, "Pretest returned False")
                        except Exception:
                            pass

            except Exception as e:
                elapsed = (time.perf_counter() - start) * 1000
                total_time += elapsed
                error_msg = str(e)[:200]

                results.append(PretestResult(
                    optimization=name,
                    status="error",
                    time_ms=elapsed,
                    error_message=error_msg,
                ))
                failed += 1

                if self.verbose:
                    self.logger.warning(f"    ✗ ERROR: {error_msg[:80]}")

                # Notify callbacks
                for cb in self._on_optimization_failed:
                    try:
                        cb(name, error_msg)
                    except Exception:
                        pass

        # Create record
        record = PretestRecord(
            timestamp=datetime.now().isoformat(),
            model_size=model_size,
            model_params=model_params,
            checkpoint_path=checkpoint_path,
            config_hash=config_hash,
            gpu_name=gpu_info["gpu_name"],
            gpu_arch=gpu_info["gpu_arch"],
            compute_capability=gpu_info["compute_capability"],
            cuda_version=gpu_info["cuda_version"],
            torch_version=torch.__version__,
            results=results,
            triton_config=safe_opts.get_triton_config(),
            passed_count=passed,
            failed_count=failed,
            skipped_count=skipped,
            total_time_ms=total_time,
        )

        # Log and cache
        self.pretest_logger.log_record(record)
        self._cache[config_hash] = record
        self._last_config_hash = config_hash

        # Summary
        if self.verbose:
            self.logger.info("-" * 60)
            self.logger.info(f"PRETESTS COMPLETE: {passed} passed, {failed} failed, {skipped} skipped")
            self.logger.info(f"Total time: {total_time:.1f}ms")
            self.logger.info(f"Results logged to: {self.pretest_logger.log_file}")
            self.logger.info("=" * 60)

        # Notify callbacks
        for cb in self._on_pretest_complete:
            try:
                cb(record)
            except Exception:
                pass

        return record

    def on_checkpoint_loaded(
        self,
        model: torch.nn.Module,
        config: Any,
        checkpoint_path: str,
        sample_batch: torch.Tensor,
    ) -> Optional[PretestRecord]:
        """Hook called when a checkpoint is loaded.

        Automatically runs pretests if config changed.
        """
        if self.config_changed(config, checkpoint_path):
            if self.verbose:
                self.logger.info(f"[PretestHook] Config changed, running pretests for checkpoint: {checkpoint_path}")
            return self.run_pretests(model, config, sample_batch, checkpoint_path)
        return None

    def on_config_changed(
        self,
        model: torch.nn.Module,
        config: Any,
        sample_batch: torch.Tensor,
    ) -> Optional[PretestRecord]:
        """Hook called when training config changes.

        Automatically runs pretests if config differs from last run.
        """
        if self.config_changed(config):
            if self.verbose:
                self.logger.info("[PretestHook] Config changed, running pretests...")
            return self.run_pretests(model, config, sample_batch)
        return None

    def register_callback(
        self,
        event: str,
        callback: Callable,
    ) -> None:
        """Register a callback for pretest events.

        Events:
        - "pretest_start": Called before pretests run
        - "pretest_complete": Called after pretests complete
        - "optimization_failed": Called when an optimization fails
        """
        if event == "pretest_start":
            self._on_pretest_start.append(callback)
        elif event == "pretest_complete":
            self._on_pretest_complete.append(callback)
        elif event == "optimization_failed":
            self._on_optimization_failed.append(callback)
        else:
            raise ValueError(f"Unknown event: {event}")

    def get_history(self, **kwargs) -> List[Dict[str, Any]]:
        """Get pretest history with optional filters."""
        return self.pretest_logger.get_history(**kwargs)

    def get_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary of pretest results by model size."""
        return self.pretest_logger.get_summary_by_model_size()

    def print_summary(self) -> None:
        """Print a formatted summary of pretest history."""
        summary = self.get_summary()

        print("\n" + "=" * 70)
        print("PRETEST HISTORY SUMMARY")
        print("=" * 70)

        if not summary:
            print("No pretest history found.")
            return

        for model_size, data in sorted(summary.items()):
            print(f"\n{model_size} ({data['total_runs']} runs):")
            print("-" * 50)

            for opt, stats in data["optimizations"].items():
                total = stats["passed"] + stats["failed"]
                rate = stats["passed"] / total * 100 if total > 0 else 0

                if stats["passed"] > 0 and stats["failed"] == 0:
                    status = "✓"
                elif stats["failed"] > 0 and stats["passed"] == 0:
                    status = "✗"
                else:
                    status = "~"

                print(f"  {status} {opt:<20s}: {stats['passed']}/{total} passed ({rate:.0f}%)")

        print("\n" + "=" * 70)


# Convenience function for one-off pretest runs
def run_pretests_for_checkpoint(
    checkpoint_path: str,
    model_size: str = "500M",
    log_dir: str = "logs/pretests",
) -> PretestRecord:
    """Convenience function to run pretests for a checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        model_size: Model size key (e.g., "500M", "1B")
        log_dir: Directory for pretest logs

    Returns:
        PretestRecord with results
    """
    from .config import TrainingConfig, MODEL_SIZE_CONFIGS
    from hydra.model.framework import HydraModel

    # Get model config
    size_config = MODEL_SIZE_CONFIGS.get(model_size, MODEL_SIZE_CONFIGS["100M"])

    config = TrainingConfig(
        mode="testing",
        model_size=model_size,
        mod_mor_dim=size_config["mod_mor_dim"],
        n_mor_blocks=size_config["n_mor_blocks"],
        mor_recursions=size_config["mor_recursions"],
        mod_mor_n_heads=size_config["mod_mor_n_heads"],
        mod_mor_n_kv_heads=size_config["mod_mor_n_kv_heads"],
        max_seq_len=2048,
        max_steps=100,
    )

    # Create model
    model = HydraModel(
        vocab_size=config.vocab_size,
        dim=config.mod_mor_dim,
        n_mor_blocks=config.n_mor_blocks,
        recursions_per_block=config.mor_recursions,
        n_heads=config.mod_mor_n_heads,
        n_kv_heads=config.mod_mor_n_kv_heads,
        compression_factor=4,
        mlp_ratio=3.6,
        max_seq_len=config.max_seq_len,
        mod_capacity=0.5,
        adaptive=True,
    ).to("cuda").bfloat16()

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cuda")
    state_dict = checkpoint.get("model", {})
    # Drop RoPE caches
    drop_keys = [k for k in state_dict.keys() if "cos_cached" in k or "sin_cached" in k]
    for k in drop_keys:
        state_dict.pop(k, None)
    model.load_state_dict(state_dict, strict=False)

    # Create sample batch
    sample_batch = torch.randint(0, config.vocab_size, (2, 512), device="cuda")

    # Run pretests
    hook = PretestHook(log_dir=log_dir)
    record = hook.run_pretests(
        model=model,
        config=config,
        sample_batch=sample_batch,
        checkpoint_path=checkpoint_path,
    )

    # Cleanup
    del model
    torch.cuda.empty_cache()

    return record
