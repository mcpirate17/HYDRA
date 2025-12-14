"""
Comprehensive Diagnostics for Compressed Convolutional Grouped Query Attention (CCGQA)

This module provides detailed performance analysis and optimization recommendations
for the CCGQA attention mechanism including:

1. Speed Benchmarking
   - Forward pass latency across batch sizes and sequence lengths
   - Backward pass latency (gradient computation)
   - FLOPs measurement and efficiency analysis

2. Memory Profiling
   - Peak memory usage during forward/backward passes
   - Memory scaling characteristics
   - Activation memory vs parameter memory breakdown

3. Learning Diagnostics
   - Gradient flow analysis (vanishing/exploding gradient detection)
   - Loss descent trajectory
   - Learning capacity measurement

4. Component Analysis
   - Kernel compression effectiveness
   - QK-mean coupling impact
   - Head reshaping efficiency
   - Normalization layer behavior

5. Optimization Recommendations
   - Compression factor optimization
   - Convolution configuration tuning
   - Architecture hyperparameter suggestions
"""

import gc
import json
import math
import time
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, record_function

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class SpeedBenchmarkResult:
    """Speed benchmark metrics."""

    forward_time_ms: float
    backward_time_ms: float
    total_time_ms: float
    throughput_samples_per_sec: float
    batch_size: int
    seq_len: int
    flops: float
    flops_per_sec: float


@dataclass
class MemoryBenchmarkResult:
    """Memory benchmark metrics."""

    peak_memory_mb: float
    activation_memory_mb: float
    parameter_memory_mb: float
    allocated_memory_mb: float
    reserved_memory_mb: float
    batch_size: int
    seq_len: int
    memory_per_sample_mb: float


@dataclass
class GradientAnalysisResult:
    """Gradient flow analysis metrics."""

    mean_grad_norm: float
    max_grad_norm: float
    min_grad_norm: float
    grad_norm_q: float
    grad_norm_k: float
    grad_norm_v: float
    grad_norm_o: float
    vanishing_gradient_detected: bool
    exploding_gradient_detected: bool


@dataclass
class LearningResult:
    """Learning capability metrics."""

    initial_loss: float
    final_loss: float
    loss_improvement: float
    convergence_steps: int
    gradient_flow_score: float
    learning_capacity_score: float


@dataclass
class ComponentAnalysisResult:
    """Component-specific analysis results."""

    compression_effectiveness: Dict[str, float]
    qk_mean_impact: Dict[str, float]
    head_reshaping_efficiency: Dict[str, float]
    normalization_stats: Dict[str, float]


@dataclass
class CCGQADiagnosticsReport:
    """Complete diagnostics report for CCGQA."""

    timestamp: str
    model_config: Dict[str, Any]
    speed_results: List[SpeedBenchmarkResult]
    memory_results: List[MemoryBenchmarkResult]
    gradient_analysis: GradientAnalysisResult
    learning_results: LearningResult
    component_analysis: ComponentAnalysisResult
    optimization_recommendations: List[str]


class CCGQADiagnostician:
    """
    Comprehensive diagnostics engine for CCGQA attention.

    Performs speed, memory, learning, and component-specific analysis
    on the CCGQA attention mechanism.
    """

    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.results = {}

    def _cleanup_gpu(self):
        """Aggressively cleanup GPU memory."""
        gc.collect()
        if self.device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    def _measure_flops(
        self,
        batch_size: int,
        seq_len: int,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        compression_factor: int,
    ) -> float:
        """
        Estimate FLOPs for CCGQA forward pass.

        Main operations:
        1. Down-projections: Q, K, V linear layers
        2. Convolutions: sequence/channel convolutions on Q, K
        3. Attention: Q @ K^T, softmax, @ V
        4. Up-projection: output projection
        """
        latent_dim = dim // compression_factor
        head_dim = latent_dim // n_heads
        kv_dim = n_kv_heads * head_dim

        flops = 0

        # 1. Down-projections (linear)
        # Q: [B, S, dim] @ [dim, latent_dim]
        flops += 2 * batch_size * seq_len * dim * latent_dim
        # K: [B, S, dim] @ [dim, kv_dim]
        flops += 2 * batch_size * seq_len * dim * kv_dim
        # V: [B, S, dim] @ [dim, kv_dim]
        flops += 2 * batch_size * seq_len * dim * kv_dim

        # 2. Convolutions (if enabled)
        # Conv1d: [B, C, S] -> [B, C, S], kernel=3
        # FLOPs = 2 * B * S * C * kernel_size (roughly, per-group convs are cheaper)
        conv_kernel = 3
        # Q conv: [B, latent_dim, S]
        flops += 2 * batch_size * seq_len * latent_dim * conv_kernel
        # K conv: [B, kv_dim, S]
        flops += 2 * batch_size * seq_len * kv_dim * conv_kernel
        # Second conv (full channels)
        flops += 2 * batch_size * seq_len * latent_dim * conv_kernel
        flops += 2 * batch_size * seq_len * kv_dim * conv_kernel

        # 3. Attention
        # Q @ K^T: [B, n_heads, S, head_dim] @ [B, n_heads, S, head_dim]^T
        # = [B, n_heads, S, S]
        flops += 2 * batch_size * n_heads * seq_len * seq_len * head_dim
        # @ V: [B, n_heads, S, S] @ [B, n_heads, S, head_dim]
        flops += 2 * batch_size * n_heads * seq_len * seq_len * head_dim

        # 4. Output projection
        # [B, S, latent_dim] @ [latent_dim, dim]
        flops += 2 * batch_size * seq_len * latent_dim * dim

        return flops

    def benchmark_speed(
        self,
        attention_module: nn.Module,
        batch_sizes: List[int] = [1, 4, 16],
        seq_lengths: List[int] = [128, 512, 2048],
        num_runs: int = 10,
        warmup_runs: int = 3,
    ) -> List[SpeedBenchmarkResult]:
        """
        Benchmark forward and backward pass speed.

        Args:
            attention_module: CCGQAAttention or CCGQABlock instance
            batch_sizes: Batch sizes to test
            seq_lengths: Sequence lengths to test
            num_runs: Number of benchmark iterations
            warmup_runs: Number of warmup iterations

        Returns:
            List of speed benchmark results
        """
        results = []
        attention_module = attention_module.to(self.device)
        attention_module.eval()

        for batch_size in batch_sizes:
            for seq_len in seq_lengths:
                self._cleanup_gpu()

                # Get model config
                dim = attention_module.dim if hasattr(attention_module, "dim") else 768
                n_heads = (
                    attention_module.n_heads
                    if hasattr(attention_module, "n_heads")
                    else 16
                )
                n_kv_heads = (
                    attention_module.n_kv_heads
                    if hasattr(attention_module, "n_kv_heads")
                    else 4
                )
                compression_factor = (
                    attention_module.compression_factor
                    if hasattr(attention_module, "compression_factor")
                    else 4
                )

                # Create dummy input
                x = torch.randn(batch_size, seq_len, dim, device=self.device)

                # Warmup
                with torch.no_grad():
                    for _ in range(warmup_runs):
                        _ = attention_module(x)

                # Forward pass timing
                self._cleanup_gpu()
                torch.cuda.synchronize() if self.device == "cuda" else None

                forward_times = []
                for _ in range(num_runs):
                    start = time.perf_counter()
                    with torch.no_grad():
                        output = attention_module(x)
                    torch.cuda.synchronize() if self.device == "cuda" else None
                    end = time.perf_counter()
                    forward_times.append((end - start) * 1000)  # Convert to ms

                forward_time_ms = np.mean(forward_times)

                # Backward pass timing
                self._cleanup_gpu()
                torch.cuda.synchronize() if self.device == "cuda" else None

                backward_times = []
                for _ in range(num_runs):
                    x_grad = torch.randn(batch_size, seq_len, dim, device=self.device)
                    output = attention_module(x)

                    start = time.perf_counter()
                    loss = (output * x_grad).sum()
                    loss.backward()
                    torch.cuda.synchronize() if self.device == "cuda" else None
                    end = time.perf_counter()
                    backward_times.append((end - start) * 1000)

                    # Clear gradients
                    attention_module.zero_grad()

                backward_time_ms = np.mean(backward_times)
                total_time_ms = forward_time_ms + backward_time_ms

                # Calculate FLOPs
                flops = self._measure_flops(
                    batch_size, seq_len, dim, n_heads, n_kv_heads, compression_factor
                )
                flops_per_sec = (
                    (flops / (total_time_ms / 1000)) if total_time_ms > 0 else 0
                )

                # Throughput
                throughput = (
                    (batch_size / (total_time_ms / 1000)) if total_time_ms > 0 else 0
                )

                result = SpeedBenchmarkResult(
                    forward_time_ms=forward_time_ms,
                    backward_time_ms=backward_time_ms,
                    total_time_ms=total_time_ms,
                    throughput_samples_per_sec=throughput,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    flops=flops,
                    flops_per_sec=flops_per_sec,
                )
                results.append(result)

                print(
                    f"Speed: B={batch_size}, S={seq_len} | "
                    f"Forward={forward_time_ms:.2f}ms | "
                    f"Backward={backward_time_ms:.2f}ms | "
                    f"Total={total_time_ms:.2f}ms | "
                    f"FLOPs/s={flops_per_sec / 1e12:.2f}T"
                )

        return results

    def benchmark_memory(
        self,
        attention_module: nn.Module,
        batch_sizes: List[int] = [1, 4, 16],
        seq_lengths: List[int] = [128, 512, 2048],
    ) -> List[MemoryBenchmarkResult]:
        """
        Benchmark memory usage during forward and backward passes.

        Args:
            attention_module: CCGQAAttention or CCGQABlock instance
            batch_sizes: Batch sizes to test
            seq_lengths: Sequence lengths to test

        Returns:
            List of memory benchmark results
        """
        results = []
        attention_module = attention_module.to(self.device)

        # Calculate parameter memory
        param_memory_mb = sum(p.numel() * 4 for p in attention_module.parameters()) / (
            1024 * 1024
        )

        for batch_size in batch_sizes:
            for seq_len in seq_lengths:
                self._cleanup_gpu()

                dim = attention_module.dim if hasattr(attention_module, "dim") else 768

                x = torch.randn(batch_size, seq_len, dim, device=self.device)

                # Reset memory stats
                if self.device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()

                # Forward + Backward
                output = attention_module(x)
                loss = output.sum()

                if self.device == "cuda":
                    torch.cuda.synchronize()
                    peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
                else:
                    peak_memory = 0

                loss.backward()

                if self.device == "cuda":
                    torch.cuda.synchronize()
                    peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)

                # Estimate activation memory
                activation_memory_mb = peak_memory - param_memory_mb
                allocated_memory_mb = (
                    torch.cuda.memory_allocated() / (1024 * 1024)
                    if self.device == "cuda"
                    else 0
                )
                reserved_memory_mb = (
                    torch.cuda.memory_reserved() / (1024 * 1024)
                    if self.device == "cuda"
                    else 0
                )

                memory_per_sample = peak_memory / batch_size if batch_size > 0 else 0

                result = MemoryBenchmarkResult(
                    peak_memory_mb=peak_memory,
                    activation_memory_mb=max(0, activation_memory_mb),
                    parameter_memory_mb=param_memory_mb,
                    allocated_memory_mb=allocated_memory_mb,
                    reserved_memory_mb=reserved_memory_mb,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    memory_per_sample_mb=memory_per_sample,
                )
                results.append(result)

                print(
                    f"Memory: B={batch_size}, S={seq_len} | "
                    f"Peak={peak_memory:.2f}MB | "
                    f"Activation={activation_memory_mb:.2f}MB | "
                    f"Per-sample={memory_per_sample:.2f}MB"
                )

        return results

    def analyze_gradient_flow(
        self,
        attention_module: nn.Module,
        batch_size: int = 4,
        seq_len: int = 256,
    ) -> GradientAnalysisResult:
        """
        Analyze gradient flow through CCGQA.

        Detects vanishing/exploding gradients and measures
        gradient norms at each layer.

        Args:
            attention_module: CCGQAAttention or CCGQABlock instance
            batch_size: Batch size for analysis
            seq_len: Sequence length for analysis

        Returns:
            Gradient analysis results
        """
        attention_module = attention_module.to(self.device)
        attention_module.train()

        dim = attention_module.dim if hasattr(attention_module, "dim") else 768

        x = torch.randn(
            batch_size, seq_len, dim, device=self.device, requires_grad=True
        )

        output = attention_module(x)
        loss = output.sum()
        loss.backward()

        # Collect gradient norms
        grad_norms = {}
        for name, param in attention_module.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms[name] = grad_norm

        # Aggregate statistics
        all_grad_norms = list(grad_norms.values())
        mean_grad_norm = np.mean(all_grad_norms) if all_grad_norms else 0
        max_grad_norm = np.max(all_grad_norms) if all_grad_norms else 0
        min_grad_norm = np.min(all_grad_norms) if all_grad_norms else 0

        # Component-specific gradient norms
        grad_norm_q = max(
            [grad_norms.get(name, 0) for name in grad_norms if "q_" in name.lower()],
            default=0,
        )
        grad_norm_k = max(
            [grad_norms.get(name, 0) for name in grad_norms if "k_" in name.lower()],
            default=0,
        )
        grad_norm_v = max(
            [grad_norms.get(name, 0) for name in grad_norms if "v_" in name.lower()],
            default=0,
        )
        grad_norm_o = max(
            [
                grad_norms.get(name, 0)
                for name in grad_norms
                if "o_proj" in name.lower()
            ],
            default=0,
        )

        # Detect vanishing/exploding gradients
        vanishing = min_grad_norm < 1e-6 if all_grad_norms else False
        exploding = max_grad_norm > 1e4 if all_grad_norms else False

        result = GradientAnalysisResult(
            mean_grad_norm=mean_grad_norm,
            max_grad_norm=max_grad_norm,
            min_grad_norm=min_grad_norm,
            grad_norm_q=grad_norm_q,
            grad_norm_k=grad_norm_k,
            grad_norm_v=grad_norm_v,
            grad_norm_o=grad_norm_o,
            vanishing_gradient_detected=vanishing,
            exploding_gradient_detected=exploding,
        )

        print(
            f"Gradient Flow: Mean={mean_grad_norm:.2e} | "
            f"Max={max_grad_norm:.2e} | Min={min_grad_norm:.2e}"
        )
        if vanishing:
            print("  ⚠️  VANISHING GRADIENTS DETECTED")
        if exploding:
            print("  ⚠️  EXPLODING GRADIENTS DETECTED")

        return result

    def analyze_learning(
        self,
        attention_module: nn.Module,
        batch_size: int = 4,
        seq_len: int = 256,
        num_steps: int = 100,
        learning_rate: float = 1e-3,
    ) -> LearningResult:
        """
        Test learning capability by training on synthetic data.

        Measures loss descent trajectory and learning capacity.

        Args:
            attention_module: CCGQAAttention or CCGQABlock instance
            batch_size: Batch size for training
            seq_len: Sequence length
            num_steps: Number of gradient steps
            learning_rate: Learning rate

        Returns:
            Learning analysis results
        """
        attention_module = attention_module.to(self.device)
        attention_module.train()

        dim = attention_module.dim if hasattr(attention_module, "dim") else 768

        optimizer = torch.optim.Adam(attention_module.parameters(), lr=learning_rate)

        losses = []
        grad_norms = []

        for step in range(num_steps):
            x = torch.randn(batch_size, seq_len, dim, device=self.device)
            target = torch.randn(batch_size, seq_len, dim, device=self.device)

            output = attention_module(x)
            loss = F.mse_loss(output, target)

            optimizer.zero_grad()
            loss.backward()

            # Collect gradient norm BEFORE clipping (for diagnostics)
            grad_norm = (
                sum(
                    p.grad.norm() ** 2
                    for p in attention_module.parameters()
                    if p.grad is not None
                )
                ** 0.5
            )
            grad_norms.append(grad_norm.item())

            # Apply gradient clipping (critical for stability)
            torch.nn.utils.clip_grad_norm_(attention_module.parameters(), max_norm=1.0)

            optimizer.step()
            losses.append(loss.item())

            if (step + 1) % 20 == 0:
                print(f"Learning Step {step + 1}/{num_steps}: Loss={loss.item():.4f}")

        initial_loss = losses[0]
        final_loss = losses[-1]
        loss_improvement = (initial_loss - final_loss) / (initial_loss + 1e-8)

        # Find convergence step (loss change < 1%)
        convergence_steps = num_steps
        for i in range(1, len(losses)):
            if (losses[i - 1] - losses[i]) / (losses[i - 1] + 1e-8) < 0.01:
                convergence_steps = i
                break

        # Gradient flow score (normalized)
        grad_flow_score = np.mean(grad_norms) / (np.std(grad_norms) + 1e-8)

        # Learning capacity score (improvement + convergence)
        learning_capacity = (loss_improvement * 100) + (
            1 - convergence_steps / num_steps
        ) * 50

        result = LearningResult(
            initial_loss=initial_loss,
            final_loss=final_loss,
            loss_improvement=loss_improvement,
            convergence_steps=convergence_steps,
            gradient_flow_score=grad_flow_score,
            learning_capacity_score=learning_capacity,
        )

        print(
            f"Learning: Initial Loss={initial_loss:.4f} | Final={final_loss:.4f} | "
            f"Improvement={loss_improvement:.2%} | Convergence={convergence_steps} steps"
        )

        return result

    def analyze_components(
        self, attention_module: nn.Module
    ) -> ComponentAnalysisResult:
        """
        Analyze component-specific behaviors.

        Tests the effectiveness of:
        - Kernel compression
        - QK-mean coupling
        - Head reshaping
        - Normalization

        Args:
            attention_module: CCGQAAttention instance

        Returns:
            Component analysis results
        """
        results = ComponentAnalysisResult(
            compression_effectiveness={},
            qk_mean_impact={},
            head_reshaping_efficiency={},
            normalization_stats={},
        )

        # 1. Compression analysis
        if hasattr(attention_module, "compression_factor"):
            compression = attention_module.compression_factor
            dim = attention_module.dim
            latent_dim = attention_module.latent_dim

            param_reduction = (dim - latent_dim) / dim
            results.compression_effectiveness["compression_factor"] = compression
            results.compression_effectiveness["param_reduction_ratio"] = param_reduction
            results.compression_effectiveness["latent_dim"] = latent_dim
            results.compression_effectiveness["original_dim"] = dim

            print(
                f"Compression: Factor={compression}x | "
                f"Dim {dim} -> {latent_dim} | "
                f"Param Reduction={param_reduction:.1%}"
            )

        # 2. QK-mean coupling analysis
        if hasattr(attention_module, "use_qk_mean"):
            results.qk_mean_impact["enabled"] = attention_module.use_qk_mean
            results.qk_mean_impact["n_groups"] = (
                attention_module.n_groups
                if hasattr(attention_module, "n_groups")
                else 1
            )
            print(f"QK-Mean: Enabled={attention_module.use_qk_mean}")

        # 3. Head reshaping analysis
        if hasattr(attention_module, "n_heads"):
            n_heads = attention_module.n_heads
            n_kv_heads = (
                attention_module.n_kv_heads
                if hasattr(attention_module, "n_kv_heads")
                else n_heads
            )
            head_dim = (
                attention_module.head_dim
                if hasattr(attention_module, "head_dim")
                else 64
            )

            gqa_ratio = n_heads / n_kv_heads

            results.head_reshaping_efficiency["n_heads"] = n_heads
            results.head_reshaping_efficiency["n_kv_heads"] = n_kv_heads
            results.head_reshaping_efficiency["gqa_ratio"] = gqa_ratio
            results.head_reshaping_efficiency["head_dim"] = head_dim
            results.head_reshaping_efficiency["kv_cache_reduction"] = 1 / gqa_ratio

            print(
                f"Head Reshaping: Q={n_heads} | KV={n_kv_heads} | "
                f"GQA Ratio={gqa_ratio}x | KV-Cache Reduction={1 / gqa_ratio:.1%}"
            )

        # 4. Normalization analysis
        if hasattr(attention_module, "use_qk_norm"):
            results.normalization_stats["qk_norm_enabled"] = (
                attention_module.use_qk_norm
            )
            if attention_module.use_qk_norm and hasattr(
                attention_module, "key_temperature"
            ):
                results.normalization_stats["key_temperature"] = (
                    attention_module.key_temperature.item()
                )
            print(f"Normalization: QK-Norm Enabled={attention_module.use_qk_norm}")

        # 5. Convolution analysis
        if hasattr(attention_module, "use_convs"):
            results.compression_effectiveness["convolutions_enabled"] = (
                attention_module.use_convs
            )
            print(f"Convolutions: Enabled={attention_module.use_convs}")

        return results

    def generate_optimization_recommendations(
        self,
        speed_results: List[SpeedBenchmarkResult],
        memory_results: List[MemoryBenchmarkResult],
        gradient_analysis: GradientAnalysisResult,
        learning_result: LearningResult,
        component_analysis: ComponentAnalysisResult,
        attention_module: nn.Module,
    ) -> List[str]:
        """
        Generate optimization recommendations based on analysis results.

        Args:
            Various analysis results

        Returns:
            List of optimization recommendations
        """
        recommendations = []

        # 1. Speed optimizations
        if speed_results:
            avg_flops_per_sec = np.mean([r.flops_per_sec for r in speed_results])
            if avg_flops_per_sec < 1e11:  # Less than 100 GFLOPS
                recommendations.append(
                    "⚠️  Low compute throughput detected. Consider:"
                    "\n   - Using Flash Attention for sequence attention"
                    "\n   - Reducing kernel size for convolutions (from 3 to 1)"
                    "\n   - Decreasing sequence length sampling for training"
                )

            backward_forward_ratio = np.mean(
                [r.backward_time_ms / r.forward_time_ms for r in speed_results]
            )
            if backward_forward_ratio > 2:
                recommendations.append(
                    "⚠️  Backward pass is much slower than forward. Consider:"
                    "\n   - Enabling gradient checkpointing in full model"
                    "\n   - Using 16-bit precision (float16/bfloat16)"
                    "\n   - Reducing attention computation with lower compression"
                )

        # 2. Memory optimizations
        if memory_results:
            avg_memory_per_sample = np.mean(
                [r.memory_per_sample_mb for r in memory_results]
            )
            if avg_memory_per_sample > 100:
                recommendations.append(
                    "⚠️  High memory per sample. Consider:"
                    "\n   - Increasing compression factor (current may be too low)"
                    "\n   - Using mixed precision training (AMP)"
                    "\n   - Reducing model width or using grouped query attention"
                )

            # Check memory scaling
            large_batch = max([r.batch_size for r in memory_results])
            small_batch = min([r.batch_size for r in memory_results])
            large_mem = [
                r.peak_memory_mb for r in memory_results if r.batch_size == large_batch
            ]
            small_mem = [
                r.peak_memory_mb for r in memory_results if r.batch_size == small_batch
            ]
            if large_mem and small_mem:
                scaling_factor = large_mem[0] / small_mem[0]
                expected_scaling = large_batch / small_batch
                if scaling_factor > expected_scaling * 1.5:
                    recommendations.append(
                        "⚠️  Sublinear memory scaling detected. Check for:"
                        "\n   - Inefficient activation caching"
                        "\n   - Memory fragmentation (try batch-norm or layer-norm fusion)"
                    )

        # 3. Gradient flow
        if gradient_analysis.vanishing_gradient_detected:
            recommendations.append(
                "⚠️  Vanishing gradients detected. Fix with:"
                "\n   - Reduce model depth (fewer layers)"
                "\n   - Use better initialization (e.g., Kaiming)"
                "\n   - Decrease learning rate or use Adam optimizer"
                "\n   - Check if QK normalization is too aggressive"
            )

        if gradient_analysis.exploding_gradient_detected:
            recommendations.append(
                "⚠️  Exploding gradients detected. Fix with:"
                "\n   - Reduce learning rate"
                "\n   - Enable gradient clipping"
                "\n   - Use layer normalization before attention"
                "\n   - Decrease compression factor (too aggressive compression)"
            )

        if gradient_analysis.mean_grad_norm < 1e-5:
            recommendations.append(
                "⚠️  Very small gradient norms. Consider:"
                "\n   - Increasing learning rate"
                "\n   - Checking for dead ReLU activations"
                "\n   - Enabling residual connections properly"
            )

        # 4. Learning capacity
        if learning_result.learning_capacity_score < 50:
            recommendations.append(
                "⚠️  Low learning capacity score. Suggestions:"
                "\n   - Reduce compression factor for better expressivity"
                "\n   - Increase number of attention heads"
                "\n   - Try different kernel sizes for convolutions"
                "\n   - Verify data preprocessing is correct"
            )

        # 5. Component-specific
        if (
            component_analysis.compression_effectiveness.get("compression_factor", 1)
            > 8
        ):
            recommendations.append(
                "ℹ️  High compression factor (>8x) may lose capacity. Options:"
                "\n   - Reduce to 4x or 6x for better quality"
                "\n   - Increase head dimensions to compensate"
                "\n   - Profile actual quality loss before/after"
            )

        if component_analysis.head_reshaping_efficiency.get("gqa_ratio", 1) > 8:
            recommendations.append(
                "ℹ️  High GQA ratio (>8x). This reduces KV-cache but may limit capacity:"
                "\n   - Profile quality at current ratio"
                "\n   - Consider lowering to 4x-6x ratio"
                "\n   - Ensure sufficient latent dimension"
            )

        if not recommendations:
            recommendations.append(
                "✅ All metrics look healthy! Current configuration is well-balanced."
            )

        return recommendations

    def run_full_diagnostics(
        self,
        attention_module: nn.Module,
        batch_sizes: List[int] = [1, 4],
        seq_lengths: List[int] = [128, 512],
    ) -> CCGQADiagnosticsReport:
        """
        Run complete diagnostics suite.

        Args:
            attention_module: CCGQA module to diagnose
            batch_sizes: Batch sizes for benchmarking
            seq_lengths: Sequence lengths for benchmarking

        Returns:
            Complete diagnostics report
        """
        print("\n" + "=" * 80)
        print("CCGQA DIAGNOSTICS SUITE")
        print("=" * 80 + "\n")

        print("1️⃣  SPEED BENCHMARKING...")
        print("-" * 80)
        speed_results = self.benchmark_speed(
            attention_module, batch_sizes=batch_sizes, seq_lengths=seq_lengths
        )

        print("\n2️⃣  MEMORY PROFILING...")
        print("-" * 80)
        memory_results = self.benchmark_memory(
            attention_module, batch_sizes=batch_sizes, seq_lengths=seq_lengths
        )

        print("\n3️⃣  GRADIENT FLOW ANALYSIS...")
        print("-" * 80)
        gradient_analysis = self.analyze_gradient_flow(attention_module)

        print("\n4️⃣  LEARNING CAPABILITY...")
        print("-" * 80)
        learning_result = self.analyze_learning(attention_module)

        print("\n5️⃣  COMPONENT ANALYSIS...")
        print("-" * 80)
        component_analysis = self.analyze_components(attention_module)

        print("\n6️⃣  OPTIMIZATION RECOMMENDATIONS...")
        print("-" * 80)
        recommendations = self.generate_optimization_recommendations(
            speed_results,
            memory_results,
            gradient_analysis,
            learning_result,
            component_analysis,
            attention_module,
        )
        for rec in recommendations:
            print(f"\n{rec}")

        # Get model config
        config = {
            "dim": attention_module.dim if hasattr(attention_module, "dim") else None,
            "n_heads": attention_module.n_heads
            if hasattr(attention_module, "n_heads")
            else None,
            "n_kv_heads": (
                attention_module.n_kv_heads
                if hasattr(attention_module, "n_kv_heads")
                else None
            ),
            "compression_factor": (
                attention_module.compression_factor
                if hasattr(attention_module, "compression_factor")
                else None
            ),
            "use_convs": attention_module.use_convs
            if hasattr(attention_module, "use_convs")
            else None,
            "use_qk_mean": attention_module.use_qk_mean
            if hasattr(attention_module, "use_qk_mean")
            else None,
            "use_qk_norm": attention_module.use_qk_norm
            if hasattr(attention_module, "use_qk_norm")
            else None,
            "use_rope": attention_module.use_rope
            if hasattr(attention_module, "use_rope")
            else None,
        }

        report = CCGQADiagnosticsReport(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            model_config=config,
            speed_results=speed_results,
            memory_results=memory_results,
            gradient_analysis=gradient_analysis,
            learning_results=learning_result,
            component_analysis=component_analysis,
            optimization_recommendations=recommendations,
        )

        print("\n" + "=" * 80)
        print("DIAGNOSTICS COMPLETE")
        print("=" * 80 + "\n")

        return report
