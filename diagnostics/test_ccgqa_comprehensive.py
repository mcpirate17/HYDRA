"""
CCGQA Diagnostics Test Suite and Report Generation

Runs comprehensive diagnostics on CCGQA attention and generates
detailed reports with graphs and optimization recommendations.
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from dataclasses import asdict
from typing import List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from hydra.model.ccgqa import CCGQAAttention, CCGQABlock, CCGQAModel
from hydra.model.hybrid_attention import (
    MQAAttention,
    CCQAAttention,
    MLAAttention,
    HybridAttentionBlock,
    MacroBlock,
    HybridTransformer,
    HybridTransformerConfig,
    AttentionType,
    create_hybrid_transformer_small,
    create_hybrid_transformer_medium,
)
from diagnostics.ccgqa_diagnostics import CCGQADiagnostician, CCGQADiagnosticsReport


class CCGQATestSuite:
    """
    Comprehensive test suite for CCGQA attention mechanism.

    Runs diagnostics on various configurations and generates reports.
    """

    def __init__(self, reports_dir: str = "reports"):
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(exist_ok=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def test_attention_layer(self) -> CCGQADiagnosticsReport:
        """Test base CCGQA attention layer."""
        print("\n" + "=" * 80)
        print("TESTING: CCGQA Attention Layer")
        print("=" * 80)

        attention = CCGQAAttention(
            dim=768,
            n_heads=12,
            n_kv_heads=3,
            compression_factor=4,
            max_seq_len=2048,
            use_rope=True,
            use_qk_norm=True,
            use_convs=True,
            use_qk_mean=True,
            use_value_shift=True,
            conv_kernel_size=3,
        )

        diagnostician = CCGQADiagnostician(device=self.device)
        report = diagnostician.run_full_diagnostics(
            attention,
            batch_sizes=[1, 4, 8],
            seq_lengths=[256, 512, 1024],
        )

        return report

    def test_transformer_block(self) -> CCGQADiagnosticsReport:
        """Test CCGQA block with MLP."""
        print("\n" + "=" * 80)
        print("TESTING: CCGQA Transformer Block")
        print("=" * 80)

        block = CCGQABlock(
            dim=768,
            n_heads=12,
            n_kv_heads=3,
            compression_factor=4,
            mlp_ratio=2.67,
            max_seq_len=2048,
            norm_eps=1e-6,
        )

        diagnostician = CCGQADiagnostician(device=self.device)
        report = diagnostician.run_full_diagnostics(
            block,
            batch_sizes=[1, 4, 8],
            seq_lengths=[256, 512, 1024],
        )

        return report

    def test_full_model(self) -> CCGQADiagnosticsReport:
        """Test full CCGQA model."""
        print("\n" + "=" * 80)
        print("TESTING: Full CCGQA Model (4 layers)")
        print("=" * 80)

        model = CCGQAModel(
            vocab_size=50257,
            dim=768,
            n_layers=4,  # Reduced for faster testing
            n_heads=12,
            n_kv_heads=3,
            compression_factor=4,
            mlp_ratio=2.67,
            max_seq_len=2048,
        )

        diagnostician = CCGQADiagnostician(device=self.device)

        # Wrap model in attention-like interface for diagnostician
        class ModelWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
                self.dim = model.dim
                self.n_heads = model.layers[0].attention.n_heads
                self.n_kv_heads = model.layers[0].attention.n_kv_heads
                self.compression_factor = model.layers[0].attention.compression_factor
                self.use_rope = model.layers[0].attention.use_rope
                self.use_qk_norm = model.layers[0].attention.use_qk_norm
                self.use_convs = model.layers[0].attention.use_convs
                self.use_qk_mean = model.layers[0].attention.use_qk_mean

            def forward(self, x, mask=None):
                # For full model, x should be token IDs
                return self.model(x)

        wrapper = ModelWrapper(model)
        report = diagnostician.run_full_diagnostics(
            wrapper,
            batch_sizes=[1, 4],
            seq_lengths=[128, 256],
        )

        return report

    def test_compression_variants(self) -> Dict[int, CCGQADiagnosticsReport]:
        """Test different compression factors."""
        print("\n" + "=" * 80)
        print("TESTING: Compression Factor Variants")
        print("=" * 80)

        compression_factors = [2, 4, 8]
        reports = {}

        for cf in compression_factors:
            print(f"\nTesting compression_factor={cf}")
            attention = CCGQAAttention(
                dim=768,
                n_heads=12,
                n_kv_heads=3,
                compression_factor=cf,
                max_seq_len=2048,
            )

            diagnostician = CCGQADiagnostician(device=self.device)
            report = diagnostician.run_full_diagnostics(
                attention,
                batch_sizes=[1, 4],
                seq_lengths=[256, 512],
            )
            reports[cf] = report

        return reports

    def test_hybrid_attention_layers(self) -> Dict[str, CCGQADiagnosticsReport]:
        """Test hybrid attention layer variants (MQA, CCQA, MLA)."""
        print("\n" + "=" * 80)
        print("TESTING: Hybrid Attention Layer Variants")
        print("=" * 80)

        reports = {}
        dim = 768
        n_heads = 12

        # Test MQA
        print("\nTesting MQA (Multi-Query Attention)...")
        mqa = MQAAttention(dim=dim, n_heads=n_heads, max_seq_len=2048)
        diagnostician = CCGQADiagnostician(device=self.device)
        report_mqa = diagnostician.run_full_diagnostics(
            mqa,
            batch_sizes=[1, 4],
            seq_lengths=[256, 512],
        )
        reports["mqa"] = report_mqa

        # Test CCQA
        print("\nTesting CCQA (Compressed Convolutional Query Attention)...")
        ccqa = CCQAAttention(
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=3,
            compression_factor=4,
            max_seq_len=2048,
            qk_modulation_gain=0.5,
        )
        diagnostician = CCGQADiagnostician(device=self.device)
        report_ccqa = diagnostician.run_full_diagnostics(
            ccqa,
            batch_sizes=[1, 4],
            seq_lengths=[256, 512],
        )
        reports["ccqa"] = report_ccqa

        # Test MLA
        print("\nTesting MLA (Multi-head Latent Attention)...")
        mla = MLAAttention(
            dim=dim,
            n_heads=n_heads,
            latent_ratio=0.5,
            max_seq_len=2048,
        )
        diagnostician = CCGQADiagnostician(device=self.device)
        report_mla = diagnostician.run_full_diagnostics(
            mla,
            batch_sizes=[1, 4],
            seq_lengths=[256, 512],
        )
        reports["mla"] = report_mla

        return reports

    def test_hybrid_macro_block(self) -> CCGQADiagnosticsReport:
        """Test hybrid 8-layer macro-block."""
        print("\n" + "=" * 80)
        print("TESTING: Hybrid Macro-Block (8-layer pattern)")
        print("=" * 80)

        macro_block = MacroBlock(
            dim=768,
            n_heads=12,
            n_kv_heads=3,
            compression_factor=4,
            mlp_ratio=3.5,
            max_seq_len=2048,
        )

        diagnostician = CCGQADiagnostician(device=self.device)
        report = diagnostician.run_full_diagnostics(
            macro_block,
            batch_sizes=[1, 4],
            seq_lengths=[128, 256],
        )

        return report

    def test_hybrid_transformer(self) -> CCGQADiagnosticsReport:
        """Test full hybrid transformer model (24 layers)."""
        print("\n" + "=" * 80)
        print("TESTING: Hybrid Transformer (~220M params)")
        print("=" * 80)

        model = create_hybrid_transformer_small()

        # Wrapper that uses just the attention/MLP stack (not embedding)
        # This allows the diagnostician to pass float tensors directly
        class HybridModelWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.macro_blocks = model.macro_blocks
                self.final_norm = model.final_norm
                self.dim = model.config.dim
                self.n_heads = model.config.n_heads
                self.n_kv_heads = model.config.n_kv_heads
                self.compression_factor = model.config.compression_factor
                self.use_rope = True
                self.use_qk_norm = True
                self.use_convs = True
                self.use_qk_mean = True
                # Add latent_dim for component analysis
                self.latent_dim = model.config.dim // model.config.compression_factor

            def forward(self, x, mask=None):
                # x is already float tensor from diagnostician
                h = x
                for macro_block in self.macro_blocks:
                    h = macro_block(h, mask=mask)
                return self.final_norm(h)

        wrapper = HybridModelWrapper(model)
        diagnostician = CCGQADiagnostician(device=self.device)
        report = diagnostician.run_full_diagnostics(
            wrapper,
            batch_sizes=[1, 2],
            seq_lengths=[64, 128],
        )

        return report

    def save_report_json(self, report: CCGQADiagnosticsReport, name: str) -> Path:
        """Save report to JSON file."""

        # Convert report to dict, handling nested dataclasses
        def convert_to_dict(obj):
            if isinstance(obj, list):
                return [convert_to_dict(item) for item in obj]
            elif hasattr(obj, "__dataclass_fields__"):
                return {k: convert_to_dict(v) for k, v in asdict(obj).items()}
            elif isinstance(obj, (bool, np.bool_)):
                return bool(obj)
            elif isinstance(obj, (int, np.integer)):
                return int(obj)
            elif isinstance(obj, (float, np.floating)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_dict(v) for k, v in obj.items()}
            else:
                return obj

        report_dict = convert_to_dict(report)

        filepath = self.reports_dir / f"{name}_report.json"
        with open(filepath, "w") as f:
            json.dump(report_dict, f, indent=2, default=str)

        print(f"✅ Report saved: {filepath}")
        return filepath

    def generate_speed_graphs(self, report: CCGQADiagnosticsReport, name: str) -> Path:
        """Generate speed benchmark graphs."""
        if not report.speed_results:
            return None

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"CCGQA Speed Benchmarks - {name}", fontsize=16, fontweight="bold")

        # Extract data
        batch_sizes = sorted(set(r.batch_size for r in report.speed_results))
        seq_lengths = sorted(set(r.seq_len for r in report.speed_results))

        # Forward time vs seq length (per batch size)
        ax = axes[0, 0]
        for bs in batch_sizes:
            results = [r for r in report.speed_results if r.batch_size == bs]
            results = sorted(results, key=lambda x: x.seq_len)
            ax.plot(
                [r.seq_len for r in results],
                [r.forward_time_ms for r in results],
                marker="o",
                label=f"B={bs}",
            )
        ax.set_xlabel("Sequence Length")
        ax.set_ylabel("Forward Time (ms)")
        ax.set_title("Forward Pass Latency")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Backward time vs seq length
        ax = axes[0, 1]
        for bs in batch_sizes:
            results = [r for r in report.speed_results if r.batch_size == bs]
            results = sorted(results, key=lambda x: x.seq_len)
            ax.plot(
                [r.seq_len for r in results],
                [r.backward_time_ms for r in results],
                marker="s",
                label=f"B={bs}",
            )
        ax.set_xlabel("Sequence Length")
        ax.set_ylabel("Backward Time (ms)")
        ax.set_title("Backward Pass Latency")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Throughput
        ax = axes[1, 0]
        for bs in batch_sizes:
            results = [r for r in report.speed_results if r.batch_size == bs]
            results = sorted(results, key=lambda x: x.seq_len)
            ax.plot(
                [r.seq_len for r in results],
                [r.throughput_samples_per_sec for r in results],
                marker="^",
                label=f"B={bs}",
            )
        ax.set_xlabel("Sequence Length")
        ax.set_ylabel("Throughput (samples/sec)")
        ax.set_title("Throughput")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # FLOPs efficiency
        ax = axes[1, 1]
        for bs in batch_sizes:
            results = [r for r in report.speed_results if r.batch_size == bs]
            results = sorted(results, key=lambda x: x.seq_len)
            ax.plot(
                [r.seq_len for r in results],
                [r.flops_per_sec / 1e12 for r in results],  # Convert to TFLOPs
                marker="d",
                label=f"B={bs}",
            )
        ax.set_xlabel("Sequence Length")
        ax.set_ylabel("Compute Efficiency (TFLOPS)")
        ax.set_title("FLOPs Efficiency")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        filepath = self.reports_dir / f"{name}_speed_benchmarks.png"
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"✅ Speed graphs saved: {filepath}")
        return filepath

    def generate_memory_graphs(self, report: CCGQADiagnosticsReport, name: str) -> Path:
        """Generate memory profiling graphs."""
        if not report.memory_results:
            return None

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"CCGQA Memory Profiling - {name}", fontsize=16, fontweight="bold")

        batch_sizes = sorted(set(r.batch_size for r in report.memory_results))

        # Peak memory vs seq length
        ax = axes[0, 0]
        for bs in batch_sizes:
            results = [r for r in report.memory_results if r.batch_size == bs]
            results = sorted(results, key=lambda x: x.seq_len)
            ax.plot(
                [r.seq_len for r in results],
                [r.peak_memory_mb for r in results],
                marker="o",
                label=f"B={bs}",
            )
        ax.set_xlabel("Sequence Length")
        ax.set_ylabel("Peak Memory (MB)")
        ax.set_title("Peak Memory Usage")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Memory breakdown
        ax = axes[0, 1]
        if report.memory_results:
            r = report.memory_results[0]
            labels = ["Parameters", "Activations"]
            sizes = [r.parameter_memory_mb, r.activation_memory_mb]
            colors = ["#ff9999", "#66b3ff"]
            ax.pie(
                sizes, labels=labels, autopct="%1.1f%%", colors=colors, startangle=90
            )
            ax.set_title("Memory Breakdown (typical)")

        # Memory per sample vs batch size
        ax = axes[1, 0]
        batch_size_list = sorted(set(r.batch_size for r in report.memory_results))
        for sl in [
            r.seq_len
            for r in report.memory_results
            if r.batch_size == batch_size_list[0]
        ]:
            results = [r for r in report.memory_results if r.seq_len == sl]
            results = sorted(results, key=lambda x: x.batch_size)
            ax.plot(
                [r.batch_size for r in results],
                [r.memory_per_sample_mb for r in results],
                marker="s",
                label=f"S={sl}",
            )
        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Memory per Sample (MB)")
        ax.set_title("Memory per Sample Scaling")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Memory utilization stacked bar
        ax = axes[1, 1]
        if report.memory_results:
            x_labels = [
                f"B{r.batch_size}S{r.seq_len}" for r in report.memory_results[:5]
            ]
            param_mem = [r.parameter_memory_mb for r in report.memory_results[:5]]
            act_mem = [r.activation_memory_mb for r in report.memory_results[:5]]

            x_pos = np.arange(len(x_labels))
            ax.bar(x_pos, param_mem, label="Parameters", color="#ff9999")
            ax.bar(
                x_pos, act_mem, bottom=param_mem, label="Activations", color="#66b3ff"
            )
            ax.set_xticks(x_pos)
            ax.set_xticklabels(x_labels, rotation=45)
            ax.set_ylabel("Memory (MB)")
            ax.set_title("Memory Composition")
            ax.legend()
            ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        filepath = self.reports_dir / f"{name}_memory_profiling.png"
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"✅ Memory graphs saved: {filepath}")
        return filepath

    def generate_analysis_graphs(
        self, report: CCGQADiagnosticsReport, name: str
    ) -> Path:
        """Generate gradient flow and learning analysis graphs."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"CCGQA Analysis - {name}", fontsize=16, fontweight="bold")

        # Gradient norms comparison
        ax = axes[0, 0]
        grad_data = {
            "Q": report.gradient_analysis.grad_norm_q,
            "K": report.gradient_analysis.grad_norm_k,
            "V": report.gradient_analysis.grad_norm_v,
            "Output": report.gradient_analysis.grad_norm_o,
        }
        bars = ax.bar(
            grad_data.keys(),
            grad_data.values(),
            color=["#ff6b6b", "#4ecdc4", "#45b7d1", "#ffa502"],
        )
        ax.set_ylabel("Gradient Norm")
        ax.set_title("Gradient Norms by Component")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3, axis="y")

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.2e}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        # Gradient statistics
        ax = axes[0, 1]
        ax.axis("off")
        gradient_text = (
            f"Gradient Flow Statistics\n"
            f"{'=' * 40}\n"
            f"Mean Norm: {report.gradient_analysis.mean_grad_norm:.2e}\n"
            f"Max Norm: {report.gradient_analysis.max_grad_norm:.2e}\n"
            f"Min Norm: {report.gradient_analysis.min_grad_norm:.2e}\n"
            f"\n"
            f"Status:\n"
            f"  Vanishing: {report.gradient_analysis.vanishing_gradient_detected}\n"
            f"  Exploding: {report.gradient_analysis.exploding_gradient_detected}\n"
        )
        ax.text(
            0.1,
            0.5,
            gradient_text,
            fontfamily="monospace",
            fontsize=10,
            verticalalignment="center",
            bbox=dict(boxstyle="round", facecolor="#f0f0f0"),
        )

        # Learning curve
        ax = axes[1, 0]
        learning = report.learning_results
        ax.text(
            0.5,
            0.8,
            f"Learning Diagnostics",
            fontsize=12,
            fontweight="bold",
            ha="center",
            transform=ax.transAxes,
        )
        learning_text = (
            f"Initial Loss: {learning.initial_loss:.4f}\n"
            f"Final Loss: {learning.final_loss:.4f}\n"
            f"Improvement: {learning.loss_improvement:.2%}\n"
            f"Convergence Steps: {learning.convergence_steps}\n"
            f"Gradient Flow Score: {learning.gradient_flow_score:.2f}\n"
            f"Learning Capacity Score: {learning.learning_capacity_score:.2f}"
        )
        ax.text(
            0.5,
            0.4,
            learning_text,
            fontsize=11,
            ha="center",
            transform=ax.transAxes,
            verticalalignment="center",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="#e8f5e9"),
        )
        ax.axis("off")

        # Component summary
        ax = axes[1, 1]
        ax.axis("off")
        comp = report.component_analysis
        comp_text = (
            f"Component Analysis\n"
            f"{'=' * 40}\n"
            f"Compression: {comp.compression_effectiveness.get('compression_factor', 'N/A')}x\n"
            f"Param Reduction: {comp.compression_effectiveness.get('param_reduction_ratio', 0):.1%}\n"
            f"Latent Dim: {comp.compression_effectiveness.get('latent_dim', 'N/A')}\n"
            f"\n"
            f"GQA Ratio: {comp.head_reshaping_efficiency.get('gqa_ratio', 'N/A')}x\n"
            f"KV-Cache Reduction: {comp.head_reshaping_efficiency.get('kv_cache_reduction', 0):.1%}\n"
            f"\n"
            f"QK-Mean: {comp.qk_mean_impact.get('enabled', False)}\n"
            f"QK-Norm: {comp.normalization_stats.get('qk_norm_enabled', False)}\n"
            f"Convolutions: {comp.compression_effectiveness.get('convolutions_enabled', False)}"
        )
        ax.text(
            0.1,
            0.5,
            comp_text,
            fontfamily="monospace",
            fontsize=10,
            verticalalignment="center",
            bbox=dict(boxstyle="round", facecolor="#fff3e0"),
        )

        plt.tight_layout()
        filepath = self.reports_dir / f"{name}_analysis.png"
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"✅ Analysis graphs saved: {filepath}")
        return filepath

    def generate_summary_report(
        self, reports: Dict[str, CCGQADiagnosticsReport]
    ) -> Path:
        """Generate a text summary report."""
        filepath = self.reports_dir / "CCGQA_DIAGNOSTICS_SUMMARY.txt"

        with open(filepath, "w") as f:
            f.write("=" * 100 + "\n")
            f.write("COMPREHENSIVE CCGQA DIAGNOSTICS REPORT\n")
            f.write("=" * 100 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Device: cuda\n\n")

            for test_name, report in reports.items():
                f.write("\n" + "=" * 100 + "\n")
                f.write(f"TEST: {test_name}\n")
                f.write("=" * 100 + "\n\n")

                # Model config
                f.write("MODEL CONFIGURATION\n")
                f.write("-" * 100 + "\n")
                for key, value in report.model_config.items():
                    f.write(f"  {key}: {value}\n")

                # Speed summary
                if report.speed_results:
                    f.write("\n\nSPEED BENCHMARKS\n")
                    f.write("-" * 100 + "\n")
                    f.write(
                        f"{'Batch':<8} {'Seq Len':<10} {'Forward':<12} {'Backward':<12} {'Total':<12} {'FLOPS/s':<15}\n"
                    )
                    f.write("-" * 100 + "\n")
                    for r in report.speed_results:
                        f.write(
                            f"{r.batch_size:<8} {r.seq_len:<10} {r.forward_time_ms:<12.2f} "
                            f"{r.backward_time_ms:<12.2f} {r.total_time_ms:<12.2f} {r.flops_per_sec / 1e12:<15.2f}T\n"
                        )

                # Memory summary
                if report.memory_results:
                    f.write("\n\nMEMORY PROFILING\n")
                    f.write("-" * 100 + "\n")
                    f.write(
                        f"{'Batch':<8} {'Seq Len':<10} {'Peak MB':<12} {'Activation MB':<15} {'Per-Sample MB':<15}\n"
                    )
                    f.write("-" * 100 + "\n")
                    for r in report.memory_results:
                        f.write(
                            f"{r.batch_size:<8} {r.seq_len:<10} {r.peak_memory_mb:<12.2f} "
                            f"{r.activation_memory_mb:<15.2f} {r.memory_per_sample_mb:<15.2f}\n"
                        )

                # Gradient analysis
                f.write("\n\nGRADIENT FLOW ANALYSIS\n")
                f.write("-" * 100 + "\n")
                g = report.gradient_analysis
                f.write(f"  Mean Gradient Norm: {g.mean_grad_norm:.2e}\n")
                f.write(f"  Max Gradient Norm: {g.max_grad_norm:.2e}\n")
                f.write(f"  Min Gradient Norm: {g.min_grad_norm:.2e}\n")
                f.write(f"  Vanishing Gradients: {g.vanishing_gradient_detected}\n")
                f.write(f"  Exploding Gradients: {g.exploding_gradient_detected}\n")

                # Learning results
                f.write("\n\nLEARNING CAPABILITY\n")
                f.write("-" * 100 + "\n")
                l = report.learning_results
                f.write(f"  Initial Loss: {l.initial_loss:.4f}\n")
                f.write(f"  Final Loss: {l.final_loss:.4f}\n")
                f.write(f"  Loss Improvement: {l.loss_improvement:.2%}\n")
                f.write(f"  Convergence Steps: {l.convergence_steps}\n")
                f.write(f"  Learning Capacity Score: {l.learning_capacity_score:.2f}\n")

                # Component analysis
                f.write("\n\nCOMPONENT ANALYSIS\n")
                f.write("-" * 100 + "\n")
                c = report.component_analysis
                f.write("\nCompression Effectiveness:\n")
                for key, value in c.compression_effectiveness.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\nHead Reshaping Efficiency:\n")
                for key, value in c.head_reshaping_efficiency.items():
                    f.write(f"  {key}: {value}\n")

                # Recommendations
                f.write("\n\nOPTIMIZATION RECOMMENDATIONS\n")
                f.write("-" * 100 + "\n")
                for i, rec in enumerate(report.optimization_recommendations, 1):
                    # Remove any non-ASCII characters from recommendations
                    clean_rec = rec.encode("ascii", "ignore").decode("ascii")
                    f.write(f"\n{i}. {clean_rec}\n")

        print(f"✅ Summary report saved: {filepath}")
        return filepath

    def run_all_tests(self):
        """Run all diagnostic tests and generate reports."""
        print("\n" + "=" * 100)
        print("CCGQA & HYBRID ATTENTION COMPREHENSIVE DIAGNOSTICS TEST SUITE")
        print("=" * 100)

        reports = {}

        # Test 1: Attention Layer
        print("\n[1/6] Testing CCGQA Attention Layer...")
        report_attn = self.test_attention_layer()
        reports["01_attention_layer"] = report_attn
        self.save_report_json(report_attn, "01_attention_layer")
        self.generate_speed_graphs(report_attn, "01_attention_layer")
        self.generate_memory_graphs(report_attn, "01_attention_layer")
        self.generate_analysis_graphs(report_attn, "01_attention_layer")

        # Test 2: Transformer Block
        print("\n[2/6] Testing CCGQA Transformer Block...")
        report_block = self.test_transformer_block()
        reports["02_transformer_block"] = report_block
        self.save_report_json(report_block, "02_transformer_block")
        self.generate_speed_graphs(report_block, "02_transformer_block")
        self.generate_memory_graphs(report_block, "02_transformer_block")
        self.generate_analysis_graphs(report_block, "02_transformer_block")

        # Test 3: Compression Variants
        print("\n[3/6] Testing Compression Factor Variants...")
        compression_reports = self.test_compression_variants()
        for cf, report in compression_reports.items():
            test_name = f"03_compression_factor_{cf}x"
            reports[test_name] = report
            self.save_report_json(report, test_name)
            self.generate_speed_graphs(report, test_name)
            self.generate_memory_graphs(report, test_name)
            self.generate_analysis_graphs(report, test_name)

        # Test 4: Hybrid Attention Layers (MQA, CCQA, MLA)
        print("\n[4/6] Testing Hybrid Attention Layers...")
        hybrid_layer_reports = self.test_hybrid_attention_layers()
        for attn_type, report in hybrid_layer_reports.items():
            test_name = f"04_hybrid_{attn_type}"
            reports[test_name] = report
            self.save_report_json(report, test_name)
            self.generate_speed_graphs(report, test_name)
            self.generate_memory_graphs(report, test_name)
            self.generate_analysis_graphs(report, test_name)

        # Test 5: Hybrid Macro-Block
        print("\n[5/6] Testing Hybrid Macro-Block...")
        report_macro = self.test_hybrid_macro_block()
        reports["05_hybrid_macro_block"] = report_macro
        self.save_report_json(report_macro, "05_hybrid_macro_block")
        self.generate_speed_graphs(report_macro, "05_hybrid_macro_block")
        self.generate_memory_graphs(report_macro, "05_hybrid_macro_block")
        self.generate_analysis_graphs(report_macro, "05_hybrid_macro_block")

        # Test 6: Hybrid Transformer
        print("\n[6/6] Testing Hybrid Transformer (~220M)...")
        report_hybrid_model = self.test_hybrid_transformer()
        reports["06_hybrid_transformer"] = report_hybrid_model
        self.save_report_json(report_hybrid_model, "06_hybrid_transformer")
        self.generate_speed_graphs(report_hybrid_model, "06_hybrid_transformer")
        self.generate_memory_graphs(report_hybrid_model, "06_hybrid_transformer")
        self.generate_analysis_graphs(report_hybrid_model, "06_hybrid_transformer")

        # Generate summary
        print("\n[7/7] Generating Summary Report...")
        self.generate_summary_report(reports)

        print("\n" + "=" * 100)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 100)
        print(f"\n[REPORT] Reports saved to: {self.reports_dir.absolute()}")
        print("\nGenerated files:")
        for file in sorted(self.reports_dir.glob("*")):
            print(f"  - {file.name}")


if __name__ == "__main__":
    test_suite = CCGQATestSuite(reports_dir="reports")
    test_suite.run_all_tests()
