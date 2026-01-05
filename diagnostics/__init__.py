"""HYDRA.diagnostics: convenience API for diagnostic tools.

This package intentionally keeps heavy scripts as separate modules. The
package API provides a small programmatic surface for running diagnostics
from Python and listing available tools.

Notes:
- Heavy/optional tools (GPU, external kernels, transformers) remain as
  standalone scripts under the diagnostics/ folder. Use the CLI entry
  points or `run_tool()` to invoke them programmatically.
- For attention-level benchmarks (CCGQA vs GQA/Flash/SDPA), see:
  hydra.attention.backends.ccgqa.benchmarks
"""

from importlib import import_module
from typing import Dict, List

# Lightweight mapping of module short-names -> module path. Keep this list
# curated: add new tools here if they should be exposed via the package API.
_tools: Dict[str, str] = {
    "main": "diagnostics.__main__",
    "benchmark_hydra_models": "diagnostics.benchmark_hydra_models",
    "collect_pt_by_seq": "diagnostics.collect_pt_by_seq",
    "convert_nemotron": "diagnostics.convert_nemotron_jsonl_to_pt",
    "convert_small_chat": "diagnostics.convert_small_chat_to_pt",
    "deep_diagnosis": "diagnostics.deep_diagnosis",
    "diagnose_learning": "diagnostics.diagnose_learning",
    "eval_checkpoint": "diagnostics.eval_checkpoint",
    "eval_sanity_diagnostic": "diagnostics.eval_sanity_diagnostic",
    "lightning_attn_healthcheck": "diagnostics.lightning_attn_healthcheck",
    "loss_component_diagnostic": "diagnostics.loss_component_diagnostic",
    "model_health_check": "diagnostics.model_health_check",
    "moe_specialization": "diagnostics.moe_specialization",
    "moe_watchdog": "diagnostics.moe_watchdog",
    "moe_weight_balance": "diagnostics.moe_weight_balance",
    "routing_healthcheck": "diagnostics.routing_healthcheck",
    "scaling_analysis": "diagnostics.scaling_analysis",
    "seq_len_vram_sweep": "diagnostics.seq_len_vram_sweep",
    "sweep_training_knobs": "diagnostics.sweep_training_knobs",
    "tall_skinny_bench": "diagnostics.tall_skinny_bench",
    "throughput_sweep": "diagnostics.throughput_sweep",
}


def list_tools() -> List[str]:
    """Return available diagnostic tool names exposed by this package."""
    return sorted(_tools.keys())


def run_tool(name: str, *args, **kwargs):
    """Import a diagnostics module by name and call its `main()`.

    Example:
        from diagnostics import run_tool
        run_tool("scaling_analysis", args=(...), kwargs={})
    """
    if name not in _tools:
        raise KeyError(f"Unknown diagnostics tool: {name}")
    mod = import_module(_tools[name])
    if hasattr(mod, "main"):
        return mod.main(*args, **kwargs)
    raise AttributeError(f"Module {_tools[name]} has no callable `main`")


def run_suite():
    """Programmatic entry to the diagnostics suite (same as `python -m diagnostics`)."""
    return run_tool("main")
