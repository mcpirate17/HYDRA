"""HYDRA.diagnostics: convenience API for diagnostic tools.

This package intentionally keeps heavy scripts as separate modules. The
package API provides a small programmatic surface for running diagnostics
from Python and listing available tools.

Notes:
- Heavy/optional tools (GPU, external kernels, transformers) remain as
  standalone scripts under the diagnostics/ folder. Use the CLI entry
  points or `run_tool()` to invoke them programmatically.
"""

from importlib import import_module
from typing import Dict, List

# Lightweight mapping of module short-names -> module path. Keep this list
# curated: add new tools here if they should be exposed via the package API.
_tools: Dict[str, str] = {
	"main": "diagnostics.__main__",
	"benchmark_ccgqa": "diagnostics.benchmark_ccgqa",
	"benchmark_ccgqa_cpu": "diagnostics.benchmark_ccgqa_cpu",
	"collect_pt_by_seq": "diagnostics.collect_pt_by_seq",
	"convert_nemotron": "diagnostics.convert_nemotron_jsonl_to_pt",
	"convert_small_chat": "diagnostics.convert_small_chat_to_pt",
	"deep_diagnosis": "diagnostics.deep_diagnosis",
	"scaling_analysis": "diagnostics.scaling_analysis",
	"throughput_sweep": "diagnostics.throughput_sweep",
	"moe_watchdog": "diagnostics.moe_watchdog",
	"moe_specialization": "diagnostics.moe_specialization",
	"moe_weight_balance": "diagnostics.moe_weight_balance",
	"model_health_check": "diagnostics.model_health_check",
	"routing_healthcheck": "diagnostics.routing_healthcheck",
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

