# HYDRA Diagnostics

Short overview of diagnostic tools in `diagnostics/`.

Categories
- Quick CLI entry
  - `__main__.py` — top-level runner for common diagnostics.

- Benchmarks / Throughput
  - `benchmark_ccgqa.py` — GPU benchmark for CCGQA and models. REQUIRES CUDA for full results.
  - `benchmark_ccgqa_cpu.py` — CPU micro-bench for regression signals.
  - `tall_skinny_bench.py`, `throughput_sweep.py`, `seq_len_vram_sweep.py` — throughput/VRAM sweeps. REQUIRES CUDA.

- Model / Training diagnostics
  - `deep_diagnosis.py` — gradient flow and layer-level checks. GPU recommended.
  - `model_health_check.py` — CPU-friendly model sanity checks.
  - `eval_checkpoint.py` — load/evaluate a checkpoint (default device: CUDA).
  - `eval_sanity_diagnostic.py`, `loss_component_diagnostic.py` — short-run diagnostics focused on loss components.

- MoE-specific tools
  - `moe_specialization.py` — expert specialization analysis (inference mode uses CUDA; weight-only mode runs on CPU).
  - `moe_watchdog.py` — runtime monitor for checkpoints/logs (runs on CPU but benefits from GPU checkpoints).
  - `moe_weight_balance.py` — checkpoint weight-norm analysis (CPU).

- Data converters / helpers
  - `convert_nemotron_jsonl_to_pt.py`, `convert_small_chat_to_pt.py`, `collect_pt_by_seq.py` — dataset converters/collectors. `transformers` required for tokenization converters.

- External kernel checks
  - `lightning_attn_healthcheck.py` — tests for lightning-attention kernels. REQUIRES CUDA and `lightning_attn`.

Usage notes
- Many diagnostics are heavy and intended to run on dedicated GPU nodes. Read the short header at the top of each script for required dependencies.
- Programmatic API: from Python you can call `diagnostics.list_tools()` and `diagnostics.run_tool(name)` to invoke modules that expose `main()`.

Next steps
- Run `pytest` to validate moved tests: `pytest -q`.
- I can annotate more scripts with dependency hints, consolidate overlapping diagnostics, or add CI hooks to run a smoke subset.
