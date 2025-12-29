# Copilot Instructions (Authoritative)

These rules are mandatory for all suggestions and edits in this repository.

---

## 0) Mandatory Repo-First Workflow
Before proposing any code change, you MUST:

1) Identify the exact file(s) to modify (full path + reason).
2) Search the repo for existing implementations of the same concept (loader, logger, scheduler, checkpointing, metrics, etc.) and name the files found.
3) Default to extending existing code; new modules require justification.
4) If unsure whether functionality already exists, propose ONE repo search command and wait.

Hard rule: Do not create parallel or duplicate systems. Prefer integrating a known, production-grade open-source library over inventing new frameworks.

---

## 1) Environment Rules (Non-Negotiable)
- Primary venv: `/home/tim/venvs/llm`
- Never run Python or pip without activating it.
- Every shell command MUST be prefixed with:

  `source /home/tim/venvs/llm/bin/activate && <command>`

- Do NOT suggest creating new virtual environments.
- Use `/home/tim/venvs/llm/bin/python` when referencing the interpreter.
- For new dependencies, provide only the `pip install ...` command.

---

## 2) Output Rules
- Do not reprint unchanged code; use `... existing code ...`.
- Default to delta edits, but refactors are encouraged when they reduce duplication or improve performance.
- Whole-file rewrites are allowed when they improve coherence or efficiency.
- Do not create new files or modules if equivalent functionality already exists in-repo or via a clean OSS dependency.
- Never introduce parallel implementations of the same concept.

---

## 3) torch.compile Hard Constraints
Inside `forward()` or any compiled region:
- No `.item()`, `.tolist()`, `.cpu()`, `.numpy()`, or host sync.
- No printing or logging.
- No Python control flow based on tensor values.
- Avoid Python loops that create tensors; vectorize.
- No dynamic shape changes unless explicitly guarded.

All logging, metrics, and diagnostics must occur outside compiled regions.

---

## 4) Performance-First Coding Defaults

### Required Performance Intent
Before coding, briefly state:
- Expected bottleneck (compute, memory, Python overhead, IO).
- Chosen strategy (vectorize, fuse, cache, reuse existing module).
- Expected improvement (fewer allocs, fewer kernel launches, less Python overhead).

### Implementation Rules
- Performance, memory efficiency, and compile-friendliness take priority over readability.
- Prefer vectorized tensor programs over Python loops.
- Prefer library primitives and fused ops (Torch, Triton, Flash-Attn, xFormers) over bespoke logic.
- Prefer in-place ops and `out=` arguments where safe.
- Avoid slow fallback/reference implementations unless explicitly requested.
- Cache locals and pre-allocate buffers in hot paths.
- Use strict type hints when they aid static analysis or compile stability.

Memory hygiene:
- Prefer `.view()`, `.expand()`, `.as_strided()` over `.reshape()` or `.repeat()` when avoiding copies matters.
- Use `register_buffer()` for all non-parameter state.

---

## 5) DRY & Abstraction Rules
- If logic appears more than once, refactor into a shared function, mixin, or canonical utility.
- Prefer composition and data-driven configuration over copy/paste variants.
- Use inheritance only when it removes duplication without adding indirection.
- Avoid abstraction layers that do not provide measurable performance or structural benefit.

---

## 6) Numba Policy
- Use `@numba.njit(cache=True, fastmath=True)` only when vectorization is not feasible.
- Must be `nopython=True`; refactor rather than falling back to object mode.
- Use `parallel=True` and `numba.prange` only when safe.
- Pass all state explicitly; no globals.
- Keep Numba code outside torch.compile regions.

---

## 7) Project Structure (Authoritative)
- Importable code: `hydra/` only.
- Training entrypoint: `trainer.py` (repo root).
- Attention backends: `hydra/attention/backends/<backend>/`
  - Assets colocated: `benchmarks/`, `docs/`, `tests/`, optional `kernels/`
  - Public API shims: `hydra/attention/<backend>.py`
- Kernels: `hydra/kernels/` (CPU-safe imports required).
- Model wiring/factories: `hydra/model/framework/`
  - `hydra/model/ccgqa/` is back-compat only; keep thin.
- Diagnostics: `diagnostics/` (no pytest discovery).
- Tests: `tests/` only.

Rules:
- No new root-level shims.
- Back-compat requires thin internal shims, not new roots.
- Use canonical import paths; do not reach into backend internals.
- Structural changes require updated imports, docs, and a passing pytest run.
- Never modify or delete stable shim APIs.

---

## 8) Observability Defaults
Reuse existing observability tooling; do not invent new systems.
Prefer TensorBoard or Weights & Biases if already present.

Always log (outside forward/compile):
- Loss components, LR, grad norm, tokens/sec, VRAM, step time
- NaNs/overflows
- Checkpoint save/load events

Prefer structured (JSON) logging for training metrics.
