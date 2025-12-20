# Mandatory Environment Activation Rules
- **Environment Location**: The primary environment is at `/home/tim/venvs/llm`.
- **Mandatory Activation**: Never run a Python or Pip command in isolation. You MUST prefix every shell command with the activation source.
- **Unified Command Template**: Use this exact format for all execution: 
  `source /home/tim/venvs/llm/bin/activate && <command>`
- **No New Venvs**: Under no circumstances should you suggest creating a new virtual environment (e.g., `python -m venv .venv`). Use the existing one at `/home/tim/venvs/llm`.
- **Absolute Paths**: When referencing the Python interpreter inside the project, use `/home/tim/venvs/llm/bin/python`.
- **Dependency Management**: If new packages are needed, only provide the `pip install` command; do not wrap it in environment creation logic.
# GitHub Copilot Behavior Guidelines
- **No Code Duplication**: Do not re-output code already in the file. Use `...` or `// ... existing code ...` for unchanged sections.
- **Delta Updates Only**: Provide only the specific functions or lines requiring modification.
- **Conciseness**: Keep responses as brief as possible while maintaining functional integrity.

# High-Performance Python & AI Rules
- **Vectorization First**: Always prefer NumPy, PyTorch, or JAX vectorized operations over explicit Python loops or comprehensions.
- **Object Overhead**: Use `__slots__` in high-frequency classes to minimize memory footprint and accelerate attribute access.
- **Memory & In-place Ops**: Prioritize in-place operations (`x.add_()`, `+=`, `*=`) and `out=` arguments to minimize temporary allocations. 
- **Efficiency Contexts**: Use `torch.no_grad()` for inference/aux-calculations and `torch.inference_mode()` where applicable.
- **Scientific Stack**: Use `scipy.special` or `scipy.stats` for complex math to ensure numerical stability and avoid manual overflow handling.
- **Type Hinting**: Use strict type hints to enable better optimization by static analyzers and `torch.compile`.

# Systems & Architectural Rules
- **No Implicit Copies**: Use `.view()`, `.expand()`, or `.as_strided()` instead of `.reshape()` or `.repeat()` to avoid unnecessary memory moves.
- **Graph Integrity**: Never use `.item()`, `.numpy()`, or Python control flow on tensors inside `forward()` passes; maintain a pure-tensor graph for `torch.compile`.
- **Buffer Hygiene**: Use `register_buffer` for all non-parameter state (moving averages, indices, histograms) to ensure device-parity.
- **Data Layout**: Use `memory_format=torch.channels_last` for 4D tensors to leverage hardware Tensor Cores.
- **Collection Efficiency**: Use `collections.deque(maxlen=N)` for moving windows; never use `list.insert(0)`.

# JIT & Scientific Optimization (Numba)
- **Numba njit**: Use `@numba.njit(cache=True, fastmath=True)` for complex logic that cannot be vectorized in PyTorch.
- **GIL-Free Execution**: Enforce `nopython=True`. Refactor code to remove unsupported Python objects rather than falling back to object mode.
- **Persistent Caching**: Always enable `cache=True` to avoid "first-call" latency spikes in production environments.
- **Parallelism**: Use `parallel=True` and `numba.prange` for CPU-bound data-parallel tasks without loop dependencies.
- **Global State**: Pass all necessary data as arguments to jitted functions; do not rely on global variables.
- **Hybrid Guard**: Ensure Numba-jitted functions are called outside of `torch.compile` regions to avoid complex trace-time conflicts.
