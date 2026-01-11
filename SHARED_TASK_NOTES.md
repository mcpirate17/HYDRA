# Shared Task Notes - Test Coverage Improvement

## Current Status
- Test count: 466 tests (up from 396)
- All tests passing (465 pass, 1 skipped)
- New test files added this iteration:
  - `tests/test_routing_ops.py` - 31 tests for routing operations
  - `tests/test_utils.py` - 7 tests for utility functions
  - `tests/test_optimizers.py` - 32 tests for Lion/Muon/Sophia optimizers

## Fixed Issues
- Fixed flaky `test_import` tests in fused kernel tests - these were asserting
  specific values of feature flags (USE_FUSED_*) which depend on module import
  order. Now only check that Triton is available.

## Priority Areas for Next Iteration

### High Priority (0-25% coverage)
1. **hydra/training/db.py** (25%) - Training metrics database
2. **hydra/training/checkpointing.py** (26%) - Checkpoint save/load
3. **hydra/training/gradients.py** (7%) - Gradient utilities
4. **hydra/training/step_diagnostics.py** (18%) - Step diagnostics
5. **hydra/training/spike_diagnostics.py** (19%) - Spike detection

### Medium Priority (25-50% coverage)
1. **hydra/data/data_filter.py** (25%) - Data filtering
2. **hydra/training/curriculum.py** (41%) - Curriculum learning
3. **hydra/training/safe_optimizations.py** (44%) - Safe optimization wrappers
4. **hydra/kernels/liger_integration.py** (45%) - Liger CE integration

### Notes
- Lightning Attention ops (0%) are third-party vendored code - skip
- Benchmark files (0%) are diagnostic tools, not core functionality - lower priority
- The CLI (hydra/training/cli.py at 10%) would need integration tests

## Quick Commands
```bash
# Run all tests
source /home/tim/venvs/llm/bin/activate && pytest

# Run with coverage for specific module
source /home/tim/venvs/llm/bin/activate && pytest --cov=hydra.training.db --cov-report=term-missing

# Run fast tests only
source /home/tim/venvs/llm/bin/activate && pytest -m "not slow"
```
