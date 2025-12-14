"""Pytest configuration and fixtures for HYDRA tests."""

import pytest
import torch
import sys
from pathlib import Path

# Add project root to path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


@pytest.fixture(scope="session")
def device():
    """Get the appropriate device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def dtype():
    """Get the default dtype for testing."""
    return torch.float32


@pytest.fixture
def small_model_config():
    """Configuration for a small test model."""
    return {
        "vocab_size": 1000,
        "dim": 256,
        "n_mor_blocks": 2,
        "recursions": 2,
        "n_heads": 4,
        "n_kv_heads": 2,
        "compression_factor": 4,
        "capacity_ratio": 0.75,
        "max_seq_len": 128,
    }


@pytest.fixture
def medium_model_config():
    """Configuration for a medium test model (100M scale)."""
    return {
        "vocab_size": 32000,
        "dim": 768,
        "n_mor_blocks": 8,
        "recursions": 4,
        "n_heads": 12,
        "n_kv_heads": 3,
        "compression_factor": 4,
        "capacity_ratio": 0.75,
        "max_seq_len": 512,
    }


def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")
    config.addinivalue_line("markers", "large: marks tests for large model variants")
