"""
Tests for hydra/utils.py - General utility functions.
"""

import os
import tempfile
import pytest
import torch
import torch.nn as nn

from hydra.utils import save_model_architecture


class TestSaveModelArchitecture:
    """Tests for save_model_architecture function."""

    def test_creates_file(self):
        """Test that the function creates the output file."""
        model = nn.Linear(10, 5)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "arch.txt")
            save_model_architecture(model, path)

            assert os.path.exists(path)

    def test_creates_directory(self):
        """Test that the function creates parent directories."""
        model = nn.Linear(10, 5)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "nested", "dir", "arch.txt")
            save_model_architecture(model, path)

            assert os.path.exists(path)

    def test_contains_model_str(self):
        """Test that output contains model string representation."""
        model = nn.Linear(10, 5)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "arch.txt")
            save_model_architecture(model, path)

            with open(path) as f:
                content = f.read()

            assert "Linear" in content
            assert "10" in content  # in_features
            assert "5" in content   # out_features

    def test_contains_param_summary(self):
        """Test that output contains parameter summary."""
        model = nn.Linear(10, 5, bias=True)  # 10*5 + 5 = 55 params

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "arch.txt")
            save_model_architecture(model, path)

            with open(path) as f:
                content = f.read()

            assert "Total parameters:" in content
            assert "Trainable parameters:" in content
            assert "Non-trainable parameters:" in content
            assert "55" in content  # Total params

    def test_handles_complex_model(self):
        """Test with a more complex model."""
        model = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "arch.txt")
            save_model_architecture(model, path)

            with open(path) as f:
                content = f.read()

            assert "Sequential" in content
            assert "ReLU" in content

    def test_frozen_params_counted(self):
        """Test that frozen parameters are correctly counted as non-trainable."""
        model = nn.Linear(10, 5)
        # Freeze the model
        for p in model.parameters():
            p.requires_grad = False

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "arch.txt")
            save_model_architecture(model, path)

            with open(path) as f:
                content = f.read()

            # Should have 0 trainable, 55 non-trainable
            assert "Trainable parameters: 0" in content
            assert "Non-trainable parameters: 55" in content

    def test_no_directory_path(self):
        """Test saving to current directory (no dirname)."""
        model = nn.Linear(10, 5)
        filename = "temp_arch_test.txt"

        try:
            save_model_architecture(model, filename)
            assert os.path.exists(filename)
        finally:
            if os.path.exists(filename):
                os.remove(filename)
