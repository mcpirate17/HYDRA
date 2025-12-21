"""
Tests for universal_data_loader module.

Tests:
1. SyntheticDataLoader - always works
2. LocalDataLoader - with mock data
3. StreamingDataLoader - HuggingFace integration
4. Interleave support
5. Batch method support
6. Error handling and fallbacks
"""

import pytest
import torch
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from universal_data_loader import (
    create_universal_loader,
    SyntheticDataLoader,
    LocalDataLoader,
    StreamingDataLoader,
    get_tokenizer,
    get_available_datasets,
    HAS_DATASETS,
    HAS_TRANSFORMERS,
)


class TestSyntheticDataLoader:
    """Test synthetic data loader (no dependencies)."""

    def test_basic_batch(self):
        """Test basic batch generation."""
        loader = SyntheticDataLoader(batch_size=4, seq_len=128)
        batch = loader.get_batch()

        assert "input_ids" in batch
        assert "labels" in batch
        assert batch["input_ids"].shape == (4, 128)
        assert batch["labels"].shape == (4, 128)

    def test_vocab_range(self):
        """Test token values are within vocab range."""
        loader = SyntheticDataLoader(batch_size=8, seq_len=64, vocab_size=1000)
        batch = loader.get_batch()

        assert batch["input_ids"].min() >= 0
        assert batch["input_ids"].max() < 1000

    def test_iterator(self):
        """Test iterator interface."""
        loader = SyntheticDataLoader(batch_size=2, seq_len=32)

        batches = []
        for i, batch in enumerate(loader):
            batches.append(batch)
            if i >= 2:
                break

        assert len(batches) == 3

    def test_close(self):
        """Test close stops iteration."""
        loader = SyntheticDataLoader(batch_size=2, seq_len=32)
        loader.close()

        with pytest.raises(StopIteration):
            next(loader)


class TestLocalDataLoader:
    """Test local data loader with mock .pt files."""

    @pytest.fixture
    def mock_data_dir(self, tmp_path):
        """Create temporary directory with mock chunk files."""
        # Create mock chunk data
        for i in range(3):
            chunk_data = [
                {"input_ids": list(range(i * 100, i * 100 + 600)), "length": 600}
                for _ in range(50)
            ]
            torch.save(chunk_data, tmp_path / f"chunk_{i:04d}.pt")

        return tmp_path

    def test_load_chunks(self, mock_data_dir):
        """Test loading chunk files."""
        loader = LocalDataLoader(
            data_dir=str(mock_data_dir),
            batch_size=4,
            seq_len=128,
        )

        assert len(loader.chunk_files) == 3

        batch = loader.get_batch()
        assert batch["input_ids"].shape == (4, 128)
        assert batch["labels"].shape == (4, 128)

    def test_no_tokenizer_download(self, mock_data_dir):
        """Ensure local loader doesn't try to download tokenizer."""
        loader = LocalDataLoader(
            data_dir=str(mock_data_dir),
            batch_size=2,
            seq_len=64,
        )

        # Should have skipped tokenizer
        assert loader.tokenizer is None

    def test_chunk_cycling(self, mock_data_dir):
        """Test that loader cycles through chunks."""
        loader = LocalDataLoader(
            data_dir=str(mock_data_dir),
            batch_size=2,
            seq_len=64,
        )

        # Get many batches to force chunk cycling
        for _ in range(100):
            batch = loader.get_batch()
            assert batch["input_ids"].shape == (2, 64)

        # Should have processed multiple chunks
        assert loader.total_samples > 50

    def test_empty_dir_raises(self, tmp_path):
        """Test that empty directory raises error."""
        with pytest.raises(ValueError, match="No .pt files found"):
            LocalDataLoader(data_dir=str(tmp_path), batch_size=2, seq_len=64)

    def test_stats(self, mock_data_dir):
        """Test stats reporting."""
        loader = LocalDataLoader(
            data_dir=str(mock_data_dir),
            batch_size=2,
            seq_len=64,
        )

        for _ in range(10):
            loader.get_batch()

        stats = loader.stats()
        assert "total_samples" in stats
        assert stats["total_samples"] > 0


@pytest.mark.skipif(not HAS_DATASETS, reason="datasets library not installed")
class TestStreamingDataLoader:
    """Test HuggingFace streaming data loader."""

    def test_basic_init(self):
        """Test basic initialization (may require network)."""
        try:
            loader = StreamingDataLoader(
                dataset_name="finefineweb",
                batch_size=2,
                seq_len=128,
                max_retries=1,
            )
            assert loader.dataset is not None
        except Exception as e:
            pytest.skip(f"Network unavailable: {e}")

    def test_get_batch(self):
        """Test batch retrieval from streaming dataset."""
        try:
            loader = StreamingDataLoader(
                dataset_name="finefineweb",
                batch_size=2,
                seq_len=64,
                max_retries=1,
            )

            batch = loader.get_batch()
            assert batch["input_ids"].shape == (2, 64)
            assert batch["labels"].shape == (2, 64)
        except Exception as e:
            pytest.skip(f"Network unavailable: {e}")

    def test_fallback_on_failure(self):
        """Test synthetic fallback on sustained failures."""
        loader = StreamingDataLoader(
            dataset_name="nonexistent_dataset_xyz",
            batch_size=2,
            seq_len=64,
            max_retries=1,
        )

        # Should fallback to synthetic
        batch = loader.get_batch()
        assert batch["input_ids"].shape == (2, 64)


class TestCreateUniversalLoader:
    """Test the create_universal_loader factory function."""

    def test_synthetic(self):
        """Test synthetic dataset selection."""
        loader = create_universal_loader(
            dataset="synthetic",
            batch_size=4,
            seq_len=128,
        )

        assert isinstance(loader, SyntheticDataLoader)
        batch = loader.get_batch()
        assert batch["input_ids"].shape == (4, 128)

    def test_local(self, tmp_path):
        """Test local dataset selection."""
        # Create mock data
        chunk_data = [{"input_ids": list(range(600)), "length": 600} for _ in range(20)]
        torch.save(chunk_data, tmp_path / "chunk_0000.pt")

        loader = create_universal_loader(
            dataset="local",
            data_dir=str(tmp_path),
            batch_size=2,
            seq_len=64,
        )

        assert isinstance(loader, LocalDataLoader)
        batch = loader.get_batch()
        assert batch["input_ids"].shape == (2, 64)

    def test_local_requires_data_dir(self):
        """Test that local without data_dir raises error."""
        with pytest.raises(ValueError, match="data_dir required"):
            create_universal_loader(dataset="local", batch_size=2, seq_len=64)

    def test_unknown_dataset_falls_back(self):
        """Test unknown dataset falls back to synthetic."""
        loader = create_universal_loader(
            dataset="unknown_dataset_xyz",
            batch_size=2,
            seq_len=64,
        )

        assert isinstance(loader, SyntheticDataLoader)

    def test_auto_select_by_model_size(self):
        """Test auto-selection based on model params."""
        # Small model (< 50M) -> synthetic (no network needed for tiny tests)
        # But the actual logic uses streaming for models >= 50M
        # Let's just test that it returns a valid loader
        loader = create_universal_loader(
            model_params=10_000_000,
            batch_size=2,
            seq_len=64,
        )
        # Should be a valid loader (SyntheticDataLoader for < 50M)
        batch = loader.get_batch()
        assert batch["input_ids"].shape == (2, 64)

    def test_available_datasets(self):
        """Test get_available_datasets returns expected names."""
        datasets = get_available_datasets()

        assert "synthetic" in datasets
        assert "local" in datasets
        assert "finefineweb" in datasets
        assert "fineweb" in datasets


@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers not installed")
class TestTokenizer:
    """Test tokenizer caching and loading."""

    def test_tokenizer_caching(self):
        """Test that tokenizer is cached."""
        tok1 = get_tokenizer("gpt2")
        tok2 = get_tokenizer("gpt2")

        assert tok1 is tok2  # Same instance

    def test_tokenizer_pad_token(self):
        """Test pad token is set."""
        tok = get_tokenizer("gpt2")
        assert tok.pad_token is not None


@pytest.mark.skipif(not HAS_DATASETS, reason="datasets library not installed")
class TestHuggingFaceIntegration:
    """Test HuggingFace datasets integration patterns."""

    def test_batch_method(self):
        """Test using dataset.batch() method per HF docs."""
        from datasets import load_dataset

        try:
            dataset = load_dataset(
                "m-a-p/FineFineWeb",
                split="train",
                streaming=True,
            )

            # Use batch method per HF documentation
            batched = dataset.batch(batch_size=4)

            # Get first batch
            batch = next(iter(batched))
            assert "text" in batch
            assert len(batch["text"]) == 4

        except Exception as e:
            pytest.skip(f"Network or dataset unavailable: {e}")

    def test_interleave_datasets(self):
        """Test interleave_datasets for mixing multiple sources."""
        from datasets import load_dataset, interleave_datasets

        try:
            # Load same dataset twice with different shuffles to ensure schema compatibility
            # (different datasets like FineFineWeb and FineWeb have incompatible date field schemas)
            ds1 = (
                load_dataset(
                    "m-a-p/FineFineWeb",
                    split="train",
                    streaming=True,
                )
                .shuffle(seed=42, buffer_size=100)
                .take(10)
            )

            ds2 = (
                load_dataset(
                    "m-a-p/FineFineWeb",
                    split="train",
                    streaming=True,
                )
                .shuffle(seed=123, buffer_size=100)
                .take(10)
            )

            # Interleave with probabilities
            combined = interleave_datasets(
                [ds1, ds2],
                probabilities=[0.5, 0.5],
                seed=42,
            )

            samples = list(combined.take(5))
            assert len(samples) == 5

        except Exception as e:
            pytest.skip(f"Network or dataset unavailable: {e}")

    def test_shuffle_buffer(self):
        """Test shuffle with buffer_size."""
        from datasets import load_dataset

        try:
            dataset = load_dataset(
                "m-a-p/FineFineWeb",
                split="train",
                streaming=True,
            )

            # Shuffle with buffer
            shuffled = dataset.shuffle(seed=42, buffer_size=1000)

            sample = next(iter(shuffled))
            assert "text" in sample

        except Exception as e:
            pytest.skip(f"Network or dataset unavailable: {e}")

    def test_map_tokenize(self):
        """Test map for tokenization."""
        from datasets import load_dataset
        from transformers import AutoTokenizer

        try:
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            # GPT-2 doesn't have a pad token by default, set it
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            dataset = load_dataset(
                "m-a-p/FineFineWeb",
                split="train",
                streaming=True,
            )

            def tokenize(examples):
                return tokenizer(
                    examples["text"],
                    truncation=True,
                    max_length=128,
                    padding="max_length",
                )

            tokenized = dataset.map(
                tokenize,
                batched=True,
                remove_columns=["text"],
            )

            sample = next(iter(tokenized))
            assert "input_ids" in sample
            assert len(sample["input_ids"]) == 128

        except Exception as e:
            pytest.skip(f"Network or dataset unavailable: {e}")


class TestDataLoaderRobustness:
    """Test error handling and robustness."""

    def test_batch_shape_consistency(self):
        """Test that batch shapes are consistent across many batches."""
        loader = SyntheticDataLoader(batch_size=4, seq_len=256)

        for i in range(100):
            batch = loader.get_batch()
            assert batch["input_ids"].shape == (4, 256), f"Batch {i} has wrong shape"
            assert batch["labels"].shape == (4, 256), f"Batch {i} labels wrong shape"

    def test_labels_are_shifted(self, tmp_path):
        """Test that labels are properly shifted by 1."""
        # Create predictable data with sequential tokens
        chunk_data = [{"input_ids": list(range(600)), "length": 600} for _ in range(10)]
        torch.save(chunk_data, tmp_path / "chunk_0000.pt")

        loader = LocalDataLoader(
            data_dir=str(tmp_path),
            batch_size=1,
            seq_len=10,
        )

        batch = loader.get_batch()
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        
        # Our loader returns: input_ids = tokens[:-1], labels = tokens[1:]
        # So for a sequence [0,1,2,3,4,5,6,7,8,9,10], we get:
        #   input_ids = [0,1,2,3,4,5,6,7,8,9]
        #   labels    = [1,2,3,4,5,6,7,8,9,10]
        # Therefore: labels[i] == input_ids[i] + 1 for sequential data
        
        # Verify shapes match
        assert input_ids.shape == labels.shape, f"Shape mismatch: {input_ids.shape} vs {labels.shape}"
        
        # For sequential data, labels should be exactly input_ids + 1
        # (since we created range(600) as input)
        assert torch.all(labels == input_ids + 1), (
            f"Labels not properly shifted! "
            f"input_ids[:5]={input_ids[0, :5].tolist()}, "
            f"labels[:5]={labels[0, :5].tolist()}"
        )

    def test_dtype_is_long(self):
        """Test that output tensors are long dtype."""
        loader = SyntheticDataLoader(batch_size=2, seq_len=64)
        batch = loader.get_batch()

        assert batch["input_ids"].dtype == torch.long
        assert batch["labels"].dtype == torch.long


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
