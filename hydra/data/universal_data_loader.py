"""
Universal Data Loader v2 for DMTA Training.

Refactored to leverage HuggingFace's native PyTorch integration for:
- Multi-worker parallel data loading (4-8x faster)
- Background prefetching (no GPU stalls)
- Batched tokenization in workers
- Native PyTorch DataLoader with pin_memory
- Checkpointing/resume support
- Distributed training ready
- HF Transfer for faster downloads (requires: pip install hf-transfer)

Environment Variables:
    HF_HUB_ENABLE_HF_TRANSFER: Set to "1" to enable faster HuggingFace downloads.
        This module enables it by default if not already set. To disable,
        set HF_HUB_ENABLE_HF_TRANSFER="0" before importing this module.

Usage:
    # Simple - auto-select based on model size
    loader = create_universal_loader(model_params=60_000_000, batch_size=16, seq_len=512)

    # Explicit dataset with num_workers
    loader = create_universal_loader(
        dataset="finefineweb",
        batch_size=16,
        seq_len=512,
        num_workers=4  # Parallel loading!
    )

    # Training loop (same API as before)
    for batch in loader:
        input_ids = batch["input_ids"]  # [B, seq_len] - already torch tensors
        labels = batch["labels"]        # [B, seq_len]
"""

import os

# Enable HF Transfer for faster downloads (5-10x speedup)
# Requires: pip install hf-transfer
# Only set if not already configured by user (respects existing env)
if "HF_HUB_ENABLE_HF_TRANSFER" not in os.environ:
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import torch
import threading
import logging
import time
import glob
import random
from pathlib import Path
from typing import Dict, Any, Optional, List
from collections import deque
import getpass

logger = logging.getLogger("dmta.universal_data")

try:
    from datasets import (
        load_dataset,
        interleave_datasets,
        IterableDataset as HFIterableDataset,
    )

    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    load_dataset = None
    interleave_datasets = None
    HFIterableDataset = None

try:
    from transformers import AutoTokenizer

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    AutoTokenizer = None

try:
    from huggingface_hub import snapshot_download

    HAS_HUGGINGFACE_HUB = True
except ImportError:
    HAS_HUGGINGFACE_HUB = False
    snapshot_download = None

try:
    import pyarrow
    import pyarrow.dataset

    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False
    pyarrow = None


# ============================================
# TOKENIZER CACHE (prevent memory leaks)
# ============================================
_TOKENIZER_CACHE: Dict[str, Any] = {}


def get_tokenizer(name: str = "gpt2"):
    """Get or create a cached tokenizer instance."""
    if name in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[name]

    try:
        # use_fast=True for Rust-based tokenizer (3-10x faster)
        tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        _TOKENIZER_CACHE[name] = tokenizer
        return tokenizer
    except Exception as e:
        logger.error(f"Failed to load tokenizer {name}: {e}")
        return None


# ============================================
# DATASET CONFIGURATIONS
# ============================================

# FineFineWeb domains for local download (actual folder names from HF)
FINEFINEWEB_DOMAINS = [
    # High-volume general domains
    "news", "economics", "entertainment", "sports", "politics", "finance",
    "health", "hobby", "travel", "food", "fashion", "beauty", "pet",
    # Knowledge/Academic domains
    "law", "history", "geography", "literature", "philosophy", "psychology",
    "sociology", "journalism_and_media_communication",
    # Science domains
    "biology", "physics", "chemistry", "mathematics", "astronomy",
    "environmental_science", "atmospheric_science", "ocean_science",
    "materials_science", "statistics", "systems_science",
    # Technology/Engineering domains
    "computer_science_and_technology", "electronic_science", "mechanical_engineering",
    "civil_engineering", "automotive", "aerospace", "transportation_engineering",
    "communication_engineering", "optical_engineering", "instrument_science",
    # Creative/Cultural domains
    "game", "movie", "music_and_dance", "drama_and_film", "artistic", "painting",
    "photo", "design", "landscape_architecture", "urban_planning",
    # Medical/Health domains
    "medical", "agronomy", "nuclear_science",
    # Other specialized domains
    "celebrity", "topicality", "relationship", "library",
    "public_administration", "mining_engineering", "hydraulic_engineering",
    "petroleum_and_natural_gas_engineering", "textile_science", "weapons_science",
    "christianity", "gamble",
]

# Default cache directory for FineFineWeb local data
# Configurable via HYDRA_CACHE_DIR environment variable
# NOTE: Default points to user's existing cache on NVMe
def _autodetect_data_path(rel_path: str) -> Optional[str]:
    """Best-effort discovery for local dataset paths.

    Supports nested paths like "LLM/training_pleias_synth/processed".
    This is only used when explicit env vars are not set.
    """

    rel_path = str(rel_path).lstrip("/")
    candidates: list[str] = []

    data_root = os.environ.get("HYDRA_DATA_ROOT")
    if data_root:
        candidates.append(os.path.join(data_root, rel_path))

    # Common fast disks / stable mount points.
    candidates.extend(
        [
            os.path.join("/mnt/hydra_data", rel_path),
            os.path.join("/mnt/data", rel_path),
            os.path.join("/mnt/nvme0", rel_path),
            os.path.join("/mnt/nvme1", rel_path),
            os.path.join("/mnt", rel_path),
            os.path.join("/data", rel_path),
            os.path.join("/datasets", rel_path),
        ]
    )

    # Mounted volumes under common roots (e.g. /mnt/NewVolume/<rel_path>).
    candidates.extend(glob.glob(f"/mnt/*/{rel_path}"))
    candidates.extend(glob.glob(f"/data/*/{rel_path}"))
    candidates.extend(glob.glob(f"/datasets/*/{rel_path}"))

    user = os.environ.get("USER")
    if not user:
        try:
            user = getpass.getuser()
        except Exception:
            user = None

    if user:
        # Common user-space locations.
        candidates.extend(
            [
                os.path.join("/home", user, rel_path),
                os.path.join("/home", user, "data", rel_path),
                os.path.join("/home", user, "datasets", rel_path),
            ]
        )

        # GUI automount paths (one or two levels deep).
        candidates.extend(glob.glob(f"/media/{user}/*/{rel_path}"))
        candidates.extend(glob.glob(f"/media/{user}/*/*/{rel_path}"))
        candidates.extend(glob.glob(f"/run/media/{user}/*/{rel_path}"))
        candidates.extend(glob.glob(f"/run/media/{user}/*/*/{rel_path}"))

    # Return first existing directory.
    for p in candidates:
        if os.path.isdir(p):
            return p
    return None


FINEFINEWEB_CACHE_DIR = (
    os.environ.get("HYDRA_CACHE_DIR")
    or _autodetect_data_path("hf_finefineweb")
    or _autodetect_data_path("LLM/hf_finefineweb")
    or "/mnt/hydra_data/hf_finefineweb"
)

# Optional: local Nemotron instruction-following chat dumps (JSONL) and converted .pt shards.
# These are used only if you run the converter script and point the configs at the output dir.
def _dir_has_pt_shards(dir_path: str) -> bool:
    return bool(glob.glob(os.path.join(dir_path, "chunk_*.pt"))) or bool(
        glob.glob(os.path.join(dir_path, "*.pt"))
    )


def _autodetect_data_dir(dir_name: str) -> Optional[str]:
    """Best-effort discovery for local dataset roots.

    Tries common mount points so training doesn't depend on GUI automount paths.
    This is only used when explicit env vars are not set.
    """

    # Back-compat helper for callers that pass a single folder name.
    return _autodetect_data_path(dir_name)


# Optional: local Nemotron instruction-following chat dumps (JSONL) and converted .pt shards.
# These are used only if you run the converter script and point the configs at the output dir.
NEMOTRON_JSONL_DIR = os.environ.get(
    "HYDRA_NEMOTRON_JSONL_DIR"
)
if not NEMOTRON_JSONL_DIR:
    NEMOTRON_JSONL_DIR = (
        _autodetect_data_path("nvidia_nemotron_instruction_following")
        or _autodetect_data_path("LLM/nvidia_nemotron_instruction_following")
        or "/mnt/hydra_data/nvidia_nemotron_instruction_following"
    )
NEMOTRON_PT_DIR = (
    os.environ.get("HYDRA_NEMOTRON_PT_DIR")
    or _autodetect_data_dir("hydra_nemotron_pt")
    or "/mnt/hydra_data/hydra_nemotron_pt"
)


def _nemotron_path(dataset: str, seq_len: int) -> str:
    """Resolve Nemotron converted dataset directory.

    Preferred standardized layout:
        <NEMOTRON_PT_DIR>/<seq_len>/<dataset>

    Backward-compatible legacy layout:
        <NEMOTRON_PT_DIR>/<dataset>_<seq_len>
    """

    seq_dir = os.path.join(NEMOTRON_PT_DIR, str(int(seq_len)), dataset)
    legacy_dir = os.path.join(NEMOTRON_PT_DIR, f"{dataset}_{int(seq_len)}")

    # Prefer directories that actually contain shards.
    if os.path.isdir(seq_dir) and _dir_has_pt_shards(seq_dir):
        return seq_dir
    if os.path.isdir(legacy_dir) and _dir_has_pt_shards(legacy_dir):
        return legacy_dir

    # Fall back to standardized layout if it exists at all.
    if os.path.isdir(seq_dir):
        return seq_dir
    return legacy_dir

# Local small instruction/chat datasets converted to `.pt` shards.
SMALL_CHAT_PT_DIR = (
    os.environ.get("HYDRA_SMALL_CHAT_PT_DIR")
    or _autodetect_data_dir("hydra_small_chat_pt")
    or "/mnt/hydra_data/hydra_small_chat_pt"
)


def _small_chat_path(dataset: str, seq_len: int) -> str:
    """Resolve small_chat converted dataset directory.

    Preferred standardized layout:
        <SMALL_CHAT_PT_DIR>/<seq_len>/<dataset>

    Backward-compatible legacy layout:
        <SMALL_CHAT_PT_DIR>/<dataset>_<seq_len>
    """

    seq_dir = os.path.join(SMALL_CHAT_PT_DIR, str(int(seq_len)), dataset)
    legacy_dir = os.path.join(SMALL_CHAT_PT_DIR, f"{dataset}_{int(seq_len)}")

    # Prefer directories that actually contain shards.
    if os.path.isdir(seq_dir) and _dir_has_pt_shards(seq_dir):
        return seq_dir
    if os.path.isdir(legacy_dir) and _dir_has_pt_shards(legacy_dir):
        return legacy_dir

    # Fall back to standardized layout if it exists at all.
    if os.path.isdir(seq_dir):
        return seq_dir
    return legacy_dir


def download_finefineweb_subset(
    cache_dir: str = FINEFINEWEB_CACHE_DIR,
    domains: Optional[List[str]] = None,
    max_files_per_domain: int = 10,
) -> str:
    """
    Download FineFineWeb subset to local disk for fast streaming.
    
    This avoids 502 errors from HuggingFace API by downloading files
    directly without tree enumeration.
    
    Args:
        cache_dir: Local directory to cache the dataset
        domains: List of domains to download (default: all 38 domains)
        max_files_per_domain: Max files per domain (e.g., 10 = first 10 files)
    
    Returns:
        Path to local directory containing downloaded files
    """
    from huggingface_hub import hf_hub_download
    
    if domains is None:
        domains = FINEFINEWEB_DOMAINS
    
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading FineFineWeb subset to {cache_dir}")
    logger.info(f"Domains: {len(domains)}, max files per domain: {max_files_per_domain}")
    
    # Download each file directly (no tree enumeration)
    downloaded = 0
    failed = 0
    for domain in domains:
        domain_dir = cache_path / domain
        domain_dir.mkdir(exist_ok=True)
        
        for i in range(max_files_per_domain):
            # Files use 6-digit numbering: domain_000000.jsonl
            filename = f"{domain}/{domain}_{i:06d}.jsonl"
            local_path = cache_path / filename
            
            if local_path.exists():
                downloaded += 1
                continue
            
            try:
                hf_hub_download(
                    repo_id="m-a-p/FineFineWeb",
                    filename=filename,
                    repo_type="dataset",
                    local_dir=cache_dir,
                )
                downloaded += 1
                if downloaded % 50 == 0:
                    logger.info(f"Downloaded {downloaded} files...")
            except Exception:
                # Some domains may not have all file indices
                failed += 1
                if failed <= 5:
                    logger.debug(f"File not found (expected for some domains): {filename}")
    
    logger.info(f"Downloaded {downloaded} files ({failed} not found) to {cache_dir}")
    return str(cache_path)



def load_finefineweb_local(
    cache_dir: str = FINEFINEWEB_CACHE_DIR,
    domains: Optional[List[str]] = None,
    streaming: bool = True,
    auto_download: bool = True,
    max_files_per_domain: int = 10,
):
    """
    Load FineFineWeb from local cache with optional auto-download.
    
    This is the preferred method for reliable, fast data loading.
    First call downloads the subset, subsequent calls stream from disk.
    
    Args:
        cache_dir: Local cache directory
        domains: List of domains to load
        streaming: If True, stream from disk (memory efficient)
        auto_download: If True and cache missing, download first
        max_files_per_domain: Max files per domain for download
    
    Returns:
        HuggingFace dataset (streaming or in-memory)
    """
    if domains is None:
        domains = FINEFINEWEB_DOMAINS
    
    # Check if cache exists
    cache_path = Path(cache_dir)
    if not cache_path.exists() or not any(cache_path.iterdir()):
        if auto_download:
            logger.info("Cache not found, downloading FineFineWeb subset...")
            download_finefineweb_subset(cache_dir, domains, max_files_per_domain)
        else:
            raise FileNotFoundError(
                f"Cache not found at {cache_dir}. "
                "If you have a local FineFineWeb cache, set HYDRA_CACHE_DIR to its path "
                "(or set HYDRA_DATA_ROOT and place 'hf_finefineweb/' under it). "
                "If this is an external/NVMe drive, ensure it is mounted. "
                "To download a subset instead, set auto_download=True."
            )
    
    # Collect all JSONL files from cached domains
    files = []
    for d in domains:
        domain_files = glob.glob(os.path.join(cache_dir, d, f"{d}_*.jsonl"))
        files.extend(sorted(domain_files))
    
    if not files:
        raise FileNotFoundError(f"No JSONL files found in {cache_dir}")
    
    logger.info(f"Loading {len(files)} files from local cache")
    
    ds = load_dataset(
        "json",
        data_files=files,
        split="train",
        streaming=streaming,
    )
    
    return ds


def load_finefineweb_hybrid(
    cache_dir: str = FINEFINEWEB_CACHE_DIR,
    domains: Optional[List[str]] = None,
    local_weight: float = 0.7,
    streaming: bool = True,
):
    """
    Hybrid loader: interleave local cache + remote HuggingFace streaming.
    
    Start immediately with local files (fast, reliable), while also 
    streaming from HuggingFace to get more domain diversity.
    
    Args:
        cache_dir: Local cache directory
        domains: List of domains to load locally
        local_weight: Probability of sampling from local vs remote (0.7 = 70% local)
        streaming: Must be True for this mode
    
    Returns:
        Interleaved HuggingFace streaming dataset
    """
    from datasets import interleave_datasets
    
    if domains is None:
        domains = FINEFINEWEB_DOMAINS
    
    datasets_to_interleave = []
    weights = []
    
    # 1. Load local files (fast, reliable)
    cache_path = Path(cache_dir)
    local_files = []
    if cache_path.exists():
        for d in domains:
            domain_files = glob.glob(os.path.join(cache_dir, d, f"{d}_*.jsonl"))
            local_files.extend(sorted(domain_files))
    
    if local_files:
        logger.info(f"Hybrid: loading {len(local_files)} local files (weight={local_weight})")
        local_ds = load_dataset(
            "json",
            data_files=local_files,
            split="train",
            streaming=True,
        )
        datasets_to_interleave.append(local_ds)
        weights.append(local_weight)
    
    # 2. Stream from HuggingFace (more diversity, may have occasional errors)
    # Use FineWeb-Edu (smaller, better API support) instead of FineFineWeb
    logger.info(f"Hybrid: connecting to remote FineWeb-Edu (weight={1-local_weight})")
    remote_ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",  # 10B token sample, manageable size
        split="train",
        streaming=True,
    )
    datasets_to_interleave.append(remote_ds)
    weights.append(1 - local_weight)
    logger.info("Remote FineWeb-Edu connected successfully")
    
    if len(datasets_to_interleave) == 1:
        return datasets_to_interleave[0]
    
    # Interleave with weighted sampling
    return interleave_datasets(
        datasets_to_interleave,
        probabilities=weights,
        stopping_strategy="all_exhausted",
    )


DATASET_CONFIGS = {
    # FineFineWeb - hybrid mode: bootstrap with local, then stream from web
    "finefineweb": {
        "path": "m-a-p/FineFineWeb",
        "name": None,
        "text_column": "text",
        "format": "jsonl",
        "use_hybrid": True,
        "local_weight": 0.1,  # 10% local (bootstrap), 90% remote (main source)
        "cache_dir": FINEFINEWEB_CACHE_DIR,
        "domains": FINEFINEWEB_DOMAINS,
        "max_files_per_domain": 10,
        "description": "Hybrid: 10% local bootstrap + 90% remote streaming",
    },
    # Local-only mode (no remote streaming, 100% reliable)
    "finefineweb-local": {
        "path": "m-a-p/FineFineWeb",
        "name": None,
        "text_column": "text",
        "format": "jsonl",
        "use_local_cache": True,
        "cache_dir": FINEFINEWEB_CACHE_DIR,
        "domains": FINEFINEWEB_DOMAINS,
        "max_files_per_domain": 10,
        "description": "Local cache only (67 domains × 10 files, ~113GB)",
    },
    # Local pre-tokenized datasets
    "pleias_synth": {
        "local": True,
        "path": (
            os.environ.get("HYDRA_PLEIAS_SYNTH_DIR")
            or _autodetect_data_path("training_pleias_synth/processed")
            or _autodetect_data_path("LLM/training_pleias_synth/processed")
            or "/mnt/hydra_data/training_pleias_synth/processed"
        ),
        "description": "PleIAs SYNTH synthetic reasoning (~1.5B tokens, 1756 shards, pre-tokenized)",
    },

    # Small instruction/chat datasets (convert local raw dumps -> .pt via diagnostics script)
    "alpaca_cleaned_512": {
        "local": True,
        "path": _small_chat_path("alpaca_cleaned", 512),
        "description": "Alpaca (cleaned) instruction data, seq=512 pre-tokenized",
    },
    "alpaca_cleaned_1024": {
        "local": True,
        "path": _small_chat_path("alpaca_cleaned", 1024),
        "description": "Alpaca (cleaned) instruction data, seq=1024 pre-tokenized",
    },
    "alpaca_cleaned_2048": {
        "local": True,
        "path": _small_chat_path("alpaca_cleaned", 2048),
        "description": "Alpaca (cleaned) instruction data, seq=2048 pre-tokenized",
    },
    "code_alpaca_512": {
        "local": True,
        "path": _small_chat_path("code_alpaca", 512),
        "description": "CodeAlpaca-20k instruction data, seq=512 pre-tokenized",
    },
    "code_alpaca_1024": {
        "local": True,
        "path": _small_chat_path("code_alpaca", 1024),
        "description": "CodeAlpaca-20k instruction data, seq=1024 pre-tokenized",
    },
    "code_alpaca_2048": {
        "local": True,
        "path": _small_chat_path("code_alpaca", 2048),
        "description": "CodeAlpaca-20k instruction data, seq=2048 pre-tokenized",
    },
    "dolly_512": {
        "local": True,
        "path": _small_chat_path("dolly", 512),
        "description": "Databricks Dolly 15k, seq=512 pre-tokenized",
    },
    "dolly_1024": {
        "local": True,
        "path": _small_chat_path("dolly", 1024),
        "description": "Databricks Dolly 15k, seq=1024 pre-tokenized",
    },
    "oo_labeled_correct_1024": {
        "local": True,
        "path": _small_chat_path("oo_labeled_correct", 1024),
        "description": "ShareGPT-labeled_correct (GPT-4) single-turn extraction, seq=1024 pre-tokenized",
    },
    "dolly_2048": {
        "local": True,
        "path": _small_chat_path("dolly", 2048),
        "description": "Databricks Dolly 15k, seq=2048 pre-tokenized",
    },

    # Flat per-seq consolidated folders (produced by diagnostics/collect_pt_by_seq.py --flat)
    # These contain chunk_######.pt directly under <SMALL_CHAT_PT_DIR>/<seq_len>/.
    "chat_flat_512": {
        "local": True,
        "path": os.path.join(SMALL_CHAT_PT_DIR, "512"),
        "description": "Flat merged chat shards at seq=512 (all sources consolidated)",
    },
    "chat_flat_1024": {
        "local": True,
        "path": os.path.join(SMALL_CHAT_PT_DIR, "1024"),
        "description": "Flat merged chat shards at seq=1024 (all sources consolidated)",
    },
    "chat_flat_2048": {
        "local": True,
        "path": os.path.join(SMALL_CHAT_PT_DIR, "2048"),
        "description": "Flat merged chat shards at seq=2048 (all sources consolidated)",
    },
    "chat_flat_4096": {
        "local": True,
        "path": os.path.join(SMALL_CHAT_PT_DIR, "4096"),
        "description": "Flat merged chat shards at seq=4096 (all sources consolidated)",
    },
    "chat_flat_8192": {
        "local": True,
        "path": os.path.join(SMALL_CHAT_PT_DIR, "8192"),
        "description": "Flat merged chat shards at seq=8192 (all sources consolidated)",
    },

    # Nemotron instruction-following chat (convert JSONL -> .pt via diagnostics script)
    "nemotron_chat_if_512": {
        "local": True,
        "path": _nemotron_path("chat_if", 512),
        "description": "Nemotron instruction-following chat (seq=512) pre-tokenized",
    },
    "nemotron_chat_if_1024": {
        "local": True,
        "path": _nemotron_path("chat_if", 1024),
        "description": "Nemotron instruction-following chat (seq=1024) pre-tokenized",
    },
    "nemotron_chat_if_2048": {
        "local": True,
        "path": _nemotron_path("chat_if", 2048),
        "description": "Nemotron instruction-following chat (seq=2048) pre-tokenized",
    },
    "nemotron_structured_outputs_512": {
        "local": True,
        "path": _nemotron_path("structured_outputs", 512),
        "description": "Nemotron structured-outputs chat (seq=512) pre-tokenized",
    },
    "nemotron_structured_outputs_1024": {
        "local": True,
        "path": _nemotron_path("structured_outputs", 1024),
        "description": "Nemotron structured-outputs chat (seq=1024) pre-tokenized",
    },
    "nemotron_structured_outputs_2048": {
        "local": True,
        "path": _nemotron_path("structured_outputs", 2048),
        "description": "Nemotron structured-outputs chat (seq=2048) pre-tokenized",
    },

    # Nemotron instruction-following chat (single-turn extraction: last user->assistant)
    "nemotron_chat_if_last_pair_512": {
        "local": True,
        "path": _nemotron_path("chat_if_last_pair", 512),
        "description": "Nemotron chat_if last_pair extraction (seq=512) pre-tokenized",
    },
    "nemotron_chat_if_last_pair_1024": {
        "local": True,
        "path": _nemotron_path("chat_if_last_pair", 1024),
        "description": "Nemotron chat_if last_pair extraction (seq=1024) pre-tokenized",
    },
    "nemotron_chat_if_last_pair_2048": {
        "local": True,
        "path": _nemotron_path("chat_if_last_pair", 2048),
        "description": "Nemotron chat_if last_pair extraction (seq=2048) pre-tokenized",
    },
    "nemotron_chat_if_last_pair_4096": {
        "local": True,
        "path": _nemotron_path("chat_if_last_pair", 4096),
        "description": "Nemotron chat_if last_pair extraction (seq=4096) pre-tokenized",
    },
    "nemotron_chat_if_last_pair_8192": {
        "local": True,
        "path": _nemotron_path("chat_if_last_pair", 8192),
        "description": "Nemotron chat_if last_pair extraction (seq=8192) pre-tokenized",
    },
    "fineweb": {
        "path": "HuggingFaceFW/fineweb",
        "name": "sample-10BT",
        "text_column": "text",
        "format": "parquet",
        "description": "Large-scale curated web text (~10B tokens)",
    },
    "fineweb_edu": {
        "path": "HuggingFaceFW/fineweb-edu",
        "name": "sample-10BT",
        "text_column": "text",
        "format": "parquet",
        "description": "Educational web content (~10B tokens)",
    },
    # Small datasets for testing
    "wikitext2": {
        "path": "wikitext",
        "name": "wikitext-2-raw-v1",
        "text_column": "text",
        "description": "Small Wikipedia text (~2M tokens)",
    },
    "tinystories": {
        "path": "roneneldan/TinyStories",
        "name": None,
        "text_column": "text",
        "description": "Synthetic simple stories for small models",
    },
    # Specialized
    "math": {
        "path": "TIGER-Lab/MathInstruct",
        "name": None,
        "text_column": None,
        "formatter": "math",
        "description": "Math instruction tuning data",
    },
    "code": {
        "path": "m-a-p/CodeFeedback-Filtered-Instruction",
        "name": None,
        "text_column": None,
        "formatter": "code",
        "description": "Code instruction tuning data",
    },
    "chat": {
        "path": "HuggingFaceH4/ultrachat_200k",
        "name": None,
        "split": "train_sft",
        "text_column": None,
        "formatter": "chat",
        "description": "Chat/QA instruction data",
    },

    # ============================================
    # REASONING / CoT DATASETS
    # ============================================
    # These provide diverse math reasoning to avoid memorization on MathInstruct
    "open_math_instruct": {
        "path": "nvidia/OpenMathInstruct-2",
        "name": None,
        "split": "train_1M",  # Use 1M subset to avoid overwhelming the mix
        "text_column": None,
        "formatter": "open_math_instruct",
        "description": "NVIDIA OpenMathInstruct-2: diverse math problems with CoT solutions (1M subset)",
    },
    "open_thoughts": {
        "path": "open-thoughts/OpenThoughts-114k",
        "name": None,
        "text_column": None,
        "formatter": "open_thoughts",
        "description": "OpenThoughts-114k: math/code/science reasoning traces from DeepSeek-R1",
    },
    "bespoke_stratos": {
        "path": "bespokelabs/Bespoke-Stratos-17k",
        "name": None,
        "text_column": None,
        "formatter": "bespoke_stratos",
        "description": "Bespoke-Stratos-17k: high-quality reasoning traces (math/code/science)",
    },

    # Sequence-aware small chat interface:
    # - At 512/1024: only small local instruction datasets (no long multi-turn garbage)
    # - At 2048+: add UltraChat (works better at long context)
    "small_chat_seqaware": {
        "mixed_by_seq": {
            "512": {
                "sources": [
                    {"name": "alpaca_cleaned_512", "weight": 0.30},
                    {"name": "dolly_512", "weight": 0.20},
                    {"name": "code_alpaca_512", "weight": 0.20},
                    {"name": "nemotron_chat_if_last_pair_512", "weight": 0.30},
                ]
            },
            "1024": {
                "sources": [
                    {"name": "alpaca_cleaned_1024", "weight": 0.20},
                    {"name": "dolly_1024", "weight": 0.15},
                    {"name": "code_alpaca_1024", "weight": 0.15},
                    {"name": "oo_labeled_correct_1024", "weight": 0.25},
                    {"name": "nemotron_chat_if_last_pair_1024", "weight": 0.25},
                ]
            },
            "2048": {
                "sources": [
                    {"name": "chat", "weight": 0.50},
                    {"name": "alpaca_cleaned_2048", "weight": 0.15},
                    {"name": "dolly_2048", "weight": 0.10},
                    {"name": "code_alpaca_2048", "weight": 0.10},
                    {"name": "nemotron_chat_if_last_pair_2048", "weight": 0.15},
                ]
            },
        },
        "description": "Seq-aware mix: Alpaca/Dolly/CodeAlpaca at 512/1024; adds UltraChat at 2048+",
    },

    # Seq-aware preset that uses the flat consolidated folders directly.
    # Use this if you want the *merged* per-seq directories to be the training source of truth.
    "small_chat_seqaware_flat": {
        "mixed_by_seq": {
            "512": {"sources": [{"name": "chat_flat_512", "weight": 1.0}]},
            "1024": {"sources": [{"name": "chat_flat_1024", "weight": 1.0}]},
            "2048": {"sources": [{"name": "chat_flat_2048", "weight": 1.0}]},
            "4096": {"sources": [{"name": "chat_flat_4096", "weight": 1.0}]},
            "8192": {"sources": [{"name": "chat_flat_8192", "weight": 1.0}]},
        },
        "description": "Seq-aware flat: uses consolidated per-seq folders under HYDRA_SMALL_CHAT_PT_DIR",
    },
    "production_672m_creative": {
        "mixed": True,
        "sources": [
            # 1. The World Knowledge (55%):
            # Still the backbone. You need this for facts and vocabulary.
            {"name": "finefineweb-sequential", "weight": 0.55},

            # 2. The Logic (30%): 
            # Crucial. Without this, the stories/text will wander aimlessly.
            # Pleias provides the "plot logic" that TinyStories lacks.
            {"name": "pleias_synth", "weight": 0.30},

            # 3. The Grammar Glue (10%): TinyStories
            # Replaces some chat data. Teaches the model to maintain 
            # a narrative thread without the "As an AI..." bloat.
            {"name": "tinystories", "weight": 0.10},

            # 4. The Minimal Interface (5%): Chat
            # The bare minimum. Just enough so it recognizes "<|user|>" 
            # and doesn't crash when you ask it a question.
            {"name": "small_chat_seqaware_flat", "weight": 0.05}, 
        ],
        "description": "672M Creative: 55% Web + 30% Reasoning + 10% Stories + 5% Chat",
    },
    # Pretraining: seq-aware 60/30/10 mix.
    # - 60% FineFineWeb (streaming; uses local cache/transition via finefineweb-sequential)
    # - 30% local small chat (your consolidated flat per-seq folders)
    # - 10% local Pleias (pre-tokenized)
    "pretrain_ffw60_chat30_pleias10_seqaware": {
        "mixed_by_seq": {
            "512": {
                "sources": [
                    {"name": "finefineweb-sequential", "weight": 0.60},
                    {"name": "chat_flat_512", "weight": 0.30},
                    {"name": "pleias_synth", "weight": 0.10},
                ]
            },
            "1024": {
                "sources": [
                    {"name": "finefineweb-sequential", "weight": 0.60},
                    {"name": "chat_flat_1024", "weight": 0.30},
                    {"name": "pleias_synth", "weight": 0.10},
                ]
            },
            "2048": {
                "sources": [
                    {"name": "finefineweb-sequential", "weight": 0.60},
                    {"name": "chat_flat_2048", "weight": 0.30},
                    {"name": "pleias_synth", "weight": 0.10},
                ]
            },
        },
        "description": "Seq-aware pretrain mix: 60% FineFineWeb + 30% local merged small-chat + 10% Pleias",
    },
    # Phase 2: SFT mix - instruction tuning for chat capability
    "sft_chat": {
        "mixed": True,
        "sources": [
            {"name": "chat", "weight": 0.70},             # UltraChat conversations
            {"name": "math", "weight": 0.20},             # Math reasoning
            {"name": "code", "weight": 0.10},             # Code generation
        ],
        "description": "SFT mix for chat: 70% conversation + 20% math + 10% code",
    },
    # ============================================
    # DEFAULT PRETRAINING MIX (Recommended)
    # ============================================
    # Web-heavy with small amounts of structured data for diversity
    # - 69% FineFineWeb: Main knowledge source
    # - 11% TinyStories: Narrative coherence, story structure
    # - 5% Math: Mathematical reasoning
    # - 5% Code: Programming patterns
    # - 4% Pleias: Synthetic reasoning, cleaner examples
    # - 3% Chat (local): Basic conversational patterns
    # - 1% UltraChat: Instruction-following
    # - 2% WikiText2: Narrative long-form stability
    "pretrain_default": {
        "mixed": True,
        "sources": [
            {"name": "finefineweb-local", "weight": 0.69},
            {"name": "tinystories", "weight": 0.11},
            {"name": "math", "weight": 0.05},
            {"name": "code", "weight": 0.05},
            {"name": "pleias_synth", "weight": 0.04},
            # Use the per-dataset seq-aware mix rather than consolidated flat shards.
            # This avoids dependency on a separately maintained <SMALL_CHAT_PT_DIR>/<seq_len>/chunk_*.pt folder.
            {"name": "small_chat_seqaware", "weight": 0.03},
            {"name": "chat", "weight": 0.01},
            {"name": "wikitext2", "weight": 0.02},
        ],
        "description": "Default pretrain: 69% FineFineWeb + 11% TinyStories + 4% math + 6% code + 4% Pleias + 3% chat + 1% UltraChat + 2% WikiText2",
    },

    # ============================================
    # CONSERVATIVE REASONING BOOST (pretrain_reasoning_lite)
    # ============================================
    # Designed to wean off MathInstruct (model memorizing) by diversifying reasoning sources.
    # Middle ground between pretrain_default (69% web, 10% math+code) and aggressive mixes.
    # - 65% Web: Slightly reduced to make room for reasoning
    # - 14% Math/Code: 7% each (40% boost over default, but not doubled)
    # - Diversified math: splits between MathInstruct + OpenMathInstruct to reduce memorization
    # - Small CoT traces: 2% open_thoughts for reasoning patterns without shocking router
    "pretrain_reasoning_lite": {
        "mixed": True,
        "sources": [
            # Base Knowledge (reduced from 69% to 65%)
            {"name": "finefineweb-local", "weight": 0.65},

            # Diversified Math (total 7%): split to reduce memorization on single source
            {"name": "math", "weight": 0.02},              # Reduced from 5%
            {"name": "open_math_instruct", "weight": 0.05}, # New: diverse math problems

            # Code (boosted from 5% to 7%)
            {"name": "code", "weight": 0.07},

            # Narrative coherence (reduced from 11% to 6%)
            {"name": "tinystories", "weight": 0.06},

            # Synthetic reasoning (kept at 4%)
            {"name": "pleias_synth", "weight": 0.04},

            # Small CoT traces (new, conservative 4%)
            {"name": "open_thoughts", "weight": 0.04},

            # Chat/Instruction (kept similar)
            {"name": "small_chat_seqaware", "weight": 0.03},
            {"name": "chat", "weight": 0.01},

            # Long-form stability
            {"name": "wikitext2", "weight": 0.03},
        ],
        "description": "Conservative reasoning boost: 65% Web + 7% diversified math + 7% code + 4% CoT + 6% TinyStories + 4% Pleias + 4% chat + 3% WikiText2",
    },

    # ============================================
    # AGENTIC REASONING MIX (pretrain_agentic)
    # ============================================
    # Designed for models that will have tool-use (web search, etc.)
    # Goals:
    # - Confident enough to chat naturally
    # - Calibrated uncertainty (knows when to say "let me look that up")
    # - Strong reasoning (step-by-step thinking reveals uncertainty)
    # - Less "confidently wrong" web patterns
    #
    # Key insight: CoT traces teach "let me think..." patterns which naturally
    # express uncertainty. Raw web text teaches confident assertions.
    #
    # Mix rationale:
    # - 45% Web: Reduced from 65% - still need world knowledge, but less confident-wrong patterns
    # - 20% Reasoning/CoT: Heavy reasoning teaches when to think vs. assert
    # - 12% Code: Formal reasoning, verifiable correctness teaches calibration
    # - 10% Chat: Conversational patterns, instruction following
    # - 8% Math: Structured problem solving
    # - 5% Narrative: Coherent output, story completion
    "pretrain_agentic": {
        "mixed": True,
        "sources": [
            # World Knowledge (reduced - less confidently-wrong patterns)
            {"name": "finefineweb-local", "weight": 0.45},

            # Heavy Reasoning/CoT (teaches "let me think" vs instant answers)
            {"name": "open_thoughts", "weight": 0.10},      # DeepSeek-R1 reasoning traces
            {"name": "bespoke_stratos", "weight": 0.05},    # High-quality reasoning
            {"name": "pleias_synth", "weight": 0.05},       # Synthetic reasoning

            # Code (formal reasoning, teaches precision)
            {"name": "code", "weight": 0.12},

            # Chat/Instruction (conversational ability)
            {"name": "small_chat_seqaware", "weight": 0.06},
            {"name": "chat", "weight": 0.04},               # UltraChat

            # Math (structured problem-solving)
            {"name": "open_math_instruct", "weight": 0.05},
            {"name": "math", "weight": 0.03},

            # Narrative coherence (reduced)
            {"name": "tinystories", "weight": 0.03},
            {"name": "wikitext2", "weight": 0.02},
        ],
        "description": "Agentic reasoning: 45% Web + 20% CoT/reasoning + 12% code + 10% chat + 8% math + 5% narrative",
    },

    # ============================================
    # AGENTIC BRIDGE MIX (pretrain_agentic_bridge)
    # ============================================
    # Halfway point between pretrain_reasoning_lite and pretrain_agentic.
    # Use this as a curriculum transition step to avoid distribution shock.
    # Key changes from reasoning_lite:
    # - Web: 65% -> 55% (halfway to 45%)
    # - CoT: 4% -> 12% (halfway to 20%)
    # - Code: 7% -> 10% (halfway to 12%)
    # - Chat: 4% -> 7% (halfway to 10%)
    "pretrain_agentic_bridge": {
        "mixed": True,
        "sources": [
            # Web: 55% (halfway between 65% and 45%)
            {"name": "finefineweb-local", "weight": 0.55},

            # CoT/Reasoning: 12% (halfway between 4% and 20%)
            {"name": "open_thoughts", "weight": 0.07},
            {"name": "pleias_synth", "weight": 0.05},

            # Code: 10% (halfway between 7% and 12%)
            {"name": "code", "weight": 0.10},

            # Chat: 7% (halfway between 4% and 10%)
            {"name": "small_chat_seqaware", "weight": 0.05},
            {"name": "chat", "weight": 0.02},

            # Math: 7% (stable)
            {"name": "open_math_instruct", "weight": 0.05},
            {"name": "math", "weight": 0.02},

            # Narrative: 4%
            {"name": "tinystories", "weight": 0.04},
        ],
        "description": "Bridge mix: 55% Web + 12% CoT + 10% code + 7% chat + 7% math + 4% narrative",
    },

    # ============================================
    # AGENTIC CURRICULUM (pretrain_agentic_curriculum)
    # ============================================
    # Automated 3-phase curriculum that transitions from reasoning_lite to agentic.
    # Uses mix_schedule with curriculum_transition type for smooth weight interpolation.
    #
    # Phase 1 (steps 0 - phase1_end): Start weights (reasoning_lite-like)
    # Phase 2 (phase1_end - phase2_end): Linear interpolation to end weights
    # Phase 3 (phase2_end+): End weights (agentic)
    #
    # Default schedule (override via CLI or resume config):
    # - phase1_end_step: 45000 (consolidate on current distribution)
    # - phase2_end_step: 115000 (70K step transition window)
    #
    # Note: Steps are RELATIVE to when you start using this dataset.
    # If resuming at step 235K, set phase1_end_step=280000, phase2_end_step=350000.
    "pretrain_agentic_curriculum": {
        "mixed": True,
        "sources": [
            # All sources needed for both start and end distributions
            # Start weights: slightly boosted reasoning/code/math vs reasoning_lite
            {"name": "finefineweb-local", "weight": 0.62},  # Reduced from 65% (-3%)
            {"name": "open_thoughts", "weight": 0.05},      # Reasoning: +1%
            {"name": "bespoke_stratos", "weight": 0.00},    # 0 at start, ramps up
            {"name": "pleias_synth", "weight": 0.04},
            {"name": "code", "weight": 0.08},               # Code: +1%
            {"name": "small_chat_seqaware", "weight": 0.03},
            {"name": "chat", "weight": 0.01},
            {"name": "open_math_instruct", "weight": 0.06}, # Hard math: +1%
            {"name": "math", "weight": 0.03},               # Math: +1%
            {"name": "tinystories", "weight": 0.05},        # Reduced from 6% (-1%)
            {"name": "wikitext2", "weight": 0.03},
        ],
        "mix_schedule": {
            "type": "curriculum_transition",
            # Phase 1 end: consolidate current distribution
            "phase1_end_step": 280000,
            # Phase 2 end: finish transition to agentic
            "phase2_end_step": 350000,
            # Target weights at end of curriculum (agentic distribution)
            "end_weights": {
                "finefineweb-local": 0.45,
                "open_thoughts": 0.10,
                "bespoke_stratos": 0.05,
                "pleias_synth": 0.05,
                "code": 0.12,
                "small_chat_seqaware": 0.06,
                "chat": 0.03,                   # Reduced from 4% to balance
                "open_math_instruct": 0.06,     # Keep at 6% (model mastered simple math)
                "math": 0.03,
                "tinystories": 0.03,
                "wikitext2": 0.02,
            },
        },
        "description": "Automated curriculum: reasoning_lite -> agentic over configurable step range",
    },

    # ============================================
    # EVAL MIX: Pretrain-like, disjoint sources
    # ============================================
    # Goal: keep eval domain mix closer to `pretrain_default` without reusing
    # training sources (FineFineWeb/TinyStories/Pleias/local small-chat).
    # - 80% FineWeb-Edu: web-like distribution, independent corpus
    # - 12% WikiText2: narrative-ish long-form text (small but stable)
    # - 5% MathInstruct: structured reasoning/instruction
    # - 3% UltraChat: conversational instruction
    "pretrain_default_eval": {
        "mixed": True,
        "sources": [
            {"name": "fineweb_edu", "weight": 0.80},
            {"name": "wikitext2", "weight": 0.12},
            {"name": "open_math_instruct", "weight": 0.05},
            {"name": "chat", "weight": 0.03},
        ],
        "description": "Eval-only mix: 80% FineWeb-Edu + 12% WikiText2 + 5% OpenMathInstruct + 3% UltraChat",
    },

    # Back-compat preset name expected by tests/configs.
    # Constant (batch-level) ratio mix of local FineFineWeb + Pleias + UltraChat.
    "ffw_pleias_ultrachat_const": {
        "mixed": True,
        "sources": [
            {"name": "finefineweb-local", "weight": 0.70},
            {"name": "pleias_synth", "weight": 0.15},
            {"name": "chat", "weight": 0.15},
        ],
        "description": "Constant mix: 70% FineFineWeb-local + 15% Pleias + 15% UltraChat",
    },
    # ============================================
    # PRETRAINING MIX: Local FineFineWeb + UltraChat + TinyStories
    # ============================================
    # For SMALL models (250M-500M): More curated data, less noisy web
    # - Smaller models learn better from structured/clean data
    # - UltraChat teaches conversational patterns
    # - TinyStories provides coherent narrative structure
    # - Web data supplements with knowledge but can be noisy
    "pretrain_mix": {
        "mixed": True,
        "sources": [
            {"name": "finefineweb-local", "weight": 0.60},  # 120GB local web knowledge
            {"name": "pleias_synth", "weight": 0.15},       # 4GB local synthetic reasoning (~1.5B tokens)
            {"name": "tinystories", "weight": 0.15},        # Streamed narrative coherence
            # Prefer the per-dataset seq-aware mix over consolidated flat shards.
            {"name": "small_chat_seqaware", "weight": 0.10},  # local chat/instruction mix
        ],
        "description": "Local-first mix: 60% FineFineWeb-local + 15% Pleias + 15% TinyStories (streamed) + 10% local chat",
    },
    # For MEDIUM+ models (750M+): More web data, they can handle noise
    "pretrain_web": {
        "mixed": True,
        "sources": [
            {"name": "finefineweb-local", "weight": 0.70},  # Heavy web focus
            {"name": "chat", "weight": 0.15},               # Some chat
            {"name": "tinystories", "weight": 0.15},        # Some stories
        ],
        "description": "Medium+ model mix: 70% FineFineWeb + 15% UltraChat + 15% TinyStories",
    },
    # Chat-heavy variant for conversational models
    "pretrain_chat": {
        "mixed": True,
        "sources": [
            {"name": "finefineweb-local", "weight": 0.30},  # Minimal web
            {"name": "chat", "weight": 0.45},               # Heavy chat focus
            {"name": "tinystories", "weight": 0.25},        # Good narrative base
        ],
        "description": "Chat-focused: 30% FineFineWeb + 45% UltraChat + 25% TinyStories",
    },

    # ============================================
    # MOE 4-EXPERT SPECIALIZATION MIX
    # ============================================
    # Balanced mix for training 4-expert MoE with domain specialization.
    # Each source maps to one expert:
    #   math -> Expert 0 (reasoning/logic)
    #   code -> Expert 1 (programming)
    #   chat -> Expert 2 (conversational)
    #   finefineweb-local -> Expert 3 (general knowledge/web)
    # Use with: --moe_domain_expert_map "math:0,code:1,chat:2,finefineweb-local:3"
    "moe_4expert_balanced": {
        "mixed": True,
        "sources": [
            {"name": "math", "weight": 0.25},              # Expert 0: Math/Reasoning
            {"name": "code", "weight": 0.25},              # Expert 1: Code
            {"name": "chat", "weight": 0.25},              # Expert 2: Chat/Creative
            {"name": "finefineweb-local", "weight": 0.25}, # Expert 3: General Knowledge
        ],
        "description": "4-expert MoE specialization: 25% math + 25% code + 25% chat + 25% web (use with --moe_domain_expert_map)",
    },

    # Variation with more web data (common pretraining distribution)
    "moe_4expert_webfocus": {
        "mixed": True,
        "sources": [
            {"name": "math", "weight": 0.15},              # Expert 0: Math/Reasoning
            {"name": "code", "weight": 0.20},              # Expert 1: Code
            {"name": "chat", "weight": 0.20},              # Expert 2: Chat/Creative
            {"name": "finefineweb-local", "weight": 0.45}, # Expert 3: General Knowledge
        ],
        "description": "4-expert MoE with web focus: 15% math + 20% code + 20% chat + 45% web",
    },
}


# ============================================
# TEXT FORMATTERS
# ============================================
def get_formatter(formatter_type: Optional[str] = None):
    """Get text formatter function for dataset type."""
    if formatter_type == "math":
        return (
            lambda x: f"### Problem:\n{x.get('instruction', x.get('question', ''))}\n\n### Solution:\n{x.get('output', x.get('answer', ''))}"
        )
    elif formatter_type == "open_math_instruct":
        # nvidia/OpenMathInstruct-2: problem, generated_solution, expected_answer
        return (
            lambda x: f"### Problem:\n{x.get('problem', '')}\n\n### Solution:\n{x.get('generated_solution', '')}\n\n### Answer: {x.get('expected_answer', '')}"
        )
    elif formatter_type == "open_thoughts":
        # open-thoughts/OpenThoughts-114k: conversations list with from/value pairs
        def format_open_thoughts(x):
            convs = x.get("conversations", [])
            parts = []
            for turn in convs:
                role = turn.get("from", "user")
                content = turn.get("value", "")
                parts.append(f"<|{role}|>\n{content}")
            return "\n".join(parts)
        return format_open_thoughts
    elif formatter_type == "bespoke_stratos":
        # bespokelabs/Bespoke-Stratos-17k: conversations list with from/value pairs
        def format_bespoke_stratos(x):
            convs = x.get("conversations", [])
            parts = []
            for turn in convs:
                role = turn.get("from", "user")
                content = turn.get("value", "")
                parts.append(f"<|{role}|>\n{content}")
            return "\n".join(parts)
        return format_bespoke_stratos
    elif formatter_type == "code":
        return (
            lambda x: f"### Task:\n{x.get('query', x.get('instruction', ''))}\n\n### Code:\n{x.get('answer', x.get('response', ''))}"
        )
    elif formatter_type == "chat":

        def format_chat(x):
            messages = x.get("messages", [])
            parts = [
                f"<|{m.get('role', 'user')}|>\n{m.get('content', '')}" for m in messages
            ]
            return "\n".join(parts)

        return format_chat
    else:
        return lambda x: x.get("text", "")


# ============================================
# TOKENIZATION FUNCTION (for map)
# ============================================
def create_tokenize_function(
    tokenizer, seq_len: int, text_column: str = "text", formatter=None
):
    """Create a tokenization function for dataset.map()."""

    def tokenize_and_chunk(examples):
        """Tokenize and chunk text into fixed-length sequences."""
        # Get texts - either from column or using formatter
        if formatter:
            texts = [formatter(ex) for ex in _dict_to_list_of_dicts(examples)]
        else:
            texts = examples.get(text_column, examples.get("text", []))
            if isinstance(texts, str):
                texts = [texts]

        # Filter empty texts
        texts = [t for t in texts if t and len(t) > 50]

        if not texts:
            return {"input_ids": [], "labels": []}

        # Tokenize all texts at once (batched) - optimized with list comprehension
        eos_token = tokenizer.eos_token_id
        all_tokens = [
            token
            for text in texts
            for token in tokenizer.encode(text, add_special_tokens=False) + [eos_token]
        ]

        # Chunk into sequences of seq_len + 1 - optimized with list comprehension
        chunk_size = seq_len + 1
        tokens_len = len(all_tokens)
        chunks = [
            all_tokens[i : i + chunk_size]
            for i in range(0, tokens_len - chunk_size + 1, chunk_size)
        ]

        if not chunks:
            return {"input_ids": [], "labels": []}

        # Split into input_ids and labels
        input_ids = [chunk[:-1] for chunk in chunks]
        labels = [chunk[1:] for chunk in chunks]

        return {"input_ids": input_ids, "labels": labels}

    return tokenize_and_chunk


def _dict_to_list_of_dicts(batch_dict):
    """Convert {key: [values]} to [{key: value}, ...]."""
    if not batch_dict:
        return []
    keys = list(batch_dict.keys())
    if not keys:
        return []
    n = len(batch_dict[keys[0]])
    return [{k: batch_dict[k][i] for k in keys} for i in range(n)]


# ============================================
# PYTORCH-NATIVE STREAMING LOADER
# ============================================

# Configure PyArrow for optimized streaming (per HF blog Oct 2025)
# https://huggingface.co/blog/streaming-datasets
if HAS_PYARROW:
    # Increase prefetch and buffer size for better throughput
    # Default is 32MiB, increasing to 128MiB for better performance
    PARQUET_FRAGMENT_SCAN_OPTIONS = pyarrow.dataset.ParquetFragmentScanOptions(
        cache_options=pyarrow.CacheOptions(
            prefetch_limit=2,  # Prefetch 2 chunks ahead
            range_size_limit=128 << 20,  # 128MiB chunks (vs 32MiB default)
        ),
    )
else:
    PARQUET_FRAGMENT_SCAN_OPTIONS = None


class HFStreamingDataLoader:
    """
    HuggingFace streaming dataset with efficient batching.

    Features:
    - Buffered token loading for efficient batching
    - Automatic iterator reset on exhaustion
    - Fallback to synthetic data on sustained failures
    - Works on Windows (no multiprocessing pickling issues)

    Note: For true multi-worker loading, use torch.utils.data.DataLoader
    with a map-style dataset. Streaming datasets have limitations.
    """

    def __init__(
        self,
        dataset_name: str,
        batch_size: int = 4,
        seq_len: int = 512,
        vocab_size: int = 50257,
        device: str = "cuda",
        tokenizer_name: str = "gpt2",
        num_workers: int = 0,  # Used as background prefetch thread count for streaming
        prefetch_factor: int = 2,
        buffer_size: int = 10000,
        max_retries: int = 3,
        seed: Optional[int] = None,  # For reproducible shuffling
        **kwargs,
    ):
        self.dataset_name = dataset_name.lower()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.device = device
        self.buffer_size = buffer_size
        self.max_retries = max_retries
        # Use provided seed or fall back to time-based seed
        self.seed = seed if seed is not None else int(time.time())

        self.tokenizer = get_tokenizer(tokenizer_name)
        self._closed = False
        self.total_batches = 0

        # Streaming datasets are iterable and can be hard to parallelize with
        # multiprocessing safely across platforms. We instead overlap CPU-side
        # fetch+tokenize with GPU compute via background prefetch threads.
        self._prefetch_threads: List[threading.Thread] = []
        self._prefetch_stop = threading.Event()
        self._buffer_lock = threading.Lock()
        self._buffer_cv = threading.Condition(self._buffer_lock)
        # HuggingFace streaming iterators are not re-entrant/thread-safe.
        # Serialize all iterator consumption/reset through this lock.
        self._refill_lock = threading.Lock()
        self._prefetch_factor = int(prefetch_factor)
        # At most one prefetch thread is supported safely for streaming iterators.
        self._num_prefetch_threads = 1 if int(num_workers) > 0 else 0

        # Token buffer for efficient batching
        self.token_buffer = deque(maxlen=self.batch_size * (self.seq_len + 1) * 20)

        # Dataset state
        self.dataset = None
        self.iterator = None
        self.formatter = None
        self.text_column = "text"
        
        # Gradual transition mode (local -> HF based on % of training)
        self.gradual_transition_mode = False
        self.local_end_pct = 0.30       # 100% local until this % of max_steps
        self.transition_end_pct = 0.60  # 100% HF by this % of max_steps
        self.max_steps = kwargs.get("max_steps", 100000)  # Total training steps
        self.current_step = 0
        
        # Dataset sources for gradual transition
        self.local_dataset = None
        self.local_iterator = None
        self.hf_dataset = None
        self.hf_iterator = None
        self.hf_config = None
        self.hf_initialized = False
        
        # Legacy sequential mode (kept for backward compat)
        self.sequential_mode = False
        self.local_epochs_target = 0
        self.local_epochs_completed = 0
        self.using_hf_phase2 = False
        self.hf_phase2_config = None

        self._init_dataset()

        # Start background prefetching after dataset init so tokenization can
        # overlap training compute.
        self._start_prefetch_threads()

    def _start_prefetch_threads(self) -> None:
        if self._num_prefetch_threads <= 0 or self._closed:
            return
        if self._prefetch_threads:
            return

        def _worker() -> None:
            # Keep a safety margin of buffered tokens.
            # Larger prefetch_factor increases margin.
            margin_mult = max(5, 2 * max(1, self._prefetch_factor))
            target = self.batch_size * (self.seq_len + 1) * margin_mult
            while not self._prefetch_stop.is_set() and not self._closed:
                try:
                    with self._buffer_cv:
                        buf_len = len(self.token_buffer)
                        if buf_len >= target:
                            self._buffer_cv.wait(timeout=0.02)
                            continue
                    self._refill_buffer()
                except Exception:
                    # Never crash training due to background prefetch.
                    self._prefetch_stop.wait(0.05)

        for _ in range(self._num_prefetch_threads):
            t = threading.Thread(target=_worker, daemon=True)
            t.start()
            self._prefetch_threads.append(t)

    def _init_dataset(self):
        """Initialize streaming dataset with optimized PyArrow settings."""
        config = DATASET_CONFIGS.get(self.dataset_name)
        if not config:
            logger.error(f"Unknown dataset: {self.dataset_name}")
            return

        self.text_column = config.get("text_column", "text")

        for attempt in range(self.max_retries):
            try:
                logger.info(
                    f"Loading {self.dataset_name} (attempt {attempt + 1}/{self.max_retries})..."
                )

                # Load streaming dataset with optimized PyArrow settings
                load_kwargs = {
                    "split": config.get("split", "train"),
                    "streaming": True,
                }

                # Add optimized fragment scan options for Parquet datasets only
                # Per HF blog: https://huggingface.co/blog/streaming-datasets
                # Note: Only works with Parquet, not JSONL files
                is_parquet = config.get("format", "").lower() == "parquet"
                if is_parquet and PARQUET_FRAGMENT_SCAN_OPTIONS is not None:
                    load_kwargs["fragment_scan_options"] = PARQUET_FRAGMENT_SCAN_OPTIONS
                    logger.info("Using optimized PyArrow prefetch (128MiB chunks)")

                # Use local cache for FineFineWeb (download once, stream locally)
                # This avoids 502 errors and is much faster than remote streaming
                if config.get("use_gradual_transition"):
                    # Gradual transition mode: local -> HF based on % of training
                    self.gradual_transition_mode = True
                    self.local_end_pct = config.get("local_end_pct", 0.33)
                    self.transition_end_pct = config.get("transition_end_pct", 0.66)
                    self.hf_config = {
                        "path": config.get("hf_dataset", "m-a-p/FineFineWeb"),
                        "name": config.get("hf_dataset_name"),
                    }
                    
                    cache_dir = config.get("cache_dir", FINEFINEWEB_CACHE_DIR)
                    domains = config.get("domains", FINEFINEWEB_DOMAINS)
                    max_files = config.get("max_files_per_domain", 10)
                    auto_download = config.get("auto_download", True)
                    
                    # Check if local cache exists
                    cache_path = Path(cache_dir)
                    cache_exists = cache_path.exists() and any(cache_path.iterdir())
                    
                    if cache_exists:
                        # Initialize local dataset from existing cache
                        self.local_dataset = load_finefineweb_local(
                            cache_dir=cache_dir,
                            domains=domains,
                            streaming=True,
                            auto_download=False,  # Don't download, use existing
                            max_files_per_domain=max_files,
                        )
                        if self.buffer_size and self.buffer_size > 0:
                            self.local_dataset = self.local_dataset.shuffle(
                                seed=self.seed, buffer_size=min(self.buffer_size, 2048)
                            )
                        self.local_batched = self.local_dataset.batch(batch_size=32)
                        self.local_iterator = iter(self.local_batched)
                        
                        # HF dataset initialized lazily when needed
                        self.hf_initialized = False
                        
                        # Point main dataset/iterator to local for now
                        self.dataset = self.local_dataset
                        self.iterator = self.local_iterator
                        
                        logger.info(f"Gradual transition mode: max_steps={self.max_steps}")
                        logger.info(f"  0-{self.local_end_pct*100:.0f}%: 100% local")
                        logger.info(f"  {self.local_end_pct*100:.0f}-{self.transition_end_pct*100:.0f}%: gradual HF phase-in")
                        logger.info(f"  {self.transition_end_pct*100:.0f}-100%: 100% HF streaming")
                    else:
                        # No local cache - use 100% HF streaming from the start
                        logger.warning(f"Local cache not found at {cache_dir}, using 100% HF streaming")
                        self.gradual_transition_mode = False  # Disable gradual mode
                        self.local_dataset = None
                        self.local_iterator = None
                        self.hf_initialized = True
                        
                        # Load directly from HuggingFace
                        hf_path = self.hf_config["path"]
                        hf_name = self.hf_config.get("name")
                        if hf_name:
                            self.dataset = load_dataset(hf_path, hf_name, split="train", streaming=True)
                        else:
                            self.dataset = load_dataset(hf_path, split="train", streaming=True)
                        
                        if self.buffer_size and self.buffer_size > 0:
                            self.dataset = self.dataset.shuffle(seed=self.seed, buffer_size=self.buffer_size)
                        self.batched = self.dataset.batch(batch_size=32)
                        self.iterator = iter(self.batched)
                        logger.info(f"Using 100% HF streaming from {hf_path}")
                elif config.get("use_sequential"):
                    # Sequential mode: Phase 1 (local cache) then Phase 2 (HF streaming)
                    self.sequential_mode = True
                    self.local_epochs_target = config.get("local_epochs", 5)
                    self.local_epochs_completed = 0
                    self.using_hf_phase2 = False
                    self.hf_phase2_config = {
                        "path": config.get("hf_phase2", config.get("hf_fallback", "m-a-p/FineFineWeb")),
                        "name": config.get("hf_phase2_name", config.get("hf_fallback_name")),
                    }
                    
                    cache_dir = config.get("cache_dir", FINEFINEWEB_CACHE_DIR)
                    domains = config.get("domains", FINEFINEWEB_DOMAINS)
                    max_files = config.get("max_files_per_domain", 10)
                    auto_download = config.get("auto_download", False)
                    
                    self.dataset = load_finefineweb_local(
                        cache_dir=cache_dir,
                        domains=domains,
                        streaming=True,
                        auto_download=auto_download,
                        max_files_per_domain=max_files,
                    )
                    logger.info(f"Sequential mode: local cache for {self.local_epochs_target} epochs, then HF streaming")
                elif config.get("use_hybrid"):
                    # Hybrid mode: local cache + remote streaming interleaved
                    cache_dir = config.get("cache_dir", FINEFINEWEB_CACHE_DIR)
                    domains = config.get("domains", FINEFINEWEB_DOMAINS)
                    local_weight = config.get("local_weight", 0.7)
                    
                    self.dataset = load_finefineweb_hybrid(
                        cache_dir=cache_dir,
                        domains=domains,
                        local_weight=local_weight,
                        streaming=True,
                    )
                    logger.info(f"Hybrid mode: {local_weight*100:.0f}% local, {(1-local_weight)*100:.0f}% remote")
                elif config.get("use_local_cache"):
                    # Local-only mode: 100% from disk cache (no downloads)
                    cache_dir = config.get("cache_dir", FINEFINEWEB_CACHE_DIR)
                    domains = config.get("domains", FINEFINEWEB_DOMAINS)
                    max_files = config.get("max_files_per_domain", 10)
                    
                    self.dataset = load_finefineweb_local(
                        cache_dir=cache_dir,
                        domains=domains,
                        streaming=True,
                        auto_download=False,  # Local-only = no downloads ever
                        max_files_per_domain=max_files,
                    )
                    logger.info(f"Loaded from local cache: {cache_dir}")
                elif config.get("name"):
                    self.dataset = load_dataset(
                        config["path"], config["name"], **load_kwargs
                    )
                else:
                    self.dataset = load_dataset(config["path"], **load_kwargs)

                # JSONL streaming: small shuffle buffer (huge buffers stall on init)
                if self.buffer_size and self.buffer_size > 0:
                    self.dataset = self.dataset.shuffle(
                        seed=self.seed, buffer_size=min(self.buffer_size, 2048)
                    )

                # Get formatter if needed
                if config.get("formatter"):
                    self.formatter = get_formatter(config["formatter"])
                else:
                    self.formatter = None

                # Use HF's batch() method for efficient iteration
                self.batched_dataset = self.dataset.batch(batch_size=32)
                self.iterator = iter(self.batched_dataset)

                # Skip test batch probe - it stalls on large streaming datasets
                logger.info(f"Dataset {self.dataset_name} initialized (streaming mode)")
                return

            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {type(e).__name__}: {e}")
                if attempt < self.max_retries - 1:
                    wait_time = 5 * (attempt + 1)
                    logger.info(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)

        logger.error(f"Failed to load {self.dataset_name} after all retries")

    def _refill_buffer(self):
        """Fill token buffer from streaming dataset using HF batch()."""
        # Serialize iterator use to avoid HF "generator already executing" failures.
        with self._refill_lock:
            if self.dataset is None or self.iterator is None:
                self._add_synthetic_tokens(self.batch_size * (self.seq_len + 1))
                return

            needed = self.batch_size * (self.seq_len + 1) * 5
            consecutive_failures = 0
            max_consecutive_failures = 20

            while consecutive_failures < max_consecutive_failures:
                with self._buffer_lock:
                    if len(self.token_buffer) >= needed:
                        break

                try:
                    # Gradual transition mode: mix local and HF based on step %
                    if self.gradual_transition_mode:
                        batch = self._get_gradual_batch()
                    else:
                        # Standard mode: single source
                        batch = next(self.iterator)

                    # Process each sample in the batch
                    if self.formatter:
                        # Convert batch dict to list of dicts for formatter
                        samples = _dict_to_list_of_dicts(batch)
                        texts = [self.formatter(s) for s in samples]
                    else:
                        texts = batch.get(self.text_column, batch.get("text", []))

                    # Filter valid texts
                    valid_texts = [t for t in texts if t and len(t) >= 50]

                    if not valid_texts:
                        consecutive_failures += 1
                        continue

                    # Batched tokenization (5-10x faster than per-sample loop)
                    encoded = self.tokenizer(
                        valid_texts,
                        add_special_tokens=False,
                        max_length=self.seq_len * 2,
                        truncation=True,
                        padding=False,  # No padding for variable length
                        return_attention_mask=False,
                    )

                    eos = self.tokenizer.eos_token_id
                    tokens_to_add: List[int] = []
                    for toks in encoded["input_ids"]:
                        if len(toks) > 10:
                            tokens_to_add.extend(toks)
                            tokens_to_add.append(eos)

                    if tokens_to_add:
                        with self._buffer_cv:
                            self.token_buffer.extend(tokens_to_add)
                            self._buffer_cv.notify_all()
                        consecutive_failures = 0
                    else:
                        consecutive_failures += 1

                except StopIteration:
                    # Handle epoch completion for sequential mode
                    if self.sequential_mode:
                        self.local_epochs_completed += 1
                        logger.info(
                            f"Local epoch {self.local_epochs_completed}/{self.local_epochs_target} completed"
                        )

                        if (
                            self.local_epochs_completed >= self.local_epochs_target
                            and not self.using_hf_phase2
                        ):
                            # Phase 1 complete -> Switch to Phase 2 (HF streaming)
                            self._switch_to_hf_phase2()
                            consecutive_failures = 0
                            continue

                    logger.info("Iterator exhausted, resetting...")
                    self.dataset = self.dataset.shuffle(
                        seed=self.seed, buffer_size=self.buffer_size
                    )
                    self.batched_dataset = self.dataset.batch(batch_size=32)
                    self.iterator = iter(self.batched_dataset)
                    consecutive_failures = 0

                except Exception as e:
                    consecutive_failures += 1
                    if consecutive_failures % 5 == 0:
                        logger.warning(f"Fetch error ({consecutive_failures}): {e}")

            if consecutive_failures >= max_consecutive_failures:
                logger.warning("Too many failures, adding synthetic tokens")
                with self._buffer_lock:
                    missing = max(0, needed - len(self.token_buffer))
                if missing:
                    self._add_synthetic_tokens(missing)

    def _switch_to_hf_phase2(self):
        """Switch from local cache (Phase 1) to HuggingFace streaming (Phase 2).
        
        This is the INTENDED transition after N epochs on local cache:
        - Phase 1: Fast training on local cache (no network latency)
        - Phase 2: Fresh data from HF streaming (unlimited, diverse)
        """
        if not self.hf_phase2_config:
            logger.warning("No HF Phase 2 configured, continuing with local cache")
            return
        
        logger.info("=" * 60)
        logger.info(f"PHASE 2: Switching to HF streaming after {self.local_epochs_completed} local epochs")
        logger.info(f"HF dataset: {self.hf_phase2_config['path']}")
        logger.info("=" * 60)
        
        try:
            self.dataset = load_dataset(
                self.hf_phase2_config["path"],
                self.hf_phase2_config.get("name"),
                split="train",
                streaming=True,
            )
            
            if self.buffer_size and self.buffer_size > 0:
                self.dataset = self.dataset.shuffle(
                    seed=self.seed, buffer_size=min(self.buffer_size, 2048)
                )
            
            self.batched_dataset = self.dataset.batch(batch_size=32)
            self.iterator = iter(self.batched_dataset)
            self.using_hf_phase2 = True
            
            logger.info("Phase 2 active: Now streaming from HuggingFace!")
        except Exception as e:
            logger.error(f"Failed to switch to HF Phase 2: {e}")
            logger.info("Continuing with local cache (will loop)")
            # Reset local iterator
            self.dataset = self.dataset.shuffle(
                seed=self.seed, buffer_size=self.buffer_size
            )
            self.batched_dataset = self.dataset.batch(batch_size=32)
            self.iterator = iter(self.batched_dataset)

    def _init_hf_streaming(self):
        """Initialize HuggingFace streaming dataset (lazy, called when needed)."""
        if self.hf_initialized or not self.hf_config:
            return
        
        logger.info("=" * 60)
        logger.info(f"Initializing HF streaming: {self.hf_config['path']}")
        logger.info("=" * 60)
        
        try:
            self.hf_dataset = load_dataset(
                self.hf_config["path"],
                self.hf_config.get("name"),
                split="train",
                streaming=True,
            )
            
            if self.buffer_size and self.buffer_size > 0:
                self.hf_dataset = self.hf_dataset.shuffle(
                    seed=self.seed, buffer_size=min(self.buffer_size, 2048)
                )
            
            self.hf_batched = self.hf_dataset.batch(batch_size=32)
            self.hf_iterator = iter(self.hf_batched)
            self.hf_initialized = True
            
            logger.info("HF streaming initialized successfully!")
        except Exception as e:
            logger.error(f"Failed to initialize HF streaming: {e}")
            self.hf_initialized = False
    
    def _get_gradual_batch(self):
        """Get batch with gradual local->HF transition based on training progress.
        
        Schedule:
        - 0% to local_end_pct: 100% local
        - local_end_pct to transition_end_pct: linear interpolation
        - transition_end_pct to 100%: 100% HF
        """
        import random
        
        progress = self.current_step / max(self.max_steps, 1)
        
        # Calculate HF probability
        if progress < self.local_end_pct:
            hf_prob = 0.0
        elif progress >= self.transition_end_pct:
            hf_prob = 1.0
        else:
            # Linear interpolation
            hf_prob = (progress - self.local_end_pct) / (self.transition_end_pct - self.local_end_pct)
        
        # Decide source for this batch
        use_hf = random.random() < hf_prob
        
        if use_hf:
            # Ensure HF is initialized
            if not self.hf_initialized:
                self._init_hf_streaming()
            
            if self.hf_initialized and self.hf_iterator is not None:
                try:
                    return next(self.hf_iterator)
                except StopIteration:
                    # HF exhausted (shouldn't happen with streaming), reset
                    self.hf_dataset = self.hf_dataset.shuffle(
                        seed=self.seed, buffer_size=min(self.buffer_size, 2048)
                    )
                    self.hf_batched = self.hf_dataset.batch(batch_size=32)
                    self.hf_iterator = iter(self.hf_batched)
                    return next(self.hf_iterator)
        
        # Use local
        try:
            return next(self.local_iterator)
        except StopIteration:
            # Local exhausted, reset and continue
            logger.debug("Local iterator exhausted, resetting...")
            self.local_dataset = self.local_dataset.shuffle(
                seed=self.seed, buffer_size=min(self.buffer_size, 2048)
            )
            self.local_batched = self.local_dataset.batch(batch_size=32)
            self.local_iterator = iter(self.local_batched)
            return next(self.local_iterator)

    def _add_synthetic_tokens(self, count: int):
        """Add synthetic random tokens as fallback."""
        synthetic = torch.randint(0, self.vocab_size, (count,)).tolist()
        with self._buffer_cv:
            self.token_buffer.extend(synthetic)
            self._buffer_cv.notify_all()

    def get_batch(self) -> Dict[str, torch.Tensor]:
        """Get next batch."""
        if self._closed:
            raise StopIteration

        needed = self.batch_size * (self.seq_len + 1)

        start_wait = time.time()
        while True:
            with self._buffer_cv:
                have = len(self.token_buffer)
                if have >= needed:
                    break
                # If a prefetch thread is running, wait briefly for it.
                if self._num_prefetch_threads > 0 and (time.time() - start_wait) < 0.5:
                    self._buffer_cv.wait(timeout=0.02)
                    continue

            # Either no prefetch thread, or it's stalled: refill synchronously.
            start_wait = time.time()
            self._refill_buffer()

        # Extract tokens from buffer
        with self._buffer_lock:
            batch_tokens = [self.token_buffer.popleft() for _ in range(needed)]
        # If training on CUDA, pinned memory allows non_blocking H2D copies.
        pin = (self.device != "cpu") and torch.cuda.is_available()
        tokens = torch.tensor(batch_tokens, dtype=torch.long, pin_memory=pin)
        tokens = tokens.view(self.batch_size, self.seq_len + 1)

        self.total_batches += 1

        return {
            "input_ids": tokens[:, :-1],
            "labels": tokens[:, 1:],
        }
    
    def set_step(self, step: int):
        """Update current training step (for gradual transition mode)."""
        self.current_step = step
    
    def set_max_steps(self, max_steps: int):
        """Update max training steps (for gradual transition mode)."""
        self.max_steps = max_steps
        if self.gradual_transition_mode:
            logger.info(f"Updated max_steps to {max_steps}")

    def __iter__(self):
        return self

    def __next__(self) -> Dict[str, torch.Tensor]:
        return self.get_batch()

    def close(self):
        """Clean up resources."""
        self._closed = True
        self._prefetch_stop.set()
        with self._buffer_cv:
            self._buffer_cv.notify_all()
        for t in self._prefetch_threads:
            t.join(timeout=0.2)

    def stats(self) -> Dict[str, Any]:
        """Return loader statistics."""
        stats = {
            "total_batches": self.total_batches,
            "batch_size": self.batch_size,
            "buffer_size": len(self.token_buffer),
            "dataset": self.dataset_name,
        }
        if self.gradual_transition_mode:
            progress = self.current_step / max(self.max_steps, 1)
            if progress < self.local_end_pct:
                hf_pct = 0
                phase = "Phase 1 (100% local)"
            elif progress >= self.transition_end_pct:
                hf_pct = 100
                phase = "Phase 3 (100% HF streaming)"
            else:
                hf_pct = int(100 * (progress - self.local_end_pct) / (self.transition_end_pct - self.local_end_pct))
                phase = f"Phase 2 (transition: {100-hf_pct}% local, {hf_pct}% HF)"
            stats.update({
                "gradual_transition_mode": True,
                "current_step": self.current_step,
                "max_steps": self.max_steps,
                "progress_pct": progress * 100,
                "hf_probability_pct": hf_pct,
                "phase": phase,
            })
        elif self.sequential_mode:
            stats.update({
                "sequential_mode": True,
                "local_epochs_completed": self.local_epochs_completed,
                "local_epochs_target": self.local_epochs_target,
                "using_hf_phase2": self.using_hf_phase2,
                "phase": "Phase 2 (HF streaming)" if self.using_hf_phase2 else f"Phase 1 (local, epoch {self.local_epochs_completed + 1}/{self.local_epochs_target})",
            })
        return stats


# ============================================
# INTERLEAVED MULTI-DATASET LOADER
# ============================================
class InterleavedDataLoader:
    """Interleave multiple datasets with configurable weights.

    This mixes at the *batch* level by sampling one component loader per batch.
    Component loaders are created via HYDRA's native loaders (e.g. FineFineWeb
    local cache / sequential modes, local pre-tokenized .pt datasets).

    This design ensures a mixed preset can safely combine:
    - local pre-tokenized `.pt` (e.g. `LocalDataLoader`)
    - FineFineWeb local JSONL cache / sequential transition (`HFStreamingDataLoader`)
    - normal HF streaming datasets (`HFStreamingDataLoader`)
    """

    def __init__(
        self,
        datasets_config: List[Dict[str, Any]],
        probabilities: Optional[List[float]] = None,
        batch_size: int = 4,
        seq_len: int = 512,
        vocab_size: int = 50257,
        device: str = "cuda",
        tokenizer_name: str = "gpt2",
        buffer_size: int = 5000,
        **kwargs,
    ):
        """
        Args:
            datasets_config: List of dataset configs, each with 'name' key
            probabilities: Sampling probabilities for each dataset
        """
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.device = device
        self.buffer_size = buffer_size
        self.tokenizer_name = tokenizer_name

        self._closed = False
        self.total_batches = 0
        self.current_step = 0
        self.max_steps = int(kwargs.get("max_steps", 100000))
        self.seed = kwargs.get("seed", None)

        # Optional dynamic mixing schedule (updates probabilities based on step/max_steps)
        self.mix_schedule: Optional[Dict[str, Any]] = kwargs.get("mix_schedule")

        self.datasets_config = datasets_config
        self.dataset_names: list[str] = [
            str(cfg.get("name", cfg.get("dataset", f"dataset_{i}")))
            for i, cfg in enumerate(self.datasets_config)
        ]
        self.base_probabilities = probabilities or [1.0 / len(datasets_config)] * len(datasets_config)
        self.probabilities = list(self.base_probabilities)

        # Loader state
        self.loaders: list[Optional[BaseDataLoader]] = []
        self.samples_by_dataset: list[int] = []

        self._init_loaders()
        self._apply_mix_schedule()

    def _apply_mix_schedule(self) -> None:
        """Update sampling probabilities based on current step/max_steps.

        This is intentionally simple and only supports the mix schedules we define
        in DATASET_CONFIGS. Weights do not need to sum to 1.0.

        Supported schedule types:
        - tinystories_ffw_ramp: Legacy ramp from TinyStories/Pleias to FineFineWeb
        - curriculum_transition: Linear interpolation between start and end weights
        """
        if not self.mix_schedule:
            return

        sched_type = self.mix_schedule.get("type")

        # Handle curriculum_transition schedule
        if sched_type == "curriculum_transition":
            self._apply_curriculum_transition()
            return

        if sched_type != "tinystories_ffw_ramp":
            return

        names = [cfg.get("name", cfg.get("dataset")) for cfg in self.datasets_config]
        base = {names[i]: float(self.base_probabilities[i]) for i in range(len(names)) if names[i] is not None}

        tiny = self.mix_schedule.get("tiny_name", "tinystories")
        pleias = self.mix_schedule.get("pleias_name", "pleias_synth")
        ffw = self.mix_schedule.get("ffw_name", "finefineweb-sequential")

        ramp_start = int(self.mix_schedule.get("ramp_start_step", 2000))
        ramp_every = int(self.mix_schedule.get("ramp_every_steps", 1000))
        ffw_step_inc = float(self.mix_schedule.get("ffw_step_increase", 0.10))

        ffw_target = float(self.mix_schedule.get("ffw_target", 0.70))
        tiny_target = float(self.mix_schedule.get("tiny_target", 0.10))
        pleias_target = float(self.mix_schedule.get("pleias_target", 0.10))

        # Start from base weights, then apply scheduled transfers.
        weights = dict(base)

        # Optional late-phase: switch to FineFineWeb + UltraChat in last N% of long runs.
        late_if_gt = float(self.mix_schedule.get("late_chat_if_max_steps_gt", 40000))
        late_start_pct = float(self.mix_schedule.get("late_chat_start_pct", 0.90))
        late_weights = self.mix_schedule.get("late_chat_weights")
        if (
            late_weights
            and self.max_steps > late_if_gt
            and self.current_step >= int(self.max_steps * late_start_pct)
        ):
            # Drop other sources; user requested Pleias/TinyStories can be dropped.
            for n in list(weights.keys()):
                weights[n] = 0.0
            for n, w in late_weights.items():
                weights[str(n)] = float(w)
        else:
            # Ramp begins strictly AFTER ramp_start, increasing every ramp_every steps.
            # Example: ramp_start=2000 => first increment at step 3000.
            k = 0
            if ramp_every > 0 and self.current_step >= (ramp_start + ramp_every):
                k = (self.current_step - ramp_start) // ramp_every
            desired_ffw = min(ffw_target, float(weights.get(ffw, 0.0)) + ffw_step_inc * k)
            delta_ffw = max(0.0, desired_ffw - float(weights.get(ffw, 0.0)))

            # Remove weight from TinyStories first, then Pleias, until targets are reached.
            weights[ffw] = desired_ffw
            if delta_ffw > 0:
                tiny_room = max(0.0, float(weights.get(tiny, 0.0)) - tiny_target)
                take_tiny = min(delta_ffw, tiny_room)
                weights[tiny] = float(weights.get(tiny, 0.0)) - take_tiny
                delta_ffw -= take_tiny

            if delta_ffw > 0:
                pleias_room = max(0.0, float(weights.get(pleias, 0.0)) - pleias_target)
                take_pleias = min(delta_ffw, pleias_room)
                weights[pleias] = float(weights.get(pleias, 0.0)) - take_pleias
                delta_ffw -= take_pleias

            # Clamp minimums to targets (protect against overshoot when base differs).
            if tiny in weights:
                weights[tiny] = max(float(weights[tiny]), tiny_target)
            if pleias in weights:
                weights[pleias] = max(float(weights[pleias]), pleias_target)
            if ffw in weights:
                weights[ffw] = min(float(weights[ffw]), ffw_target)

        # Apply back to probability list in-place.
        new_probs: list[float] = []
        for i, n in enumerate(names):
            if n is None:
                new_probs.append(float(self.base_probabilities[i]))
            else:
                new_probs.append(float(weights.get(n, float(self.base_probabilities[i]))))

        if any(p > 0 for p in new_probs):
            self.probabilities = new_probs

    def _apply_curriculum_transition(self) -> None:
        """Apply curriculum_transition schedule: linear interpolation between phases.

        Schedule:
        - Phase 1 (step < phase1_end): Use base weights (from sources config)
        - Phase 2 (phase1_end <= step < phase2_end): Linear interpolation
        - Phase 3 (step >= phase2_end): Use end_weights

        Config keys:
        - phase1_end_step: Step at which to begin transition (default: 45000)
        - phase2_end_step: Step at which transition completes (default: 115000)
        - end_weights: Dict of dataset_name -> target_weight
        """
        phase1_end = int(self.mix_schedule.get("phase1_end_step", 45000))
        phase2_end = int(self.mix_schedule.get("phase2_end_step", 115000))
        end_weights = self.mix_schedule.get("end_weights", {})

        if not end_weights:
            logger.warning("curriculum_transition: no end_weights specified, skipping")
            return

        # Build name -> index mapping
        names = [cfg.get("name", cfg.get("dataset")) for cfg in self.datasets_config]
        name_to_idx = {n: i for i, n in enumerate(names) if n is not None}

        # Get start weights from base_probabilities
        start_weights = {n: float(self.base_probabilities[i]) for n, i in name_to_idx.items()}

        # Determine interpolation factor
        step = self.current_step
        if step < phase1_end:
            # Phase 1: 100% start weights
            alpha = 0.0
            phase = "Phase 1 (consolidate)"
        elif step >= phase2_end:
            # Phase 3: 100% end weights
            alpha = 1.0
            phase = "Phase 3 (agentic)"
        else:
            # Phase 2: linear interpolation
            alpha = (step - phase1_end) / max(phase2_end - phase1_end, 1)
            phase = f"Phase 2 (transition {alpha*100:.0f}%)"

        # Interpolate weights
        new_probs = list(self.base_probabilities)
        for name, idx in name_to_idx.items():
            start_w = start_weights.get(name, 0.0)
            end_w = float(end_weights.get(name, start_w))
            interpolated = start_w + alpha * (end_w - start_w)
            new_probs[idx] = interpolated

        # Log transition info periodically (every 1000 steps or at phase boundaries)
        if step % 1000 == 0 or step == phase1_end or step == phase2_end:
            logger.info(f"Curriculum {phase} at step {step}:")
            for name, idx in sorted(name_to_idx.items(), key=lambda x: -new_probs[x[1]]):
                start_w = start_weights.get(name, 0.0)
                end_w = float(end_weights.get(name, start_w))
                curr_w = new_probs[idx]
                if curr_w > 0.001 or end_w > 0.001:
                    logger.info(f"  {name}: {start_w*100:.1f}% -> {curr_w*100:.1f}% -> {end_w*100:.1f}%")

        if any(p > 0 for p in new_probs):
            self.probabilities = new_probs

    def _init_loaders(self) -> None:
        logger.info("Loading interleaved datasets (batch-level mixing)...")
        self.loaders = []
        self.samples_by_dataset = []

        for cfg in self.datasets_config:
            name = cfg.get("name", cfg.get("dataset"))
            if not name:
                self.loaders.append(None)
                self.samples_by_dataset.append(0)
                continue

            dataset_name = str(name).lower()
            config = DATASET_CONFIGS.get(dataset_name)
            if not config:
                logger.warning(f"  ✗ Unknown dataset: {dataset_name}, skipping")
                self.loaders.append(None)
                self.samples_by_dataset.append(0)
                continue

            if config.get("mixed"):
                logger.warning(f"  ✗ Nested mixed dataset not supported: {dataset_name}, skipping")
                self.loaders.append(None)
                self.samples_by_dataset.append(0)
                continue

            # Handle mixed_by_seq configs: resolve to the appropriate seq-specific source
            if config.get("mixed_by_seq"):
                seq_key = str(self.seq_len)
                seq_config = config["mixed_by_seq"].get(seq_key)
                if not seq_config:
                    # Fall back to closest available seq_len
                    available_seqs = sorted([int(k) for k in config["mixed_by_seq"].keys()])
                    closest = min(available_seqs, key=lambda x: abs(x - self.seq_len))
                    seq_key = str(closest)
                    seq_config = config["mixed_by_seq"].get(seq_key)
                    logger.info(f"  → {dataset_name}: no config for seq={self.seq_len}, using seq={seq_key}")
                
                if seq_config and seq_config.get("sources"):
                    # Use the first source (or could weight-sample, but typically single source)
                    resolved_name = seq_config["sources"][0]["name"]
                    resolved_config = DATASET_CONFIGS.get(resolved_name)
                    if resolved_config:
                        logger.info(f"  → {dataset_name} resolved to {resolved_name} for seq={seq_key}")
                        config = resolved_config
                        dataset_name = resolved_name
                    else:
                        logger.warning(f"  ✗ {dataset_name}: resolved source {resolved_name} not found, skipping")
                        self.loaders.append(None)
                        self.samples_by_dataset.append(0)
                        continue
                else:
                    logger.warning(f"  ✗ {dataset_name}: no sources for seq={seq_key}, skipping")
                    self.loaders.append(None)
                    self.samples_by_dataset.append(0)
                    continue

            try:
                if config.get("local"):
                    loader: BaseDataLoader = LocalDataLoader(
                        data_dir=config["path"],
                        batch_size=self.batch_size,
                        seq_len=self.seq_len,
                        vocab_size=self.vocab_size,
                        device="cpu",
                    )
                else:
                    loader = HFStreamingDataLoader(
                        dataset_name=dataset_name,
                        batch_size=self.batch_size,
                        seq_len=self.seq_len,
                        vocab_size=self.vocab_size,
                        device="cpu",
                        tokenizer_name=self.tokenizer_name,
                        buffer_size=self.buffer_size,
                        seed=self.seed,
                        max_steps=self.max_steps,
                    )
                self.loaders.append(loader)
                self.samples_by_dataset.append(0)
                logger.info(f"  ✓ Loaded {dataset_name}")
            except Exception as e:
                logger.warning(f"  ✗ Failed to initialize {dataset_name}: {e}")
                self.loaders.append(None)
                self.samples_by_dataset.append(0)

        available = [i for i, loader in enumerate(self.loaders) if loader is not None]
        if not available:
            logger.warning("No datasets available for interleaving")
            return

        if len(available) < len(self.datasets_config):
            logger.warning(f"Only {len(available)}/{len(self.datasets_config)} datasets available")

    def get_batch(self) -> Dict[str, torch.Tensor]:
        """Get next batch."""
        if self._closed:
            raise StopIteration

        available_indices = [i for i, loader in enumerate(self.loaders) if loader is not None]
        if not available_indices:
            tokens = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len + 1))
            return {"input_ids": tokens[:, :-1], "labels": tokens[:, 1:]}

        weights = [self.probabilities[i] for i in available_indices]
        idx = random.choices(available_indices, weights=weights)[0]
        loader = self.loaders[idx]
        assert loader is not None

        batch = loader.get_batch()
        # Attach source metadata so the trainer can do domain-aware MoE routing/teacher.
        # Keep it lightweight and backwards-compatible.
        try:
            batch["source_id"] = int(idx)
            batch["source_name"] = self.dataset_names[int(idx)]
        except Exception:
            pass
        self.total_batches += 1
        try:
            self.samples_by_dataset[idx] += int(batch["input_ids"].shape[0])
        except Exception:
            pass
        return batch

    def __iter__(self):
        return self

    def __next__(self):
        return self.get_batch()

    def close(self):
        self._closed = True
        for loader in self.loaders:
            if loader is not None and hasattr(loader, "close"):
                loader.close()

    def set_step(self, step: int):
        self.current_step = int(step)
        self._apply_mix_schedule()
        for loader in self.loaders:
            if loader is not None and hasattr(loader, "set_step"):
                loader.set_step(self.current_step)

    def set_max_steps(self, max_steps: int):
        self.max_steps = int(max_steps)
        self._apply_mix_schedule()
        for loader in self.loaders:
            if loader is not None and hasattr(loader, "set_max_steps"):
                loader.set_max_steps(self.max_steps)

    def stats(self) -> Dict[str, Any]:
        return {
            "total_batches": self.total_batches,
            "samples_by_dataset": self.samples_by_dataset,
            "current_step": self.current_step,
            "max_steps": self.max_steps,
        }


# ============================================
# BACKWARDS-COMPATIBLE BASE CLASSES
# ============================================
class BaseDataLoader:
    """Base class for all data loaders (backwards compatible)."""

    def __init__(
        self,
        batch_size: int = 4,
        seq_len: int = 512,
        vocab_size: int = 50257,
        device: str = "cuda",
        tokenizer_name: str = "gpt2",
        skip_tokenizer: bool = False,
    ):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.device = device
        if skip_tokenizer:
            self.tokenizer = None
        else:
            self.tokenizer = get_tokenizer(tokenizer_name)
        self._closed = False

    def get_batch(self) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def __iter__(self):
        return self

    def __next__(self) -> Dict[str, torch.Tensor]:
        if self._closed:
            raise StopIteration
        return self.get_batch()

    def close(self):
        self._closed = True

    def stats(self) -> Dict[str, Any]:
        return {}


class SyntheticDataLoader(BaseDataLoader):
    """Generate random token sequences for testing."""

    def get_batch(self) -> Dict[str, torch.Tensor]:
        tokens = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len + 1))
        return {
            "input_ids": tokens[:, :-1],
            "labels": tokens[:, 1:],
        }


class LocalDataLoader(BaseDataLoader):
    """Load preprocessed .pt chunk files."""

    def __init__(self, data_dir: str, **kwargs):
        kwargs["skip_tokenizer"] = True
        super().__init__(**kwargs)
        self.data_dir = Path(data_dir)

        # Padding policy:
        # - Never pad with token id 0 (can create pathological batches).
        # - Prefer EOS as pad for GPT-2-style vocabularies.
        _pad_override = os.environ.get("HYDRA_LOCAL_PT_PAD_TOKEN_ID", "").strip()
        if _pad_override:
            try:
                self._pad_token_id = int(_pad_override)
            except Exception:
                self._pad_token_id = int(self.vocab_size - 1) if int(self.vocab_size) > 0 else 0
        else:
            self._pad_token_id = int(self.vocab_size - 1) if int(self.vocab_size) > 0 else 0

        # Resampling guard: avoid batches dominated by padding (keeps loss scaling stable).
        self._max_pad_frac = float(os.environ.get("HYDRA_LOCAL_PT_MAX_PAD_FRAC", "0.25") or 0.25)
        self._max_resample_tries = int(os.environ.get("HYDRA_LOCAL_PT_MAX_RESAMPLE_TRIES", "200") or 200)
        self._warned_excessive_padding = False
        self._skipped_too_padded = 0
        self._forced_accept_padded = 0

        if not self.data_dir.exists():
            raise FileNotFoundError(
                "LocalDataLoader data_dir does not exist. "
                f"Got: {self.data_dir}. "
                "If this is an external drive, it may not be mounted or your env vars are not set. "
                "Set HYDRA_DATA_ROOT or HYDRA_SMALL_CHAT_PT_DIR / HYDRA_NEMOTRON_PT_DIR to a stable mount point (e.g. /mnt/hydra_data)."
            )
        if not self.data_dir.is_dir():
            raise NotADirectoryError(f"LocalDataLoader data_dir is not a directory: {self.data_dir}")

        self.chunk_files = sorted(glob.glob(str(self.data_dir / "chunk_*.pt")))
        if not self.chunk_files:
            self.chunk_files = sorted(glob.glob(str(self.data_dir / "*.pt")))

        if not self.chunk_files:
            # Common confusion: pointing at a parent dir that contains per-dataset or per-seq folders.
            child_shards = sorted(glob.glob(str(self.data_dir / "*" / "chunk_*.pt")))
            if not child_shards:
                child_shards = sorted(glob.glob(str(self.data_dir / "*" / "*.pt")))

            entries = []
            try:
                entries = sorted([p.name for p in self.data_dir.iterdir()])[:16]
            except Exception:
                entries = []

            msg = f"No .pt files found in {self.data_dir}"
            if entries:
                msg += f". Directory entries (first {len(entries)}): {entries}"
            if child_shards:
                msg += (
                    ". However, .pt shards exist in subdirectories. "
                    "This usually means you passed the wrong folder level (e.g. you passed <root>/<seq_len> but need <root>/<seq_len>/<dataset>). "
                    "If you intend to train on the consolidated per-seq flat shards, use dataset 'chat_flat_512'/'chat_flat_1024'/... or 'small_chat_seqaware_flat'."
                )
            raise ValueError(msg)

        random.shuffle(self.chunk_files)
        logger.info(f"LocalDataLoader: Found {len(self.chunk_files)} chunk files")

        self.current_chunk = None
        self.current_idx = 0
        self.chunk_file_idx = 0
        self.total_samples = 0
        self._load_next_chunk()

    def _load_next_chunk(self):
        if self.chunk_file_idx >= len(self.chunk_files):
            random.shuffle(self.chunk_files)
            self.chunk_file_idx = 0

        self.current_chunk = torch.load(
            self.chunk_files[self.chunk_file_idx], weights_only=False
        )
        self.current_idx = 0
        self.chunk_file_idx += 1

    def get_batch(self) -> Dict[str, torch.Tensor]:
        sequences = []
        pad_masks = []  # Track which positions are padding

        # Compute in local vars (hot path)
        target_len = int(self.seq_len) + 1
        max_pad = int(self._max_pad_frac * float(target_len))
        max_pad = max(0, min(max_pad, target_len))
        tries = 0

        while len(sequences) < self.batch_size:
            tries += 1
            if self.current_idx >= len(self.current_chunk):
                self._load_next_chunk()

            sample = self.current_chunk[self.current_idx]
            self.current_idx += 1
            self.total_samples += 1

            if isinstance(sample, dict):
                tokens = sample.get("input_ids", sample.get("tokens", []))
                if isinstance(tokens, torch.Tensor):
                    tokens = tokens.tolist()
            elif isinstance(sample, (list, torch.Tensor)):
                tokens = sample if isinstance(sample, list) else sample.tolist()
            else:
                continue

            tok_len = len(tokens)
            if tok_len >= target_len:
                sequences.append(tokens[:target_len])
                pad_masks.append([False] * target_len)  # No padding
                continue

            # Skip extremely short samples (legacy heuristic).
            if tok_len <= 64:
                continue

            n_pad = target_len - tok_len

            # If this sample would be heavily padded, resample a different sample.
            # This prevents tiny valid_count batches which can amplify gradients.
            if n_pad > max_pad and tries <= self._max_resample_tries:
                self._skipped_too_padded += 1
                continue

            if n_pad > max_pad and (not self._warned_excessive_padding):
                logger.warning(
                    "LocalDataLoader: forced to accept heavily-padded sequences "
                    f"(n_pad={n_pad}, max_pad={max_pad}, target_len={target_len}). "
                    "Consider increasing dataset packing or lowering HYDRA_LOCAL_PT_MAX_PAD_FRAC."
                )
                self._warned_excessive_padding = True
                self._forced_accept_padded += 1

            padded = tokens + [self._pad_token_id] * n_pad
            sequences.append(padded[:target_len])
            pad_masks.append([False] * tok_len + [True] * n_pad)

        batch = torch.tensor(sequences[: self.batch_size], dtype=torch.long)
        input_ids = batch[:, :-1]
        labels = batch[:, 1:].clone()
        
        # Set labels to -100 for padded positions (CE ignore_index)
        pad_mask_tensor = torch.tensor(pad_masks[:self.batch_size], dtype=torch.bool)[:, 1:]
        labels[pad_mask_tensor] = -100
        
        # Create attention mask for input_ids (1 for valid, 0 for padded)
        # pad_masks matches the full sequence (input + 1 for label shift).
        # We need the mask corresponding to input_ids ([:, :-1]).
        input_pad_mask = torch.tensor(pad_masks[:self.batch_size], dtype=torch.bool)[:, :-1]
        attention_mask = (~input_pad_mask).long()

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

    def stats(self) -> Dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "chunk_files": len(self.chunk_files),
            "current_chunk_idx": self.chunk_file_idx,
            "skipped_too_padded": self._skipped_too_padded,
            "forced_accept_padded": self._forced_accept_padded,
            "pad_token_id": int(getattr(self, "_pad_token_id", 0)),
        }


class LocalJSONLDataLoader(BaseDataLoader):
    """
    Load pre-filtered JSONL data from data_filter.py output.
    
    Supports:
    - Single .jsonl file
    - Directory with multiple .jsonl files
    - Streaming (memory-efficient for large datasets)
    """
    
    def __init__(self, data_dir: str, **kwargs):
        super().__init__(**kwargs)
        self.data_path = Path(data_dir)
        
        # Find JSONL files
        if self.data_path.is_file():
            self.jsonl_files = [self.data_path]
        else:
            self.jsonl_files = sorted(self.data_path.glob("**/*.jsonl"))
        
        if not self.jsonl_files:
            raise ValueError(f"No .jsonl files found in {data_dir}")
        
        random.shuffle(self.jsonl_files)
        logger.info(f"LocalJSONLDataLoader: Found {len(self.jsonl_files)} JSONL files")
        
        self.current_file_idx = 0
        self.current_file = None
        self.total_samples = 0
        self.buffer: List[str] = []
        self.buffer_idx = 0
        self.buffer_size = 1000  # Read 1000 lines at a time
        
        self._open_next_file()
    
    def _open_next_file(self):
        """Open the next JSONL file."""
        if self.current_file:
            self.current_file.close()
        
        if self.current_file_idx >= len(self.jsonl_files):
            random.shuffle(self.jsonl_files)
            self.current_file_idx = 0
        
        self.current_file = open(self.jsonl_files[self.current_file_idx], 'r')
        self.current_file_idx += 1
        self._refill_buffer()
    
    def _refill_buffer(self):
        """Read more lines into the buffer."""
        self.buffer = []
        self.buffer_idx = 0
        
        for _ in range(self.buffer_size):
            line = self.current_file.readline()
            if not line:
                break
            self.buffer.append(line)
        
        if not self.buffer:
            self._open_next_file()
    
    def _get_next_text(self) -> str:
        """Get the next text sample."""
        import json
        
        if self.buffer_idx >= len(self.buffer):
            self._refill_buffer()
        
        line = self.buffer[self.buffer_idx]
        self.buffer_idx += 1
        
        try:
            sample = json.loads(line)
            return sample.get('text', '')
        except json.JSONDecodeError:
            return ''
    
    def get_batch(self) -> Dict[str, torch.Tensor]:
        sequences = []
        pad_masks = []  # Track which positions are padding
        
        while len(sequences) < self.batch_size:
            text = self._get_next_text()
            self.total_samples += 1
            
            if not text:
                continue
            
            # Tokenize
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            
            if len(tokens) >= self.seq_len + 1:
                # Take a random chunk if longer than seq_len
                if len(tokens) > self.seq_len + 1:
                    start = random.randint(0, len(tokens) - self.seq_len - 1)
                    tokens = tokens[start:start + self.seq_len + 1]
                sequences.append(tokens[:self.seq_len + 1])
                pad_masks.append([False] * (self.seq_len + 1))  # No padding
            elif len(tokens) > 64:
                # Pad shorter sequences - track padding positions
                n_pad = self.seq_len + 1 - len(tokens)
                pad_token = self.tokenizer.eos_token_id or 0
                padded = tokens + [pad_token] * n_pad
                sequences.append(padded[:self.seq_len + 1])
                # Mask: False for real tokens, True for padding
                pad_masks.append([False] * len(tokens) + [True] * n_pad)
        
        batch = torch.tensor(sequences[:self.batch_size], dtype=torch.long)
        input_ids = batch[:, :-1]
        labels = batch[:, 1:].clone()
        
        # Set labels to -100 for padded positions (CE ignore_index)
        # This prevents the model from learning to predict EOS at padding
        pad_mask_tensor = torch.tensor(pad_masks[:self.batch_size], dtype=torch.bool)[:, 1:]
        labels[pad_mask_tensor] = -100
        
        return {
            "input_ids": input_ids,
            "labels": labels,
        }
    
    def stats(self) -> Dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "jsonl_files": len(self.jsonl_files),
            "current_file_idx": self.current_file_idx,
        }
    
    def __del__(self):
        if hasattr(self, 'current_file') and self.current_file:
            self.current_file.close()


# Keep old StreamingDataLoader for backwards compatibility
StreamingDataLoader = HFStreamingDataLoader


# ============================================
# UNIVERSAL LOADER FACTORY
# ============================================
def create_universal_loader(
    dataset: str = "auto",
    model_params: Optional[int] = None,
    batch_size: int = 4,
    seq_len: int = 512,
    vocab_size: int = 50257,
    device: str = "cuda",
    tokenizer_name: str = "gpt2",
    data_dir: Optional[str] = None,
    num_workers: int = 4,  # NEW: parallel workers
    prefetch_factor: int = 2,  # NEW: prefetch batches per worker
    seed: Optional[int] = None,  # For reproducible shuffling
    # Mix weights for interleaved datasets
    text_weight: float = 0.40,
    math_weight: float = 0.20,
    code_weight: float = 0.20,
    qa_weight: float = 0.20,
    **kwargs,
) -> BaseDataLoader:
    """
    Create a universal data loader with native PyTorch integration.

    Args:
        dataset: Dataset name or type:
            - "auto": Auto-select based on model_params
            - "synthetic": Random tokens (for testing)
            - "synthetic_mix" / "mix": Interleaved math + code + Q&A
            - "local": Local .pt files (requires data_dir)
            - "finefineweb", "fineweb", "fineweb_edu": Streaming web text
            - "wikitext2", "tinystories": Small test datasets
        model_params: Number of model parameters (for "auto" selection)
        batch_size: Batch size
        seq_len: Sequence length
        vocab_size: Vocabulary size
        device: Target device
        tokenizer_name: Tokenizer to use
        data_dir: Directory for local data (required if dataset="local")
        num_workers: Number of parallel data loading workers (default: 4)
        prefetch_factor: Batches to prefetch per worker (default: 2)
        seed: Random seed for reproducible shuffling (default: None = time-based)

    Returns:
        DataLoader instance
    """
    dataset = dataset.lower()
    def _resolve_mixed_by_seq_sources(cfg: Dict[str, Any], seq: int) -> Optional[List[Dict[str, Any]]]:
        by_seq = cfg.get("mixed_by_seq")
        if not isinstance(by_seq, dict):
            return None

        keys: list[int] = []
        for k in by_seq.keys():
            try:
                keys.append(int(k))
            except Exception:
                continue
        if not keys:
            return None

        seq_i = int(seq)
        chosen = max([k for k in keys if k <= seq_i], default=min(keys))
        entry = by_seq.get(str(chosen), {})
        if isinstance(entry, dict):
            sources = entry.get("sources")
            if isinstance(sources, list) and sources:
                return sources
        return None

    # Auto-select based on model size
    if dataset == "auto":
        if model_params is None:
            model_params = 100_000_000

        if model_params < 50_000_000:
            dataset = "wikitext2"
        elif model_params < 100_000_000:
            dataset = "tinystories"
        elif model_params < 500_000_000:
            dataset = "finefineweb"
        else:
            dataset = "fineweb"

        logger.info(
            f"Auto-selected dataset '{dataset}' for {model_params / 1e6:.0f}M params"
        )

    # Common kwargs
    common_kwargs = {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "vocab_size": vocab_size,
        "device": device,
        "tokenizer_name": tokenizer_name,
        "seed": seed,
    }
    
    # Kwargs without seed for loaders that don't support it
    basic_kwargs = {k: v for k, v in common_kwargs.items() if k != "seed"}

    # Create loader based on type
    if dataset == "synthetic":
        return SyntheticDataLoader(**basic_kwargs)

    elif dataset == "local":
        if not data_dir:
            raise ValueError("data_dir required for local dataset")
        return LocalDataLoader(data_dir=data_dir, **basic_kwargs)
    
    elif dataset == "local_jsonl" or dataset == "jsonl":
        if not data_dir:
            raise ValueError("data_dir required for local_jsonl dataset")
        return LocalJSONLDataLoader(data_dir=data_dir, **basic_kwargs)

    # Check if it's a configured local dataset (pre-tokenized .pt files)
    elif dataset in DATASET_CONFIGS and DATASET_CONFIGS[dataset].get("local"):
        config = DATASET_CONFIGS[dataset]
        return LocalDataLoader(data_dir=config["path"], **basic_kwargs)

    elif dataset in ["synthetic_mix", "mix"]:
        # Use interleaved loader with multiple datasets
        datasets_config = [
            {"name": "finefineweb"},
            {"name": "math"},
            {"name": "code"},
            {"name": "chat"},
        ]
        probabilities = [text_weight, math_weight, code_weight, qa_weight]

        return InterleavedDataLoader(
            datasets_config=datasets_config,
            probabilities=probabilities,
            num_workers=num_workers,
            **common_kwargs,
            **kwargs,
        )
    # Sequence-aware mixed dataset configs (select sources based on seq_len)
    elif dataset in DATASET_CONFIGS and isinstance(DATASET_CONFIGS[dataset], dict) and DATASET_CONFIGS[dataset].get("mixed_by_seq"):
        config = DATASET_CONFIGS[dataset]
        sources = _resolve_mixed_by_seq_sources(config, seq_len)
        if not sources:
            logger.warning(f"mixed_by_seq dataset '{dataset}' has no valid sources; falling back to synthetic")
            return SyntheticDataLoader(**basic_kwargs)

        datasets_config = [{"name": s["name"]} for s in sources]
        probabilities = [s.get("weight", 0.0) for s in sources]

        logger.info(f"Creating seq-aware mixed dataset '{dataset}' for seq_len={seq_len}:")
        for s in sources:
            logger.info(f"  - {s['name']}: {float(s.get('weight', 0.0))*100:.0f}%")

        return InterleavedDataLoader(
            datasets_config=datasets_config,
            probabilities=probabilities,
            num_workers=num_workers,
            **common_kwargs,
            **kwargs,
        )

    # Support for mixed dataset configs (pretrain_1b, sft_chat, etc.)
    elif dataset in DATASET_CONFIGS and DATASET_CONFIGS[dataset].get("mixed"):
        config = DATASET_CONFIGS[dataset]
        sources = config["sources"]
        
        datasets_config = [{"name": s["name"]} for s in sources]
        probabilities = [s["weight"] for s in sources]
        
        logger.info(f"Creating mixed dataset '{dataset}':")
        for s in sources:
            logger.info(f"  - {s['name']}: {s['weight']*100:.0f}%")
        
        return InterleavedDataLoader(
            datasets_config=datasets_config,
            probabilities=probabilities,
            num_workers=num_workers,
            mix_schedule=config.get("mix_schedule"),
            **common_kwargs,
            **kwargs,
        )

    elif dataset in DATASET_CONFIGS:
        return HFStreamingDataLoader(
            dataset_name=dataset,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            **common_kwargs,
            **kwargs,
        )

    else:
        logger.warning(f"Unknown dataset '{dataset}', using synthetic")
        return SyntheticDataLoader(**basic_kwargs)


# ============================================
# CONVENIENCE FUNCTIONS
# ============================================
def get_available_datasets() -> List[str]:
    """Return list of available dataset names."""
    return ["synthetic", "synthetic_mix", "local"] + list(DATASET_CONFIGS.keys())


def test_loader(dataset: str = "finefineweb", num_batches: int = 3):
    """Test a data loader."""
    print(f"Testing {dataset} loader...")

    try:
        loader = create_universal_loader(
            dataset=dataset,
            batch_size=2,
            seq_len=128,
            num_workers=2,
        )

        start = time.time()
        for i, batch in enumerate(loader):
            if i >= num_batches:
                break
            print(
                f"  Batch {i + 1}: input_ids={batch['input_ids'].shape}, labels={batch['labels'].shape}"
            )
            print(
                f"    Token range: [{batch['input_ids'].min().item()}, {batch['input_ids'].max().item()}]"
            )

        elapsed = time.time() - start
        print(f"  Stats: {loader.stats()}")
        print(
            f"  Time for {num_batches} batches: {elapsed:.2f}s ({elapsed / num_batches:.3f}s/batch)"
        )
        loader.close()
        print("  ✓ Test passed!")
        return True

    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("Universal Data Loader v2 Tests")
    print("=" * 60)

    # Test synthetic first (always works)
    test_loader("synthetic")

    # Test streaming
    test_loader("finefineweb")
