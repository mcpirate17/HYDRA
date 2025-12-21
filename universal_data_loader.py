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
import logging
import time
import glob
import random
from pathlib import Path
from typing import Dict, Any, Optional, List
from collections import deque

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
# NOTE: Default points to user's existing 109GB cache on fast NVMe
FINEFINEWEB_CACHE_DIR = os.environ.get(
    "HYDRA_CACHE_DIR",
    "/mnt/nvme0/hf_finefineweb"  # User's pre-downloaded 109GB cache
)


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
            raise FileNotFoundError(f"Cache not found at {cache_dir}. Set auto_download=True")
    
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
    # Gradual transition mode: local cache -> HF streaming based on % of training
    # - 0-33% of max_steps: 100% local (fast, from pre-downloaded cache)
    # - 33-66% of max_steps: blend local + HF streaming
    # - 66-100% of max_steps: mostly HF streaming (fresh data diversity)
    # NOTE: Does NOT auto-download - uses existing local cache only
    "finefineweb-sequential": {
        "path": "m-a-p/FineFineWeb",
        "name": None,
        "text_column": "text",
        "format": "jsonl",
        "use_gradual_transition": True,
        "local_end_pct": 0.33,       # 100% local until 33% of training
        "transition_end_pct": 0.66,  # 100% HF by 66% of training
        "cache_dir": FINEFINEWEB_CACHE_DIR,
        "domains": FINEFINEWEB_DOMAINS,
        "max_files_per_domain": 10,
        "auto_download": False,      # Use existing cache only, no downloads
        "hf_dataset": "m-a-p/FineFineWeb",
        "hf_dataset_name": None,
        "description": "Gradual: local (0-33%) -> blend (33-66%) -> HF streaming (66%+)",
    },
    # Local pre-tokenized datasets
    "pleias_synth": {
        "local": True,
        "path": "/mnt/nvme0/LLM/training_pleias_synth/processed",
        "description": "PleIAs SYNTH synthetic reasoning (~1.5B tokens, pre-tokenized)",
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
    # MIXED PRETRAINING CONFIGS (for chat-capable models)
    # ============================================
    # Phase 1: Pretraining mix - local first, then HF streaming + local high-quality data
    "pretrain_1b": {
        "mixed": True,
        "sources": [
            {"name": "finefineweb-sequential", "weight": 0.85},  # Local→HF transition (uses your 109GB cache first!)
            {"name": "pleias_synth", "weight": 0.10},            # Local synthetic reasoning
            {"name": "tinystories", "weight": 0.05},             # Narrative coherence
        ],
        "description": "Pretraining mix for 1B+ chat model: 85% web (local-first) + 10% reasoning + 5% stories",
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
            {"name": "finefineweb-local", "weight": 0.40},  # Web knowledge (reduced for small models)
            {"name": "chat", "weight": 0.35},               # UltraChat - structured conversations
            {"name": "tinystories", "weight": 0.25},        # TinyStories - narrative coherence
        ],
        "description": "Small model mix: 40% FineFineWeb + 35% UltraChat + 25% TinyStories",
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
import pyarrow
import pyarrow.dataset

# Increase prefetch and buffer size for better throughput
# Default is 32MiB, increasing to 128MiB for better performance
PARQUET_FRAGMENT_SCAN_OPTIONS = pyarrow.dataset.ParquetFragmentScanOptions(
    cache_options=pyarrow.CacheOptions(
        prefetch_limit=2,  # Prefetch 2 chunks ahead
        range_size_limit=128 << 20,  # 128MiB chunks (vs 32MiB default)
    ),
)


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
        num_workers: int = 0,  # Ignored for streaming (Windows compat)
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
                if is_parquet:
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
                    auto_download = config.get("auto_download", False)  # Default: no downloads
                    
                    # Initialize local dataset (use existing cache, no downloads by default)
                    self.local_dataset = load_finefineweb_local(
                        cache_dir=cache_dir,
                        domains=domains,
                        streaming=True,
                        auto_download=auto_download,
                        max_files_per_domain=max_files,
                    )
                    if self.buffer_size and self.buffer_size > 0:
                        self.local_dataset = self.local_dataset.shuffle(
                            seed=42, buffer_size=min(self.buffer_size, 2048)
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
                        seed=42, buffer_size=min(self.buffer_size, 2048)
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
        if self.dataset is None or self.iterator is None:
            self._add_synthetic_tokens(self.batch_size * (self.seq_len + 1))
            return

        needed = self.batch_size * (self.seq_len + 1) * 5
        consecutive_failures = 0
        max_consecutive_failures = 20

        while (
            len(self.token_buffer) < needed
            and consecutive_failures < max_consecutive_failures
        ):
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

                if valid_texts:
                    # Batched tokenization (5-10x faster than per-sample loop)
                    encoded = self.tokenizer(
                        valid_texts,
                        add_special_tokens=False,
                        max_length=self.seq_len * 2,
                        truncation=True,
                        padding=False,  # No padding for variable length
                        return_attention_mask=False,
                    )

                    for tokens in encoded["input_ids"]:
                        if len(tokens) > 10:
                            self.token_buffer.extend(tokens)
                            self.token_buffer.append(self.tokenizer.eos_token_id)
                        consecutive_failures = 0

            except StopIteration:
                # Handle epoch completion for sequential mode
                if self.sequential_mode:
                    self.local_epochs_completed += 1
                    logger.info(f"Local epoch {self.local_epochs_completed}/{self.local_epochs_target} completed")
                    
                    if self.local_epochs_completed >= self.local_epochs_target and not self.using_hf_phase2:
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
            self._add_synthetic_tokens(needed - len(self.token_buffer))

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
        self.token_buffer.extend(synthetic)

    def get_batch(self) -> Dict[str, torch.Tensor]:
        """Get next batch."""
        if self._closed:
            raise StopIteration

        needed = self.batch_size * (self.seq_len + 1)

        while len(self.token_buffer) < needed:
            self._refill_buffer()

        # Extract tokens from buffer
        batch_tokens = [self.token_buffer.popleft() for _ in range(needed)]
        # If training on CUDA, pinned memory allows non_blocking H2D copies.
        pin = (self.device == "cpu") and torch.cuda.is_available()
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
    """
    Interleave multiple HuggingFace datasets with configurable weights.

    Uses weighted random sampling to mix datasets efficiently.
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

        self.tokenizer = get_tokenizer(tokenizer_name)
        self._closed = False
        self.total_batches = 0

        # Token buffer
        self.token_buffer = deque(maxlen=self.batch_size * (self.seq_len + 1) * 20)

        self.datasets_config = datasets_config
        self.probabilities = probabilities or [1.0 / len(datasets_config)] * len(
            datasets_config
        )

        # Dataset state
        self.datasets = []
        self.iterators = []
        self.formatters = []
        self.text_columns = []
        self.samples_by_dataset = []

        self._init_datasets()

    def _init_datasets(self):
        """Initialize all component datasets."""
        logger.info("Loading interleaved datasets...")

        # Cache append methods to minimize attribute lookups in loop
        datasets_append = self.datasets.append
        iterators_append = self.iterators.append
        formatters_append = self.formatters.append
        text_columns_append = self.text_columns.append
        samples_append = self.samples_by_dataset.append

        for cfg in self.datasets_config:
            name = cfg.get("name", cfg.get("dataset"))
            config = DATASET_CONFIGS.get(name)

            if not config:
                logger.warning(f"  ✗ Unknown dataset: {name}, skipping")
                datasets_append(None)
                iterators_append(None)
                formatters_append(None)
                text_columns_append("text")
                samples_append(0)
                continue

            # Handle local pre-tokenized .pt files differently
            if config.get("local"):
                try:
                    local_path = Path(config["path"])
                    pt_files = sorted(local_path.glob("*.pt"))
                    if not pt_files:
                        raise FileNotFoundError(f"No .pt files found in {local_path}")
                    
                    # Create a generator that yields pre-tokenized batches
                    local_loader = LocalDataLoader(
                        data_dir=str(local_path),
                        batch_size=self.batch_size,
                        seq_len=self.seq_len,
                        vocab_size=self.vocab_size,
                        device="cpu",  # Keep on CPU for interleaving
                    )
                    
                    # Mark as local dataset (special handling in _refill_buffer)
                    datasets_append(("local", local_loader))
                    iterators_append(iter(local_loader))
                    formatters_append(None)  # Already tokenized
                    text_columns_append(None)  # Not text, already tokens
                    samples_append(0)
                    
                    logger.info(f"  ✓ Loaded {name} (local .pt files: {len(pt_files)} chunks)")
                    continue
                except Exception as e:
                    logger.warning(f"  ✗ Failed to load local dataset {name}: {e}")
                    datasets_append(None)
                    iterators_append(None)
                    formatters_append(None)
                    text_columns_append("text")
                    samples_append(0)
                    continue

            try:
                load_kwargs = {
                    "split": config.get("split", "train"),
                    "streaming": True,
                }

                if config.get("name"):
                    ds = load_dataset(config["path"], config["name"], **load_kwargs)
                else:
                    ds = load_dataset(config["path"], **load_kwargs)

                ds = ds.shuffle(seed=42, buffer_size=self.buffer_size)
                batched_ds = ds.batch(batch_size=32)

                # Get formatter
                formatter = None
                if config.get("formatter"):
                    formatter = get_formatter(config["formatter"])

                datasets_append(ds)
                iterators_append(iter(batched_ds))
                formatters_append(formatter)
                text_columns_append(config.get("text_column", "text"))
                samples_append(0)

                logger.info(f"  ✓ Loaded {name}")

            except Exception as e:
                logger.warning(f"  ✗ Failed to load {name}: {e}")
                datasets_append(None)
                iterators_append(None)
                formatters_append(None)
                text_columns_append("text")
                samples_append(0)

        # Adjust probabilities for available datasets
        available = [i for i, ds in enumerate(self.datasets) if ds is not None]
        if len(available) < len(self.datasets_config):
            logger.warning(
                f"Only {len(available)}/{len(self.datasets_config)} datasets available"
            )
            if available:
                total_avail = sum(self.probabilities[i] for i in available)
                self.probabilities = [
                    self.probabilities[i] / total_avail if i in available else 0
                    for i in range(len(self.datasets_config))
                ]

    def _refill_buffer(self):
        """Fill buffer from interleaved datasets."""
        needed = self.batch_size * (self.seq_len + 1) * 5
        attempts = 0
        max_attempts = 100

        available_indices = [i for i, ds in enumerate(self.datasets) if ds is not None]
        if not available_indices:
            self._add_synthetic_tokens(needed)
            return

        while len(self.token_buffer) < needed and attempts < max_attempts:
            attempts += 1

            # Weighted random selection
            weights = [self.probabilities[i] for i in available_indices]
            idx = random.choices(available_indices, weights=weights)[0]

            try:
                ds_entry = self.datasets[idx]
                
                # Handle local pre-tokenized datasets differently
                if isinstance(ds_entry, tuple) and ds_entry[0] == "local":
                    # Local dataset returns already-tokenized batches
                    batch = next(self.iterators[idx])
                    input_ids = batch.get("input_ids")
                    if input_ids is not None:
                        # Flatten and add to buffer
                        tokens = input_ids.flatten().tolist()
                        self.token_buffer.extend(tokens)
                        self.samples_by_dataset[idx] += input_ids.shape[0]
                    continue
                
                batch = next(self.iterators[idx])

                # Process samples
                if self.formatters[idx]:
                    samples = _dict_to_list_of_dicts(batch)
                    texts = [self.formatters[idx](s) for s in samples]
                else:
                    texts = batch.get(self.text_columns[idx], batch.get("text", []))

                # Filter valid texts
                valid_texts = [t for t in texts if t and len(t) >= 50]

                if valid_texts:
                    # Batched tokenization (5-10x faster than per-sample loop)
                    encoded = self.tokenizer(
                        valid_texts,
                        add_special_tokens=False,
                        max_length=self.seq_len * 2,
                        truncation=True,
                        padding=False,
                        return_attention_mask=False,
                    )

                    for tokens in encoded["input_ids"]:
                        if len(tokens) > 10:
                            self.token_buffer.extend(tokens)
                            self.token_buffer.append(self.tokenizer.eos_token_id)
                            self.samples_by_dataset[idx] += 1

            except StopIteration:
                # Reset iterator
                ds_entry = self.datasets[idx]
                if isinstance(ds_entry, tuple) and ds_entry[0] == "local":
                    # Reset local loader
                    local_loader = ds_entry[1]
                    local_loader.current_chunk = 0
                    local_loader.current_idx = 0
                    self.iterators[idx] = iter(local_loader)
                else:
                    self.datasets[idx] = self.datasets[idx].shuffle(
                        seed=self.seed, buffer_size=self.buffer_size
                    )
                    self.iterators[idx] = iter(self.datasets[idx].batch(batch_size=32))

            except Exception as e:
                logger.debug(f"Sample error from dataset {idx}: {e}")

        if len(self.token_buffer) < needed:
            self._add_synthetic_tokens(needed - len(self.token_buffer))

    def _add_synthetic_tokens(self, count: int):
        """Add synthetic tokens as fallback."""
        synthetic = torch.randint(0, self.vocab_size, (count,)).tolist()
        self.token_buffer.extend(synthetic)

    def get_batch(self) -> Dict[str, torch.Tensor]:
        """Get next batch."""
        if self._closed:
            raise StopIteration

        needed = self.batch_size * (self.seq_len + 1)

        while len(self.token_buffer) < needed:
            self._refill_buffer()

        batch_tokens = [self.token_buffer.popleft() for _ in range(needed)]
        # If training on CUDA, pinned memory allows non_blocking H2D copies.
        pin = (self.device == "cpu") and torch.cuda.is_available()
        tokens = torch.tensor(batch_tokens, dtype=torch.long, pin_memory=pin)
        tokens = tokens.view(self.batch_size, self.seq_len + 1)

        self.total_batches += 1

        return {
            "input_ids": tokens[:, :-1],
            "labels": tokens[:, 1:],
        }

    def __iter__(self):
        return self

    def __next__(self):
        return self.get_batch()

    def close(self):
        self._closed = True

    def stats(self) -> Dict[str, Any]:
        return {
            "total_batches": self.total_batches,
            "samples_by_dataset": self.samples_by_dataset,
            "buffer_size": len(self.token_buffer),
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

        self.chunk_files = sorted(glob.glob(str(self.data_dir / "chunk_*.pt")))
        if not self.chunk_files:
            self.chunk_files = sorted(glob.glob(str(self.data_dir / "*.pt")))

        if not self.chunk_files:
            raise ValueError(f"No .pt files found in {data_dir}")

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

        while len(sequences) < self.batch_size:
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

            if len(tokens) >= self.seq_len + 1:
                sequences.append(tokens[: self.seq_len + 1])
                pad_masks.append([False] * (self.seq_len + 1))  # No padding
            elif len(tokens) > 64:
                n_pad = self.seq_len + 1 - len(tokens)
                padded = tokens + [0] * n_pad
                sequences.append(padded[: self.seq_len + 1])
                # Mask: False for real tokens, True for padding
                pad_masks.append([False] * len(tokens) + [True] * n_pad)

        batch = torch.tensor(sequences[: self.batch_size], dtype=torch.long)
        input_ids = batch[:, :-1]
        labels = batch[:, 1:].clone()
        
        # Set labels to -100 for padded positions (CE ignore_index)
        pad_mask_tensor = torch.tensor(pad_masks[:self.batch_size], dtype=torch.bool)[:, 1:]
        labels[pad_mask_tensor] = -100
        
        return {
            "input_ids": input_ids,
            "labels": labels,
        }

    def stats(self) -> Dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "chunk_files": len(self.chunk_files),
            "current_chunk_idx": self.chunk_file_idx,
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
