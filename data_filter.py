"""
Fast Data Filtering for HYDRA Training

Provides both offline pre-filtering and online (during training) filtering.

Key strategies:
1. Heuristic filters (fast, CPU-based)
2. Loss-based filtering (skip batches that would destabilize training)
3. Repetition detection
4. Quality scoring
5. Datatrove integration (industry-standard, used for FineWeb)

Usage:
    # Offline: Pre-filter with built-in filters
    python data_filter.py --input fineweb --output fineweb-filtered --workers 8
    
    # Offline: Pre-filter with datatrove (recommended for 1B+ scale)
    python data_filter.py --use-datatrove --input fineweb --output fineweb-hydra
    
    # Online: Integrated into trainer via BatchFilter class
"""

import re
import hashlib
from dataclasses import dataclass
from typing import Optional, List, Tuple, Set, Iterator, Dict, Any
from collections import Counter
import unicodedata
from pathlib import Path


@dataclass
class FilterConfig:
    """Configuration for data filtering."""
    # Length filters
    min_length: int = 50  # Minimum characters
    max_length: int = 100_000  # Maximum characters
    min_words: int = 10  # Minimum word count
    
    # Character quality
    min_alpha_ratio: float = 0.60  # At least 60% alphabetic
    max_special_ratio: float = 0.15  # At most 15% special chars
    max_digit_ratio: float = 0.20  # At most 20% digits
    
    # Repetition filters
    max_char_repetition: int = 10  # "aaaaaaaaaa" = bad
    max_word_repetition: float = 0.30  # 30% same word = bad
    max_line_repetition: float = 0.40  # 40% duplicate lines = bad
    max_ngram_repetition: float = 0.20  # 20% repeated 4-grams = bad
    
    # Loss-based filtering (online)
    loss_spike_threshold: float = 2.5  # Skip if loss > 2.5x running average
    loss_ema_alpha: float = 0.01  # Slow EMA for running average
    max_skips_per_epoch: float = 0.05  # Don't skip more than 5% of data
    
    # Deduplication
    enable_dedup: bool = True
    dedup_ngram_size: int = 13  # For MinHash-style dedup


class TextQualityFilter:
    """Fast heuristic-based text quality filter."""
    
    def __init__(self, config: Optional[FilterConfig] = None):
        self.config = config or FilterConfig()
        
        # Precompile regex patterns for speed
        self._whitespace_re = re.compile(r'\s+')
        self._alpha_re = re.compile(r'[a-zA-Z]')
        self._special_re = re.compile(r'[^a-zA-Z0-9\s\.,!?\'\"-]')
        self._digit_re = re.compile(r'\d')
        self._repeated_char_re = re.compile(r'(.)\1{9,}')  # 10+ repeated chars
        self._url_re = re.compile(r'https?://\S+|www\.\S+')
        self._email_re = re.compile(r'\S+@\S+\.\S+')
        
        # Common boilerplate patterns
        self._boilerplate_patterns = [
            re.compile(r'cookie\s*(policy|consent|settings)', re.I),
            re.compile(r'(subscribe|sign\s*up)\s*(to|for)?\s*(our)?\s*newsletter', re.I),
            re.compile(r'all\s*rights\s*reserved', re.I),
            re.compile(r'terms\s*(of|and)\s*(service|use|conditions)', re.I),
            re.compile(r'privacy\s*policy', re.I),
            re.compile(r'click\s*here\s*to', re.I),
            re.compile(r'share\s*(this|on)\s*(facebook|twitter|linkedin)', re.I),
        ]
    
    def check_text(self, text: str) -> Tuple[bool, str]:
        """
        Check if text passes quality filters.
        
        Returns:
            (passed: bool, reason: str)
        """
        cfg = self.config
        
        # Length check
        if len(text) < cfg.min_length:
            return False, f"too_short:{len(text)}"
        if len(text) > cfg.max_length:
            return False, f"too_long:{len(text)}"
        
        # Word count
        words = text.split()
        if len(words) < cfg.min_words:
            return False, f"few_words:{len(words)}"
        
        # Character ratio checks
        alpha_count = len(self._alpha_re.findall(text))
        special_count = len(self._special_re.findall(text))
        digit_count = len(self._digit_re.findall(text))
        total = len(text)
        
        alpha_ratio = alpha_count / total
        special_ratio = special_count / total
        digit_ratio = digit_count / total
        
        if alpha_ratio < cfg.min_alpha_ratio:
            return False, f"low_alpha:{alpha_ratio:.2f}"
        if special_ratio > cfg.max_special_ratio:
            return False, f"high_special:{special_ratio:.2f}"
        if digit_ratio > cfg.max_digit_ratio:
            return False, f"high_digit:{digit_ratio:.2f}"
        
        # Repeated character check
        if self._repeated_char_re.search(text):
            return False, "char_repetition"
        
        # Word repetition check
        word_counts = Counter(w.lower() for w in words)
        if word_counts:
            most_common_ratio = word_counts.most_common(1)[0][1] / len(words)
            if most_common_ratio > cfg.max_word_repetition:
                return False, f"word_repetition:{most_common_ratio:.2f}"
        
        # Line repetition check
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        if len(lines) > 3:
            line_counts = Counter(lines)
            dup_lines = sum(c - 1 for c in line_counts.values() if c > 1)
            dup_ratio = dup_lines / len(lines)
            if dup_ratio > cfg.max_line_repetition:
                return False, f"line_repetition:{dup_ratio:.2f}"
        
        # N-gram repetition (fast approximation)
        if len(words) > 20:
            ngram_rep = self._check_ngram_repetition(words)
            if ngram_rep > cfg.max_ngram_repetition:
                return False, f"ngram_repetition:{ngram_rep:.2f}"
        
        # Boilerplate check
        boilerplate_count = sum(1 for p in self._boilerplate_patterns if p.search(text))
        if boilerplate_count >= 3:
            return False, f"boilerplate:{boilerplate_count}"
        
        return True, "ok"
    
    def _check_ngram_repetition(self, words: List[str], n: int = 4) -> float:
        """Check for repeated n-grams (fast approximation)."""
        if len(words) < n * 2:
            return 0.0
        
        # Sample every 3rd position for speed
        ngrams = []
        for i in range(0, len(words) - n, 3):
            ngram = ' '.join(words[i:i+n])
            ngrams.append(ngram)
        
        if not ngrams:
            return 0.0
        
        unique = len(set(ngrams))
        return 1.0 - (unique / len(ngrams))
    
    def get_quality_score(self, text: str) -> float:
        """
        Get a quality score from 0-1 (higher = better).
        Useful for soft filtering or ranking.
        """
        passed, reason = self.check_text(text)
        if not passed:
            return 0.0
        
        score = 1.0
        words = text.split()
        
        # Prefer medium-length texts
        if len(text) < 200:
            score *= 0.8
        elif len(text) > 50000:
            score *= 0.9
        
        # Penalize high URL density
        url_count = len(self._url_re.findall(text))
        url_ratio = url_count / max(len(words), 1)
        if url_ratio > 0.1:
            score *= 0.7
        
        # Reward diverse vocabulary
        unique_words = len(set(w.lower() for w in words))
        vocab_diversity = unique_words / len(words) if words else 0
        score *= (0.5 + 0.5 * vocab_diversity)
        
        return min(1.0, max(0.0, score))


class BatchFilter:
    """
    Online batch filtering during training.
    
    Tracks running loss statistics and skips batches that would
    destabilize training (loss spikes from bad data).
    """
    
    def __init__(self, config: Optional[FilterConfig] = None):
        self.config = config or FilterConfig()
        
        # Running statistics
        self.loss_ema = 0.0
        self.loss_ema_sq = 0.0  # For variance
        self.n_samples = 0
        self.n_skipped = 0
        self.n_total = 0
        
        # Track skipped batch info for debugging
        self.skip_history: List[Tuple[int, float, str]] = []
        
        # Text filter for checking individual samples
        self.text_filter = TextQualityFilter(config)
    
    def should_skip_batch(self, loss: float, step: int) -> Tuple[bool, str]:
        """
        Check if a batch should be skipped based on loss.
        
        Call this AFTER computing loss but BEFORE backward pass.
        
        Args:
            loss: The computed loss for this batch
            step: Current training step
            
        Returns:
            (should_skip: bool, reason: str)
        """
        cfg = self.config
        self.n_total += 1
        
        # Initialize EMA with first few samples
        if self.n_samples < 10:
            self.loss_ema = (self.loss_ema * self.n_samples + loss) / (self.n_samples + 1)
            self.n_samples += 1
            return False, "warmup"
        
        # Check skip budget
        skip_ratio = self.n_skipped / max(self.n_total, 1)
        if skip_ratio >= cfg.max_skips_per_epoch:
            # Already skipped too many, must accept
            self._update_ema(loss)
            return False, "budget_exceeded"
        
        # Check for loss spike
        threshold = self.loss_ema * cfg.loss_spike_threshold
        
        if loss > threshold:
            self.n_skipped += 1
            reason = f"loss_spike:{loss:.3f}>{threshold:.3f}"
            self.skip_history.append((step, loss, reason))
            
            # Keep only last 100 skips
            if len(self.skip_history) > 100:
                self.skip_history = self.skip_history[-100:]
            
            return True, reason
        
        # Update EMA and accept batch
        self._update_ema(loss)
        return False, "ok"
    
    def _update_ema(self, loss: float) -> None:
        """Update exponential moving average of loss."""
        alpha = self.config.loss_ema_alpha
        self.loss_ema = alpha * loss + (1 - alpha) * self.loss_ema
        self.n_samples += 1
    
    def get_stats(self) -> dict:
        """Get filtering statistics."""
        return {
            "loss_ema": self.loss_ema,
            "n_samples": self.n_samples,
            "n_skipped": self.n_skipped,
            "n_total": self.n_total,
            "skip_ratio": self.n_skipped / max(self.n_total, 1),
            "recent_skips": self.skip_history[-10:] if self.skip_history else [],
        }
    
    def reset_epoch(self) -> None:
        """Reset per-epoch counters (keep EMA)."""
        self.n_skipped = 0
        self.n_total = 0


class DedupFilter:
    """
    Fast approximate deduplication using MinHash-style fingerprints.
    
    Use during offline preprocessing, not online (too slow).
    """
    
    def __init__(self, ngram_size: int = 13):
        self.ngram_size = ngram_size
        self.seen_hashes: Set[str] = set()
    
    def get_fingerprint(self, text: str) -> str:
        """Get a fingerprint for near-duplicate detection."""
        # Normalize
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        words = text.split()
        
        if len(words) < self.ngram_size:
            # For short texts, hash the whole thing
            return hashlib.md5(text.encode()).hexdigest()[:16]
        
        # Get n-grams and hash them
        ngrams = []
        for i in range(0, len(words) - self.ngram_size + 1, 3):  # Sample every 3rd
            ngram = ' '.join(words[i:i + self.ngram_size])
            h = hashlib.md5(ngram.encode()).hexdigest()[:8]
            ngrams.append(h)
        
        # Use min hash as fingerprint
        if ngrams:
            return min(ngrams)
        return hashlib.md5(text.encode()).hexdigest()[:16]
    
    def is_duplicate(self, text: str) -> bool:
        """Check if text is a near-duplicate of something we've seen."""
        fp = self.get_fingerprint(text)
        if fp in self.seen_hashes:
            return True
        self.seen_hashes.add(fp)
        return False
    
    def clear(self) -> None:
        """Clear seen hashes."""
        self.seen_hashes.clear()


def filter_dataset_streaming(
    input_dataset,
    output_path: str,
    config: Optional[FilterConfig] = None,
    max_samples: Optional[int] = None,
    verbose: bool = True,
) -> dict:
    """
    Filter a HuggingFace dataset in streaming mode.
    
    Args:
        input_dataset: HF dataset (streaming or regular)
        output_path: Where to save filtered data (jsonl)
        config: Filter configuration
        max_samples: Maximum samples to process
        verbose: Print progress
        
    Returns:
        Statistics dict
    """
    import json
    from tqdm import tqdm
    
    cfg = config or FilterConfig()
    text_filter = TextQualityFilter(cfg)
    dedup = DedupFilter(cfg.dedup_ngram_size) if cfg.enable_dedup else None
    
    stats = {
        "total": 0,
        "passed": 0,
        "rejected": Counter(),
        "duplicates": 0,
    }
    
    with open(output_path, 'w') as f:
        iterator = tqdm(input_dataset) if verbose else input_dataset
        
        for i, sample in enumerate(iterator):
            if max_samples and i >= max_samples:
                break
            
            stats["total"] += 1
            
            # Get text (handle different dataset formats)
            text = sample.get('text') or sample.get('content') or str(sample)
            
            # Quality filter
            passed, reason = text_filter.check_text(text)
            if not passed:
                stats["rejected"][reason] += 1
                continue
            
            # Dedup filter
            if dedup and dedup.is_duplicate(text):
                stats["duplicates"] += 1
                continue
            
            # Passed all filters
            stats["passed"] += 1
            f.write(json.dumps({"text": text}) + '\n')
    
    # Summary
    if verbose:
        print(f"\n=== Filtering Complete ===")
        print(f"Total processed: {stats['total']:,}")
        print(f"Passed: {stats['passed']:,} ({stats['passed']/stats['total']*100:.1f}%)")
        print(f"Duplicates removed: {stats['duplicates']:,}")
        print(f"\nRejection reasons:")
        for reason, count in stats["rejected"].most_common(10):
            print(f"  {reason}: {count:,}")
    
    return stats


# ============================================
# Datatrove Integration (Industry Standard)
# ============================================

def check_datatrove_available() -> bool:
    """Check if datatrove is installed."""
    try:
        import datatrove
        return True
    except ImportError:
        return False


def create_datatrove_pipeline(
    input_path: str,
    output_path: str,
    config: Optional[FilterConfig] = None,
    num_workers: int = 8,
    tasks_per_worker: int = 10,
) -> "Pipeline":
    """
    Create a datatrove pipeline for high-quality data filtering.
    
    This uses the same filters that produced FineWeb/FineWeb-Edu.
    
    Args:
        input_path: HuggingFace dataset name or local path
        output_path: Where to save filtered data
        config: Filter configuration
        num_workers: Number of parallel workers
        tasks_per_worker: Tasks per worker for load balancing
        
    Returns:
        Configured datatrove Pipeline
    """
    from datatrove.pipeline.readers import HuggingFaceDatasetReader
    from datatrove.pipeline.writers import JsonlWriter
    from datatrove.pipeline.filters import (
        GopherQualityFilter,
        GopherRepetitionFilter,
        LanguageFilter,
        URLFilter,
        C4QualityFilter,
    )
    from datatrove.pipeline.dedup import MinhashDedupSignature, MinhashDedupCluster, MinhashDedupFilter
    from datatrove.executor import LocalPipelineExecutor
    from datatrove.pipeline.base import Pipeline
    
    cfg = config or FilterConfig()
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Stage 1: Quality filtering
    quality_pipeline = Pipeline([
        # Read from HuggingFace
        HuggingFaceDatasetReader(
            dataset=input_path,
            streaming=True,
            text_key="text",
        ),
        
        # Language filter (English only by default)
        LanguageFilter(
            languages=["en"],
            language_threshold=0.65,
        ),
        
        # Gopher quality filter (used by DeepMind)
        # - Word count, mean word length, symbol ratios
        # - Stop word presence, ellipsis limit
        GopherQualityFilter(
            min_doc_words=cfg.min_words,
            max_doc_words=100_000,
            min_avg_word_length=3,
            max_avg_word_length=10,
            max_symbol_word_ratio=cfg.max_special_ratio,
            max_bullet_lines_ratio=0.90,
            max_ellipsis_lines_ratio=0.30,
            max_non_alpha_words_ratio=1.0 - cfg.min_alpha_ratio,
            min_stop_words=2,
        ),
        
        # Gopher repetition filter
        # - Character/word/line repetition
        # - N-gram repetition
        GopherRepetitionFilter(
            top_n_grams=[2, 3, 4],
            top_n_grams_thresholds=[0.20, 0.18, 0.16],
            dup_line_frac=cfg.max_line_repetition,
            dup_para_frac=0.30,
            dup_line_char_frac=0.40,
            dup_para_char_frac=0.40,
        ),
        
        # C4 quality filter (additional checks)
        C4QualityFilter(
            filter_no_terminal_punct=False,  # Don't require period at end
            min_num_sentences=1,
            min_words_per_line=0,
            max_word_length=1000,
            filter_lorem_ipsum=True,
            filter_javascript=True,
            filter_curly_bracket=True,
            filter_policy=True,  # Remove privacy policies, ToS
        ),
        
        # URL filter (remove known bad domains)
        URLFilter(),
        
        # Write filtered output
        JsonlWriter(
            output_folder=str(output_dir / "filtered"),
            output_filename="data.jsonl",
        ),
    ])
    
    return quality_pipeline


def create_dedup_pipeline(
    input_path: str,
    output_path: str,
    num_workers: int = 8,
) -> List["Pipeline"]:
    """
    Create datatrove deduplication pipelines (3 stages).
    
    MinHash deduplication is done in 3 stages:
    1. Signature computation
    2. Clustering 
    3. Filtering
    
    Args:
        input_path: Path to filtered data (from quality pipeline)
        output_path: Final output path
        num_workers: Number of workers
        
    Returns:
        List of 3 Pipeline objects to run sequentially
    """
    from datatrove.pipeline.readers import JsonlReader
    from datatrove.pipeline.writers import JsonlWriter
    from datatrove.pipeline.dedup import (
        MinhashDedupSignature,
        MinhashDedupCluster,
        MinhashDedupFilter,
        MinhashConfig,
    )
    from datatrove.pipeline.base import Pipeline
    
    output_dir = Path(output_path)
    sig_dir = output_dir / "minhash_sigs"
    cluster_dir = output_dir / "minhash_clusters"
    
    # MinHash config
    minhash_config = MinhashConfig(
        num_buckets=14,
        hashes_per_bucket=8,
        n_grams=5,
    )
    
    # Stage 1: Compute signatures
    sig_pipeline = Pipeline([
        JsonlReader(str(output_dir / "filtered")),
        MinhashDedupSignature(
            output_folder=str(sig_dir),
            config=minhash_config,
        ),
    ])
    
    # Stage 2: Cluster similar documents
    cluster_pipeline = Pipeline([
        MinhashDedupCluster(
            input_folder=str(sig_dir),
            output_folder=str(cluster_dir),
            config=minhash_config,
        ),
    ])
    
    # Stage 3: Filter duplicates
    filter_pipeline = Pipeline([
        JsonlReader(str(output_dir / "filtered")),
        MinhashDedupFilter(
            input_folder=str(cluster_dir),
            exclusion_writer=JsonlWriter(str(output_dir / "removed_duplicates")),
        ),
        JsonlWriter(
            output_folder=str(output_dir / "final"),
            output_filename="data.jsonl",
        ),
    ])
    
    return [sig_pipeline, cluster_pipeline, filter_pipeline]


def run_datatrove_pipeline(
    input_path: str,
    output_path: str,
    config: Optional[FilterConfig] = None,
    num_workers: int = 8,
    skip_dedup: bool = False,
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run full datatrove filtering pipeline.
    
    Args:
        input_path: HuggingFace dataset name (e.g., "HuggingFaceFW/fineweb")
        output_path: Output directory
        config: Filter configuration
        num_workers: Number of parallel workers
        skip_dedup: Skip deduplication (faster, but may have duplicates)
        max_samples: Maximum samples to process (for testing)
        
    Returns:
        Statistics dict
    """
    from datatrove.executor import LocalPipelineExecutor
    
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("HYDRA Data Pipeline (powered by datatrove)")
    print("=" * 60)
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Workers: {num_workers}")
    print(f"Dedup: {'Enabled' if not skip_dedup else 'Disabled'}")
    print("=" * 60)
    
    # Stage 1: Quality filtering
    print("\n📊 Stage 1: Quality Filtering...")
    quality_pipeline = create_datatrove_pipeline(
        input_path, output_path, config, num_workers
    )
    
    executor = LocalPipelineExecutor(
        pipeline=quality_pipeline,
        workers=num_workers,
        tasks=num_workers * 10,
    )
    executor.run()
    
    stats = {"quality_filtered": True}
    
    # Stage 2-4: Deduplication (optional)
    if not skip_dedup:
        print("\n🔄 Stage 2-4: MinHash Deduplication...")
        dedup_pipelines = create_dedup_pipeline(
            str(output_dir / "filtered"),
            output_path,
            num_workers,
        )
        
        for i, pipeline in enumerate(dedup_pipelines, 2):
            print(f"  Running dedup stage {i-1}/3...")
            executor = LocalPipelineExecutor(
                pipeline=pipeline,
                workers=num_workers if i < 4 else 1,  # Clustering is single-threaded
                tasks=num_workers * 10 if i < 4 else 1,
            )
            executor.run()
        
        stats["deduplicated"] = True
        final_path = output_dir / "final"
    else:
        final_path = output_dir / "filtered"
    
    print("\n✅ Pipeline complete!")
    print(f"Output: {final_path}")
    
    return stats


def create_perplexity_filter_pipeline(
    input_path: str,
    output_path: str,
    model_name: str = "kenlm",  # or path to a KenLM model
    percentile_threshold: float = 0.30,  # Keep top 30% by perplexity
    num_workers: int = 8,
) -> "Pipeline":
    """
    Create a perplexity-based quality filter.
    
    This filters data based on how "surprised" a language model is.
    Lower perplexity = more natural/common language patterns.
    
    Used by FineWeb-Edu and other high-quality datasets.
    
    Args:
        input_path: Input data path
        output_path: Output path
        model_name: KenLM model path or "kenlm" for default
        percentile_threshold: Keep documents below this perplexity percentile
        num_workers: Number of workers
        
    Returns:
        Pipeline object
    """
    try:
        from datatrove.pipeline.readers import JsonlReader
        from datatrove.pipeline.writers import JsonlWriter
        from datatrove.pipeline.filters import PerplexityFilter
        from datatrove.pipeline.base import Pipeline
    except ImportError:
        raise ImportError(
            "Perplexity filtering requires datatrove with kenlm support. "
            "Install with: pip install datatrove[kenlm]"
        )
    
    return Pipeline([
        JsonlReader(input_path),
        PerplexityFilter(
            model_path=model_name,
            max_perplexity=percentile_threshold,  # Will be computed as percentile
        ),
        JsonlWriter(
            output_folder=output_path,
            output_filename="data.jsonl",
        ),
    ])


# ============================================
# Educational Quality Filter (FineWeb-Edu style)
# ============================================

def create_educational_filter(
    input_path: str,
    output_path: str,
    min_score: float = 3.0,  # FineWeb-Edu uses 3.0 threshold
    num_workers: int = 8,
) -> None:
    """
    Filter for educational content quality (FineWeb-Edu style).
    
    This uses a classifier trained to score content on educational value.
    Scores range from 0-5, with 3+ being "educational quality".
    
    Note: Requires the HuggingFace classifier model.
    
    Args:
        input_path: Input data path  
        output_path: Output path
        min_score: Minimum educational score (0-5, default 3.0)
        num_workers: Number of workers
    """
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch
    except ImportError:
        raise ImportError("Educational filter requires transformers: pip install transformers")
    
    print("Loading educational quality classifier...")
    model_name = "HuggingFaceFW/fineweb-edu-classifier"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    def score_text(text: str) -> float:
        """Score text for educational quality (0-5)."""
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512
        )
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            score = outputs.logits.squeeze().item()
        
        return score
    
    # Process dataset
    import json
    from tqdm import tqdm
    
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    stats = {"total": 0, "passed": 0, "scores": []}
    
    with open(output_dir / "data.jsonl", "w") as out_f:
        # Read input
        input_files = list(Path(input_path).glob("**/*.jsonl"))
        
        for input_file in input_files:
            with open(input_file) as in_f:
                for line in tqdm(in_f, desc=f"Scoring {input_file.name}"):
                    sample = json.loads(line)
                    text = sample.get("text", "")
                    
                    stats["total"] += 1
                    score = score_text(text[:2000])  # Score first 2K chars
                    stats["scores"].append(score)
                    
                    if score >= min_score:
                        stats["passed"] += 1
                        sample["edu_score"] = score
                        out_f.write(json.dumps(sample) + "\n")
    
    # Print stats
    import numpy as np
    scores = np.array(stats["scores"])
    print(f"\n=== Educational Filter Stats ===")
    print(f"Total: {stats['total']:,}")
    print(f"Passed (score >= {min_score}): {stats['passed']:,} ({stats['passed']/stats['total']*100:.1f}%)")
    print(f"Score distribution: mean={scores.mean():.2f}, std={scores.std():.2f}")
    print(f"Percentiles: 25th={np.percentile(scores, 25):.2f}, 50th={np.percentile(scores, 50):.2f}, 75th={np.percentile(scores, 75):.2f}")


# ============================================
# CLI for offline filtering
# ============================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Filter dataset for quality (HYDRA data pipeline)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic filtering with built-in filters
    python data_filter.py -i fineweb -o filtered.jsonl
    
    # Use datatrove (recommended for scale)
    python data_filter.py --use-datatrove -i HuggingFaceFW/fineweb -o ./data/fineweb-hydra
    
    # Datatrove without deduplication (faster)
    python data_filter.py --use-datatrove --skip-dedup -i fineweb -o ./data/fineweb-quick
    
    # Educational quality filter (FineWeb-Edu style)
    python data_filter.py --edu-filter -i ./data/fineweb-hydra/final -o ./data/fineweb-edu
    
    # Full pipeline: quality + dedup + educational
    python data_filter.py --full-pipeline -i HuggingFaceFW/fineweb -o ./data/fineweb-hydra-edu
        """
    )
    
    # Input/output
    parser.add_argument("--input", "-i", required=True, help="Input dataset name or path")
    parser.add_argument("--output", "-o", required=True, help="Output path")
    parser.add_argument("--split", default="train", help="Dataset split (default: train)")
    parser.add_argument("--max-samples", "-n", type=int, help="Max samples to process")
    
    # Processing mode
    parser.add_argument("--use-datatrove", action="store_true", 
                        help="Use datatrove for filtering (recommended for 1B+ scale)")
    parser.add_argument("--streaming", action="store_true", 
                        help="Use streaming mode (for built-in filters)")
    parser.add_argument("--workers", "-w", type=int, default=8,
                        help="Number of parallel workers (default: 8)")
    
    # Datatrove options
    parser.add_argument("--skip-dedup", action="store_true",
                        help="Skip deduplication stage (faster)")
    parser.add_argument("--edu-filter", action="store_true",
                        help="Apply educational quality filter")
    parser.add_argument("--edu-threshold", type=float, default=3.0,
                        help="Educational score threshold (default: 3.0)")
    parser.add_argument("--full-pipeline", action="store_true",
                        help="Run full pipeline: quality + dedup + edu filter")
    
    # Filter parameters
    parser.add_argument("--min-length", type=int, default=50)
    parser.add_argument("--min-words", type=int, default=10)
    parser.add_argument("--min-alpha", type=float, default=0.60)
    parser.add_argument("--no-dedup", action="store_true", help="Disable deduplication")
    
    args = parser.parse_args()
    
    # Configure filter
    config = FilterConfig(
        min_length=args.min_length,
        min_words=args.min_words,
        min_alpha_ratio=args.min_alpha,
        enable_dedup=not args.no_dedup,
    )
    
    # Datatrove mode
    if args.use_datatrove or args.full_pipeline:
        if not check_datatrove_available():
            print("ERROR: datatrove not installed. Install with:")
            print("  pip install datatrove")
            print("\nFor full features (perplexity, dedup):")
            print("  pip install 'datatrove[all]'")
            exit(1)
        
        # Run datatrove pipeline
        stats = run_datatrove_pipeline(
            args.input,
            args.output,
            config=config,
            num_workers=args.workers,
            skip_dedup=args.skip_dedup,
            max_samples=args.max_samples,
        )
        
        # Educational filter (optional)
        if args.edu_filter or args.full_pipeline:
            final_dir = Path(args.output) / ("final" if not args.skip_dedup else "filtered")
            edu_output = Path(args.output) / "edu_filtered"
            create_educational_filter(
                str(final_dir),
                str(edu_output),
                min_score=args.edu_threshold,
                num_workers=args.workers,
            )
            print(f"\n✅ Educational filtered data: {edu_output}")
    
    # Educational filter only mode
    elif args.edu_filter:
        create_educational_filter(
            args.input,
            args.output,
            min_score=args.edu_threshold,
            num_workers=args.workers,
        )
    
    # Built-in filters mode
    else:
        from datasets import load_dataset
        
        print(f"Loading dataset: {args.input}")
        ds = load_dataset(args.input, split=args.split, streaming=args.streaming)
        
        stats = filter_dataset_streaming(
            ds, 
            args.output,
            config=config,
            max_samples=args.max_samples,
        )
        
        print(f"\nFiltered data saved to: {args.output}")
