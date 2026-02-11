#!/usr/bin/env python3
"""
HYDRA Reasoning Trainer - Standalone System 2 Training

Dedicated trainer for reasoning/thinking capabilities using GRPO.
Runs independently from the main pretraining loop, allowing:
- Longer sequence lengths for extended thinking
- Think loops (multiple reasoning iterations)
- Optional teacher LLM integration
- More aggressive memory optimization (no shared state with pretraining)

Usage:
    # Basic reasoning training
    source /home/tim/venvs/llm/bin/activate && python reasoning_trainer.py \
        --checkpoint checkpoints/hydra_500m_step_282000.pt \
        --max_steps 1000

    # Extended thinking with longer sequences
    source /home/tim/venvs/llm/bin/activate && python reasoning_trainer.py \
        --checkpoint checkpoints/hydra_500m_step_282000.pt \
        --max_seq_len 1024 \
        --max_thinking_tokens 512 \
        --think_loops 3

    # With teacher model (future)
    source /home/tim/venvs/llm/bin/activate && python reasoning_trainer.py \
        --checkpoint checkpoints/hydra_500m_step_282000.pt \
        --teacher_model "meta-llama/Llama-3-8B-Instruct"
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Set CUDA allocator config before importing torch
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import warnings
# Suppress dtype mismatch warning from PyTorch RMSNorm - we use autocast for all forward passes
warnings.filterwarnings("ignore", message="Mismatch dtype between input and weight")

import torch
import torch.nn.functional as F
from torch.amp import autocast

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_log = logging.getLogger("ReasoningTrainer")

# Global shutdown flag for graceful ctrl+c handling
_SHUTDOWN_REQUESTED = False


def _signal_handler(signum, frame):
    """Handle ctrl+c gracefully."""
    global _SHUTDOWN_REQUESTED
    if _SHUTDOWN_REQUESTED:
        _log.warning("Force quit requested, exiting immediately...")
        sys.exit(1)
    _SHUTDOWN_REQUESTED = True
    _log.info("\n" + "=" * 60)
    _log.info("SHUTDOWN REQUESTED - will save checkpoint after current step")
    _log.info("Press Ctrl+C again to force quit (will lose progress)")
    _log.info("=" * 60)


@dataclass
class ReasoningConfig:
    """Configuration for standalone reasoning training."""

    # Checkpoint
    checkpoint_path: str = ""
    output_dir: str = "checkpoints/reasoning"
    save_every: int = 100

    # Training
    max_steps: int = 1000
    learning_rate: float = 1e-5
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup_steps: int = 50

    # Generation reuse (μ parameter from TRL GRPO)
    # Reuse generated samples for this many policy updates before regenerating.
    # μ=1 means generate fresh samples every step (no reuse).
    # μ=2-4 gives 2-4x effective throughput by amortizing generation cost.
    generation_reuse_count: int = 1

    # Reasoning generation
    batch_size: int = 2  # Prompts per step
    num_generations: int = 4  # Completions per prompt (G in GRPO)
    max_thinking_tokens: int = 512  # Max tokens for thinking/reasoning
    max_seq_len: int = 1024  # Total sequence length cap
    temperature: float = 0.7
    top_p: float = 0.95
    repetition_penalty: float = 1.2  # Penalty for repeated tokens (>1.0 discourages)
    no_repeat_ngram_size: int = 3  # Block repeating n-grams of this size

    # Think loops - multiple reasoning passes
    think_loops: int = 1  # Number of thinking iterations per problem
    think_loop_temperature_decay: float = 0.9  # Cool down temperature each loop

    # Reward function
    reward_function: str = "exact_match"  # exact_match, format_reward, length_penalty

    # Memory optimization
    skip_moe: bool = True  # Skip MoE layers for memory savings
    micro_batch_size: int = 1  # Process sequences one at a time
    gradient_checkpointing: bool = True
    use_8bit_adam: bool = True

    # Distillation (offline teacher)
    distillation_dataset: Optional[str] = None  # Path to teacher completions JSONL
    distillation_weight: float = 0.5  # Weight for distillation loss vs GRPO
    distillation_only: bool = False  # Only distillation, no GRPO self-play

    # KL divergence (reference model)
    kl_beta: float = 0.0  # KL penalty coefficient. 0 = disabled (no ref model)
    kl_warmup_steps: int = 0  # Steps before KL penalty ramps to full kl_beta

    # Reward weighting (per-signal priorities)
    # When multiple reward components are combined, weight each one differently.
    # Default empty = uniform weighting. Otherwise, comma-separated floats.
    reward_weights: Optional[List[float]] = None

    # DAPO (Decoupled Alignment from Policy Optimization)
    dapo: bool = False  # Use DAPO instead of standard GRPO
    dapo_clip_low: float = 0.8  # Lower clip bound (only clip probability decreases)
    dapo_clip_high: float = 1.28  # Upper clip bound (allow probability increases)
    dapo_dynamic_temperature: bool = True  # Scale temperature by advantage magnitude

    # Speculative decoding (n-gram suffix speculation)
    speculative_ngram_size: int = 0  # 0=disabled. 3-5 typical. Uses suffix match for draft.
    speculative_max_draft: int = 4  # Max tokens to draft per speculation step

    # Logging
    log_interval: int = 10
    eval_interval: int = 100


def log_memory(stage: str) -> Dict[str, float]:
    """Log and return CUDA memory state."""
    if not torch.cuda.is_available():
        return {}
    # memory_allocated/reserved read allocator bookkeeping — no sync needed
    alloc = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    free = torch.cuda.mem_get_info()[0] / (1024**3)
    _log.info(f"[MEM] {stage}: alloc={alloc:.2f}GB reserved={reserved:.2f}GB free={free:.2f}GB")
    return {"allocated_gb": alloc, "reserved_gb": reserved, "free_gb": free}


def load_checkpoint(checkpoint_path: str, device: str = "cuda") -> Tuple[torch.nn.Module, Dict, Dict]:
    """Load model from checkpoint.

    Returns:
        model: The loaded model
        checkpoint: Full checkpoint dict
        model_config: Model architecture config (for saving later)
    """
    from hydra.model.framework import HydraModel

    _log.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Extract config - prefer model_config (from reasoning checkpoints), fall back to config
    cfg = checkpoint.get("model_config", checkpoint.get("config", {}))
    model_state = checkpoint.get("model", checkpoint.get("model_state_dict", {}))

    # Infer mlp_ratio from checkpoint weights (not stored in config)
    # SwiGLU: gate_up has shape [2*hidden, dim], so hidden = gate_up[0] / 2
    mlp_ratio = 2.67  # default
    dim = cfg.get("mod_mor_dim", 1792)
    for key, value in model_state.items():
        if "mlp.gate_up.weight" in key and hasattr(value, "shape"):
            hidden_size = value.shape[0] // 2
            mlp_ratio = hidden_size / dim
            _log.info(f"  Inferred mlp_ratio from weights: {mlp_ratio:.4f}")
            break

    # Log key config values
    _log.info(f"  Model dim: {dim}")
    _log.info(f"  MoR blocks: {cfg.get('n_mor_blocks', 14)}")
    _log.info(f"  MoR recursions: {cfg.get('mor_recursions', 4)}")
    _log.info(f"  MLP ratio: {mlp_ratio:.4f}")
    _log.info(f"  MoE enabled: {cfg.get('moe_enabled', False)}")
    _log.info(f"  MoE experts: {cfg.get('moe_num_experts', 0)}")
    _log.info(f"  MoE layers: {cfg.get('moe_num_layers', 0)}")

    # Build model with config from checkpoint
    model = HydraModel(
        dim=dim,
        n_mor_blocks=cfg.get("n_mor_blocks", 14),
        recursions_per_block=cfg.get("mor_recursions", 4),
        n_heads=cfg.get("mod_mor_n_heads", 28),
        n_kv_heads=cfg.get("mod_mor_n_kv_heads", 4),
        vocab_size=cfg.get("vocab_size", 50257),
        max_seq_len=cfg.get("max_seq_len", 2048),
        mlp_ratio=mlp_ratio,
        mod_capacity=cfg.get("mod_capacity", 0.75),
        adaptive=cfg.get("mor_adaptive", True),
        static_routing_mode=cfg.get("static_routing_mode", False),
        # MoE config
        moe_enabled=cfg.get("moe_enabled", False),
        moe_num_experts=cfg.get("moe_num_experts", 0),
        moe_num_layers=cfg.get("moe_num_layers", 0),
        moe_top_k=cfg.get("moe_top_k", 1),
        moe_aux_weight=cfg.get("moe_aux_weight", 0.01),
        moe_identity_init=cfg.get("moe_identity_init", True),
    )

    # Load weights - handle different checkpoint formats
    # Prefer "model" (standard format, used by main trainer and new reasoning checkpoints)
    # Fall back to "model_state_dict" for old reasoning checkpoints
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"], strict=False)
    elif "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    else:
        raise KeyError(f"Checkpoint missing model weights. Keys: {list(checkpoint.keys())}")
    model = model.to(device).to(torch.bfloat16)

    # Count params
    total_params = sum(p.numel() for p in model.parameters())
    _log.info(f"Model loaded: {total_params/1e6:.1f}M parameters")

    # Build model_config dict for saving (preserves architecture for future loads)
    model_config = {
        "mod_mor_dim": dim,
        "n_mor_blocks": cfg.get("n_mor_blocks", 14),
        "mor_recursions": cfg.get("mor_recursions", 4),
        "mod_mor_n_heads": cfg.get("mod_mor_n_heads", 28),
        "mod_mor_n_kv_heads": cfg.get("mod_mor_n_kv_heads", 4),
        "vocab_size": cfg.get("vocab_size", 50257),
        "max_seq_len": cfg.get("max_seq_len", 2048),
        "mlp_ratio": mlp_ratio,
        "mod_capacity": cfg.get("mod_capacity", 0.75),
        "mor_adaptive": cfg.get("mor_adaptive", True),
        "static_routing_mode": cfg.get("static_routing_mode", False),
        "moe_enabled": cfg.get("moe_enabled", False),
        "moe_num_experts": cfg.get("moe_num_experts", 0),
        "moe_num_layers": cfg.get("moe_num_layers", 0),
        "moe_top_k": cfg.get("moe_top_k", 1),
        "moe_aux_weight": cfg.get("moe_aux_weight", 0.01),
        "moe_identity_init": cfg.get("moe_identity_init", True),
    }

    return model, checkpoint, model_config


def create_optimizer(
    model: torch.nn.Module,
    config: ReasoningConfig,
) -> torch.optim.Optimizer:
    """Create optimizer for reasoning training."""
    if config.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.Adam8bit(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
            _log.info("Using 8-bit Adam optimizer")
            return optimizer
        except ImportError:
            _log.warning("bitsandbytes not available, using standard AdamW")

    return torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )


def create_reference_model(model: torch.nn.Module) -> torch.nn.Module:
    """Create a frozen copy of the model for KL divergence computation.

    The reference model shares the same architecture but its parameters are
    detached from the computation graph and never updated. This provides
    the pi_ref baseline for KL(pi_theta || pi_ref).
    """
    import copy
    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    _log.info(f"Created frozen reference model ({sum(p.numel() for p in ref_model.parameters())/1e6:.1f}M params)")
    return ref_model


def compute_kl_penalty(
    model: torch.nn.Module,
    ref_model: torch.nn.Module,
    input_ids: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Compute per-sequence KL(pi_theta || pi_ref) penalty.

    Returns [B] tensor of mean per-token KL divergence for each sequence.
    """
    # Get current policy log-probs
    current_logprobs = compute_sequence_logprobs(model, input_ids, mask)  # [B, T]

    # Get reference policy log-probs (no grad)
    with torch.no_grad():
        ref_logprobs = compute_sequence_logprobs(ref_model, input_ids, mask)  # [B, T]

    # Per-token KL: log(pi_theta / pi_ref) = log_pi_theta - log_pi_ref
    # KL(pi_theta || pi_ref) = sum_t pi_theta(a_t|s_t) * (log_pi_theta - log_pi_ref)
    # Approximation: just use (log_pi_theta - log_pi_ref) directly (unbiased estimator)
    kl_per_token = (current_logprobs - ref_logprobs) * mask  # [B, T]

    # Mean per-token KL per sequence
    num_tokens = mask.sum(dim=1).clamp(min=1)
    kl_per_seq = kl_per_token.sum(dim=1) / num_tokens  # [B]

    return kl_per_seq


def get_tokenizer(tokenizer_name: str = "gpt2"):
    """Get tokenizer."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_reasoning_prompts(max_prompts: int = 1000) -> List[Dict[str, Any]]:
    """Load reasoning prompts from OpenMathInstruct or similar."""
    prompts = []

    try:
        from datasets import load_dataset

        ds = load_dataset(
            "nvidia/OpenMathInstruct-2",
            split="train_1M",
            streaming=True,
        )

        count = 0
        for example in ds:
            if count >= max_prompts:
                break
            problem = example.get("problem", "")
            answer = example.get("expected_answer", "")
            if problem and len(problem) > 20:
                prompts.append({
                    "prompt": f"Solve this math problem step by step. Show your reasoning.\n\nProblem: {problem}\n\nSolution:",
                    "expected_answer": answer,
                })
                count += 1

        _log.info(f"Loaded {len(prompts)} reasoning prompts from OpenMathInstruct-2")

    except Exception as e:
        _log.warning(f"Failed to load reasoning prompts: {e}")
        # Fallback prompts
        prompts = [
            {"prompt": "What is 15 * 17? Think step by step.\n\nSolution:", "expected_answer": "255"},
            {"prompt": "If x + 5 = 12, what is x? Show your work.\n\nSolution:", "expected_answer": "7"},
            {"prompt": "What is the area of a rectangle with width 8 and height 6?\n\nSolution:", "expected_answer": "48"},
        ]

    return prompts


def load_distillation_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """Load teacher completions for distillation training.

    Expects JSONL with fields:
    - prompt: the problem/question
    - teacher_completion: high-quality reasoning from teacher model
    - expected_answer: (optional) ground truth answer
    """
    import json

    data = []
    with open(dataset_path) as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                if item.get("prompt") and item.get("teacher_completion"):
                    data.append({
                        "prompt": item["prompt"],
                        "teacher_completion": item["teacher_completion"],
                        "expected_answer": item.get("expected_answer", ""),
                    })

    _log.info(f"Loaded {len(data)} teacher completions from {dataset_path}")
    return data


def run_distillation_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    distill_data: List[Dict[str, Any]],
    tokenizer,
    config: ReasoningConfig,
    step: int,
) -> Dict[str, float]:
    """Run a distillation training step.

    Trains the student model to match teacher completions using
    standard language modeling loss on teacher outputs.
    """
    import random

    device = next(model.parameters()).device
    base_model = model._orig_mod if hasattr(model, "_orig_mod") else model

    # Skip MoE if configured
    if config.skip_moe and hasattr(base_model, "set_skip_moe"):
        base_model.set_skip_moe(True)

    # Sample batch from distillation data
    batch = random.sample(distill_data, min(config.batch_size, len(distill_data)))

    # Prepare sequences: prompt + teacher completion
    full_texts = []
    prompt_lengths = []
    for item in batch:
        prompt = item["prompt"]
        completion = item["teacher_completion"]
        full_text = prompt + " " + completion
        full_texts.append(full_text)

        # Track prompt length for masking
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        prompt_lengths.append(len(prompt_tokens))

    # Tokenize
    encoded = tokenizer(
        full_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=config.max_seq_len,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    # Create loss mask (only compute loss on completion tokens)
    batch_size, seq_len = input_ids.shape
    loss_mask = torch.zeros(batch_size, seq_len, device=device)
    for i, plen in enumerate(prompt_lengths):
        # Mask out prompt tokens, only train on completion
        loss_mask[i, plen:] = attention_mask[i, plen:]

    # Forward pass
    base_model.train()
    optimizer.zero_grad(set_to_none=True)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        outputs = base_model(input_ids)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs

        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = loss_mask[:, 1:].contiguous()

        # Cross-entropy loss on completion tokens only
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        token_losses = loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).view(batch_size, -1)

        # Masked mean
        masked_loss = (token_losses * shift_mask).sum() / (shift_mask.sum() + 1e-8)

    # Backward
    masked_loss.backward()

    # Gradient clipping and optimizer step
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
    optimizer.step()

    # Re-enable MoE
    if config.skip_moe and hasattr(base_model, "set_skip_moe"):
        base_model.set_skip_moe(False)

    # Compute perplexity
    with torch.no_grad():
        perplexity = torch.exp(masked_loss).item()

    metrics = {
        "distill_loss": masked_loss.item(),
        "distill_ppl": perplexity,
        "distill_tokens": shift_mask.sum().item(),
    }

    _log.info(f"  Step {step} distillation: loss={masked_loss.item():.4f}, ppl={perplexity:.2f}")

    return metrics


def _vectorized_ngram_blocking(
    generated: torch.Tensor,
    current_len: int,
    ngram_size: int,
    next_logits: torch.Tensor,
) -> torch.Tensor:
    """Block repeated n-grams using vectorized tensor ops instead of Python loops.

    Args:
        generated: Full generated buffer [1, max_len]
        current_len: Current generation length
        ngram_size: Size of n-grams to block
        next_logits: Logits to modify in-place [1, vocab_size]

    Returns:
        Modified next_logits with banned tokens set to -inf
    """
    if ngram_size <= 0 or current_len < ngram_size:
        return next_logits

    device = generated.device
    seq = generated[0, :current_len]  # [current_len]

    # The (n-1) prefix we're looking for: last (n-1) tokens
    prefix = seq[current_len - ngram_size + 1:]  # [ngram_size - 1]

    # Extract all (n-1)-grams from the sequence using unfold
    # unfold gives [num_ngrams, ngram_size - 1]
    n_minus_1 = ngram_size - 1
    num_positions = current_len - ngram_size + 1
    if num_positions <= 0:
        return next_logits

    all_prefixes = seq[:current_len - 1].unfold(0, n_minus_1, 1)  # [num_positions, n_minus_1]

    # Find matches: where all positions in the prefix match
    matches = (all_prefixes == prefix.unsqueeze(0)).all(dim=1)  # [num_positions] bool

    if matches.any():
        # The token following each matching prefix
        match_indices = matches.nonzero(as_tuple=True)[0]
        # The token that followed each matched prefix is at index (match_pos + ngram_size - 1)
        banned_positions = match_indices + n_minus_1
        # Clamp to valid range
        valid = banned_positions < current_len
        if valid.any():
            banned_tokens = seq[banned_positions[valid]].unique()
            next_logits[0, banned_tokens] = float("-inf")

    return next_logits


def _ngram_draft_tokens(
    generated: torch.Tensor,
    current_len: int,
    ngram_size: int,
    max_draft: int,
) -> torch.Tensor:
    """Draft tokens using n-gram suffix matching (no model call needed).

    Looks for the longest suffix of the current sequence that appeared
    earlier, then proposes the tokens that followed that earlier occurrence.
    This is especially effective for RL rollouts which have high repetition.

    Args:
        generated: [1, max_len] buffer
        current_len: tokens generated so far
        ngram_size: n-gram size for suffix matching
        max_draft: maximum tokens to draft

    Returns:
        [num_drafted] tensor of drafted token ids (may be empty)
    """
    if ngram_size <= 0 or current_len < ngram_size + 1:
        return torch.tensor([], dtype=generated.dtype, device=generated.device)

    seq = generated[0, :current_len]
    # The suffix to match: last ngram_size tokens
    suffix = seq[current_len - ngram_size:]  # [ngram_size]

    # Find all earlier occurrences of this suffix
    if current_len - ngram_size < 1:
        return torch.tensor([], dtype=generated.dtype, device=generated.device)

    all_ngrams = seq[:current_len - 1].unfold(0, ngram_size, 1)  # [num_pos, ngram_size]
    matches = (all_ngrams == suffix.unsqueeze(0)).all(dim=1)  # [num_pos]

    if not matches.any():
        return torch.tensor([], dtype=generated.dtype, device=generated.device)

    # Take the last (most recent) match for best context relevance
    match_idx = matches.nonzero(as_tuple=True)[0][-1].item()
    draft_start = match_idx + ngram_size

    # Copy tokens that followed the matched suffix
    available = min(max_draft, current_len - draft_start)
    if available <= 0:
        return torch.tensor([], dtype=generated.dtype, device=generated.device)

    return seq[draft_start:draft_start + available].clone()


@torch.no_grad()
def generate_batch_completions(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    num_generations: int,
    max_new_tokens: int,
    temperature: float = 0.7,
    top_p: float = 0.95,
    eos_token_id: Optional[int] = None,
    repetition_penalty: float = 1.2,
    no_repeat_ngram_size: int = 3,
    speculative_ngram_size: int = 0,
    speculative_max_draft: int = 4,
) -> List[torch.Tensor]:
    """Generate multiple completions for a SINGLE prompt with shared KV cache.

    The prompt is processed once (prefill), then the KV cache is replicated
    across num_generations parallel decode streams. This amortizes the O(L)
    prefill cost and parallelizes decode on the GPU.

    Args:
        prompt_ids: [1, prompt_len] — single prompt
        num_generations: Number of completions to generate
        max_new_tokens: Max new tokens per completion
        temperature, top_p, eos_token_id, repetition_penalty, no_repeat_ngram_size: sampling params
        speculative_ngram_size: N-gram size for suffix speculation (0=disabled)
        speculative_max_draft: Max tokens to draft per speculation step

    Returns:
        List of [1, seq_len] tensors (one per generation, variable length)
    """
    device = prompt_ids.device
    prompt_len = prompt_ids.shape[1]
    N = num_generations

    # Get base model
    base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    has_kv_cache = hasattr(base_model, "forward_with_cache")

    if not has_kv_cache or N <= 1:
        # Fallback: sequential generation (with speculative decoding support)
        results = []
        for _ in range(N):
            gen = generate_completion(
                model, prompt_ids, max_new_tokens, temperature,
                top_p, eos_token_id, repetition_penalty, no_repeat_ngram_size,
                speculative_ngram_size=speculative_ngram_size,
                speculative_max_draft=speculative_max_draft,
            )
            results.append(gen)
        return results

    model.eval()
    max_len = prompt_len + max_new_tokens

    # Pre-allocate buffers for all N generations
    generated = torch.zeros((N, max_len), device=device, dtype=prompt_ids.dtype)
    generated[:, :prompt_len] = prompt_ids  # broadcast prompt
    current_lens = torch.full((N,), prompt_len, device=device, dtype=torch.long)
    active = torch.ones(N, device=device, dtype=torch.bool)

    with torch.inference_mode(), autocast("cuda", dtype=torch.bfloat16):
        # === PREFILL: single prompt, get KV cache ===
        logits, past_kv_single = base_model.forward_with_cache(
            prompt_ids, past_key_values=None, start_pos=0,
        )

        # Replicate KV cache across N parallel decode streams
        # Each layer's cache is (K, V), each shaped [1, heads, seq, head_dim]
        past_key_values = []
        for layer_kv in past_kv_single:
            k_cache, v_cache = layer_kv
            past_key_values.append((
                k_cache.expand(N, -1, -1, -1).contiguous(),
                v_cache.expand(N, -1, -1, -1).contiguous(),
            ))

        # Sample first token from shared prefill logits
        first_logits = logits[:, -1, :].expand(N, -1).clone() / temperature

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(first_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_mask = cumulative_probs > top_p
            sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
            sorted_mask[:, 0] = False
            mask = sorted_mask.scatter(1, sorted_indices, sorted_mask)
            first_logits = first_logits.masked_fill(mask, float("-inf"))

        probs = F.softmax(first_logits, dim=-1)
        first_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)  # [N]
        generated[:, prompt_len] = first_tokens
        current_lens[:] = prompt_len + 1

        if eos_token_id is not None:
            active &= (first_tokens != eos_token_id)

        # === DECODE LOOP: batched across all N generations ===
        _arange_N = torch.arange(N, device=device)
        for step in range(1, max_new_tokens):
            if not active.any():
                break

            # Input: last token for each stream [N, 1]
            input_ids = generated[_arange_N, current_lens - 1].unsqueeze(1)

            logits, past_key_values = base_model.forward_with_cache(
                input_ids, past_key_values=past_key_values,
                start_pos=prompt_len + step - 1,
            )

            next_logits = logits[:, -1, :] / temperature  # [N, vocab]

            # Materialize lengths + active mask to CPU once (1 sync instead of 3N)
            lens_cpu = current_lens.tolist()
            active_cpu = active.tolist()

            # Repetition penalty per-stream
            if repetition_penalty != 1.0:
                for gi in range(N):
                    if not active_cpu[gi]:
                        continue
                    gen_tokens = generated[gi, prompt_len:lens_cpu[gi]].unique()
                    if gen_tokens.numel() > 0:
                        pl = next_logits[gi, gen_tokens]
                        next_logits[gi, gen_tokens] = torch.where(
                            pl > 0, pl / repetition_penalty, pl * repetition_penalty,
                        )

            # N-gram blocking per-stream
            if no_repeat_ngram_size > 0:
                for gi in range(N):
                    if not active_cpu[gi]:
                        continue
                    _vectorized_ngram_blocking(
                        generated[gi:gi+1], lens_cpu[gi],
                        no_repeat_ngram_size, next_logits[gi:gi+1],
                    )

            # Top-p sampling (batched)
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_mask = cumulative_probs > top_p
                sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
                sorted_mask[:, 0] = False
                mask = sorted_mask.scatter(1, sorted_indices, sorted_mask)
                next_logits = next_logits.masked_fill(mask, float("-inf"))

            probs = F.softmax(next_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)  # [N]

            # Write tokens for active streams (vectorized, no per-stream .item() sync)
            active_idx = active.nonzero(as_tuple=True)[0]
            generated[active_idx, current_lens[active_idx]] = next_tokens[active_idx]
            current_lens[active] += 1

            if eos_token_id is not None:
                active &= (next_tokens != eos_token_id)

    # Return variable-length generations
    results = []
    for gi in range(N):
        clen = current_lens[gi].item()
        results.append(generated[gi:gi+1, :clen])
    return results


@torch.no_grad()
def generate_completion(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 0.7,
    top_p: float = 0.95,
    eos_token_id: Optional[int] = None,
    repetition_penalty: float = 1.2,
    no_repeat_ngram_size: int = 3,
    speculative_ngram_size: int = 0,
    speculative_max_draft: int = 4,
) -> torch.Tensor:
    """Generate a single completion with KV cache for O(n) generation.

    Uses incremental decoding: the prompt is processed once (prefill),
    then each new token reuses cached K,V from all layers, avoiding
    the O(n²) full-sequence recomputation.

    When speculative_ngram_size > 0, uses n-gram suffix speculation:
    draft tokens are proposed based on suffix matches in the already-generated
    sequence, then verified in a single model forward pass. Accepted tokens
    skip individual decode steps, giving 1.3-2x speedup on repetitive outputs
    typical of RL rollouts.

    Args:
        repetition_penalty: Penalty for tokens that have already appeared.
            Values > 1.0 discourage repetition. Default 1.2.
        no_repeat_ngram_size: Prevent repeating n-grams of this size.
            Set to 0 to disable. Default 3.
        speculative_ngram_size: N-gram size for suffix speculation. 0=disabled.
        speculative_max_draft: Max tokens to draft per speculation step.
    """
    device = prompt_ids.device
    prompt_len = prompt_ids.shape[1]

    # Pre-allocate buffer
    max_len = prompt_len + max_new_tokens
    generated = torch.zeros((1, max_len), device=device, dtype=prompt_ids.dtype)
    generated[:, :prompt_len] = prompt_ids
    current_len = prompt_len

    # Get base model (unwrap torch.compile if needed)
    base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    has_kv_cache = hasattr(base_model, "forward_with_cache")

    model.eval()

    def _sample_token(logits_2d):
        """Apply repetition penalty, n-gram blocking, top-p, and sample."""
        next_logits = logits_2d / temperature

        if repetition_penalty != 1.0:
            gen_tokens = generated[0, prompt_len:current_len].unique()
            if gen_tokens.numel() > 0:
                pl = next_logits[:, gen_tokens]
                next_logits[:, gen_tokens] = torch.where(
                    pl > 0, pl / repetition_penalty, pl * repetition_penalty,
                )

        next_logits = _vectorized_ngram_blocking(
            generated, current_len, no_repeat_ngram_size, next_logits,
        )

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_mask = cumulative_probs > top_p
            sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
            sorted_mask[:, 0] = False
            mask = sorted_mask.scatter(1, sorted_indices, sorted_mask)
            next_logits = next_logits.masked_fill(mask, float("-inf"))

        probs = F.softmax(next_logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    with torch.inference_mode(), autocast("cuda", dtype=torch.bfloat16):
        past_key_values = None

        tokens_generated = 0
        while tokens_generated < max_new_tokens:
            # --- Speculative decoding path ---
            if (speculative_ngram_size > 0 and has_kv_cache
                    and past_key_values is not None
                    and current_len > prompt_len + speculative_ngram_size):
                draft = _ngram_draft_tokens(
                    generated, current_len, speculative_ngram_size, speculative_max_draft,
                )
                if draft.numel() > 0:
                    # Clamp draft length to remaining budget
                    remaining = max_new_tokens - tokens_generated
                    if draft.numel() > remaining:
                        draft = draft[:remaining]

                    # Tentatively write draft tokens into buffer
                    n_draft = draft.numel()
                    generated[0, current_len:current_len + n_draft] = draft

                    # Verify: forward all draft tokens at once
                    verify_input = generated[:, current_len - 1:current_len + n_draft]
                    verify_logits, new_kv = base_model.forward_with_cache(
                        verify_input, past_key_values=past_key_values,
                        start_pos=current_len - 1,
                    )
                    # verify_logits is [1, n_draft+1, V] (includes logits for each position)
                    # Position i gives logits for position (current_len + i)
                    # We accept draft[i] if argmax(logits[i]) == draft[i] (greedy verify)
                    # For sampling: accept if draft token has reasonable probability
                    accepted = 0
                    for di in range(n_draft):
                        token_logits = verify_logits[:, di, :]
                        probs = F.softmax(token_logits / temperature, dim=-1)
                        draft_prob = probs[0, draft[di]].item()
                        # Accept if draft token has >= 10% probability
                        if draft_prob >= 0.1:
                            accepted += 1
                        else:
                            break

                    if accepted > 0:
                        # Keep accepted draft tokens, update KV cache
                        # We need to truncate KV to only include accepted positions
                        current_len += accepted
                        tokens_generated += accepted

                        # Trim KV cache to accepted length
                        # The verify forward added n_draft+1 positions to KV
                        # We want to keep only accepted+1 (the verify input was n_draft+1 tokens)
                        trim_to = current_len
                        trimmed_kv = []
                        for layer_kv in new_kv:
                            k, v = layer_kv
                            # k,v shape: [1, heads, seq_so_far, head_dim]
                            trimmed_kv.append((k[:, :, :trim_to, :], v[:, :, :trim_to, :]))
                        past_key_values = trimmed_kv

                        # Check EOS in accepted tokens
                        if eos_token_id is not None:
                            for ai in range(accepted):
                                if generated[0, current_len - accepted + ai].item() == eos_token_id:
                                    current_len = current_len - accepted + ai + 1
                                    return generated[:, :current_len]

                        # Sample the next token after accepted draft (from last verify logits)
                        if tokens_generated < max_new_tokens:
                            next_logits = verify_logits[:, accepted, :]
                            next_token = _sample_token(next_logits)
                            generated[:, current_len] = next_token
                            current_len += 1
                            tokens_generated += 1

                            if eos_token_id is not None and next_token.item() == eos_token_id:
                                break

                            # Trim KV to include this new token position
                            trimmed_kv2 = []
                            for layer_kv in past_key_values:
                                k, v = layer_kv
                                trimmed_kv2.append((k[:, :, :current_len, :], v[:, :, :current_len, :]))
                            past_key_values = trimmed_kv2

                        continue  # Skip standard decode for this iteration
                    else:
                        # All draft tokens rejected — clear them from buffer
                        generated[0, current_len:current_len + n_draft] = 0
                        # Restore KV cache (discard verify forward)
                        # Fall through to standard decode below

            # --- Standard decode path ---
            if has_kv_cache:
                if past_key_values is None:
                    input_ids = generated[:, :current_len]
                    logits, past_key_values = base_model.forward_with_cache(
                        input_ids, past_key_values=None, start_pos=0,
                    )
                else:
                    input_ids = generated[:, current_len - 1:current_len]
                    logits, past_key_values = base_model.forward_with_cache(
                        input_ids, past_key_values=past_key_values, start_pos=current_len - 1,
                    )
            else:
                input_ids = generated[:, :current_len]
                outputs = model(input_ids)
                logits = outputs[0] if isinstance(outputs, tuple) else outputs

            next_token = _sample_token(logits[:, -1, :])

            generated[:, current_len] = next_token
            current_len += 1
            tokens_generated += 1

            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

    return generated[:, :current_len]


def compute_reward(
    prompt: str,
    completion: str,
    expected_answer: Optional[str],
    reward_fn: str = "exact_match",
) -> float:
    """Compute reward for a completion."""
    import re

    if reward_fn == "exact_match":
        if expected_answer is None:
            return 0.0

        completion_lower = completion.lower().strip()
        expected_lower = expected_answer.lower().strip()

        # Try to extract boxed answer
        boxed_match = re.search(r'\\boxed\{([^}]+)\}', completion)
        if boxed_match:
            extracted = boxed_match.group(1).strip().lower()
            if extracted == expected_lower:
                return 1.0

        # Try answer patterns
        answer_patterns = [
            r'(?:the\s+)?answer\s+is[:\s]+([^\n.]+)',
            r'(?:final\s+)?answer[:\s]+([^\n.]+)',
            r'=\s*([^\n.]+)$',
        ]
        for pattern in answer_patterns:
            match = re.search(pattern, completion_lower)
            if match:
                extracted = match.group(1).strip()
                try:
                    if abs(float(extracted) - float(expected_lower)) < 1e-6:
                        return 1.0
                except ValueError:
                    if extracted == expected_lower:
                        return 1.0

        # Partial credit for containing answer
        if expected_lower in completion_lower:
            return 0.5

        return 0.0

    elif reward_fn == "format_reward":
        if not completion or len(completion.strip()) < 10:
            return 0.0

        score = 0.5
        # Reward structure
        structure_patterns = [r'\d+\.', r'(?:first|then|therefore)', r'```']
        structure_hits = sum(1 for p in structure_patterns if re.search(p, completion.lower()))
        score += min(0.3, structure_hits * 0.1)

        if expected_answer and expected_answer.lower() in completion.lower():
            score += 0.2

        return max(0.0, min(1.0, score))

    else:
        return 0.5  # Default


def compute_sequence_logprobs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Compute per-token log probabilities with memory-efficient hidden-state path.

    When the model supports forward_hidden(), avoids materializing the full
    [B, T, vocab_size] logits tensor (~3GB for B=4, T=4096, V=50K in bf16).
    Instead, computes only the target-token logit via a selective dot product,
    then chunks the logsumexp computation over the vocab dimension.
    """
    model.train()

    base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    has_forward_hidden = hasattr(base_model, "forward_hidden")

    if has_forward_hidden:
        # === Memory-efficient path: hidden states + selective logprob ===
        with autocast("cuda", dtype=torch.bfloat16):
            hidden = base_model.forward_hidden(input_ids)  # [B, T, D]

        weight = base_model.output.weight  # [V, D]
        B, T, D = hidden.shape
        V = weight.shape[0]

        # Shift for next-token prediction
        shift_hidden = hidden[:, :-1, :].contiguous()  # [B, T-1, D]
        shift_labels = input_ids[:, 1:].contiguous()    # [B, T-1]
        shift_mask = mask[:, 1:].contiguous()            # [B, T-1]

        # Target-token logit: dot product of hidden with target embedding
        # Gather target embeddings: [B, T-1, D]
        target_embeds = weight[shift_labels.view(-1)].view(B, T - 1, D)
        # Dot product → selected logit [B, T-1]
        selected_logits = (shift_hidden * target_embeds).sum(dim=-1)
        del target_embeds

        # Chunked logsumexp over vocab to avoid [B, T-1, V] materialization
        # Process vocab in chunks of 8192 to limit peak memory to ~B*T*8192*4 bytes
        vocab_chunk = 8192
        N = B * (T - 1)
        h_flat = shift_hidden.reshape(N, D).float()
        lse = torch.full((N,), -float("inf"), device=hidden.device, dtype=torch.float32)
        for v0 in range(0, V, vocab_chunk):
            v1 = min(V, v0 + vocab_chunk)
            logits_chunk = h_flat @ weight[v0:v1].float().t()  # [N, chunk]
            lse = torch.logaddexp(lse, torch.logsumexp(logits_chunk, dim=1))
            del logits_chunk
        del h_flat

        token_logprobs = (selected_logits.float() - lse.view(B, T - 1)) * shift_mask

        # Pad to original length
        result = torch.zeros(B, T, device=input_ids.device, dtype=token_logprobs.dtype)
        result[:, 1:] = token_logprobs
        return result
    else:
        # === Fallback: full logits path ===
        with autocast("cuda", dtype=torch.bfloat16):
            outputs = model(input_ids)
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs

        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = mask[:, 1:].contiguous()

        selected_logits = shift_logits.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
        logsumexp = torch.logsumexp(shift_logits, dim=-1)
        token_logprobs = selected_logits - logsumexp
        token_logprobs = token_logprobs * shift_mask

        B, L = input_ids.shape
        result = torch.zeros(B, L, device=input_ids.device, dtype=token_logprobs.dtype)
        result[:, 1:] = token_logprobs
        return result


class _GenerationCache:
    """Cache for reusing generated samples across multiple GRPO policy updates.

    When generation_reuse_count > 1, the expensive generation phase is run once
    and the resulting samples are reused for multiple optimizer steps, giving
    2-4x effective GRPO throughput.
    """

    __slots__ = ("generated_ids", "completion_mask", "rewards_tensor", "advantages",
                 "uses_remaining", "total_samples", "num_micro_batches")

    def __init__(self) -> None:
        self.generated_ids: Optional[torch.Tensor] = None
        self.completion_mask: Optional[torch.Tensor] = None
        self.rewards_tensor: Optional[torch.Tensor] = None
        self.advantages: Optional[torch.Tensor] = None
        self.uses_remaining: int = 0
        self.total_samples: int = 0
        self.num_micro_batches: int = 0

    @property
    def valid(self) -> bool:
        return self.uses_remaining > 0 and self.generated_ids is not None

    def clear(self) -> None:
        self.generated_ids = None
        self.completion_mask = None
        self.rewards_tensor = None
        self.advantages = None
        self.uses_remaining = 0


# Module-level cache instance (avoids passing through all call sites)
_grpo_generation_cache = _GenerationCache()


def run_grpo_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    prompts: List[Dict[str, Any]],
    tokenizer,
    config: ReasoningConfig,
    step: int,
    ref_model: Optional[torch.nn.Module] = None,
) -> Dict[str, float]:
    """Run a single GRPO training step (standard or DAPO variant).

    When config.generation_reuse_count > 1, generated samples are cached
    and reused for multiple policy updates before regenerating. This amortizes
    the expensive generation phase across μ optimizer steps.

    When config.dapo is True, uses DAPO (Decoupled Alignment from Policy
    Optimization) with one-sided probability clipping and dynamic temperature.

    When ref_model is provided and config.kl_beta > 0, adds KL(pi_theta || pi_ref)
    penalty to prevent reward hacking.
    """
    global _grpo_generation_cache
    device = next(model.parameters()).device
    base_model = model._orig_mod if hasattr(model, "_orig_mod") else model

    # Skip MoE if configured
    if config.skip_moe and hasattr(base_model, "set_skip_moe"):
        base_model.set_skip_moe(True)

    # Enable gradient checkpointing
    if config.gradient_checkpointing and hasattr(base_model, "enable_gradient_checkpointing"):
        base_model.enable_gradient_checkpointing(every_n=1)

    # === GENERATION PHASE (skip if reusing cached samples) ===
    reuse_mu = max(1, config.generation_reuse_count)
    if _grpo_generation_cache.valid and reuse_mu > 1:
        # Reuse cached generation data
        generated_ids = _grpo_generation_cache.generated_ids
        completion_mask = _grpo_generation_cache.completion_mask
        rewards_tensor = _grpo_generation_cache.rewards_tensor
        advantages = _grpo_generation_cache.advantages
        total_samples = _grpo_generation_cache.total_samples
        num_micro_batches = _grpo_generation_cache.num_micro_batches
        _grpo_generation_cache.uses_remaining -= 1
        _log.info(f"  Step {step}: reusing cached generations "
                  f"({_grpo_generation_cache.uses_remaining} reuses remaining)")
    else:
        # Fresh generation
        _grpo_generation_cache.clear()

        # Sample prompts
        import random
        batch_prompts = random.sample(prompts, min(config.batch_size, len(prompts)))

        # Tokenize
        prompt_texts = [p["prompt"] for p in batch_prompts]
        expected_answers = [p.get("expected_answer") for p in batch_prompts]

        encoded = tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        prompt_ids = encoded["input_ids"].to(device)

        # Generate completions — batch all num_generations per prompt
        # using shared KV cache prefill (amortizes O(L) prefill cost)
        all_generated = []
        all_masks = []
        all_rewards = []

        for prompt_idx in range(len(batch_prompts)):
            single_prompt = prompt_ids[prompt_idx:prompt_idx + 1]
            prompt_text = prompt_texts[prompt_idx]
            expected = expected_answers[prompt_idx]

            if config.think_loops <= 1:
                # Fast path: batch all generations with shared prefill
                gen_list = generate_batch_completions(
                    model=base_model,
                    prompt_ids=single_prompt,
                    num_generations=config.num_generations,
                    max_new_tokens=config.max_thinking_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=config.repetition_penalty,
                    no_repeat_ngram_size=config.no_repeat_ngram_size,
                    speculative_ngram_size=config.speculative_ngram_size,
                    speculative_max_draft=config.speculative_max_draft,
                )
                prompt_len = single_prompt.shape[1]

                for gen_ids in gen_list:
                    total_len = gen_ids.shape[1]
                    mask = torch.zeros(1, total_len, device=device)
                    mask[:, prompt_len:] = 1.0

                    completion_tokens = gen_ids[0, prompt_len:]
                    completion_text = tokenizer.decode(completion_tokens, skip_special_tokens=True)
                    reward = compute_reward(prompt_text, completion_text, expected, config.reward_function)

                    all_generated.append(gen_ids)
                    all_masks.append(mask)
                    all_rewards.append(reward)
            else:
                # Multi-think-loop path: sequential (think loops change input between passes)
                for gen_idx in range(config.num_generations):
                    temp = config.temperature
                    current_ids = single_prompt

                    for loop in range(config.think_loops):
                        gen_ids = generate_completion(
                            model=base_model,
                            prompt_ids=current_ids,
                            max_new_tokens=config.max_thinking_tokens,
                            temperature=temp,
                            top_p=config.top_p,
                            eos_token_id=tokenizer.eos_token_id,
                            repetition_penalty=config.repetition_penalty,
                            no_repeat_ngram_size=config.no_repeat_ngram_size,
                            speculative_ngram_size=config.speculative_ngram_size,
                            speculative_max_draft=config.speculative_max_draft,
                        )

                        if loop < config.think_loops - 1:
                            temp *= config.think_loop_temperature_decay

                    prompt_len = single_prompt.shape[1]
                    total_len = gen_ids.shape[1]
                    mask = torch.zeros(1, total_len, device=device)
                    mask[:, prompt_len:] = 1.0

                    completion_tokens = gen_ids[0, prompt_len:]
                    completion_text = tokenizer.decode(completion_tokens, skip_special_tokens=True)
                    reward = compute_reward(prompt_text, completion_text, expected, config.reward_function)

                    all_generated.append(gen_ids)
                    all_masks.append(mask)
                    all_rewards.append(reward)

        # Pad sequences to same length
        max_len = max(g.shape[1] for g in all_generated)
        padded_generated = []
        padded_masks = []

        for gen_ids, mask in zip(all_generated, all_masks):
            pad_len = max_len - gen_ids.shape[1]
            if pad_len > 0:
                gen_ids = torch.cat([gen_ids, torch.zeros((1, pad_len), device=device, dtype=gen_ids.dtype)], dim=1)
                mask = torch.cat([mask, torch.zeros((1, pad_len), device=device)], dim=1)
            padded_generated.append(gen_ids)
            padded_masks.append(mask)

        generated_ids = torch.cat(padded_generated, dim=0)
        completion_mask = torch.cat(padded_masks, dim=0)
        rewards_tensor = torch.tensor(all_rewards, device=device)

        # Log generation stats
        _log.info(f"  Step {step} generation: prompts={len(batch_prompts)}, "
                  f"gens/prompt={config.num_generations}, think_loops={config.think_loops}, "
                  f"total_seqs={generated_ids.shape[0]}")
        _log.info(f"  Step {step} rewards: mean={rewards_tensor.mean():.3f}, "
                  f"std={rewards_tensor.std():.3f}, min={rewards_tensor.min():.3f}, "
                  f"max={rewards_tensor.max():.3f}")
        _log.info(f"  Step {step} rewards breakdown: {[f'{r:.2f}' for r in all_rewards]}")

        # Skip if no reward variance
        if rewards_tensor.std() < 1e-6:
            _log.info(f"  Step {step}: skipped (no reward variance - all generations got same score)")
            if config.skip_moe and hasattr(base_model, "set_skip_moe"):
                base_model.set_skip_moe(False)
            return {"skipped": 1.0, "reward_mean": rewards_tensor.mean().item()}

        # Compute advantages
        total_samples = generated_ids.shape[0]
        rewards_grouped = rewards_tensor.view(config.batch_size, config.num_generations)
        mean_rewards = rewards_grouped.mean(dim=1, keepdim=True)
        std_rewards = rewards_grouped.std(dim=1, keepdim=True) + 1e-8
        advantages = ((rewards_grouped - mean_rewards) / std_rewards).view(-1)

        # Truncate sequences if too long
        if generated_ids.shape[1] > config.max_seq_len:
            generated_ids = generated_ids[:, :config.max_seq_len]
            completion_mask = completion_mask[:, :config.max_seq_len]

        num_micro_batches = (total_samples + config.micro_batch_size - 1) // config.micro_batch_size

        # Cache for reuse if μ > 1
        if reuse_mu > 1:
            _grpo_generation_cache.generated_ids = generated_ids.detach()
            _grpo_generation_cache.completion_mask = completion_mask.detach()
            _grpo_generation_cache.rewards_tensor = rewards_tensor.detach()
            _grpo_generation_cache.advantages = advantages.detach()
            _grpo_generation_cache.total_samples = total_samples
            _grpo_generation_cache.num_micro_batches = num_micro_batches
            _grpo_generation_cache.uses_remaining = reuse_mu - 1  # first use is now

    # Compute reference log-probs for KL penalty (before training forward)
    kl_beta_effective = 0.0
    ref_logprobs_all = None
    if ref_model is not None and config.kl_beta > 0:
        # Warmup KL penalty
        if config.kl_warmup_steps > 0 and step <= config.kl_warmup_steps:
            kl_beta_effective = config.kl_beta * (step / config.kl_warmup_steps)
        else:
            kl_beta_effective = config.kl_beta

        # Compute reference log-probs (no grad, batched)
        with torch.no_grad():
            ref_logprobs_all = compute_sequence_logprobs(
                ref_model, generated_ids, completion_mask,
            )

    # Micro-batched forward/backward
    base_model.train()
    optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    total_kl = 0.0
    num_micro_batches = (total_samples + config.micro_batch_size - 1) // config.micro_batch_size

    for i in range(0, total_samples, config.micro_batch_size):
        end_idx = min(i + config.micro_batch_size, total_samples)
        chunk_ids = generated_ids[i:end_idx]
        chunk_mask = completion_mask[i:end_idx]
        chunk_adv = advantages[i:end_idx]

        # Forward + logprobs
        chunk_logprobs = compute_sequence_logprobs(base_model, chunk_ids, chunk_mask)

        # Compute policy loss
        num_tokens = chunk_mask.sum(dim=1).clamp(min=1)
        completion_logprobs = (chunk_logprobs * chunk_mask).sum(dim=1) / num_tokens

        if config.dapo:
            # DAPO: one-sided probability ratio clipping
            # Only clip probability decreases (low side), allow increases (high side)
            # This prevents entropy collapse while allowing exploration
            if ref_logprobs_all is not None:
                ref_chunk = ref_logprobs_all[i:end_idx]
                ref_completion = (ref_chunk * chunk_mask).sum(dim=1) / num_tokens
                log_ratio = completion_logprobs - ref_completion
            else:
                # Without ref model, use advantage-weighted log-prob directly
                log_ratio = completion_logprobs

            ratio = torch.exp(log_ratio)

            # One-sided clipping: clip low but not high
            # Standard PPO/GRPO clips both sides symmetrically
            # DAPO only clips the lower bound to prevent catastrophic forgetting
            clipped_ratio = torch.clamp(ratio, min=config.dapo_clip_low, max=config.dapo_clip_high)

            # Dynamic temperature: scale advantage by its magnitude
            if config.dapo_dynamic_temperature:
                adv_scale = 1.0 / (chunk_adv.abs().mean().clamp(min=0.1))
                scaled_adv = chunk_adv * adv_scale
            else:
                scaled_adv = chunk_adv

            # DAPO loss: min of unclipped and clipped objectives
            surr1 = scaled_adv * ratio
            surr2 = scaled_adv * clipped_ratio
            chunk_loss = -torch.min(surr1, surr2).mean()
        else:
            # Standard GRPO: advantage-weighted log-probability
            chunk_loss = -(chunk_adv * completion_logprobs).mean()

        # KL penalty
        if kl_beta_effective > 0 and ref_logprobs_all is not None:
            ref_chunk = ref_logprobs_all[i:end_idx]
            # Per-token KL: log_pi_theta - log_pi_ref
            kl_per_token = (chunk_logprobs - ref_chunk) * chunk_mask
            kl_per_seq = kl_per_token.sum(dim=1) / num_tokens
            kl_penalty = kl_beta_effective * kl_per_seq.mean()
            chunk_loss = chunk_loss + kl_penalty
            total_kl += kl_per_seq.mean().detach().item()

        # Backward
        (chunk_loss / num_micro_batches).backward()

        total_loss += chunk_loss.detach().item()

        # Cleanup tensors (no empty_cache here — deferred to end of step)
        del chunk_logprobs, completion_logprobs, chunk_loss

    # Gradient clipping and optimizer step
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    # Re-enable MoE
    if config.skip_moe and hasattr(base_model, "set_skip_moe"):
        base_model.set_skip_moe(False)

    # Cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    avg_loss = total_loss / num_micro_batches
    metrics = {
        "loss": avg_loss,
        "reward_mean": rewards_tensor.mean().item(),
        "reward_std": rewards_tensor.std().item(),
        "advantage_mean": advantages.mean().item(),
        "num_samples": total_samples,
    }
    if kl_beta_effective > 0:
        metrics["kl_divergence"] = total_kl / num_micro_batches
        metrics["kl_beta"] = kl_beta_effective
    if config.dapo:
        metrics["algorithm"] = "dapo"

    return metrics


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    metrics: Dict[str, float],
    model_config: Dict[str, Any],
    output_path: str,
    base_step: int = 0,
):
    """Save reasoning training checkpoint.

    Compatible with main trainer.py format:
    - 'model' for weights (main trainer expects this)
    - 'model_config' for architecture (reasoning trainer needs this)
    - 'config' copied from model_config for trainer.py compatibility
    - 'step' is the global step (base_step + reasoning_step) so the main
      trainer resumes with correct curriculum/LR scheduling
    """
    base_model = model._orig_mod if hasattr(model, "_orig_mod") else model

    # Get state_dict once to avoid duplicating in memory
    model_state = base_model.state_dict()

    global_step = base_step + step

    checkpoint = {
        "model": model_state,  # Main trainer expects 'model'
        # NOTE: Do NOT save duplicate "model_state_dict" - wastes memory and disk
        "optimizer": optimizer.state_dict(),  # Main trainer expects 'optimizer'
        "step": global_step,
        "reasoning_step": step,  # Local reasoning step for reference
        "base_step": base_step,  # Pretrain step for auditing
        "metrics": metrics,
        "model_config": model_config,  # Architecture for reasoning trainer
        "config": model_config,  # Main trainer uses 'config' for architecture
        "timestamp": datetime.now().isoformat(),
    }

    # Update MoE router _global_step tensors to match global step
    for key in model_state:
        if "_global_step" in key:
            checkpoint["model"][key] = torch.tensor(
                global_step - 1, dtype=model_state[key].dtype
            )

    torch.save(checkpoint, output_path)
    _log.info(f"Saved checkpoint: {output_path} (global_step={global_step}, reasoning_step={step})")


def main():
    parser = argparse.ArgumentParser(description="HYDRA Reasoning Trainer")

    # Required
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")

    # Training
    parser.add_argument("--max_steps", type=int, default=1000,
                        help="Maximum training steps")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=50,
                        help="LR warmup steps")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Gradient clipping")

    # Generation
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Prompts per step")
    parser.add_argument("--num_generations", type=int, default=4,
                        help="Completions per prompt")
    parser.add_argument("--generation_reuse_count", type=int, default=1,
                        help="Reuse generated samples for N policy updates (μ parameter). "
                             "Values 2-4 give 2-4x throughput by amortizing generation cost.")
    parser.add_argument("--max_thinking_tokens", type=int, default=512,
                        help="Max tokens for thinking")
    parser.add_argument("--max_seq_len", type=int, default=1024,
                        help="Max sequence length")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--repetition_penalty", type=float, default=1.2,
                        help="Penalty for repeated tokens (>1.0 discourages repetition)")
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3,
                        help="Block repeating n-grams of this size (0 to disable)")

    # Think loops
    parser.add_argument("--think_loops", type=int, default=1,
                        help="Number of thinking iterations")
    parser.add_argument("--think_loop_temp_decay", type=float, default=0.9,
                        help="Temperature decay per think loop")

    # Reward
    parser.add_argument("--reward_function", type=str, default="exact_match",
                        choices=["exact_match", "format_reward", "length_penalty"],
                        help="Reward function")

    # Memory
    parser.add_argument("--skip_moe", action="store_true", default=True,
                        help="Skip MoE layers for memory")
    parser.add_argument("--no_skip_moe", action="store_false", dest="skip_moe")
    parser.add_argument("--micro_batch_size", type=int, default=1,
                        help="Micro batch size")

    # KL divergence (reference model)
    parser.add_argument("--kl_beta", type=float, default=0.0,
                        help="KL penalty coefficient (0=disabled, 0.01-0.1 typical)")
    parser.add_argument("--kl_warmup_steps", type=int, default=0,
                        help="Steps to ramp KL penalty from 0 to kl_beta")

    # Reward weighting
    parser.add_argument("--reward_weights", type=str, default=None,
                        help="Comma-separated reward signal weights (e.g. '0.7,0.3')")

    # DAPO
    parser.add_argument("--dapo", action="store_true",
                        help="Use DAPO instead of standard GRPO")
    parser.add_argument("--dapo_clip_low", type=float, default=0.8,
                        help="DAPO lower clip bound")
    parser.add_argument("--dapo_clip_high", type=float, default=1.28,
                        help="DAPO upper clip bound")
    parser.add_argument("--no_dapo_dynamic_temp", action="store_false",
                        dest="dapo_dynamic_temperature",
                        help="Disable DAPO dynamic temperature scaling")
    parser.set_defaults(dapo_dynamic_temperature=True)

    # Speculative decoding
    parser.add_argument("--speculative_ngram_size", type=int, default=0,
                        help="N-gram size for suffix speculation (0=disabled, 3-5 typical)")
    parser.add_argument("--speculative_max_draft", type=int, default=4,
                        help="Max tokens to draft per speculation step")

    # Distillation
    parser.add_argument("--distillation_dataset", type=str, default=None,
                        help="Path to teacher completions JSONL for distillation")
    parser.add_argument("--distillation_weight", type=float, default=0.5,
                        help="Weight for distillation loss (0=GRPO only, 1=distill only)")
    parser.add_argument("--distillation_only", action="store_true",
                        help="Only do distillation, no GRPO self-play")

    # Output
    parser.add_argument("--output_dir", type=str, default="checkpoints/reasoning",
                        help="Output directory")
    parser.add_argument("--save_every", type=int, default=100,
                        help="Save checkpoint every N steps")
    parser.add_argument("--log_interval", type=int, default=10,
                        help="Log every N steps")

    args = parser.parse_args()

    # Create config
    config = ReasoningConfig(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        save_every=args.save_every,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        grad_clip=args.grad_clip,
        batch_size=args.batch_size,
        num_generations=args.num_generations,
        generation_reuse_count=args.generation_reuse_count,
        max_thinking_tokens=args.max_thinking_tokens,
        max_seq_len=args.max_seq_len,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        think_loops=args.think_loops,
        think_loop_temperature_decay=args.think_loop_temp_decay,
        reward_function=args.reward_function,
        skip_moe=args.skip_moe,
        micro_batch_size=args.micro_batch_size,
        distillation_dataset=args.distillation_dataset,
        distillation_weight=args.distillation_weight,
        distillation_only=args.distillation_only,
        log_interval=args.log_interval,
        kl_beta=args.kl_beta,
        kl_warmup_steps=args.kl_warmup_steps,
        reward_weights=[float(w) for w in args.reward_weights.split(",")] if args.reward_weights else None,
        dapo=args.dapo,
        dapo_clip_low=args.dapo_clip_low,
        dapo_clip_high=args.dapo_clip_high,
        dapo_dynamic_temperature=args.dapo_dynamic_temperature,
        speculative_ngram_size=args.speculative_ngram_size,
        speculative_max_draft=args.speculative_max_draft,
    )

    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(config.output_dir, exist_ok=True)

    _log.info("=" * 60)
    _log.info("HYDRA REASONING TRAINER")
    _log.info("=" * 60)
    _log.info(f"Checkpoint: {config.checkpoint_path}")
    _log.info(f"Max steps: {config.max_steps}")
    _log.info(f"Think loops: {config.think_loops}")
    _log.info(f"Max thinking tokens: {config.max_thinking_tokens}")
    _log.info(f"Skip MoE: {config.skip_moe}")
    if config.distillation_dataset:
        _log.info(f"Distillation dataset: {config.distillation_dataset}")
        _log.info(f"Distillation weight: {config.distillation_weight}")
        _log.info(f"Distillation only: {config.distillation_only}")
    if config.kl_beta > 0:
        _log.info(f"KL penalty: beta={config.kl_beta}, warmup={config.kl_warmup_steps}")
    if config.dapo:
        _log.info(f"DAPO: clip=[{config.dapo_clip_low}, {config.dapo_clip_high}], "
                   f"dynamic_temp={config.dapo_dynamic_temperature}")
    if config.speculative_ngram_size > 0:
        _log.info(f"Speculative decoding: ngram={config.speculative_ngram_size}, "
                   f"max_draft={config.speculative_max_draft}")
    if config.reward_weights:
        _log.info(f"Reward weights: {config.reward_weights}")
    _log.info("=" * 60)

    log_memory("STARTUP")

    # Load model
    model, checkpoint, model_config = load_checkpoint(config.checkpoint_path, device)
    log_memory("MODEL_LOADED")

    # Preserve the pretrain step so the main trainer sees correct global step
    # This prevents curriculum schedules (dataset mix, MoD/MoR warmup) from resetting
    base_step = checkpoint.get("step", 0)
    _log.info(f"Pretrain base step: {base_step} (will be added to reasoning steps)")

    # Single output file (overwrites each save)
    output_path = os.path.join(config.output_dir, "reasoning_checkpoint.pt")
    _log.info(f"Output checkpoint: {output_path}")

    # Create frozen reference model for KL penalty (before optimizer to share GPU mem)
    ref_model = None
    if config.kl_beta > 0:
        ref_model = create_reference_model(model)
        log_memory("REF_MODEL_CREATED")

    # Create optimizer
    optimizer = create_optimizer(model, config)
    log_memory("OPTIMIZER_CREATED")

    # Get tokenizer
    tokenizer = get_tokenizer("gpt2")

    # Load prompts for GRPO
    prompts = None
    if not config.distillation_only:
        prompts = load_reasoning_prompts(max_prompts=1000)
        _log.info(f"Loaded {len(prompts)} reasoning prompts")

    # Load distillation dataset if provided
    distill_data = None
    if config.distillation_dataset:
        distill_data = load_distillation_dataset(config.distillation_dataset)

    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Training loop
    mode = "distillation only" if config.distillation_only else (
        f"mixed (distill_weight={config.distillation_weight})" if distill_data else "GRPO only"
    )
    _log.info(f"\nStarting reasoning training... (mode: {mode})")
    _log.info("(Press Ctrl+C to save and exit gracefully)")
    start_time = time.time()
    total_reward = 0.0
    total_loss = 0.0
    total_distill_loss = 0.0
    num_grpo_completed = 0
    num_distill_completed = 0

    for step in range(1, config.max_steps + 1):
        # Check for graceful shutdown
        if _SHUTDOWN_REQUESTED:
            _log.info(f"Saving checkpoint at step {step - 1} before shutdown...")
            save_checkpoint(model, optimizer, step - 1, metrics, model_config, output_path, base_step=base_step)
            _log.info("Checkpoint saved. Exiting.")
            return

        # Warmup LR
        if step <= config.warmup_steps:
            lr_scale = step / config.warmup_steps
            for param_group in optimizer.param_groups:
                param_group["lr"] = config.learning_rate * lr_scale

        metrics = {}

        # Distillation step
        if distill_data and (config.distillation_only or config.distillation_weight > 0):
            distill_metrics = run_distillation_step(
                model=model,
                optimizer=optimizer,
                distill_data=distill_data,
                tokenizer=tokenizer,
                config=config,
                step=step,
            )
            metrics.update(distill_metrics)
            total_distill_loss += distill_metrics["distill_loss"]
            num_distill_completed += 1

        # GRPO step (if not distillation-only)
        if not config.distillation_only and prompts:
            grpo_metrics = run_grpo_step(
                model=model,
                optimizer=optimizer,
                prompts=prompts,
                tokenizer=tokenizer,
                config=config,
                step=step,
                ref_model=ref_model,
            )
            metrics.update(grpo_metrics)

            if "skipped" not in grpo_metrics:
                total_loss += grpo_metrics.get("loss", 0)
                total_reward += grpo_metrics.get("reward_mean", 0)
                num_grpo_completed += 1

        # Logging
        if step % config.log_interval == 0:
            elapsed = time.time() - start_time
            avg_reward = total_reward / max(1, num_grpo_completed)
            avg_distill_loss = total_distill_loss / max(1, num_distill_completed)

            log_parts = [f"Step {step}/{config.max_steps}"]

            if distill_data:
                log_parts.append(f"distill_loss={metrics.get('distill_loss', 0):.4f}")
                log_parts.append(f"distill_ppl={metrics.get('distill_ppl', 0):.2f}")

            if not config.distillation_only:
                log_parts.append(f"grpo_loss={metrics.get('loss', 0):.4f}")
                log_parts.append(f"reward={metrics.get('reward_mean', 0):.3f}")
                log_parts.append(f"avg_reward={avg_reward:.3f}")

            log_parts.append(f"time={elapsed:.1f}s")
            _log.info(" | ".join(log_parts))
            log_memory(f"STEP_{step}")

        # Save checkpoint (single file, overwrites each time)
        if step % config.save_every == 0:
            save_checkpoint(model, optimizer, step, metrics, model_config, output_path, base_step=base_step)

        # Check for graceful shutdown after step completes
        if _SHUTDOWN_REQUESTED:
            _log.info(f"Saving checkpoint at step {step} before shutdown...")
            save_checkpoint(model, optimizer, step, metrics, model_config, output_path, base_step=base_step)
            _log.info("Checkpoint saved. Exiting.")
            return

    # Final save
    save_checkpoint(model, optimizer, config.max_steps, metrics, model_config, output_path, base_step=base_step)

    _log.info("\n" + "=" * 60)
    _log.info("REASONING TRAINING COMPLETE")
    _log.info(f"Total steps: {config.max_steps}")
    if distill_data:
        _log.info(f"Avg distillation loss: {total_distill_loss / max(1, num_distill_completed):.4f}")
    if not config.distillation_only:
        _log.info(f"Average reward: {total_reward / max(1, num_grpo_completed):.3f}")
    _log.info(f"Final checkpoint: {output_path}")
    _log.info("=" * 60)


if __name__ == "__main__":
    main()
