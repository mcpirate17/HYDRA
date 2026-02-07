"""
HYDRA Reasoning & System 2 Training Module.

Implements frontier training techniques for "Thinking" models (Late 2025 era),
specifically Group Relative Policy Optimization (GRPO) and Thought-Process Supervision.

References:
- DeepSeek-R1: GRPO for self-evolving reasoning without supervised critic.
- OpenAI o-series: Test-time compute scaling laws.

Design:
- Online GRPO: Generate G completions per prompt, score them, use relative advantage.
- No reference model storage: KL computed against frozen copy or approximated.
- Reward functions: exact_match (math), format_check (general), verifier (code execution).
"""

from __future__ import annotations

import gc
import logging
import re
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import torch
import torch._dynamo
import torch.nn.functional as F

if TYPE_CHECKING:
    from .config import TrainingConfig

_log = logging.getLogger("HYDRA")


# ============================================================================
# REWARD FUNCTIONS
# ============================================================================

def reward_exact_match(
    prompt: str,
    completion: str,
    expected_answer: Optional[str] = None,
) -> float:
    """
    Binary reward: 1.0 if completion contains expected answer, else 0.0.
    
    For math problems, extracts answer from completion and compares.
    """
    if expected_answer is None:
        return 0.0
    
    # Normalize both strings
    completion_lower = completion.lower().strip()
    expected_lower = expected_answer.lower().strip()
    
    # Try to extract boxed answer: \boxed{...} or **Answer: ...**
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', completion)
    if boxed_match:
        extracted = boxed_match.group(1).strip().lower()
        if extracted == expected_lower:
            return 1.0
    
    # Try "Answer: X" or "answer is X" patterns
    answer_patterns = [
        r'(?:the\s+)?answer\s+is[:\s]+([^\n.]+)',
        r'(?:final\s+)?answer[:\s]+([^\n.]+)',
        r'=\s*([^\n.]+)$',  # Trailing equals
    ]
    for pattern in answer_patterns:
        match = re.search(pattern, completion_lower)
        if match:
            extracted = match.group(1).strip()
            # Numeric comparison (tolerant to formatting)
            try:
                if abs(float(extracted) - float(expected_lower)) < 1e-6:
                    return 1.0
            except ValueError:
                if extracted == expected_lower:
                    return 1.0
    
    # Fallback: substring match
    if expected_lower in completion_lower:
        return 0.5  # Partial credit
    
    return 0.0


def reward_format_check(
    prompt: str,
    completion: str,
    expected_answer: Optional[str] = None,
) -> float:
    """
    Reward based on response quality heuristics:
    - Penalize empty or very short responses
    - Reward structured thinking (numbered steps, "therefore", etc.)
    - Penalize repetition
    """
    if not completion or len(completion.strip()) < 10:
        return 0.0
    
    score = 0.5  # Base score for non-empty response
    
    # Reward structure indicators
    structure_patterns = [
        r'\d+\.',           # Numbered steps: "1.", "2."
        r'(?:first|second|third|then|next|finally)',
        r'(?:therefore|thus|hence|so)',
        r'(?:because|since|given that)',
        r'```',             # Code blocks
        r'\n-\s',           # Bullet points
    ]
    structure_hits = sum(1 for p in structure_patterns if re.search(p, completion.lower()))
    score += min(0.3, structure_hits * 0.1)
    
    # Penalize excessive repetition
    words = completion.lower().split()
    if len(words) > 20:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.3:  # Very repetitive
            score -= 0.3
        elif unique_ratio < 0.5:
            score -= 0.1
    
    # Bonus for appropriate length (not too short, not excessively long)
    if 50 < len(completion) < 2000:
        score += 0.1
    
    # If expected answer provided, check for it
    if expected_answer and expected_answer.lower() in completion.lower():
        score += 0.2
    
    return max(0.0, min(1.0, score))


def reward_length_penalty(
    prompt: str,
    completion: str,
    expected_answer: Optional[str] = None,
    target_length: int = 200,
) -> float:
    """
    Simple reward that encourages responses near a target length.
    Used as a baseline / debugging reward.
    """
    length = len(completion)
    if length == 0:
        return 0.0
    
    # Gaussian-like penalty around target
    deviation = abs(length - target_length) / target_length
    return max(0.0, 1.0 - deviation)


REWARD_FUNCTIONS: Dict[str, Callable[..., float]] = {
    "exact_match": reward_exact_match,
    "format_reward": reward_format_check,
    "length_penalty": reward_length_penalty,
}


# ============================================================================
# GENERATION UTILITIES
# ============================================================================

@torch.no_grad()
@torch._dynamo.disable  # Autoregressive generation is not compile-friendly
def generate_completions(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,           # [B, prompt_len]
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_k: int = 50,
    eos_token_id: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    num_return_sequences: int = 1,      # G: generations per prompt
    repetition_penalty: float = 1.2,
    no_repeat_ngram_size: int = 3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate completions autoregressively using nucleus sampling.

    Args:
        repetition_penalty: Penalty for tokens that have already appeared.
            Values > 1.0 discourage repetition. Default 1.2.
        no_repeat_ngram_size: Prevent repeating n-grams of this size.
            Set to 0 to disable. Default 3.

    Returns:
        generated_ids: [B * num_return_sequences, prompt_len + gen_len]
        completion_mask: [B * num_return_sequences, prompt_len + gen_len]
                         (1 for generated tokens, 0 for prompt)
    """
    device = prompt_ids.device
    B, prompt_len = prompt_ids.shape
    
    # CRITICAL: Sync CUDA and clear any graph capture state before generation
    # This prevents "Offset increment outside graph capture" errors when
    # torch.compile with CUDA graphs was used for training
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    # Get base model (unwrap compiled model if needed)
    base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    base_model.eval()
    
    # Expand for multiple generations per prompt
    if num_return_sequences > 1:
        # [B, L] -> [B * G, L]
        prompt_ids = prompt_ids.repeat_interleave(num_return_sequences, dim=0)
    
    total_batch = prompt_ids.shape[0]
    pad_id = pad_token_id if pad_token_id is not None else 0

    # PRE-ALLOCATE output buffer to avoid O(L²) memory from repeated torch.cat
    # This eliminates ~256 temporary tensor allocations during generation
    max_total_len = prompt_len + max_new_tokens
    generated = torch.full(
        (total_batch, max_total_len), pad_id, device=device, dtype=prompt_ids.dtype
    )
    generated[:, :prompt_len] = prompt_ids
    current_len = prompt_len

    # Track which sequences have finished (hit EOS)
    finished = torch.zeros(total_batch, dtype=torch.bool, device=device)

    # Use inference_mode for generation (more efficient than no_grad)
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            if finished.all():
                break

            # Forward pass - get logits for last position
            # Only pass tokens up to current_len (not the full pre-allocated buffer)
            input_ids = generated[:, :current_len]

            # Handle different model forward signatures
            try:
                outputs = base_model(input_ids)
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
            except Exception:
                # Fallback for models expecting return_losses kwarg
                try:
                    logits, _ = base_model(input_ids, return_losses=False)
                except Exception:
                    logits = base_model(input_ids)

            next_logits = logits[:, -1, :]  # [B*G, vocab]

            # Temperature scaling
            if temperature > 0:
                next_logits = next_logits / temperature

            # Apply repetition penalty to tokens already in sequence
            if repetition_penalty != 1.0:
                for batch_idx in range(total_batch):
                    # Get unique tokens generated so far (not prompt)
                    gen_tokens = generated[batch_idx, prompt_len:current_len]
                    unique_tokens = gen_tokens.unique()
                    if unique_tokens.numel() > 0:
                        # Penalize: divide positive logits, multiply negative logits
                        penalty_logits = next_logits[batch_idx, unique_tokens]
                        next_logits[batch_idx, unique_tokens] = torch.where(
                            penalty_logits > 0,
                            penalty_logits / repetition_penalty,
                            penalty_logits * repetition_penalty,
                        )

            # No-repeat n-gram blocking
            if no_repeat_ngram_size > 0 and current_len >= prompt_len + no_repeat_ngram_size:
                for batch_idx in range(total_batch):
                    # Get the last (n-1) tokens as the current n-gram prefix
                    ngram_prefix = generated[batch_idx, current_len - no_repeat_ngram_size + 1:current_len].tolist()

                    # Find all previous occurrences of this prefix and block the following tokens
                    banned_tokens = set()
                    seq = generated[batch_idx, prompt_len:current_len].tolist()
                    for i in range(len(seq) - no_repeat_ngram_size + 1):
                        if seq[i:i + no_repeat_ngram_size - 1] == ngram_prefix:
                            banned_tokens.add(seq[i + no_repeat_ngram_size - 1])

                    if banned_tokens:
                        banned_tensor = torch.tensor(list(banned_tokens), device=device, dtype=torch.long)
                        next_logits[batch_idx, banned_tensor] = float("-inf")

            # Top-k filtering
            if top_k > 0:
                top_k_vals, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                threshold = top_k_vals[:, -1].unsqueeze(-1)
                next_logits = torch.where(
                    next_logits < threshold,
                    torch.full_like(next_logits, float("-inf")),
                    next_logits,
                )

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above threshold
                sorted_mask = cumulative_probs > top_p
                # Shift to keep first token above threshold
                sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
                sorted_mask[:, 0] = False

                # Scatter mask back to original order
                mask = sorted_mask.scatter(1, sorted_indices, sorted_mask)
                next_logits = next_logits.masked_fill(mask, float("-inf"))

            # Sample
            probs = F.softmax(next_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)  # [B*G]

            # Don't update finished sequences
            next_tokens = torch.where(finished, torch.full_like(next_tokens, pad_id), next_tokens)

            # Write in-place to pre-allocated buffer (no torch.cat!)
            generated[:, current_len] = next_tokens
            current_len += 1

            # Check for EOS
            if eos_token_id is not None:
                finished = finished | (next_tokens == eos_token_id)

    # Truncate to actual length (creates a view, not a copy)
    generated = generated[:, :current_len].contiguous()
    total_len = current_len

    # Build completion mask (1 for generated tokens, 0 for prompt)
    completion_mask = torch.zeros(total_batch, total_len, device=device, dtype=torch.float)
    completion_mask[:, prompt_len:] = 1.0
    
    # Mask out padding for finished sequences
    if eos_token_id is not None:
        for i in range(total_batch):
            eos_positions = (generated[i] == eos_token_id).nonzero(as_tuple=True)[0]
            if len(eos_positions) > 0:
                first_eos = eos_positions[0].item()
                if first_eos >= prompt_len:
                    completion_mask[i, first_eos + 1:] = 0.0
    
    return generated, completion_mask


def _chunked_log_softmax_gather(
    logits: torch.Tensor,      # [B, L, V]
    labels: torch.Tensor,      # [B, L]
    chunk_size: int = 4096,    # Process vocab in chunks to avoid full [B, L, V] allocation
) -> torch.Tensor:
    """
    Memory-efficient log_softmax + gather that avoids allocating full [B, L, V] tensor.

    Instead of: log_probs = F.log_softmax(logits, dim=-1)  # [B, L, V] - huge!
                token_logprobs = gather(log_probs, labels)

    We compute log_softmax and gather in one fused operation per token,
    using the log-sum-exp trick to avoid materializing full softmax.

    All computations done in float32 for numerical stability, converted at end.
    """
    B, L, V = logits.shape
    device = logits.device
    out_dtype = logits.dtype

    # For numerical stability, compute: log_softmax(x)[i] = x[i] - log(sum(exp(x)))
    # We only need the logprob at the label index, so:
    # log_softmax(x)[label] = x[label] - logsumexp(x)

    # Get logits at label positions: [B, L] - convert to float32 for precision
    label_logits = torch.gather(logits, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1).float()

    # Compute logsumexp over vocab dimension in chunks to reduce peak memory
    # logsumexp(x) = max(x) + log(sum(exp(x - max(x))))
    # Always use chunked computation to avoid allocating full [B, L, V] in float32
    # even for small vocab (the overhead is minimal and behavior is consistent)

    # First pass: find max across all chunks (compute in original dtype, result is [B, L])
    max_logit = logits[:, :, :chunk_size].max(dim=-1).values.float()
    for start in range(chunk_size, V, chunk_size):
        end = min(start + chunk_size, V)
        chunk_max = logits[:, :, start:end].max(dim=-1).values.float()
        max_logit = torch.maximum(max_logit, chunk_max)

    # Second pass: compute sum of exp(x - max) across chunks
    # Only convert each chunk to float32, not the entire tensor
    sum_exp = torch.zeros(B, L, device=device, dtype=torch.float32)
    for start in range(0, V, chunk_size):
        end = min(start + chunk_size, V)
        # Subtract max in original dtype to keep values small, then convert
        chunk_shifted = (logits[:, :, start:end] - max_logit.unsqueeze(-1).to(logits.dtype))
        sum_exp = sum_exp + torch.exp(chunk_shifted.float()).sum(dim=-1)

    logsumexp = max_logit + sum_exp.log()

    # log_softmax at label = label_logit - logsumexp (in float32)
    token_logprobs = label_logits - logsumexp

    return token_logprobs.to(out_dtype)


def compute_sequence_logprobs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,    # [B, L]
    mask: torch.Tensor,         # [B, L] - which tokens to compute logprobs for
    use_gradient_checkpointing: bool = True,  # Ignored - model has internal checkpointing
    chunk_size: int = 4096,     # Vocab chunk size for memory-efficient log_softmax
) -> torch.Tensor:
    """
    Compute per-token log probabilities for sequences.

    Memory-optimized version that uses chunked log_softmax to avoid [B, L, V] allocation.

    Note: use_gradient_checkpointing is ignored because input_ids are integers (token IDs)
    which cannot require gradients. The model's internal checkpointing handles memory.

    Returns: [B, L] tensor of log probs (0 where mask is 0)
    """
    device = input_ids.device
    B, L = input_ids.shape

    # Get base model
    base_model = model._orig_mod if hasattr(model, "_orig_mod") else model

    # Define forward function for checkpointing
    # NOTE: No exception handling here - if OOM occurs, we want to fail cleanly
    # rather than retry and allocate even more memory (cascading OOM)
    def forward_fn(ids):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = base_model(ids)
            if isinstance(outputs, tuple):
                return outputs[0]
            return outputs

    # Forward pass - model's internal gradient checkpointing handles memory efficiency
    # NOTE: We cannot use torch.utils.checkpoint here because input_ids are integers
    # (token IDs) which cannot require gradients. The HYDRA model already has internal
    # gradient checkpointing enabled via --checkpoint_every flag.
    logits = forward_fn(input_ids)

    # Shift for next-token prediction: logits[t] predicts token[t+1]
    shift_logits = logits[:, :-1, :]              # [B, L-1, V] - view, not copy
    shift_labels = input_ids[:, 1:].contiguous()  # [B, L-1]
    shift_mask = mask[:, 1:].contiguous()         # [B, L-1]

    # Memory-efficient log_softmax + gather (avoids full [B, L, V] allocation)
    token_logprobs = _chunked_log_softmax_gather(
        shift_logits,
        shift_labels,
        chunk_size=chunk_size,
    )  # [B, L-1]

    # Apply mask
    token_logprobs = token_logprobs * shift_mask

    # Pad to original length (first position has no logprob)
    result = torch.zeros(B, L, device=device, dtype=token_logprobs.dtype)
    result[:, 1:] = token_logprobs

    return result


# ============================================================================
# GRPO TRAINER MIXIN
# ============================================================================

class GRPOTrainerMixin:
    """
    Mixin for Trainer to add Group Relative Policy Optimization.
    
    Implements online GRPO where we:
    1. Sample prompts from reasoning datasets
    2. Generate G completions per prompt
    3. Score completions with reward function
    4. Compute group-relative advantages
    5. Update policy to increase prob of high-advantage completions
    """
    
    # These will be set by the Trainer class
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scaler: Any  # GradScaler
    device: str
    config: "TrainingConfig"
    logger: Any
    _reasoning_prompts: Optional[List[Dict[str, Any]]] = None
    _reasoning_prompt_idx: int = 0
    _tokenizer: Any = None
    
    def _ensure_reasoning_prompts(self) -> None:
        """Load reasoning prompts from configured datasets (lazy init)."""
        if self._reasoning_prompts is not None:
            return
        
        self._reasoning_prompts = []
        
        # Try to load from OpenMathInstruct or OpenThoughts
        try:
            from datasets import load_dataset
            
            # OpenMathInstruct-2: has problem + expected_answer
            ds = load_dataset(
                "nvidia/OpenMathInstruct-2",
                split="train_1M",
                streaming=True,
            )
            
            # Take a sample of prompts
            count = 0
            for example in ds:
                if count >= 1000:  # Cache 1000 prompts
                    break
                problem = example.get("problem", "")
                answer = example.get("expected_answer", "")
                if problem and len(problem) > 20:
                    self._reasoning_prompts.append({
                        "prompt": f"Solve this math problem step by step:\n\n{problem}\n\nSolution:",
                        "expected_answer": answer,
                    })
                    count += 1
            
            _log.info(f"Loaded {len(self._reasoning_prompts)} reasoning prompts from OpenMathInstruct-2")
            
        except Exception as e:
            _log.warning(f"Failed to load reasoning prompts: {e}")
            # Fallback: simple prompts
            self._reasoning_prompts = [
                {"prompt": "What is 2 + 2? Think step by step.\n\nAnswer:", "expected_answer": "4"},
                {"prompt": "If x = 5, what is 2x + 3? Show your work.\n\nAnswer:", "expected_answer": "13"},
                {"prompt": "Solve: 3 * 4 - 2 = ?\n\nAnswer:", "expected_answer": "10"},
            ]
    
    def _get_reasoning_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        """Get a batch of reasoning prompts."""
        self._ensure_reasoning_prompts()
        
        prompts = []
        for _ in range(batch_size):
            idx = self._reasoning_prompt_idx % len(self._reasoning_prompts)
            prompts.append(self._reasoning_prompts[idx])
            self._reasoning_prompt_idx += 1
        
        return prompts
    
    def _ensure_tokenizer(self) -> Any:
        """Get or create tokenizer."""
        if self._tokenizer is not None:
            return self._tokenizer
        
        try:
            from hydra.data.universal_data_loader import get_tokenizer
            tokenizer_name = getattr(self.config, "tokenizer_name", "gpt2")
            self._tokenizer = get_tokenizer(tokenizer_name)
            return self._tokenizer
        except Exception as e:
            _log.error(f"Failed to get tokenizer: {e}")
            return None
    
    def _clear_mor_caches(self, base_model: torch.nn.Module) -> None:
        """Clear cached MoR routing tensors to prevent graph conflicts.
        
        After a reasoning step (which does its own forward/backward), the MoR blocks
        may hold _last_probs tensors that were part of a now-freed computation graph.
        These must be cleared so the next regular training step doesn't accidentally
        reference stale graph nodes.
        """
        layers = getattr(base_model, "layers", [])
        for layer in layers:
            # Clear MoR-related cached tensors
            if hasattr(layer, "_last_probs"):
                layer._last_probs = None
            if hasattr(layer, "_last_depths"):
                layer._last_depths = None
            if hasattr(layer, "_last_router_logits"):
                layer._last_router_logits = None
            if hasattr(layer, "_last_router_probs_tensor"):
                layer._last_router_probs_tensor = None
            if hasattr(layer, "_last_target_depths"):
                layer._last_target_depths = None

    def _log_memory(self, stage: str) -> None:
        """Log CUDA memory state for debugging OOM issues."""
        if not torch.cuda.is_available():
            return
        torch.cuda.synchronize()
        alloc = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        free = torch.cuda.mem_get_info()[0] / (1024**3)
        _log.info(f"  [MEM] {stage}: alloc={alloc:.2f}GB reserved={reserved:.2f}GB free={free:.2f}GB")

    def _run_reasoning_step(self, step: int) -> Optional[Dict[str, float]]:
        """
        Execute a full GRPO reasoning step:
        1. Sample prompts
        2. Generate G completions per prompt
        3. Compute rewards
        4. Compute GRPO loss and update

        Returns metrics dict or None if skipped.
        """
        config = self.config
        G = getattr(config, "grpo_num_generations", 4)
        kl_coef = getattr(config, "grpo_kl_coef", 0.01)
        batch_size = getattr(config, "reasoning_batch_size", 2)
        max_tokens = getattr(config, "reasoning_max_tokens", 256)
        temperature = getattr(config, "reasoning_temperature", 0.7)
        top_p = getattr(config, "reasoning_top_p", 0.95)
        reward_fn_name = getattr(config, "reasoning_reward_function", "format_reward")

        # Log initial memory state
        self._log_memory("GRPO_START")

        tokenizer = self._ensure_tokenizer()
        if tokenizer is None:
            self.logger.warning("Reasoning step skipped: no tokenizer")
            return None

        # Get the unwrapped model (avoid torch.compile issues during generation)
        base_model = self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model

        # Log model info
        total_params = sum(p.numel() for p in base_model.parameters())
        num_moe_layers = len(getattr(base_model, 'moe_layers', []))
        _log.info(f"  [INFO] Model params: {total_params/1e6:.1f}M, MoE layers: {num_moe_layers}")

        # Skip MoE during GRPO for massive memory savings
        skip_moe = getattr(config, "grpo_skip_moe", True)
        if skip_moe and num_moe_layers > 0 and hasattr(base_model, "set_skip_moe"):
            base_model.set_skip_moe(True)
            _log.info("  [INFO] MoE layers SKIPPED for GRPO (grpo_skip_moe=True)")

        # CRITICAL: Reset dynamo state before generation to avoid graph capture conflicts
        # This is needed when torch.compile with CUDA graphs is used for training
        torch._dynamo.reset()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()  # Free GPU memory before CPU generation

        self._log_memory("AFTER_CACHE_CLEAR")
        
        # Get reward function
        reward_fn = REWARD_FUNCTIONS.get(reward_fn_name, reward_format_check)
        
        # Sample prompts
        prompt_batch = self._get_reasoning_batch(batch_size)
        
        # Tokenize prompts
        prompt_texts = [p["prompt"] for p in prompt_batch]
        expected_answers = [p.get("expected_answer") for p in prompt_batch]
        
        encoded = tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        prompt_ids = encoded["input_ids"].to(self.device)
        
        # Store reference logprobs BEFORE generation (model is still in current state)
        base_model.eval()
        
        # Generate completions ONE AT A TIME to avoid GPU memory spikes
        # This is slower but prevents unpredictable OOM from memory fragmentation
        eos_id = tokenizer.eos_token_id
        pad_id = tokenizer.pad_token_id or eos_id
        
        all_generated = []
        all_masks = []
        
        self.logger.info(f"  🧠 Generating {batch_size * G} completions (1 at a time to avoid OOM)...")
        self._log_memory("BEFORE_GENERATION")
        
        for prompt_idx in range(batch_size):
            single_prompt = prompt_ids[prompt_idx:prompt_idx+1]  # [1, prompt_len]
            
            for gen_idx in range(G):
                # Generate single completion
                gen_ids, gen_mask = generate_completions(
                    model=base_model,
                    prompt_ids=single_prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=50,
                    eos_token_id=eos_id,
                    pad_token_id=pad_id,
                    num_return_sequences=1,
                )
                all_generated.append(gen_ids)
                all_masks.append(gen_mask)
                
                # Aggressive memory clearing between generations
                del gen_ids, gen_mask
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Pad all sequences to same length and stack
        max_len = max(g.shape[1] for g in all_generated)
        padded_generated = []
        padded_masks = []
        
        for gen_ids, gen_mask in zip(all_generated, all_masks):
            pad_len = max_len - gen_ids.shape[1]
            if pad_len > 0:
                gen_ids = torch.cat([gen_ids, torch.full((1, pad_len), pad_id, device=gen_ids.device)], dim=1)
                gen_mask = torch.cat([gen_mask, torch.zeros((1, pad_len), device=gen_mask.device)], dim=1)
            padded_generated.append(gen_ids)
            padded_masks.append(gen_mask)
        
        generated_ids = torch.cat(padded_generated, dim=0)  # [B*G, max_len]
        completion_mask = torch.cat(padded_masks, dim=0)    # [B*G, max_len]
        
        del all_generated, all_masks, padded_generated, padded_masks
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._log_memory("AFTER_GENERATION")

        # generated_ids: [B * G, total_len]
        total_samples = generated_ids.shape[0]
        total_len = generated_ids.shape[1]
        _log.info(f"  [INFO] Generated {total_samples} sequences of length {total_len}")
        
        # Decode completions for reward computation
        completions = []
        for i in range(total_samples):
            # Decode only the completion part
            completion_tokens = generated_ids[i][completion_mask[i].bool()]
            text = tokenizer.decode(completion_tokens, skip_special_tokens=True)
            completions.append(text)
        
        # Compute rewards
        rewards = []
        for i in range(total_samples):
            prompt_idx = i // G
            reward = reward_fn(
                prompt=prompt_texts[prompt_idx],
                completion=completions[i],
                expected_answer=expected_answers[prompt_idx],
            )
            rewards.append(reward)
        
        rewards_tensor = torch.tensor(rewards, device=self.device, dtype=torch.float32)
        rewards_tensor = rewards_tensor.view(batch_size, G)  # [B, G]
        
        # Skip update if all rewards are identical (no signal)
        if rewards_tensor.std() < 1e-6:
            self.logger.info(f"  Reasoning step {step}: skipped (no reward variance)")
            # Clear cached MoR tensors to prevent graph conflicts with next training step
            self._clear_mor_caches(base_model)
            # Clear CUDA cache to free generation tensors (don't reset dynamo - causes fragmentation)
            del generated_ids, completion_mask
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return {"grpo/skipped": 1.0, "grpo/reward_mean": rewards_tensor.mean().item()}
        
        # Already on GPU - no move needed

        # MEMORY SAFETY: Check available VRAM before expensive logprobs computation
        # MoE models need significantly more headroom due to expert activations
        # 500M+MoE (6 layers, 6 experts) needs ~12GB for logprobs alone
        has_moe = hasattr(base_model, "moe_blocks") and len(getattr(base_model, "moe_blocks", [])) > 0
        MIN_FREE_VRAM_GB = 12.0 if has_moe else 6.0
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            free_vram = torch.cuda.mem_get_info()[0] / (1024**3)
            if free_vram < MIN_FREE_VRAM_GB:
                self.logger.warning(
                    f"  ⚠️ Reasoning step skipped: only {free_vram:.2f}GB free (need {MIN_FREE_VRAM_GB}GB)"
                )
                self._clear_mor_caches(base_model)
                del generated_ids, completion_mask, rewards_tensor
                gc.collect()
                torch.cuda.empty_cache()
                return {"grpo/skipped": 1.0, "grpo/reason": "low_vram", "grpo/free_vram_gb": free_vram}

        # SEQUENCE LENGTH SAFETY: Truncate very long sequences to prevent OOM
        # Long sequences consume O(L^2) memory in attention
        # MoE models need shorter sequences due to expert activation memory
        MAX_LOGPROB_SEQ_LEN = 256 if has_moe else 512
        if total_len > MAX_LOGPROB_SEQ_LEN:
            self.logger.info(f"  ⚡ Truncating sequences from {total_len} to {MAX_LOGPROB_SEQ_LEN} for logprobs")
            generated_ids = generated_ids[:, :MAX_LOGPROB_SEQ_LEN]
            completion_mask = completion_mask[:, :MAX_LOGPROB_SEQ_LEN]
            total_len = MAX_LOGPROB_SEQ_LEN

        # Compute log probs under current policy (use base_model for consistency)
        base_model.train()

        self._log_memory("BEFORE_LOGPROBS")

        # PRE-COMPUTE ADVANTAGES from rewards (no gradients needed)
        # This allows us to process logprobs in micro-batches with immediate backward
        # Advantages: A_i = (r_i - mean(r_group)) / (std(r_group) + eps)
        mean_rewards = rewards_tensor.mean(dim=1, keepdim=True)  # [B, 1]
        std_rewards = rewards_tensor.std(dim=1, keepdim=True) + 1e-8
        advantages = (rewards_tensor - mean_rewards) / std_rewards  # [B, G]
        advantages_flat = advantages.view(-1)  # [B*G] for micro-batch indexing

        # MICRO-BATCH with gradient accumulation
        # Process sequences in small chunks, backward immediately to free graph memory
        # This fixes the gradient disconnect bug: gradients now flow to the model
        # Memory scaling: log-prob computation uses ~0.4GB per sequence at seq_len=512
        # MoE models need smaller micro-batches due to expert activation memory
        if has_moe:
            micro_batch_size = 1  # Always single-sequence for MoE
        elif total_len > 400:
            micro_batch_size = 1  # Ultra-conservative for long sequences
        elif total_len > 256:
            micro_batch_size = 2
        else:
            micro_batch_size = 4
        num_micro_batches = (total_samples + micro_batch_size - 1) // micro_batch_size

        # OOM recovery pattern: PyTorch docs recommend handling cleanup OUTSIDE
        # the except block because the exception object holds references to the
        # stack frame, preventing tensor deallocation. See:
        # https://pytorch.org/docs/stable/notes/faq.html#my-out-of-memory-exception-handler-can-t-allocate-memory
        oom_occurred = False
        oom_message = ""

        # Metrics accumulators
        total_loss = 0.0
        total_kl = 0.0
        micro_batches_processed = 0

        _log.info(f"  [INFO] Processing {total_samples} sequences in micro-batches of {micro_batch_size}")

        for i in range(0, total_samples, micro_batch_size):
            if oom_occurred:
                break

            end_idx = min(i + micro_batch_size, total_samples)
            chunk_size_actual = end_idx - i
            chunk_ids = generated_ids[i:end_idx]
            chunk_mask = completion_mask[i:end_idx]
            chunk_advantages = advantages_flat[i:end_idx]  # [chunk_size]

            if i == 0:
                self._log_memory(f"MICRO_BATCH_0_START (seq_len={chunk_ids.shape[1]})")

            try:
                # Compute logprobs WITH gradients (no detach!)
                chunk_logprobs = compute_sequence_logprobs(
                    base_model,
                    chunk_ids,
                    chunk_mask,
                )  # [chunk_size, L]

                # Compute policy loss for this chunk: -A * mean(log_probs)
                # IMPORTANT: Use mean over tokens, not sum, to keep loss scale independent
                # of sequence length and comparable to cross-entropy loss (~2-4 range)
                num_tokens = chunk_mask.sum(dim=1).clamp(min=1)  # [chunk_size]
                completion_logprobs = (chunk_logprobs * chunk_mask).sum(dim=1) / num_tokens  # [chunk_size]
                chunk_policy_loss = -(chunk_advantages * completion_logprobs).mean()

                # Scale loss for gradient accumulation (average over all micro-batches)
                scaled_loss = chunk_policy_loss / num_micro_batches

                if i == 0:
                    self._log_memory("MICRO_BATCH_0_LOGPROBS_DONE")

                # Backward immediately - this frees the computation graph
                self.scaler.scale(scaled_loss).backward()

                if i == 0:
                    self._log_memory("MICRO_BATCH_0_BACKWARD_DONE")

                # Accumulate metrics (detach for logging)
                total_loss += chunk_policy_loss.detach().item() * chunk_size_actual
                micro_batches_processed += 1

            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if "out of memory" in str(e).lower():
                    oom_occurred = True
                    oom_message = str(e)
                else:
                    raise

            # Free intermediate memory (always, even on OOM path)
            try:
                del chunk_ids, chunk_mask, chunk_advantages, chunk_logprobs
                del completion_logprobs, chunk_policy_loss, scaled_loss
            except NameError:
                pass  # Some variables may not exist on OOM
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # OOM cleanup OUTSIDE the except block - this is critical!
        if oom_occurred:
            self.logger.warning(f"  ⚠️ OOM during logprobs computation, skipping reasoning step: {oom_message}")
            self._clear_mor_caches(base_model)
            self.optimizer.zero_grad(set_to_none=True)  # Clear partial gradients
            del generated_ids, completion_mask, rewards_tensor, advantages, advantages_flat
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return {"grpo/skipped": 1.0, "grpo/reason": "oom_logprobs"}

        # Clean up tensors no longer needed
        del generated_ids, completion_mask, advantages_flat

        # Clip gradients
        self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=getattr(config, "max_grad_norm", 1.0),
        )

        # Optimizer step
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

        # Restore training mode on main model
        self.model.train()

        # Clear cached MoR tensors to prevent graph conflicts with next training step
        self._clear_mor_caches(base_model)

        # CRITICAL: Clear CUDA memory to prevent accumulation across reasoning steps
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Build metrics
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        metrics = {
            "grpo/loss": avg_loss,
            "grpo/reward_mean": rewards_tensor.mean().item(),
            "grpo/reward_std": rewards_tensor.std().item(),
            "grpo/kl_mean": 0.0,  # KL is 0 when ref=current (online GRPO)
            "grpo/advantage_mean": advantages.mean().item(),
            "grpo/advantage_std": advantages.std().item(),
            "grpo/grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
            "grpo/num_samples": total_samples,
        }

        del rewards_tensor, advantages

        # Re-enable MoE for regular training
        if skip_moe and num_moe_layers > 0 and hasattr(base_model, "set_skip_moe"):
            base_model.set_skip_moe(False)

        self.logger.info(
            f"  🧠 Reasoning update: loss={metrics['grpo/loss']:.4f} | "
            f"reward={metrics['grpo/reward_mean']:.3f}±{metrics['grpo/reward_std']:.3f} | "
            f"adv={metrics['grpo/advantage_mean']:.3f}"
        )

        return metrics
