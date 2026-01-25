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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import torch
import torch._dynamo
import torch.nn.functional as F

if TYPE_CHECKING:
    from .config import TrainingConfig

_log = logging.getLogger("HYDRA")


@dataclass
class ReasoningConfig:
    """Configuration for System 2 / Reasoning training."""
    enabled: bool = False
    
    # GRPO settings
    num_generations: int = 4      # G: Number of samples to generate per prompt
    kl_coef: float = 0.01         # Beta: KL penalty coefficient
    clip_epsilon: float = 0.2     # PPO-style clipping epsilon (unused in simple GRPO)
    max_completion_length: int = 512  # Max tokens to generate per completion
    temperature: float = 0.7      # Sampling temperature for diversity
    top_p: float = 0.95           # Nucleus sampling threshold
    
    # Execution parameters
    reasoning_interval: int = 100       # Run reasoning step every N training steps
    reasoning_batch_size: int = 2       # Prompts per reasoning step (total samples = batch * G)
    reasoning_grad_accum: int = 1       # Gradient accumulation for reasoning updates
    
    # "Thinking" token definitions (for process masking)
    start_thought_token_id: Optional[int] = None
    end_thought_token_id: Optional[int] = None
    
    # Reward configuration
    reward_function: str = "format_reward"  # 'exact_match', 'format_reward', 'length_penalty'


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
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate completions autoregressively using nucleus sampling.
    
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
    generated = prompt_ids.clone()
    
    # Track which sequences have finished (hit EOS)
    finished = torch.zeros(total_batch, dtype=torch.bool, device=device)
    
    # Use inference_mode for generation (more efficient than no_grad)
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            if finished.all():
                break
            
            # Forward pass - get logits for last position
            # Handle different model forward signatures
            try:
                outputs = base_model(generated)
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
            except Exception:
                # Fallback for models expecting return_losses kwarg
                try:
                    logits, _ = base_model(generated, return_losses=False)
                except Exception:
                    logits = base_model(generated)
            
            next_logits = logits[:, -1, :]  # [B*G, vocab]
            
            # Temperature scaling
            if temperature > 0:
                next_logits = next_logits / temperature
            
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
            next_tokens = torch.multinomial(probs, num_samples=1)  # [B*G, 1]
            
            # Don't update finished sequences
            next_tokens = torch.where(
                finished.unsqueeze(-1),
                torch.full_like(next_tokens, pad_token_id or 0),
                next_tokens,
            )
            
            generated = torch.cat([generated, next_tokens], dim=1)
            
            # Check for EOS
            if eos_token_id is not None:
                finished = finished | (next_tokens.squeeze(-1) == eos_token_id)
    
    # Build completion mask (1 for generated tokens, 0 for prompt)
    total_len = generated.shape[1]
    completion_mask = torch.zeros_like(generated, dtype=torch.float)
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
    if V <= chunk_size:
        # Small vocab - just compute directly in float32
        logsumexp = torch.logsumexp(logits.float(), dim=-1)  # [B, L]
    else:
        # Large vocab - chunked computation in float32
        # First pass: find max across all chunks
        max_logit = logits[:, :, :chunk_size].float().max(dim=-1).values
        for start in range(chunk_size, V, chunk_size):
            end = min(start + chunk_size, V)
            chunk_max = logits[:, :, start:end].float().max(dim=-1).values
            max_logit = torch.maximum(max_logit, chunk_max)

        # Second pass: compute sum of exp(x - max) across chunks
        sum_exp = torch.zeros(B, L, device=device, dtype=torch.float32)
        for start in range(0, V, chunk_size):
            end = min(start + chunk_size, V)
            chunk = logits[:, :, start:end].float()
            sum_exp = sum_exp + torch.exp(chunk - max_logit.unsqueeze(-1)).sum(dim=-1)

        logsumexp = max_logit + sum_exp.log()

    # log_softmax at label = label_logit - logsumexp (in float32)
    token_logprobs = label_logits - logsumexp

    return token_logprobs.to(out_dtype)


def compute_sequence_logprobs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,    # [B, L]
    mask: torch.Tensor,         # [B, L] - which tokens to compute logprobs for
    use_gradient_checkpointing: bool = True,
    chunk_size: int = 4096,     # Vocab chunk size for memory-efficient log_softmax
) -> torch.Tensor:
    """
    Compute per-token log probabilities for sequences.

    Memory-optimized version that:
    1. Uses gradient checkpointing to reduce activation memory
    2. Uses chunked log_softmax to avoid [B, L, V] allocation

    Returns: [B, L] tensor of log probs (0 where mask is 0)
    """
    device = input_ids.device
    B, L = input_ids.shape

    # Get base model
    base_model = model._orig_mod if hasattr(model, "_orig_mod") else model

    # Define forward function for checkpointing
    def forward_fn(ids):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            try:
                outputs = base_model(ids)
                if isinstance(outputs, tuple):
                    return outputs[0]
                return outputs
            except Exception:
                try:
                    logits, _ = base_model(ids, return_losses=False)
                    return logits
                except Exception:
                    return base_model(ids)

    # Forward pass with optional gradient checkpointing
    if use_gradient_checkpointing and input_ids.requires_grad:
        from torch.utils.checkpoint import checkpoint
        logits = checkpoint(forward_fn, input_ids, use_reentrant=False)
    else:
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
    
    def compute_grpo_loss(
        self,
        model_logprobs: torch.Tensor,     # [B, G, SeqLen]
        ref_logprobs: torch.Tensor,       # [B, G, SeqLen]
        rewards: torch.Tensor,            # [B, G]
        mask: torch.Tensor,               # [B, G, SeqLen] - 1 for generated tokens
        kl_coef: float = 0.01,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Computes GRPO loss (simplified DeepSeek-R1 style).
        
        L = -E[ A_i * log π(y|x) ] + β * KL(π || π_ref)
        
        Where Advantage A_i is computed relative to the GROUP:
        A_i = (r_i - mean(r_group)) / (std(r_group) + eps)
        """
        B, G, S = model_logprobs.shape
        
        # 1. Group-Relative Advantages
        mean_rewards = rewards.mean(dim=1, keepdim=True)  # [B, 1]
        std_rewards = rewards.std(dim=1, keepdim=True) + 1e-8
        advantages = (rewards - mean_rewards) / std_rewards  # [B, G]
        
        # 2. KL Divergence (token-level)
        # Approximate KL = log(π) - log(π_ref)
        token_kl = (model_logprobs - ref_logprobs)  # [B, G, S]
        
        # 3. Policy Loss
        # Sum log probs over completion tokens
        completion_logprobs = (model_logprobs * mask).sum(dim=2)  # [B, G]
        
        # Weighted by advantage: -A * log π(completion)
        policy_loss = -(advantages * completion_logprobs)  # [B, G]
        
        # 4. KL Penalty
        kl_per_sample = (token_kl * mask).sum(dim=2)  # [B, G]
        kl_loss = kl_coef * kl_per_sample
        
        # Total loss (mean over batch and group)
        total_loss = (policy_loss + kl_loss).mean()
        
        # Metrics (resolve to Python floats outside compiled regions)
        metrics = {
            "grpo/loss": total_loss.detach().item(),
            "grpo/reward_mean": rewards.mean().item(),
            "grpo/reward_std": rewards.std().item(),
            "grpo/kl_mean": token_kl.mean().item(),
            "grpo/advantage_mean": advantages.mean().item(),
            "grpo/advantage_std": advantages.std().item(),
        }
        
        return total_loss, metrics

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

    @torch.no_grad()
    def _snapshot_model_state(self) -> Dict[str, torch.Tensor]:
        """Create lightweight copy of model params for reference policy."""
        base_model = self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model
        return {k: v.clone() for k, v in base_model.state_dict().items()}
    
    def _restore_model_state(self, state: Dict[str, torch.Tensor]) -> None:
        """Restore model from snapshot (for reference policy)."""
        base_model = self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model
        base_model.load_state_dict(state)
    
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
        
        tokenizer = self._ensure_tokenizer()
        if tokenizer is None:
            self.logger.warning("Reasoning step skipped: no tokenizer")
            return None
        
        # Get the unwrapped model (avoid torch.compile issues during generation)
        base_model = self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model
        
        # CRITICAL: Reset dynamo state before generation to avoid graph capture conflicts
        # This is needed when torch.compile with CUDA graphs is used for training
        torch._dynamo.reset()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
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
        
        # Generate G completions per prompt (using unwrapped model for generation)
        eos_id = tokenizer.eos_token_id
        pad_id = tokenizer.pad_token_id or eos_id
        
        generated_ids, completion_mask = generate_completions(
            model=base_model,
            prompt_ids=prompt_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=50,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
            num_return_sequences=G,
        )
        
        # generated_ids: [B * G, total_len]
        total_samples = generated_ids.shape[0]
        total_len = generated_ids.shape[1]
        
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
            # Reset dynamo state to clear any compiled graph remnants
            torch._dynamo.reset()
            # Clear CUDA cache to free generation tensors
            del generated_ids, completion_mask
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return {"grpo/skipped": 1.0, "grpo/reward_mean": rewards_tensor.mean().item()}
        
        # Compute log probs under current policy (use base_model for consistency)
        base_model.train()
        
        # Reshape for batch processing: [B*G, L] -> compute logprobs -> [B, G, L]
        current_logprobs = compute_sequence_logprobs(
            base_model,
            generated_ids,
            completion_mask,
        )  # [B*G, L]
        current_logprobs = current_logprobs.view(batch_size, G, total_len)
        
        # For simple online GRPO, use current logprobs as reference (KL ≈ 0)
        # In a full implementation, you'd snapshot the model before generation
        ref_logprobs = current_logprobs.detach()
        
        # Reshape mask
        completion_mask_3d = completion_mask.view(batch_size, G, total_len)
        
        # Compute GRPO loss
        loss, metrics = self.compute_grpo_loss(
            model_logprobs=current_logprobs,
            ref_logprobs=ref_logprobs,
            rewards=rewards_tensor,
            mask=completion_mask_3d,
            kl_coef=kl_coef,
        )
        
        # Backward pass (use self.model to ensure compiled graph gets gradients)
        self.scaler.scale(loss).backward()
        
        # Explicitly delete large tensors to free memory before optimizer step
        del current_logprobs, ref_logprobs, completion_mask_3d, loss
        del generated_ids, completion_mask, rewards_tensor
        
        # Optional: clip gradients
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
        
        # Reset dynamo state to clear any compiled graph remnants after reasoning step
        torch._dynamo.reset()
        
        # CRITICAL: Clear CUDA memory to prevent accumulation across reasoning steps
        # The logprobs computation creates large intermediate tensors that can fragment memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Log
        metrics["grpo/grad_norm"] = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
        metrics["grpo/num_samples"] = total_samples
        
        self.logger.info(
            f"  🧠 Reasoning update: loss={metrics['grpo/loss']:.4f} | "
            f"reward={metrics['grpo/reward_mean']:.3f}±{metrics['grpo/reward_std']:.3f} | "
            f"adv={metrics['grpo/advantage_mean']:.3f}"
        )
        
        return metrics


# ============================================================================
# THOUGHT BOUNDARY DETECTION
# ============================================================================

def detect_thought_boundaries(
    input_ids: torch.Tensor, 
    start_id: int, 
    end_id: int
) -> torch.Tensor:
    """
    Returns a mask where 1 = inside a thought block, 0 = outside.
    Useful for 'Thought Masking' (blocking gradients on thought tokens 
    to prevent mimicking human errors, or vice versa).
    """
    B, L = input_ids.shape
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    
    # Simple state machine for vectorized thinking detection
    is_thinking = torch.zeros((B,), dtype=torch.bool, device=input_ids.device)
    
    for i in range(L):
        token = input_ids[:, i]
        # Entering thought
        starts = (token == start_id)
        is_thinking = is_thinking | starts
        
        mask[:, i] = is_thinking
        
        # Exiting thought
        ends = (token == end_id)
        is_thinking = is_thinking & (~ends)
        
    return mask
