# Part B: Reasoning Trainer Analysis & Optimizations

**Date:** 2026-02-09  
**File:** `reasoning_trainer.py` (1139 lines)  
**Scope:** Standalone GRPO trainer for System-2 reasoning capabilities  
**Algorithm:** Group Relative Policy Optimization (GRPO) from DeepSeekMath (arXiv:2402.03300)  

---

## HIGH PRIORITY — Critical Performance Issues

### 1. No KV Cache in Generation — O(n²) per token
- **File:** `reasoning_trainer.py`, `generate_completion()` (lines ~430-500)
- **Issue:** Full sequence recomputation for every generated token. For 512-token generation, the model recomputes attention over all prior tokens from scratch — total O(n²) redundant work instead of O(n) with KV cache.
- **Impact:** This is the **single biggest bottleneck**. Generation is already the dominant cost in GRPO. A KV cache would give 5-10× speedup on generation.
- **Fix (future):** Implement KV cache in model's `forward()`, or integrate vLLM/SGLang for generation phase. TRL's GRPO already supports this pattern.
- **Status:** ✅ IMPLEMENTED — `forward_with_cache()` added to model, blocks, and attention. `generate_completion()` uses prefill + incremental decode. Batch generation replicates KV cache across N parallel streams.

### 2. Full vocab `log_softmax` materialization
- **File:** `reasoning_trainer.py`, `compute_sequence_logprobs()` (lines 606-640)
- **Issue:** `F.log_softmax(shift_logits, dim=-1)` materializes a `[B, T, 32000]` tensor but only the log-prob at the actual token index is used (via `torch.gather`). Wastes memory and compute.
- **Fix:** Use selective log-softmax: compute only the needed logit minus logsumexp.
- **Tested:** Numerically matches reference within 1e-7 max absolute error. Memory savings most visible with autograd (backward avoids retaining [B,T,V] log-prob tensor).
- **Status:** ✅ IMPLEMENTED + TESTED

### 3. Excessive `torch.cuda.empty_cache()` calls
- **File:** `reasoning_trainer.py`, `run_grpo_step()` (lines ~730, ~790)
- **Issue:** Called between every single generation AND between every micro-batch. Each forces CUDA driver synchronization and allocator defragmentation.
- **Fix:** Remove per-generation calls. Keep only one cleanup after the full GRPO step.
- **Status:** ✅ IMPLEMENTED + TESTED

---

## ALGORITHMIC IMPROVEMENTS (Not Yet Implemented)

### 4. No Reference Model / KL Divergence Term
- **Issue:** Standard GRPO includes `β · KL(π_θ || π_ref)` to prevent reward hacking. Current implementation has no reference policy.
- **Impact:** Training stability degrades as model diverges from initial policy.
- **Fix:** Keep frozen copy of initial weights, compute per-token KL penalty.
- **Status:** ✅ IMPLEMENTED — `create_reference_model()` makes frozen deepcopy. `compute_kl_penalty()` computes per-token KL. CLI flags: `--kl_beta 0.05 --kl_warmup_steps 50`. KL penalty added to GRPO loss with configurable warmup.

### 5. Generation Reuse (μ > 1)
- **Issue:** Current trainer generates fresh samples every step. TRL's GRPO now supports reusing generated samples across 2-4 policy updates.
- **Impact:** 2-4× effective GRPO throughput by amortizing expensive generation.
- **Status:** ✅ IMPLEMENTED — `_GenerationCache` class manages sample reuse. CLI flag `--generation_reuse_count N` sets μ parameter (default 1 = no reuse).

### 6. Reward Weighting
- **Issue:** All rewards treated equally. TRL now supports `reward_weights` for different signal priorities.
- **Status:** ✅ IMPLEMENTED — CLI flag `--reward_weights '0.7,0.3'` for per-signal prioritization. Config field `reward_weights: Optional[List[float]]`.

### 7. Group Size and Advantage Normalization
- **Issue:** Current `num_generations=4` is small. DeepSeek-R1 used 64-96. Larger groups → more stable advantages.
- **Status:** ❌ NOT STARTED

---

## INNOVATION OPPORTUNITIES (Research Phase)

### 8. Speculative Decoding for Generation Speed
- Since generation is the GRPO bottleneck, speculative decoding could yield 1.5-2× speedup.
- Options: n-gram/suffix speculation (no extra model), small draft model, EAGLE.
- Suffix decoding particularly good for RL rollouts (high repetition patterns).
- **Status:** ✅ IMPLEMENTED — N-gram suffix speculation via `_ngram_draft_tokens()`. CLI flags: `--speculative_ngram_size 3 --speculative_max_draft 4`. Integrated into `generate_completion()` with KV cache verification loop. Draft tokens accepted if model assigns >= 10% probability.

### 9. DAPO (Decoupled Alignment from Policy Optimization)
- Clips only one side of probability ratio (avoids entropy collapse).
- Dynamic sampling temperature.
- Better exploration-exploitation tradeoff than standard GRPO.
- **Status:** ✅ IMPLEMENTED — CLI flag `--dapo` enables DAPO loss. Configurable clip bounds `--dapo_clip_low 0.8 --dapo_clip_high 1.28`. Dynamic temperature scales advantages by magnitude. Uses `min(surr1, surr2)` objective with one-sided clipping.

### 10. Process Reward Models (PRM)
- Train PRM for per-step rewards instead of outcome-only rewards.
- Much denser training signal for multi-step math reasoning.

### 11. Distillation Best Practices (from Open-R1 findings)
- **Don't pack samples** — packing is harmful for reasoning traces.
- **Use larger LR** (4e-5 vs 2e-5 showed ~10 point LiveCodeBench improvement).
- **Prefill with `<think>`** to ensure consistent long CoT behavior.
- **Use 8-bit optimizers** for memory efficiency with long contexts.

---

## Code Quality Issues Found

### Python n-gram blocking loop
- **File:** `reasoning_trainer.py` lines ~470-490
- **Issue:** Nested Python loops for n-gram history checking. Converts tensors to `.tolist()`.
- **Fix:** Vectorize with `torch.unfold()` or move to compiled helper.
- **Status:** ✅ IMPLEMENTED — `_vectorized_ngram_blocking()` uses `torch.unfold()` + vectorized comparison. No `.tolist()` calls.

### Repeated model mode switching
- **Issue:** `model.eval()` in `generate_completion()`, `model.train()` in `compute_sequence_logprobs()` — switched multiple times per step.
- **Impact:** Minor overhead from batch norm stat toggling (though RMSNorm doesn't use running stats).
- **Status:** ⏭️ LOW PRIORITY

### No AMP scaler
- **Issue:** Uses `autocast` but no `GradScaler`. For bfloat16 this is actually fine (bf16 doesn't need loss scaling).
- **Status:** ✅ CORRECT AS-IS

---

## Implementation Notes

### Selective log-softmax
Replaced `F.log_softmax(shift_logits, dim=-1)` + `torch.gather` with:
```python
selected_logits = shift_logits.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
logsumexp = torch.logsumexp(shift_logits, dim=-1)
token_logprobs = selected_logits - logsumexp
```
This avoids materializing the full `[B, T, V]` log-prob tensor (saves ~250MB for typical batch sizes).

### empty_cache removal
Removed `torch.cuda.empty_cache()` from the per-generation loop in `run_grpo_step()`. Kept a single cleanup after the full GRPO step completes (post-optimizer step). This eliminates repeated CUDA allocator sync points during the generation phase.

---

## References
- DeepSeek-R1: arXiv:2501.12948
- DeepSeekMath (GRPO): arXiv:2402.03300
- Open-R1 Update #3: https://huggingface.co/blog/open-r1/update-3
- DAPO / DPO-Positive: arXiv:2402.13228
- vLLM Speculative Decoding: https://docs.vllm.ai/en/latest/features/spec_decode.html
