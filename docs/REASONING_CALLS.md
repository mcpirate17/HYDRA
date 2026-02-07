# Reasoning call map (Trainer integration)

This table lists reasoning-related functions/methods and where they are invoked from training/trainer.py and trainer.py.

| Reasoning function/method | Defined at | Called in training/trainer.py | Called in trainer.py | Notes |
|---|---|---|---|---|
| `GRPOTrainerMixin` | [hydra/training/reasoning.py](hydra/training/reasoning.py#L452) | [hydra/training/trainer.py](hydra/training/trainer.py#L61) | Indirect via [trainer.py](trainer.py#L43) | Mixed into `Trainer` for System 2/GRPO behavior. |
| `reward_exact_match()` | [hydra/training/reasoning.py](hydra/training/reasoning.py#L65) | None | None | Used inside reasoning module via `REWARD_FUNCTIONS`. |
| `reward_format_check()` | [hydra/training/reasoning.py](hydra/training/reasoning.py#L114) | None | None | Used inside reasoning module via `REWARD_FUNCTIONS`. |
| `reward_length_penalty()` | [hydra/training/reasoning.py](hydra/training/reasoning.py#L162) | None | None | Used inside reasoning module via `REWARD_FUNCTIONS`. |
| `generate_completions()` | [hydra/training/reasoning.py](hydra/training/reasoning.py#L194) | None | None | Called internally by `_run_reasoning_step()`. |
| `_chunked_log_softmax_gather()` | [hydra/training/reasoning.py](hydra/training/reasoning.py#L323) | None | None | Helper for `compute_sequence_logprobs()`. |
| `compute_sequence_logprobs()` | [hydra/training/reasoning.py](hydra/training/reasoning.py#L379) | None | None | Called internally by `_run_reasoning_step()`. |
| `_ensure_reasoning_prompts()` | [hydra/training/reasoning.py](hydra/training/reasoning.py#L475) | None | None | Called internally by `_get_reasoning_batch()`. |
| `_get_reasoning_batch()` | [hydra/training/reasoning.py](hydra/training/reasoning.py#L518) | None | None | Called internally by `_run_reasoning_step()`. |
| `_ensure_tokenizer()` | [hydra/training/reasoning.py](hydra/training/reasoning.py#L530) | None | None | Called internally by `_run_reasoning_step()`. |
| `_clear_mor_caches()` | [hydra/training/reasoning.py](hydra/training/reasoning.py#L597) | None | None | Called internally by `_run_reasoning_step()`. |
| `_snapshot_model_state()` | [hydra/training/reasoning.py](hydra/training/reasoning.py#L620) | None | None | Present for reference policy snapshots; not called in training/trainer.py. |
| `_restore_model_state()` | [hydra/training/reasoning.py](hydra/training/reasoning.py#L625) | None | None | Present for reference policy snapshots; not called in training/trainer.py. |
| `_run_reasoning_step()` | [hydra/training/reasoning.py](hydra/training/reasoning.py#L630) | [hydra/training/trainer.py](hydra/training/trainer.py#L1984) | Indirect via [trainer.py](trainer.py#L45) | Main GRPO step invoked inside the training loop. |
| `_log_grpo_diagnostics()` | [hydra/training/trainer.py](hydra/training/trainer.py#L2878) | [hydra/training/trainer.py](hydra/training/trainer.py#L1989) | Indirect via [trainer.py](trainer.py#L45) | Trainer-side logging for GRPO metrics. |
| `detect_thought_boundaries()` | [hydra/training/reasoning.py](hydra/training/reasoning.py#L906) | None | None | Utility; not used by training/trainer.py. |
