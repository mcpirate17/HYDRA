#!/usr/bin/env python3
"""
GRPO Memory Profiler - Identifies memory spikes during reasoning training.

Profiles each stage of _run_reasoning_step to find OOM culprits:
1. Generation phase (autoregressive sampling)
2. Reward computation
3. Log-probability computation (forward pass with gradients)
4. Backward pass
5. Optimizer step

Usage:
    source /home/tim/venvs/llm/bin/activate && python diagnostics/profile_grpo_memory.py

Options:
    --model_size: Model size (default: 100M, use 500M to match production)
    --moe: Enable MoE (default: True to match production)
    --seq_len: Sequence length for generation (default: 256)
    --batch_size: Reasoning batch size (default: 2)
    --num_generations: Generations per prompt (default: 4)
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Set CUDA allocator config before importing torch
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F


@dataclass
class MemorySnapshot:
    """Single memory measurement."""
    stage: str
    allocated_gb: float
    reserved_gb: float
    free_gb: float
    peak_allocated_gb: float

    def __str__(self) -> str:
        return (
            f"{self.stage:40s} | "
            f"Alloc: {self.allocated_gb:6.2f}GB | "
            f"Reserved: {self.reserved_gb:6.2f}GB | "
            f"Free: {self.free_gb:6.2f}GB | "
            f"Peak: {self.peak_allocated_gb:6.2f}GB"
        )


@dataclass
class MemoryProfile:
    """Collection of memory snapshots."""
    snapshots: List[MemorySnapshot] = field(default_factory=list)

    def snapshot(self, stage: str) -> MemorySnapshot:
        """Take a memory snapshot at the current stage."""
        torch.cuda.synchronize()

        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        free = torch.cuda.mem_get_info()[0] / (1024**3)
        peak = torch.cuda.max_memory_allocated() / (1024**3)

        snap = MemorySnapshot(
            stage=stage,
            allocated_gb=allocated,
            reserved_gb=reserved,
            free_gb=free,
            peak_allocated_gb=peak,
        )
        self.snapshots.append(snap)
        return snap

    def reset_peak(self) -> None:
        """Reset peak memory tracking."""
        torch.cuda.reset_peak_memory_stats()

    def print_report(self) -> None:
        """Print full memory profile report."""
        print("\n" + "=" * 100)
        print("GRPO MEMORY PROFILE REPORT")
        print("=" * 100)

        print(f"\n{'Stage':40s} | {'Allocated':>12s} | {'Reserved':>12s} | {'Free':>12s} | {'Peak':>12s}")
        print("-" * 100)

        for snap in self.snapshots:
            print(snap)

        # Find biggest jumps
        print("\n" + "-" * 100)
        print("MEMORY DELTAS (biggest allocations):")
        print("-" * 100)

        deltas = []
        for i in range(1, len(self.snapshots)):
            prev = self.snapshots[i-1]
            curr = self.snapshots[i]
            delta = curr.allocated_gb - prev.allocated_gb
            deltas.append((curr.stage, delta, prev.stage))

        # Sort by absolute delta
        deltas.sort(key=lambda x: abs(x[1]), reverse=True)

        for stage, delta, prev_stage in deltas[:10]:
            sign = "+" if delta > 0 else ""
            print(f"  {prev_stage} -> {stage}: {sign}{delta:.3f}GB")

        # Peak analysis
        print("\n" + "-" * 100)
        print("PEAK MEMORY ANALYSIS:")
        print("-" * 100)

        max_peak = max(s.peak_allocated_gb for s in self.snapshots)
        max_alloc = max(s.allocated_gb for s in self.snapshots)
        min_free = min(s.free_gb for s in self.snapshots)

        print(f"  Maximum peak allocated: {max_peak:.2f}GB")
        print(f"  Maximum allocated:      {max_alloc:.2f}GB")
        print(f"  Minimum free VRAM:      {min_free:.2f}GB")

        # Find which stage had the peak
        for snap in self.snapshots:
            if snap.peak_allocated_gb == max_peak:
                print(f"  Peak occurred at:       {snap.stage}")
                break


def get_gpu_info() -> Dict[str, Any]:
    """Get GPU information."""
    if not torch.cuda.is_available():
        return {"available": False}

    props = torch.cuda.get_device_properties(0)
    total_mem = props.total_memory / (1024**3)

    return {
        "available": True,
        "name": props.name,
        "total_memory_gb": total_mem,
        "compute_capability": f"{props.major}.{props.minor}",
    }


def create_model(
    model_size: str = "100M",
    moe_enabled: bool = False,
    moe_num_experts: int = 6,
    moe_num_layers: int = 6,
    device: str = "cuda",
) -> torch.nn.Module:
    """Create HYDRA model for profiling."""
    from hydra.training.config import MODEL_SIZE_CONFIGS, TrainingConfig
    from hydra.model.framework import HydraModel

    size_config = MODEL_SIZE_CONFIGS.get(model_size, MODEL_SIZE_CONFIGS["100M"])

    config = TrainingConfig(
        model_size=model_size,
        mod_mor_dim=size_config["mod_mor_dim"],
        n_mor_blocks=size_config["n_mor_blocks"],
        mor_recursions=size_config["mor_recursions"],
        mod_mor_n_heads=size_config["mod_mor_n_heads"],
        mod_mor_n_kv_heads=size_config["mod_mor_n_kv_heads"],
        moe_enabled=moe_enabled,
        moe_num_experts=moe_num_experts if moe_enabled else 0,
        moe_num_layers=moe_num_layers if moe_enabled else 0,
        gradient_checkpointing=True,
        checkpoint_every_n=1,
    )

    model = HydraModel(
        dim=config.mod_mor_dim,
        n_blocks=config.n_mor_blocks,
        n_recursions=config.mor_recursions,
        n_heads=config.mod_mor_n_heads,
        n_kv_heads=config.mod_mor_n_kv_heads,
        vocab_size=config.vocab_size,
        max_seq_len=config.max_seq_len,
        mod_capacity=0.75,
        mor_adaptive=True,
        moe_num_experts=config.moe_num_experts if moe_enabled else 0,
        moe_num_layers=config.moe_num_layers if moe_enabled else 0,
    )

    model = model.to(device).to(torch.bfloat16)
    model.enable_gradient_checkpointing(every_n=1)

    return model, config


def create_optimizer(model: torch.nn.Module, use_8bit: bool = True) -> torch.optim.Optimizer:
    """Create optimizer matching production setup."""
    if use_8bit:
        try:
            import bitsandbytes as bnb
            return bnb.optim.Adam8bit(model.parameters(), lr=1e-5)
        except ImportError:
            print("Warning: bitsandbytes not available, using standard AdamW")

    return torch.optim.AdamW(model.parameters(), lr=1e-5)


@torch.no_grad()
def generate_sequences(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    profile: MemoryProfile,
    num_sequences: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate sequences with memory profiling at each step."""
    device = prompt_ids.device
    B, prompt_len = prompt_ids.shape

    profile.snapshot("gen_start")

    # Expand for multiple generations
    if num_sequences > 1:
        prompt_ids = prompt_ids.repeat_interleave(num_sequences, dim=0)

    total_batch = prompt_ids.shape[0]
    max_total_len = prompt_len + max_new_tokens

    # Pre-allocate buffer
    generated = torch.zeros(
        (total_batch, max_total_len), device=device, dtype=prompt_ids.dtype
    )
    generated[:, :prompt_len] = prompt_ids
    current_len = prompt_len

    profile.snapshot("gen_buffer_allocated")

    base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    base_model.eval()

    # Generate tokens
    with torch.inference_mode():
        for i in range(max_new_tokens):
            input_ids = generated[:, :current_len]

            outputs = base_model(input_ids)
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs

            next_logits = logits[:, -1, :]
            next_tokens = torch.argmax(next_logits, dim=-1)

            generated[:, current_len] = next_tokens
            current_len += 1

            # Profile at intervals
            if i == 0:
                profile.snapshot("gen_first_token")
            elif i == max_new_tokens // 2:
                profile.snapshot("gen_mid_tokens")

    profile.snapshot("gen_complete")

    generated = generated[:, :current_len].contiguous()
    completion_mask = torch.zeros(total_batch, current_len, device=device)
    completion_mask[:, prompt_len:] = 1.0

    return generated, completion_mask


def compute_logprobs_profiled(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    mask: torch.Tensor,
    profile: MemoryProfile,
    micro_batch_size: int = 1,
) -> torch.Tensor:
    """Compute log probabilities with detailed memory profiling."""
    device = input_ids.device
    B, L = input_ids.shape

    profile.snapshot("logprobs_start")

    base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    base_model.train()

    all_logprobs = []

    for i in range(0, B, micro_batch_size):
        end_idx = min(i + micro_batch_size, B)
        chunk_ids = input_ids[i:end_idx]
        chunk_mask = mask[i:end_idx]

        profile.snapshot(f"logprobs_chunk_{i}_start")

        # Forward pass
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = base_model(chunk_ids)
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs

        profile.snapshot(f"logprobs_chunk_{i}_forward_done")

        # Compute log probs
        shift_logits = logits[:, :-1, :]
        shift_labels = chunk_ids[:, 1:]
        shift_mask = chunk_mask[:, 1:]

        # Memory-efficient log_softmax + gather
        log_probs = F.log_softmax(shift_logits, dim=-1)

        profile.snapshot(f"logprobs_chunk_{i}_logsoftmax_done")

        token_logprobs = torch.gather(
            log_probs, dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        token_logprobs = token_logprobs * shift_mask

        # Pad to original length
        result = torch.zeros(end_idx - i, L, device=device, dtype=token_logprobs.dtype)
        result[:, 1:] = token_logprobs

        all_logprobs.append(result)

        profile.snapshot(f"logprobs_chunk_{i}_complete")

        # Clean up intermediate tensors
        del logits, shift_logits, log_probs, token_logprobs
        torch.cuda.empty_cache()

        profile.snapshot(f"logprobs_chunk_{i}_cleanup")

    return torch.cat(all_logprobs, dim=0)


def run_backward_profiled(
    logprobs: torch.Tensor,
    mask: torch.Tensor,
    advantages: torch.Tensor,
    scaler: torch.cuda.amp.GradScaler,
    profile: MemoryProfile,
) -> torch.Tensor:
    """Run backward pass with profiling."""
    profile.snapshot("backward_start")

    # Compute loss
    num_tokens = mask.sum(dim=1).clamp(min=1)
    completion_logprobs = (logprobs * mask).sum(dim=1) / num_tokens
    policy_loss = -(advantages * completion_logprobs).mean()

    profile.snapshot("backward_loss_computed")

    # Backward
    scaler.scale(policy_loss).backward()

    profile.snapshot("backward_complete")

    return policy_loss.detach()


def run_optimizer_step_profiled(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    profile: MemoryProfile,
) -> None:
    """Run optimizer step with profiling."""
    profile.snapshot("optimizer_start")

    scaler.unscale_(optimizer)

    profile.snapshot("optimizer_unscaled")

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    profile.snapshot("optimizer_clipped")

    scaler.step(optimizer)
    scaler.update()

    profile.snapshot("optimizer_stepped")

    optimizer.zero_grad(set_to_none=True)

    profile.snapshot("optimizer_zeroed")


def profile_full_grpo_step(
    model_size: str = "100M",
    moe_enabled: bool = True,
    moe_num_experts: int = 6,
    moe_num_layers: int = 6,
    batch_size: int = 2,
    num_generations: int = 4,
    max_tokens: int = 256,
    prompt_len: int = 128,
) -> MemoryProfile:
    """Profile a complete GRPO reasoning step."""

    device = "cuda"
    profile = MemoryProfile()

    print(f"\n{'='*80}")
    print("GRPO MEMORY PROFILER")
    print(f"{'='*80}")

    gpu_info = get_gpu_info()
    print(f"\nGPU: {gpu_info['name']}")
    print(f"Total VRAM: {gpu_info['total_memory_gb']:.1f}GB")

    print(f"\nConfiguration:")
    print(f"  Model size: {model_size}")
    print(f"  MoE enabled: {moe_enabled}")
    if moe_enabled:
        print(f"  MoE experts: {moe_num_experts}")
        print(f"  MoE layers: {moe_num_layers}")
    print(f"  Batch size: {batch_size}")
    print(f"  Num generations: {num_generations}")
    print(f"  Max tokens: {max_tokens}")
    print(f"  Prompt length: {prompt_len}")
    print(f"  Total sequences: {batch_size * num_generations}")

    # Clear GPU
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    profile.snapshot("initial_clean")

    # Create model
    print("\n[1/7] Creating model...")
    model, config = create_model(
        model_size=model_size,
        moe_enabled=moe_enabled,
        moe_num_experts=moe_num_experts,
        moe_num_layers=moe_num_layers,
        device=device,
    )

    profile.snapshot("model_created")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params / 1e6:.1f}M")

    # Create optimizer
    print("\n[2/7] Creating optimizer...")
    optimizer = create_optimizer(model, use_8bit=True)
    # Note: GradScaler not needed for bfloat16 (has sufficient dynamic range)
    # We'll use a dummy scaler that passes through for API compatibility
    use_scaler = False  # bfloat16 doesn't need scaling

    profile.snapshot("optimizer_created")

    # Simulate FULL training state - do multiple forward/backward at training seq_len
    # This forces optimizer to materialize ALL momentum buffers
    print("\n[3/7] Simulating full training state (10 steps @ seq_len=1024)...")
    training_seq_len = 1024
    training_batch = 4  # Match production batch size

    for warmup_step in range(10):
        dummy_input = torch.randint(0, 50257, (training_batch, training_seq_len), device=device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            dummy_out = model(dummy_input)
            if isinstance(dummy_out, tuple):
                dummy_loss = dummy_out[0].mean()
            else:
                dummy_loss = dummy_out.mean()
        dummy_loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        del dummy_input, dummy_out, dummy_loss

        if warmup_step == 0:
            profile.snapshot("first_training_step")

    # DON'T clear cache - we want to see memory state during actual training
    # gc.collect()
    # torch.cuda.empty_cache()

    profile.snapshot("optimizer_warmed_up")

    # Create prompt tokens
    print("\n[4/7] Generating sequences...")
    prompt_ids = torch.randint(0, 50257, (batch_size, prompt_len), device=device)

    profile.snapshot("prompts_created")

    # Generation phase
    profile.reset_peak()
    generated_ids, completion_mask = generate_sequences(
        model=model,
        prompt_ids=prompt_ids,
        max_new_tokens=max_tokens,
        profile=profile,
        num_sequences=num_generations,
    )

    total_samples = generated_ids.shape[0]
    total_len = generated_ids.shape[1]
    print(f"  Generated {total_samples} sequences of length {total_len}")

    # Simulate reward computation
    print("\n[5/7] Computing rewards...")
    rewards = torch.rand(batch_size, num_generations, device=device)
    mean_rewards = rewards.mean(dim=1, keepdim=True)
    std_rewards = rewards.std(dim=1, keepdim=True) + 1e-8
    advantages = ((rewards - mean_rewards) / std_rewards).view(-1)

    profile.snapshot("rewards_computed")

    # Log probability computation
    print("\n[6/7] Computing log probabilities...")
    gc.collect()
    torch.cuda.empty_cache()
    profile.reset_peak()

    profile.snapshot("pre_logprobs_cleanup")

    # Test different sequence length caps
    for max_seq_len in [128, 256, 512]:
        if total_len <= max_seq_len:
            test_ids = generated_ids
            test_mask = completion_mask
        else:
            test_ids = generated_ids[:, :max_seq_len]
            test_mask = completion_mask[:, :max_seq_len]

        print(f"\n  Testing seq_len={max_seq_len} (actual={test_ids.shape[1]})...")

        gc.collect()
        torch.cuda.empty_cache()
        profile.reset_peak()

        profile.snapshot(f"logprobs_seq{max_seq_len}_start")

        try:
            # Single sequence test
            single_ids = test_ids[:1]
            single_mask = test_mask[:1]

            model.train()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(single_ids)
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs

            profile.snapshot(f"logprobs_seq{max_seq_len}_1seq_forward")

            # Full forward pass memory
            log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)

            profile.snapshot(f"logprobs_seq{max_seq_len}_1seq_logsoftmax")

            # Backward
            fake_loss = log_probs.mean()
            fake_loss.backward()

            profile.snapshot(f"logprobs_seq{max_seq_len}_1seq_backward")

            optimizer.zero_grad(set_to_none=True)
            del logits, log_probs, fake_loss
            gc.collect()
            torch.cuda.empty_cache()

            profile.snapshot(f"logprobs_seq{max_seq_len}_1seq_cleanup")

        except torch.cuda.OutOfMemoryError as e:
            profile.snapshot(f"logprobs_seq{max_seq_len}_OOM")
            print(f"    OOM at seq_len={max_seq_len}!")
            gc.collect()
            torch.cuda.empty_cache()
            continue

    # Full GRPO step simulation
    print("\n[7/7] Simulating full GRPO backward...")
    gc.collect()
    torch.cuda.empty_cache()
    profile.reset_peak()

    # Use conservative settings
    test_seq_len = min(256, total_len)
    test_ids = generated_ids[:, :test_seq_len]
    test_mask = completion_mask[:, :test_seq_len]

    profile.snapshot("grpo_full_start")

    total_loss = 0.0
    micro_batch_size = 1

    for i in range(0, total_samples, micro_batch_size):
        end_idx = min(i + micro_batch_size, total_samples)
        chunk_ids = test_ids[i:end_idx]
        chunk_mask = test_mask[i:end_idx]
        chunk_adv = advantages[i:end_idx]

        profile.snapshot(f"grpo_micro{i}_start")

        try:
            model.train()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(chunk_ids)
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs

            profile.snapshot(f"grpo_micro{i}_forward")

            # Log probs
            shift_logits = logits[:, :-1, :]
            shift_labels = chunk_ids[:, 1:]
            shift_mask = chunk_mask[:, 1:]

            log_probs = F.log_softmax(shift_logits, dim=-1)
            token_logprobs = torch.gather(
                log_probs, dim=-1, index=shift_labels.unsqueeze(-1)
            ).squeeze(-1)
            token_logprobs = token_logprobs * shift_mask

            profile.snapshot(f"grpo_micro{i}_logprobs")

            # Loss
            num_tokens = shift_mask.sum(dim=1).clamp(min=1)
            completion_logprobs = token_logprobs.sum(dim=1) / num_tokens
            chunk_loss = -(chunk_adv * completion_logprobs).mean()

            # Backward (no scaler needed for bfloat16)
            (chunk_loss / (total_samples // micro_batch_size)).backward()

            profile.snapshot(f"grpo_micro{i}_backward")

            total_loss += chunk_loss.detach().item()

            del logits, shift_logits, log_probs, token_logprobs, chunk_loss
            gc.collect()
            torch.cuda.empty_cache()

            profile.snapshot(f"grpo_micro{i}_cleanup")

        except torch.cuda.OutOfMemoryError as e:
            profile.snapshot(f"grpo_micro{i}_OOM")
            print(f"    OOM at micro-batch {i}!")
            gc.collect()
            torch.cuda.empty_cache()
            break

    # Optimizer step (no scaler for bfloat16)
    try:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        profile.snapshot("grpo_clipped")

        optimizer.step()
        profile.snapshot("grpo_stepped")

        optimizer.zero_grad(set_to_none=True)
        profile.snapshot("grpo_zeroed")

    except torch.cuda.OutOfMemoryError:
        profile.snapshot("grpo_optimizer_OOM")
        print("    OOM during optimizer step!")

    # Final cleanup
    gc.collect()
    torch.cuda.empty_cache()
    profile.snapshot("final_cleanup")

    return profile


def main():
    parser = argparse.ArgumentParser(description="Profile GRPO memory usage")
    parser.add_argument("--model_size", type=str, default="100M",
                        choices=["debug", "100M", "250M", "500M"],
                        help="Model size to profile")
    parser.add_argument("--moe", action="store_true", default=True,
                        help="Enable MoE (default: True)")
    parser.add_argument("--no-moe", action="store_false", dest="moe",
                        help="Disable MoE")
    parser.add_argument("--moe_experts", type=int, default=6,
                        help="Number of MoE experts")
    parser.add_argument("--moe_layers", type=int, default=6,
                        help="Number of MoE layers")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Reasoning batch size")
    parser.add_argument("--num_generations", type=int, default=4,
                        help="Generations per prompt")
    parser.add_argument("--max_tokens", type=int, default=256,
                        help="Max tokens to generate")
    parser.add_argument("--prompt_len", type=int, default=128,
                        help="Prompt length")

    args = parser.parse_args()

    profile = profile_full_grpo_step(
        model_size=args.model_size,
        moe_enabled=args.moe,
        moe_num_experts=args.moe_experts,
        moe_num_layers=args.moe_layers,
        batch_size=args.batch_size,
        num_generations=args.num_generations,
        max_tokens=args.max_tokens,
        prompt_len=args.prompt_len,
    )

    profile.print_report()

    print("\n" + "=" * 80)
    print("RECOMMENDATIONS:")
    print("=" * 80)

    # Find the peak stage
    max_snap = max(profile.snapshots, key=lambda s: s.peak_allocated_gb)
    print(f"\n1. Peak memory ({max_snap.peak_allocated_gb:.2f}GB) occurred at: {max_snap.stage}")

    # Check if log_softmax is the culprit
    logsoftmax_snaps = [s for s in profile.snapshots if "logsoftmax" in s.stage.lower()]
    if logsoftmax_snaps:
        max_ls = max(logsoftmax_snaps, key=lambda s: s.allocated_gb)
        print(f"\n2. Log-softmax peak: {max_ls.allocated_gb:.2f}GB at {max_ls.stage}")
        print("   -> Consider using chunked log-softmax to avoid [B, L, V] allocation")

    # Check forward pass memory
    forward_snaps = [s for s in profile.snapshots if "forward" in s.stage.lower()]
    if forward_snaps:
        max_fwd = max(forward_snaps, key=lambda s: s.allocated_gb)
        print(f"\n3. Forward pass peak: {max_fwd.allocated_gb:.2f}GB at {max_fwd.stage}")

    # Memory delta analysis
    print("\n4. Largest memory jumps indicate allocation hotspots")
    print("   -> Focus optimization on stages with biggest deltas")


if __name__ == "__main__":
    main()
