#!/usr/bin/env python3
"""
Interactive chat with a trained HYDRA model.

Usage:
    python diagnostics/chat.py checkpoints/hydra_500m_final.pt
    python diagnostics/chat.py checkpoints/hydra_500m_final.pt --temperature 0.7
    python diagnostics/chat.py checkpoints/hydra_500m_final.pt --system "You are a helpful assistant."

    # With reasoning (model thinks before responding)
    python diagnostics/chat.py checkpoints/reasoning/reasoning_checkpoint.pt --format chat --reasoning
    python diagnostics/chat.py checkpoints/reasoning/reasoning_checkpoint.pt --format chat --reasoning --show-thinking
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from hydra.model.framework import HydraModel


def load_checkpoint(ckpt_path: str, device: str = "cuda") -> tuple:
    """Load checkpoint and reconstruct model."""
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    config = ckpt.get("config")
    if config is None:
        raise ValueError("Checkpoint has no 'config' key")

    def get_cfg(key, default=None):
        if hasattr(config, key):
            return getattr(config, key)
        elif isinstance(config, dict):
            return config.get(key, default)
        return default

    vocab_size = get_cfg("vocab_size", 50257)
    seq_len = get_cfg("max_seq_len", get_cfg("seq_len", 1024))
    dim = get_cfg("mod_mor_dim", 1024)
    n_mor_blocks = get_cfg("n_mor_blocks", 10)
    mor_recursions = get_cfg("mor_recursions", 4)
    n_heads = get_cfg("mod_mor_n_heads", 16)
    n_kv_heads = get_cfg("mod_mor_n_kv_heads", 4)
    mod_capacity = get_cfg("mod_capacity", 0.5)
    mor_adaptive = get_cfg("mor_adaptive", True)
    attention_backend = get_cfg("attention_backend", "ccgqa")

    print(f"  Model: {get_cfg('model_size', 'unknown')}, Step: {ckpt.get('step', '?')}")
    print(f"  Dim: {dim}, Blocks: {n_mor_blocks}x{mor_recursions}")

    model = HydraModel(
        vocab_size=vocab_size,
        dim=dim,
        n_mor_blocks=n_mor_blocks,
        recursions_per_block=mor_recursions,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        compression_factor=4,
        mlp_ratio=3.6,
        max_seq_len=seq_len,
        mod_capacity=mod_capacity,
        adaptive=mor_adaptive,
        tie_weights=True,
        attention_backend=attention_backend,
        mod_mlp_warmup=0,
        mor_warmup=0,
    )

    state_dict = ckpt.get("model_state_dict", ckpt.get("model"))
    clean_state = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            clean_state[k[10:]] = v
        else:
            clean_state[k] = v

    model.load_state_dict(clean_state, strict=False)
    model = model.to(device)
    model.eval()

    return model, seq_len


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.2,
    stop_strings: list[str] = None,
    device: str = "cuda",
    show_thinking: bool = True,
    reasoning_mode: bool = False,
) -> tuple[str, str]:
    """Generate text with streaming output.

    Args:
        reasoning_mode: If True, the prompt already ends with <think>\\n so
            generation starts inside the think block.

    Returns:
        (thinking, response) tuple. thinking is empty if no <think> block.
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated = input_ids
    generated_text = ""

    stop_strings = stop_strings or []

    # Track thinking state for streaming display
    # If reasoning_mode, we already injected <think> in the prompt so we start inside it
    in_think_block = reasoning_mode
    think_block_done = False
    thinking_text = ""

    if reasoning_mode and show_thinking:
        print("\033[2m[thinking] ", end="", flush=True)

    for _ in range(max_new_tokens):
        logits = model(generated)
        next_logits = logits[:, -1, :] / max(temperature, 0.01)

        # Repetition penalty
        if repetition_penalty != 1.0:
            for token_id in set(generated[0].tolist()):
                if next_logits[0, token_id] < 0:
                    next_logits[0, token_id] *= repetition_penalty
                else:
                    next_logits[0, token_id] /= repetition_penalty

        # Top-k
        if top_k > 0:
            indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
            next_logits[indices_to_remove] = float("-inf")

        # Top-p
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            next_logits[indices_to_remove] = float("-inf")

        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        generated = torch.cat([generated, next_token], dim=1)

        # Decode new token
        new_text = tokenizer.decode(next_token[0], skip_special_tokens=False)
        generated_text += new_text

        # Detect <think> / </think> transitions
        if not in_think_block and not think_block_done and "<think>" in generated_text:
            in_think_block = True
            if show_thinking:
                print("\033[2m[thinking] ", end="", flush=True)
            continue

        if in_think_block and "</think>" in generated_text:
            in_think_block = False
            think_block_done = True
            # Extract thinking content
            think_start = generated_text.find("<think>") + 7
            think_end = generated_text.find("</think>")
            thinking_text = generated_text[think_start:think_end].strip()
            if show_thinking:
                print("\033[0m", flush=True)  # Reset dim
            continue

        # Stream output
        if in_think_block:
            if show_thinking:
                print(new_text, end="", flush=True)
        elif think_block_done:
            # After </think>, only print non-tag content
            # Skip printing until we're past the </think> tag remnants
            after_think = generated_text.split("</think>", 1)[-1]
            if len(after_think) == len(new_text):
                print(new_text, end="", flush=True)
            elif after_think:
                # Transitional token that partially overlaps with tag
                print(after_think[-len(new_text):] if len(new_text) <= len(after_think) else after_think, end="", flush=True)
        else:
            # No think block at all, stream normally
            print(new_text, end="", flush=True)

        # Stop conditions
        if next_token.item() == tokenizer.eos_token_id:
            break

        for stop in stop_strings:
            if stop in generated_text:
                print()
                response = generated_text.split(stop)[0]
                if "</think>" in response:
                    response = response.split("</think>", 1)[-1].strip()
                return thinking_text, response

    print()
    response = generated_text
    if "</think>" in response:
        response = response.split("</think>", 1)[-1].strip()
    return thinking_text, response


def main():
    parser = argparse.ArgumentParser(description="Chat with HYDRA model")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--repetition-penalty", type=float, default=1.2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--system", type=str, default=None, help="System prompt")
    parser.add_argument("--format", type=str, default="raw",
                       choices=["raw", "chat", "instruct"],
                       help="Prompt format: raw (just text), chat (<|user|>), instruct (### Instruction)")
    parser.add_argument("--reasoning", action="store_true",
                       help="Enable reasoning mode: prompt model to think before responding")
    parser.add_argument("--show-thinking", action="store_true",
                       help="Show the <think> block content (dimmed). Hidden by default.")
    args = parser.parse_args()

    if args.reasoning and args.format not in ("chat", "raw"):
        print("Warning: --reasoning works best with --format chat or raw. Switching to chat format.")
        args.format = "chat"

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model, max_seq_len = load_checkpoint(args.checkpoint, args.device)

    print("\n" + "=" * 60)
    print("HYDRA Chat - Type 'quit' to exit, 'clear' to reset")
    mode_str = f"Format: {args.format} | Temp: {args.temperature} | Max tokens: {args.max_tokens}"
    if args.reasoning:
        mode_str += " | Reasoning: ON"
        if args.show_thinking:
            mode_str += " (visible)"
    print(mode_str)
    print("=" * 60 + "\n")

    history = ""
    if args.system:
        history = f"{args.system}\n\n"

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Goodbye!")
            break
        if user_input.lower() == "clear":
            if args.format == "chat":
                history = "<|system|>\nYou are a helpful assistant.\n"
            else:
                history = args.system + "\n\n" if args.system else ""
            print("[History cleared]")
            continue

        # Format prompt based on style
        if args.format == "chat":
            # HYDRA was trained on this exact format from Alpaca/Dolly
            if not history:
                history = "<|system|>\nYou are a helpful assistant.\n"
            prompt = f"{history}<|user|>\n{user_input}\n<|assistant|>\n"
            if args.reasoning:
                prompt += "<think>\n"
            stop_strings = ["<|user|>", "<|endoftext|>", "<|system|>"]
        elif args.format == "instruct":
            prompt = f"{history}### Instruction:\n{user_input}\n\n### Response:\n"
            stop_strings = ["### Instruction:", "###"]
        else:  # raw
            prompt = f"{history}{user_input}\n"
            if args.reasoning:
                prompt += "<think>\n"
            stop_strings = []

        print("HYDRA: ", end="", flush=True)
        thinking, response = generate(
            model, tokenizer, prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            stop_strings=stop_strings,
            device=args.device,
            show_thinking=args.show_thinking,
            reasoning_mode=args.reasoning,
        )

        # Update history (store the response only, not thinking)
        if args.format == "chat":
            history = f"{history}<|user|>\n{user_input}\n<|assistant|>\n{response}\n"
        elif args.format == "instruct":
            history = f"{prompt}{response}\n\n"
        else:
            history = f"{prompt}{response}\n"

        print()


if __name__ == "__main__":
    main()
