#!/usr/bin/env python3
"""
Evaluate a trained HYDRA checkpoint.

Usage:
    python diagnostics/eval_checkpoint.py checkpoints/hydra_250m_step_10000.pt
    python diagnostics/eval_checkpoint.py checkpoints/hydra_250m_step_10000.pt --prompt "Once upon a time"
    python diagnostics/eval_checkpoint.py checkpoints/hydra_250m_step_10000.pt --eval-loss --samples 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hydra.model.framework import HydraModel


def load_checkpoint(ckpt_path: str, device: str = "cuda") -> tuple:
    """Load checkpoint and reconstruct model."""
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    # Extract config
    config = ckpt.get("config")
    if config is None:
        raise ValueError("Checkpoint has no 'config' key - cannot reconstruct model")
    
    # Handle both object and dict configs
    def get_cfg(key, default=None):
        if hasattr(config, key):
            return getattr(config, key)
        elif isinstance(config, dict):
            return config.get(key, default)
        return default
    
    # Get model parameters directly from checkpoint config
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
    
    model_size = get_cfg("model_size", "unknown")
    
    print(f"  Model size: {model_size}")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Seq len: {seq_len}")
    print(f"  Dim: {dim}, Heads: {n_heads}, KV Heads: {n_kv_heads}")
    print(f"  MoR Blocks: {n_mor_blocks}, Recursions: {mor_recursions}")
    print(f"  Attention backend: {attention_backend}")
    print(f"  Step: {ckpt.get('step', 'unknown')}")
    
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
        # Routing already trained, no warmup needed
        mod_mlp_warmup=0,
        mor_warmup=0,
    )
    
    # Load weights
    state_dict = ckpt.get("model_state_dict", ckpt.get("model"))
    if state_dict is None:
        raise ValueError("Checkpoint has no model state dict")
    
    # Handle compiled model prefix
    clean_state = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            clean_state[k[10:]] = v  # Strip "_orig_mod."
        else:
            clean_state[k] = v
    
    model.load_state_dict(clean_state, strict=False)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")
    
    model = model.to(device)
    model.eval()
    
    return model, vocab_size, seq_len


def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    device: str = "cuda",
) -> str:
    """Generate text from a prompt."""
    model.eval()
    
    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    generated = input_ids
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Forward pass
            logits = model(generated)
            
            # Get next token logits
            next_logits = logits[:, -1, :] / temperature
            
            # Top-k filtering
            if top_k > 0:
                indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
                next_logits[indices_to_remove] = float("-inf")
            
            # Top-p filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_logits[indices_to_remove] = float("-inf")
            
            # Sample
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop on EOS
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    return tokenizer.decode(generated[0], skip_special_tokens=True)


def compute_eval_loss(model, tokenizer, texts: list[str], device: str = "cuda") -> float:
    """Compute average cross-entropy loss on texts."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for text in texts:
            tokens = tokenizer.encode(text, return_tensors="pt").to(device)
            if tokens.size(1) < 2:
                continue
            
            # Forward
            logits = model(tokens[:, :-1])
            targets = tokens[:, 1:]
            
            # Loss
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                reduction="sum",
            )
            total_loss += loss.item()
            total_tokens += targets.numel()
    
    return total_loss / max(total_tokens, 1)


def main():
    parser = argparse.ArgumentParser(description="Evaluate HYDRA checkpoint")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint file")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt for generation")
    parser.add_argument("--samples", type=int, default=3, help="Number of samples to generate")
    parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p (nucleus) sampling")
    parser.add_argument("--eval-loss", action="store_true", help="Compute eval loss on sample texts")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    args = parser.parse_args()
    
    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model, vocab_size, seq_len = load_checkpoint(args.checkpoint, args.device)
    
    print("\n" + "=" * 70)
    print("MODEL LOADED SUCCESSFULLY")
    print("=" * 70)
    
    # Default prompts if none provided
    default_prompts = [
        "The meaning of life is",
        "In a distant galaxy,",
        "The scientist discovered that",
        "Once upon a time, there was",
        "The future of artificial intelligence",
    ]
    
    prompts = [args.prompt] if args.prompt else default_prompts[:args.samples]
    
    # Generate samples
    print("\n" + "=" * 70)
    print("GENERATION SAMPLES")
    print("=" * 70)
    
    for i, prompt in enumerate(prompts):
        print(f"\n--- Sample {i+1} ---")
        print(f"Prompt: {prompt}")
        print("-" * 40)
        
        try:
            output = generate_text(
                model,
                tokenizer,
                prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                device=args.device,
            )
            print(output)
        except Exception as e:
            print(f"Generation failed: {e}")
    
    # Eval loss
    if args.eval_loss:
        print("\n" + "=" * 70)
        print("EVALUATION LOSS")
        print("=" * 70)
        
        eval_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "In 1969, humans first landed on the moon.",
            "Python is a popular programming language for data science.",
            "The capital of France is Paris, a beautiful city.",
        ]
        
        loss = compute_eval_loss(model, tokenizer, eval_texts, args.device)
        ppl = 2.718281828 ** loss  # e^loss = perplexity
        
        print(f"  Eval Loss: {loss:.4f}")
        print(f"  Perplexity: {ppl:.2f}")
        
        # Sanity check
        random_loss = torch.log(torch.tensor(float(vocab_size))).item()
        print(f"  Random baseline: {random_loss:.4f}")
        
        if loss < random_loss * 0.8:
            print("  ✅ Model has learned (loss << random)")
        elif loss < random_loss:
            print("  ⚠️  Model shows some learning")
        else:
            print("  ❌ Model may not have learned (loss >= random)")
    
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
