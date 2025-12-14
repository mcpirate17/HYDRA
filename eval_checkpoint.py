#!/usr/bin/env python3
"""
CPU Evaluation Script for HYDRA Checkpoints

Evaluates model checkpoints on CPU without requiring GPU resources.
Useful for:
- Checking model quality during/after training
- Comparing checkpoints to find best one
- Validating model hasn't degraded
- Running evals on machines without GPU

Usage:
    python eval_checkpoint.py checkpoints/hydra_100m_step_69500.pt
    python eval_checkpoint.py checkpoints/hydra_100m_step_69500.pt --compare checkpoints/hydra_100m_step_56000.pt
    python eval_checkpoint.py checkpoints/hydra_100m_step_69500.pt --batches 32 --dataset fineweb-edu
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F


def load_model_from_checkpoint(checkpoint_path: str, device: str = "cpu"):
    """Load model and config from checkpoint."""
    from hydra.model.ccgqa import CCGQAMoDMoRModel
    
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt['config']
    step = ckpt.get('step', 'unknown')
    
    # Determine model architecture from config
    arch = config.get('architecture', 'mod_mor')
    
    if arch == 'mod_mor':
        model = CCGQAMoDMoRModel(
            vocab_size=config['vocab_size'],
            dim=config['mod_mor_dim'],
            n_mor_blocks=config['n_mor_blocks'],
            recursions_per_block=config['mor_recursions'],
            n_heads=config['mod_mor_n_heads'],
            n_kv_heads=config['mod_mor_n_kv_heads'],
            compression_factor=4,
            mlp_ratio=3.5,
            max_seq_len=config['max_seq_len'],
            mod_capacity=config['mod_capacity'],
            adaptive=config.get('mor_adaptive', True),
            tie_weights=True,
        )
    else:
        raise ValueError(f"Unsupported architecture: {arch}")
    
    model.load_state_dict(ckpt['model'])
    model.eval()
    
    return model, config, step


def evaluate_model(
    model,
    config: dict,
    n_batches: int = 16,
    batch_size: int = 4,
    dataset: str = None,
    device: str = "cpu",
    verbose: bool = True,
):
    """Run evaluation on model and return metrics."""
    from universal_data_loader import create_universal_loader
    
    # Use dataset from config if not specified
    if dataset is None:
        dataset = config.get('dataset_name', 'fineweb-edu')
    
    seq_len = config.get('max_seq_len', 512)
    vocab_size = config.get('vocab_size', 50257)
    tokenizer = config.get('tokenizer_name', 'gpt2')
    
    if verbose:
        print(f"Creating eval loader: dataset={dataset}, batch_size={batch_size}, seq_len={seq_len}")
    
    eval_loader = create_universal_loader(
        dataset=dataset,
        batch_size=batch_size,
        seq_len=seq_len,
        vocab_size=vocab_size,
        device=device,
        tokenizer_name=tokenizer,
    )
    
    if verbose:
        print(f"Running evaluation on {n_batches} batches...")
    
    total_ce = 0.0
    total_tokens = 0
    batch_losses = []
    
    start_time = time.time()
    
    with torch.no_grad():
        for i in range(n_batches):
            batch = eval_loader.get_batch()
            x = batch['input_ids']
            y = batch['labels']
            
            # Check if model is MoD/MoR (has return_losses)
            if hasattr(model, 'forward'):
                try:
                    logits, aux = model(x, return_losses=True)
                except TypeError:
                    logits = model(x)
                    aux = {}
            
            # Use reshape for non-contiguous tensors
            ce_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
                ignore_index=-100,
            )
            
            batch_loss = ce_loss.item()
            total_ce += batch_loss
            batch_losses.append(batch_loss)
            
            # Count non-padding tokens
            n_tokens = (y != -100).sum().item()
            total_tokens += n_tokens
            
            if verbose and (i + 1) % 4 == 0:
                print(f"  Batch {i+1}/{n_batches}: CE={batch_loss:.4f}")
    
    elapsed = time.time() - start_time
    avg_ce = total_ce / n_batches
    
    # Compute statistics
    import statistics
    std_ce = statistics.stdev(batch_losses) if len(batch_losses) > 1 else 0.0
    min_ce = min(batch_losses)
    max_ce = max(batch_losses)
    
    return {
        'avg_ce': avg_ce,
        'std_ce': std_ce,
        'min_ce': min_ce,
        'max_ce': max_ce,
        'n_batches': n_batches,
        'total_tokens': total_tokens,
        'elapsed_sec': elapsed,
        'tokens_per_sec': total_tokens / elapsed if elapsed > 0 else 0,
    }


def compare_checkpoints(ckpt1_path: str, ckpt2_path: str, **eval_kwargs):
    """Compare two checkpoints and report which is better."""
    print("=" * 60)
    print("CHECKPOINT COMPARISON")
    print("=" * 60)
    
    # Evaluate first checkpoint
    print(f"\n[1] {ckpt1_path}")
    model1, config1, step1 = load_model_from_checkpoint(ckpt1_path)
    results1 = evaluate_model(model1, config1, **eval_kwargs)
    print(f"    Step {step1}: CE = {results1['avg_ce']:.4f} ± {results1['std_ce']:.4f}")
    
    # Evaluate second checkpoint
    print(f"\n[2] {ckpt2_path}")
    model2, config2, step2 = load_model_from_checkpoint(ckpt2_path)
    results2 = evaluate_model(model2, config2, **eval_kwargs)
    print(f"    Step {step2}: CE = {results2['avg_ce']:.4f} ± {results2['std_ce']:.4f}")
    
    # Compare
    diff = results1['avg_ce'] - results2['avg_ce']
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    print(f"Checkpoint 1 (step {step1}): {results1['avg_ce']:.4f}")
    print(f"Checkpoint 2 (step {step2}): {results2['avg_ce']:.4f}")
    print(f"Difference: {diff:+.4f}")
    
    if abs(diff) < 0.05:
        verdict = "SIMILAR (within noise)"
    elif diff < 0:
        verdict = f"Checkpoint 1 (step {step1}) is BETTER"
    else:
        verdict = f"Checkpoint 2 (step {step2}) is BETTER"
    
    print(f"Verdict: {verdict}")
    
    return results1, results2


def analyze_training_diagnostics(diagnostics_path: str = "checkpoints/training_diagnostics.json"):
    """Analyze training diagnostics for loss trends and issues."""
    print("\n" + "=" * 60)
    print("TRAINING DIAGNOSTICS ANALYSIS")
    print("=" * 60)
    
    try:
        with open(diagnostics_path) as f:
            diag = json.load(f)
    except FileNotFoundError:
        print(f"Diagnostics file not found: {diagnostics_path}")
        return None
    
    if not diag:
        print("No diagnostic entries found")
        return None
    
    # Extract losses and grad norms
    entries = []
    for entry in diag:
        step = entry.get('step', 0)
        losses = entry.get('losses', {})
        total = losses.get('total')
        grad = entry.get('grad_norm')
        lr = entry.get('lr')
        if total is not None:
            entries.append({
                'step': step,
                'loss': total,
                'grad_norm': grad,
                'lr': lr,
            })
    
    if not entries:
        print("No valid entries with loss data")
        return None
    
    # Group by 1000-step windows
    windows = {}
    for e in entries:
        window = (e['step'] // 1000) * 1000
        if window not in windows:
            windows[window] = []
        windows[window].append(e['loss'])
    
    print("\nLoss by 1000-step windows (last 15):")
    print("Window     Min      Max      Avg      Count")
    print("-" * 50)
    
    sorted_windows = sorted(windows.keys())
    for w in sorted_windows[-15:]:
        vals = windows[w]
        print(f"{w:6d}   {min(vals):6.3f}   {max(vals):6.3f}   {sum(vals)/len(vals):6.3f}   {len(vals):3d}")
    
    # Check gradient norms
    grad_norms = [e['grad_norm'] for e in entries[-50:] if e.get('grad_norm')]
    if grad_norms:
        avg_grad = sum(grad_norms) / len(grad_norms)
        max_grad = max(grad_norms)
        print(f"\nGradient norms (last 50): avg={avg_grad:.3f}, max={max_grad:.3f}")
        if max_grad > 5 * avg_grad:
            print("⚠️  WARNING: Large gradient spike detected!")
        else:
            print("✓ Gradient norms look stable")
    
    # Check trend
    if len(sorted_windows) >= 5:
        last_windows = sorted_windows[-5:]
        avgs = [sum(windows[w])/len(windows[w]) for w in last_windows]
        trend = "INCREASING ↑" if avgs[-1] > avgs[0] + 0.1 else "DECREASING ↓" if avgs[-1] < avgs[0] - 0.1 else "STABLE →"
        print(f"\nLoss trend (last 5K steps): {trend}")
        print(f"  {avgs[0]:.4f} → {avgs[-1]:.4f} ({avgs[-1]-avgs[0]:+.4f})")
    
    return entries


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate HYDRA checkpoints on CPU",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Evaluate a single checkpoint
    python eval_checkpoint.py checkpoints/hydra_100m_step_69500.pt
    
    # Compare two checkpoints
    python eval_checkpoint.py checkpoints/hydra_100m_step_69500.pt --compare checkpoints/hydra_100m_step_56000.pt
    
    # Evaluate with more batches and different dataset
    python eval_checkpoint.py checkpoints/hydra_100m_step_69500.pt --batches 32 --dataset fineweb-edu
    
    # Analyze training diagnostics only
    python eval_checkpoint.py --diagnostics-only
        """
    )
    
    parser.add_argument("checkpoint", nargs="?", help="Path to checkpoint file")
    parser.add_argument("--compare", "-c", help="Second checkpoint to compare against")
    parser.add_argument("--batches", "-n", type=int, default=16, help="Number of batches to evaluate (default: 16)")
    parser.add_argument("--batch-size", "-b", type=int, default=4, help="Batch size (default: 4)")
    parser.add_argument("--dataset", "-d", help="Dataset to use (default: from checkpoint config)")
    parser.add_argument("--diagnostics", action="store_true", help="Also analyze training diagnostics")
    parser.add_argument("--diagnostics-only", action="store_true", help="Only analyze training diagnostics")
    parser.add_argument("--quiet", "-q", action="store_true", help="Less verbose output")
    
    args = parser.parse_args()
    
    # Diagnostics only mode
    if args.diagnostics_only:
        analyze_training_diagnostics()
        return
    
    if not args.checkpoint:
        parser.print_help()
        print("\nError: checkpoint path required (or use --diagnostics-only)")
        sys.exit(1)
    
    # Check checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    eval_kwargs = {
        'n_batches': args.batches,
        'batch_size': args.batch_size,
        'dataset': args.dataset,
        'verbose': not args.quiet,
    }
    
    if args.compare:
        # Compare two checkpoints
        if not Path(args.compare).exists():
            print(f"Error: Comparison checkpoint not found: {args.compare}")
            sys.exit(1)
        compare_checkpoints(args.checkpoint, args.compare, **eval_kwargs)
    else:
        # Evaluate single checkpoint
        print("=" * 60)
        print("CHECKPOINT EVALUATION")
        print("=" * 60)
        
        model, config, step = load_model_from_checkpoint(args.checkpoint)
        results = evaluate_model(model, config, **eval_kwargs)
        
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"Checkpoint: {args.checkpoint}")
        print(f"Step: {step}")
        print(f"Avg CE Loss: {results['avg_ce']:.4f} ± {results['std_ce']:.4f}")
        print(f"Min/Max CE:  {results['min_ce']:.4f} / {results['max_ce']:.4f}")
        print(f"Tokens evaluated: {results['total_tokens']:,}")
        print(f"Eval speed: {results['tokens_per_sec']:.0f} tok/s")
    
    # Optionally analyze diagnostics
    if args.diagnostics:
        analyze_training_diagnostics()


if __name__ == "__main__":
    main()
