#!/usr/bin/env python3
"""
Deep Model Health Check - CPU only

Checks for:
1. Gradient flow through all layers
2. Hidden state evolution (are layers actually transforming data?)
3. Attention patterns (are we attending or collapsed?)
4. Output head connectivity
5. Weight magnitudes and initialization sanity
"""

import sys
import os
import torch
import torch.nn.functional as F
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hydra.model import HydraModel, create_hydra_model


def check_weight_stats(model):
    """Check weight statistics for initialization sanity."""
    print("\n" + "=" * 70)
    print("WEIGHT STATISTICS")
    print("=" * 70)
    
    critical_issues = []
    
    for name, param in model.named_parameters():
        if param.numel() == 0:
            continue
        
        mean = param.data.mean().item()
        std = param.data.std().item()
        absmax = param.data.abs().max().item()
        
        # Check for issues
        issues = []
        if abs(mean) > 1.0:
            issues.append(f"HIGH_MEAN={mean:.3f}")
        if std < 1e-6:
            issues.append(f"ZERO_STD={std:.2e}")
        if std > 10.0:
            issues.append(f"HIGH_STD={std:.3f}")
        if absmax > 100:
            issues.append(f"HIGH_MAX={absmax:.3f}")
        if torch.isnan(param.data).any():
            issues.append("HAS_NAN")
        if torch.isinf(param.data).any():
            issues.append("HAS_INF")
        
        if issues:
            critical_issues.append((name, issues))
            print(f"  ⚠️  {name}: {', '.join(issues)}")
    
    if not critical_issues:
        print("  ✅ All weights look healthy")
    
    return len(critical_issues) == 0


def check_gradient_flow(model, vocab_size=1000, seq_len=64, batch_size=2):
    """Check if gradients flow through all parameters."""
    print("\n" + "=" * 70)
    print("GRADIENT FLOW CHECK")
    print("=" * 70)
    
    model.train()
    model.zero_grad()
    
    # Create dummy input
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    y = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Forward pass with losses
    logits, aux_losses = model(x, return_losses=True)
    
    # Compute loss
    ce_loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    aux_loss = aux_losses.get("aux_loss", torch.tensor(0.0))
    ponder_loss = aux_losses.get("ponder_loss", torch.tensor(0.0))
    total_loss = ce_loss + 0.1 * aux_loss + 0.01 * ponder_loss
    
    print(f"  CE Loss: {ce_loss.item():.4f}")
    print(f"  Aux Loss: {aux_loss.item() if hasattr(aux_loss, 'item') else aux_loss:.4f}")
    print(f"  Ponder Loss: {ponder_loss.item() if hasattr(ponder_loss, 'item') else ponder_loss:.4f}")
    print(f"  Total Loss: {total_loss.item():.4f}")
    
    # Backward
    total_loss.backward()
    
    # Check gradients
    no_grad_params = []
    zero_grad_params = []
    healthy_params = 0
    total_params = 0
    
    grad_norms = {}
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        total_params += 1
        
        if param.grad is None:
            no_grad_params.append(name)
        elif param.grad.abs().sum() == 0:
            zero_grad_params.append(name)
        else:
            healthy_params += 1
            grad_norms[name] = param.grad.norm().item()
    
    print(f"\n  Parameters with gradients: {healthy_params}/{total_params}")
    
    if no_grad_params:
        print(f"\n  ❌ NO GRADIENT (disconnected from loss):")
        for name in no_grad_params[:10]:
            print(f"     - {name}")
        if len(no_grad_params) > 10:
            print(f"     ... and {len(no_grad_params) - 10} more")
    
    if zero_grad_params:
        print(f"\n  ⚠️  ZERO GRADIENT (might be dead):")
        for name in zero_grad_params[:10]:
            print(f"     - {name}")
        if len(zero_grad_params) > 10:
            print(f"     ... and {len(zero_grad_params) - 10} more")
    
    # Show gradient distribution by layer type
    print(f"\n  Gradient norms by component:")
    component_grads = {}
    for name, norm in grad_norms.items():
        # Extract component type
        if "tok_emb" in name:
            key = "tok_emb"
        elif "output" in name:
            key = "output"
        elif "attention" in name:
            key = "attention"
        elif "mlp" in name or "gate_proj" in name or "up_proj" in name or "down_proj" in name:
            key = "mlp"
        elif "norm" in name:
            key = "norm"
        elif "router" in name:
            key = "router"
        else:
            key = "other"
        
        if key not in component_grads:
            component_grads[key] = []
        component_grads[key].append(norm)
    
    for comp, norms in sorted(component_grads.items()):
        avg = sum(norms) / len(norms)
        print(f"     {comp}: avg={avg:.2e}, min={min(norms):.2e}, max={max(norms):.2e}")
    
    all_ok = len(no_grad_params) == 0 and len(zero_grad_params) < total_params * 0.1
    if all_ok:
        print("\n  ✅ Gradient flow looks healthy")
    else:
        print("\n  ❌ Gradient flow has issues!")
    
    return all_ok


def check_hidden_state_evolution(model, vocab_size=1000, seq_len=64, batch_size=2):
    """Check if hidden states actually change through layers."""
    print("\n" + "=" * 70)
    print("HIDDEN STATE EVOLUTION")
    print("=" * 70)
    
    model.eval()
    
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Hook to capture hidden states
    hidden_states = []
    hooks = []
    
    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            hidden_states.append((name, out.detach().clone()))
        return hook
    
    # Register hooks on each layer
    base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    
    # Hook embedding
    hooks.append(base_model.tok_emb.register_forward_hook(make_hook("tok_emb")))
    
    # Hook each MoR block
    for i, layer in enumerate(base_model.layers):
        hooks.append(layer.register_forward_hook(make_hook(f"layer_{i}")))
    
    # Hook final norm
    hooks.append(base_model.norm.register_forward_hook(make_hook("final_norm")))
    
    # Forward pass
    with torch.no_grad():
        _ = model(x)
    
    # Remove hooks
    for h in hooks:
        h.remove()
    
    # Analyze hidden states
    print(f"\n  Layer-wise hidden state statistics:")
    prev_state = None
    issues = []
    
    for name, state in hidden_states:
        mean = state.mean().item()
        std = state.std().item()
        absmax = state.abs().max().item()
        
        # Check for collapse or explosion
        if std < 1e-4:
            issues.append(f"{name}: COLLAPSED (std={std:.2e})")
        if absmax > 1000:
            issues.append(f"{name}: EXPLODED (max={absmax:.1f})")
        
        # Check if state changed from previous
        if prev_state is not None:
            diff = (state - prev_state).abs().mean().item()
            rel_change = diff / (prev_state.abs().mean().item() + 1e-8)
            
            if rel_change < 1e-4:
                issues.append(f"{name}: NO CHANGE from previous (rel_change={rel_change:.2e})")
            
            print(f"    {name:20s}: mean={mean:+.4f}, std={std:.4f}, max={absmax:.2f}, Δ={rel_change:.4f}")
        else:
            print(f"    {name:20s}: mean={mean:+.4f}, std={std:.4f}, max={absmax:.2f}")
        
        prev_state = state
    
    if issues:
        print(f"\n  ⚠️  Issues found:")
        for issue in issues:
            print(f"     - {issue}")
    else:
        print(f"\n  ✅ Hidden states evolve normally through layers")
    
    return len(issues) == 0


def check_output_connectivity(model, vocab_size=1000, seq_len=64, batch_size=2):
    """Check if output logits actually depend on input."""
    print("\n" + "=" * 70)
    print("OUTPUT CONNECTIVITY CHECK")
    print("=" * 70)
    
    model.eval()
    
    # Two different inputs
    x1 = torch.randint(0, vocab_size, (batch_size, seq_len))
    x2 = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    with torch.no_grad():
        logits1 = model(x1)
        logits2 = model(x2)
    
    # Check if outputs are different
    diff = (logits1 - logits2).abs().mean().item()
    
    print(f"  Logits difference for different inputs: {diff:.4f}")
    
    if diff < 1e-4:
        print("  ❌ CRITICAL: Outputs are identical for different inputs!")
        print("     This suggests the model is not actually processing input.")
        return False
    else:
        print("  ✅ Outputs vary with input (model is connected)")
    
    # Check logit distribution
    print(f"\n  Logit statistics:")
    print(f"    Mean: {logits1.mean().item():.4f}")
    print(f"    Std: {logits1.std().item():.4f}")
    print(f"    Min: {logits1.min().item():.4f}")
    print(f"    Max: {logits1.max().item():.4f}")
    
    # Check if logits are reasonable for cross-entropy
    # Random init should give ~log(vocab_size) loss
    expected_random_loss = torch.log(torch.tensor(float(vocab_size))).item()
    
    y = torch.randint(0, vocab_size, (batch_size, seq_len))
    ce = F.cross_entropy(logits1.view(-1, vocab_size), y.view(-1)).item()
    
    print(f"\n  CE loss with random labels: {ce:.4f}")
    print(f"  Expected for random predictions: {expected_random_loss:.4f}")
    
    if ce > expected_random_loss * 1.5:
        print("  ⚠️  Loss is higher than random - possible initialization issue")
    elif ce < expected_random_loss * 0.5:
        print("  ⚠️  Loss is much lower than random - possible data leak or collapse")
    else:
        print("  ✅ Loss is in reasonable range")
    
    return True


def check_attention_patterns(model, vocab_size=1000, seq_len=64, batch_size=1):
    """Check if attention is actually attending or collapsed."""
    print("\n" + "=" * 70)
    print("ATTENTION PATTERN CHECK")
    print("=" * 70)
    
    # This is a simplified check - we look at Q/K similarity
    model.eval()
    
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # We'll check if Q and K projections produce varied outputs
    base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    
    # Get first layer attention
    first_layer = base_model.layers[0]
    
    # Check if it's a MoR block with attention inside
    if hasattr(first_layer, 'attention'):
        attn = first_layer.attention
    elif hasattr(first_layer, 'block') and hasattr(first_layer.block, 'attention'):
        attn = first_layer.block.attention
    else:
        print("  Could not find attention module for detailed check")
        return True
    
    # Get embedding
    with torch.no_grad():
        h = base_model.tok_emb(x) * (base_model.dim ** 0.5)
        
        # Check Q/K/V projections
        if hasattr(attn, 'q_proj'):
            q = attn.q_proj(h)
            k = attn.k_proj(h)
            v = attn.v_proj(h)
        elif hasattr(attn, 'wq'):
            q = attn.wq(h)
            k = attn.wk(h)
            v = attn.wv(h)
        else:
            print("  Could not find Q/K/V projections")
            return True
        
        print(f"  Q stats: mean={q.mean().item():.4f}, std={q.std().item():.4f}")
        print(f"  K stats: mean={k.mean().item():.4f}, std={k.std().item():.4f}")
        print(f"  V stats: mean={v.mean().item():.4f}, std={v.std().item():.4f}")
        
        # Check if Q and K produce varied patterns
        q_var = q.std(dim=1).mean().item()  # Variance across positions
        k_var = k.std(dim=1).mean().item()
        
        print(f"\n  Position-wise variance:")
        print(f"    Q variance across positions: {q_var:.4f}")
        print(f"    K variance across positions: {k_var:.4f}")
        
        if q_var < 1e-4 or k_var < 1e-4:
            print("  ⚠️  Attention may be collapsed (low position variance)")
            return False
        else:
            print("  ✅ Attention projections show healthy variance")
    
    return True


def check_tied_weights(model):
    """Check if tied weights are actually tied."""
    print("\n" + "=" * 70)
    print("WEIGHT TYING CHECK")
    print("=" * 70)
    
    base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    
    emb_weight = base_model.tok_emb.weight
    out_weight = base_model.output.weight
    
    is_tied = emb_weight.data_ptr() == out_weight.data_ptr()
    
    if is_tied:
        print("  ✅ Embedding and output weights are tied (same tensor)")
    else:
        # Check if they're at least similar
        diff = (emb_weight - out_weight).abs().mean().item()
        print(f"  ⚠️  Weights are NOT tied (diff={diff:.4f})")
        print("     This is not necessarily wrong, but worth noting.")
    
    return True


def main():
    print("=" * 70)
    print("HYDRA MODEL HEALTH CHECK (CPU)")
    print("=" * 70)
    
    # Create a small test model
    print("\nCreating small test model...")
    model = create_hydra_model(
        vocab_size=1000,  # Small vocab for fast testing
        dim=256,
        n_mor_blocks=4,
        recursions_per_block=2,
        n_heads=4,
        n_kv_heads=2,
        compression_factor=2,
        mlp_ratio=2.0,
        max_seq_len=128,
        mod_capacity=0.5,
        aux_loss_weight=0.01,
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model created: {total_params:,} parameters")
    
    results = {}
    
    results["weights"] = check_weight_stats(model)
    results["gradient_flow"] = check_gradient_flow(model, vocab_size=1000)
    results["hidden_evolution"] = check_hidden_state_evolution(model, vocab_size=1000)
    results["output_connectivity"] = check_output_connectivity(model, vocab_size=1000)
    results["attention_patterns"] = check_attention_patterns(model, vocab_size=1000)
    results["weight_tying"] = check_tied_weights(model)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    all_pass = True
    for check, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {check:25s}: {status}")
        if not passed:
            all_pass = False
    
    if all_pass:
        print("\n✅ All checks passed - model architecture appears sound")
    else:
        print("\n❌ Some checks failed - investigate the issues above")
    
    return all_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
