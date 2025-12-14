#!/usr/bin/env python3
"""
Debug script to verify the CCGQAMoDMoRModel can actually learn.

This tests:
1. Forward pass works
2. Backward pass works (gradients flow)
3. Loss decreases on repeated data (overfitting test)
4. Router statistics are reasonable

If the model can overfit on 1 batch, it can learn.
If it can't, there's a bug in the architecture.
"""

import torch
import torch.nn.functional as F
from hydra.model.ccgqa import CCGQAMoDMoRModel

def test_forward_backward():
    """Test basic forward/backward pass."""
    print("=" * 60)
    print("TEST 1: Forward/Backward Pass")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Small model for quick testing
    model = CCGQAMoDMoRModel(
        vocab_size=1000,
        dim=256,
        n_mor_blocks=2,
        recursions_per_block=2,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=64,
        mod_capacity=0.5,
    ).to(device)
    
    params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {params/1e6:.2f}M")
    
    # Random input
    x = torch.randint(0, 1000, (2, 64), device=device)
    
    # Forward
    model.train()
    logits, losses = model(x, return_losses=True)
    print(f"Forward OK: logits shape = {logits.shape}")
    print(f"Aux losses: {losses}")
    
    # Compute loss
    targets = torch.randint(0, 1000, (2, 64), device=device)
    ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    total_loss = ce_loss + 0.01 * losses.get("aux_loss", 0) + 0.001 * losses.get("ponder_loss", 0)
    print(f"CE Loss: {ce_loss.item():.4f}")
    print(f"Total Loss: {total_loss.item():.4f}")
    
    # Backward
    total_loss.backward()
    
    # Check gradients
    grad_norms = []
    zero_grads = 0
    for name, p in model.named_parameters():
        if p.grad is not None:
            norm = p.grad.norm().item()
            grad_norms.append(norm)
            if norm == 0:
                zero_grads += 1
        else:
            zero_grads += 1
    
    print(f"Backward OK: {len(grad_norms)} params with gradients")
    print(f"Zero gradients: {zero_grads}")
    print(f"Grad norm range: [{min(grad_norms):.6f}, {max(grad_norms):.6f}]")
    print(f"Mean grad norm: {sum(grad_norms)/len(grad_norms):.6f}")
    
    return True


def test_overfitting():
    """Test if model can overfit on a single batch (proves it can learn)."""
    print("\n" + "=" * 60)
    print("TEST 2: Overfitting Test (Can the model learn?)")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Small model
    model = CCGQAMoDMoRModel(
        vocab_size=1000,
        dim=256,
        n_mor_blocks=2,
        recursions_per_block=2,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=64,
        mod_capacity=0.5,
    ).to(device)
    
    # Fixed batch (same data every step)
    torch.manual_seed(42)
    x = torch.randint(0, 1000, (4, 64), device=device)
    y = torch.randint(0, 1000, (4, 64), device=device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    print("Training on SAME batch for 100 steps...")
    print("If loss decreases significantly, model can learn.")
    print()
    
    losses = []
    model.train()
    
    for step in range(100):
        optimizer.zero_grad()
        
        logits, aux_losses = model(x, return_losses=True)
        ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        aux = aux_losses.get("aux_loss", 0)
        ponder = aux_losses.get("ponder_loss", 0)
        total_loss = ce_loss + 0.01 * aux + 0.001 * ponder
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        losses.append(ce_loss.item())
        
        if step % 20 == 0:
            print(f"Step {step:3d}: CE Loss = {ce_loss.item():.4f}, Aux = {aux:.4f}, Ponder = {ponder:.4f}")
    
    initial_loss = losses[0]
    final_loss = losses[-1]
    reduction = (initial_loss - final_loss) / initial_loss * 100
    
    print()
    print(f"Initial loss: {initial_loss:.4f}")
    print(f"Final loss:   {final_loss:.4f}")
    print(f"Reduction:    {reduction:.1f}%")
    
    if reduction > 50:
        print("✅ PASS: Model can learn (>50% loss reduction on overfit test)")
        return True
    elif reduction > 20:
        print("⚠️  PARTIAL: Model learns slowly (20-50% reduction)")
        return True
    else:
        print("❌ FAIL: Model cannot learn (<20% reduction)")
        return False


def test_router_behavior():
    """Test if MoD/MoR routers are working correctly."""
    print("\n" + "=" * 60)
    print("TEST 3: Router Behavior (MoD + MoR)")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = CCGQAMoDMoRModel(
        vocab_size=1000,
        dim=256,
        n_mor_blocks=2,
        recursions_per_block=2,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=64,
        mod_capacity=0.5,
    ).to(device)
    
    x = torch.randint(0, 1000, (2, 64), device=device)
    
    model.eval()
    with torch.no_grad():
        logits, losses = model(x, return_losses=True)
    
    print(f"Aux loss (MoD load balancing): {losses.get('aux_loss', 'N/A')}")
    print(f"Ponder loss (MoR depth cost): {losses.get('ponder_loss', 'N/A')}")
    
    # Check router stats if available
    for i, block in enumerate(model.mor_blocks):
        if hasattr(block, '_last_router_probs_mean'):
            print(f"Block {i} router prob mean: {block._last_router_probs_mean:.4f}")
        if hasattr(block, '_last_depth_histogram'):
            print(f"Block {i} depth histogram: {block._last_depth_histogram}")
    
    return True


def test_gradient_flow_per_component():
    """Test gradients flow through each component."""
    print("\n" + "=" * 60)
    print("TEST 4: Gradient Flow Per Component")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = CCGQAMoDMoRModel(
        vocab_size=1000,
        dim=256,
        n_mor_blocks=2,
        recursions_per_block=2,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=64,
        mod_capacity=0.5,
    ).to(device)
    
    x = torch.randint(0, 1000, (2, 64), device=device)
    y = torch.randint(0, 1000, (2, 64), device=device)
    
    model.train()
    logits, losses = model(x, return_losses=True)
    ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
    total_loss = ce_loss + 0.01 * losses.get("aux_loss", 0) + 0.001 * losses.get("ponder_loss", 0)
    total_loss.backward()
    
    # Group gradients by component
    components = {
        "embed": [],
        "mor_block": [],
        "router": [],
        "attention": [],
        "ffn": [],
        "norm": [],
        "lm_head": [],
        "other": [],
    }
    
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        norm = p.grad.norm().item()
        
        if "embed" in name or "tok_emb" in name:
            components["embed"].append(norm)
        elif "router" in name:
            components["router"].append(norm)
        elif "attn" in name or "q_proj" in name or "k_proj" in name or "v_proj" in name:
            components["attention"].append(norm)
        elif "ffn" in name or "mlp" in name or "w1" in name or "w2" in name or "w3" in name:
            components["ffn"].append(norm)
        elif "norm" in name:
            components["norm"].append(norm)
        elif "lm_head" in name:
            components["lm_head"].append(norm)
        elif "mor_block" in name:
            components["mor_block"].append(norm)
        else:
            components["other"].append(norm)
    
    print("\nGradient norms by component:")
    for comp, norms in components.items():
        if norms:
            print(f"  {comp:15s}: mean={sum(norms)/len(norms):.6f}, max={max(norms):.6f}, count={len(norms)}")
        else:
            print(f"  {comp:15s}: NO GRADIENTS")
    
    # Check for vanishing/exploding gradients
    all_norms = [n for norms in components.values() for n in norms]
    if all_norms:
        mean_norm = sum(all_norms) / len(all_norms)
        if mean_norm < 1e-7:
            print("\n⚠️  WARNING: Very small gradients (vanishing)")
        elif mean_norm > 100:
            print("\n⚠️  WARNING: Very large gradients (exploding)")
        else:
            print(f"\n✅ Gradient magnitudes look healthy (mean={mean_norm:.6f})")
    
    return True


def test_longer_training():
    """Test training for 500 steps to see learning curve."""
    print("\n" + "=" * 60)
    print("TEST 5: Extended Training (500 steps)")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Slightly larger model
    model = CCGQAMoDMoRModel(
        vocab_size=1000,
        dim=384,
        n_mor_blocks=3,
        recursions_per_block=2,
        n_heads=6,
        n_kv_heads=2,
        max_seq_len=128,
        mod_capacity=0.5,
    ).to(device)
    
    params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {params/1e6:.2f}M")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    
    print("Training on random data for 500 steps...")
    print("Loss should decrease even on random data (memorization).\n")
    
    model.train()
    losses = []
    
    for step in range(500):
        # Different random data each step (harder than overfitting)
        x = torch.randint(0, 1000, (8, 128), device=device)
        y = torch.randint(0, 1000, (8, 128), device=device)
        
        optimizer.zero_grad()
        
        logits, aux_losses = model(x, return_losses=True)
        ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        aux = aux_losses.get("aux_loss", 0)
        ponder = aux_losses.get("ponder_loss", 0)
        total_loss = ce_loss + 0.01 * aux + 0.001 * ponder
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        losses.append(ce_loss.item())
        
        if step % 100 == 0:
            avg_loss = sum(losses[-100:]) / len(losses[-100:]) if len(losses) >= 100 else sum(losses) / len(losses)
            print(f"Step {step:3d}: CE Loss = {ce_loss.item():.4f}, Avg(100) = {avg_loss:.4f}")
    
    # Analyze learning curve
    first_100_avg = sum(losses[:100]) / 100
    last_100_avg = sum(losses[-100:]) / 100
    
    print()
    print(f"First 100 steps avg loss: {first_100_avg:.4f}")
    print(f"Last 100 steps avg loss:  {last_100_avg:.4f}")
    
    if last_100_avg < first_100_avg * 0.95:
        print("✅ Loss is decreasing - model is learning")
    else:
        print("⚠️  Loss not decreasing significantly")
    
    return True


if __name__ == "__main__":
    print("HYDRA CCGQAMoDMoRModel Learning Debug")
    print("=" * 60)
    
    try:
        test_forward_backward()
        test_overfitting()
        test_router_behavior()
        test_gradient_flow_per_component()
        test_longer_training()
        
        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETE")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
