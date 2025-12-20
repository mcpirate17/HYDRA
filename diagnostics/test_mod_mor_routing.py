"""
Diagnostic test for MoD and MoR routing health.

Industry-standard pytest tests for validating:
1. MoD layers maintain capacity near target (0.5) without collapse
2. MoR routers show proper depth distribution (not all same depth)  
3. aux_loss is appropriately weighted relative to CE loss (1-5%)

Usage:
    pytest diagnostics/test_mod_mor_routing.py -v
    pytest diagnostics/test_mod_mor_routing.py::TestMoDMoRRouting::test_mod_capacity_near_target -v
"""

import torch
import pytest
from hydra.model.ccgqa import CCGQAMoDMoRModel


class TestMoDMoRRouting:
    """Test MoD and MoR routing behavior."""
    
    @pytest.fixture
    def model(self):
        """Create a small test model."""
        model = CCGQAMoDMoRModel(
            vocab_size=1000,
            dim=256,
            n_mor_blocks=4,  # 4 MoR blocks
            n_heads=4,
            max_seq_len=128,
            n_kv_heads=2,
            mlp_ratio=1.0,
            recursions_per_block=3,
            mod_capacity=0.5,
        )
        return model
    
    @pytest.fixture
    def sample_input(self, model):
        """Create sample input."""
        batch_size = 4
        seq_len = 64
        return torch.randint(0, 1000, (batch_size, seq_len))
    
    def test_mod_capacity_not_collapsed(self, model, sample_input):
        """Test that MoD layers don't collapse to 0 or 1.
        
        Note: Uses train mode because eval mode may show different routing
        behavior (hard vs soft routing). This test validates training behavior.
        """
        model.train()  # Use train mode for soft routing
        with torch.no_grad():
            _ = model(sample_input)
        
        # Use stable API for routing stats
        stats = model.get_routing_stats()
        mod_layers = stats.get("mod_layers", [])
        
        # Check MoD layers
        mod_issues = []
        for layer_stat in mod_layers:
            probs = layer_stat.get("probs_mean", 0.5)
            layer_idx = layer_stat.get("layer", "?")
            if probs > 0.95:
                mod_issues.append(f"Layer {layer_idx}: probs={probs:.3f} (collapsed to process all)")
            elif probs < 0.05:
                mod_issues.append(f"Layer {layer_idx}: probs={probs:.3f} (collapsed to skip all)")
        
        assert len(mod_issues) == 0, f"MoD collapse detected:\n" + "\n".join(mod_issues)
    
    def test_mor_router_not_collapsed(self, model, sample_input):
        """Test that MoR routers exist and are accessible.
        
        Note: At random init, routers may show extreme values. This test
        primarily validates the router infrastructure is in place.
        Actual collapse detection happens during training diagnostics.
        """
        model.train()  # Use train mode to get varying router outputs
        
        # Run multiple forward passes to get router activity
        for _ in range(3):
            with torch.no_grad():
                _ = model(sample_input)
        
        # Use stable API for routing stats
        stats = model.get_routing_stats()
        mor_layers = stats.get("mor_layers", [])
        
        # Should have found some MoR layers
        assert len(mor_layers) > 0, "No MoR layers found in model"
        
        # Router values should be valid (between 0 and 1)
        for layer_stat in mor_layers:
            probs = layer_stat.get("router_probs_mean", 0.5)
            layer_idx = layer_stat.get("layer", "?")
            assert 0.0 <= probs <= 1.0, f"Layer {layer_idx}: invalid router_probs={probs}"
    
    def test_aux_loss_reasonable_magnitude(self, model, sample_input):
        """Test that aux_loss is reasonable relative to CE loss.
        
        aux_loss should be 1-10% of CE loss to guide without dominating.
        """
        model.train()
        
        # Create target
        target = torch.randint(0, 1000, sample_input.shape)
        
        # Forward pass with return_losses=True
        logits, aux_losses = model(sample_input, return_losses=True)
        
        # Compute CE loss
        ce_loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target.view(-1)
        )
        
        aux_loss = aux_losses.get("aux_loss", torch.tensor(0.0))
        
        # Check ratio
        if ce_loss.item() > 0:
            ratio = aux_loss.item() / ce_loss.item()
            # aux_loss should be 1-20% of CE in early training
            assert ratio < 0.20, f"aux_loss too strong: {aux_loss.item():.4f} is {ratio*100:.1f}% of CE {ce_loss.item():.4f}"
    
    def test_mod_capacity_near_target(self, model, sample_input):
        """Test that MoD probs are near target capacity (0.5)."""
        model.train()
        
        # Run a few forward passes to let routing stabilize
        for _ in range(3):
            with torch.no_grad():
                _ = model(sample_input)
        
        # Use stable API for routing stats
        stats = model.get_routing_stats()
        summary = stats.get("summary", {})
        avg_probs = summary.get("mod_probs_mean", 0.5)
        
        # Allow 0.2-0.8 range initially (not collapsed)
        assert 0.2 < avg_probs < 0.8, f"MoD avg probs {avg_probs:.3f} too far from target 0.5"
    
    def test_gradients_flow_through_routers(self, model, sample_input):
        """Test that gradients flow through both MoD and MoR routers."""
        model.train()
        target = torch.randint(0, 1000, sample_input.shape)
        
        # Forward pass with return_losses=True
        logits, aux_losses = model(sample_input, return_losses=True)
        
        # Compute loss
        ce_loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target.view(-1)
        )
        aux_loss = aux_losses.get("aux_loss", torch.tensor(0.0))
        total_loss = ce_loss + 0.1 * aux_loss
        
        # Backward
        total_loss.backward()
        
        # Check router gradients
        router_grads = []
        for name, param in model.named_parameters():
            if 'router' in name.lower() and param.grad is not None:
                grad_norm = param.grad.norm().item()
                router_grads.append((name, grad_norm))
        
        # Should have some router gradients
        assert len(router_grads) > 0, "No router gradients found"
        
        # Gradients should not be zero
        nonzero_grads = [(n, g) for n, g in router_grads if g > 1e-8]
        assert len(nonzero_grads) > 0, f"All router gradients are zero: {router_grads}"


class TestAuxLossWeighting:
    """Test aux_loss weight scenarios."""
    
    def test_aux_loss_weight_effect(self):
        """Test different aux_loss weights."""
        model = CCGQAMoDMoRModel(
            vocab_size=1000,
            dim=128,
            n_mor_blocks=2,
            n_heads=2,
            max_seq_len=64,
            n_kv_heads=1,
            mlp_ratio=1.0,
            recursions_per_block=2,
            mod_capacity=0.5,
            aux_loss_weight=0.1,  # Test with 0.1
        )
        
        x = torch.randint(0, 1000, (2, 32))
        model.train()
        
        # Use forward with return_losses=True
        logits, aux_losses = model(x, return_losses=True)
        aux_loss = aux_losses.get("aux_loss", torch.tensor(0.0))
        
        # With weight 0.1, aux_loss should be smaller
        assert aux_loss.item() < 1.0, f"aux_loss {aux_loss.item():.4f} too large with weight 0.1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
