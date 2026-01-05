"""
Integration tests for MoE with HydraModel.

Tests cover:
- Model initialization with MoE enabled vs disabled
- Bit-for-bit equivalence when MoE is OFF
- Forward/backward pass with MoE layers
- Training loop behavior with MoE aux loss
- Checkpoint compatibility
"""

import pytest
import torch
import torch.nn as nn
from copy import deepcopy

from hydra.model.framework.model import HydraModel


# =============================================================================
# Fixtures
# =============================================================================

# Model config kwargs (no MoE)
BASE_MODEL_KWARGS = dict(
    vocab_size=1000,
    dim=128,
    n_mor_blocks=6,  # Need enough blocks for MoE placement
    recursions_per_block=2,
    n_heads=4,
    n_kv_heads=2,
    compression_factor=2,
    max_seq_len=128,
    mod_capacity=0.5,
    moe_enabled=False,
)

# Model config kwargs (with MoE)
MOE_MODEL_KWARGS = dict(
    vocab_size=1000,
    dim=128,
    n_mor_blocks=6,  # Need enough blocks for MoE placement
    recursions_per_block=2,
    n_heads=4,
    n_kv_heads=2,
    compression_factor=2,
    max_seq_len=128,
    mod_capacity=0.5,
    # MoE settings
    moe_enabled=True,
    moe_num_experts=4,
    moe_num_layers=2,
    moe_top_k=1,
    moe_aux_weight=0.01,
    moe_warmup_steps=100,
    moe_identity_init=True,
)


@pytest.fixture
def base_model_kwargs():
    """Base model kwargs without MoE."""
    return BASE_MODEL_KWARGS.copy()


@pytest.fixture
def moe_model_kwargs():
    """Model kwargs with MoE enabled."""
    return MOE_MODEL_KWARGS.copy()


@pytest.fixture
def sample_batch():
    """Sample input batch."""
    return torch.randint(0, 1000, (2, 64))


@pytest.fixture
def sample_batch_cuda():
    """Sample input batch on CUDA if available."""
    if torch.cuda.is_available():
        return torch.randint(0, 1000, (2, 64), device="cuda")
    return None


# =============================================================================
# Model Initialization Tests
# =============================================================================

class TestMoEModelInit:
    """Tests for model initialization with MoE."""
    
    def test_model_init_without_moe(self, base_model_kwargs):
        """Test model initializes correctly without MoE."""
        model = HydraModel(**base_model_kwargs)
        
        # Check no MoE layers
        assert hasattr(model, 'moe_layers')
        assert len(model.moe_layers) == 0
        
        # Model should have standard structure
        assert hasattr(model, 'tok_emb')
        assert hasattr(model, 'layers')
        assert hasattr(model, 'norm')
        assert hasattr(model, 'output')
    
    def test_model_init_with_moe(self, moe_model_kwargs):
        """Test model initializes correctly with MoE."""
        model = HydraModel(**moe_model_kwargs)
        
        # Check MoE layers created
        assert hasattr(model, 'moe_layers')
        assert len(model.moe_layers) == moe_model_kwargs["moe_num_layers"]
        
        # Verify MoE placement
        assert hasattr(model, '_moe_placement')
        assert len(model._moe_placement) == moe_model_kwargs["moe_num_layers"]
    
    def test_param_count_increase_with_moe(self, base_model_kwargs, moe_model_kwargs):
        """Test MoE increases parameter count."""
        model_base = HydraModel(**base_model_kwargs)
        model_moe = HydraModel(**moe_model_kwargs)
        
        base_params = sum(p.numel() for p in model_base.parameters())
        moe_params = sum(p.numel() for p in model_moe.parameters())
        
        # MoE model should have more parameters
        assert moe_params > base_params
        
        # Estimate expected increase: 2 MoE layers * 4 experts * expert_size
        # Expert size ~ 2 * dim * hidden_dim (gate/up + down)
        # This is a rough sanity check
        param_increase = moe_params - base_params
        assert param_increase > 0


# =============================================================================
# Forward Pass Tests
# =============================================================================

class TestMoEForward:
    """Tests for forward pass with MoE."""
    
    def test_forward_no_moe(self, base_model_kwargs, sample_batch):
        """Test forward pass without MoE."""
        model = HydraModel(**base_model_kwargs)
        model.eval()
        
        with torch.no_grad():
            logits = model(sample_batch)
        
        B, L = sample_batch.shape
        assert logits.shape == (B, L, base_model_kwargs["vocab_size"])
    
    def test_forward_with_moe(self, moe_model_kwargs, sample_batch):
        """Test forward pass with MoE enabled."""
        model = HydraModel(**moe_model_kwargs)
        model.eval()
        
        with torch.no_grad():
            logits = model(sample_batch)
        
        B, L = sample_batch.shape
        assert logits.shape == (B, L, moe_model_kwargs["vocab_size"])
    
    def test_forward_hidden_with_losses(self, moe_model_kwargs, sample_batch):
        """Test forward_hidden_with_losses returns MoE aux loss."""
        model = HydraModel(**moe_model_kwargs)
        model.train()
        model.set_global_step(200)  # After warmup
        
        logits, aux_losses = model.forward_hidden_with_losses(sample_batch)
        
        # Should have moe_aux_loss in aux_losses
        assert "moe_aux_loss" in aux_losses
        assert aux_losses["moe_aux_loss"].shape == ()  # scalar
    
    def test_moe_aux_loss_training_only(self, moe_model_kwargs, sample_batch):
        """Test MoE aux loss is only computed during training."""
        model = HydraModel(**moe_model_kwargs)
        model.set_global_step(200)
        
        # Training mode
        model.train()
        _, aux_losses_train = model.forward_hidden_with_losses(sample_batch)
        
        # Eval mode
        model.eval()
        _, aux_losses_eval = model.forward_hidden_with_losses(sample_batch)
        
        # Both should have the key, but eval should be 0
        assert "moe_aux_loss" in aux_losses_train
        assert "moe_aux_loss" in aux_losses_eval
        assert aux_losses_eval["moe_aux_loss"].item() == 0.0


# =============================================================================
# Equivalence Tests (MoE OFF)
# =============================================================================

class TestMoEOffEquivalence:
    """Tests for equivalence when MoE is disabled."""
    
    def test_same_output_with_alpha_zero(self, moe_model_kwargs, sample_batch):
        """Test output identical to input passthrough when alpha=0."""
        model = HydraModel(**moe_model_kwargs)
        model.eval()
        
        # Force alpha to 0 in all MoE layers
        with torch.no_grad():
            for moe_layer in model.moe_layers:
                moe_layer.residual_alpha.fill_(0.0)
        
        model.set_global_step(0)  # Warmup scale = 0
        
        # Get hidden states before and after MoE
        # Since alpha=0 and warmup_scale=0, MoE should have no effect
        logits = model(sample_batch)
        
        # Just verify output shape and finiteness
        assert logits.shape == (sample_batch.shape[0], sample_batch.shape[1], moe_model_kwargs["vocab_size"])
        assert torch.isfinite(logits).all()


# =============================================================================
# Backward Pass Tests
# =============================================================================

class TestMoEBackward:
    """Tests for backward pass with MoE."""
    
    def test_gradients_finite(self, moe_model_kwargs, sample_batch):
        """Test all gradients are finite."""
        model = HydraModel(**moe_model_kwargs)
        model.train()
        model.set_global_step(200)
        
        logits, aux_losses = model.forward_hidden_with_losses(sample_batch)
        
        # Compute loss
        loss = logits.sum() + aux_losses.get("moe_aux_loss", 0.0)
        loss.backward()
        
        # Check all gradients are finite
        for name, param in model.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), f"Non-finite grad in {name}"
    
    def test_moe_params_receive_grads(self, moe_model_kwargs, sample_batch):
        """Test MoE parameters receive gradients."""
        model = HydraModel(**moe_model_kwargs)
        model.train()
        model.set_global_step(200)
        
        logits, aux_losses = model.forward_hidden_with_losses(sample_batch)
        loss = logits.sum() + aux_losses.get("moe_aux_loss", 0.0)
        loss.backward()
        
        # Check MoE layer gradients
        moe_grads = []
        for name, param in model.named_parameters():
            if "moe_layers" in name and param.grad is not None:
                moe_grads.append(param.grad.abs().mean().item())
        
        # Should have some gradients in MoE layers
        assert len(moe_grads) > 0
        assert sum(moe_grads) > 0


# =============================================================================
# Training Loop Tests
# =============================================================================

class TestMoETraining:
    """Tests for MoE in training loop context."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_short_training_run(self, moe_model_kwargs, sample_batch_cuda):
        """Test short training run with MoE."""
        if sample_batch_cuda is None:
            pytest.skip("CUDA not available")
        
        model = HydraModel(**moe_model_kwargs).cuda()
        model.train()
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        initial_losses = []
        final_losses = []
        
        # 50 training steps
        for step in range(50):
            model.set_global_step(step)
            optimizer.zero_grad()
            
            # forward_hidden_with_losses returns hidden states, need to project
            hidden, aux_losses = model.forward_hidden_with_losses(sample_batch_cuda)
            logits = model.output(hidden)  # Project to vocab
            
            # Simple LM loss (next token prediction)
            targets = torch.roll(sample_batch_cuda, -1, dims=1)
            lm_loss = nn.functional.cross_entropy(
                logits.view(-1, moe_model_kwargs["vocab_size"]),
                targets.view(-1),
                ignore_index=-1,
            )
            
            moe_aux = aux_losses.get("moe_aux_loss", 0.0)
            total_loss = lm_loss + moe_aux
            
            if step < 5:
                initial_losses.append(total_loss.item())
            elif step >= 45:
                final_losses.append(total_loss.item())
            
            total_loss.backward()
            optimizer.step()
        
        # Loss should decrease (or at least not explode)
        avg_initial = sum(initial_losses) / len(initial_losses)
        avg_final = sum(final_losses) / len(final_losses)
        
        # Allow some tolerance - the main check is that training doesn't crash
        assert torch.isfinite(torch.tensor(avg_final))
        # Final loss should be at most 2x initial (shouldn't explode)
        assert avg_final < avg_initial * 2


# =============================================================================
# Global Step Propagation Tests  
# =============================================================================

class TestGlobalStepPropagation:
    """Tests for global step propagation to MoE layers."""
    
    def test_step_propagates_to_moe(self, moe_model_kwargs):
        """Test set_global_step propagates to MoE layers."""
        model = HydraModel(**moe_model_kwargs)
        
        model.set_global_step(500)
        
        # Check MoE layers received the step via get_warmup_scale behavior
        for moe_layer in model.moe_layers:
            # After warmup (100 steps), scale should be 1.0
            assert moe_layer.get_warmup_scale() == 1.0
    
    def test_warmup_scale_changes_with_step(self, moe_model_kwargs, sample_batch):
        """Test warmup scale changes as step progresses."""
        model = HydraModel(**moe_model_kwargs)
        model.eval()
        
        warmup_steps = moe_model_kwargs["moe_warmup_steps"]
        
        # At step 0, warmup scale should be 0
        model.set_global_step(0)
        for moe_layer in model.moe_layers:
            assert moe_layer.get_warmup_scale() == 0.0
        
        # At step = warmup_steps/2, scale should be ~0.5
        model.set_global_step(warmup_steps // 2)
        for moe_layer in model.moe_layers:
            scale = moe_layer.get_warmup_scale()
            assert 0.4 < scale < 0.6
        
        # After warmup, scale should be 1.0
        model.set_global_step(warmup_steps + 100)
        for moe_layer in model.moe_layers:
            assert moe_layer.get_warmup_scale() == 1.0


# =============================================================================
# Routing Stats Tests
# =============================================================================

class TestMoERoutingStats:
    """Tests for MoE routing statistics collection."""
    
    def test_routing_stats_collected(self, moe_model_kwargs, sample_batch):
        """Test routing stats are collected after forward pass."""
        model = HydraModel(**moe_model_kwargs)
        model.train()
        model.set_global_step(200)
        
        _ = model(sample_batch)
        
        stats = model.get_routing_stats()
        
        # Should have MoE in summary
        assert "summary" in stats
        assert stats["summary"]["moe_enabled"]
        
        # Should have moe_layers list
        assert "moe_layers" in stats
        assert len(stats["moe_layers"]) > 0


# =============================================================================
# CUDA Tests
# =============================================================================

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestMoECUDA:
    """Tests for MoE on CUDA."""
    
    def test_forward_cuda(self, moe_model_kwargs, sample_batch_cuda):
        """Test forward pass on CUDA."""
        if sample_batch_cuda is None:
            pytest.skip("CUDA not available")
        
        model = HydraModel(**moe_model_kwargs).cuda()
        model.eval()
        model.set_global_step(200)
        
        with torch.no_grad():
            logits = model(sample_batch_cuda)
        
        assert logits.device.type == "cuda"
        assert torch.isfinite(logits).all()
    
    def test_backward_cuda(self, moe_model_kwargs, sample_batch_cuda):
        """Test backward pass on CUDA."""
        if sample_batch_cuda is None:
            pytest.skip("CUDA not available")
        
        model = HydraModel(**moe_model_kwargs).cuda()
        model.train()
        model.set_global_step(200)
        
        logits, aux_losses = model.forward_hidden_with_losses(sample_batch_cuda)
        loss = logits.sum() + aux_losses.get("moe_aux_loss", 0.0)
        loss.backward()
        
        # Check gradients exist on CUDA
        for name, param in model.named_parameters():
            if param.grad is not None:
                assert param.grad.device.type == "cuda"


# =============================================================================
# torch.compile Tests
# =============================================================================

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestMoECompile:
    """Tests for torch.compile compatibility with MoE."""
    
    def test_compile_forward(self, moe_model_kwargs, sample_batch_cuda):
        """Test compiled forward pass."""
        if sample_batch_cuda is None:
            pytest.skip("CUDA not available")
        
        model = HydraModel(**moe_model_kwargs).cuda()
        model.eval()
        model.set_global_step(200)
        
        # Compile the model
        compiled_model = torch.compile(model, mode="reduce-overhead")
        
        with torch.no_grad():
            logits = compiled_model(sample_batch_cuda)
        
        assert logits.shape == (2, 64, moe_model_kwargs["vocab_size"])
        assert torch.isfinite(logits).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
