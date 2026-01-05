"""
Benchmark HYDRA full models: HydraBaseModel, HydraModel (MoD+MoR), memory usage.

This script benchmarks complete model forward/backward passes, NOT attention kernels.
For attention-level benchmarks, see: hydra/attention/backends/ccgqa/benchmark.py

Run with:
    python -m diagnostics.benchmark_hydra_models

Notes:
- REQUIRES: CUDA for full GPU benchmarks. Running on CPU will work for very small shapes
  but results are not indicative of GPU throughput.
"""

import torch
import time
from typing import Tuple

from hydra.attention import CCGQAAttention
from hydra.model.framework import HydraBaseModel, HydraModel, create_base_model, create_hydra_model


def benchmark_ccgqa_attention():
    """Quick benchmark of CCGQA attention (for model context).
    
    For comprehensive attention benchmarks, run:
        python hydra/attention/backends/ccgqa/benchmark.py --save --plot
    """
    print("=" * 80)
    print("CCGQA Attention Quick Check")
    print("(For full attention benchmarks: python hydra/attention/backends/ccgqa/benchmark.py)")
    print("=" * 80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    
    B, S, D = 2, 512, 1344
    x = torch.randn(B, S, D, device=device, dtype=dtype)
    
    attn = CCGQAAttention(
        dim=D,
        n_heads=21,
        n_kv_heads=3,
        compression_factor=4,
        max_seq_len=S * 2,
    ).to(device=device, dtype=dtype)
    
    # Warm up
    for _ in range(5):
        _ = attn(x)
    
    # Benchmark
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        out = attn(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - start
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Time per forward: {elapsed / 100 * 1000:.2f}ms")
    print(f"Device: {device}, dtype: {dtype}")
    
    # Parameter comparison
    n_params = sum(p.numel() for p in attn.parameters())
    print(f"\nCCGQA Attention params: {n_params:,}")
    
    # Compare to standard GQA
    gqa_params = D * D + D * (2 * D // 7) + D * D
    print(f"Standard GQA params (approx): {gqa_params:,}")
    print(f"Compression ratio: {gqa_params / n_params:.2f}x")
    print()


def benchmark_base_model():
    """Benchmark HydraBaseModel (CCGQA transformer without MoD/MoR)."""
    print("=" * 80)
    print("Benchmarking HydraBaseModel (CCGQA Transformer)")
    print("=" * 80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    
    class MockSpec:
        vocab_size = 50257
        dim = 1344
        n_layers = 24
        n_heads = 21
        n_kv_heads = 3
        compression_factor = 4
        mlp_ratio = 2.67
        max_seq_len = 8192
    
    model = create_base_model(MockSpec()).to(device=device, dtype=dtype)
    
    tokens = torch.randint(0, 50257, (2, 512), device=device)
    
    # Warm up
    for _ in range(5):
        _ = model(tokens)
    
    # Benchmark
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(20):
        logits = model(tokens)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - start
    
    print(f"Tokens shape: {tokens.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Time per forward: {elapsed / 20 * 1000:.2f}ms")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal model params: {total_params:,} ({total_params / 1e6:.1f}M)")
    print()


def benchmark_hydra_model():
    """Benchmark full HydraModel (CCGQA + MoD + MoR)."""
    print("=" * 80)
    print("Benchmarking HydraModel (CCGQA + MoD + MoR)")
    print("=" * 80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    
    mod_mor_model = create_hydra_model(
        dim=2048,
        n_mor_blocks=8,
        recursions_per_block=4,
        n_heads=32,
        n_kv_heads=4,
        mlp_ratio=4.0,
    ).to(device=device, dtype=dtype)
    
    mod_mor_model.train()
    mod_mor_model.set_global_step(200)  # Enable hard routing
    
    tokens = torch.randint(0, 50257, (2, 256), device=device)
    
    # Warm up
    for _ in range(5):
        _, _ = mod_mor_model(tokens, return_losses=True)
    
    # Benchmark
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(20):
        logits, losses = mod_mor_model(tokens, return_losses=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - start
    
    print(f"Tokens shape: {tokens.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Time per forward: {elapsed / 20 * 1000:.2f}ms")
    
    total_params = sum(p.numel() for p in mod_mor_model.parameters())
    print(f"\nTotal model params: {total_params:,} ({total_params / 1e6:.1f}M)")
    print(f"Effective layers: {mod_mor_model.effective_layers}")
    print(f"Aux loss: {losses['aux_loss'].item():.4f}")
    print(f"Ponder loss: {losses['ponder_loss'].item():.4f}")
    
    # Routing stats
    stats = mod_mor_model.get_routing_stats()
    if stats.get('mor_stats'):
        print("\nMoR Routing Stats:")
        for i, layer_stats in enumerate(stats['mor_stats']):
            avg_depth = layer_stats.get('avg_depth', -1)
            if avg_depth >= 0:
                print(f"  Layer {i}: avg_depth={avg_depth:.2f}")
    print()


def benchmark_memory_usage():
    """Benchmark memory usage with different configurations."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping memory benchmark")
        return
    
    print("=" * 80)
    print("Memory Usage Comparison")
    print("=" * 80)
    
    def measure_memory(model_fn, tokens) -> Tuple[float, float]:
        """Measure peak memory usage."""
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        model = model_fn().cuda().to(torch.bfloat16)
        tokens = tokens.cuda()
        
        # Forward
        _ = model(tokens)
        forward_mem = torch.cuda.max_memory_allocated() / 1e9
        
        # Backward
        if hasattr(model, 'forward'):
            out, losses = model(tokens, return_losses=True) \
                if 'return_losses' in model.forward.__code__.co_varnames else (model(tokens), {})
            loss = out.sum()
            if isinstance(losses, dict):
                for v in losses.values():
                    loss = loss + v
            loss.backward()
        
        backward_mem = torch.cuda.max_memory_allocated() / 1e9
        
        del model, tokens
        torch.cuda.empty_cache()
        
        return forward_mem, backward_mem
    
    tokens = torch.randint(0, 50257, (2, 512))
    
    # Base model
    def base_model():
        return create_base_model(type('Spec', (), {
            'vocab_size': 50257, 'dim': 1344, 'n_layers': 12,
            'n_heads': 21, 'n_kv_heads': 3, 'compression_factor': 4,
            'mlp_ratio': 2.67, 'max_seq_len': 8192
        }))
    
    fwd, bwd = measure_memory(base_model, tokens)
    print(f"HydraBaseModel (12L):       Forward={fwd:.2f}GB, Backward={bwd:.2f}GB")
    
    # MoD+MoR model
    def mod_mor_model():
        return create_hydra_model(
            dim=1344, n_mor_blocks=6, recursions_per_block=2,
            n_heads=21, n_kv_heads=3, mlp_ratio=2.67
        )
    
    fwd, bwd = measure_memory(mod_mor_model, tokens)
    print(f"HydraModel (6x2L MoD+MoR):  Forward={fwd:.2f}GB, Backward={bwd:.2f}GB")
    print()


def main():
    """Run all model benchmarks."""
    print("\n")
    print("=" * 80)
    print("HYDRA Model Benchmark Suite")
    print("=" * 80)
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
    print()
    
    benchmark_ccgqa_attention()
    benchmark_base_model()
    benchmark_hydra_model()
    
    if torch.cuda.is_available():
        benchmark_memory_usage()
    
    print("=" * 80)
    print("Model Benchmark Complete")
    print("=" * 80)
    print("\nFor attention kernel benchmarks, run:")
    print("  python hydra/attention/backends/ccgqa/benchmark.py --save --plot")


if __name__ == "__main__":
    main()
