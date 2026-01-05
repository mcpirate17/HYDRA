# Lightning Attention 3 Small Transformer Test

This folder contains a standalone test for the Lightning Attention 3 backend, disconnected from the main HYDRA project.

## Purpose

- Verify that the gradient scaling fix in `lightning_attn_interface.py` prevents exploding gradients
- Test end-to-end training of a small transformer using Lightning Attention
- Monitor training metrics: tokens/sec, memory usage, gradient norms

## Model Configuration

- **Layers**: 2
- **Hidden dim**: 512
- **Heads**: 8
- **FF dim**: 2048
- **Max seq len**: 512
- **Vocab size**: 50257 (GPT-2)

## Dataset

- TinyStories dataset (~2M tokens)
- Sequence length: 512
- Batch size: 4

## Usage

```bash
./run.sh
```

This will:
1. Install dependencies
2. Train for 500 steps
3. Monitor for gradient explosions
4. Report tokens/sec and memory usage

## Expected Output

If the gradient fix works, training should complete without NaN/inf gradients, with stable loss decrease and reasonable throughput.