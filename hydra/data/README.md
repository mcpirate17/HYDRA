# HYDRA Data Loading & Curriculum System

This module provides flexible data loading with support for automated curriculum transitions between dataset mixes.

## Quick Start

```bash
# Standard training with fixed dataset
--dataset pretrain_reasoning_lite

# Automated curriculum (transitions reasoning_lite -> agentic)
--dataset pretrain_agentic_curriculum
```

## Available Dataset Mixes

| Dataset | Web | CoT | Code | Chat | Math | Use Case |
|---------|-----|-----|------|------|------|----------|
| `pretrain_reasoning_lite` | 65% | 4% | 7% | 4% | 7% | Balanced reasoning focus |
| `pretrain_agentic_bridge` | 55% | 12% | 10% | 7% | 7% | Manual curriculum midpoint |
| `pretrain_agentic` | 45% | 20% | 12% | 10% | 8% | Agentic/tool-use focus |
| `pretrain_agentic_curriculum` | varies | varies | varies | varies | varies | Automated transition |

## Automated Curriculum (`pretrain_agentic_curriculum`)

The curriculum system automatically interpolates dataset weights based on training step:

```
Phase 1 (step < 280,000):     Use reasoning_lite weights (consolidate)
Phase 2 (280,000 - 350,000):  Linear interpolation (70K step transition)
Phase 3 (step >= 350,000):    Use full agentic weights
```

### Weight Transition Schedule

| Component | Phase 1 | Phase 2 (midpoint) | Phase 3 |
|-----------|---------|-------------------|---------|
| Web (finefineweb-local) | 65% | 55% | 45% |
| CoT (open_thoughts) | 4% | 7% | 10% |
| Reasoning (bespoke_stratos) | 0% | 2.5% | 5% |
| Synthetic (pleias_synth) | 4% | 4.5% | 5% |
| Code | 7% | 9.5% | 12% |
| Chat (small_chat_seqaware) | 3% | 4.5% | 6% |
| Chat (ultrachat) | 1% | 2.5% | 4% |
| Math (open_math_instruct) | 5% | 5% | 5% |
| Math (mathinstruct) | 2% | 2.5% | 3% |
| Narrative (tinystories) | 6% | 4.5% | 3% |
| Long-form (wikitext2) | 3% | 2.5% | 2% |

### Monitoring

The curriculum logs weight changes every 1000 steps:
```
Curriculum Phase 2 (transition 50%) at step 315000:
  finefineweb-local: 65.0% -> 55.0% -> 45.0%
  open_thoughts: 4.0% -> 7.0% -> 10.0%
  code: 7.0% -> 9.5% -> 12.0%
  ...
```

## Command Line Arguments

### Dataset Selection

| Argument | Description | Default |
|----------|-------------|---------|
| `--dataset NAME` | Dataset config name | `pretrain_reasoning_lite` |

### Training Parameters (relevant to curriculum)

| Argument | Description | Default |
|----------|-------------|---------|
| `--max_steps N` | Total training steps | varies by model |
| `--resume PATH` | Resume from checkpoint | None |
| `--batch_size N` | Batch size per GPU | 4 |
| `--grad_accum N` | Gradient accumulation steps | 8 |
| `--seq_len N` | Sequence length | 1024 |

### Model & Routing

| Argument | Description | Default |
|----------|-------------|---------|
| `--model_size SIZE` | Model variant (100M, 250M, 500M, 1B) | required |
| `--mod_capacity FLOAT` | MoD token capacity (0.5-1.0) | 0.75 |
| `--mor_already_enabled` | Skip MoR warmup (for resumed runs) | False |

### MoE (Mixture of Experts)

| Argument | Description | Default |
|----------|-------------|---------|
| `--moe` | Enable Mixture of Experts | False |
| `--moe_num_layers N` | Number of MoE layers | 0 |
| `--moe_num_experts N` | Experts per MoE layer | 4 |
| `--moe_aux_weight FLOAT` | Auxiliary load balancing loss | 0.01 |

### Optimization

| Argument | Description | Default |
|----------|-------------|---------|
| `--min_lr FLOAT` | Minimum learning rate | varies |
| `--max_lr FLOAT` | Maximum learning rate | varies |
| `--8bit_adam` | Use 8-bit Adam (saves VRAM) | False |
| `--no-adaptive_lr` | Disable adaptive LR scheduling | False |
| `--grad_clip_max FLOAT` | Maximum gradient clip value | 500.0 |
| `--grad_clip_k FLOAT` | Dynamic clip multiplier | 1.5 |

### Memory & Performance

| Argument | Description | Default |
|----------|-------------|---------|
| `--chunked_ce_size N` | Chunked cross-entropy size | 2048 |
| `--triton_kernels` | Enable fused Triton kernels | True |

## Example Commands

### Continue Training with Current Dataset

```bash
source /home/tim/venvs/llm/bin/activate && cd /home/tim/Projects/LLM/HYDRA && \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python trainer.py \
    --model_size 500M --mode production --batch_size 4 \
    --min_lr 4.15e-5 --max_lr 4.25e-5 \
    --grad_accum 8 --seq_len 1024 --max_steps 280000 \
    --resume checkpoints/hydra_500m_step_235000.pt \
    --mod_capacity 0.75 --mod_mlp_warmup_steps 46000 --mor_already_enabled \
    --mor_advantage_loss_scale 0.0025 \
    --no-adaptive_lr \
    --moe --moe_num_layers 6 \
    --moe_aux_weight 0.01 \
    --chunked_ce_size 2048 \
    --dataset pretrain_reasoning_lite \
    --8bit_adam \
    --grad_clip_max 500.0 \
    --grad_clip_k 1.5
```

### Switch to Automated Curriculum

```bash
source /home/tim/venvs/llm/bin/activate && cd /home/tim/Projects/LLM/HYDRA && \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python trainer.py \
    --model_size 500M --mode production --batch_size 4 \
    --min_lr 4.15e-5 --max_lr 4.25e-5 \
    --grad_accum 8 --seq_len 1024 --max_steps 400000 \
    --resume checkpoints/hydra_500m_step_280000.pt \
    --mod_capacity 0.75 --mod_mlp_warmup_steps 46000 --mor_already_enabled \
    --mor_advantage_loss_scale 0.0025 \
    --no-adaptive_lr \
    --moe --moe_num_layers 6 \
    --moe_aux_weight 0.01 \
    --chunked_ce_size 2048 \
    --dataset pretrain_agentic_curriculum \
    --8bit_adam \
    --grad_clip_max 500.0 \
    --grad_clip_k 1.5
```

### Manual 3-Phase Curriculum

For more control, manually switch datasets at milestones:

```bash
# Phase 1: Steps 235K -> 280K (consolidate on reasoning_lite)
--dataset pretrain_reasoning_lite --max_steps 280000

# Phase 2: Steps 280K -> 350K (bridge mix)
--dataset pretrain_agentic_bridge --max_steps 350000

# Phase 3: Steps 350K+ (full agentic)
--dataset pretrain_agentic --max_steps 500000
```

## Customizing the Curriculum

To create a custom curriculum schedule, add a new config to `DATASET_CONFIGS` in `universal_data_loader.py`:

```python
"my_custom_curriculum": {
    "mixed": True,
    "sources": [
        {"name": "finefineweb-local", "weight": 0.65},  # Start weights
        {"name": "open_thoughts", "weight": 0.04},
        {"name": "code", "weight": 0.07},
        # ... all sources needed
    ],
    "mix_schedule": {
        "type": "curriculum_transition",
        "phase1_end_step": 300000,    # When to start transition
        "phase2_end_step": 400000,    # When transition completes
        "end_weights": {
            "finefineweb-local": 0.45,  # Target weights
            "open_thoughts": 0.15,
            "code": 0.12,
            # ... target for each source
        },
    },
    "description": "My custom curriculum",
},
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `HYDRA_DATA_ROOT` | Parent directory for dataset shards |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | Reduce VRAM fragmentation |

## Troubleshooting

### Dataset not found
Ensure local datasets are available:
- `finefineweb-local`: Requires cached FineFineWeb JSONL files
- `pleias_synth`: Requires `HYDRA_PLEIAS_SYNTH_DIR` env var
- `small_chat_seqaware`: Requires local pre-tokenized chat data

### Curriculum not transitioning
- Check that `set_step()` is being called on the data loader (trainer does this automatically)
- Verify `current_step` is within the transition range
- Look for curriculum log messages every 1000 steps

### Memory issues with mixed datasets
- Reduce `num_workers` if running out of CPU memory
- Each source in a mixed dataset maintains its own iterator
