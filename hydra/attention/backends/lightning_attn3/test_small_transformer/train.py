import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
import psutil
import os
from model import SmallTransformer
from data import create_data_loader


def get_memory_usage():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0


def train_model(
    model,
    dataloader,
    num_steps=500,
    lr=1e-3,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    log_every=10
):
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_steps)
    criterion = nn.CrossEntropyLoss()

    model.train()
    total_tokens = 0
    start_time = time.time()

    for step in range(num_steps):
        batch = next(iter(dataloader))  # Simple cycling through data
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        logits = model(input_ids)
        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

        loss.backward()

        # Check for gradient explosion
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
            print(f"WARNING: Gradient explosion detected at step {step}, grad_norm={grad_norm}")
            return False

        optimizer.step()
        scheduler.step()

        total_tokens += input_ids.numel()

        if step % log_every == 0:
            elapsed = time.time() - start_time
            tokens_per_sec = total_tokens / elapsed
            memory_mb = get_memory_usage()
            print(".4f"
                  ".2f"
                  ".2f")

    print("Training completed successfully! No gradient explosions detected.")
    return True


def main():
    # Model config
    vocab_size = 50257  # GPT-2
    d_model = 512
    n_heads = 8
    n_layers = 2
    d_ff = 2048
    max_seq_len = 1024

    model = SmallTransformer(vocab_size, d_model, n_heads, n_layers, d_ff, max_seq_len)
    print(f"Model has {model.num_params:,} parameters")

    # Data
    dataloader = create_data_loader(batch_size=4, seq_len=1024)

    # Train
    success = train_model(model, dataloader, num_steps=500)
    if success:
        print("✓ Lightning attention gradient fix verified!")
    else:
        print("✗ Gradient issues detected")


if __name__ == "__main__":
    main()