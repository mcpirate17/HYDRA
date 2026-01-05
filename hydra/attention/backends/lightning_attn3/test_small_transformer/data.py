import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
import datasets


class TinyStoriesDataset(Dataset):
    def __init__(self, tokenizer, seq_len=512, split='train'):
        self.tokenizer = tokenizer
        self.seq_len = seq_len

        # Load dataset
        self.dataset = datasets.load_dataset("roneneldan/TinyStories", split=split)

        # Tokenize and prepare chunks
        self.chunks = []
        for example in self.dataset:
            text = example['text']
            tokens = tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=self.seq_len) + [tokenizer.eos_token_id]
            # Create chunks
            for i in range(0, len(tokens) - self.seq_len, self.seq_len):
                chunk = tokens[i:i + self.seq_len + 1]  # +1 for labels
                if len(chunk) == self.seq_len + 1:
                    self.chunks.append(chunk)

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
        labels = torch.tensor(chunk[1:], dtype=torch.long)
        return {'input_ids': input_ids, 'labels': labels}


def create_data_loader(batch_size=4, seq_len=1024, num_workers=0):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = TinyStoriesDataset(tokenizer, seq_len)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader