import torch

import torch.nn as nn
from torch.nn import functional as F

# Encoder: text → numbers
encode = lambda s: [stoi[c] for c in s]

# Decoder: numbers → text
decode = lambda l: ''.join([itos[i] for i in l])



def get_batch(data, block_size, batch_size):
    """Generate a batch of data"""

    # Pick random starting positions
    ix = torch.randint(len(data) - block_size, (batch_size,))

    # Stack sequences into batches
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])

    return x, y

class SimpleBigramLanguageModel(nn.Module):
    """Simplest possible language model"""

    def __init__(self, vocab_size):
        super().__init__()
        # Each token predicts logits for next token
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx: (B, T) tensor of token indices
        # logits: (B, T, C) tensor of predictions
        logits = self.token_embedding_table(idx)

        if targets is None:
            loss = None
        else:
            # Reshape for cross-entropy loss
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """Generate new tokens"""
        for _ in range(max_new_tokens):
            # Get predictions
            logits, _ = self(idx)
            # Focus on last time step
            logits = logits[:, -1, :]  # (B, C)
            # Convert to probabilities
            probs = F.softmax(logits, dim=-1)
            # Sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
    
