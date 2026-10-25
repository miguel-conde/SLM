import torch

# hyperparameters
batch_size = 16  #16
# how many independent sequences will we process in parallel?
#Specifies the number of sequences processed simultaneously in each batch.
#Increasing batch_size allows more data to be processed in parallel, which can stabilize training and potentially reduce training time. However, larger batches require more memory, especially on a GPU.
#Typical values can range from 8 to 64, depending on memory availability.

block_size = 32 #64 #32 # what is the maximum context length for predictions?
#Determines the maximum context length for predictions, i.e., the number of tokens considered in each input sequence.
#A larger block_size enables the model to consider a broader context when predicting, but also increases memory usage. For smaller datasets or shorter dependencies, a smaller value (like 32) may suffice. For long-range dependencies, increasing it to values like 128 or 256 can improve accuracy, given sufficient computational resources.

max_iters = 1000
#The total number of training iterations. This defines how long the model will train.
#Lower values might result in underfitting, while excessive values can lead to overfitting, especially if not paired with a validation-based stopping criterion.

eval_interval = 100   #The interval (in iterations) at which model performance is evaluated on validation data.

learning_rate = 1e-3
#Specifies the rate at which the model learns by updating weights. This parameter is crucial for training stability.
#Lower values (e.g., 1e-4) offer slower but more stable learning, especially useful for fine-tuning or sensitive models. Higher values (e.g., 1e-2) can speed up learning but may cause instability, particularly for complex models.

device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 400 #200
#Number of batches used to estimate validation loss.
#Higher values improve accuracy of the loss estimate but add computational overhead. If set too low, validation loss estimates may be noisy.

n_embd = 64 #128 #64
#Embedding dimension that defines the dimensionality of each token's representation.
#Smaller values (e.g., 64) reduce model capacity and training time but may limit the model's ability to capture intricate relationships. Higher values (e.g., 256) increase capacity but demand more memory and computational power.

n_head = 8 #8
#Specifies the number of attention heads in each layer. Multi-head attention allows the model to focus on different parts of the sequence simultaneously.
#Lower values (e.g., 4) may reduce model performance on tasks requiring diverse dependencies, while higher values (e.g., 16) allow for more comprehensive attention patterns at the cost of additional memory.

n_layer = 8 #8
#Number of Transformer layers (blocks) in the model. Each block adds complexity and depth, allowing the model to capture more nuanced patterns.
#Reducing the number of layers (e.g., 4) can help if resources are limited or for simpler datasets. More layers (e.g., 12 or 24) improve the model’s representational capacity but increase memory requirements and training time.

dropout = 0.0
#Dropout is a regularization technique that randomly “drops out” units during training to prevent overfitting.
#Setting dropout to 0.1 or 0.2 can improve generalization by forcing the model to learn more robust representations. In this code, dropout=0.0 suggests no dropout is applied, which may work well if overfitting is not a concern but can lead to overfitting on smaller datasets.

# ------------

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    # Explicitly move tensors to the correct device
    x, y = x.to(device), y.to(device)
    return x, y


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        #B is the batch size, T is the sequence length (number of tokens), and C is the embedding dimension.
        #This line unpacks the shape of the input tensor x, which has the shape (B, T, C).

        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        #These lines project x into key (k) and query (q) representations using learned linear transformations.
        #Each key, query, and value representation captures different aspects of the information in x for calculating attention.
        #After transformation, k and q still have the shape (B, T, C).

        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        #The @ operator performs matrix multiplication between q and the transpose of k.
        #k.transpose(-2, -1) changes the shape of k to (B, C, T), so q @ k.transpose(-2, -1) produces a matrix of shape (B, T, T).
        #This operation calculates the raw attention scores, indicating the relationship between each query and key pair. These scores are scaled by C**-0.5 to prevent overly large values, which helps stabilize gradients during training.

        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        #This line applies a mask to the wei matrix using a lower triangular matrix (tril) to enforce causality.
        #The masking sets upper-triangular values in wei to -inf, meaning only past and current tokens are attended to, not future ones. This ensures that predictions for each token only depend on past tokens in the sequence.

        wei = F.softmax(wei, dim=-1) # (B, T, T)
        #The softmax function normalizes the attention scores wei along the last dimension (sequence length), turning them into a probability distribution that sums to 1.
        #This distribution represents how much attention each token in the sequence should pay to other tokens.
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        #Applies dropout to wei to prevent overfitting by randomly setting a fraction of the attention probabilities to zero during training, helping improve generalization.

        v = self.value(x) # (B,T,C)
        #The input x is projected to obtain the values v using another linear transformation.
        # v has the shape (B, T, C), holding information from x to be aggregated based on the attention scores.
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        #This line performs the weighted sum of the values v, using the attention weights in wei.
        #The matrix multiplication wei @ v yields an output of shape (B, T, C), where each token’s representation in the sequence is adjusted based on its attention to other tokens.
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# super simple bigram model
#The BigramLanguageModel class is a simple language model that generates text one token at a time, based on bigrams (considering two tokens at a time for predictions).
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        #creates an embedding layer that maps each token in the vocabulary to a vector of size n_embd, effectively learning representations for each token.

        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        #adds position embeddings, allowing the model to incorporate positional information for each token up to block_size.

        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        # is a sequence of Transformer blocks (Block instances), which apply multi-head attention and feedforward layers to create a context-aware representation for each token.

        self.ln_f = nn.LayerNorm(n_embd) #  a final layer normalization to stabilize training.
        self.lm_head = nn.Linear(n_embd, vocab_size)
        # projects the final token representations back to the vocabulary size, outputting logits (unnormalized scores) for each possible next token.

    def forward(self, idx, targets=None):
        B, T = idx.shape
        #Maps the input token indices (idx) to embeddings using token_embedding_table and adds position embeddings from position_embedding_table.
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)
        #Applies a final layer normalization and outputs logits for each token in the vocabulary.
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    #this function produces new tokens based on the model's predictions in an auto-regressive manner (using previously generated tokens as context).
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        #It samples the next token based on probabilities from the model’s logits for the last generated token and appends it to the sequence until the specified number of tokens (max_new_tokens) are generated.
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx