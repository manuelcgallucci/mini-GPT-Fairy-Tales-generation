import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperparameters
# =============
batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
# =============

torch.manual_seed(42)

with open("all_stories.txt", "r", encoding="utf-8") as f:
    dataset = f.read()

chars = sorted(list(set(dataset)))
vocab_size = len(chars)
# Mapping from vocab chars to ints 
# Very small vocal size tokenizer, gpt-2 uses 50k 
str2int = {ch:i for i, ch in enumerate(chars)}
int2str = {i:ch for i, ch in enumerate(chars)}
encode = lambda s: [str2int[c] for c in s]
decode = lambda l: "".join([int2str[i] for i in l])

data = torch.Tensor(encode(dataset)).to(torch.int64).to(device)
# Split training / validation at 85%
n = int(0.85 * len(data))
train_data = data[:n]
val_data = data[:n]

# data loading 
def get_batch(split):
    # generate a batch of data at input X and target Y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

@torch.no_grad() # Do not call / save backward mem for this function
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _logits_, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # Each token directly rteads off the logits for thenext token from a lookup table 
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, idx, targets=None):
        # idx and targets are (B, T) tensors of ints
        #  every int in idx takes out one row of length vocab_size from the embedding table
        logits = self.token_embedding_table(idx) # (B, T, C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(-1) # same as B*T
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            # Only get the last time dim step
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=1)
            # sample one from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append index to running idx sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel(vocab_size)
m = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {iter}: Train Loss {losses['train']:.4f}, Val Loss {losses['val']:.4f}")
    
    xb, yb = get_batch("train")

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), device=device, dtype=torch.long)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))