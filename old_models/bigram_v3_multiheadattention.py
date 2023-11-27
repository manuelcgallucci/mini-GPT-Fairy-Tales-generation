import torch
import torch.nn as nn
from torch.nn import functional as F

'''
(v2)
Added position embeddings 
Added LM head with n_embd layer
Added self-attention head (n_embd) (grab info from the past depending on the present)
    query: what am i looking for
    key: what do i contain
    dot product between the query at time N and keys at times 0, N-1 (in a N,N matrix)
    Notes: 
        Attention is a communication mechanism with data-dependent weights and can be seen as a graph
        In this case the nodes would only communicate with the past, never the future 
        Self attention is because the keys querys and values come from the same source, they could come from other encoder blocks 
        Scaled attention normalizes the Q @ K^T matrix by 1/sqrt(head_size). It normalized the matrix variance to be 1,
            it is important since it feeds into softmax so the numbers at init are "more diffused" after the softmax
            Has to get better 
(v3)
Added multiple parallel attention heads and concatenated the results
'''

# Hyperparameters
# =============
batch_size = 32
block_size = 8
max_iters = 5000
eval_interval = 300
learning_rate = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embd = 32 # Number of embedding dimensions
n_heads = 4
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


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        v = self.value(x) # (B,T,C)
        # Compute attention scores 
        wei = q @ k.transpose(-2, -1) # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf")) # (B, T, T)
        wei = F.softmax(wei, dim=1)
        # weighted aggregation of the values 
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    """Multiple self attention heads in parallel"""
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    
    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1)
    
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Each token directly rteads off the logits for thenext token from a lookup table 
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size) # Language modeling head
        self.sa_heads = MultiHeadAttention(n_heads, n_embd // n_heads) # 4 heads of 8-dim self attention

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are (B, T) tensors of ints
        #  every int in idx takes out one row of length vocab_size from the embedding table
        token_embeddings = self.token_embedding_table(idx) # (B, T, C)
        position_embeddings = self.position_embedding_table(torch.arange(T, device=device)) #  (T, C) arange = 0,1,2 
        x = token_embeddings + position_embeddings # Encoded information
        x = self.sa_heads(x)
        logits = self.lm_head(x) # (B, T, vocab_size)

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
            # crop idx to blac_size tokens
            idx_crop = idx[:, -block_size:]
            logits, loss = self(idx_crop)
            # Only get the last time dim step
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=1)
            # sample one from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append index to running idx sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel()
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

"""
Step 4800: Train Loss 0.5881, Val Loss 0.5900
Example output:



il ill aspidl sarlo ring ho it hadyi.HThe Inandnany poks and wo


cor sadat o thedear
c
king hough as ofuld thy, ""ses aracrt anillling wothor, has het not daint allinding wighears rus cout hint is afr ave wi.w ollde and ilfe and hathe the tlat otulte whorwitt re

"om wa puin franded mewe at rhey r
is ury one o tlastS cackecerpe any rwalefe s on haput ther al of;a.w theve and

fusee whort on, hat and.ad Therint asn.

f
"Hes!helte tha wappor oliegut agered his leset Llit, hJof and and and sore
"""