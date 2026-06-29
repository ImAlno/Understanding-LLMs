"""
A GPT — the full Transformer, from scratch.
===========================================
Chapter 4 built ONE attention head and got structured gibberish. Here we assemble the real
thing — the architecture behind GPT-2 and ChatGPT — by adding four ideas around attention:

  • Multi-head attention — several heads in parallel, so the model attends to several things
    at once, then a projection mixes their findings.
  • A feed-forward net — a per-position MLP that lets each token "think" about what it gathered.
  • Residual connections — `x = x + sublayer(x)` — so gradients flow through deep stacks.
  • LayerNorm — normalizes activations to keep training stable.

We stack these into Blocks, stack Blocks into a GPT, and train on Shakespeare. The loss drops
well below Chapter 4's, and the samples become recognizably English.

Run:  python gpt.py     (trains a few minutes on a laptop CPU)
"""
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

DATA = Path(__file__).resolve().parent.parent / "data" / "input.txt"

# ---- config (CPU-friendly; scale these up on a GPU for sharper text) ----
block_size = 64       # context length
n_embd = 128          # embedding dimension
n_head = 4            # number of attention heads (each of size n_embd // n_head)
n_layer = 4           # number of Transformer blocks
dropout = 0.1
batch_size = 32
max_steps = 3000
learning_rate = 3e-4
torch.manual_seed(1337)

# ---- data ----
text = DATA.read_text()
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]


def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
    return x, y


class Head(nn.Module):
    """One self-attention head (Chapter 4's head, now with dropout)."""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k, q = self.key(x), self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = self.dropout(F.softmax(wei, dim=-1))
        return wei @ self.value(x)


class MultiHeadAttention(nn.Module):
    """Several heads in parallel, concatenated, then projected back to n_embd."""

    def __init__(self, n_head, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_head)])
        self.proj = nn.Linear(head_size * n_head, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)   # (B, T, n_head*head_size)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    """A per-position MLP — lets each token process what it gathered."""

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
    """A Transformer block: attention + feed-forward, each with a residual + pre-LayerNorm."""

    def __init__(self, n_embd, n_head):
        super().__init__()
        self.sa = MultiHeadAttention(n_head, n_embd // n_head)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))      # communicate (attention), then add back (residual)
        x = x + self.ffwd(self.ln2(x))    # think (feed-forward), then add back
        return x


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_embedding(idx) + self.position_embedding(torch.arange(T))
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            B, T, Vc = logits.shape
            loss = F.cross_entropy(logits.view(B * T, Vc), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -block_size:])
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, num_samples=1)], dim=1)
        return idx


@torch.no_grad()
def estimate_loss(model, eval_iters=50):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            _, loss = model(*get_batch(split))
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    model = GPT()
    print(f"vocab {vocab_size} | {sum(p.nelement() for p in model.parameters())/1e3:.0f}K parameters")
    print(f"config: n_embd {n_embd}, {n_head} heads, {n_layer} layers, block_size {block_size}\n")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for step in range(max_steps):
        _, loss = model(*get_batch("train"))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 500 == 0 or step == max_steps - 1:
            L = estimate_loss(model)
            print(f"step {step:5d} | train {L['train']:.4f} | val {L['val']:.4f}")

    print("\n(Chapter 4's single head got ~2.34; the full Transformer goes well below)\n")
    print("--- sample ---")
    start = torch.zeros((1, 1), dtype=torch.long)
    print(decode(model.generate(start, max_new_tokens=500)[0].tolist()))


if __name__ == "__main__":
    main()
