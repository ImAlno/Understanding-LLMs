"""
Self-attention from scratch — a single attention head.
======================================================
Chapter 3's MLP saw a FIXED, tiny context (3 characters) and mashed them together the same
way every time. **Attention** lets every position look back at *all* earlier positions and
decide, on the fly, which ones matter — the mechanism behind every modern LLM.

This builds one self-attention head: each token emits a **query** ("what am I looking
for?") and a **key** ("what do I contain?"); their dot-product says how much one token
should attend to another; a **causal mask** keeps it from peeking at the future; softmax
turns the scores into weights; and we take a weighted sum of each token's **value**.

We train it on ~1MB of Shakespeare (character level) and generate new text. The loss beats
the bigram baseline, and — more tellingly — the samples start to look like English.

We also meet PyTorch's `nn.Module`, `nn.Linear`, and `nn.Embedding` here: tidy wrappers for
the exact weight matrices we built by hand in Chapter 3 (a `Linear` is just `x @ W + b`).

Run:  python attention.py
"""
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

DATA = Path(__file__).resolve().parent.parent / "data" / "input.txt"

# ---- config ----
block_size = 32      # context length: how many previous characters the model can attend to
n_embd = 64          # embedding dimension
batch_size = 32
max_steps = 5000
learning_rate = 1e-3
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
    """Grab batch_size random chunks of length block_size, with their next-char targets."""
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
    return x, y


class Head(nn.Module):
    """One self-attention head."""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # a lower-triangular matrix of ones; used to mask out the future
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)        # (B, T, head_size) — what each token contains
        q = self.query(x)      # (B, T, head_size) — what each token is looking for
        # attention scores: how much each token should attend to each other token
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5     # (B, T, T), scaled
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # causal: no peeking ahead
        wei = F.softmax(wei, dim=-1)                            # weights sum to 1 over the past
        v = self.value(x)      # (B, T, head_size) — what each token offers if attended to
        return wei @ v         # (B, T, head_size) — weighted sum of past values


class AttentionLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)     # one vector per character
        self.position_embedding = nn.Embedding(block_size, n_embd)  # one vector per position
        self.sa_head = Head(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.token_embedding(idx)                      # (B, T, C)
        pos = self.position_embedding(torch.arange(T))       # (T, C) — broadcast over the batch
        x = tok + pos                                        # tokens now know WHERE they are
        x = self.sa_head(x)                                  # (B, T, C) — attention
        logits = self.lm_head(x)                             # (B, T, vocab_size)
        loss = None
        if targets is not None:
            B, T, Vc = logits.shape
            loss = F.cross_entropy(logits.view(B * T, Vc), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]                  # last block_size tokens
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]                        # focus on the last position
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
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
    model = AttentionLM()
    print(f"vocab {vocab_size} | {sum(p.nelement() for p in model.parameters())} parameters\n")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)  # a better optimizer (Ch 7)

    for step in range(max_steps):
        xb, yb = get_batch("train")
        _, loss = model(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 1000 == 0 or step == max_steps - 1:
            L = estimate_loss(model)
            print(f"step {step:5d} | train {L['train']:.4f} | val {L['val']:.4f}")

    print("\n(bigram baseline on this data is ~2.5; one attention head should beat it)\n")
    print("--- sample ---")
    start = torch.zeros((1, 1), dtype=torch.long)            # start from a newline (id 0)
    print(decode(model.generate(start, max_new_tokens=400)[0].tolist()))


if __name__ == "__main__":
    main()
