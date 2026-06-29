"""
E02 — Visualize the attention weights.
======================================
Train a small head, then plot its (T, T) attention matrix for one sequence. The black
upper triangle is the masked future; brighter cells are where each token "looks."

Run:  python e02_visualize.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))

import torch
import torch.nn as nn
import torch.nn.functional as F
import attention as A

vocab_size, train_data, val_data, itos = A.vocab_size, A.train_data, A.val_data, A.itos
torch.manual_seed(1337)
block_size, n_embd, batch_size, steps = 16, 32, 32, 1500


def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    return (torch.stack([d[i:i + block_size] for i in ix]),
            torch.stack([d[i + 1:i + block_size + 1] for i in ix]))


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.last_wei = None

    def forward(self, x):
        B, T, C = x.shape
        k, q = self.key(x), self.query(x)
        wei = q @ k.transpose(-2, -1) * C ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        self.last_wei = wei.detach()          # stash for inspection
        return wei @ self.value(x)


class LM(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, n_embd)
        self.pos = nn.Embedding(block_size, n_embd)
        self.head = Head()
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T))
        logits = self.lm_head(self.head(x))
        loss = None if targets is None else F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss


model = LM()
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
print(f"training a small head ({steps} steps)...")
for i in range(steps):
    xb, yb = get_batch("train")
    _, loss = model(xb, yb)
    opt.zero_grad(); loss.backward(); opt.step()

xb, _ = get_batch("val")
model(xb[:1])                                  # one forward pass to populate last_wei
wei = model.head.last_wei[0]                   # (T, T)
labels = [itos[i] for i in xb[0].tolist()]

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 7))
    plt.imshow(wei, cmap="viridis")
    plt.xticks(range(block_size), labels, fontsize=8)
    plt.yticks(range(block_size), labels, fontsize=8)
    plt.xlabel("attends to (key position)")
    plt.ylabel("query position")
    plt.colorbar(); plt.title("Attention weights — one head")
    plt.tight_layout(); plt.savefig("attention.png", dpi=120)
    print("saved attention.png — the upper triangle is black (the masked future).")
except ImportError:
    print("(install matplotlib to see the heatmap)")
