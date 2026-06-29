"""
Shared, configurable GPT for the Chapter 5 exercises.
=====================================================
A compact version of code/gpt.py whose architecture you can toggle: residual connections on/off,
LayerNorm on/off, number of layers/heads, and the feed-forward activation. The exercises import
`train` from here and flip one knob.

(Uses the data + vocab from code/gpt.py. Small config so each run takes ~15-20s on a CPU.)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))

import torch
import torch.nn as nn
import torch.nn.functional as F
import gpt as G

vocab_size, train_data, val_data = G.vocab_size, G.train_data, G.val_data


def get_batch(split, block_size, batch_size=32):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    return (torch.stack([d[i:i + block_size] for i in ix]),
            torch.stack([d[i + 1:i + block_size + 1] for i in ix]))


class Head(nn.Module):
    def __init__(self, n_embd, head_size, block_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k, q = self.key(x), self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        return F.softmax(wei, dim=-1) @ self.value(x)


class MultiHead(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        hs = n_embd // n_head
        self.heads = nn.ModuleList([Head(n_embd, hs, block_size) for _ in range(n_head)])
        self.proj = nn.Linear(hs * n_head, n_embd)

    def forward(self, x):
        return self.proj(torch.cat([h(x) for h in self.heads], dim=-1))


class FeedForward(nn.Module):
    def __init__(self, n_embd, activation):
        super().__init__()
        act = {"relu": nn.ReLU(), "gelu": nn.GELU(), "tanh": nn.Tanh()}[activation]
        self.net = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), act, nn.Linear(4 * n_embd, n_embd))

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, residual, norm, activation):
        super().__init__()
        self.sa = MultiHead(n_embd, n_head, block_size)
        self.ffwd = FeedForward(n_embd, activation)
        self.ln1 = nn.LayerNorm(n_embd) if norm else nn.Identity()
        self.ln2 = nn.LayerNorm(n_embd) if norm else nn.Identity()
        self.residual = residual

    def forward(self, x):
        a = self.sa(self.ln1(x))
        x = x + a if self.residual else a            # ← the residual connection (toggle)
        f = self.ffwd(self.ln2(x))
        x = x + f if self.residual else f
        return x


class GPT(nn.Module):
    def __init__(self, n_layer, n_head, n_embd, block_size, residual, norm, activation):
        super().__init__()
        self.block_size = block_size
        self.tok = nn.Embedding(vocab_size, n_embd)
        self.pos = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, residual, norm, activation)
                                      for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) if norm else nn.Identity()
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T))
        logits = self.lm_head(self.ln_f(self.blocks(x)))
        loss = None if targets is None else F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss


def train(n_layer=4, n_head=4, n_embd=64, block_size=32,
          residual=True, norm=True, activation="relu", steps=1500, lr=3e-4, seed=1337):
    """Train a configurable GPT and return (val_loss, model)."""
    torch.manual_seed(seed)
    model = GPT(n_layer, n_head, n_embd, block_size, residual, norm, activation)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for _ in range(steps):
        _, loss = model(*get_batch("train", block_size))
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        val = sum(model(*get_batch("val", block_size))[1].item() for _ in range(30)) / 30
    return val, model
