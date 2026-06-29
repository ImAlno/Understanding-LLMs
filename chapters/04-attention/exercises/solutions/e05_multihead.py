"""
E05 — Multi-head attention (a first taste of the Transformer).
==============================================================
Run several smaller heads in parallel and concatenate them, so the model can attend to
different things at once. Compare to the single head.

Run:  python e05_multihead.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))

import torch
import torch.nn as nn
import torch.nn.functional as F
import attention as A

vocab_size, train_data, val_data = A.vocab_size, A.train_data, A.val_data
block_size, n_embd, batch_size, steps = 16, 64, 32, 2500


def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    return (torch.stack([d[i:i + block_size] for i in ix]),
            torch.stack([d[i + 1:i + block_size + 1] for i in ix]))


class Head(nn.Module):
    def __init__(self, head_size):
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
        wei = F.softmax(wei, dim=-1)
        return wei @ self.value(x)


class MultiHead(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd // n_heads) for _ in range(n_heads)])

    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1)   # concat back to n_embd


class LM(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, n_embd)
        self.pos = nn.Embedding(block_size, n_embd)
        self.sa = MultiHead(n_heads)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T))
        logits = self.lm_head(self.sa(x))
        loss = None if targets is None else F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss


def run(n_heads):
    torch.manual_seed(1337)
    model = LM(n_heads)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for i in range(steps):
        _, loss = model(*get_batch("train"))
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return sum(model(*get_batch("val"))[1].item() for _ in range(20)) / 20


print(f"  1 head   (head_size {n_embd})  ->  val {run(1):.4f}")
print(f"  4 heads  (head_size {n_embd // 4})   ->  val {run(4):.4f}")
print("\nMulti-head lets the model attend to several patterns at once — a core piece of the")
print("Transformer you'll assemble in Chapter 5.")
