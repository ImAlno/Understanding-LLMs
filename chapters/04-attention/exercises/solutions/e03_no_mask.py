"""
E03 — Break the causal mask.
============================
Train two identical heads — one causal (masked), one not. Without the mask, every token can
see the future and the model "cheats" to a much lower loss... that means nothing, because at
generation time there is no future to read.

Run:  python e03_no_mask.py
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
    def __init__(self, causal):
        super().__init__()
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.causal = causal

    def forward(self, x):
        B, T, C = x.shape
        k, q = self.key(x), self.query(x)
        wei = q @ k.transpose(-2, -1) * C ** -0.5
        if self.causal:                                       # the line E03 removes
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        return wei @ self.value(x)


class LM(nn.Module):
    def __init__(self, causal):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, n_embd)
        self.pos = nn.Embedding(block_size, n_embd)
        self.head = Head(causal)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T))
        logits = self.lm_head(self.head(x))
        loss = None if targets is None else F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss


def run(causal):
    torch.manual_seed(1337)
    model = LM(causal)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for i in range(steps):
        _, loss = model(*get_batch("train"))
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return sum(model(*get_batch("val"))[1].item() for _ in range(20)) / 20


print(f"  WITH causal mask    (honest)   ->  val {run(True):.4f}")
print(f"  WITHOUT causal mask (cheating) ->  val {run(False):.4f}")
print("\nThe no-mask loss is much lower — but it's a lie: each token peeked at the next one.")
print("At generation time there IS no future to peek at, so a no-mask model is useless.")
