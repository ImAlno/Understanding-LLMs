"""
E01 — Tune the MLP on the dev set.
==================================
Wrap "build data → init → train → dev loss" in a `run(...)` and sweep a few configs.
Change one thing at a time; pick the winner by DEV loss; touch test only at the very end.

Run:  python e01_tune.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))

import random
import torch
import torch.nn.functional as F
from mlp import load_words, build_vocab, build_dataset

words = load_words()
stoi, itos = build_vocab(words)
V = len(stoi)
random.seed(42)
random.shuffle(words)
n1, n2 = int(0.8 * len(words)), int(0.9 * len(words))
train_words, dev_words = words[:n1], words[n1:n2]


def run(block_size=3, n_embd=10, n_hidden=200, steps=8000):
    Xtr, Ytr = build_dataset(train_words, stoi, block_size)
    Xdev, Ydev = build_dataset(dev_words, stoi, block_size)
    g = torch.Generator().manual_seed(2147483647)
    C = torch.randn((V, n_embd), generator=g)
    W1 = torch.randn((block_size * n_embd, n_hidden), generator=g) * 0.2
    b1 = torch.randn(n_hidden, generator=g) * 0.01
    W2 = torch.randn((n_hidden, V), generator=g) * 0.01
    b2 = torch.zeros(V)
    params = [C, W1, b1, W2, b2]
    for p in params:
        p.requires_grad = True

    def fwd(X):
        emb = C[X]
        h = torch.tanh(emb.view(emb.shape[0], -1) @ W1 + b1)
        return h @ W2 + b2

    for i in range(steps):
        ix = torch.randint(0, Xtr.shape[0], (32,), generator=g)
        loss = F.cross_entropy(fwd(Xtr[ix]), Ytr[ix])
        for p in params:
            p.grad = None
        loss.backward()
        lr = 0.1 if i < 0.6 * steps else 0.01
        for p in params:
            p.data += -lr * p.grad
    with torch.no_grad():
        return F.cross_entropy(fwd(Xdev), Ydev).item()


print("sweeping configs (dev loss — lower is better):\n")
configs = [
    dict(block_size=3, n_embd=10, n_hidden=200),   # baseline
    dict(block_size=4, n_embd=10, n_hidden=200),   # more context
    dict(block_size=4, n_embd=15, n_hidden=300),   # bigger net + richer embedding
]
for cfg in configs:
    dev = run(**cfg)
    print(f"  block_size={cfg['block_size']}  n_embd={cfg['n_embd']:>2}  "
          f"n_hidden={cfg['n_hidden']:>3}  ->  dev {dev:.4f}")
print("\nMore context + a bigger net usually helps (up to a point). Tune on dev; test once.")
