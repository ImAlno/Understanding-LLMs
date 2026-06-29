"""
E05 — A deeper MLP (two hidden layers).
=======================================
Add a second hidden layer, and compare it to the 1-layer baseline AT MATCHED COMPUTE (the
same number of steps) — otherwise the comparison isn't fair.

Run:  python e05_deeper.py
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

block_size, n_embd, n_hidden, steps = 3, 10, 200, 8000
Xtr, Ytr = build_dataset(words[:n1], stoi, block_size)
Xdev, Ydev = build_dataset(words[n1:n2], stoi, block_size)


def run(deep):
    g = torch.Generator().manual_seed(2147483647)
    C = torch.randn((V, n_embd), generator=g)
    W1 = torch.randn((block_size * n_embd, n_hidden), generator=g) * 0.2
    b1 = torch.randn(n_hidden, generator=g) * 0.01
    params = [C, W1, b1]
    if deep:                                                   # add a 2nd hidden layer
        W2b = torch.randn((n_hidden, n_hidden), generator=g) * 0.1
        b2b = torch.randn(n_hidden, generator=g) * 0.01
        params += [W2b, b2b]
    Wo = torch.randn((n_hidden, V), generator=g) * 0.01        # output layer
    bo = torch.zeros(V)
    params += [Wo, bo]
    for p in params:
        p.requires_grad = True

    def fwd(X):
        emb = C[X]
        h = torch.tanh(emb.view(emb.shape[0], -1) @ W1 + b1)
        if deep:
            h = torch.tanh(h @ W2b + b2b)                      # the extra layer
        return h @ Wo + bo

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


print(f"  1 hidden layer   ->  dev {run(False):.4f}")
print(f"  2 hidden layers  ->  dev {run(True):.4f}")
print(f"\n(Both trained for {steps} steps — a fair comparison. Depth gives a small edge here")
print(" at best; it doesn't reliably pay off until the architecture and training tricks of")
print(" Chapters 5 and 7. More layers is not a free win.)")
