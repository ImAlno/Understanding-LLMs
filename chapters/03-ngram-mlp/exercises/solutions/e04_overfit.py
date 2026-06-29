"""
E04 — Watch a model overfit.
============================
Train on only 200 names (full batch, many steps). The train loss dives toward 0 while the
dev loss stays high — the model memorizes instead of generalizing. That gap is overfitting.

Run:  python e04_overfit.py
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

block_size, n_embd, n_hidden = 3, 10, 200
tiny = words[:200]                                   # a deliberately tiny training set
Xtr, Ytr = build_dataset(tiny, stoi, block_size)
Xdev, Ydev = build_dataset(words[n1:n2], stoi, block_size)

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


print(f"training on only {len(tiny)} names ({Xtr.shape[0]} examples), full batch:\n")
for i in range(3000):
    loss = F.cross_entropy(fwd(Xtr), Ytr)            # full batch (the set is tiny)
    for p in params:
        p.grad = None
    loss.backward()
    for p in params:
        p.data += -0.1 * p.grad
    if i % 500 == 0 or i == 2999:
        with torch.no_grad():
            dev = F.cross_entropy(fwd(Xdev), Ydev).item()
        print(f"  step {i:4d} | train {loss.item():.3f} | dev {dev:.3f}")

print("\nTrain loss dives (it memorizes 200 names); dev loss stays high / rises.")
print("That divergence is overfitting — exactly why we judge models on held-out data.")
