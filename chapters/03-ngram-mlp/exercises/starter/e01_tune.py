"""
E01 — Tune the MLP on the dev set (STARTER scaffold).
=====================================================
Fill in the TODOs in `run(...)`, then sweep a few configs. A full reference is in
../solutions/e01_tune.py.

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
    """Build data, init params, train, and return the DEV loss for this config."""
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
        # ✍️ TODO #1: compute the loss for this minibatch (cross_entropy of fwd(...) vs targets)
        loss = None
        for p in params:
            p.grad = None
        # ✍️ TODO #2: backprop, then nudge every param downhill (lr is below)
        lr = 0.1 if i < 0.6 * steps else 0.01
        # ...

    with torch.no_grad():
        return F.cross_entropy(fwd(Xdev), Ydev).item()


# ✍️ TODO #3: call run(...) for a few configs (vary ONE thing at a time) and print the dev
#    losses, e.g. baseline vs block_size=4 vs a bigger n_hidden.
print("baseline dev loss:", run())
