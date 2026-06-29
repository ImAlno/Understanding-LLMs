"""
E03 — tanh vs GELU.
===================
Same model, two activations. GELU is what GPT-style models use.

Run:  python e03_gelu.py
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


def run(activation):
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
        h = activation(emb.view(emb.shape[0], -1) @ W1 + b1)   # the only line that changes
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


print(f"  tanh  ->  dev {run(torch.tanh):.4f}")
print(f"  GELU  ->  dev {run(F.gelu):.4f}")
print("\nGELU is the activation in GPT-style models. Here it's usually a small win;")
print("its advantage grows in the deeper networks of later chapters.")
