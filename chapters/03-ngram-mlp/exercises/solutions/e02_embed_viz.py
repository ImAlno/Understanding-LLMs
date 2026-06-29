"""
E02 — Visualize the learned 2-D embeddings.
===========================================
Train with n_embd=2 so we can plot the 27 character embeddings on a flat graph, each
labelled by its character. You'll see structure the model invented on its own.

Run:  python e02_embed_viz.py
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
n1 = int(0.8 * len(words))

block_size, n_embd, n_hidden, steps = 3, 2, 200, 12000
Xtr, Ytr = build_dataset(words[:n1], stoi, block_size)

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


print(f"training with n_embd=2 ({steps} steps)...")
for i in range(steps):
    ix = torch.randint(0, Xtr.shape[0], (32,), generator=g)
    loss = F.cross_entropy(fwd(Xtr[ix]), Ytr[ix])
    for p in params:
        p.grad = None
    loss.backward()
    lr = 0.1 if i < 0.6 * steps else 0.01
    for p in params:
        p.data += -lr * p.grad

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Cd = C.detach()
    plt.figure(figsize=(8, 8))
    plt.scatter(Cd[:, 0], Cd[:, 1], s=300, c="lightblue")
    for i in range(V):
        plt.text(Cd[i, 0].item(), Cd[i, 1].item(), itos[i],
                 ha="center", va="center", fontsize=13)
    plt.title("Learned 2-D character embeddings")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("embeddings.png", dpi=120)
    print("saved embeddings.png — look for the vowels (a, e, i, o, u) clustering together.")
except ImportError:
    print("(install matplotlib to see the plot)")
