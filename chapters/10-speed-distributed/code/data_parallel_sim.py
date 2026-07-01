"""
Why data parallelism is *correct* — proven on one CPU, no multiprocessing.
=========================================================================
DDP splits a batch across N workers and AVERAGES their gradients. The reason that's allowed: the
average of the N shard-gradients is **exactly** the gradient you'd get from the whole batch on one
worker. So DDP changes only the *speed*, never the answer. Here's the proof on real numbers — no
processes, no gloo, just linear algebra.

Run:  python data_parallel_sim.py
"""
import torch
import torch.nn as nn


def grad_of(model, x, y):
    """Return the flattened gradient of the mean-squared-error over (x, y)."""
    model.zero_grad()
    loss = ((model(x) - y) ** 2).mean()
    loss.backward()
    return torch.cat([p.grad.flatten() for p in model.parameters()])


torch.manual_seed(0)
model = nn.Linear(4, 2)
X = torch.randn(16, 4)
Y = torch.randn(16, 2)

# The single-worker answer: the gradient over the whole 16-row batch.
full = grad_of(model, X, Y)

# The DDP way: split into N equal shards, take each shard's gradient, average them.
N = 4
shard_grads = [grad_of(model, X[r::N], Y[r::N]) for r in range(N)]   # rank r sees rows r, r+N, …
averaged = torch.stack(shard_grads).mean(dim=0)

print(f"full-batch gradient   (first 4 numbers): {[round(v, 4) for v in full[:4].tolist()]}")
print(f"avg of {N} shard grads (first 4 numbers): {[round(v, 4) for v in averaged[:4].tolist()]}")
print(f"\nmax difference: {(full - averaged).abs().max().item():.2e}")
print("→ identical. Averaging N shard-gradients == the full-batch gradient — that's why DDP is exact,")
print("  and why N workers give you an effective batch N× larger at (ideally) N× the speed.")
