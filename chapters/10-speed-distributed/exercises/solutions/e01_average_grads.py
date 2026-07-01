"""
E01 — Average the gradients (the heart of DDP).
===============================================
DDP splits a batch across ranks and averages their gradients. Prove, in one process, that this
average equals the gradient over the whole batch — so DDP gives the same answer, just in parallel.

Run:  python e01_average_grads.py
"""
import torch
import torch.nn as nn


def grad_of(model, x, y):
    model.zero_grad()
    (((model(x) - y) ** 2).mean()).backward()
    return torch.cat([p.grad.flatten() for p in model.parameters()])


torch.manual_seed(0)
model = nn.Linear(4, 2)
X = torch.randn(12, 4)
Y = torch.randn(12, 2)

full = grad_of(model, X, Y)                                   # whole-batch gradient (one worker)

N = 3
shard_grads = [grad_of(model, X[r::N], Y[r::N]) for r in range(N)]
averaged = torch.stack(shard_grads).mean(dim=0)               # the all-reduce AVG

diff = (full - averaged).abs().max().item()
print(f"full-batch grad   (first 3): {[round(v, 4) for v in full[:3].tolist()]}")
print(f"averaged shard grad (first 3): {[round(v, 4) for v in averaged[:3].tolist()]}")
print(f"max difference: {diff:.2e}")
print("MATCH — averaging N shard-gradients == the full-batch gradient." if diff < 1e-5 else "mismatch!")
