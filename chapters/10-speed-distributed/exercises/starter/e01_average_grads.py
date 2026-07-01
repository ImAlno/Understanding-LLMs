"""
E01 — Average the gradients (STARTER scaffold).
===============================================
Fill the TODO: compute the all-reduce AVERAGE of the shard gradients, and confirm it equals the
full-batch gradient. Reference: ../solutions/e01_average_grads.py.

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

# ✍️ TODO: average the N shard gradients across ranks (hint: torch.stack(...).mean(dim=0))
averaged = None      # replace

if averaged is None:
    raise SystemExit("Fill the TODO (average the shard gradients), then run again.")

diff = (full - averaged).abs().max().item()
print(f"max difference vs full-batch gradient: {diff:.2e}")
print("MATCH — DDP is exact." if diff < 1e-5 else "mismatch — average across the shards (dim=0).")
