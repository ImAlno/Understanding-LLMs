"""
E01 — Prove the matmul trick = a loop (STARTER scaffold).
=========================================================
Fill the 2 TODOs. A full reference is in ../solutions/e01_avg_proof.py.

Run:  python e01_avg_proof.py
"""
import torch

torch.manual_seed(0)
B, T, C = 2, 6, 4
x = torch.randn(B, T, C)

# ✍️ TODO #1: explicit loop — set xbow_loop[b, t] to the MEAN of x[b] over positions 0..t
xbow_loop = torch.zeros(B, T, C)
for b in range(B):
    for t in range(T):
        pass   # your line here: the running mean up to and including position t

# ✍️ TODO #2: the matrix trick — build a lower-triangular matrix, normalize each row to sum
#             to 1, then matrix-multiply it with x
xbow_matmul = None   # replace

if xbow_matmul is None:
    raise SystemExit("Fill in TODO #2 (and TODO #1), then run again.")
print("match:", torch.allclose(xbow_loop, xbow_matmul, atol=1e-6))
