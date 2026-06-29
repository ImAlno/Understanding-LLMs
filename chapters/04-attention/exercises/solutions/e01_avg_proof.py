"""
E01 — Prove the triangular matmul averages the past.
====================================================
Two ways to compute, for every position, the running average of all positions up to it: an
explicit double loop, and one matrix multiply. They must agree.

Run:  python e01_avg_proof.py
"""
import torch

torch.manual_seed(0)
B, T, C = 2, 6, 4
x = torch.randn(B, T, C)

# Method 1 — explicit loop: position t = mean of x over positions 0..t
xbow_loop = torch.zeros(B, T, C)
for b in range(B):
    for t in range(T):
        xbow_loop[b, t] = x[b, :t + 1].mean(dim=0)

# Method 2 — the matrix trick: a lower-triangular, row-normalized weight matrix
tril = torch.tril(torch.ones(T, T))
wei = tril / tril.sum(1, keepdim=True)
xbow_matmul = wei @ x

match = torch.allclose(xbow_loop, xbow_matmul, atol=1e-6)
print(f"  loop vs matmul match: {match}")
print(f"  max difference: {(xbow_loop - xbow_matmul).abs().max().item():.2e}")
print("\nThe triangular matmul does exactly what the loop does — in one fast op, and (unlike")
print("the loop) with weights we can later make learned and data-dependent. That's attention.")
