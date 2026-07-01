"""
E01 — Implement the cache (STARTER scaffold).
=============================================
Fill the TODO: append each new token's key/value to the running cache, so attention from the cache
matches attention from scratch. Reference: ../solutions/e01_implement_cache.py.

Run:  python e01_implement_cache.py
"""
import torch
import torch.nn as nn

torch.manual_seed(0)
D = 16
Wq, Wk, Wv = (nn.Linear(D, D, bias=False) for _ in range(3))
seq = torch.randn(6, D)

# FULL attention for the last token — the naive answer to match:
K_all, V_all, q_last = Wk(seq), Wv(seq), Wq(seq[-1:])
full = torch.softmax(q_last @ K_all.T / D ** 0.5, dim=-1) @ V_all

cache_k, cache_v = None, None
for t in range(6):
    kt, vt = Wk(seq[t:t + 1]), Wv(seq[t:t + 1])              # only THIS token's k, v
    # ✍️ TODO: append kt, vt to cache_k / cache_v (handle the first token, when the cache is None).
    #          hint: torch.cat([cache_k, kt]) if cache_k is not None else kt
    cache_k = None      # replace
    cache_v = None      # replace

if cache_k is None:
    raise SystemExit("Fill the TODO (append kt, vt to the cache), then run again.")

cached = torch.softmax(q_last @ cache_k.T / D ** 0.5, dim=-1) @ cache_v
assert torch.allclose(full, cached, atol=1e-6), "cached != full — check the append order (past, then new)"
print("✓ Cached attention matches full attention exactly — the cache is a lossless optimization.")
