"""
E01 — Implement the cache (and prove it's exact).
=================================================
The whole KV-cache trick in miniature: instead of recomputing every past token's key/value, keep a
running cache and append only the new token's. Prove that attention computed from the cache equals
attention computed from scratch.

Run:  python e01_implement_cache.py
"""
import torch
import torch.nn as nn

torch.manual_seed(0)
D = 16
Wq, Wk, Wv = (nn.Linear(D, D, bias=False) for _ in range(3))
seq = torch.randn(6, D)                       # 6 token "embeddings"

# FULL attention for the last token — recompute k/v for ALL tokens (the naive way):
K_all, V_all, q_last = Wk(seq), Wv(seq), Wq(seq[-1:])
full = torch.softmax(q_last @ K_all.T / D ** 0.5, dim=-1) @ V_all

# CACHED — process one token at a time, appending to a running cache (never recompute the past):
cache_k, cache_v = None, None
for t in range(6):
    kt, vt = Wk(seq[t:t + 1]), Wv(seq[t:t + 1])              # only THIS token's k, v
    cache_k = torch.cat([cache_k, kt]) if cache_k is not None else kt
    cache_v = torch.cat([cache_v, vt]) if cache_v is not None else vt

cached = torch.softmax(q_last @ cache_k.T / D ** 0.5, dim=-1) @ cache_v

print(f"cache holds {cache_k.shape[0]} keys/values (one per token)")
print(f"cached attention == full attention: {torch.allclose(full, cached, atol=1e-6)}")
assert torch.allclose(full, cached, atol=1e-6)
print("The cache stores EXACTLY what naive recomputes — so the output is identical, just cheaper.")
