"""
E04 — A KV-cache memory table.
==============================
The cache's size = 2 (K and V) x n_layers x n_heads x head_dim x seq_len x batch x bytes. Build a
table across real models and contexts, and see where the cache overtakes the model's own weights.

Run:  python e04_memory.py
"""


def kv_cache_gb(n_layers, n_heads, head_dim, seq_len, batch=1, bytes_per=2):
    return 2 * n_layers * n_heads * head_dim * seq_len * batch * bytes_per / 1e9


# (name, n_layers, n_heads, head_dim, approx weights GB in bf16)
MODELS = [
    ("GPT-2 (124M)", 12, 12, 64, 0.25),
    ("Llama-2-7B", 32, 32, 128, 14),
    ("Llama-2-70B", 80, 64, 128, 140),
]
CONTEXTS = [2048, 8192, 32768]

print(f"KV-cache size (bf16, batch 1), and vs the model's weights:\n")
print(f"{'model':>16} {'weights':>9} | " + " ".join(f"{c:>9}" for c in CONTEXTS))
for name, nl, nh, hd, wgt in MODELS:
    cells = [f"{kv_cache_gb(nl, nh, hd, c):>8.1f}G" for c in CONTEXTS]
    print(f"{name:>16} {wgt:>7.2f}G | " + " ".join(f"{c:>9}" for c in cells))

big = kv_cache_gb(32, 32, 128, 32768)
print(f"\nLlama-2-7B's cache at 32k context is {big:.0f} GB — larger than its ~14 GB of weights.")
print("That's why long context is the expensive part of inference, and why GQA/paging exist.")
