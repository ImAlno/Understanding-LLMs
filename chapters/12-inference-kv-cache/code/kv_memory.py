"""
The KV-cache's price: memory.
=============================
The cache trades memory for speed. It stores, for every token and every layer, that token's key
AND value vectors — so its size grows *linearly* with the sequence length:

    cache bytes = 2 (K and V) × n_layers × n_heads × head_dim × seq_len × batch × bytes_per_number

For a big model at a long context, that's gigabytes — often more than the model's own weights, and
the main reason long-context inference is expensive. This computes it.

Run:  python kv_memory.py
"""


def kv_cache_gb(n_layers, n_heads, head_dim, seq_len, batch=1, bytes_per=2):
    """bytes_per=2 for bf16/fp16 (the usual inference dtype)."""
    return 2 * n_layers * n_heads * head_dim * seq_len * batch * bytes_per / 1e9


print("KV-cache size (bf16):\n")
print(f"  our tiny model (3 layers), 512 ctx      : {kv_cache_gb(3, 4, 32, 512) * 1000:.2f} MB")
print(f"  Llama-2-7B  (32 layers),  4,096 ctx     : {kv_cache_gb(32, 32, 128, 4096):.2f} GB")
print(f"  Llama-2-7B  (32 layers), 32,768 ctx     : {kv_cache_gb(32, 32, 128, 32768):.2f} GB")
print(f"  Llama-2-7B,  4,096 ctx, batch of 32     : {kv_cache_gb(32, 32, 128, 4096, batch=32):.1f} GB")

print("\nThe 7B model's *weights* are ~14 GB (bf16). At 32k context its KV-cache alone is ~17 GB —")
print("bigger than the model. This is why long context is costly, and why the next tricks exist:")
print("  • Multi-Query / Grouped-Query Attention (MQA/GQA): share K/V across heads -> smaller cache")
print("  • PagedAttention (vLLM): don't pre-allocate the whole cache; page it like virtual memory")
