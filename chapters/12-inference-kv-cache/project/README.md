# ⚡ Mini-Project — KV-Cache Your GPT

You've built the cache in miniature; now **add it to a real GPT's generation** and claim the two
payoffs that matter: the output is **unchanged**, and the tokens-per-second **jump**.

> 💻 **No GPU needed** — generation is forward passes; the speedup shows up on a CPU.

> **How this works:** [`starter/kv_cache_gpt.py`](./starter/kv_cache_gpt.py) is a self-contained GPT
> with `generate_naive` and `generate_cached` already wired (prefill + decode). It has **one TODO** —
> implement the actual KV-cache *inside the attention* (append the new k/v to the past, save it back).
> A reference is in [`solution/kv_cache_gpt.py`](./solution/kv_cache_gpt.py).

## 🎯 What it does

```bash
python starter/kv_cache_gpt.py
```

Generates the same tokens two ways — naive (recompute everything) and cached — checks they're
**identical**, then benchmarks **tokens/second** for each. With the cache implemented you'll see the
same output at several times the throughput (~4–5× on this small model; far more on a real one at
long context).

## 🛠️ Your TODO

In `Attn.forward`, implement the cache when one is passed in:

```python
if cache["k"] is not None:
    k = torch.cat([cache["k"], k], dim=2)      # append the new keys onto the cached past...
    v = torch.cat([cache["v"], v], dim=2)      # ...and values
cache["k"], cache["v"] = k, v                  # save, so the next decode step reuses them
```

Until you do, `generate_cached` forgets the past (each decode step sees only the current token), its
output differs from `generate_naive`, and the script refuses to run — no false success.

## ✅ Checking your work

- **Identical output:** the script asserts `generate_cached == generate_naive`. This is the
  correctness test — the cache must not change the answer.
- **Faster:** the cached generator should report several × more tokens/second than naive, and the gap
  grows with how many tokens you generate.
- Compare against [`solution/kv_cache_gpt.py`](./solution/kv_cache_gpt.py).

## 🚀 Stretch

- **Plug it into your Chapter 5 GPT.** Add the same cache-aware attention + prefill/decode loop to
  your trained Storyteller and watch a long story generate snappily — with the exact same text.
- **Sampling, not greedy.** Swap `argmax` for `torch.multinomial` on the softmax; the cache is
  unchanged (it only speeds the attention, not the token choice).
- **Measure the memory.** Print the cache's size (`2 × n_layers × n_heads × head_dim × seq_len × 2`
  bytes) as generation grows, and watch it climb linearly.
- **Grouped-Query Attention.** Share one k/v across every 2 heads and confirm the cache halves — the
  real trick behind long-context models.

Next: [Chapter 13 — Inference II: Quantization](../../13-inference-quantization/), the *other* half of
cheap inference — shrinking the weights themselves. ⚡
