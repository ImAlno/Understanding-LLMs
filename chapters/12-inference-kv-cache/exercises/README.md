# Chapter 12 — Exercises

Build the KV-cache and feel its payoff. **Try first**; solutions are in [`solutions/`](./solutions/).
Everything runs on a CPU in seconds (E03 takes a few). E02–E04 import the GPT from
[`../code/gpt.py`](../code/gpt.py).

```bash
python chapters/12-inference-kv-cache/exercises/solutions/e01_implement_cache.py
```

---

### E01 — Implement the cache 🥇
The trick in miniature: append each new token's key/value to a running cache, and prove attention
from the cache equals attention from scratch.

- 🧩 **Scaffold:** [`starter/e01_implement_cache.py`](./starter/e01_implement_cache.py).
- ✅ **Solution:** [`solutions/e01_implement_cache.py`](./solutions/e01_implement_cache.py).

  <details><summary>💡 Hint</summary>

  `cache_k = torch.cat([cache_k, kt]) if cache_k is not None else kt` — the `else` handles the very
  first token, when the cache is still empty.
  </details>

### E02 — Cached == naive (the correctness test)
On the full GPT, prove greedy `generate_cached` produces the *identical* tokens as `generate_naive`,
across several prompt lengths. The cache is an optimization, so identical output is how you know it's
right.

- ✅ Solution: [`solutions/e02_identical.py`](./solutions/e02_identical.py).

  <details><summary>💡 Hint</summary>

  `torch.equal(model.generate_naive(p, 100), model.generate_cached(p, 100))` should be `True`.
  </details>

### E03 — The speedup grows with length
Time naive vs cached at 64, 128, 256, 480 tokens and watch the ratio climb (naive is O(T²), cached is
O(T), so the win grows roughly like T).

- ✅ Solution: [`solutions/e03_speedup.py`](./solutions/e03_speedup.py).

  <details><summary>💡 Hint</summary>

  Warm up once, then time a single call. The exact ms vary run to run; the *rising trend* is the point.
  </details>

### E04 — A KV-cache memory table
Compute the cache size across models (GPT-2, Llama-2-7B/70B) and contexts, and see where it overtakes
the model's own weights.

- ✅ Solution: [`solutions/e04_memory.py`](./solutions/e04_memory.py).

  <details><summary>💡 Hint</summary>

  `2 × n_layers × n_heads × head_dim × seq_len × batch × bytes_per / 1e9`. Llama-2-7B at 32k ≈ 17 GB,
  bigger than its ~14 GB of weights.
  </details>

> 🧠 **Takeaway:** the cache appends the new k/v and reuses the rest (exact, not approximate); it
> makes generation O(T) instead of O(T²) (speedup grows with length); and its price is memory that
> scales with length × batch — eventually dwarfing the weights.
