# Chapter 12 — Inference I: The KV-Cache

> You've spent eleven chapters *training*. Now we switch to *inference* — making a trained model
> **generate** — and the first thing you notice is that the obvious way is painfully slow. A
> Transformer generates one token at a time, and the naive loop re-processes the *entire* sequence
> on every single step, redoing work it already did. The **KV-cache** fixes it with one observation:
> under causal attention, a past token's keys and values *never change*, so you compute them once and
> **cache** them. That turns generation from `O(T²)` recomputation into a fast `O(T)` incremental
> decode — the trick behind every chat UI's snappy, token-by-token responses. We build it, prove it
> gives the *identical* output, and watch the speedup grow with length.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ImAlno/Understanding-LLMs/blob/main/chapters/12-inference-kv-cache/code/explore.ipynb)

> 💻 **No GPU needed.** Generation is just forward passes — it runs fine on a CPU, and the cache's
> speedup is visible right here on a laptop. (On a real GPU serving chats, the same trick is the
> difference between snappy and unusable.)

**You will be able to:**
- explain why **autoregressive generation** is slow the naive way — `O(T²)` work (growing with the
  *square* of the length);
- state the key insight — under a **causal mask**, past tokens' **keys and values never change**;
- implement a **KV-cache**: cache each token's k/v once, then each step compute only the *new*
  token's q/k/v and attend over the cache;
- distinguish the **prefill** and **decode** phases;
- reason about the cache's **memory cost** (it grows with sequence length and can exceed the model's
  own weights) and the **latency vs throughput** trade-off;
- place **MQA/GQA** and **PagedAttention** as the tricks that tame the cache at scale.

**Prerequisites:** attention — queries, keys, values, the causal mask (Chapter 4); the GPT and its
`generate` loop (Chapter 5); and Chapter 9's memory accounting (bytes per number).

**Time:** ~2 hours. **Hardware:** any laptop CPU.

> ### 📓 Build it yourself in the notebook
> **[`code/explore.ipynb`](./code/explore.ipynb)**: see the naive waste, **implement the cache**
> (append k/v, attend over the past), prove cached output equals naive, and compute the cache's
> memory. "✍️ Your turn", "▶️ Run this", "▶️ Check your work" cells. Watch for: ✏️ **In the notebook
> → Step N**.

---

## 0. Setup

Three scripts, all CPU, all self-contained:

- [`code/gpt.py`](./code/gpt.py) — a compact GPT with hand-written attention and **two** generators:
  `generate_naive` and `generate_cached`.
- [`code/benchmark.py`](./code/benchmark.py) — proves they give identical output, then times the
  speedup as length grows.
- [`code/kv_memory.py`](./code/kv_memory.py) — the cache's memory cost, from our toy model up to a 7B.

```bash
python chapters/12-inference-kv-cache/code/gpt.py         # naive == cached output
python chapters/12-inference-kv-cache/code/benchmark.py    # the speedup
```

---

## A 2-minute primer: generation is a loop

A GPT doesn't emit a sentence in one shot. It **generates autoregressively** — one token at a time,
each token fed back in to produce the next:

```
prompt ─► [GPT] ─► token1 ─► [GPT] ─► token2 ─► [GPT] ─► token3 ─► ...
                     (append each token and run the model again)
```

That loop is the token-by-token sampling you've done since Chapter 1, wrapped into the GPT's
`generate` in Chapter 5. It's correct — but *how* you run
each step matters enormously for speed, because a Transformer's attention lets every position look at
every earlier position. Do that carelessly and you redo a mountain of work. Do it with a cache and
each step is nearly free. This chapter is that one optimization, in full.

---

## 1. The problem: the naive loop recomputes everything

> ✏️ **In the notebook → Step 1.**

Here's the naive generator (essentially what you wrote in Chapter 5):

```python
for _ in range(n_new):
    logits = model(idx)                 # run the WHOLE sequence-so-far through every layer
    next_tok = logits[:, -1].argmax(-1) # we only use the LAST position's prediction
    idx = torch.cat([idx, next_tok], 1)
```

Look at what's wasted. To generate token 100, the model runs a full forward pass over all 99 previous
tokens — computing every layer's attention keys and values for all of them — and then **throws away
99/100 of the result**, keeping only the last position's next-token prediction. The next step redoes
all of it for 100 tokens, then 101, then 102… Generating `T` tokens does work proportional to
`1 + 2 + 3 + … + T ≈ T²/2` — we write that **`O(T²)`** ("order T-squared": the work grows with the
*square* of the length). That quadratic is why naive generation crawls as the text gets longer (the
cache will bring it down to **`O(T)`** — linear in the length).

Put numbers on it: generating a **1,000-token** response the naive way runs the model over
`1 + 2 + … + 1000 ≈ 500,000` token-positions in total. With the cache it's about **1,000** — each
token processed once. That ~500× gap (at this length) is pure recomputation you get to delete.

The waste is specific and fixable: at step 100, tokens 1–99 are *exactly the same tokens* they were
at step 99. We recomputed their keys and values for nothing.

---

## 2. The insight: past keys and values never change

> ✏️ **In the notebook → Step 2.**

Recall attention (Chapter 4): each token produces a **query**, a **key**, and a **value**; a token
attends by dotting its query against every (visible) key and taking the weighted sum of values. Under
the **causal mask**, token `t` only attends to tokens `0…t` — it *cannot* see the future.

That causal rule has a beautiful consequence. Token 50's key and value depend only on token 50 (and,
through earlier layers, tokens 0–50) — **nothing after it**. So when you later append token 51, 52,
100, …, token 50's key and value are *unchanged*. They were correct when you first computed them and
they stay correct forever.

So there's no reason to recompute them. Compute each token's key and value **once**, when the token
first appears, and **store** them. That store is the **KV-cache**.

---

## 3. The KV-cache: compute once, reuse forever

> ✏️ **In the notebook → Step 2.** Implement the cache.

The cache is just a running list of every past token's keys and values, kept per layer. With it, a
generation step shrinks to: compute q, k, v for **only the one new token**, append its k and v to the
cache, and attend the new query over **all** the cached keys/values. In code (from
[`gpt.py`](./code/gpt.py)):

```python
q = self.q(x); k = self.k(x); v = self.v(x)    # x is just the NEW token(s), not the whole sequence

if cache is not None:
    if cache["k"] is not None:
        k = torch.cat([cache["k"], k], dim=2)  # prepend all the cached past keys...
        v = torch.cat([cache["v"], v], dim=2)  # ...and values
    cache["k"], cache["v"] = k, v              # the cache now holds every key/value so far

att = (q @ k.transpose(-2, -1)) / sqrt(head_dim)   # new query attends over ALL keys (cached + new)
out = softmax(att) @ v
```

Visually, the cache grows by one column each step (**prefill** = load the whole prompt at once;
**decode** = generate one token at a time — both detailed in §4):

```
   prefill "The cat sat":   cache K = [k_The, k_cat, k_sat]              (3 prompt tokens at once)
   decode  " on":           cache K = [k_The, k_cat, k_sat, k_on]        + append k_on
   decode  " the":          cache K = [k_The, k_cat, k_sat, k_on, k_the] + append k_the
                                       └────────── reused, never recomputed ──────────┘
```

(and an identical row for the values). Each step adds *one* key and one value and reuses all the
earlier ones; the naive path rebuilds that entire row every single step.

That `torch.cat` is the whole trick: instead of recomputing every past key and value, you glue the
*one* new one onto the ones you already have. The new token's query then attends over the full
history — but the history came out of memory, not out of 99 redundant forward passes.

Two details make it correct:

- **Positions.** The new token still needs its *absolute* position for the position embedding and the
  causal mask (a token at position 100 must know it's position 100). We track a `pos_offset` so the
  single new token is placed correctly. (In `gpt.py`, the mask keeps `key_position ≤ query_position`,
  which reduces to "attend to everything cached" for a lone new token — it's the newest, so all
  cached keys are legitimately in its past.)
- **It's exact.** The cache stores *precisely* the keys and values the naive path recomputes, so the
  math is the same (to floating-point rounding) and the generated *tokens* are identical (§5). The
  cache is an optimization, not an approximation.

---

## 4. Prefill vs decode: the two phases

Real inference has two distinct phases, and the cache is what separates them:

- **Prefill.** You feed the whole prompt in *one* forward pass. This fills the cache with the prompt's
  keys and values for every layer, and produces the first generated token. It's one big parallel pass
  over `prompt_length` tokens — efficient, because attention over a known sequence parallelizes.
- **Decode.** Then you generate one token at a time, each step processing a *single* token through the
  model and attending over the cache. Each decode step is cheap — one token's worth of new
  computation, no growing *recomputation* (it does *read* the growing cache, §7, but never rebuild it).

To make **one decode step** concrete — you've generated "The cat sat on" and want the next token:

1. Take *just* the last token, `on` — a single-token input (not the whole sentence).
2. Embed it, with `pos_offset = 4` so it knows it's the 5th position.
3. In each layer: compute `on`'s q, k, v; append its k/v to that layer's cache (now holding the
   keys/values for `The, cat, sat, on`); the query `q_on` attends over all four.
4. Out come the next-token logits — produced with **one** token's worth of new work, not five.

Then you append the sampled token and repeat. Every step is that same cheap operation, no matter how
long the text already is. This is exactly the shape of `generate_cached`: one prefill call, then a
decode loop.

```python
caches = [{"k": None, "v": None} for _ in range(n_layer)]
logits = model(prompt, caches, pos_offset=0)          # PREFILL: whole prompt at once, fills the cache
token = logits[:, -1].argmax(-1)
for _ in range(n_new - 1):                             # DECODE: one token at a time
    logits = model(token, caches, pos_offset=current_length)
    token = logits[:, -1].argmax(-1)
```

The two phases have different performance characters — prefill is compute-heavy (lots of tokens at
once), decode is memory-bound (one token, but reading the whole growing cache each step) — which is
why serving systems tune them separately. A rough rule: a long prompt with a short answer is
*prefill-bound*; a short prompt with a long answer (a chat reply, a story) is *decode-bound* — and
decode is exactly where the KV-cache earns its keep.

---

## 5. It's the *same* output — just faster

> ✏️ **In the notebook → Step 3.** Prove it, then time it.

Because the cache stores exactly what the naive path recomputes, cached generation produces the
**identical tokens** — the logits agree to floating-point precision, and greedy `argmax` is robust to
that last-bit wiggle, so the chosen token ids come out *exactly* equal. That's the first thing
[`benchmark.py`](./code/benchmark.py) checks — and it's true:

```
identical output (naive vs cached): True
```

Then it times both as the number of generated tokens grows (real CPU output; exact ms vary run to
run, the *trend* is the point):

```
  tokens   naive ms   cached ms   speedup
      64         35          17      2.0x
     128         85          35      2.4x
     256        245          74      3.3x
     480        819         158      5.2x
```

The speedup **climbs with length** — because naive is `O(T²)` and cached is `O(T)`, so the ratio
grows roughly like `T`. Generate 64 tokens and the cache saves you 2×; generate 480 and it's over 5×;
generate the thousands of tokens a real chat response needs and it's the difference between a snappy
reply and a spinner. This is a rare optimization that is *both* faster *and* exact.

---

## 6. The price: memory

> ✏️ **In the notebook → Step 4.**

The cache isn't free — it trades **memory** for speed. It stores a key *and* a value vector for every
token, every layer, so its size grows **linearly with sequence length**:

```
cache bytes = 2 (K and V) × n_layers × n_heads × head_dim × seq_len × batch × bytes_per_number
```

[`kv_memory.py`](./code/kv_memory.py) puts real numbers on it (bf16, 2 bytes/number):

```
  our tiny model (3 layers), 512 ctx   : 0.79 MB
  Llama-2-7B  (32 layers),  4,096 ctx  : 2.15 GB
  Llama-2-7B  (32 layers), 32,768 ctx  : 17.18 GB
  Llama-2-7B,  4,096 ctx, batch of 32  : 68.7 GB
```

Sit with those. Llama-2-7B's *weights* are ~14 GB in bf16. At a 32k context, its **KV-cache alone is
~17 GB — bigger than the model itself.** And it scales with the batch size (serving 32 users at once
is 32 caches). This is the central tension of LLM serving: the cache is what makes generation fast,
but it's also the thing that fills your GPU memory and caps how long a context or how many users you
can handle. Long-context inference is expensive *because of the KV-cache.*

This reframes a limit you've probably hit: when a chat app caps your context length or how many
requests it takes at once, the wall it's usually hitting is **KV-cache memory**, not the model's
size. Doubling the context doubles the cache; the weights don't move.

---

## 7. Latency vs throughput

Serving isn't one number. Two matter, and they pull against each other:

- **Latency** — how fast *one* response comes back (tokens/second for a single user). The KV-cache is
  mostly a latency win: each decode step is cheap.
- **Throughput** — how many tokens/second you serve across *all* users at once. You raise it by
  **batching** many users' decode steps together (one big matmul instead of many small ones) — but
  every user in the batch needs their own KV-cache, so throughput is capped by cache memory (§6).

Concretely: one user's decode step is a tiny matmul that barely troubles the GPU; batch 32 users and
it's a single matmul 32× as large — nearly free extra throughput — *until* those 32 caches exhaust
memory. That's why §6's numbers matter so much: **cache memory literally sets how many users a GPU
can serve at once.**

The whole game of an inference server (vLLM, TGI) is juggling these: batch enough requests to keep the
GPU busy (throughput) without blowing the memory budget on caches or making any one user wait
(latency). The KV-cache sits at the center of that trade-off.

---

## 8. Taming the cache: MQA, GQA, PagedAttention

Because the cache is *the* bottleneck, a lot of cleverness targets it:

- **Multi-Query Attention (MQA)** and **Grouped-Query Attention (GQA)** — instead of a separate key
  and value per attention head, **share** them across heads (all heads, or small groups). Fewer
  distinct k/v means a proportionally **smaller cache** (GQA with 8 groups instead of 32 heads is a
  4× cache saving) for almost no quality loss. Modern models (Llama-2/3, Mistral) use GQA largely for
  this reason.
- **PagedAttention (vLLM)** — don't pre-allocate one big contiguous cache per request (which wastes
  memory on padding and fragments badly). Instead, store the cache in fixed-size **pages** and hand
  them out on demand, exactly like an operating system's virtual memory (the paging idea from Chapter
  11). This packs far more concurrent requests into the same GPU.

You won't implement these here, but they're all the same story: *the KV-cache is the cost of fast
generation, so squeeze it.*

---

## 🔌 How this plugs into the Storyteller

Your Chapter 5 GPT already generates — this makes it generate *fast*. Swap its plain sampling loop for
the prefill-then-decode-with-cache loop, and a long Storyteller response goes from quadratic-slow to
linear-snappy, with the exact same text. The mini-project does precisely this to your GPT and
benchmarks the win. Combined with Chapter 13's quantization next, it's the "make the trained model
cheap to *run*" half of the course — as important in practice as training it.

---

## 🐛 Building it yourself: what trips people up

- **Feeding the whole sequence during decode.** The point of the cache is to pass in *only the new
  token* each step. If you keep passing the full `idx`, you've kept the `O(T²)` cost *and* added a
  cache — slower, not faster.
- **Forgetting the position offset.** The new token needs its *absolute* position (for the position
  embedding and mask). Pass a `pos_offset`; treating every decoded token as "position 0" garbles the
  output.
- **Re-creating the cache each step.** The cache must **persist** across the whole decode loop (it's
  the accumulated past). Allocate it once before the loop, not inside it.
- **Expecting different output.** Cached and naive must match *exactly* (greedy) — if they don't, your
  cache logic is wrong. It's an optimization, so identical output is the correctness test.
- **Ignoring the context limit.** A basic cache (and position embedding) only covers `block_size`
  positions; past that you must slide the window or use longer positional schemes. Our demo stays
  within `block_size`.

---

## 🤔 Common questions

- **Does the KV-cache change the model's output?** No — it stores exactly the keys/values the naive
  path recomputes, so the generated tokens are identical. It's pure speed (§5).
- **Why cache keys and values but not queries?** Because a past token's query is never needed again —
  only the *current* token's query attends. But every past token's key and value *are* attended to by
  every future token, so those are what's worth keeping.
- **Where does the `O(T²) → O(T)` come from?** Naive redoes a full-sequence pass every step
  (`1+2+…+T ≈ T²`); with the cache each step processes one new token (`T` steps × constant work).
- **Why is the cache so big for real models?** It's `2 × layers × heads × head_dim` numbers *per
  token*, times the sequence length and batch. For a 7B model at long context that's tens of GB —
  more than the weights (§6).
- **Does the cache only work with greedy decoding?** No — it's independent of *how* you pick the next
  token (greedy, sampling, top-k, beam). The cache makes the *attention* cheap; the sampling happens
  afterward on the logits. (We use greedy in the demo only so "identical output" is a clean test.)
- **Why cache k/v and not the whole hidden state?** Because keys and values are *exactly* what future
  tokens read from a past token — nothing else about a past token is reused. Caching k/v is the
  minimal sufficient store.
- **Is this really what makes ChatGPT fast?** Yes — every production LLM server uses a KV-cache; the
  prefill/decode split, GQA, and PagedAttention are all built around it.

## ✅ Check your understanding

<details>
<summary>1. Why is naive autoregressive generation O(T²)?</summary>

Each step runs a full forward pass over the *entire* sequence so far (recomputing all past keys and
values) but keeps only the last position's prediction. Over `T` tokens that's `1 + 2 + … + T ≈ T²/2`
work — quadratic in the length.
</details>

<details>
<summary>2. What is the key property of causal attention that makes caching correct?</summary>

A token attends only to itself and earlier tokens, so a past token's key and value depend only on
tokens up to it — **not** on anything appended later. They never change, so they can be computed once
and cached.
</details>

<details>
<summary>3. In one decode step with a cache, what do you compute, and what do you reuse?</summary>

You compute the query, key, and value for **only the new token**; you **reuse** all past keys and
values from the cache. You append the new k/v to the cache, then the new query attends over the whole
(cached + new) set.
</details>

<details>
<summary>4. What are the prefill and decode phases?</summary>

**Prefill**: one forward pass over the whole prompt at once, filling the cache and producing the first
token. **Decode**: generate one token at a time, each step processing a single token and attending
over the cache.
</details>

<details>
<summary>5. Why can the KV-cache use more memory than the model's weights?</summary>

The cache is `2 × layers × heads × head_dim` numbers *per token*, growing with sequence length and
batch. At a long context (e.g. 32k) and/or large batch, that product exceeds the fixed size of the
weights — for Llama-2-7B, ~17 GB of cache vs ~14 GB of weights.
</details>

## 🎓 Key takeaways

- **Autoregressive generation** is a loop; the naive version re-runs the whole sequence each step —
  `O(T²)` wasted recomputation.
- Under **causal attention**, a past token's **keys and values never change**, so you compute them
  once and **cache** them.
- A cached decode step computes q/k/v for **only the new token**, appends its k/v to the cache, and
  attends over the whole cache — turning generation into `O(T)`, with **identical output**.
- Inference is **prefill** (whole prompt at once, fills the cache) then **decode** (one token at a
  time). The speedup grows with length.
- The cache's cost is **memory** — it grows with length × batch and can exceed the weights, which is
  why **GQA/MQA** and **PagedAttention** exist and why long context is expensive.

## 📖 New vocabulary

`inference` · `autoregressive generation` · `KV-cache` · `keys` / `values` · `causal mask` ·
`prefill` vs `decode` · `incremental decoding` · `pos_offset` · `O(T²)` vs `O(T)` ·
`latency` vs `throughput` · `batching` · `MQA` / `GQA` · `PagedAttention`.

## 🧪 Practice & build

1. **The notebook** ([Colab](https://colab.research.google.com/github/ImAlno/Understanding-LLMs/blob/main/chapters/12-inference-kv-cache/code/explore.ipynb))
   — see the waste, implement the cache, prove it's identical, compute its memory. Do this first.
2. **Exercises** — [`exercises/`](./exercises/): implement the cached decode step, prove cached == naive,
   measure how the speedup grows with length, and compute a KV-cache memory table. Tiered hints +
   solutions; a starter for the first.
3. **Mini-project** — [`project/`](./project/): **"KV-Cache Your GPT"** — add a KV-cache to a GPT's
   `generate`, confirm the output is unchanged, and benchmark the tokens/second win.

## 🔗 Go deeper (optional)

- 📄 [HuggingFace — *KV Caching Explained*](https://huggingface.co/blog/not-lain/kv-caching) — a clear,
  code-first walkthrough.
- 📄 [Pope et al. (2022), *Efficiently Scaling Transformer Inference*](https://arxiv.org/abs/2211.05102)
  — the trade-offs of serving big models (prefill/decode, batching, the cache).
- 📄 [Kwon et al. (2023), *PagedAttention / vLLM*](https://arxiv.org/abs/2309.06180) — paging the
  KV-cache like virtual memory.

---

| ⬅️ Previous | Up | Next ➡️ |
|:--|:-:|--:|
| [Chapter 11 — Datasets](../11-datasets/) | [Syllabus](../../README.md#-syllabus) | [Chapter 13 — Quantization](../13-inference-quantization/) |
