# 📚 Mini-Project — The Storyteller's Data Pipeline

You've built the pieces; now **assemble the whole fuel line** end to end: generate a story corpus,
pack it into `.bin` shards, sample training batches, and *prove they're ready to train* — the batches
decode back to real stories **and** feed cleanly through a fresh model (loss ≈ `ln(vocab)`, an
untrained baseline).

> 💻 **No GPU, no download.** Pure CPU + NumPy on a corpus you generate locally.

> **How this works:** [`starter/build_pipeline.py`](./starter/build_pipeline.py) does the
> generate → pack → sample → verify pipeline and has **one TODO** — implement the `get_batch`
> sampler. A reference is in [`solution/build_pipeline.py`](./solution/build_pipeline.py).

## 🎯 What it does

```bash
python starter/build_pipeline.py
```

Runs all five steps: **(1)** generate 2,000 synthetic stories, **(2)** tokenize and pack them to
`.bin` shards, **(3)** sample a batch with your `get_batch`, **(4)** decode it to confirm it's real
story text, and **(5)** feed it through a fresh embedding to confirm the loss is ~`ln(vocab)` — i.e.
the batch is a valid model input.

## 🛠️ Your TODO

Implement the sampler in `get_batch`:

```python
x = torch.stack([torch.from_numpy(data[i     : i + BLOCK    ].astype(np.int64)) for i in ix])
y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + BLOCK].astype(np.int64)) for i in ix])
```

Until you fill it, the script detects `x`/`y` are `None` and asks you to.

## ✅ Checking your work

- **It decodes to stories:** step 4 should print recognizable text like
  `'k. One day, Lily found a key. Lily was very happ'`.
- **It's model-ready:** step 5's loss should land near `ln(37) ≈ 3.61` (a fresh model is maximally
  unsure over the 37-char vocab — Chapter 7's loss-at-init, now on your own data). The script asserts
  this.
- Compare against [`solution/build_pipeline.py`](./solution/build_pipeline.py).

## 🚀 Stretch

- **Swap in your Chapter 6 BPE tokenizer.** Replace the char-level encoding in `prepare.py` with your
  `RegexTokenizer` — the `.bin`/memmap/`get_batch` machinery doesn't change, only the ids do.
- **Use the *real* TinyStories.** `pip install datasets`, then
  `load_dataset("roneneldan/TinyStories")`, and run it through the same `prepare` → `get_batch`
  pipeline. (Needs the internet; everything else here doesn't.)
- **Actually train.** Point your Chapter 5 GPT's training loop at `get_batch` and watch it learn to
  write stories — the payoff of the whole "Need for Speed" + "Datasets" arc.
- **Multiple shards.** Real pipelines split a huge corpus into many `.bin` shards; make `get_batch`
  pick a random shard, then a random window within it.

Next: [Chapter 12 — Inference I: the KV-cache](../../12-inference-kv-cache/), where we make the
trained model *generate* fast. 📚
