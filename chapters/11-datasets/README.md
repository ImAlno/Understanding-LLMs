# Chapter 11 — Datasets: Feeding the Storyteller

> A model is what it eats. You've spent ten chapters learning to *train* — but a Transformer is only
> as good as the text you pour through it, and pouring it efficiently is its own craft. This chapter
> builds the **data pipeline**: raw text → tokens → a compact binary file on disk → a fast stream of
> training batches. Along the way we meet **memory-mapping** (how you sample from a dataset far
> bigger than your RAM), PyTorch's **`Dataset`/`DataLoader`**, and **synthetic data** — including
> **TinyStories**, the "simple text so a *small* model can learn" idea that our Storyteller is built
> on. Everything runs on your laptop, on a synthetic story corpus we generate from scratch.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ImAlno/Understanding-LLMs/blob/main/chapters/11-datasets/code/explore.ipynb)

> 💻 **No GPU, no download needed.** The whole pipeline is CPU work, and we generate our own corpus
> locally (no network). At the end we show how to swap in the *real* TinyStories with one
> `datasets` call — optional, and the only part that needs the internet.

**You will be able to:**
- explain why **data quality and quantity** decide what a model can become;
- build the **tokenize-once-to-disk** pipeline: text → `uint16` token ids → `.bin` files;
- **memory-map** a token file and sample training batches from data bigger than RAM (the
  `get_batch` pattern), with `x`/`y` shifted by one for next-token prediction;
- **recognize** PyTorch's **`Dataset`/`DataLoader`** and say when to use it vs `get_batch`;
- keep a clean **train/val split**, and reason about **epochs vs steps**;
- explain **synthetic data** and why **TinyStories** lets a tiny model write coherent English.

**Prerequisites:** the training loop and the `x`/`y` next-token setup (Chapters 1–5), and
tokenization (Chapter 6 — here we use a simple char tokenizer to stay self-contained, but the
pipeline is identical with BPE). A little NumPy.

**Time:** ~2 hours. **Hardware:** any laptop CPU.

> ### 📓 Build it yourself in the notebook
> **[`code/explore.ipynb`](./code/explore.ipynb)**: generate synthetic stories, pack them into a
> `.bin`, **implement `get_batch`** by memory-mapping it, and decode a batch back to text. "✍️ Your
> turn", "▶️ Run this", "▶️ Check your work" cells. Watch for: ✏️ **In the notebook → Step N**.

---

## 0. Setup

Three small scripts form the pipeline (all self-contained, no downloads):

- [`code/synth_stories.py`](./code/synth_stories.py) — generate a simple story corpus from templates.
- [`code/prepare.py`](./code/prepare.py) — tokenize the corpus and pack it into `.bin` files.
- [`code/dataloader.py`](./code/dataloader.py) — memory-map the files and sample batches.

```bash
python chapters/11-datasets/code/prepare.py       # writes data/train.bin + val.bin
python chapters/11-datasets/code/dataloader.py     # samples a batch and decodes it
```

---

## A 2-minute primer: the model is what it eats

Every capability a language model has, it learned from its training text. Feed it clean, diverse,
well-written text and it writes well; feed it garbage and it writes garbage. Two levers matter:

- **Quantity.** Bigger models need *more* data to fill their capacity — the scaling laws of
  Chapter 10 grow parameters *and* tokens together. Modern pretraining runs on **trillions** of
  tokens.
- **Quality.** Duplicate, spammy, or broken text actively hurts. A huge fraction of the work behind
  a good model is *cleaning and curating* the data — deduplication, filtering, balancing sources.

(How much *is* enough? A rule of thumb from the **Chinchilla** result is roughly **20 tokens per
parameter** for compute-optimal training — so a 1B model wants ~20B tokens, a 70B model ~1.4T. In
practice people train well *past* that ratio, because a smaller model fed more data is cheaper to
*run* forever after.)

So "get a dataset" is really three jobs: **acquire** the text, **process** it into a form training
can consume fast, and **curate** it so it's worth learning from. This chapter is mostly the middle
job — the *pipeline* — because that's the part you build and run yourself. But keep the other two in
mind: they're where most of the real-world effort goes.

---

## 1. The pipeline, end to end

> ✏️ **In the notebook → Step 1.**

Here's the whole journey from raw text to a training batch:

```
   raw text          "Once upon a time, there was a little cat named Lily. ..."
      │  tokenize (Chapter 6) — do this ONCE
      ▼
   token ids         [45, 67, 12, 8, 45, ...]   (uint16, 2 bytes each)
      │  split + write to disk
      ▼
   train.bin / val.bin        a flat binary file of token ids
      │  memory-map + sample random windows, every step
      ▼
   batch  (x, y)     x = a window of tokens,  y = the same window shifted one right
      │  feed the model (Chapters 5–10)
      ▼
   loss.backward()
```

The key design choice is that **tokenizing happens once, offline**, and training just reads the
prepared token ids. Re-tokenizing text on every training step would waste enormous time; instead we
pay that cost a single time and store the result in the most training-friendly form there is — a
flat array of integers.

---

## 2. Tokenize once, pack to disk

> ✏️ **In the notebook → Step 2.**

To make our own corpus, [`synth_stories.py`](./code/synth_stories.py) generates simple stories from
templates (more on *why* in §6). A corpus of 2,000 of them is about **344,000 characters** of text
like:

```
Once upon a time, there was a little fox named Sam. Sam loved to sing in the garden.
One day, Sam found a leaf. Sam was very glad. Sam and the leaf played all day. The end.
```

[`prepare.py`](./code/prepare.py) then does the packing: build a vocabulary, turn every character
into an integer id, split off a validation slice, and write the ids to disk as raw `uint16`:

```python
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
ids = np.array([stoi[c] for c in text], dtype=np.uint16)   # the whole corpus as token ids
n_train = int(len(ids) * 0.9)
ids[:n_train].tofile("train.bin")                          # raw bytes, no wrapper
ids[n_train:].tofile("val.bin")
```

Real output:

```
corpus: 344,498 chars -> 344,498 tokens, vocab 37
wrote train.bin (310,048) + val.bin (34,450) = 0.69 MB of uint16
```

Why `uint16`? A token id is a small integer (0 … vocab-1). `uint16` holds `0 … 65,535` — enough for
any BPE vocab (GPT-2's is 50,257) — at just **2 bytes per token**. So a billion-token corpus is a
2 GB file: dense, contiguous, and trivial to read. (`.tofile` writes the raw bytes with no header —
the file *is* the array.) That's the "pack" step, and it's exactly what nanoGPT and real pretraining
pipelines do, just with BPE and many more tokens.

Concretely, tokenizing a big corpus can take minutes to hours (BPE has to scan every character and
apply its merges); doing it once and reading the `.bin` afterward is *instant*. The rule generalizes:
any expensive, one-time transform of your data belongs **offline** in `prepare.py`, never in the
training loop — the same "keep slow work out of the hot loop" instinct as Chapter 8.

---

## 3. Memory-mapping: sampling from data bigger than RAM

> ✏️ **In the notebook → Steps 3–4.** Implement `get_batch`, then decode a batch back to text.

Here's the problem at scale: FineWeb (a modern web corpus) is ~15 *trillion* tokens — about **30 TB**
as `uint16`. You cannot load that into RAM. The solution is **memory-mapping**: `np.memmap` treats
the file *as if* it were an array, but the operating system pages in only the bytes you actually
touch. You can index a 30 TB file from a laptop and only the windows you read ever enter memory.

(Why not just `open()` the file and `read()` the bytes you need? You could — but `memmap` lets you
*index it like a NumPy array* (`data[i : i+block]`) while the OS transparently handles caching,
read-ahead, and eviction. It turns "manage file offsets by hand" into "treat 30 TB on disk as if it
were `data[...]` in memory.")

A **page** is a fixed-size chunk (a few KB) the OS shuttles between disk and RAM; memory-mapping
works page by page:

```
   train.bin on disk (30 TB)              RAM (a few GB)
   ┌─────────────────────────────┐        ┌────────────────┐
   │ ....##....###.........##.... │  ───►  │  ##  ###   ##   │   only the pages holding the
   └─────────────────────────────┘        └────────────────┘   windows you sampled are loaded;
        ^ your random windows                                   the OS evicts old ones when it
                                                                needs room — so RAM never fills up
```

So even though you sample windows from all over a 30 TB file, only the handful of pages you're
*currently* touching are ever resident — the OS pages them in on demand and evicts the rest. That's
what lets a laptop stream from a dataset thousands of times larger than its memory.

Given the memmap, one training batch is a handful of **random windows**:

```python
data = np.memmap("train.bin", dtype=np.uint16, mode="r")     # the file, as an array
ix = torch.randint(len(data) - block_size, (batch_size,))    # random start offsets
x = torch.stack([torch.from_numpy(data[i     : i + block_size    ].astype(np.int64)) for i in ix])
y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix])
```

`x` is a window of `block_size` tokens; `y` is *the same window shifted one token right* — every
position's target is the next token, exactly the setup since Chapter 1. Real output from
[`dataloader.py`](./code/dataloader.py):

```
one batch: x (16, 64), y (16, 64), dtype torch.int64
x[0] decodes to: 'k. One day, Lily found a key. Lily was very happy.'
y[0] decodes to: '. One day, Lily found a key. Lily was very happy. '
```

(These numbers use `block_size=64, batch_size=16`; the notebook uses a smaller `32/8` for speed —
same idea, smaller windows.) Look at the two decoded strings: `y` is `x` slid left by one character — position 0 of `x` (`k`)
predicts position 0 of `y` (`.`), and so on. (We `.astype(np.int64)` because PyTorch wants `int64`
indices for the embedding lookup, and re-open the memmap each call — cheap, and the right thing to do
when many worker processes share the file.) This random-window sampler *is* how GPT-style models are
trained: no epochs, no fixed order — just endless random windows from an ocean of tokens.

Why *random* offsets instead of reading front to back? Randomness decorrelates a batch's examples
(consecutive windows overlap heavily and would give a biased, low-variance gradient), and it makes
every step an unbiased sample of the whole corpus — the "stochastic" in stochastic gradient descent
(Chapter 7), fed straight from disk.

---

## 4. `Dataset` and `DataLoader`: PyTorch's way

> ✏️ **Not a notebook step** — the `DataLoader` here is context for the *finite-dataset* work in
> Chapters 14–15; the notebook stays on the pretraining `get_batch` path.

The `get_batch` above is bespoke. PyTorch also offers a standard abstraction worth knowing, because
you'll use it constantly for *finite* datasets (fine-tuning, classification): a **`Dataset`** says
"how many examples are there, and how do I fetch item `i`?", and a **`DataLoader`** wraps it to
handle **batching**, **shuffling**, and **parallel loading**:

```python
from torch.utils.data import Dataset, DataLoader

class TokenWindows(Dataset):
    def __init__(self, data, block_size):
        self.data, self.block = data, block_size
    def __len__(self):
        return len(self.data) - self.block
    def __getitem__(self, i):
        chunk = torch.from_numpy(self.data[i : i + self.block + 1].astype(np.int64))
        return chunk[:-1], chunk[1:]          # (x, y): the window and itself shifted by one

loader = DataLoader(TokenWindows(data, 64), batch_size=16, shuffle=True, num_workers=2)
for x, y in loader:                            # yields shuffled (x, y) batches
    ...
```

(This sketch shares one `data` handle for brevity; a production memmap `Dataset` opens the file
*inside* `__getitem__` — or via a `worker_init_fn` — so each worker gets its own, for the same reason
`get_batch` in §3 re-opens it.)

- **`shuffle=True`** hands out examples in random order each epoch (so the model doesn't learn the
  order of the data).
- **`num_workers=N`** loads batches in `N` background processes, so data prep overlaps with the
  GPU's compute — the same "hide the wait behind useful work" idea as Chapters 8 and 10. (A subtlety
  with `memmap`: each worker should open its *own* handle, not share one across the forked processes —
  which is exactly why `get_batch` in §3 re-opens the file every call.)

**When to use which?** For *pretraining* on a giant corpus, the memmap `get_batch` (§3) is simpler
and standard — you don't need epochs over trillions of tokens, just infinite random windows. For
*finite* datasets you'll iterate many times (fine-tuning in Chapters 14–15), the `DataLoader` — with
its shuffling and multiprocessing — is the right tool. Both produce the same thing: `(x, y)` batches.

---

## 5. Train/val split, epochs vs steps

Two disciplines the pipeline has to get right:

- **The train/val split must not leak.** We split the token stream into `train.bin` and `val.bin`
  *before* sampling any windows, so no validation token is ever trained on. (We split by *position*
  here; real pipelines split by *document* so a story can't straddle the boundary.) Validation loss
  on held-out text is your only honest signal of whether the model is *learning* rather than
  *memorizing* — the overfitting story from Chapter 3, now at the data level.
- **Epochs vs steps.** One **epoch** is one full pass over the dataset; one **step** is one batch /
  one optimizer update. On small data you train for many epochs (seeing each example repeatedly). On
  giant pretraining corpora you often do **less than one epoch** — there's so much text that the
  model never sees the same token twice, so people count **steps** (or tokens), not epochs. Our
  random-window sampler doesn't even track epochs; it just draws steps forever.

A concrete feel: our synthetic corpus is ~310K training tokens. With `block_size = 64` and
`batch_size = 16`, one step consumes `64 × 16 = 1,024` tokens, so a single epoch is ~300 steps. Train
for 5,000 steps and you've done ~16 passes over this tiny corpus — fine for toy data. Now scale up:
FineWeb's 15T tokens at the same 1,024 tokens/step is ~15 *billion* steps for **one** epoch — which
is exactly why nobody counts epochs there.

---

## 6. Synthetic data: making the text you need

> ✏️ **In the notebook → Step 1.**

Sometimes you can't just *download* the right data — it doesn't exist, or it's not clean, simple, or
targeted enough. So you **generate** it. That's **synthetic data**, and it's increasingly central to
how models are built (for pretraining, for fine-tuning, for evaluation).

Our [`synth_stories.py`](./code/synth_stories.py) is a toy version: fill sentence templates from
small word lists to produce endless simple stories.

```python
f"Once upon a time, there was a {adj} {animal} named {name}. "
f"{name} loved to {verb} in the {place}. "
f"One day, {name} found a {obj}. ..."
```

It's deliberately simple and repetitive — which, it turns out, is exactly the point.

Synthetic data has a clear appeal and a clear risk. The **appeal**: you get *exactly* the
distribution you want — simple enough for a small model, or targeted at a skill you're teaching — in
unlimited quantity, cleanly formatted. The **risk**: the data can only be as good as its generator,
so a model trained on it inherits the generator's quirks and blind spots (and, if you're careless,
its *mistakes* dressed up as facts). Used well — as TinyStories does — it's one of the most powerful
tools in the modern toolkit; used lazily, it quietly caps how good your model can get.

---

## 7. TinyStories, and real datasets

**TinyStories** (Eldan & Li, 2023) is the idea our whole course rests on. The authors used GPT-3.5/4
to generate millions of very short stories using only words a **3–4-year-old** would know. The
finding was striking: a model with just a *few million* parameters — hundreds of times smaller than
GPT-2 — trained on TinyStories learns to produce **fluent, coherent, grammatical** short stories.
Normal web text is too hard for a tiny model to ever sound coherent; *simple* text lets a tiny model
succeed. That's why our from-scratch Storyteller can actually work on a laptop: we match the data's
difficulty to the model's size.

Our template stories are a stand-in you can generate offline; the real thing is one call away (this
is the only networked, optional part of the chapter):

```python
# optional — needs `pip install datasets` and the internet
from datasets import load_dataset
ds = load_dataset("roneneldan/TinyStories", split="train")
print(ds[0]["text"])        # a real GPT-generated tiny story
```

For *bigger* models you reach for bigger, messier corpora: **FineWeb** (15T filtered web tokens),
**The Pile** (an 825 GB mix of books, code, papers, web). Those need the full curation pipeline —
deduplication, quality filtering, source balancing — but they feed into the *exact same* tokenize →
`.bin` → memmap → batch machinery you just built.

---

## 8. A word on quality and curation

The unglamorous truth of modern pretraining: **most of the effort is in the data, not the model.**
The model architecture (Chapters 4–5) barely changes run to run; what separates a good model from a
mediocre one is usually the data pipeline's *curation*:

- **Deduplication** — the web is full of copies; training on the same text many times wastes compute
  and encourages memorization.
- **Quality filtering** — drop boilerplate, spam, broken encodings, and machine-generated junk
  (often with a small classifier trained to recognize "good" text).
- **Source mixing** — deliberately balance web, books, code, math, etc., because the mix shapes what
  the model is good at.

To put a number on it: FineWeb's creators started from ~**100 trillion** tokens of raw Common Crawl
and filtered and deduplicated down to ~**15 trillion** — discarding *85%* of it. The kept 15% trains
a *better* model than the full 100% would. Curation isn't tidying up around the edges; it's most of
the quality.

You won't build a trillion-token curation pipeline here, but knowing that this is *where the work is*
is half of understanding how real models are made.

---

## 🔌 How this plugs into the Storyteller

This chapter is the fuel line. When you train the Storyteller for real, you'll run `prepare.py` on
TinyStories (or our synthetic corpus) to make `.bin` files once, then point the Chapter 5 GPT's
training loop at `get_batch` — every step draws fresh random windows straight from disk, tokenized
and ready. Combined with Chapters 8–10 (device, precision, distribution), you now have the complete
picture: a model, made fast, fed well. The mini-project builds exactly this pipeline end to end.

---

## 🐛 Building it yourself: what trips people up

- **Forgetting `x` and `y` overlap.** `y` is `x` shifted by **one** token, not a separate chunk —
  they share `block_size - 1` tokens. Sample a window of `block_size + 1` and slice `[:-1]` / `[1:]`.
- **`np.memmap` dtype mismatch.** If you wrote `uint16` but read `uint8` (or `int16`), every value is
  garbage. The dtype on read must match the dtype you wrote. (There's no header to catch you.)
- **Not casting to `int64`.** Token ids feed an `nn.Embedding`, which wants `int64` (`long`) indices;
  passing `uint16`/`int32` errors. Cast when you build the batch.
- **Leaking val into train.** Split the token stream *before* you sample windows. Sampling first and
  splitting batches later lets a validation token sneak into a training window.
- **Re-tokenizing every run.** Tokenizing is slow; do it once in `prepare.py` and memmap the result.
  If your training loop calls the tokenizer, you've put the slow step in the hot loop.

---

## 🤔 Common questions

- **Why store tokens as a binary file instead of just keeping the text?** Speed and size. Training
  reads integers, not characters — pre-tokenizing to a flat `uint16` array means every step is a
  fast memory read, no parsing, and the file is compact (2 bytes/token).
- **How does memory-mapping let me use a file bigger than RAM?** The OS loads only the *pages* you
  actually read into memory, on demand, and evicts them when it needs room. You index the file like
  an array; only your sampled windows are ever resident.
- **Do LLMs really train for less than one epoch?** On the largest corpora, yes — there's so much
  text that a single pass already exhausts the compute budget, so the model may see most tokens only
  once. People track **tokens** or **steps**, not epochs.
- **Why is TinyStories the key to a laptop-sized model?** It matches the data's difficulty to the
  model's size: simple vocabulary and short, formulaic stories are learnable by a few-million-param
  model, where full web text never would be.
- **Is synthetic data "cheating"?** No — it's a mainstream technique. TinyStories, instruction-tuning
  sets, and many eval suites are synthetic. The catch is that a model can only be as good as the
  generator that made its data (and can inherit its blind spots).
- **Why split by document, not just by position?** If you cut the token stream at an arbitrary
  offset, one story can land half in train and half in val — a subtle leak, since the model could
  "predict" the val half from the train half it memorized. Splitting on document boundaries keeps
  each story wholly on one side. (We split by position here only because our toy corpus is uniform.)

## ✅ Check your understanding

<details>
<summary>1. Why do we tokenize the whole corpus once and store <code>uint16</code> ids on disk?</summary>

Because re-tokenizing on every training step is slow; doing it once and storing a flat `uint16`
array means each step is a fast memory read of integers (no parsing). `uint16` (2 bytes) holds any
token id up to 65,535 — enough for a BPE vocab — so the file is compact and contiguous.
</details>

<details>
<summary>2. What does <code>np.memmap</code> buy you, and why does it matter for real corpora?</summary>

It lets you index a file as if it were an array while the OS pages in only the bytes you actually
read. So you can sample random windows from a dataset far larger than RAM (FineWeb ~30 TB) from a
laptop — only the sampled windows enter memory.
</details>

<details>
<summary>3. In a batch, how are <code>x</code> and <code>y</code> related?</summary>

`y` is `x` shifted one token to the right: each position of `x` predicts the *next* token, which sits
at the same position in `y`. You get both from one window of `block_size + 1` tokens (`[:-1]` and
`[1:]`).
</details>

<details>
<summary>4. When would you use a <code>DataLoader</code> instead of the memmap <code>get_batch</code>?</summary>

For *finite* datasets you pass over many times (fine-tuning, classification), where you want epochs,
shuffling, and parallel loading. For *pretraining* on a giant corpus, the memmap `get_batch` — infinite
random windows, no epochs — is simpler and standard.
</details>

<details>
<summary>5. Why does TinyStories let a tiny model write coherent English when web text doesn't?</summary>

It matches data difficulty to model size: TinyStories uses only simple words and short, formulaic
stories, which a few-million-parameter model *can* learn to produce fluently. Full web text is too
complex for so small a model to ever sound coherent.
</details>

## 🎓 Key takeaways

- A model is **what it eats**: quantity fills capacity, quality decides what it's worth — and most
  real-world effort is **data curation**, not architecture.
- The pipeline is **tokenize once → pack to a `uint16` `.bin` → memory-map → sample batches**. Doing
  the slow tokenize step offline is the whole point.
- **Memory-mapping** lets you stream random windows from a corpus far bigger than RAM; `x`/`y` are
  one window and itself shifted by one (next-token targets).
- Use the memmap **`get_batch`** for giant-corpus pretraining, a **`DataLoader`** for finite datasets
  you epoch over. Keep a leak-free **train/val split**.
- **Synthetic data** (like **TinyStories**) is how you get simple, targeted text — and matching data
  difficulty to model size is why a laptop-sized Storyteller can work at all.

## 📖 New vocabulary

`data pipeline` · `tokenize-and-pack` · `.bin shard` · `uint16` · `memory-mapping` / `np.memmap` ·
`get_batch` · `Dataset` / `DataLoader` · `shuffle` · `num_workers` · `train/val split` · `epoch` vs
`step` · `synthetic data` · `TinyStories` · `FineWeb` / `The Pile` · `deduplication` ·
`quality filtering` · `data curation`.

## 🧪 Practice & build

1. **The notebook** ([Colab](https://colab.research.google.com/github/ImAlno/Understanding-LLMs/blob/main/chapters/11-datasets/code/explore.ipynb))
   — generate stories, pack a `.bin`, implement `get_batch`, decode a batch. Do this first.
2. **Exercises** — [`exercises/`](./exercises/): implement `get_batch`, verify `x`/`y` are shifted by
   one, prove the train/val split doesn't leak, and check batch shapes/dtypes. Tiered hints +
   solutions; a starter for the first.
3. **Mini-project** — [`project/`](./project/): **"The Storyteller's Data Pipeline"** — generate a
   corpus, pack it into `.bin` files, build a batch sampler, and confirm the batches decode back to real
   story text, ready to train on.

## 🔗 Go deeper (optional)

- 📄 [Eldan & Li (2023), *TinyStories*](https://arxiv.org/abs/2305.07759) — how simple synthetic
  stories let tiny models learn coherent English.
- 📄 [HuggingFace — *FineWeb*](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1) — a
  deep look at building and cleaning a 15-trillion-token web corpus.
- 📄 [Gao et al. (2020), *The Pile*](https://arxiv.org/abs/2101.00027) — an 825 GB curated mix, and
  the reasoning behind mixing sources.

---

| ⬅️ Previous | Up | Next ➡️ |
|:--|:-:|--:|
| [Chapter 10 — Distributed](../10-speed-distributed/) | [Syllabus](../../README.md#-syllabus) | [Chapter 12 — KV-cache](../12-inference-kv-cache/) |
