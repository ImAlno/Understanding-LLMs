# Chapter 6 — Tokenization: Byte-Pair Encoding

> Every chapter so far fed the model **characters**. Real LLMs feed it **tokens** — subword
> chunks like `" the"`, `"ing"`, or `" Citizen"` — produced by **byte-pair encoding (BPE)**.
> This chapter builds that tokenizer from scratch. No neural nets, no training loop — just
> bytes, dictionaries, and one beautifully simple idea: *repeatedly merge the most common
> pair*.

**You will be able to:**
- explain why **subword** tokens beat both characters and whole words;
- turn text into **UTF-8 bytes** and back;
- implement the **BPE merge algorithm** (`get_stats`, `merge`, `train`);
- **encode** new text into tokens and **decode** tokens back to text;
- reason about **vocabulary size vs. compression**;
- understand the **regex splitting** GPT-2/GPT-4 use, and real tokenizers (`tiktoken`).

**Prerequisites:** comfort with Python lists, dictionaries, and loops. That's it — this is the
one chapter with **no PyTorch and no math**. A clean break before the heavy chapters ahead.

**Time:** ~2–3 hours. **Hardware:** anything — it's pure Python.

> ### 📓 Build it yourself in the notebook
> **[`code/explore.ipynb`](./code/explore.ipynb)** builds the whole tokenizer piece by piece —
> `get_stats`, `merge`, `train`, `encode`, `decode` — then measures the compression. "✍️ Your
> turn", "📖 Study & run", "▶️ Check your work" cells. Watch for: ✏️ **In the notebook → Step N**.

---

## 0. Setup

The reference implementation is [`code/bpe.py`](./code/bpe.py); we train on the same Shakespeare
([`data/input.txt`](./data/input.txt)).

```bash
uv venv --python 3.13 .venv && uv pip install -r requirements.txt   # adds `regex`
.venv/bin/jupyter lab        # open chapters/06-tokenization/code/explore.ipynb
```

---

## A 2-minute primer: characters, words, or something in between?

We need to turn text into a sequence of integers (the model can only do math on numbers). Three
choices:

- **Characters** (what we've used): a tiny vocabulary (~65 for Shakespeare, ~150k for all of
  Unicode), but sequences are **long** — every letter is a step — so the model wastes its
  limited context on spelling, and burns compute per character.
- **Whole words**: short sequences, but a **gigantic** vocabulary (every word *and* form:
  "run", "runs", "running", "ran"…), and any word it never saw is a dead end (out-of-vocabulary).
- **Subwords** (BPE): the sweet spot. Common words become **single tokens** (`" the"`,
  `" and"`); rare words split into a few pieces (`" Storyteller"` → `" Story" + "teller"`); and
  because we start from raw bytes, **nothing is ever out-of-vocabulary** — the worst case is
  falling back to single bytes. Fixed vocab, short-ish sequences, no dead ends.

BPE is how we *build* a subword vocabulary, automatically, from data. Here's the whole idea in
one sentence: **start with bytes, then repeatedly merge the most frequent adjacent pair into a
new token, until the vocabulary is as big as you want.**

---

## 1. Text → bytes

> ✏️ **In the notebook → Step 1.** See text become bytes.

Computers store text as **Unicode code points** (every character — letters, emoji, anything —
has a number). To get a compact, universal starting alphabet, we encode those code points as
**UTF-8 bytes**: a sequence of integers, each **0–255**.

```python
list("hi 🧙".encode("utf-8"))
# [104, 105, 32, 240, 159, 167, 153]
#   h    i   ' '  └──── the wizard emoji, 4 bytes ────┘
```

Notice `h` became **1** byte (`104`) but the emoji became **4** (`240, 159, 167, 153`): UTF-8 is
*variable-length* — plain ASCII characters are 1 byte each, while rarer ones take 2–4 (that's the
"why 4 bytes?"). So **every possible text** becomes a list of integers in `0..255`. Those 256
byte-values are our **starting vocabulary** (tokens `0`–`255`), and BPE grows it from there.

> 🧰 **New here: the `bytes` type.** Python writes a bytes value with a `b` prefix: `b"hi"` is
> *two bytes* `[104, 105]`, not the text `"hi"`. You only need a few operations on it this chapter:
> `list(b"hi")` → `[104, 105]` (bytes to ints); `bytes([104, 105])` → `b"hi"` (ints to bytes);
> `b"hi" + b"!"` → `b"hi!"` (glue two with `+`); `b"".join([b"hi", b"!"])` → `b"hi!"` (glue a
> list); and `b"hi".decode("utf-8")` → `"hi"` (bytes back to text). That's the whole toolkit.

---

## 2. Counting pairs

> ✏️ **In the notebook → Step 2.** Write `get_stats`.

To find what to merge, count how often each **adjacent pair** of tokens appears:

```python
def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):       # zip(ids, ids[1:]) walks every neighbor pair
        counts[pair] = counts.get(pair, 0) + 1
    return counts

get_stats([1, 2, 1, 2, 3])               # {(1,2): 2, (2,1): 1, (2,3): 1}
```

`zip(ids, ids[1:])` is the idiom for "every neighboring pair": it pairs each element with the
one after it. `counts.get(pair, 0) + 1` is the standard "increment a dict counter, defaulting to
0." The most common pair is the one we'll merge next.

---

## 3. Merging a pair

> ✏️ **In the notebook → Step 3.** Write `merge`.

"Merging" a pair means: pick a brand-new token id, and replace every occurrence of that pair
with it.

```python
def merge(ids, pair, idx):
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            newids.append(idx)           # found the pair → emit the new token, skip both
            i += 2
        else:
            newids.append(ids[i])        # otherwise copy this token through
            i += 1
    return newids

merge([1, 2, 1, 2, 3], (1, 2), 99)       # [99, 99, 3]
```

We walk the list with an index `i`. When the next two tokens are the pair, we append the new id
and jump **two** forward; otherwise we copy one token and step **one** forward. (The
`i < len(ids) - 1` guard stops us reading past the end when checking `ids[i+1]`.)

---

## 4. Training: merge, and repeat

> ✏️ **In the notebook → Step 4.** Write the training loop.

Training is just: **find the most common pair, merge it into a new token, repeat** until we've
added enough tokens to reach `vocab_size`. We remember every merge (so we can re-apply it later)
and build a `vocab` mapping each token id back to the actual bytes it stands for.

```python
def train(text, vocab_size):
    num_merges = vocab_size - 256                    # we start with 256 byte-tokens
    ids = list(text.encode("utf-8"))
    merges = {}                                      # (int, int) -> new token id
    vocab = {idx: bytes([idx]) for idx in range(256)}
    for i in range(num_merges):
        stats = get_stats(ids)
        pair = max(stats, key=stats.get)             # the most frequent pair
        idx = 256 + i                                # its new token id
        ids = merge(ids, pair, idx)                  # apply it to the whole sequence
        merges[pair] = idx
        vocab[idx] = vocab[pair[0]] + vocab[pair[1]] # the new token = the two pieces' bytes
    return merges, vocab
```

`max(stats, key=stats.get)` returns the **key** (a pair) with the largest **value** (its count).
The `key=` argument is the trick: you hand `max` a *function*, and it ranks the items by what
that function returns. Here that function is `stats.get` — the same `.get` from §2, but written
*without* parentheses, because we're handing `max` the method to call on each pair, not calling it
ourselves. Each new token's bytes are just its two parts glued together (with `+`), which is why
decoding will work. Trained on Shakespeare, the first merges
it discovers are exactly the common chunks you'd guess:

```
b'e' + b' '  ->  b'e '        b't' + b'h'  ->  b'th'
b't' + b' '  ->  b't '        b'o' + b'u'  ->  b'ou'
```

---

## 5. Encoding new text

> ✏️ **In the notebook → Step 5.** Write `encode`.

To tokenize *new* text, we re-apply the learned merges — and crucially, **in the order we
learned them** (a later merge like `" the"` depends on an earlier one like `"th"` already
existing).

```python
def encode(text, merges):
    ids = list(text.encode("utf-8"))
    while len(ids) >= 2:
        stats = get_stats(ids)
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break                                    # no merge-able pair left → done
        ids = merge(ids, pair, merges[pair])
    return ids
```

The clever line is `min(stats, key=lambda p: merges.get(p, float("inf")))`. The `key=` works just
like §4's, but the ranking function here is a **`lambda`** — a one-line throwaway function.
`lambda p: merges.get(p, float("inf"))` reads "given a pair `p`, return its merge id," and `min`
calls it on each present pair for us. We want, among the pairs currently present, the one we
learned **earliest** — i.e. the smallest merge id.
`merges.get(p, float("inf"))` looks up each present pair's merge id, returning **infinity** for
any pair that was never a merge. Taking the `min` therefore picks the earliest-learned merge,
and pushes all non-merges to the back. When even the `min` isn't in `merges`, nothing is
merge-able and we stop. (We re-`get_stats` each loop only to know *which pairs are present*; the
counts don't matter here — the merge order does.)

---

## 6. Decoding

> ✏️ **In the notebook → Step 6.** Write `decode`, then check the round-trip.

Decoding is easy because `vocab` already maps each token id to its bytes: look each one up, glue
the bytes together, and decode the UTF-8.

```python
def decode(ids, vocab):
    text_bytes = b"".join(vocab[idx] for idx in ids)
    return text_bytes.decode("utf-8", errors="replace")
```

The golden test of any tokenizer is the **round-trip**: `decode(encode(text))` must return the
original text, exactly — for *any* string, including emoji and accents (that's the payoff of
working in bytes). (`errors="replace"` guards the rare case of decoding an arbitrary token
sequence that isn't valid UTF-8; for real round-trips it never triggers.)

---

## 7. Vocabulary size vs. compression

More merges = bigger vocabulary = each token covers more text = **fewer tokens per sentence**.
That ratio (bytes ÷ tokens) is the **compression**, and on our Shakespeare it climbs with vocab
size — with diminishing returns:

| vocab size | compression (bytes → tokens) |
|-----------:|:-----------------------------|
|   256 (raw bytes) | 1.00× |
|   512 | 1.89× |
|  1024 | 2.35× |
|  2048 | 2.72× |

(Exercise E01 reproduces this curve; exact numbers shift a little with the training slice.)
This is the core trade-off. **Bigger vocab** → shorter sequences (cheaper, longer effective
context) but a fatter embedding table and rarer tokens that are harder to learn. Real models
pick a sweet spot: GPT-2 uses **50,257** tokens (~4× compression on English), GPT-4 ~**100k**.

---

## 8. One more trick: split before you merge (the GPT-2 way)

Our basic tokenizer happily merges across spaces and punctuation — its very first merge on
Shakespeare is `'e' + ' '` → `'e '`. That glues word-endings to the next space, which is a bit
silly: `"dog"` and `"dog."` and `"dog!"` would each get different tokens.

GPT-2's fix: **split the text into word-ish chunks first** (with a regex), then run BPE *within*
each chunk, never across the boundaries. The pattern keeps a **leading** space with its word, so
you get tokens like `" the"`, `" Citizen"` — which is why, in real tokenizers, a word at the
start of a line tokenizes differently than the same word mid-sentence.

```python
import regex as re
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
re.findall(re.compile(GPT2_SPLIT_PATTERN), "First Citizen: we're poor.")
# ['First', ' Citizen', ':', ' we', "'re", ' poor', '.']
```

Don't be intimidated by that pattern — it's just a list of alternatives separated by `|`, each
matching one kind of chunk:

- `'(?:[sdmt]|ll|ve|re)` — contractions (`'s`, `'t`, `'ll`, `'ve`, `'re`);
- `" ?\p{L}+"` — an optional leading space, then a run of **letters** (`\p{L}`) → `" Citizen"`;
- `" ?\p{N}+"` — an optional space, then a run of **digits** (`\p{N}`);
- `" ?[^\s\p{L}\p{N}]+"` — an optional space, then **punctuation** (anything that isn't space,
  letter, or digit);
- `\s+(?!\S)|\s+` — leftover runs of whitespace.

The leading `" ?"` is what keeps a space *with* the word that follows it (so you get `" the"`,
not `"the"`). You don't have to be able to *write* regex like this — just recognize that it chops
text into words, numbers, punctuation, and spaces. The reference code's `RegexTokenizer` does
exactly this; you'll wire it up in the exercises.

---

## 9. Real tokenizers, and their famous quirks

Production models don't ship a from-scratch loop — they use fast libraries like OpenAI's
[`tiktoken`](https://github.com/openai/tiktoken) with pre-trained vocabularies. But it's the
**same algorithm you just built**, scaled up. Knowing how it works explains a lot of LLM
weirdness:

- **Numbers tokenize oddly** ("127" might be one token, "128" two) — which is why models flub
  arithmetic.
- **Trailing spaces matter** — `"hello"` and `"hello "` are different tokens, so a stray space
  in a prompt can change the output.
- **Non-English text costs more tokens** (its byte sequences weren't common in training), so the
  same sentence is "more expensive" in some languages.
- **Glitch tokens** like the infamous `SolidGoldMagikarp` — rare strings that got a token but
  almost no training — make models behave bizarrely.

Every one of these is a tokenization artifact, not a reasoning failure.

---

## 🔌 How this plugs into the Storyteller

Swap the Chapter 5 GPT's character vocabulary for a BPE one and two things happen: each training
sequence holds **~2–4× more text** (so the same `block_size` sees more context), and the model
predicts **word-like units** instead of letters — both make the generated text sharper. The
plumbing (an `nn.Embedding` of size `vocab_size`) is unchanged; only `vocab_size`, `encode`, and
`decode` differ. That swap is the mini-project's stretch goal.

---

## 🤔 Common questions

- **Why bytes and not characters?** Bytes give a fixed 256-symbol alphabet that covers *every*
  possible text (emoji, accents, any language) with no out-of-vocabulary holes. Characters would
  need a ~150k Unicode alphabet and still miss things.
- **Why merge the *most frequent* pair?** Greedy compression: merging the commonest pair removes
  the most tokens per step. It's not provably optimal, but it's simple and works extremely well.
- **Why re-apply merges in learned order when encoding?** Later tokens are built *out of* earlier
  ones (`" the"` needs `"th"` to exist first), so you must replay merges oldest-first.
- **Is this really what GPT-4 uses?** Yes — byte-level BPE with regex pre-splitting, exactly the
  `RegexTokenizer` here, just trained to ~100k tokens on far more text.
- **Does a bigger vocab always win?** No — it shortens sequences but inflates the embedding table
  and creates rarer tokens that are harder to learn. It's a trade-off (§7).

## ✅ Check your understanding

<details>
<summary>1. Why do we start from bytes (0–255) instead of characters?</summary>

256 byte-values form a fixed, universal alphabet that can represent *any* text — every language,
emoji, symbol — with **no out-of-vocabulary** possibility. BPE then grows the vocabulary from
that floor, and rare inputs gracefully fall back to single bytes.
</details>

<details>
<summary>2. In one sentence, what does BPE training do?</summary>

It repeatedly finds the **most frequent adjacent pair** of tokens and **merges** it into a new
token, recording each merge, until the vocabulary reaches the target size.
</details>

<details>
<summary>3. Why must <code>encode</code> apply merges in the order they were learned?</summary>

Because later tokens are composed of earlier ones (e.g. `" the"` is built from `"th"` + `"e"`).
Applying a late merge before its prerequisites would give wrong (or impossible) tokenizations,
so we always merge the earliest-learned available pair first.
</details>

<details>
<summary>4. What's the trade-off in choosing a vocabulary size?</summary>

Bigger vocab → more compression → shorter sequences (cheaper compute, longer effective context),
**but** a larger embedding table and rarer tokens that are harder for the model to learn well. You
pick a sweet spot (GPT-2: ~50k).
</details>

## 🎓 Key takeaways

- Models read **tokens**, not characters; **subword** tokens balance vocab size against sequence
  length and never go out-of-vocabulary.
- **BPE** = start from UTF-8 **bytes**, then repeatedly **merge the most frequent adjacent pair**
  into a new token.
- **`get_stats`** counts pairs, **`merge`** replaces one, **`train`** loops them; **`encode`**
  replays merges oldest-first, **`decode`** glues each token's bytes back.
- The **round-trip** `decode(encode(text)) == text` is the correctness test.
- **Vocab size** trades compression against embedding size / token rarity.
- **Regex pre-splitting** (GPT-2/4) keeps merges inside word-ish chunks; real tokenizers
  (`tiktoken`) are this same algorithm at scale, and explain many LLM quirks.

## 📖 New vocabulary

`token` · `tokenizer` · `Unicode code point` · `UTF-8` · `byte` · `byte-pair encoding (BPE)` ·
`merge` · `vocabulary` · `encode` / `decode` · `compression ratio` · `subword` ·
`out-of-vocabulary` · `regex pre-splitting` · `tiktoken`.

## 🧪 Practice & build

1. **The notebook** — [`code/explore.ipynb`](./code/explore.ipynb): build `get_stats`, `merge`,
   `train`, `encode`, `decode`, and verify the round-trip + compression.
2. **Exercises** — [`exercises/`](./exercises/): plot the vocab-size↔compression curve, stress-test
   the round-trip on emoji/accents, inspect the learned merges, and wire up the GPT-2
   `RegexTokenizer`. Tiered hints + solutions; a starter for the first.
3. **Mini-project** — [`project/`](./project/): **"Train Your Own Tokenizer"** — train BPE on a
   text *you* pick, inspect its vocabulary, measure compression, and (stretch) swap it into your
   Chapter 5 GPT.

## 🔗 Go deeper (optional)

- 🎥 Karpathy, [*Let's build the GPT Tokenizer*](https://www.youtube.com/watch?v=zduSFxRajkE) —
  this chapter follows it closely.
- 📄 [Sennrich et al. (2016), *Neural Machine Translation of Rare Words with Subword Units*](https://arxiv.org/abs/1508.07909)
  — the paper that brought BPE to NLP.
- 💻 [karpathy/minbpe](https://github.com/karpathy/minbpe) — the clean reference our `bpe.py`
  follows. · [openai/tiktoken](https://github.com/openai/tiktoken) — the fast production library.

---

| ⬅️ Previous | Up | Next ➡️ |
|:--|:-:|--:|
| [Chapter 5 — Transformer](../05-transformer/) | [Syllabus](../../README.md#-syllabus) | [Chapter 7 — Optimization](../07-optimization/) |
