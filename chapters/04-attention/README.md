# Chapter 4 — Attention: Letting the Model Look Back

> *The idea that made modern LLMs possible.* Chapter 3's model saw a fixed three
> characters and blended them the same way every time. **Attention** lets every position
> look back at *all* the earlier positions and decide, on the fly, which ones matter — so
> a token can reach back to whatever it needs. We build one self-attention head from
> scratch, train it on a megabyte of Shakespeare, and watch the gibberish reorganize into
> something that looks like English.

**You will be able to:**
- explain **self-attention** — queries, keys, and values — and what each one is *for*;
- do the "weighted average of the past" trick with a triangular matrix multiply;
- build a **causal** self-attention head (no peeking at the future);
- add **positional encoding** so the model knows the order of its input;
- train it on Shakespeare and generate text — and recognize this as the heart of the
  Transformer we assemble in Chapter 5.

**Prerequisites:** [Chapter 3](../03-ngram-mlp/) (embeddings, the MLP, tensors,
`cross_entropy`) and [Chapter 2](../02-micrograd/) (backprop). We introduce three PyTorch
conveniences (`nn.Module`, `nn.Linear`, `nn.Embedding`) and explain each — they're just
tidy wrappers for the weight matrices you built by hand in Chapter 3.

**Time:** ~3 hours building along. **Hardware:** a laptop CPU — the model trains in ~10s.

> ### 📓 Build it yourself in the notebook
> **[`code/explore.ipynb`](./code/explore.ipynb)** walks you through the heart of it — the
> averaging trick, then a real attention head, then training and sampling — with "✍️ Your
> turn" fill-ins, "📖 Study & run" cells, and "▶️ Check your work" cells. Watch for: ✏️ **In
> the notebook → Step N**.

---

## 0. Setup

We switch datasets here. Names were too short for attention to show its worth, so we move
to **tiny Shakespeare** — about 1MB of his plays as plain text, already downloaded to
[`data/input.txt`](./data/input.txt). (Generating text is still squarely in our
Storyteller wheelhouse.)

```bash
uv venv --python 3.13 .venv && uv pip install -r requirements.txt
.venv/bin/jupyter lab        # open chapters/04-attention/code/explore.ipynb
```

The finished reference is [`code/attention.py`](./code/attention.py).

---

## A 2-minute primer: what attention does

When *you* read "the cat sat on the mat because **it** was tired," you instantly know
"it" = the cat. You **looked back** at an earlier word and decided it was the relevant one.
That is attention: for each position, gather information from the earlier positions that
matter *to it*, and ignore the rest.

Chapter 3 couldn't do this. It crammed a fixed 3-character window into one vector and
treated every slot identically. Attention is the opposite: the context can be long, and
each token *chooses* — by learning — which earlier tokens to pull information from. No fixed
window, no fixed pattern. That flexibility is the whole reason Transformers work.

---

## 1. The plan, and the shape convention

We'll feed the model chunks of text and have **every position predict the next character**,
using only the characters up to and including itself. To pull that off, each position
gathers context from its past via attention.

One piece of vocabulary first, because every tensor in this chapter is **3-D**:

> **`(B, T, C)`** = **(Batch, Time, Channels)** = (how many sequences at once · how many
> positions per sequence · how many numbers per position). E.g. `(4, 8, 64)` is 4 sequences
> of 8 characters, each character represented by 64 numbers.

"Time" just means *position in the sequence* (token 0, 1, 2, …), and "channels" is just *the
numbers that represent each token* (its embedding). This is the same shape idea as Chapter
3's `(N, 3, 10)` block — with one crucial difference: there we **flattened** the 3 context
characters into a single vector and made one prediction. Here we **keep** all `T` positions
and let **every one make its own prediction at once**. That's why, at the end, the logits are
`(B, T, vocab)` — a prediction at each of the T positions — and why we'll `.view(B*T, vocab)`
to score them exactly like Chapter 3's `(N, vocab)`.

**Reading a matmul on 3-D tensors.** Chapter 3's matmuls were 2-D: `(N, 30) @ (30, 200)`. The
ones here look scarier but follow one rule: **`@` works on the *last two* dimensions and loops
over the rest.** So `(B, T, hs) @ (B, hs, T)` does the ordinary 2-D matmul `(T, hs) @ (hs, T) =
(T, T)` *once per batch element*, giving `(B, T, T)`. (And a 2-D matrix times a 3-D tensor,
like `(T, T) @ (B, T, C)`, applies that one matrix to every batch slice → `(B, T, C)`.) The
one other tool we need is **`.transpose(-2, -1)`**, which swaps the last two axes — e.g.
turning `(B, T, hs)` into `(B, hs, T)` so the inner dimensions line up for the matmul. Keep
those two rules handy and every shape below reads like a sentence.

---

## 2. Three new tools: `nn.Module`, `nn.Linear`, `nn.Embedding`

In Chapter 3 we made weight matrices by hand (`W1 = torch.randn(...)`) and tracked them in a
`params` list. PyTorch has tidy wrappers for exactly that — same math, less bookkeeping:

- **`nn.Linear(in, out)`** *is* Chapter 3's `x @ W + b`. It creates and tracks the weight
  matrix `W` (shape `in × out`) and bias `b` for you; calling `layer(x)` computes `x @ W + b`.
- **`nn.Embedding(num, dim)`** *is* Chapter 3's embedding table `C` and the `C[X]` lookup,
  wrapped: `emb(ids)` returns the rows for those ids.
- **`nn.Module`** is a base class you inherit from. It auto-collects every `nn.Linear` /
  `nn.Embedding` you put inside, so `model.parameters()` hands you all the weights to train —
  no manual list. (You'll write `class Head(nn.Module)` with `super().__init__()` as the first
  line of `__init__` — boilerplate that lets the base class set itself up.)

That's the whole abstraction. When you see `self.key = nn.Linear(...)` below, read it as "a
weight matrix, managed for me."

---

## 3. The data: many predictions at once

> ✏️ **In the notebook → Step 1.** Load the data and grab a batch.

We turn the text into character ids, then grab random chunks. A chunk `x` is `block_size`
characters; its target `y` is the *same chunk shifted one character to the right*:

```python
x = data[i : i + block_size]          # e.g. "First Citi"
y = data[i + 1 : i + block_size + 1]  #      "irst Citiz"  ← each char's "next char"
```

So position 0's target is `x`'s character 1, position 1's target is character 2, and so on.
**One chunk gives `block_size` predictions at once** — position `t` predicts character `t+1`
using only characters `0…t`. (That "only `0…t`" is the *causal* rule, enforced by the mask in
§6.) Training on whole chunks like this is what makes Transformers efficient: every position
learns simultaneously.

---

## 4. The core trick: a weighted average of the past

> ✏️ **In the notebook → Step 2.** Build this — it's the engine of attention.

Forget queries and keys for a moment. Suppose each token just wanted the **average of itself
and every token before it** — a crude way to "summarize the past." The slow way is a loop.
The fast way is one matrix multiply, and it's the trick the whole chapter rests on.

Build a `(T, T)` matrix `wei` that is **lower-triangular and row-normalized** — row `t` has
equal weights in columns `0…t` and zeros after:

```
wei = [[1.00, 0,    0,    0   ],
       [0.50, 0.50, 0,    0   ],
       [0.33, 0.33, 0.33, 0   ],
       [0.25, 0.25, 0.25, 0.25]]
```

Then `wei @ x` gives, for every position, the average of `x` over all positions up to `t`.
The lower-triangular shape is what enforces "only look at the past":

```python
tril = torch.tril(torch.ones(T, T))        # lower-triangular ones
wei = tril / tril.sum(1, keepdim=True)      # normalize each row to sum to 1
xbow = wei @ x                              # (T, T) @ (B, T, C) → (B, T, C)
```

(`tril.sum(1, keepdim=True)` sums each row and *keeps* the result a column — the `keepdim`
trick from Chapter 1's §3 — so the divide broadcasts across the row; and `wei @ x` is the
"2-D matrix × 3-D tensor" case from §1.)

**Attention is this exact operation** — except the weights in `wei` aren't fixed and uniform;
they're *learned* and *depend on the tokens*. So a token can weight a relevant earlier token
heavily and ignore the rest. To get there, we build `wei` a slightly different (equivalent)
way that leaves room for learned scores:

```python
wei = torch.zeros(T, T)
wei = wei.masked_fill(tril == 0, float("-inf"))   # the future → -inf
wei = F.softmax(wei, dim=-1)                       # → the same uniform triangular weights
```

(New ops there: `masked_fill(cond, value)` overwrites the entries where `cond` is true —
`tril == 0` is true exactly on the future, the upper triangle; `float("-inf")` is negative
infinity; and `softmax(..., dim=-1)` runs along the *last* axis, i.e. across each row. Softmax
of all-equal numbers is uniform, and `-inf` becomes `0` after softmax.) This looks
like a roundabout way to average — but the instant those zeros become real, data-dependent
scores, softmax turns them into a **learned attention pattern**. That's the next step.

---

## 5. Self-attention: queries, keys, values

> ✏️ **In the notebook → Step 3.** Build a real attention head.

Now we make the weights smart. Each token produces three vectors, each from its own
`nn.Linear` layer:

- a **query** `q` — *"what am I looking for?"*
- a **key** `k` — *"what do I contain?"*
- a **value** `v` — *"what will I tell you, if you attend to me?"*

The attention score from token *i* to token *j* is the **dot product** of *i*'s query with
*j*'s key — large when they "match." We compute every pair's score in one matrix multiply:

```python
k = self.key(x)      # (B, T, head_size)
q = self.query(x)    # (B, T, head_size)
wei = q @ k.transpose(-2, -1)        # (B,T,hs) @ (B,hs,T) → (B, T, T)
```

`wei[b, i, j]` is now "how much should token *i* attend to token *j*?", *learned* from the
tokens. We mask the future, softmax each row into weights, and take the weighted sum of the
**values**:

```python
wei = wei.masked_fill(tril[:T, :T] == 0, float("-inf"))   # causal: no peeking ahead
wei = F.softmax(wei, dim=-1)                                # each row sums to 1, over the past
v = self.value(x)                                          # (B, T, head_size)
out = wei @ v                                              # (B,T,T) @ (B,T,hs) → (B, T, head_size)
```

`out` is the head's output: for each token, a custom blend of earlier tokens' *values*,
weighted by how well its query matched their keys. That's it — that's attention. (Why
project to `value` instead of summing the raw `x`? So the model can choose *what information*
to pass along, separately from *what to match on*.)

---

## 6. Two finishing touches: scaling and the mask

**Scaling.** We multiply the scores by `1 / sqrt(head_size)`:

```python
wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
```

Without it, the dot products grow large (they're sums of `head_size` terms), softmax
saturates toward a one-hot spike, and the gradients through it shrink to nothing. The scale
keeps the scores tame so attention stays soft and trainable.

**The causal mask.** `masked_fill(tril[:T,:T] == 0, float("-inf"))` sets every "future"
position's score to `-inf` *before* the softmax — and `softmax(-inf) = 0`, so each token's
weights land entirely on itself and the past. This is what makes it a *language model*
(predicting the next token) rather than a model that cheats by reading ahead. (Drop the mask
and you get **non-causal** attention — useful for tasks like translation, but not for
generation.)

---

## 7. Positional encoding: attention is order-blind

Here's a subtle problem: attention is a *weighted sum*, and a sum doesn't care about order —
shuffle the tokens and you'd get the same blend. But word order obviously matters in
language! Attention as built has **no idea** which token came first. (You might object that
the causal mask already involves order — but the mask only controls *which* tokens are
visible, the past; among those, the *weighting* depends purely on content, so the model still
can't tell a token at position 2 from the same token at position 5.)

The fix is simple: alongside the **token** embedding (what each character is), add a
**position** embedding (where it sits) — a learned vector for each slot `0 … block_size-1`:

```python
tok = self.token_embedding(idx)                  # (B, T, C): what each token is
pos = self.position_embedding(torch.arange(T))   # (T, C): where each token is
x = tok + pos                                    # each token now knows its content AND position
```

Now a token's representation carries both *what* it is and *where* it is, so attention can
learn order-dependent patterns ("attend to the character two back").

---

## 8. The whole head, the model, and training

Putting §5–§7 together, a single attention **`Head`** is:

```python
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
    def forward(self, x):
        B, T, C = x.shape
        k, q = self.key(x), self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        return wei @ self.value(x)
```

(`register_buffer` just stores `tril` on the module without making it a trainable weight.)
The model wraps it: token + position embeddings → the head → a final `nn.Linear` to
`vocab_size` **logits**. We train with **`torch.optim.AdamW`** — a smarter optimizer than our
hand-rolled SGD (it adapts the step size per parameter; Chapter 7 explains it) — but the loop
is the same reset → backward → update:

```python
optimizer.zero_grad(); loss.backward(); optimizer.step()
```

🔮 **Predict before you run:** the bigram baseline on Shakespeare bottoms out around **2.5**.
What does one attention head get?

<details><summary>👉 Reveal</summary>

```
step     0 | train 4.1552 | val 4.1572
step  4999 | train 2.2997 | val 2.3394
```

A real drop below the bigram — but the loss isn't the headline. The *samples* are:
</details>

```
POr,
Coll. I Tshat ke thisingus?
Wemeisares wiflithrei tius ming Plithown wo D:
...
STres heaguty, prett foomout re hto dimee thy
```

Still gibberish — but look: word-sized clusters, capitalization, line breaks, even
`Speaker:`-shaped lines. One head learned the *shape* of English from characters alone. The
full Transformer (Chapter 5) drives this to actually-readable Shakespeare.

---

## 🤔 Common questions

- **Why three separate vectors (q, k, v) instead of one?** They do different jobs: the query
  and key decide *who attends to whom* (the weights), while the value decides *what gets
  passed along*. Splitting them lets the model match on one thing and communicate another.
- **Why divide by `sqrt(head_size)`?** To keep the dot-product scores from blowing up with
  `head_size`. Large scores make softmax nearly one-hot, which kills the gradient. The scale
  keeps attention soft and learnable.
- **What exactly does the `-inf` masking do?** It zeroes out the future *after* softmax (since
  `softmax(-inf)=0`), so each position attends only to itself and earlier positions — the rule
  that makes this a next-token predictor.
- **Why add position embeddings — doesn't the order come for free?** No. Attention is a sum
  over weighted values, and sums ignore order. The position embedding is what tells the model
  token 5 came after token 4.
- **Is this really how ChatGPT works?** Yes — this *is* the core operation. ChatGPT uses many
  attention heads in parallel (**multi-head**, Chapter 5), stacks dozens of layers, and adds a
  feed-forward net + residuals + LayerNorm around each — but every one of those layers runs
  the exact attention you just built.

---

## ✅ Check your understanding

<details>
<summary>1. In one sentence each, what are the query, key, and value?</summary>

The **query** is what a token is looking for; the **key** is what a token contains (query·key
= how much to attend); the **value** is the information a token passes along when attended to.
</details>

<details>
<summary>2. What does the lower-triangular mask accomplish, and why <code>-inf</code> (not 0)?</summary>

It makes attention **causal** — each token can attend only to itself and earlier tokens, never
the future. We use `-inf` *before* softmax because `softmax(-inf) = 0`; setting the score
(not the weight) to `-inf` lets softmax renormalize the remaining weights correctly.
</details>

<details>
<summary>3. <code>q</code> and <code>k</code> are <code>(B, T, hs)</code>. What shape is <code>q @ k.transpose(-2,-1)</code>, and what does entry <code>[b,i,j]</code> mean?</summary>

`(B, T, T)` — an attention-score matrix per sequence. Entry `[b, i, j]` is how much token `i`
should attend to token `j` (its query dotted with `j`'s key).
</details>

<details>
<summary>4. Why does the model need a <i>position</i> embedding on top of the <i>token</i> embedding?</summary>

Attention is a weighted sum, which is order-independent — without position info, "abc" and
"cba" would look the same. The position embedding injects *where* each token is, so the model
can learn order-dependent patterns.
</details>

## 🎓 Key takeaways

- **Attention** lets each position gather information from a learned, weighted selection of
  the *earlier* positions — no fixed context window.
- The engine is a **weighted average of the past**, done as a triangular matrix multiply; the
  weights come from **softmax of query·key scores**.
- **Query / key / value**: q·k sets the weights, v is what's passed along. Scale by
  `1/sqrt(head_size)`; mask the future with `-inf` before softmax.
- Attention is **order-blind**, so we add a **position embedding** to the token embedding.
- `nn.Linear` / `nn.Embedding` / `nn.Module` are just Chapter 3's matrices, wrapped.
- One head on Shakespeare already learns the *shape* of English. Stacking many is the
  Transformer — Chapter 5.

## 📖 New vocabulary

`attention` · `self-attention` · `query` · `key` · `value` · `attention weights` ·
`scaled dot-product` · `causal mask` · `lower-triangular` · `(B, T, C)` convention ·
`token embedding` · `position embedding` · `head` / `head_size` · `nn.Module` · `nn.Linear` ·
`nn.Embedding` · `AdamW`.

## 🧪 Practice & build

1. **The notebook** — [`code/explore.ipynb`](./code/explore.ipynb): the averaging trick, a
   real head, training, sampling. Do this first.
2. **Exercises** — [`exercises/`](./exercises/): prove the triangular-matmul trick equals a
   loop, visualize the attention weights, remove the mask and see what breaks, tune
   `block_size`. Tiered hints + solutions; a starter for the first.
3. **Mini-project** — [`project/`](./project/): **"The Attention Microscope"** — train the
   head, then *watch* which earlier characters it looks at when predicting the next one.

## 🔗 Go deeper (optional)

- 🎥 Karpathy, [*Let's build GPT: from scratch, in code, spelled out*](https://www.youtube.com/watch?v=kCc8FmEb1nY)
  — this chapter follows the attention portion closely.
- 📄 [Bahdanau et al. (2014), *Neural Machine Translation by Jointly Learning to Align and Translate*](https://arxiv.org/abs/1409.0473)
  — the paper that introduced attention.
- 📄 [Jay Alammar, *The Illustrated Transformer*](https://jalammar.github.io/illustrated-transformer/)
  — the classic visual, intuition-first walkthrough of attention.

---

| ⬅️ Previous | Up | Next ➡️ |
|:--|:-:|--:|
| [Chapter 3 — N-gram MLP](../03-ngram-mlp/) | [Syllabus](../../README.md#-syllabus) | [Chapter 5 — Transformer](../05-transformer/) |
