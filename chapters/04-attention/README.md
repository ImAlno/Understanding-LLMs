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

Why Shakespeare? It's long enough that *context* genuinely matters — quotes open and close,
speakers alternate, lines have structure — so a model that can **look back** has a real edge over
one that can't, which is exactly what we're testing. (Names had none of that to reward.) It's also
the classic `makemore`/`nanoGPT` benchmark, so our loss numbers line up with Karpathy's.

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

To feel the difference: Chapter 3's model always blended its 3 context characters with the *same*
fixed recipe, whatever they were. Attention picks a *different* blend for every token, based on
*what the tokens actually are* — the character after a comma can reach back to the start of the
clause; a closing quote can reach back to the opening one, however far apart they sit. Nobody
programs these patterns; the model **learns** which tokens should talk to which, from data.
*Content-based, learned, variable-distance routing* — that's the superpower, and it's why this one
idea swept away a whole zoo of earlier architectures.

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

A tiny concrete case: with `B = 2`, a `q` of shape `(2, 3, 4)` times a `k.transpose` of `(2, 4, 3)`
gives `(2, 3, 3)` — for *each* of the 2 batch elements, the ordinary `(3,4) @ (4,3) = (3,3)` matmul,
stacked back into a batch. The batch axis just rides along untouched; the real work is the 2-D
matmul happening once per slice.

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

A 10-second check: `nn.Linear(4, 3)` holds a `(4, 3)` weight and a length-3 bias, so feeding it a
`(2, 4)` input returns `(2, 3)` — exactly `x @ W + b` with `(2,4) @ (4,3) = (2,3)`. It's the
Chapter-3 layer, just with `W` and `b` created and tracked for you. (We'll pass `bias=False` for the
query/key/value layers below — the bias adds nothing useful there.)

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

In a batch we stack `batch_size` such chunks, so `x` and `y` are both `(batch_size, block_size)` —
say `(32, 32)`. The model turns that into `(32, 32, vocab)` logits: a next-character prediction at
*every* position of *every* chunk — `32 × 32 = 1024` predictions scored in a single step. That
parallelism (every position of every chunk, all at once) is a big part of why Transformers train so
much faster than the older one-character-at-a-time models they replaced.

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

To *see* it average, take one number per token, `x = [10, 20, 30, 40]`: then `wei @ x` gives
`[10, 15, 20, 25]` — position 0 is just `10`, position 1 is `(10+20)/2 = 15`, position 2 is
`(10+20+30)/3 = 20`, and position 3 is the full average `25`. Each position summarizes everything
up to and including itself, and *nothing* after it.

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

To check it really gives the same uniform triangle: row 2 starts as `[0, 0, 0, −∞]` (the future
masked out), and `softmax([0, 0, 0, −∞]) = [1/3, 1/3, 1/3, 0]` — exactly row 2 of the `wei` matrix
above. Identical averaging — except now those three `0`s are *placeholders* that real query·key
scores will replace.

---

## 5. Self-attention: queries, keys, values

> ✏️ **In the notebook → Step 3.** Build a real attention head.

Now we make the weights smart. Each token produces three vectors, each from its own
`nn.Linear` layer:

- a **query** `q` — *"what am I looking for?"*
- a **key** `k` — *"what do I contain?"*
- a **value** `v` — *"what will I tell you, if you attend to me?"*

A search-engine analogy: your **query** is what you type into a library search; each book's **key**
is its catalog entry (what it's *about*); the **value** is the book's actual *contents*. You match
your query against every key to decide which books are relevant, then read the contents of the
relevant ones, weighted by how relevant. Self-attention does exactly this — except every token is
*simultaneously* doing a search (its query) and *being* searchable (its key), and "reading" means
blending in the matched tokens' values.

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

**See one head on tiny numbers.** Take 3 tokens (`T = 3`), `head_size = 2`, and suppose the
queries and keys came out as:

```
q = [[1, 0],        k = [[1, 0],
     [0, 1],             [1, 1],
     [1, 1]]             [0, 1]]
```

The scores `q @ kᵀ` dot every query with every key:

```
        k0   k1   k2
q0 →  [  1,   1,   0 ]     token 0's query matches keys 0 and 1
q1 →  [  0,   1,   1 ]     token 1 matches keys 1 and 2
q2 →  [  1,   2,   1 ]     token 2 matches key 1 most strongly
```

Now mask the future (keep the lower triangle → `-inf` above the diagonal) and softmax each row:

```
q0 → [ 1, −∞, −∞ ]  → [1.00, 0,    0   ]   token 0 can only see itself
q1 → [ 0,  1, −∞ ]  → [0.27, 0.73, 0   ]   token 1 leans on token 1
q2 → [ 1,  2,  1 ]  → [0.21, 0.58, 0.21]   token 2 leans on token 1
```

Those three rows *are* the attention weights. Each token's output is its row times the value
matrix `v` — a weighted blend of the past tokens' values, with the blend **chosen by the content**.
Change `q` or `k` (which the model learns during training) and the blend changes. That single
worked example is the entire mechanism; everything else is making it fast and stacking it up.

---

## 6. Two finishing touches: scaling and the mask

**Scaling.** We multiply the scores by `1 / sqrt(head_size)`:

```python
wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
```

Without it, the dot products grow large (they're sums of `head_size` terms), softmax
saturates toward a one-hot spike, and the gradients through it shrink to nothing. The scale
keeps the scores tame so attention stays soft and trainable.

Concretely: softmax of `[1, 2, 3]` is a soft `[0.09, 0.24, 0.67]` — attention *spread* over the
tokens. But scale those up to `[10, 20, 30]` (the size unscaled scores reach at `head_size ≈ 100`)
and softmax collapses to `[0.000, 0.000, 1.000]` — a hard one-hot. A one-hot has almost no gradient
(nudging the scores barely changes an all-or-nothing output), so the head stops learning. Dividing
by `sqrt(head_size)` keeps the scores — and therefore the softmax — in the soft, trainable regime.

**The causal mask.** `masked_fill(tril[:T,:T] == 0, float("-inf"))` sets every "future"
position's score to `-inf` *before* the softmax — and `softmax(-inf) = 0`, so each token's
weights land entirely on itself and the past. This is what makes it a *language model*
(predicting the next token) rather than a model that cheats by reading ahead.

Drop the mask and you get **non-causal** attention, where every token sees the whole sequence —
past *and* future. That's wrong for *generating* text left-to-right, but exactly right for tasks
that read a fixed input all at once (classification, translation). It's the difference between
GPT-style models (causal — what we're building) and BERT-style models (non-causal). For a
Storyteller that writes one character at a time, causal is the only option: you can't attend to
words you haven't written yet.

---

## 7. Positional encoding: attention is order-blind

Here's a subtle problem: attention is a *weighted sum*, and a sum doesn't care about order —
shuffle the tokens and you'd get the same blend. But word order obviously matters in
language! Attention as built has **no idea** which token came first. (You might object that
the causal mask already involves order — but the mask only controls *which* tokens are
visible, the past; among those, the *weighting* depends purely on content, so the model still
can't tell a token at position 2 from the same token at position 5.)

Concretely: if the character `e` sits at position 2 *and* again at position 5, attention treats the
two identically — same key, same value, same content — even though "the `e` two back" and "the `e`
five back" usually matter very differently. With no position signal, the model has no handle to
tell them apart, so it can't learn order-dependent rules at all.

The fix is simple: alongside the **token** embedding (what each character is), add a
**position** embedding (where it sits) — a learned vector for each slot `0 … block_size-1`:

```python
tok = self.token_embedding(idx)                  # (B, T, C): what each token is
pos = self.position_embedding(torch.arange(T))   # (T, C): where each token is
x = tok + pos                                    # each token now knows its content AND position
```

(The shapes: `tok` is `(B, T, C)` and `pos` is `(T, C)` — adding them **broadcasts** the same `T`
position vectors across all `B` sequences, so position `t` gets the same positional nudge in every
sequence of the batch. And these position vectors are *learned* — an `nn.Embedding(block_size, C)`,
just like the token table — so the model works out for itself how to encode "first," "second," and
so on. Some models use fixed sine-wave patterns instead; learned ones are simpler and work fine at a
fixed context length.)

Now a token's representation carries both *what* it is and *where* it is, so attention can
learn order-dependent patterns ("attend to the character two back").

On tiny numbers: if `e`'s token embedding is `[0.2, −0.1]` and position 2's vector is `[0.5, 0.0]`,
then `e`-at-position-2 becomes `[0.7, −0.1]`; the *same* `e` at position 5 (vector `[−0.3, 0.4]`)
becomes `[−0.1, 0.3]` — a genuinely different vector. The single addition is what lets attention
finally tell the two apart.

---

## 8. The whole head, the model, and training

Putting §5–§7 together, here's the data flow — three projections of `x`, then the weights, then one
weighted sum:

```
        ┌─► query ─┐
   x ───┼─► key  ──┴─► q·kᵀ · scale · mask(−∞) · softmax ─► wei ─┐
        └─► value ─────────────────────────────────────────────┴─► wei @ v ─► out
```

In code, a single attention **`Head`** is:

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

Trace the shapes through `forward`, with `B = 1, T = 8, n_embd = 64, head_size = 16`:

```
x                            (1, 8, 64)   8 tokens, 64 numbers each
k, q, v = key/query/value(x) (1, 8, 16)   project each token down to head_size
q @ k.transpose(-2,-1)       (1, 8, 8)    every token's score against every token
masked_fill + softmax        (1, 8, 8)    causal weights; each row sums to 1 over the past
wei @ v                      (1, 8, 16)    each token: a weighted blend of past values
```

So the head takes `(B, T, 64)` in and hands back `(B, T, 16)` — a *context-aware* version of each
token, where every token has mixed in information from the relevant earlier ones.

The model wraps it: token + position embeddings → the head → a final `nn.Linear` to
`vocab_size` **logits**. We train with **`torch.optim.AdamW`** — a smarter optimizer than our
hand-rolled SGD (it adapts the step size per parameter; Chapter 7 explains it) — but the loop
is the same reset → backward → update:

```python
optimizer.zero_grad(); loss.backward(); optimizer.step()
```

(What's actually training here? Three weight matrices per head — `key`, `query`, `value`, each
`n_embd × head_size` — plus the token and position embeddings and the final output `nn.Linear`. For
our tiny config that's a few tens of thousands of numbers, all tuned by the same backprop you built
in Chapter 2, now wrapped in AdamW.)

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

One head can track only *one kind* of relationship at a time — it has a single query/key/value, so
it learns a single notion of "what's relevant." Real models run **many heads in parallel**: one
might follow subject–verb agreement, another match brackets and quotes, another handle spacing —
each with its own q/k/v — then combine them. Adding exactly that is the first step of Chapter 5.

---

## 🐛 Building it yourself: what trips people up

The concepts are clean; the *shapes* are where people stumble. The usual snags:

- **`.transpose(-2, -1)`, not `.T`.** You only want to swap the *last two* axes (`(B,T,hs)` →
  `(B,hs,T)`), leaving the batch axis alone. Plain `.T` reverses all three and the matmul breaks.
- **Mask the score, not the weight.** Set future *scores* to `-inf` **before** softmax — not the
  weights to 0 after. `-inf` lets softmax renormalize the survivors so each row still sums to 1;
  zeroing weights afterward leaves rows that don't.
- **Don't forget the scale.** Drop the `* head_size ** -0.5` and training still runs but plateaus
  high (the softmax saturates, §6). Easy to miss, because nothing errors.
- **`tril[:T, :T]`, not `tril`.** The buffer is `block_size × block_size`, but a batch may be
  shorter (`T ≤ block_size`), so slice it to the current `T` or the shapes won't line up.
- **When in doubt, print the shape.** Almost every attention bug is a shape that isn't what you
  think it is. `print(wei.shape)` settles it faster than staring.

None of these are conceptual — they're the handful of shape papercuts everyone meets building their
first head.

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
- **How is this different from the Chapter 3 MLP?** The MLP blended a *fixed* window with *fixed*
  weights. Attention blends a *variable-length* past with weights *computed from the tokens
  themselves*, so one model handles "look one character back" and "look fifty back" with no change
  in structure — it just learns the right query/key patterns.
- **Does every token really attend to every earlier token, every step?** Yes — the `(T, T)` score
  matrix is all pairs at once. That's `T²` comparisons, which is why *very* long contexts get
  expensive (a real cost later chapters address). At our `block_size` it's cheap.
- **Why did attention beat older approaches like RNNs?** Two reasons. *Parallelism* — an RNN walks
  a sequence one token at a time (token 5 must wait for token 4), while attention scores every
  position at once, which suits modern hardware far better. *Long range* — an RNN carries
  information forward step by step and it fades; attention reaches any earlier token **directly**,
  in a single hop, however far back. Direct, parallel, learned routing is why the field switched
  over almost completely.

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

<details>
<summary>5. Why divide the attention scores by √(head_size) before softmax?</summary>

The scores are dot products of `head_size`-long vectors, so they grow with `head_size`. Large
scores push softmax toward a one-hot, and a one-hot has almost no gradient — the head stops
learning. Dividing by √(head_size) keeps the scores (and the softmax) in a soft, trainable range.
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
