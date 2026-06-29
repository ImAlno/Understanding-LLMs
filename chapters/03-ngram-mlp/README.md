# Chapter 3 — The N-gram MLP: Embeddings & More Context

> *Our Storyteller gets a memory and a mind's eye.* The bigram model in Chapter 1 saw
> exactly **one** previous character. Here we let the model see the previous **three** —
> and, more importantly, it learns its own little vector of numbers for each character,
> called an **embedding**. With tensors doing the heavy lifting (the autograd you built
> in Chapter 2, now applied to whole arrays), the loss drops well below the bigram's and
> the names become genuinely name-like: `jahmer`, `kaeli`, `nellara`, `kaleigh`.

**You will be able to:**
- explain what an **embedding** is, and why a learned vector beats a one-hot;
- give the model a **context** of several characters (`block_size`);
- build a **multi-layer perceptron (MLP)** language model with PyTorch **tensors**;
- train it with **minibatches** and the **cross-entropy** loss;
- reach a markedly lower loss (~2.15 vs the bigram's 2.45) and better samples;
- see how *real* LLMs represent their tokens — this is the same idea, scaled up.

**Prerequisites:** [Chapter 1](../01-bigram/) (a language model = probabilities over the
next token, the loss, sampling) and [Chapter 2](../02-micrograd/) (so `loss.backward()`
is no longer mysterious). Basic Python + the tensor basics from Chapter 1.

**Time:** ~2–3 hours building along. **Hardware:** a laptop CPU is plenty — the whole
model trains in about **5 seconds**.

> ### 📓 Build it yourself in the notebook
> **[`code/explore.ipynb`](./code/explore.ipynb)** walks you through building the model
> piece by piece — the embedding lookup, the forward pass, the loss, training, sampling —
> with "✍️ Your turn" fill-ins (hint + answer one click away), "📖 Study & run" cells for
> the mechanical parts, and "▶️ Check your work" cells. Read a section here, build it
> there. Watch for: ✏️ **In the notebook → Step N**.

---

## 0. Setup

If Chapter 1 is set up, you're ready (same dataset, `data/names.txt`, already copied in).
Otherwise, from the repo root:

```bash
uv venv --python 3.13 .venv && uv pip install -r requirements.txt
.venv/bin/jupyter lab        # open chapters/03-ngram-mlp/code/explore.ipynb
```

The finished reference is [`code/mlp.py`](./code/mlp.py) — run it any time with
`python chapters/03-ngram-mlp/code/mlp.py`.

---

## A 2-minute primer: embeddings

The one big new idea this chapter is the **embedding**. Here's the problem it solves.

In Chapter 1's neural net, we fed a character to the model as a **one-hot** vector — 27
numbers, a single `1` and 26 zeros. That works, but it's *dumb*: every character is
equally far from every other. The one-hot for `a` knows nothing about the one for `e`,
even though vowels behave alike. The model can't share what it learns about `a` with `e`.

An **embedding** fixes this. We give each character a short vector of, say, **10
numbers** — and we let the model **learn** those numbers during training:

```
one-hot 'e'  =  [0,0,0,0,1,0,0, ... ,0]        (27 numbers; says nothing about 'e')
embedding 'e' = [0.31, -1.2, 0.04, ... , 0.7]  (10 learned numbers; a little profile of 'e')
```

Think of it as a learned *personality profile* for each character. Nothing tells the
model to do this, but to lower its loss it tends to place similar characters near each
other (vowels in one neighbourhood, say), so a lesson learned about `a` partly transfers
to `e`. That **sharing** is why embeddings generalize better than one-hots — and it's
exactly how every modern LLM represents its tokens. We store all 27 profiles in one
table `C` of shape `(27, 10)`; row `i` is character `i`'s embedding.

---

## 1. The plan

Two upgrades over the bigram:
1. **More context.** Predict the next character from the previous `block_size` characters
   (we'll use 3) instead of just one. More context → better guesses.
2. **Learned representations.** Feed those characters in as **embeddings**, not one-hots,
   and pass them through a small **MLP** (a couple of layers) to make the prediction.

This is the model of Bengio et al. (2003) — the first neural language model — at the
character level. Everything runs on PyTorch **tensors**: the same autograd from Chapter 2,
but operating on whole arrays at once, which is what makes it fast.

---

## 2. The data, now with context

> ✏️ **In the notebook → Step 2.** Build the dataset yourself.

For each name we slide a window of `block_size` characters and record *(context → next
char)*. We pad the start with the `.` token (id 0). For `emma` with `block_size = 3`:

```
context        next
. . .    →     e
. . e    →     m
. e m    →     m
e m m    →     a
m m a    →     .   (end)
```

In code, we keep a rolling `context` list and slide it forward one step at a time:

```python
X, Y = [], []
for w in words:
    context = [0] * block_size                 # start with '...'
    for ch in w + ".":                         # walk the name, then the end token
        ix = stoi[ch]
        X.append(context)                      # the 3 ids we're predicting from
        Y.append(ix)                           # the id we want to predict
        context = context[1:] + [ix]           # drop the oldest, append the new char
X, Y = torch.tensor(X), torch.tensor(Y)
```

`X` ends up with shape `(N, 3)` — `N` examples, each a row of 3 context ids — and `Y` has
shape `(N,)`, one target id per example. (The trailing comma in `(N,)` just marks a **1-D**
tensor — a flat list of `N` numbers — versus the 2-D grid `(N, 3)`.) We also split the names
80/10/10 into train/dev/test, the discipline from Chapter 1's exercises.

---

## 3. Embeddings: a lookup table

> ✏️ **In the notebook → Step 4.** Do the embedding lookup yourself.

The embedding table is just a tensor of random numbers to start — the model will tune it:

```python
C = torch.randn((27, 10))      # 27 characters, a 10-number embedding each
```

How do we look things up in it? Let's build the idea up one step at a time:

```python
C[5]            # one id      →  that character's embedding,  shape (10,)
C[[5, 7]]       # a list of 2 ids  →  2 embeddings stacked,   shape (2, 10)
C[X]            # X is (N, 3) ids  →                          shape (N, 3, 10)
```

That last line is the move, and it's called **fancy indexing**: when you index `C` with a
tensor of ids, PyTorch replaces *every id* with that id's row of `C`, keeping the shape of
the ids and tacking the embedding length (`10`) onto the end. So our `(N, 3)` grid of ids
becomes an `(N, 3, 10)` block — **picture it as `N` pages, each a `3 × 10` grid**: for every
example, its 3 context characters, each now a 10-number vector.

```python
emb = C[X]      # (N, 3) ids  →  (N, 3, 10) embeddings
```

(This is the *batched* big sibling of Chapter 1's exercise E04, where we indexed a table
with a **1-D** list of ids and got a 2-D result. Here the ids are 2-D, so the result gains a
dimension — same lookup idea, one dimension up. If you skipped E04, no problem: the three
lines above are the whole story.)

Before the MLP can use it, we **flatten** each example's `3 × 10` block into one long row
of `30` numbers:

```python
x = emb.view(emb.shape[0], -1)     # (N, 3, 10)  →  (N, 30)
```

`.view(...)` just **re-labels** a tensor's shape — it rearranges nothing, only regroups the
same numbers — and `-1` means "you work out this dimension" (here, `3 × 10 = 30`). The 30
numbers stay in order (character 1's ten, then character 2's ten, then character 3's ten),
so nothing is lost; the network simply sees one flat row per example.

---

## 4. The MLP

> ✏️ **In the notebook → Step 5.** Build the forward pass.

A **multi-layer perceptron** is just linear layers with a nonlinearity between them. Ours
has one hidden layer:

```python
W1 = torch.randn((30, 200)) * 0.2;   b1 = torch.randn(200) * 0.01   # hidden layer: 30 → 200
W2 = torch.randn((200, 27)) * 0.01;  b2 = torch.zeros(27)           # output: 200 → 27 (small init — see §7)

h = torch.tanh(x @ W1 + b1)        # (N, 30) @ (30, 200) → (N, 200), squashed by tanh
logits = h @ W2 + b2               # (N, 200) @ (200, 27) → (N, 27): a score per next-char
```

If the shapes feel fast, slow down — they're the whole story (and `@` is the matrix
multiply from Chapter 1's §6: `(N, 30) @ (30, 200) → (N, 200)`, the inner `30`s cancel):

- `x` is `(N, 30)` — the flattened embeddings.
- `x @ W1 + b1` mixes those 30 numbers into `200` hidden values per example. (Adding the
  length-`200` bias `b1` to the `(N, 200)` result **broadcasts** — that align-from-the-right
  rule from Chapter 1's §3 — so the same bias is added to every row.) Then `tanh` squashes
  each value into `(−1, 1)` — the nonlinearity from Chapter 2 that lets a network learn
  curved relationships.
- `h @ W2 + b2` turns the 200 hidden values into `27` **logits** — one raw score for each
  possible next character, exactly like Chapter 1.

> 🔭 **GELU, for later.** We use `tanh` (it's familiar from Chapter 2). Modern networks
> usually swap it for **GELU** or ReLU, which tend to train a little better — you'll try
> that in the exercises, and we lean on it from the Transformer onward.

---

## 5. The loss: `cross_entropy`

Those 27 logits become probabilities with **softmax**, and we score them with the
**negative log-likelihood** — exactly Chapter 1. PyTorch rolls both steps into one call:

```python
loss = F.cross_entropy(logits, Y)
```

`F.cross_entropy(logits, Y)` does *softmax then negative-log-likelihood of the correct
character* in a single, numerically-stable step. (You proved it equals the hand-written
version in Chapter 1's exercise E05; from here on we always use it — it's faster and won't
overflow when logits get large.)

---

## 6. Minibatches: train on a handful at a time

Our training set has ~182,000 examples. Computing the loss and gradient over *all* of them
every step would be correct but slow. Instead, each step we grab a **random minibatch** of
32 examples and update on those:

```python
ix = torch.randint(0, Xtr.shape[0], (32,))     # 32 random example indices
Xb, Yb = Xtr[ix], Ytr[ix]                       # this step's minibatch
```

(`torch.randint(0, Xtr.shape[0], (32,))` draws `32` random whole numbers between `0` and the
number of examples — the `(32,)` just says "give me 32 of them" — and `Xtr[ix]` grabs those
32 rows.)

The gradient from 32 random examples is a *noisy but cheap* estimate of the true gradient —
and it turns out that taking many fast, slightly-wrong steps beats taking few slow, perfect
ones. This is **stochastic gradient descent**, and every large model is trained this way.

---

## 7. Training, and the payoff

> ✏️ **In the notebook → Step 6.** Run the training loop and watch the loss fall.

The loop is the one you already know — forward, backward, update — now on tensors:

```python
for step in range(30000):
    ix = torch.randint(0, Xtr.shape[0], (32,))
    logits = forward(Xtr[ix])
    loss = F.cross_entropy(logits, Ytr[ix])
    for p in params: p.grad = None
    loss.backward()                              # the Chapter-2 engine, on tensors
    lr = 0.1 if step < 18000 else 0.01           # drop the step size partway through
    for p in params: p.data += -lr * p.grad
```

Two lines that deserve a clarification each. **`p.grad = None`** resets the gradients first,
because PyTorch *adds into* `.grad` on every `backward()` (the same accumulation you built in
Chapter 2 — so we clear it each step or gradients would pile up). And **`loss.backward()` is
genuinely the Chapter-2 engine**: it walks the graph and applies each operation's local
gradient rule — PyTorch just knows the rules for tensor ops like `@` and `tanh`, so the exact
same backprop runs on whole arrays instead of one scalar at a time.

🔮 **Predict before you run:** the bigram bottomed out at **2.45**, the trigram-by-counting
at **2.21**. Where will an MLP with 3 characters of context and learned embeddings land?

<details><summary>👉 Reveal the result</summary>

```
train loss 2.1224 | dev loss 2.1488
```

Below the trigram, and — unlike a giant count table — it *generalizes*: the dev loss (on
names it never trained on) is right next to the train loss.
</details>

The **dev** loss is the honest number (names the model never saw). At **~2.15** it beats
every Chapter 1 model, and it does so with only ~12,000 parameters that *generalize*,
rather than a giant lookup table.

---

## 8. Sampling: meet the new names

> ✏️ **In the notebook → Step 7.** Generate names from your model.

Same autoregressive loop as Chapter 1 — but now we feed the last 3 characters through the
MLP each step, sample the next, and slide the context forward:

```
jahmer   kaeli   nellara   kaleigh   quint   suline   carmahza   kanden   kimrix
```

Compare those to the bigram's `cexze` and `staiyaubrtthrigotai`. Three characters of memory
and a learned representation make a real difference — and the path from here (Chapters 4–5)
is simply: *much* more context, learned even more cleverly.

---

## 🤔 Common questions

- **What's actually "learned" in the embedding table `C`?** Numbers that make the loss
  small. In practice the model nudges similar characters (e.g. vowels) toward similar
  vectors so it can share statistics. With `n_embd = 2` you can literally plot them on a
  2-D graph and see the clusters — that's an exercise.
- **Why is `loss.backward()` not magic anymore?** Because you built it in Chapter 2.
  PyTorch's autograd is the same idea on tensors instead of scalars.
- **Why does the loss bounce around during training (e.g. 2.0 then 2.5)?** That's the
  minibatch noise — each step sees a different random 32 examples. The *trend* is down;
  the per-step number wobbles. The full-split losses at the end are the real measure.
- **Why did we start the output weights tiny (`* 0.01`)?** So the first predictions are
  near-uniform and the loss starts around `log(27) ≈ 3.3` (the loss of a model that just
  guesses uniformly among the 27 characters) instead of a huge number. Initialization is a
  whole topic — Chapter 7.
- **Is this really how big LLMs work?** The shape is identical: tokens → embeddings →
  layers → logits → softmax → cross-entropy. ChatGPT swaps our one hidden layer for a deep
  Transformer (Chapter 5) and 3 characters for thousands of tokens, but you're looking at
  the skeleton.

---

## ✅ Check your understanding

<details>
<summary>1. What's the difference between a one-hot and an embedding, and why does the embedding help?</summary>

A one-hot is a fixed vector with a single `1` — every character is equidistant and carries
no information about itself. An embedding is a short vector of *learned* numbers, so the
model can place similar characters near each other and **share** what it learns across
them, which generalizes better.
</details>

<details>
<summary>2. <code>X</code> is <code>(N, 3)</code> and <code>C</code> is <code>(27, 10)</code>. What shape is <code>C[X]</code>, and why?</summary>

`(N, 3, 10)`. Indexing `C` with the id tensor `X` replaces *each* id with its 10-number
row, so every one of the `N` examples becomes `3` characters × `10` numbers.
</details>

<details>
<summary>3. Why <code>emb.view(N, -1)</code> before the hidden layer?</summary>

The MLP's first layer expects one flat vector per example, not a 3×10 grid. `.view(N, -1)`
flattens each example's `3 × 10` embeddings into a single row of `30` numbers (the `-1`
means "infer this size").
</details>

<details>
<summary>4. Why train on random minibatches of 32 instead of all ~182,000 examples each step?</summary>

Speed. A minibatch gives a noisy-but-cheap estimate of the gradient, and many fast
approximate steps reach a good model faster than a few slow exact ones. That's stochastic
gradient descent — how every large model is trained.
</details>

<details>
<summary>5. Why is the <b>dev</b> loss the number we care about, not the train loss?</summary>

The dev set is names the model never trained on, so its loss measures *generalization* —
how well the model does on new data — which is what we actually want. Train loss can look
good just by memorizing.
</details>

## 🎓 Key takeaways

- An **embedding** is a short, *learned* vector per token; it beats a one-hot because
  similar tokens can get similar vectors and **share** learning. `C[X]` looks them up.
- More **context** (`block_size` characters) means better predictions than the bigram's one.
- An **MLP** = embed → flatten → linear → **tanh** → linear → **logits**; `@` is matmul and
  the shapes tell the whole story.
- **`cross_entropy`** = softmax + negative-log-likelihood in one stable call.
- **Minibatch** SGD — many fast, noisy steps — is how real models train.
- Result: **dev loss ~2.15** (vs bigram 2.45) and much better names, from ~12k parameters
  that *generalize*. This token → embedding → layers → logits shape is exactly how LLMs work.

## 📖 New vocabulary

`embedding` · `embedding table` · `context` · `block_size` · `fancy indexing` · `.view` /
reshape · `multi-layer perceptron (MLP)` · `hidden layer` · `tanh` / `GELU` · `logits` ·
`cross-entropy` · `minibatch` · `stochastic gradient descent (SGD)` · `learning-rate
decay` · `generalization` · `train/dev/test`.

## 🧪 Practice & build

1. **The notebook** — [`code/explore.ipynb`](./code/explore.ipynb): build the dataset,
   embeddings, forward pass, training, and sampling yourself. Do this first.
2. **Exercises** — [`exercises/`](./exercises/): tune `block_size`/`n_embd`/`n_hidden` on
   the dev set, **visualize the learned embeddings** in 2-D, swap `tanh` for GELU, and more
   (tiered hints + solutions; a starter for the first).
3. **Mini-project** — [`project/`](./project/): **"The Name Forge MK II"** — upgrade
   Chapter 1's name generator into an MLP you build from a starter, and beat the bigram.

## 🔗 Go deeper (optional)

- 🎥 Karpathy, [*Building makemore Part 2: MLP*](https://www.youtube.com/watch?v=TCH_1BHY58I)
  — this chapter follows it closely.
- 📄 [Bengio et al. (2003), *A Neural Probabilistic Language Model*](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
  — the paper that introduced learned word embeddings + an MLP language model.
- 📄 [Hendrycks & Gimpel (2016), *Gaussian Error Linear Units (GELUs)*](https://arxiv.org/abs/1606.08415)
  — the activation that has largely replaced `tanh`/ReLU in modern LLMs.

---

| ⬅️ Previous | Up | Next ➡️ |
|:--|:-:|--:|
| [Chapter 2 — Micrograd](../02-micrograd/) | [Syllabus](../../README.md#-syllabus) | [Chapter 4 — Attention](../04-attention/) |
