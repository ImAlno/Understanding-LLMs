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

(Why not just feed the character's *number* — `a`=1, `b`=2, `c`=3 — as a single input? Because that
invents a **fake order**: it would tell the model that `b` sits "between" `a` and `c`, and that `c`
is "three times" `a` — nonsense for letters. The one-hot avoids that by making all characters
equidistant and unordered; the embedding then goes one better, *learning* a meaningful geometry
instead of imposing a fake one.)

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

**A concrete picture.** Imagine the model has trained and learned 2-number embeddings (made up, but
representative of what really happens):

```
a → ( 0.9, 0.2)    e → ( 0.8, 0.3)    o → ( 0.85, 0.1)    ← vowels land near each other
b → (-0.7, 0.6)    d → (-0.6, 0.5)                        ← consonants cluster elsewhere
```

The vowels `a, e, o` ended up *close together*, the consonants `b, d` in a different region.
**Nothing told the model what a vowel is** — it discovered the grouping on its own, because treating
similar characters similarly lowers the loss. Now when it learns "a vowel often follows `th`," that
lesson applies to `a`, `e`, *and* `o` at once, because their vectors are near each other. A one-hot
can't do this: its 27 characters are all *exactly* the same distance apart, so every lesson has to
be learned 27 separate times. **An embedding turns "27 unrelated symbols" into "points in a space
where nearness means similarity"** — and that space is *learned*, not hand-designed. (In the
exercises you'll train with 2-number embeddings and literally plot this.)

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

Here's the **whole model on one page** — keep it in view as we build each piece:

```
  context ids       embeddings         flatten         hidden layer        output layer
  [ . , e , m ] ─► C[·]  (3 × 10) ─► 30 numbers ─► tanh( ·@W1 + b1 ) ─► ·@W2 + b2 ─► softmax → P(next char)
  block_size = 3    look up rows      .view          200 numbers          27 scores      27 probabilities
```

Every arrow is one section below: the embedding **lookup** and the **flatten** (§3), the **hidden**
and **output** layers (§4), and **softmax + loss** (§5). Four small steps, and the whole thing is
differentiable end to end — so one `loss.backward()` tunes *every* number in it.

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

Notice two things. The window **slides one character at a time**, so each name produces *several*
training examples (`emma` gives 5) — every position is something to learn from. And we **pad the
start with `.`** so even the first character has a full 3-slot context (`. . .` → `e`): the model
always sees exactly `block_size` ids, never a ragged short one. That same `.` doubles as the *end*
token, so the model also learns when a name should *stop* (`m m a → .`). Run this over all 32,000
names and you get ~182,000 (context → next) examples — that's the dataset the model trains on.

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

**See it on tiny numbers.** Shrink everything to a vocab of 3 characters with 2-number embeddings:

```
C = [[10, 11],     # char 0's embedding
     [20, 21],     # char 1's
     [30, 31]]     # char 2's        → shape (3, 2)

X = [[0, 2],       # example 0: context ids 0 and 2
     [1, 1]]       # example 1: ids 1 and 1   → shape (2, 2)
```

Indexing `C` with `X` replaces *every id with its row*:

```
C[X] = [[[10,11], [30,31]],     # example 0 → char 0's row, then char 2's row
        [[20,21], [20,21]]]     # example 1 → char 1's row, twice   → shape (2, 2, 2)
```

Each id became a 2-number vector, so the `(2, 2)` grid of ids grew a third dimension → `(2, 2, 2)`.
Scale the toy up — vocab 3 → 27, embedding 2 → 10, context 2 → 3 — and that's *exactly* `C[X]`
turning `(N, 3)` ids into `(N, 3, 10)` embeddings.

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

**How does `C` actually get *learned*?** It's just more parameters in the graph. When
`loss.backward()` runs, the gradient flows back through the MLP and the flatten and into the
*specific rows of `C`* that were looked up this batch — nudging those characters' embeddings to
lower the loss. (Rows for characters that weren't in the batch get zero gradient and sit still until
they show up.) Over thousands of batches every row drifts into a useful spot. That's all "the
embedding is learned" means: `C` is in the parameter list, so it's tuned exactly like `W1` or `W2`.

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

Why matrices at all? Each of the 200 hidden numbers is a different *weighted sum* of all 30 inputs —
one **column** of `W1` holds the 30 weights feeding one hidden unit. The `@` computes all 200 of
those sums, for all `N` examples, in a *single* operation, instead of `N × 200` separately written
dot products. That's the whole reason deep learning runs on matrix multiplies: they pack an enormous
number of weighted sums into one fast, parallelizable step — the step the GPU chapters later
accelerate.

Here's the whole pipeline as a shape trace, with `N = 4` examples (`block_size = 3`, `n_embd = 10`,
`n_hidden = 200`):

```
X         (4, 3)        4 examples, 3 context ids each
C[X]      (4, 3, 10)    look up each id's 10-number embedding
.view     (4, 30)       flatten each example's 3×10 grid into one row
x @ W1+b1 (4, 200)      (4,30)@(30,200) → mix the 30 into 200 hidden numbers
tanh      (4, 200)      squash each hidden number into (−1, 1)
h @ W2+b2 (4, 27)       (4,200)@(200,27) → 27 logits (next-char scores) per example
```

Read it top to bottom: **ids → embeddings → flat → hidden → logits.** The entire model *is* that
pipeline; training does nothing but tune the numbers inside `C`, `W1`, `b1`, `W2`, `b2` so the
logits point at the right next character.

> 🔭 **GELU, for later.** We use `tanh` (it's familiar from Chapter 2). Modern networks
> usually swap it for **GELU** or ReLU, which tend to train a little better — you'll try
> that in the exercises, and we lean on it from the Transformer onward.

**What is that hidden layer *for*?** The 200 hidden numbers are 200 little learned *feature
detectors*. Each is a `tanh` of some weighted mix of the 30 input numbers, so each can fire for a
particular pattern in the context — "the previous character was a vowel," "we're mid-way through a
common ending like `-ly`," and so on. Nobody labels these features; the model invents whatever ones
lower the loss. The output layer then reads those 200 features to score the 27 possible next
characters. More hidden units = more features the model *can* represent = more capacity — which is
why `n_hidden` is a knob worth tuning (you will, in the exercises).

---

## 5. The loss: `cross_entropy`

Those 27 logits become probabilities with **softmax**, and we score them with the
**negative log-likelihood** — exactly Chapter 1. (**Softmax**, concretely: `exp` each logit to make
it positive, then divide by the total so they sum to 1. Three logits `[2.0, 1.0, 0.1]` become
`exp = [7.39, 2.72, 1.11]`, total `11.2`, so probabilities `[0.66, 0.24, 0.10]` — the biggest logit
gets the biggest share, but nothing is ever exactly 0. The **negative log-likelihood** then just
reads off `−log` of the probability landed on the correct character.) PyTorch rolls both steps into
one call:

```python
loss = F.cross_entropy(logits, Y)
```

`F.cross_entropy(logits, Y)` does *softmax then negative-log-likelihood of the correct
character* in a single, numerically-stable step. (You proved it equals the hand-written
version in Chapter 1's exercise E05; from here on we always use it — it's faster and won't
overflow when logits get large.)

**What the number means, on one example.** Softmax turns the 27 logits into 27 probabilities
(exponentiate each, then divide by the total so they sum to 1). The loss for that example is just
`−log(the probability the model assigned to the *correct* next character)`:

- model puts `0.9` on the right character → loss `−log(0.9) ≈ 0.11` (confident and right — tiny loss)
- model puts `0.05` on the right character → loss `−log(0.05) ≈ 3.0` (confidently wrong — big loss)
- model is unsure, `1/27 ≈ 0.037` on every character → loss `−log(1/27) ≈ 3.3` (pure guessing)

Cross-entropy is the **average of that over the batch** — literally *"how surprised the model is by
the right answers."* A loss of `2.15` means the model, on average, gave the correct character a
probability of `e^(−2.15) ≈ 0.12` — far better than the `0.037` of blind guessing. Driving this
number down is the entire objective of training.

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

Why does training on a noisy estimate even work? Each minibatch's gradient points *roughly*
downhill — not exactly, since it only saw 32 examples — but "roughly downhill, 30,000 times" still
reliably finds the valley floor, and each step is ~5,000× cheaper than scanning all 182,000
examples. The noise even *helps* a little: the jiggle can shake the model out of shallow ruts it
might otherwise settle into. A slightly-wrong direction for a vastly cheaper step is such a good
trade that *every* large model is trained this way — the only real question is how big to make the
batch.

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

(And the **learning-rate drop** — `0.1` for the first 18,000 steps, then `0.01` — is there because
big steps cover ground fast early on, but late in training they make the loss *bounce* around the
minimum instead of settling into it; smaller steps let it settle. It's a crude version of the
*learning-rate schedule* we build properly in Chapter 7.)

**How big is this model?** Count the knobs: the table `C` is `27 × 10 = 270`; `W1` is
`30 × 200 = 6,000` plus a 200-bias; `W2` is `200 × 27 = 5,400` plus a 27-bias — about **11,900
numbers** in total. Compare the bigram's `27 × 27 = 729` counts: many more parameters, but they
*generalize* (they don't just memorize the training names), which is exactly why the dev loss ends
up right next to the train loss.

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

**Why the three splits matter.** We *train* on the train set but *judge* on the dev set — names the
model has never seen. Why bother? Because a model can lower its train loss simply by **memorizing**
the training names, which teaches it nothing about new ones. The gap between train and dev loss is
the tell: a big gap means overfitting (memorizing), a small gap means the model learned *general*
rules of how names work. Here the two are nearly equal (`2.12` vs `2.15`), so our MLP is genuinely
generalizing. The third split, *test*, we touch only **once**, at the very end, as a final unbiased
check we never tuned against — peeking at it during development would quietly turn it into another
dev set.

**How far we've come.** Every model so far, on the same dev set:

| Model | What it sees | Dev loss |
|-------|--------------|---------:|
| Bigram (Ch 1) | the previous **1** character (counts) | 2.45 |
| Trigram by counting | the previous **2** characters (counts) | 2.21 |
| **MLP (this chapter)** | the previous **3** characters (**learned embeddings**) | **2.15** |

Each step down came from *more context* plus a *smarter representation* — and that's exactly the
road ahead. Chapters 4–5 don't change the goal (predict the next token, minimize cross-entropy);
they make "more context, learned more cleverly" dramatically more powerful.

---

## 8. Sampling: meet the new names

> ✏️ **In the notebook → Step 7.** Generate names from your model.

Same autoregressive loop as Chapter 1 — but now we feed the last 3 characters through the
MLP each step, sample the next, and slide the context forward:

```python
context = [0] * block_size          # start with '. . .'
out = []
while True:
    logits = forward(torch.tensor([context]))   # run the MLP on the current 3-char context
    probs = F.softmax(logits, dim=1)             # 27 probabilities for the next character
    ix = torch.multinomial(probs, 1).item()      # sample one (weighted by probability)
    context = context[1:] + [ix]                 # slide the window: drop oldest, append new
    if ix == 0:                                  # sampled '.' → end of name
        break
    out.append(itos[ix])
```

The *only* change from Chapter 1's sampler is where the next-character distribution comes from: an
MLP reading 3 characters of context (as embeddings) instead of a single-character count row. Run it
and you get names like:

```
jahmer   kaeli   nellara   kaleigh   quint   suline   carmahza   kanden   kimrix
```

Compare those to the bigram's `cexze` and `staiyaubrtthrigotai`. Three characters of memory
and a learned representation make a real difference — and the path from here (Chapters 4–5)
is simply: *much* more context, learned even more cleverly.

---

## 🐛 Building it yourself: what trips people up

A handful of papercuts everyone hits once when building this — worth knowing in advance:

- **Shape mismatch in the matmul.** `x @ W1` needs `x` of shape `(N, 30)` and `W1` of `(30, 200)` —
  the *inner* numbers must match. Forget the `.view` flatten and `x` is still `(N, 3, 10)`, so the
  matmul errors. When a shape line surprises you, `print(x.shape)` is your friend.
- **Forgetting `p.grad = None`.** Skip the reset and PyTorch *adds* this step's gradients onto last
  step's; the model lurches and the loss won't settle. (Same accumulation you built in Chapter 2.)
- **Wrong target shape.** `F.cross_entropy(logits, Y)` wants `logits` as `(N, 27)` and `Y` as
  `(N,)` — target **ids**, not one-hots. Passing one-hots is a classic mix-up.
- **Missing `requires_grad`.** Every parameter (`C, W1, b1, W2, b2`) needs `requires_grad=True`, or
  `backward()` won't compute its gradient and it never learns. The notebook sets this for you.
- **Loss stuck at ~3.3.** That's `log(27)` — the model is still guessing uniformly. Give it more
  steps, check the learning rate isn't tiny, and confirm gradients are actually flowing (the
  `.grad`s aren't `None`).

None of these are deep — they're the once-and-done snags of building your first real net. Hit them,
fix them, move on.

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
- **Why 3 characters of context — why not 10?** More context usually helps, but each extra
  character adds input width and weights to learn, with diminishing returns at this scale; 3 is a
  sweet spot for short names (you'll push it in the exercises). Chapters 4–5 change the game
  entirely — they let the model attend to *hundreds* of characters cheaply.
- **Doesn't flattening the embeddings lose the character order?** No — `.view` lays the three
  embeddings out in a fixed order (char 1's ten numbers, then char 2's, then char 3's), and that
  layout never changes. The hidden layer sees "position 1's numbers" in the first 10 input slots,
  "position 2's" in the next 10, and can treat them differently. Order is encoded by *where* each
  number sits.
- **What sets the embedding size (10) and hidden size (200)?** They're **hyperparameters** — knobs
  *you* choose, not numbers the model learns. Bigger means more capacity (lower train loss) but more
  compute and more overfitting risk. You pick them by trying values and watching the *dev* loss —
  which is exactly the first exercise.

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
