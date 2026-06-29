# Chapter 1 — The Bigram Language Model

> *Our Storyteller's very first words.* Before an AI can write a story, it has to
> learn what letters tend to follow other letters. In this chapter we build the
> simplest possible language model — one you could almost do by hand — and use it
> to **invent brand-new names** for the characters in our stories. By the end
> you'll understand what a language model *is*, how we *measure* one, and the single
> most important idea in the whole course: that **counting** and **training a
> neural network** can produce the very same model.

**You will be able to:**
- explain what a language model is, in terms of probabilities;
- build a bigram model two ways — by counting, and as a 1-layer neural network;
- generate new names by *sampling* from the model;
- measure model quality with the **negative log-likelihood** loss;
- see *why* gradient descent is the idea that scales all the way to ChatGPT.

**Prerequisites:** basic Python (variables, lists, loops, functions, dictionaries).
That's it. We also lean on a few Python niceties — `set()`, dict comprehensions, and
`enumerate()` — and explain them inline the first time they show up. Every new idea —
tensors, probability, softmax, loss, gradient descent — is explained from scratch the
moment we need it. No prior machine learning, and no remembered calculus, required.

**Time:** ~2–3 hours if you build along. **Hardware:** any laptop. No GPU needed.

> ### 📓 The most important tip: build it yourself in the notebook
>
> This chapter has a companion Jupyter notebook,
> **[`code/explore.ipynb`](./code/explore.ipynb)**, and it is the heart of the
> chapter. It's **not** a demo you click through — it's a guided build where *you
> write the code*. Each step explains an idea, shows you exactly what to type, and
> leaves an empty cell for you to write it in and run. After each one, a **"Check
> your work" cell** runs what you wrote and tells you instantly whether you nailed
> it (✅) or need a nudge (❌) — like having a tutor right beside you.
>
> **The best way to take this chapter:** read a section here, then switch to the
> notebook and build that same piece yourself. Reading *about* code and *writing*
> code are completely different experiences — only one of them sticks. We'll point
> you to the matching notebook step as we go, like this:
>
> > ✏️ **In the notebook → Step 2.** You'll build this part yourself.
>
> Launch it with `jupyter lab` (or open the file in VS Code / Cursor) and keep it
> open beside this page.

---

## 0. Setup

From the repo root, if you haven't already:

```bash
uv venv --python 3.13 .venv && uv pip install -r requirements.txt
# (or: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt)
```

Then launch the notebook:

```bash
.venv/bin/jupyter lab        # opens in your browser; navigate to chapters/01-bigram/code/explore.ipynb
```

The dataset — **32,033 names**, one per line — already lives in
[`data/names.txt`](./data/names.txt). Peek at it: `head data/names.txt` shows
`emma, olivia, ava, isabella, …`.

**Three resources, one chapter** — use them together:
1. **This lesson** — the *why* and the *intuition*. Read it first.
2. **The notebook** [`code/explore.ipynb`](./code/explore.ipynb) — the *how*. Build
   each piece yourself here. **This is where the learning happens.**
3. **The finished scripts** [`code/bigram_counts.py`](./code/bigram_counts.py) and
   [`code/bigram_nn.py`](./code/bigram_nn.py) — the clean, complete reference, for
   when you want to see it all in one place or check your work.

---

## A 2-minute primer: tensors

Almost the only new tool you need in this chapter is the **tensor**. The good news:
a tensor is just **a grid of numbers**. That's it. We use the PyTorch library
(`import torch`) to make and manipulate them, because PyTorch can do math on whole
grids at once (fast) and — later — compute gradients for us.

- A list of numbers is a **1-D** tensor (a *vector*): `[0.2, 0.5, 0.3]`.
- A table of numbers is a **2-D** tensor (a *matrix*): rows × columns.
- Its **shape** tells you the size along each dimension. A 27×27 table has shape
  `(27, 27)`.

A few operations are all you'll use:

```python
import torch
N = torch.zeros((27, 27))   # make a 27×27 grid filled with 0.0
N[3, 5] += 1                # index row 3, column 5, and add one to it
N[0]                        # grab row 0 (a length-27 vector)
```

The `(27, 27)` is just the **shape** written as a Python tuple — `(rows, columns)`.
You'll sometimes see a `dtype` ("data type") that says what kind of number fills the
grid: `int32` for whole numbers, `float` for decimals.

If you've used a spreadsheet, you already have the right mental picture: a tensor is
a spreadsheet of numbers, and indexing is "go to cell (row, column)." We'll meet two
more operations as we go — *summing along a dimension* (Section 3) and *matrix
multiply* (Section 6) — and we'll go slow at both, because they're the two spots
beginners trip on.

---

## 1. What *is* a language model?

A **language model (LM)** is a thing that answers one question, over and over:

> *Given the text so far, what comes next?*

You already use one every day: the **predictive keyboard** on your phone that
suggests the next word, or autocomplete in a search bar. A language model is exactly
that, made precise.

Crucially, it doesn't answer with a single guess. It answers with a **probability
for every possible next item** — every next word, or, in our case, every next
**character**. For example, partway through spelling a name, after the letters
`emm`, a good model of names might assign:

```
next char       'a'    'e'    'i'    'y'    ' . ' (end)   ...all others...
probability     73%    9%     6%     5%     2%            tiny
```

Notice the probabilities are all positive and add up to 100% — that's what makes it
a **probability distribution**: a complete answer that hedges across every option
instead of betting on one.

To actually *write* something, we run a loop:

1. Look at the text so far.
2. Ask the model for the probability of each possible next character.
3. **Sample** one — roll a weighted die that lands on `'a'` 73% of the time, `'e'`
   9% of the time, and so on.
4. Append it to the text, and go back to step 1.

This loop — *predict a distribution, sample one item, feed it back in* — is called
**autoregressive generation** ("auto" = self, "regressive" = referring back to its
own output). It is **exactly** how ChatGPT writes a sentence, just with a far
smarter "what comes next?" step and billions more numbers under the hood. The entire
rest of this course is about making that one prediction better and better.

> 🧩 **Why characters and not words?** Working one character at a time keeps our
> whole vocabulary tiny — just 26 letters plus one end marker, 27 things total. The
> model can also invent words that have never existed (perfect for fantasy names!),
> and we dodge the surprisingly messy question of "what *is* a word?" for now. Real
> models split text into **tokens** (chunks between a letter and a word); we build a
> proper tokenizer in Chapter 6.

### Our task: invent character names

Our Storyteller needs a cast of characters, so our first model will learn from
32,033 real names and then generate *new* ones — `cexze`, `konimittain`, `llayn` —
names that sound name-like but don't exist. It's a small task, but it contains every
core idea of language modeling in miniature: data, prediction, sampling, and a way
to measure quality. Nail it here, and the leap to sentences (Chapter 5) is mostly a
matter of scale.

---

## 2. The data, and a special token

> ✏️ **In the notebook → Step 1 & 2.** Load the data and build the vocabulary
> yourself as you read this section.

First, load the names into a Python list of strings:

```python
words = open('data/names.txt').read().splitlines()   # ['emma', 'olivia', 'ava', ...]
```

A computer can't do arithmetic on the *letter* `'a'` — math needs **numbers**. So we
build two little lookup dictionaries that translate between characters and integer
**ids**:

```python
chars = sorted(set(''.join(words)))              # the 26 distinct letters: ['a','b',...,'z']
stoi = {c: i + 1 for i, c in enumerate(chars)}   # "string to int": 'a'->1, 'b'->2, ... 'z'->26
stoi['.'] = 0                                     # a special token, id 0 (explained below)
itos = {i: c for c, i in stoi.items()}           # "int to string": the reverse, 0->'.', 1->'a', ...
```

*New Python here, in case it's unfamiliar:* `set(...)` keeps only the **distinct**
characters; `enumerate(chars)` walks the list handing you `(position, character)`
pairs; and `{c: i + 1 for i, c in ...}` is a **dict comprehension** — a one-line way
to build a dictionary. Together they number the letters `1..26`.

Now `stoi['e']` is `5`, and `itos[5]` is `'e'`. We'll convert letters to ids to do
math, then convert back to letters to read the result.

### The special `.` token

Look at that `stoi['.'] = 0`. We invent one extra symbol, `.`, that isn't a letter.
Why? Because the model needs to learn two things that pure letters can't express:

- **Where a name starts.** Names are far more likely to begin with `k` or `a` than
  with `x` or `z`. Without a "start" signal, the model has no way to learn *first
  letter* preferences.
- **Where a name ends.** Otherwise generation never knows when to stop.

The trick is elegant: wrap **every** name in a single boundary token that means both
"start" and "end":

```
emma   →   . e m m a .
```

So our vocabulary is **27 characters** total: the `.` plus the 26 letters.

### Bigrams

A **bigram** is just a pair of neighbours. Reading `. e m m a .` left to right, the
adjacent pairs are:

```
(., e)   (e, m)   (m, m)   (m, a)   (a, .)
```

Read each pair as *"given the first character, the second one came next."* So
`(., e)` says "a name started with `e`"; `(a, .)` says "after `a`, the name ended."

A bigram model makes one bold, simplifying assumption:

> **The next character depends only on the single character right before it.**

That's obviously too simple to be *good* — when you're choosing the 5th letter, the
first four surely matter! — but it's the perfect place to start, and it already
captures a shocking amount of structure.

---

## 3. Approach #1 — just count

> ✏️ **In the notebook → Step 3 & 4.** Build the bigram helper and the count grid.

Here's the beautiful thing about that "depends only on the previous character"
assumption: it means we can build the entire model by **counting**. To know what
tends to follow `m`, we just tally, across all 32,033 names, every character that
ever came right after an `m`.

Do that for all 27 possible "current" characters at once and you get a **27×27 grid
of counts** called `N`, where:

> `N[i, j]` = "how many times did character `j` follow character `i`?"

```python
import torch
N = torch.zeros((27, 27), dtype=torch.int32)   # a 27×27 grid of (integer) zeros
for w in words:
    chs = ['.'] + list(w) + ['.']              # e.g. emma -> ['.', 'e', 'm', 'm', 'a', '.']
    for ch1, ch2 in zip(chs, chs[1:]):         # walk over adjacent pairs
        N[stoi[ch1], stoi[ch2]] += 1           # bump the cell for this bigram
```

Two bits of Python worth unpacking if they're new:

- `zip(chs, chs[1:])` pairs each item with the one after it. `chs[1:]` is "the list
  starting from index 1," so zipping the list with its own tail gives you every
  adjacent pair. It's the standard idiom for "loop over neighbours."
- `N[stoi[ch1], stoi[ch2]] += 1` converts both characters to ids and adds 1 to that
  cell. Tracing `emma`: we bump `(., e)`, then `(e, m)`, `(m, m)`, `(m, a)`, and
  `(a, .)` — five cells, one per bigram.

After the loops, `N` holds **228,146** counted bigrams. For instance you'd find a
big number in `N[stoi['q'], stoi['u']]` (almost every `q` is followed by `u`) and a
`0` in `N[stoi['q'], stoi['z']]` (no name has `qz`).

> 🔎 **See it!** In the notebook you'll plot `N` as a labelled heatmap (or run
> `python code/bigram_counts.py --plot`). It's worth a long look: the bright first
> **column** is "letters that names end on", the bright first **row** is "letters
> that names start with", and patterns like `q`→`u` jump right out. This picture
> *is* the model.

### From counts to probabilities (and the #1 beginner bug)

> ✏️ **In the notebook → Step 5.** This is the trickiest line in the chapter —
> build it carefully and check the shapes as you go.

Counts aren't probabilities. The model's answer for "what follows `i`?" should be a
set of 27 probabilities that **add up to 1**. So we take each row of `N` and divide
it by its own total:

```python
P = (N + 1).float()                # +1 = "smoothing" (see §5); .float() makes the counts decimals
P /= P.sum(dim=1, keepdim=True)    # divide each row by its sum  →  each row now sums to 1
```

(`N` holds whole-number counts; `.float()` turns them into decimals first, so a
probability like `0.04` isn't rounded down to `0` when we divide.)

That second line is short but subtle, and it's the single most common place
beginners (and pros!) introduce a silent bug. Let's slow down.

`P.sum(dim=1, keepdim=True)` means *"add up the numbers along dimension 1, and keep
the result 2-D."* A `(27, 27)` grid has two dimensions: **dimension 0** runs *down* the
rows, **dimension 1** runs *across* the columns. Summing along dimension 1 walks across
each row and adds up its 27 entries — collapsing each row to a single number — so we
get **one total per row, 27 in all**. ("Sum over the columns" and "give me each row's
total" describe the very same operation; that double-wording is the bit that trips
people up.)

- `keepdim=True` keeps those 27 totals as a **column** of shape `(27, 1)` instead of
  collapsing them to a flat list of shape `(27,)`.

Why does that matter? When you divide a `(27, 27)` grid by a `(27, 1)` column,
PyTorch **broadcasts**: it stretches that single column across all 27 columns, so
**row `i` of `P` gets divided by the total of row `i`.** Exactly right.

Here's the rule behind that, worth learning once: to broadcast, PyTorch lines the two
shapes up **from the right** and stretches any dimension whose size is `1`.

- `(27, 27)` vs `(27, 1)` → the `1` on the right stretches across the columns: each
  row is divided by *its own* total. ✅
- `(27, 27)` vs `(27,)` → the `(27,)` lines up on the right as `(1, 27)` — a *row* —
  so it stretches *down*, dividing each **column** by a row-total. ❌

Concrete 2×2 check: `[[2, 2], [3, 9]]` divided by the column `[[4], [12]]` gives
`[[0.5, 0.5], [0.25, 0.75]]` — each row scaled by *its own* total (4, then 12), which
is what we want.

Same `(27, 27)` shape comes out either way, with no error — but one is a correct model
and the other is silently broken. **Lesson: whenever you sum/reduce a tensor,
sanity-check the shape.** Print `P.sum(dim=1, keepdim=True).shape` and confirm it's
`[27, 1]`.

Now `P[i]` is a genuine probability distribution over the next character. As a check,
`P[0].sum()` should be `1.0` (row 0 is "what follows the start token" — i.e., the
distribution over first letters). The model is **done** — no training, just counting.

---

## 4. Sampling — making the model talk

> ✏️ **In the notebook → Step 6.** Write the sampling loop and generate your first
> names. This is the fun part.

To generate a name we run the autoregressive loop from Section 1, using `P` as our
"what comes next?" oracle:

1. Start at the `.` token (id `0`).
2. Look up its row `P[0]` — the distribution over the first letter.
3. **Sample** one character from that distribution.
4. Move to that character's row and repeat — until we sample `.` again, meaning the
   name is finished.

```python
g = torch.Generator().manual_seed(2147483647)   # a seeded RNG → everyone gets the same names
ix = 0                                            # start at '.'
out = []
while True:
    p = P[ix]                                     # the distribution over the next char
    ix = torch.multinomial(p, num_samples=1, generator=g).item()
    if ix == 0:                                   # sampled the end token → stop
        break
    out.append(itos[ix])
print(''.join(out))
```

The new function is **`torch.multinomial`**, and it's just a **weighted die**. Give
it a list of probabilities like `[0.6, 0.1, 0.3]` and it returns index `0` about 60%
of the time, `1` about 10%, and `2` about 30%. That's the "roll a weighted die" step
made concrete. (Two small notes: `generator=g` makes the randomness reproducible, so
your names will match the ones below; and `.item()` just turns the resulting
1-element tensor into a plain Python integer we can use as an index.)

Running `python code/bigram_counts.py` produces names like:

```
cexze   konimittain   llayn   ka   da   moliellavo   ke   teda   emimmsade
```

Some are plausible (`ka`, `teda`, `moliellavo`); some are a mouthful
(`staiyaubrtthrigotai`). That's expected, and instructive: each character was chosen
knowing **only the one before it**, so the model has no sense of the name as a whole
— no notion of length, rhythm, or what it already wrote three letters ago. Wanting
more context is precisely what drives Chapters 3–5.

But don't undersell it: with nothing but counting, the model learned that names
start and end believably and that letters flow in name-like ways. Real structure,
for free.

---

## 5. Is it any good? The loss function

> ✏️ **In the notebook → Step 7.** Compute the loss yourself.

"Some names look okay" is a vibe, not a measurement. To improve a model you need a
single number that says how good it is, so you can tell whether a change *helped*.
That number, in language modeling, comes from **likelihood**.

The idea: a good model should assign **high probability to names that are real**. Our
model already assigns a probability to every bigram (that's what `P` is). The
probability it assigns to a whole name is the bigrams **multiplied together**. *Why
multiplied?* Spelling a name means making this first choice **and** that next choice
**and** the next — and the chance of a whole chain of choices is the product of each
one's probability, exactly like the chance of flipping three heads in a row is
½ × ½ × ½ = ⅛. So the probability of the *entire dataset* is every bigram of every
name multiplied together. **A better model makes that product bigger**, so our goal is
to *maximize* it.

Working with that giant product directly is impractical for two reasons, and fixing
each gives us the standard loss:

1. **Multiplying thousands of small probabilities underflows to zero.** A computer
   can't store a number that tiny. The fix is **logarithms**. A *log* is the inverse
   of the exponential `eˣ` (which we'll meet in §6); its one magic property is that it
   turns multiplication into addition — `log(a · b) = log(a) + log(b)` — so we can
   **add up the log of each probability** instead of multiplying them. Same ranking of
   models, no underflow.

   *Concretely:* a probability like `0.4` has `log(0.4) ≈ -0.92`. Log-probabilities
   are always **negative** (since probabilities are ≤ 1), and *less* negative means
   *more* likely.

2. **We want a "lower is better" score to minimize.** Maximizing a sum is the same as
   minimizing its negative, so we flip the sign. And we **average** over all bigrams
   so the number doesn't depend on dataset size.

Put together, that's the **average negative log-likelihood (NLL)** — the loss that
shows up in virtually every language model you will ever train:

```python
log_likelihood = 0.0
n = 0
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        log_likelihood += torch.log(P[stoi[ch1], stoi[ch2]])   # add log-prob of each bigram
        n += 1
nll = -log_likelihood / n          # negate, and average → our loss
```

🔮 **Predict before you read on.** A clueless model that guessed uniformly among 27
tokens would score `log(27) = 3.2958`. Do you think our bigram model beats that — and
by roughly how much?

<details>
<summary>👉 Click to reveal the loss</summary>

```
average negative log-likelihood (loss) = 2.4544
```

Comfortably below the 3.30 baseline — one character of context already teaches the
model something real.
</details>

**Is 2.4544 good?** It helps to know the worst case. A model that knew *nothing* and
guessed uniformly among 27 tokens would assign every bigram probability `1/27`, for a
loss of `log(27) = 3.2958`. Our model scores well below that, so it has genuinely
learned something. A *perfect* model would score `0` (it would assign probability 1
to every real bigram). So `2.45` sits sensibly between "clueless" (3.30) and
"perfect" (0). **Remember this number** — in the next section we'll reach the very
same `2.45` by a completely different route.

> 🩹 **What was that `+ 1`?** That's **smoothing**. If some bigram (say `jq`) never
> appears in the data, its count is `0`, so its probability is `0`, so its `log` is
> `-∞`, and the loss explodes to infinity the instant any name contains it. Adding a
> small fake count to *every* cell guarantees no probability is ever exactly zero.
> More fake count → a smoother, blander, more uniform model. (You'll tune this in the
> exercises.)

> 📐 **Aside — perplexity.** You'll often see models reported with their
> *perplexity*, which is just `exp(loss)` — the exponential function `eˣ` (we explain
> `exp` properly in §6). Our `exp(2.4544) ≈ 11.6`, loosely "the model is as unsure as
> if it were picking uniformly among ~11.6 characters at each step." Same information
> as the loss, friendlier units. We'll stick with the loss.

---

## 6. Approach #2 — the *same* model as a neural network

> ✏️ **In the notebook → Steps 8–10.** Build the dataset, the training loop, and
> sample from your trained network.

Here is the chapter's punchline, and one of the most important ideas in the whole
course. We just built a perfectly good model by counting. So why touch it?

Because **counting doesn't scale.** It works only because a bigram has just 27×27 =
729 situations to tally. The moment the model needs to consider *two* previous
characters, that becomes 27³ ≈ 19,683 cells; ten characters of context would need
27¹⁰ ≈ 10¹⁴ cells — more than could ever be filled or stored. We need a method that
**learns** good numbers from examples instead of enumerating every case. That method
is **gradient descent**, it works for models of *any* size, and it is how every
modern LLM is trained. Let's prove it reproduces our counting model exactly, on this
small problem where we can check the answer.

### The network

We'll build the world's smallest neural network: a single **27×27 weight matrix
`W`** — no hidden layer, no bias, nothing else. The 729 numbers inside `W` are the
network's **parameters**: the knobs gradient descent gets to turn. Here's the entire
forward pass for one input character:

```
character id  →  one-hot vector (length 27)  →  multiply by W  →  logits (length 27)
              →  exp(), then normalize         =  softmax        →  probabilities
```

Let's take the new pieces one at a time.

**One-hot encoding.** To feed a character id into matrix math, we turn it into a
length-27 vector that is `1` at that id's position and `0` everywhere else. The id
`5` (`'e'`) becomes:

```
index:   0  1  2  3  4  5  6 ... 26
vector: [0, 0, 0, 0, 0, 1, 0, ..., 0]
```

Here's the neat part: multiplying that one-hot row by `W` simply **picks out row 5 of
`W`** (all the zeros kill every other row). So `W`'s rows play the same role the
count rows did in `P` — except now they're *learnable numbers* instead of fixed
tallies. (In Chapter 3 we'll skip the wasteful one-hot-and-multiply and just index
the row directly — that shortcut is called an *embedding*. You'll prove they're
identical in exercise E04.)

**Matrix multiply (`@`).** `xenc @ W` is a *matrix multiplication*. You don't need the
full linear-algebra rules — here's all it does for us. Multiplying one row by the
matrix `W` builds a new row by **combining `W`'s rows**, each weighted by a number in
your input row. Because our input row is a *one-hot* (a single `1`, the rest `0`s),
that combination keeps exactly one of `W`'s rows and zeroes out the others — it
**selects the row** at the `1`'s position. Concretely, with tiny numbers:

```
[0, 1, 0] @ [[10, 11], [20, 21], [30, 31]]  =  [20, 21]   ← row 1, picked by the 1
```

We do this for every example at once by
stacking them into a grid: if `xenc` is `(num, 27)` (that's `num` examples, each a
length-27 one-hot) and `W` is `(27, 27)`, then `xenc @ W` is `(num, 27)` — one
length-27 row of scores per example. (The shape pattern, if you like them:
`(a, b) @ (b, c) → (a, c)`; the inner `b`s must match and "cancel.")

**Logits.** Those output numbers are called *logits*: raw scores that can be any real
number — positive, negative, large, small.

**Softmax.** We need to turn those 27 arbitrary scores into 27 probabilities. Two
steps. First, `exp()` — the exponential function, `exp(x) = eˣ` where `e ≈ 2.718` —
makes every score **positive** (think of it as turning a *log-count* back into a
*count*: `exp` of a negative number is a small positive one, `exp(0) = 1`, and `exp`
of a positive number is big). Second, dividing by the total makes them **sum to 1**.
That combination — exponentiate, then normalize — is called the **softmax**, and it's
the exact neural-net twin of "turn counts into probabilities" from Section 3.

```python
import torch.nn.functional as F
# xs holds every input character id; `num` is how many examples there are.
xenc = F.one_hot(xs, num_classes=27).float()   # (num, 27) — one-hot every input character
logits = xenc @ W                              # (num, 27) — raw scores (this just selects rows of W)
counts = logits.exp()                          # (num, 27) — make them positive ("counts")
probs  = counts / counts.sum(dim=1, keepdim=True)  # (num, 27) — normalize ⇒ softmax
```

(There's that `keepdim=True` again — same reason as Section 3. And heads-up: `num`
here is the **number of examples** (228,146) — *not* the 27×27 count grid `N` from
Section 3. Different thing, reused letter; we renamed it to `num` to keep them
straight.)

### Training: nudge `W` downhill

`W` starts as **random noise**, so at first the predictions are terrible — the loss
is about `3.77`, even worse than guessing. Training is a loop of three steps that we
repeat a few hundred times:

1. **Forward.** Run the forward pass above and compute the NLL loss (the very same
   loss as Section 5).
2. **Backward.** Ask: for each of the 729 numbers in `W`, *which way should I nudge
   it to make the loss smaller?* That set of directions is called the **gradient**,
   and PyTorch computes it for us with one line: `loss.backward()`.
3. **Update.** Take a small step in the downhill direction:
   `W -= learning_rate · W.grad`.

The intuition for steps 2–3: imagine the loss as a hilly landscape and `W` as a ball
sitting on it. The gradient points **uphill** (toward higher loss), so we step the
opposite way, **downhill**, to reduce it. The **learning rate** is how big each step
is — too small and training crawls; too big and the ball overshoots the valley and
bounces around. Repeat enough small downhill steps and the ball settles near the
bottom: the best `W`. (One heads-up for the code below: the learning rate there is
`50`, which *looks* enormous — but the gradients in this tiny model are very small
numbers, so each real step, `50 × gradient`, is actually small.)

First, the data the loop runs on: in the notebook (Step 8) we flatten every bigram
into two tensors — `xs` (the input character ids) and `ys` (the *correct* next-char
ids) — `num` examples in total. Then the loop:

```python
W = torch.randn((27, 27), requires_grad=True)   # random start; requires_grad = "track this for backward()"
for step in range(200):
    # 1. forward
    probs = F.softmax(xenc @ W, dim=1)   # F.softmax = the exp-then-normalize from above, in one call
    loss = -probs[torch.arange(num), ys].log().mean() + 0.01 * (W**2).mean()
    # 2. backward
    W.grad = None          # clear the previous step's gradient
    loss.backward()        # compute the new gradient of loss w.r.t. every entry of W
    # 3. update
    W.data += -50 * W.grad # step downhill (here the learning rate is 50)
```

**Reading that `loss` line** — it's the densest in the chapter, so let's unpack it
left to right:

- `probs` has one row per example; row `i` holds the model's 27 predicted
  probabilities for example `i`.
- `probs[torch.arange(num), ys]` is **paired indexing**. `torch.arange(num)` is just
  `[0, 1, 2, …, num-1]` (one index per example). Pairing it with `ys` says: *"from row
  0 take column `ys[0]`, from row 1 take column `ys[1]`, …"* — i.e. **pluck out the
  probability the model gave to the _correct_ next character, for every example at
  once.** It is the §5 loop's `P[stoi[ch1], stoi[ch2]]`, done for all 228,146 bigrams
  in a single line.
- `.log().mean()` takes the log of each of those probabilities and averages them; the
  leading `-` flips the sign. That is the **average negative log-likelihood — the very
  same loss as §5**, just computed all at once instead of in a Python loop.
- `+ 0.01 * (W**2).mean()` is a small regularization nudge (see the box below).

Two more one-liners in that block: `torch.randn(...)` fills `W` with *random* numbers
to start (versus `torch.zeros`); and we nudge `W.data` — the raw numbers inside `W` —
rather than `W` itself, so PyTorch doesn't treat our hand-made update as part of the
math it's differentiating.

> 🪄 **`loss.backward()` is doing the real magic — how?** That one line computes the
> gradient through the whole forward pass, and right now it's a black box. **Chapter
> 2 is entirely about opening that box** — we'll build the gradient machinery
> (*backpropagation*) ourselves, from scratch, in pure Python. For now, trust that
> it hands us the right "downhill" direction.

Run `python code/bigram_nn.py` and watch the loss fall:

```
step    0 | loss 3.7686
step   20 | loss 2.5823
step   60 | loss 2.5027
step  140 | loss 2.4856
```

It converges to **~2.45** — *the same loss as the counting model* — and it generates
the *same kind of names*. This is not a coincidence, and here's the intuition for
*why*, in two steps.

**Step 1 — the loss is lowest when the model predicts the true odds.** Picture a
betting game: your best long-run score comes from betting each outcome at its *real*
frequency — overconfident or underconfident bets both cost you. The NLL loss works
the same way, so it bottoms out exactly when, for each current character, the
network's predicted distribution **matches the real frequencies in the data**. And
those real frequencies are precisely our normalized counts `P`. So training pushes the
network's `softmax(W)` toward `P`.

**Step 2 — which `W` gives `softmax(W) = P`? Just `W = log(P)`.** Feed `log(P)` through
the softmax: the `exp` undoes the `log` (`exp(log P) = P`), and since each row of `P`
already sums to 1, the "divide by the total" step changes nothing — so
`softmax(log P) = P` exactly. (That's the *"give or take a constant"* caveat too:
adding the same number to all 27 logits cancels out inside the softmax, so any
`log(P) + c` works just as well.)

Put together: **minimizing the loss drives `W` toward `log(counts)`**, and the softmax
turns those learned weights right back into our count-based table `P`. **Two roads —
counting and learning — arrive at one identical model.** (The one piece we're taking
on faith is Step 1's claim that the loss bottoms out exactly at the true frequencies;
*proving* that needs a little calculus we're skipping, but the betting intuition is
exactly right.)

> 🩹 **Smoothing returns, in disguise.** That `+ 0.01 * (W**2).mean()` term is
> **regularization**: it gently pulls every weight toward `0`. Since `exp(0) = 1`,
> pulling weights toward 0 pushes the probabilities toward uniform — which is
> *exactly* what add-one smoothing did to the counts. Same idea wearing a different
> costume. Crank that `0.01` up and, just like heavy smoothing, the model gets
> blander.

### Why bother, if counting already worked?

Because **gradient descent doesn't care how big or complicated the model is.**
Counting only works when you can enumerate every situation. Gradient descent finds
good parameters for *any* network you can write down — this 1-layer bigram, a deeper
MLP (Chapter 3), a Transformer (Chapter 5), or a 100-billion-parameter ChatGPT — all
with the same three-step loop: **forward, backward, update.** This chapter is where
we earn the right to stop counting and start *learning*. Everything that follows is
this same loop on ever-more-powerful models.

---

## 7. Run it yourself

Once you've built it in the notebook, the finished scripts are handy for
experimenting:

```bash
# The counting model: sample names + report the loss
python chapters/01-bigram/code/bigram_counts.py --num 20

# Save the bigram-count heatmap (needs matplotlib)
python chapters/01-bigram/code/bigram_counts.py --plot

# The neural-net model: watch it train down to ~2.45
python chapters/01-bigram/code/bigram_nn.py --steps 300

# Experiment! more smoothing = blander, safer names:
python chapters/01-bigram/code/bigram_counts.py --smoothing 10
```

Poke at things. What happens to the names with `--smoothing 0.01`? With
`--smoothing 100`? Does training longer (`--steps 1000`) push the loss below the
counting model's 2.4544, or does it level off? (It levels off — they're the same
model.) Building the *instinct* to ask and test these questions is half of what this
course is teaching.

---

## 🤔 Common questions

- **Why is the best loss ~2.45 and not 0? Did we do something wrong?** No — `2.45`
  is genuinely the best a *bigram* model can do on this data. One character of
  context simply isn't enough to predict the next letter perfectly. Lowering that
  number requires a smarter model with more context, which is the whole arc from
  here to Chapter 5.
- **Why does the neural net land slightly above the counting model (e.g. 2.485 vs
  2.454)?** Two reasons: the regularization term nudges it a touch toward uniform,
  and we only ran a few hundred steps. Train longer with less regularization and it
  creeps right down to the counting value. They're the same model.
- **Is this *really* how ChatGPT works?** The skeleton, yes: predict a probability
  distribution over the next token, sample, repeat. ChatGPT swaps our 1-layer `W`
  for a giant Transformer and our 27 characters for ~100k tokens, but the loop and
  the loss are the same ones you just built.
- **My sampled names don't match the lesson's.** Check that you seeded the generator
  with `manual_seed(2147483647)`. Different seed → different (equally valid) names.
- **Do I need to memorize the math?** No. Aim to understand *what each step does and
  why*. The formulas will feel obvious after you've built them a couple of times.

---

## ✅ Check your understanding

Before moving on, see if you can answer these in your own words. Click to check.

<details>
<summary>1. Why do we wrap every name in the <code>.</code> token?</summary>

So the model can learn which letters tend to *start* a name and which tend to *end*
one — and so generation knows when to stop. One token does double duty as both
"start" and "end".
</details>

<details>
<summary>2. What does <code>keepdim=True</code> do when we normalize, and why does it matter?</summary>

It keeps the row-sums as a `(27, 1)` column instead of collapsing them to a flat
`(27,)`. Dividing the `(27, 27)` grid by a `(27, 1)` column broadcasts correctly —
each row is divided by its own total. Without it, PyTorch would silently divide the
wrong way and produce a broken model with no error.
</details>

<details>
<summary>3. The counting model and the neural net reach the same loss. Why is that not a coincidence?</summary>

Gradient descent drives `W` toward `log(counts)`. The softmax then computes
`exp(W)` = the counts, normalized — which is exactly our probability table `P`. Same
model, found two different ways.
</details>

<details>
<summary>4. Why can't we just keep counting once the model needs more context?</summary>

The table grows as `vocab_size ^ context_length`. Two characters of context need
27³ ≈ 20k cells; ten characters need 27¹⁰ ≈ 10¹⁴ — impossible to fill or store.
Gradient descent learns *parameters* instead of enumerating every case, so it scales.
</details>

## 🎓 Key takeaways

- A **language model** predicts a *probability distribution* over the next token,
  and **generates** text by sampling from it autoregressively — the same loop as
  ChatGPT.
- A **bigram** model assumes the next character depends only on the current one.
- **Counting** bigrams and **normalizing** each row gives a complete language model
  (mind the `keepdim=True` when you normalize!).
- We measure quality with the **average negative log-likelihood (NLL)** — lower is
  better; `log(vocab_size)` is the clueless baseline, `0` is perfect.
- **Smoothing / regularization** keeps the model from assigning probability 0 (and
  loss ∞) to things it hasn't seen.
- The same bigram model is a **1-layer neural network** (`one-hot → W → softmax`)
  trained by **gradient descent** — and it reaches the same answer. *This loop, not
  counting, is what scales to ChatGPT.*

## 📖 New vocabulary

`tensor` · `shape` · `token` · `vocabulary` · `bigram` · `probability distribution`
· `autoregressive generation` · `sampling` · `broadcasting` · `one-hot encoding` ·
`parameters` · `matrix multiply` · `logits` · `softmax` · `likelihood` · `negative
log-likelihood (NLL) / loss` · `perplexity` · `smoothing` · `regularization` ·
`gradient` · `gradient descent` · `learning rate`.

Don't worry about memorizing these — you just *used* every one of them, which is how
they stick.

## 🧪 Practice & build

1. **The notebook first.** If you only read this lesson, do
   [`code/explore.ipynb`](./code/explore.ipynb) now — building it yourself is the
   single highest-value hour in this chapter.
2. **Exercises:** [`exercises/README.md`](./exercises/) — short, focused tasks
   (build a *trigram* model, add a train/dev/test split, tune smoothing, use
   `cross_entropy`). Solutions are in
   [`exercises/solutions/`](./exercises/solutions/) — try first, peek after.
3. **Mini-project:** [`project/README.md`](./project/) — **"The Name Forge"**, an
   interactive name generator you build yourself from a scaffold of TODOs (with a
   reference solution to check against).

## 🔗 Go deeper (optional)

- 🎥 Karpathy, [*The spelled-out intro to language modeling: building makemore*](https://www.youtube.com/watch?v=PaCmpygFfXo)
  — ~2 hours, follows this exact path in video form. Hugely recommended.
- 📄 [Jurafsky & Martin, *Speech and Language Processing*, Ch. 3 — N-gram Language Models](https://web.stanford.edu/~jurafsky/slp3/3.pdf)
  — the friendly textbook treatment of n-grams, smoothing, and perplexity.

---

| ⬅️ Previous | Up | Next ➡️ |
|:--|:-:|--:|
| [Course Home](../../README.md) | [Syllabus](../../README.md#-syllabus) | [Chapter 2 — Micrograd](../02-micrograd/) |
