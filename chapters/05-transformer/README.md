# Chapter 5 — The Transformer: Building a GPT

> *This is the one.* Chapter 4 gave us a single attention head and structured gibberish. Now
> we assemble the **full Transformer** — the exact architecture behind GPT-2 and ChatGPT — by
> wrapping attention in four more ideas: **multi-head attention**, a **feed-forward net**,
> **residual connections**, and **LayerNorm**. We stack those into blocks, stack blocks into a
> GPT, train it on Shakespeare, and watch the output turn into recognizable English.

**You will be able to:**
- build **multi-head attention** (several heads in parallel + a projection);
- add a **feed-forward net** — the "think about what you gathered" step;
- explain **residual connections** and *why* deep networks need them;
- explain **LayerNorm** and where it goes (pre-norm);
- assemble these into a **Transformer block**, stack them into a **GPT**, and train it;
- recognize this as a real **decoder-only GPT** — the thing ChatGPT is, just bigger.

**Prerequisites:** [Chapter 4](../04-attention/) (self-attention, the `(B,T,C)` shapes,
`nn.Module`/`nn.Linear`) and the earlier chapters' MLP, training loop, and `cross_entropy`. We
introduce a few more PyTorch wrappers (`nn.Sequential`, `nn.ReLU`, `nn.LayerNorm`,
`nn.Dropout`) and explain each.

**Time:** ~3 hours. **Hardware:** a laptop CPU — the model trains in **under 2 minutes**.

> ### 📓 Build it yourself in the notebook
> **[`code/explore.ipynb`](./code/explore.ipynb)** walks you through each new piece — multi-head,
> feed-forward, the residual+norm block — then trains the GPT and generates text. "✍️ Your
> turn", "📖 Study & run", and "▶️ Check your work" cells. Watch for: ✏️ **In the notebook → Step N**.

---

## 0. Setup

Same dataset as Chapter 4 (tiny Shakespeare, in [`data/input.txt`](./data/input.txt)). The
finished reference is [`code/gpt.py`](./code/gpt.py).

```bash
uv venv --python 3.13 .venv && uv pip install -r requirements.txt
.venv/bin/jupyter lab        # open chapters/05-transformer/code/explore.ipynb
```

---

## A 2-minute primer: what's still missing

Chapter 4's model was *one* attention head feeding straight into the output. It learned the
*shape* of English but couldn't go further, for three reasons:

1. **One head sees one kind of relationship.** Real language needs many at once (who's the
   subject? what punctuation is due? what word started this line?).
2. **It only *gathered* information — it never *processed* it.** Attention mixes tokens'
   values together, but nothing then thinks about the mixture.
3. **It was shallow.** One layer can't build up complex structure; you need to stack many.
   But naively stacking deep layers makes training collapse.

The Transformer fixes all three: **multi-head** attention (#1), a **feed-forward** net (#2),
and **residual connections + LayerNorm** that make **deep stacks** trainable (#3). Let's add
them one at a time.

---

## 1. Multi-head attention

> ✏️ **In the notebook → Step 2.** Build multi-head attention.

Instead of one head of size `n_embd`, run **several smaller heads in parallel** — say 4 heads
of size `n_embd/4` — and concatenate their outputs back to `n_embd`. Each head can specialize.
Then a **projection** (a `nn.Linear`) mixes the heads' findings:

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_head)])  # the Ch4 head, ×n
        self.proj = nn.Linear(head_size * n_head, n_embd)                     # mix the heads
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)   # (B, T, n_head*head_size) = (B,T,C)
        return self.proj(out)
```

(`nn.ModuleList` is a list of layers that `nn.Module` tracks for you; `torch.cat(..., dim=-1)`
glues the heads' `(B,T,head_size)` outputs side by side along the last axis.) Why the
projection, when `cat` already gives the right shape? Each head computed its chunk *in
isolation* — the projection is a learned mix that lets information flow *across* heads. (If you
did Chapter 4's optional E05, you built the multi-head part; the projection here is the new piece.)

---

## 2. The feed-forward net: gather, then think

> ✏️ **In the notebook → Step 3.** Build the feed-forward net.

Attention is **communication** — each token pulls in information from others. But it never
*processes* that information. So after attention, every token passes through a small **MLP**
on its own (the Chapter 3 idea: linear → nonlinearity → linear):

```python
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),   # widen to 4× (the standard ratio)
            nn.ReLU(),                       # nonlinearity: max(0, x)
            nn.Linear(4 * n_embd, n_embd),   # back to n_embd
        )
    def forward(self, x):
        return self.net(x)
```

(`nn.Sequential` just chains layers so the output of one feeds the next; `nn.ReLU` is the
nonlinearity `max(0, x)` — a simpler cousin of Chapter 3's `tanh`/GELU.) The slogan for a
Transformer block is **"communicate, then think"**: attention gathers, the feed-forward net
digests — *per position, independently*.

---

## 3. Residual connections: the highway that makes depth trainable

> ✏️ **In the notebook → Step 4.** This is the key idea for deep networks (you'll add it inside the block).

Here's the problem with deep stacks: gradients have to travel back through *every* layer, and
they tend to shrink or explode along the way, so the early layers barely learn. Training a
deep plain stack often just... stalls.

The fix is shockingly simple. Instead of replacing `x` with a sublayer's output, **add the
output to `x`**:

```python
x = x + self.sa(x)       # NOT  x = self.sa(x)
x = x + self.ffwd(x)
```

That `+` creates a **residual connection** (a "skip connection"). Picture a highway running
straight through the network, with each sublayer as an off-ramp that computes a small change
and merges it back. Two things follow:

- **Gradients flow freely.** In a deep plain stack, the gradient gets multiplied by something at
  each layer on its way back, and those factors compound — shrinking it toward zero (so early
  layers stop learning) or blowing it up. The `+` fixes this: the slope of `x + f(x)` as `x`
  changes is `1 + (slope of f)`, and that **`1`** gives the gradient a clean path straight back
  through every layer, even when `f`'s part is tiny. Early layers actually learn.
- **Each sublayer only learns a *small adjustment*** to the running representation, rather than
  rebuilding it from scratch — which is far easier to optimize.

Residual connections are the single trick that unlocked very deep networks (they come from
ResNet, 2015). Without them, you cannot train a 12- or 96-layer Transformer.

---

## 4. LayerNorm: keep the numbers tame

> ✏️ **In the notebook → Step 4.**

The other half of "make deep stacks trainable" is **LayerNorm**. As data flows through many
layers, the activations can drift to wildly different scales, which destabilizes training.
LayerNorm fixes this by **normalizing each token's feature vector**: take the `C` numbers for a
token, subtract their average and divide by their spread (how far they typically sit from that
average), so they come out centered at 0 with a consistent scale. Then it multiplies by two
small *learned* vectors — so the model can dial how much of that normalization it actually keeps
(it isn't locked to scale 1):

```python
self.ln1 = nn.LayerNorm(n_embd)
... self.ln1(x) ...     # normalizes the C numbers of every token, independently
```

We use **pre-norm**: apply LayerNorm *before* each sublayer (this is what GPT-2 does and it
trains more stably than the original *post-norm* — LayerNorm applied *after* the sublayer, on the
merged result). So a sublayer becomes `x + sublayer(ln(x))`.
(`nn.LayerNorm(n_embd)` is a layer with two small learnable vectors; you don't have to compute
the mean/variance yourself.)

---

## 5. The Transformer block

> ✏️ **In the notebook → Step 4.** Assemble the block.

Now stack the four ideas into one reusable **block**: LayerNorm → multi-head attention →
residual, then LayerNorm → feed-forward → residual.

```python
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.sa = MultiHeadAttention(n_head, n_embd // n_head)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))      # communicate (attention), add back
        x = x + self.ffwd(self.ln2(x))    # think (feed-forward), add back
        return x
```

That's the whole Transformer block — the unit that GPT-2 stacks 12 times and GPT-3 stacks 96.

---

## 6. The GPT

> ✏️ **In the notebook → Steps 5–6.** Assemble and train the GPT.

The full model: token + position embeddings → a stack of `n_layer` blocks → a final LayerNorm
→ a `nn.Linear` to `vocab_size` logits.

```python
class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)              # a final norm
        self.lm_head = nn.Linear(n_embd, vocab_size)
    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_embedding(idx) + self.position_embedding(torch.arange(T))
        x = self.blocks(x)
        logits = self.lm_head(self.ln_f(x))
        ...                                            # cross_entropy loss, exactly as before
```

(`nn.Sequential(*[Block(...) for _ in range(n_layer)])` builds a list of `n_layer` blocks, and
the `*` *spreads* that list into separate arguments — as if you'd typed `Block(), Block(), …` by
hand — so `self.blocks(x)` runs `x` through all of them in order.)

This is a **decoder-only Transformer**. The original 2017 Transformer had two halves — an
*encoder* (which reads a whole input at once, unmasked) and a *decoder* (which generates
left-to-right). For pure text generation we keep only the decoder half — hence "decoder-only" —
whose causal mask means every position only sees the past, exactly what a next-token generator
needs. It is, to the architecture, GPT-2.

---

## 7. A note on dropout

The reference code also sprinkles in **`nn.Dropout`** — during training it randomly zeroes a
fraction of activations, which prevents the model from over-relying on any one path
(regularization, fighting the overfitting you saw in Chapter 3's exercise). It's off
automatically at generation time. One line per sublayer; you can ignore it on a first read.

---

## 8. Train it, and read the result

> 🔮 **Predict before you run:** Chapter 4's single head got **2.34**. With multi-head +
> feed-forward + 4 stacked blocks, where does the val loss land?

<details><summary>👉 Reveal</summary>

```
step     0 | train 4.1777 | val 4.1778
step  2999 | train 1.6680 | val 1.8142
```

From 4.18 to **1.81** — a huge drop. But the loss isn't the story. The *text* is:
</details>

```
KING EWIO:
Etwar, our, inte is-cry out musterford,
He have to no at see hold see

BRUKINGHABELAS:
Suare thy with arquent of taoure;
...
Their tagent comp noblers!
```

Compare that to Chapter 4's `POr, Coll. I Tshat ke`. We now have **speaker names**, line
breaks, dialogue, capitalization, and word-shaped words — recognizably (broken) English, from
a model that has only ever seen characters. The same architecture, scaled up on a GPU and
trained longer (Chapters 7–10), is what writes fluent text.

---

## 🤔 Common questions

- **Why "communicate, then think"?** Attention (communicate) only *mixes* tokens' information;
  it can't transform it nonlinearly. The feed-forward net (think) processes each token's
  gathered information on its own. A block needs both.
- **Why does adding `x +` help so much?** It gives gradients a direct route backward (the
  derivative of `x + f(x)` keeps a `+1` term), so deep stacks actually train, and each layer
  only has to learn a small *change* rather than a full transformation.
- **Why LayerNorm *before* the sublayer (pre-norm)?** It keeps the residual highway clean (the
  thing added back is normalized) and trains more stably than the original post-norm. It's what
  GPT-2 uses.
- **Is this *actually* the ChatGPT architecture?** Yes — a decoder-only Transformer of exactly
  these blocks. ChatGPT differs in scale (thousands of dims, ~100 layers), tokenization
  (Chapter 6), and the finetuning that makes it follow instructions (Chapters 14–15) — but the
  core you just built is the same.
- **Why is the text still broken?** Small model (817K params), short training, and *character*
  tokens. Bigger + longer + subword tokens (Chapter 6) sharpen it dramatically.

---

## ✅ Check your understanding

<details>
<summary>1. What does "communicate, then think" mean in a Transformer block?</summary>

**Communicate** = multi-head attention, where each token gathers information from other tokens.
**Think** = the feed-forward net, where each token then processes that gathered information on
its own. A block does both, in that order.
</details>

<details>
<summary>2. Why are residual connections (<code>x = x + sublayer(x)</code>) essential for deep models?</summary>

They give gradients a direct path backward through every layer (so early layers actually
learn), and they let each sublayer learn a small *adjustment* to the representation rather than
rebuild it. Without them, deep stacks fail to train.
</details>

<details>
<summary>3. What does LayerNorm do, and where do we put it (pre-norm)?</summary>

It normalizes each token's feature vector to mean 0 / variance 1 (then rescales with learned
parameters), keeping activations from drifting to wild scales as they pass through many layers.
Pre-norm puts it *before* each sublayer: `x + sublayer(ln(x))`.
</details>

<details>
<summary>4. Why is multi-head attention better than one big head?</summary>

Different heads can specialize in different relationships (subject-verb, punctuation, line
start, …) and attend to several of them simultaneously; a projection then mixes their findings.
One head can only express one attention pattern at a time.
</details>

## 🎓 Key takeaways

- **Multi-head attention** = several heads in parallel + a projection — attend to many things
  at once.
- A **feed-forward net** after attention is the "think" step (per-position MLP); the block's
  job is *communicate, then think*.
- **Residual connections** (`x = x + sublayer(x)`) are the highway that lets gradients reach
  deep layers — the key to training depth.
- **LayerNorm** (pre-norm) keeps activations tame and training stable.
- A **block** = LN → attention → residual, LN → feed-forward → residual; a **GPT** is a stack
  of blocks + embeddings + a final norm + an output head.
- This **decoder-only Transformer** is, architecturally, GPT-2 / ChatGPT — ours just hits
  val **1.81** and writes broken-but-recognizable Shakespeare. Scale is what's left.

## 📖 New vocabulary

`Transformer` · `multi-head attention` · `projection` · `feed-forward net` · `nn.Sequential` ·
`nn.ReLU` · `residual / skip connection` · `LayerNorm` · `pre-norm` · `dropout` ·
`Transformer block` · `decoder-only` · `GPT` · `n_layer` / `n_head`.

## 🧪 Practice & build

1. **The notebook** — [`code/explore.ipynb`](./code/explore.ipynb): build multi-head, the
   feed-forward net, the residual+norm block, then the GPT, and train it.
2. **Exercises** — [`exercises/`](./exercises/): ablate the residual connections (watch it
   break), ablate LayerNorm, scale `n_layer`/`n_head`, and compare ReLU vs GELU. Tiered hints +
   solutions; a starter for the first.
3. **Mini-project** — [`project/`](./project/): **"Your First GPT"** — assemble the blocks into
   a GPT you build from a starter, train it, and generate a paragraph of your own Shakespeare.

## 🔗 Go deeper (optional)

- 🎥 Karpathy, [*Let's build GPT: from scratch, in code, spelled out*](https://www.youtube.com/watch?v=kCc8FmEb1nY)
  — this chapter assembles exactly what the second half of that video builds.
- 📄 [Vaswani et al. (2017), *Attention Is All You Need*](https://arxiv.org/abs/1706.03762)
  — the original Transformer paper.
- 💻 [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) — the clean, scaled-up version of
  exactly this model.

---

| ⬅️ Previous | Up | Next ➡️ |
|:--|:-:|--:|
| [Chapter 4 — Attention](../04-attention/) | [Syllabus](../../README.md#-syllabus) | [Chapter 6 — Tokenization](../06-tokenization/) |
