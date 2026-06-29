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

Same dataset as Chapter 4 (tiny Shakespeare, in [`data/input.txt`](./data/input.txt)). By the end
you'll have a complete, trainable GPT in ~150 lines — the finished reference is
[`code/gpt.py`](./code/gpt.py) — and it generates recognizable Shakespeare in under two minutes on a
laptop CPU.

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

Here's the plan, mapped to the sections below: §1 adds **multi-head** attention (many heads, then a
projection); §2 adds the **feed-forward net** (the per-token "think"); §3–§4 add **residual
connections** and **LayerNorm** (the scaffolding that lets us stack §1 + §2 deep); §5 packages all
four into a reusable **block**; §6 stacks blocks into the **GPT**. Five short steps from one
attention head to a real language model — keep that map in mind as the pieces arrive.

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

**The shapes**, with `n_embd = 64` and `4` heads of size `16`: each head turns `(B, T, 64)` into
`(B, T, 16)`; `torch.cat` glues the four `(B,T,16)` outputs along the last axis back to `(B, T, 64)`;
the projection (a `64 × 64` Linear) then mixes them. Why split 64 into four 16s rather than run one
head of 64? Because each head has its *own* query/key/value, so it can learn its *own* notion of
"what's relevant" — one head might track subject–verb agreement, another match quotes and brackets,
another follow spacing. Four small heads each hunting for a different thing beat one big head forced
to squeeze every pattern into a single set of weights. The projection then lets those four findings
*talk to each other* before the layer's output moves on.

On tiny numbers the concat is literal: if head A outputs `[1, 2]` for a token and head B outputs
`[3, 4]`, then `torch.cat([A, B], dim=-1)` is `[1, 2, 3, 4]` — the heads' findings laid side by
side. The projection (a learned matrix) then blends those four numbers into the token's final
output, so what head A discovered can influence the slots head B wrote, and vice versa.

In a trained model you can often *see* the specialization by visualizing the heads: one head's
attention sharply tracks "attend to the previous space," another "attend to the matching open
quote," another "two characters back." Each becomes a small expert, and the projection pools their
expertise.

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

Why widen to **4×** in the middle? The hidden layer needs room to compute richer features than the
`n_embd`-sized input allows; `4×` is the ratio the original Transformer used, and it stuck. And
*per position, independently* is worth dwelling on: the same little MLP runs on each token's vector
*separately* — no mixing between positions happens here (attention already did that). So a block has
a clean division of labour: **attention moves information *between* tokens; the feed-forward net
transforms *each token on its own*.** Shape-wise it's `(B,T,C) → (B,T,4C) → (B,T,C)` — wide in the
middle, same size in and out, so blocks stack cleanly.

Concretely, for one token: its `n_embd` numbers expand up to `4·n_embd` (the wide `Linear`), `ReLU`
zeroes out the negatives (keeping only the features that "fired"), and a second `Linear` projects
back down. It's a little pattern-detector run on each token's just-gathered context — *"given what I
pulled in from attention, here's a transformed summary."* The same two weight matrices are applied
to every position independently.

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

To feel why, put numbers on it. Imagine 10 layers, each of which (on the way back) multiplies the
gradient by `0.7`. Stacked, the gradient reaching layer 1 is scaled by `0.7¹⁰ ≈ 0.028` — 35× weaker
than at the top, so layer 1 barely budges. (Make the factor `1.3` instead and it *explodes* to
`1.3¹⁰ ≈ 13.8`.) The residual `+` adds a parallel route that multiplies by **`1`** at every layer,
so the gradient always has a clean identity path home — it can't vanish or explode no matter how
deep the stack gets. That single `+1` per layer is what turns "10 layers" into "100 layers that
still train."

There's a second nice consequence, for *learning*: because a block computes `x + (small change)`,
an untrained block — whose sublayers start near zero — is almost the **identity**, passing `x`
through nearly untouched. So dropping a fresh block into a working network *starts* harmless and only
helps as it learns. That's a big part of why you can stack dozens of them and training stays stable
from the very first step.

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

On numbers: say a token's 4 features come out as `[1, 2, 3, 10]`. Their average is `4`, so subtract
it → `[−3, −2, −1, 6]`; their spread (standard deviation) is about `3.5`, so divide → roughly
`[−0.85, −0.57, −0.28, 1.70]`. Now they're centered at 0 with a consistent scale, *whatever* wild
values they started from. LayerNorm does this for every token, in every layer — so no matter how the
activations drift as they flow through a deep stack, each LayerNorm pulls them back into a sane
range before the next sublayer ever sees them.

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

Notice a block takes `(B, T, C)` in and returns `(B, T, C)` out — *same shape*. That's exactly why
you can stack them: each block's output is a valid input to the next. A block doesn't change the
*shape* of the representation, it *refines* it — like passing a draft through one editor after
another, each making it a little sharper.

(What do the *different* blocks learn? Roughly, earlier blocks pick up local, surface patterns —
which characters tend to follow which — and later blocks build more abstract structure on top, like
where a line or a speaker's turn should end. Nobody assigns these roles; depth plus the residual
highway lets the stack discover a hierarchy on its own.)

**Follow a token through one block.** Say we're processing the `t` in `"cat"`. (1) LayerNorm tidies
its features. (2) Attention lets it look back at `c` and `a` and pull in context — "I'm the third
letter of a short word starting `ca`" — that's *communicate*. (3) The residual `+` adds that
gathered context onto the original `t`. (4) LayerNorm again, then (5) the feed-forward net transforms
the combined vector on its own — "given a `t` after `ca`, here's a more useful representation" —
that's *think*. (6) Another residual `+`. Out comes a `t` that now carries both its own identity and
what it learned from its neighbours. Stack four of these and the representation grows
correspondingly richer.

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

Here's the whole GPT on one page:

```
   token ids  (B, T)
       │   token embedding  +  position embedding
       ▼
    x  (B, T, C)
       │        ┌──────────────── Block ────────────────┐
       ├───────►│  x = x + attention( LayerNorm(x) )      │  ← communicate
       │        │  x = x + feedforward( LayerNorm(x) )    │  ← think
       │        └────────────────────────────────────────┘
       │            (n_layer identical blocks, stacked)
       ▼
   final LayerNorm
       │   lm_head:  Linear(C → vocab)
       ▼
   logits  (B, T, vocab)   →   softmax   →   P(next token)
```

Embeddings in, a stack of identical blocks, a final norm, a linear head out. Every block does the
same two things — "communicate, then think" — each wrapped in a residual `+` and a LayerNorm. The
rest of this section is just naming the pieces in that picture.

The embeddings are exactly Chapter 4's: `token_embedding` gives each character its learned vector,
`position_embedding` gives each slot `0…T−1` its own, and we **add** them so each input carries both
content and position (`torch.arange(T)` is just the position ids `[0, 1, …, T−1]`).

This is a **decoder-only Transformer**. The original 2017 Transformer had two halves — an
*encoder* (which reads a whole input at once, unmasked) and a *decoder* (which generates
left-to-right). For pure text generation we keep only the decoder half — hence "decoder-only" —
whose causal mask means every position only sees the past, exactly what a next-token generator
needs. It is, to the architecture, GPT-2.

**A forward shape trace** (`B=32, T=64, n_embd=128, n_layer=4, vocab=65`):

```
idx                  (32, 64)         token ids
tok + pos embedding  (32, 64, 128)    each token → a 128-vector (content + position)
blocks (×4)          (32, 64, 128)    refined by 4 stacked blocks — shape unchanged
final LayerNorm      (32, 64, 128)
lm_head (→ 65)       (32, 64, 65)     logits: a next-char score at every position
```

**How big is it?** ~817K parameters for this config. The bulk is the four blocks: each has the
attention (q/k/v projections + the output projection ≈ 66K) plus the feed-forward net
(`128→512→128` ≈ 131K) ≈ ~197K per block × 4 ≈ 790K, with the embeddings (`65×128 + 64×128` ≈ 17K)
and output head (`128×65` ≈ 8K) making up the rest. That's *tiny* — GPT-3 has 175 **billion** — but
the architecture is byte-for-byte the same; only the numbers grow.

It's worth pausing on what you've built. The 2017 paper that introduced this was titled *"Attention
Is All You Need"*, and the claim held up: this one architecture, scaled, swept aside the specialized
models that came before it — first in language, later in vision, audio, and code. The reasons are
exactly the ones you've now seen: attention's parallel, long-range routing (Chapter 4) plus the
residual + LayerNorm scaffolding that makes it stackable to great depth. Everything from here
(Chapters 6–17) makes *this* model bigger, faster, better-fed, and better-behaved — it never
replaces it.

---

## 7. A note on dropout

The reference code also sprinkles in **`nn.Dropout`** — during training it randomly zeroes a
fraction (say 20%) of activations on each forward pass, then scales the rest up to compensate. Why
deliberately break things? Because it stops the model from over-relying on any single path or
neuron — every unit has to pull its weight, since it can't count on its neighbours always being
there. That's **regularization**: it fights the overfitting you met in Chapter 3 (a model
memorizing the training set instead of learning general patterns). Crucially, dropout is **off at
generation time** — `model.eval()` switches it off; you only want the noise while *learning*. One
line per sublayer, and safe to ignore on a first read, but it's a standard part of the recipe.

---

## 8. Train it, and read the result

We train it exactly as before — the same reset → backward → update loop with **AdamW**, on random
chunks of Shakespeare, for a few thousand steps (~2 minutes on a laptop). The *only* difference from
Chapter 4 is that the model is now a stack of blocks instead of one head. Then we read both the loss
and the samples.

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
a model that has only ever seen characters.

Read it closely and you can see *which* patterns it picked up: the `NAME:` line format (it learned
the dialogue structure), capital letters after newlines, spaces clustering letters into word-sized
chunks, even plausible letter runs (`musterford`, `noblers`). What it *hasn't* learned is meaning —
the words are mostly non-words — because at this size, with single-character tokens, it can capture
*form* but not *content*. Closing that gap is exactly what scale and subword tokens (Chapter 6)
begin to do. The same architecture, scaled up on a GPU and
trained longer (Chapters 7–10), is what writes fluent text.

**How does it actually generate?** The same autoregressive loop as Chapters 1, 3, and 4 — only the
next-token distribution now comes from the GPT. Start from a seed token, run the model, take the
logits at the *last* position, softmax to probabilities, sample one token, append it, repeat:

```python
for _ in range(max_new_tokens):
    logits = model(idx[:, -block_size:])          # crop to the last block_size tokens
    probs  = F.softmax(logits[:, -1, :], dim=-1)  # distribution for the NEXT token
    idx_next = torch.multinomial(probs, 1)        # sample one
    idx = torch.cat([idx, idx_next], dim=1)       # append it, and loop
```

The one new wrinkle is `idx[:, -block_size:]`: the model has a *fixed* context window, so we feed it
only the most recent `block_size` tokens. Run this 500 times and out comes the Shakespeare above —
one character at a time, each conditioned on everything generated so far.

---

## 🐛 Building it yourself: what trips people up

- **Forgetting the residual `+`.** Writing `x = self.sa(self.ln1(x))` instead of
  `x = x + self.sa(self.ln1(x))` silently drops the skip connection. It still *runs* — and even
  trains OK at 4 layers — so the bug hides until you go deeper. (The notebook's check catches it.)
- **`head_size` must divide `n_embd`.** With `n_embd=64` and 4 heads, `head_size=16` and the
  concatenation comes back to 64. A head count that doesn't divide evenly breaks the shapes.
- **Pre-norm vs post-norm.** Normalize *before* the sublayer (`x + sa(ln(x))`), not after
  (`ln(x + sa(x))`). Both "work," but pre-norm is far more stable for deep stacks — it's what GPT-2
  settled on.
- **Dropout left on at generation.** Dropout should act only during training. PyTorch toggles it
  via `model.train()` / `model.eval()`; forget to switch to `eval()` and your samples come out
  randomly corrupted.
- **The `(B, T, C)` shape, again.** As in Chapter 4, most bugs here are shape bugs. A block *must*
  return the same `(B, T, C)` it received, or the next block — and the residual `+` — will choke.

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
- **Why does the feed-forward net widen to 4× and back?** The wide middle gives the network room to
  compute richer per-token features than the input size allows; projecting back to `n_embd` keeps a
  block's output the same shape as its input so blocks stack. `4×` is the original Transformer's
  ratio.
- **How many heads / layers should I use?** Hyperparameters. More heads = more kinds of
  relationship tracked at once; more layers = more rounds of communicate-then-think (more capacity,
  but harder to train without the residual+norm scaffolding). Our config uses 4 of each; GPT-2 used
  12 and 12, GPT-3 used 96 and 96.
- **Where does the actual "understanding" live?** Spread across all of it: attention routes
  information, the feed-forward nets transform it, and stacking blocks lets later ones build on
  patterns earlier ones found. No single piece "understands" — the *stack* does.
- **Does the position embedding work the same as in Chapter 4?** Yes — `nn.Embedding(block_size,
  n_embd)` added to the token embedding, so each token knows *where* it is. Bigger models sometimes
  use fancier schemes (rotary, ALiBi), but the idea is identical.
- **What's the `lm_head`?** The final `nn.Linear(n_embd, vocab_size)` that turns each token's
  refined vector into `vocab_size` logits — one score per possible next token, just like every model
  since Chapter 1. The final LayerNorm right before it keeps those inputs tame.

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

<details>
<summary>5. A Transformer block takes <code>(B, T, C)</code> in — what shape does it return, and why does that matter?</summary>

`(B, T, C)` — the *same* shape. That's exactly what lets you stack blocks: each block's output is a
valid input to the next. A block refines the representation without changing its shape.
</details>

## 🧭 The journey so far

Every model we've trained on Shakespeare, by validation loss:

| Chapter | Model | Val loss |
|---------|-------|---------:|
| 4 | bigram baseline | ~2.5 |
| 4 | one attention head | 2.34 |
| **5** | **full GPT** (4 heads × 4 blocks + FFN + residual + LayerNorm) | **1.81** |

And the *text* went from `POr, Coll. I Tshat ke` to speaker names and dialogue. Every drop came from
the same recipe — *more capacity, better wired* — and the road ahead (Chapters 6–10) is more of it:
subword tokens, a GPU, more data, more layers.

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
