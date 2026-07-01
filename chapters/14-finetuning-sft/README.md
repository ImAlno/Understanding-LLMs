# Chapter 14 — Finetuning I: SFT, PEFT & LoRA

> Everything so far built a model that **continues text** — feed it "Once upon a time" and it writes
> a story. But ChatGPT doesn't continue your message, it *answers* it. That behaviour isn't in the
> pretrained model; it's taught, in a second stage called **supervised fine-tuning (SFT)**: train on
> **(instruction → response)** pairs, formatted with a **chat template**, so the model learns to
> respond rather than ramble. The twist that makes it work is **loss masking** — train only on the
> response, never on the user's words. And because fine-tuning *all* the weights of a big model is
> expensive, we do it the modern way: **LoRA**, tiny trainable low-rank adapters bolted onto frozen
> weights. We build all of it from scratch and watch a base model turn into an instruction-follower.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ImAlno/Understanding-LLMs/blob/main/chapters/14-finetuning-sft/code/explore.ipynb)

> 💻 **No GPU needed.** We SFT a tiny GPT on a handful of instruction pairs — it learns to follow them
> on a CPU in seconds. The *mechanism* is identical to fine-tuning a 7B model; only the scale differs.

**You will be able to:**
- explain the difference between a **base** model (continues text) and an **instruct/chat** model
  (follows instructions), and how SFT bridges them;
- format data with a **chat template** (special role tokens) and build the **loss mask** that trains
  only the response;
- run **supervised fine-tuning** and watch a base model become an instruction-follower;
- explain why **full fine-tuning** is expensive and what **PEFT** buys you;
- implement **LoRA** from scratch (freeze `W`, add a trainable low-rank `B·A`), reason about its
  parameter savings, and place **QLoRA** (LoRA on a quantized base).

**Prerequisites:** the training loop and `cross_entropy` (Chapters 3–5), the GPT and its `generate`
(Chapter 5), and — for LoRA's cousin QLoRA — quantization (Chapter 13).

**Time:** ~2.5 hours. **Hardware:** any laptop CPU.

> ### 📓 Build it yourself in the notebook
> **[`code/explore.ipynb`](./code/explore.ipynb)**: see a base model ignore an instruction, build the
> loss mask, implement `LoRALinear`, then SFT a tiny GPT and watch it start answering. "✍️ Your turn",
> "▶️ Run this", "▶️ Check your work" cells. Watch for: ✏️ **In the notebook → Step N**.

---

## 0. Setup

Three small scripts, all CPU, all self-contained:

- [`code/gpt.py`](./code/gpt.py) — a compact GPT plus the SFT data helpers (a tiny instruction set,
  the chat template, the loss mask).
- [`code/sft.py`](./code/sft.py) — supervised fine-tuning end to end, with a before/after demo.
- [`code/lora.py`](./code/lora.py) — LoRA from scratch: the adapter, its parameter savings, and a
  fit demo.

```bash
python chapters/14-finetuning-sft/code/sft.py     # base model -> instruction-follower
python chapters/14-finetuning-sft/code/lora.py     # tiny adapters, big savings
```

---

## A 2-minute primer: base vs. chat models

Pretraining (Chapters 1–11) produces a **base model**: it has read a mountain of text and learned to
**continue** it. Give a base model *"What is the capital of France?"* and it might continue with
*"What is the capital of Germany? What is the capital of…"* — happily extending the pattern, because
that's what "predict the next token" rewards. It has all the knowledge; it just doesn't know it's
supposed to *answer*.

An **instruct** (or **chat**) model does. The difference isn't more knowledge — it's a learned
*behaviour*: when it sees a question, respond; when it sees a request, fulfil it; then stop. That
behaviour is installed by **fine-tuning** the base model on examples of the behaviour you want. The
first and most important such stage is **SFT**, and it's remarkably simple: it's the same training
loop you already know, pointed at different data.

Making a chat model is really **three stages**: **pretraining** (Chapters 1–11) teaches language and
knowledge from a mountain of raw text; **SFT** (this chapter) teaches the assistant *format* and
behaviour from demonstrations; **alignment** (Chapter 15) polishes it toward human preferences. SFT
does most of the heavy lifting of "act like a helpful assistant," which is why we start here.

---

## 1. SFT: same loss, different data

> ✏️ **In the notebook → Step 4.**

Supervised fine-tuning uses the *exact* next-token cross-entropy loss from Chapter 5. The only thing
that changes is the **data**: instead of raw text, you train on **(instruction, response)** pairs —
demonstrations of the assistant doing the right thing:

```
  instruction: "hi"        response: "hello!"
  instruction: "thanks"    response: "welcome!"
  ...
```

Show the model enough of these and it generalizes the *pattern* "given an instruction, produce a
helpful response, then stop." Real SFT datasets are tens of thousands of such pairs across every kind
of task (answer questions, summarize, write code, refuse harmful requests); ours is five toy pairs,
but the machinery is identical. (A real example is the same shape, just richer:
`"Summarize: <article>"` → `<a good summary>`; `"Write a Python function to reverse a list"` →
`<the code>`; a harmful request → a polite refusal. Thousands of these, across every task, and the
model learns the *general* skill of being helpful — not any one task.) Fine-tuning is *cheap* compared to pretraining — a base model already
knows language, so a little SFT reshapes its behaviour without teaching it anything new.

---

## 2. The chat template

> ✏️ **In the notebook → Step 2.**

A raw (instruction, response) pair has no structure the model can see — where does the instruction
end and the response begin? So we wrap each turn in a **chat template**: special tokens that mark the
**roles**. Every chat model has one; a minimal version is:

```
  <user> hi <assistant> hello! <end>
```

The `<user>` / `<assistant>` tokens tell the model whose turn it is, and `<end>` tells it to stop
generating (so it doesn't run on forever). At inference you feed it `<user> your message <assistant>`
and let it generate the response up to `<end>`. Real templates (ChatGPT's, Llama's) are fancier —
system prompts, multiple turns — but they're the same idea: **structure the conversation with special
tokens** so the model can learn the turn-taking. (These are the reserved "special tokens" from
Chapter 6's tokenizer, finally earning their keep.)

---

## 3. Loss masking: train only the response

> ✏️ **In the notebook → Step 2.** Build the mask.

Here's the crucial subtlety. Each training example is the *whole* templated conversation —
`<user> hi <assistant> hello! <end>` — but you do **not** want to train the model to produce the
*user's* words. If you did, it would learn to generate questions, not answers. You only want it to
learn the **response**.

The fix is **loss masking**: compute the loss only on the response tokens, and ignore the prompt
tokens. In PyTorch, `cross_entropy(..., ignore_index=-100)` skips any target set to `-100`, so you
build the targets with the prompt masked out. From [`sft.py`](./code/sft.py), one example:

```
chat template : '<user>hi<assistant>hello!<end>'
trained target: '___hello!<end>'    ('_' = masked prompt, trained only on the response)
```

Every `_` is a position whose target is `-100` — the model still *reads* the prompt (it's in the
input), but no gradient flows from predicting it. Only `hello!<end>` contributes to the loss. This
input-vs-loss split is the key: the response is still **conditioned on** the whole instruction (the
model reads `hi` to know to say `hello!`), we just don't *train* it to produce the instruction —
**input = everything, loss = response only.** Get
this wrong (train on everything) and the model wastes capacity learning to parrot instructions;
mask correctly and every gradient teaches it to *respond*.

---

## 4. The result: base → instruct

> ✏️ **In the notebook → Step 1.** See the base model ignore you.

Put it together — chat template, loss mask, the ordinary training loop — and a base model *becomes*
an instruction-follower. [`sft.py`](./code/sft.py) shows the before and after on our tiny GPT:

```
before SFT:
  'hi'     -> 'wsbm\x02thb'          (untrained: babble)
  'bye'    -> 'wdk\x02thobirwd'
  'thanks' -> 'thyobirwd\x02w'

after SFT (loss 0.0006):
  'hi'     -> 'hello!'     ✓
  'bye'    -> 'goodbye!'   ✓
  'thanks' -> 'welcome!'   ✓
  ...
5/5 correct — the base model became an instruction-follower.
```

Before SFT the model babbles; after a few hundred steps on five masked examples, it answers every
instruction and stops cleanly at `<end>`. With only five pairs and no held-out test, our tiny model is really **memorizing** those
five mappings — but the *procedure* is identical to what you'd run on tens of thousands of pairs, where
those same steps produce genuine generalization. That's the whole of SFT in miniature — and it's exactly
how a pretrained 7B base becomes a chat assistant, just with far more (and far more varied) data.

---

## 5. Full fine-tuning is expensive → PEFT

The SFT above updated *every* weight in the model — **full fine-tuning**. That's fine for our toy GPT,
but for a real one it's painful: fine-tuning a 7B model in the usual way needs to store the weights,
their gradients, *and* the optimizer state — Chapter 9's ~16 bytes/param (fp16 weights + fp16 grads +
fp32 master + Adam's `m` and `v`), so `7B × 16 ≈ 112 GB` *before activations*, a multi-GPU job — and
a *separate full copy* of the model for every task you fine-tune. Costly to train, costly to store,
costly to serve.

**PEFT** — Parameter-Efficient Fine-Tuning — is the answer: instead of updating all the weights,
**freeze them and train only a tiny number of new ones.** You keep one frozen base model and a small
"adapter" per task. Much less memory, much less to store, and you can hot-swap adapters. The dominant
PEFT method, by far, is **LoRA**.

---

## 6. LoRA: fine-tune with tiny low-rank adapters

> ✏️ **In the notebook → Step 3.** Implement it.

LoRA (Low-Rank Adaptation) freezes the original weight matrix `W` and learns a small **low-rank
update** beside it. For a layer `output = W·x`, LoRA computes:

```
  output = W·x  +  (B·A)·x          A is r×k,  B is d×r,  rank r ≪ d, k
```

Here the **rank** `r` is how many independent *directions* a matrix can push its input along: a full
`d×k` weight can use up to `min(d, k)` of them, but `B·A` — squeezed through its skinny `r`-wide waist —
can use at most `r`. That's the "low-rank" in LoRA. `W` (shape `d×k`) stays frozen; only `A` and `B` are
trained. Their product `B·A` is a `d×k` matrix just like `W`, so it's a valid weight update — but because
it's built from two skinny matrices, it has only `r·(d+k)` parameters instead of `d·k`. With `B`
initialized to **zero**, the adapter starts as a no-op (`B·A = 0`), so fine-tuning begins exactly at the
pretrained model and gently adjusts from there. (`A` is random rather than zero so that a gradient can
reach `B` on the very first step — if *both* started at zero, `B·A` would stay zero and the adapter would
never budge.) From [`lora.py`](./code/lora.py):

```python
class LoRALinear(nn.Module):
    def __init__(self, base, r=8):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False              # freeze W
        d, k = base.out_features, base.in_features
        self.A = nn.Parameter(torch.randn(r, k) * 0.02)
        self.B = nn.Parameter(torch.zeros(d, r)) # start as a no-op
    def forward(self, x):
        return self.base(x) + (x @ self.A.T) @ self.B.T
```

(The formula writes `W·x` with column vectors; PyTorch's `nn.Linear` stores rows and computes `x @ W.T`,
which is why in code the adapter reads `(x @ A.T) @ B.T` rather than a literal `B·A·x` — same map, transposed.)

**Why so few parameters?** The savings grow with the layer size (real output, rank 8):

```
         layer         full       LoRA   LoRA %
    128×128            16,384      2,048    12.5%
    1024×1024        1,048,576     16,384     1.6%
    4096×4096       16,777,216     65,536     0.4%
```

On a 128×128 layer it's 12.5%; on a real 4096-wide layer it's 0.4%; across a whole 7B model, LoRA
trains **well under 1%** of the parameters. Yet it works, because of the **low-rank hypothesis**: the
*change* fine-tuning needs is simple — a few directions of adjustment, not a full rewrite — so a
low-rank matrix captures it. `lora.py` shows this directly: asked to fit a target that differs from
`W` by a rank-8 update, LoRA nails it exactly (`loss 6.5 → 0.000000`).

So with rank `r = 8`, the fine-tune can nudge each layer along just 8 learned directions — plenty to install "be
polite" or "answer, then stop," but far too few to relearn the layer from scratch. That's the whole
bet, and it usually pays off: *behaviour* changes are low-rank, *knowledge* changes are not — which
is exactly why you SFT behaviour and pretrain for facts.

Two more LoRA facts worth knowing:
- **Merge for free inference.** After training, you can compute `W + B·A` once and fold it back into a
  plain `nn.Linear` — so LoRA adds **zero** inference cost. During training it's cheap; at deployment
  it's invisible.
- **Adapters are tiny and swappable.** A LoRA adapter for a 7B model is a few MB. You keep one base
  model and swap in a "coding" adapter, a "medical" adapter, a "your-company-tone" adapter — each a
  small file — instead of storing a full 14 GB model per task.

---

## 7. QLoRA: LoRA on a quantized base

Combine this chapter with the last one and you get **QLoRA**, the technique that put fine-tuning
within reach of a single consumer GPU. The idea: **quantize the frozen base model to 4-bit** (Chapter
13) — so the giant, unchanging weights barely use memory — and train **LoRA adapters** (kept in
higher precision) on top. The base is int4 and never updated; only the tiny fp16 adapters learn. That
combination lets you fine-tune a **65B** model on a single 48 GB GPU, where full fine-tuning would
need a cluster. It's the natural endpoint of the inference-and-tuning arc: *quantize what you keep,
adapt with a sliver.*

One honest limit to close on: SFT can only **imitate its demonstrations**. It gets you a model that
*follows* instructions, but every response is just "copy the style of the training answers" — it has
no notion that one good answer is *better* than another merely-acceptable one, or that it should
avoid a plausible-but-harmful reply. Teaching those *preferences* — which of two responses a human
would rather receive — is a different objective, and it's the subject of Chapter 15.

---

## 🔌 How this plugs into the Storyteller

Your pretrained Storyteller (Chapters 5–11) is a *base* model — it continues stories. SFT turns it
into one that takes *requests*: fine-tune it on `(instruction, story)` pairs like
`("Write a story about a brave robot.", "Once upon a time, a little robot…")`, with the loss masked
to the story, and it learns to write on demand. Do it with LoRA and the whole fine-tune is a few MB of
adapter on top of the frozen (optionally quantized) base. The mini-project does exactly this —
instruction-tunes a GPT with LoRA — and Chapter 15 goes further, aligning it with human preferences.

---

## 🐛 Building it yourself: what trips people up

- **Not masking the prompt.** The #1 SFT bug: training on the whole conversation, so the model learns
  to generate the *user's* turn. Mask the prompt tokens to `-100`; train only the response.
- **No `<end>` token (or not training on it).** If the model never learns to stop, it rambles past
  its answer. `<end>` must be in the template *and* inside the trained (unmasked) response.
- **Forgetting to freeze the base for LoRA.** If `W` still has `requires_grad=True`, you're doing full
  fine-tuning *plus* an adapter — no savings. Freeze the base; train only `A` and `B`.
- **Initializing `B` nonzero.** Start `B` at zero so the adapter is a no-op at step 0 and training
  begins from the pretrained model. A big random `B·A` would shove the model off a cliff on step 1.
- **Rank too small (or too large).** `r` trades capacity for size. Too small and the adapter can't
  capture the task; too large and you've lost the point. `r` = 8–64 is typical.

---

## 🤔 Common questions

- **Does SFT teach the model new knowledge?** Mostly no — it teaches *behaviour* (answer, follow
  format, stop). The knowledge came from pretraining; SFT reshapes how it's expressed. (This is why a
  small, clean SFT set can have a big effect.)
- **Why mask the prompt instead of just training on the pair?** Because the objective is next-token
  prediction, and you don't want "predict the next token" to teach the model to produce *user* text.
  Masking focuses every gradient on the response.
- **How is LoRA different from just fine-tuning fewer layers?** LoRA fine-tunes *all* the layers, but
  each with a tiny low-rank *update* rather than the full matrix — so it's expressive across the whole
  model yet trains almost no parameters.
- **Does LoRA hurt quality?** Very little for most tasks — the low-rank update is enough to capture
  the behaviour shift SFT needs. Full fine-tuning can edge it out on the hardest adaptations, but LoRA
  is the default for a reason.
- **Is this really how chat models are made?** Yes — every instruct model starts with SFT on
  chat-formatted demonstrations (usually with LoRA/QLoRA at smaller scale), then adds the preference
  alignment of Chapter 15 (RLHF/DPO).
- **How much data does SFT need?** Far less than pretraining — thousands to tens of thousands of
  high-quality pairs often suffice, because you're teaching *behaviour*, not knowledge. Quality and
  diversity matter more than raw quantity (a lesson the field learned the hard way).
- **What exactly is `-100`?** PyTorch's default `ignore_index` for `cross_entropy`: any target
  position set to `-100` contributes no loss and no gradient. That's the precise mechanism loss
  masking uses to skip the prompt tokens.

## ✅ Check your understanding

<details>
<summary>1. What's the difference between a base model and a chat model, and what installs it?</summary>

A **base** model continues text; a **chat/instruct** model follows instructions and stops. The
difference is a learned *behaviour*, not more knowledge, and it's installed by **fine-tuning** on
demonstrations — starting with **SFT** on (instruction, response) pairs.
</details>

<details>
<summary>2. What is loss masking in SFT, and why is it essential?</summary>

Computing the training loss only on the **response** tokens (the prompt tokens' targets are set to
`-100` / ignored). Essential because otherwise the model would learn to generate the *user's* words,
not just the answer — every gradient should teach it to respond.
</details>

<details>
<summary>3. Write the LoRA update, and say which parts are trained.</summary>

`output = W·x + (B·A)·x`, with `A` shape `r×k`, `B` shape `d×r`, rank `r ≪ d,k`. `W` is **frozen**;
only the small `A` and `B` are trained (`B` initialized to zero so the adapter starts as a no-op).
</details>

<details>
<summary>4. Why does LoRA train so few parameters, and why does that still work?</summary>

`B·A` has `r·(d+k)` parameters instead of `d·k` — a tiny fraction for large layers (well under 1% of
a 7B model). It works because of the **low-rank hypothesis**: the *change* fine-tuning needs is
simple enough to be captured by a low-rank matrix.
</details>

<details>
<summary>5. What is QLoRA, and what does it enable?</summary>

LoRA adapters trained on top of a **4-bit quantized** (Chapter 13) frozen base. The base barely uses
memory and never updates; only the tiny adapters learn — letting you fine-tune very large models
(e.g. 65B) on a single GPU.
</details>

## 🎓 Key takeaways

- Pretraining makes a **base** model (continues text); **SFT** makes an **instruct/chat** model — the
  same next-token loss, on **(instruction, response)** pairs in a **chat template**.
- **Loss masking** (`ignore_index=-100` on the prompt) is the key: train only on the **response**, so
  the model learns to answer, not to parrot the user.
- **Full fine-tuning** is expensive (weights + grads + optimizer state, and a full copy per task); 
  **PEFT** freezes the base and trains a few new parameters.
- **LoRA** freezes `W` and learns a low-rank `B·A` beside it — **< 1%** of the parameters, no
  inference cost after merging, tiny swappable adapters — thanks to the **low-rank hypothesis**.
- **QLoRA** (LoRA on a 4-bit base) makes fine-tuning huge models affordable, tying together Chapters
  13 and 14.

## 📖 New vocabulary

`base model` vs `instruct` / `chat model` · `supervised fine-tuning (SFT)` · `chat template` ·
`role / special tokens` · `loss masking` · `ignore_index` · `full fine-tuning` · `PEFT` · `LoRA` ·
`low-rank update` · `rank r` · `adapter` · `merging` · `low-rank hypothesis` · `QLoRA`.

## 🧪 Practice & build

1. **The notebook** ([Colab](https://colab.research.google.com/github/ImAlno/Understanding-LLMs/blob/main/chapters/14-finetuning-sft/code/explore.ipynb))
   — see the base model ignore you, build the loss mask, implement LoRA, SFT a GPT. Do this first.
2. **Exercises** — [`exercises/`](./exercises/): build the loss mask, implement `LoRALinear`, measure
   LoRA's parameter savings across sizes, and show LoRA fits a low-rank target. Tiered hints +
   solutions; a starter for the first.
3. **Mini-project** — [`project/`](./project/): **"Instruction-Tune Your Storyteller"** — SFT a GPT
   with **LoRA** + loss masking on instruction pairs, and watch it go from babble to obedient (with
   only a sliver of the parameters trained).

## 🔗 Go deeper (optional)

- 📄 [Hu et al. (2021), *LoRA*](https://arxiv.org/abs/2106.09685) — the low-rank adaptation paper.
- 📄 [Dettmers et al. (2023), *QLoRA*](https://arxiv.org/abs/2305.14314) — 4-bit base + LoRA adapters.
- 📄 [HuggingFace — *Chat Templates*](https://huggingface.co/docs/transformers/main/en/chat_templating)
  — how role tokens structure real conversations.

---

| ⬅️ Previous | Up | Next ➡️ |
|:--|:-:|--:|
| [Chapter 13 — Quantization](../13-inference-quantization/) | [Syllabus](../../README.md#-syllabus) | [Chapter 15 — Finetuning: RL](../15-finetuning-rl/) |
