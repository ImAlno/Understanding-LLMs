# Chapter 7 — Optimization: Training That Actually Works

> Every chapter so far wrote `torch.optim.AdamW(...)` and trusted it. This chapter opens the
> box. Training a network well comes down to three choices: **where you start** (initialization),
> **how you step** (the optimizer), and **how big the steps are over time** (the learning-rate
> schedule). We'll build the optimizers from scratch — SGD, Momentum, Adam, AdamW — race them,
> and learn the two stability tricks (warmup/decay and gradient clipping) every real model uses.

**You will be able to:**
- explain why **initialization** matters (loss at init, saturated activations);
- implement **SGD**, **Momentum**, **Adam**, and **AdamW** from their update rules;
- say *why* Adam is the default — per-parameter adaptive steps + momentum;
- understand **learning-rate warmup + cosine decay** and how to find a learning rate;
- apply **gradient clipping** and **weight decay**, and state the modern training recipe.

**Prerequisites:** the training loop, gradients/backprop (Chapters 2–3), and a little comfort
with PyTorch tensors. The math is just running averages and one square root — we'll decode every
line. (You've *used* AdamW since Chapter 4; now we build it.)

**Time:** ~3 hours. **Hardware:** a laptop CPU — every demo is tiny.

> ### 📓 Build it yourself in the notebook
> **[`code/explore.ipynb`](./code/explore.ipynb)** has you see the initialization effects, then
> implement the four optimizers' update rules and race them. "✍️ Your turn", "▶️ Run this",
> "▶️ Check your work" cells. Watch for: ✏️ **In the notebook → Step N**.

---

## 0. Setup

The reference code is [`code/optimizers.py`](./code/optimizers.py) (the four optimizers + the
race) and [`code/init_demo.py`](./code/init_demo.py) (the initialization effects). We race them
on a tiny **two-moons** classification task (no dataset to download — it's generated).

```bash
uv venv --python 3.13 .venv && uv pip install -r requirements.txt
.venv/bin/jupyter lab        # open chapters/07-optimization/code/explore.ipynb
```

---

## 1. Where you start: initialization

> ✏️ **In the notebook → Step 1.** See the initialization effects.

Before the first step, the weights are random — but *how* random matters enormously. Two things
go wrong with a bad scale:

**Loss at init.** A freshly-initialized classifier should be *unsure* — it hasn't learned
anything, so it should assign roughly equal probability to each class. Recall cross-entropy is
**−ln(probability you put on the correct answer)** (here `log` means *natural* log, `ln` — what
`math.log` computes, not base-10). Guessing 1/n on the right class gives a starting loss of about
**−ln(1/n_classes)** — for 2 classes, `−ln(1/2) ≈ 0.69`. But if the output layer's weights are too
big, the model starts out wildly **confident and wrong**, and the loss is huge:

```
loss at initialization:
  good init (default)            0.81      (in the ballpark of 0.69 — appropriately unsure)
  bad init (output layer ×30)    5.51      (confidently wrong — a wasted starting point)
```

A model that starts confidently wrong spends its first many steps just **undoing** that
confidence — the famous "hockey-stick" loss curve that plummets and then crawls. Fixing init
removes the wasted steps.

**Saturated activations.** If weights are too large, the inputs to `tanh` (or sigmoid) are large,
so the activations slam to **±1** — where the function is flat and its **gradient is ≈ 0**. Those
neurons stop learning ("dead" gradients):

```
fraction of tanh activations stuck near ±1:
  good init (small weights)      0.2%
  bad init (large weights)      35.6%      (a third of the network, barely learning)
```

The fix is to **scale weights by roughly 1/√(fan-in)** (the number of inputs to a neuron) — the
idea behind **Xavier/Glorot** and **He/Kaiming** initialization. PyTorch's default `nn.Linear`
already does a sensible version, which is why our models have trained; this section is about
*seeing why it's there.*

---

## 2. How you step: SGD

> ✏️ **In the notebook → Step 2.** Implement SGD.

An **optimizer** takes the parameters and their gradients and decides how to update. The simplest
is **stochastic gradient descent**: step a little bit *downhill* (opposite the gradient).

```python
class SGD:
    def __init__(self, lr=0.1):
        self.lr = lr
    def step(self, params, grads):
        for p, g in zip(params, grads):
            p -= self.lr * g           # move opposite the gradient, scaled by the learning rate
```

That `p -= lr * g` *is* gradient descent — the same rule you wrote by hand back in Chapter 2.
It works, but on a stretched loss surface (steep one way, shallow another) it **zig-zags**: the
learning rate that's safe for the steep direction is painfully slow for the shallow one.

> 🧰 **Two PyTorch notes for the code.** (1) We pass `params` *and* `grads` in explicitly
> (`opt.step(params, [p.grad for p in params])`) so the update rule is right there in front of
> you; the real `torch.optim` hides this — its `optimizer.step()` reads each parameter's `.grad`
> for you. (2) The reference and notebook put **`@torch.no_grad()`** on `step` — it just tells
> PyTorch *not* to record these weight-update operations as part of the next backprop (the update
> isn't part of the model's math, just bookkeeping).

---

## 3. Momentum: build up speed

> ✏️ **In the notebook → Step 3.** Implement Momentum.

Momentum fixes the zig-zag by giving the optimizer **velocity**. Instead of stepping by the raw
gradient, keep a running average of recent gradients and step by *that*:

```python
self.v[i] = beta * self.v[i] + g     # velocity: mostly the old velocity, plus the new gradient
p -= self.lr * self.v[i]
```

Picture a ball rolling downhill: it doesn't stop and re-decide at every point — it **accumulates
momentum**. Zig-zags (which flip sign step to step) cancel out in the average, while the
consistent downhill direction builds up. `beta` (≈ 0.9) sets how much past velocity to keep. The
result reaches the bottom faster and smoother than plain SGD.

---

## 4. Adam: a custom step size for every parameter

> ✏️ **In the notebook → Step 4.** Implement Adam — the heart of the chapter.

Momentum uses one learning rate for everything. **Adam** goes further: it gives **each parameter
its own effective step size**, automatically shrinking the step for parameters with big, noisy
gradients and growing it for parameters with small, steady ones. It tracks **two** running
averages:

```python
self.m[i] = b1 * self.m[i] + (1 - b1) * g        # 1st moment: running average of the gradient
self.v[i] = b2 * self.v[i] + (1 - b2) * g * g    # 2nd moment: running average of the SQUARED gradient
mhat = self.m[i] / (1 - b1 ** t)                 # bias correction (see below)
vhat = self.v[i] / (1 - b2 ** t)
p -= self.lr * mhat / (vhat.sqrt() + eps)        # step = mean ÷ sqrt(average squared gradient)
```

Three ideas, decoded — and "moment" is just stats jargon: the 1st moment is the average of `g`,
the 2nd moment the average of `g²`.

- **The running averages** are *exponential moving averages*: a smoothed mean weighted toward
  recent values, where a gradient from `k` steps ago carries weight ~`b^k` — shrinking
  *exponentially*, hence the name (`b1 ≈ 0.9` keeps ~90% of the old average each step). Note Adam
  weights the **new** gradient by `(1 - b1)` — **unlike momentum's `beta*v + g`** — a difference
  worth remembering when you implement it.
- **The division `mhat / sqrt(vhat)`** is the magic, and it's **element-wise**: `m` and `v` hold
  one number *per parameter*, so every weight gets its own denominator. `sqrt(vhat)` is roughly
  "the typical size of this parameter's gradient" (a root-mean-square), so dividing by it makes a
  parameter with consistently huge gradients take *small* steps and one with tiny gradients take
  *large* steps. (`eps` just avoids dividing by zero.)
- **Bias correction** `/(1 - b**t)`: `m` and `v` start at 0, so early on they're pulled toward
  zero. Watch step 1 with `b1 = 0.9`: `m = 0.9·0 + 0.1·g = 0.1g` — only a tenth of the gradient.
  Dividing by `(1 - 0.9¹) = 0.1` gives `mhat = g` — fixed. (`1 - 0.9² = 0.19`, `1 - 0.9³ = 0.27`, …
  climbing to 1, so the correction fades as the averages warm up.) `t` is the step number.

**See it on two parameters.** Say weight A always has gradient `1.0` and weight B always `0.01`.
After a few steps `sqrt(vhat) ≈ 1.0` for A and `≈ 0.01` for B, while `mhat ≈ 1.0` and `≈ 0.01`. So
*both* take a step of about `lr · (mhat / sqrt(vhat)) ≈ lr · 1` — the same healthy size, even
though their gradients differ 100×. That self-tuning, separately for every parameter, is the whole
point — and it's why `Adam` (and `AdamW`) is the default optimizer for basically all modern deep
learning.

---

## 5. AdamW: Adam + weight decay

**Weight decay** gently pulls every weight toward zero each step — a form of regularization that
keeps weights small and improves generalization (it's *why* it helped in Chapter 3's overfitting
exercise). **AdamW** applies it **decoupled** from the gradient — a separate little nudge:

```python
p -= self.lr * self.weight_decay * p     # pull toward zero...
# ...then do the normal Adam step
```

What does "**decoupled**" mean? The *old* (coupled) way folded weight decay into the gradient —
`g = g + weight_decay * p` — *before* Adam's `÷ sqrt(vhat)` scaling. But that scaling then shrinks
the decay differently for each parameter, so the weights that most need reining in get decayed
least. AdamW instead applies the decay as its own separate step (the line above), *outside* the
adaptive scaling — so every weight is reined in equally. That's the "W," from the 2017 *AdamW*
paper. It's the optimizer you've been using since Chapter 4, and the one real LLMs train with.

---

## 6. The race

> ✏️ **In the notebook → Step 5.** Race them.

Train the same little network on two-moons with each optimizer and watch the final loss after
300 steps:

```
SGD       start 0.705 -> final 0.0110
Momentum  start 0.705 -> final 0.0037
Adam      start 0.705 -> final 0.0001
AdamW     start 0.705 -> final 0.0002
```

Same start, same network, same steps — **Adam reaches a loss ~100× lower than SGD.** That gap is
why nobody hand-tunes plain SGD anymore. (AdamW lands a touch higher than Adam here because its
weight decay deliberately trades a little training loss for smaller, better-generalizing weights.)

---

## 7. The learning-rate schedule

The single most important hyperparameter is the **learning rate**. Too small → glacial; too big
→ the loss diverges. Two refinements every real model uses:

- **Warmup.** Start the LR near zero and ramp it up over the first few hundred steps. Early on the
  gradients are large and the model is fragile; warmup avoids blowing up before Adam's running
  averages have settled.
- **Cosine decay.** After warmup, smoothly *anneal* (gradually lower) the LR down to near zero
  following a cosine curve. Big steps early to make fast progress, tiny steps late to settle into a
  good minimum.

```python
def lr_at(step, base_lr, warmup, total):
    if step < warmup:
        return base_lr * step / warmup                       # linear warmup
    progress = (step - warmup) / (total - warmup)
    return 0.5 * base_lr * (1 + math.cos(math.pi * progress)) # cosine decay to 0
```

Reading that last line: `progress` runs 0 → 1 across training, so `math.pi * progress` runs
0 → π; over that range `cos` runs 1 → −1; so `(1 + cos)` runs 2 → 0; times `0.5 * base_lr` makes
the LR trace a smooth curve from `base_lr` down to `0`. (Angles here are in **radians**, where π ≈
3.14 is half a turn — that's just the unit `math.cos` expects.)

To *find* a base LR, the practical trick is an **LR range test**: train for a bit while
exponentially increasing the LR, and pick one a notch below where the loss starts to blow up.

---

## 8. Gradient clipping

Occasionally a batch produces a freak, enormous gradient that would launch the weights into
nonsense. **Gradient clipping** caps the gradient's overall size before the step:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

If the gradient vector's length exceeds `max_norm`, it's scaled down to that length (same
direction, smaller size). One line, and it makes training of deep models far more robust to the
occasional bad batch.

---

## 🧩 The modern recipe

Put it together and you have how essentially every transformer is trained today:

> **AdamW** (β1=0.9, β2=0.95–0.999, weight decay ≈ 0.1) + **linear warmup → cosine decay** +
> **gradient clipping** at norm 1.0, on top of a **sane initialization**.

You'll wire exactly this into your GPT in the project.

---

## 🤔 Common questions

- **Why does the loss at init matter if it drops anyway?** A confident-wrong start wastes
  hundreds of steps undoing itself; a good init starts near the floor and makes real progress
  immediately. Free speed.
- **Momentum vs Adam — when does it matter?** On easy problems both work; the harder and more
  ill-conditioned the loss surface (deep transformers!), the more Adam's per-parameter scaling
  wins. That's why it's the default.
- **What's the difference between Adam and AdamW again?** AdamW applies weight decay *decoupled*
  from the gradient update. With Adam's adaptive scaling, the naive (coupled) version
  under-regularizes the parameters that need it most; decoupling fixes that.
- **Why warmup *and* decay?** Warmup protects the fragile early steps; decay lets the model take
  big steps while it can and small ones to settle. Together they reliably reach a lower loss.
- **Is this really how GPT-4 is trained?** The optimizer (AdamW), the schedule (warmup + cosine),
  the clipping — yes, exactly these, just at enormous scale.

## ✅ Check your understanding

<details>
<summary>1. Why should a well-initialized 2-class classifier start near loss 0.69?</summary>

Because it should be *unsure* — assigning ~50% to each class — and the cross-entropy of a uniform
guess over 2 classes is −log(1/2) ≈ 0.69. A much higher starting loss means the model is
confidently *wrong*, a sign of bad output-layer scaling.
</details>

<details>
<summary>2. What does momentum's velocity buy you over plain SGD?</summary>

It averages recent gradients, so zig-zags (which flip direction each step) cancel while the
consistent downhill direction accumulates — faster, smoother descent on stretched loss surfaces.
</details>

<details>
<summary>3. What are Adam's two running averages, and what does dividing by the second do?</summary>

`m` = average gradient (1st moment), `v` = average squared gradient (2nd moment). Dividing the
step by `sqrt(v)` gives each parameter its own effective learning rate — big-gradient parameters
take smaller steps, small-gradient ones take bigger steps.
</details>

<details>
<summary>4. What do warmup and gradient clipping each protect against?</summary>

Warmup protects the fragile early steps (large gradients, unsettled optimizer averages) from
blowing up. Gradient clipping protects against the occasional freak huge gradient that would
otherwise wreck the weights.
</details>

## 🎓 Key takeaways

- Good training = **init** (start near the loss floor, with healthy gradients) + **optimizer**
  (how to step) + **schedule** (step size over time).
- **SGD** steps downhill; **Momentum** adds velocity to cut zig-zags; **Adam** gives every
  parameter its own adaptive step (mean ÷ √variance) with bias correction; **AdamW** adds
  decoupled weight decay.
- Adam reaches a far lower loss than SGD for the same steps — the reason it's the default.
- **Warmup + cosine decay** and **gradient clipping** make training fast *and* stable.
- The modern recipe — AdamW + warmup/cosine + clip + sane init — is exactly how real
  transformers are trained.

## 📖 New vocabulary

`initialization` · `loss at init` · `saturated activation` · `Xavier/He init` · `optimizer` ·
`SGD` · `momentum` · `velocity` · `Adam` · `1st/2nd moment` · `exponential moving average` ·
`bias correction` · `AdamW` · `weight decay` · `learning-rate schedule` · `warmup` ·
`cosine decay` · `LR range test` · `gradient clipping`.

## 🧪 Practice & build

1. **The notebook** — [`code/explore.ipynb`](./code/explore.ipynb): see the init effects, then
   implement SGD, Momentum, and Adam, and race them.
2. **Exercises** — [`exercises/`](./exercises/): fix a broken initialization, sweep the learning
   rate, watch momentum's `beta`, and add a warmup+cosine schedule. Tiered hints + solutions; a
   starter for the first.
3. **Mini-project** — [`project/`](./project/): **"The Optimizer Showdown"** — plot the full loss
   *curves* of SGD vs Momentum vs Adam (and a learning-rate sweep) on one chart, and read off why
   Adam wins.

## 🔗 Go deeper (optional)

- 🎥 Karpathy, [*makemore Part 3: Activations & Gradients, BatchNorm*](https://www.youtube.com/watch?v=P6sfmUTpUmc)
  — the definitive walk-through of init and activation statistics.
- 📄 [Kingma & Ba (2014), *Adam*](https://arxiv.org/abs/1412.6980) · [Loshchilov & Hutter (2017), *Decoupled Weight Decay (AdamW)*](https://arxiv.org/abs/1711.05101).
- 📄 [He et al. (2015), *Kaiming init*](https://arxiv.org/abs/1502.01852) — the 1/√(fan-in) idea.

---

| ⬅️ Previous | Up | Next ➡️ |
|:--|:-:|--:|
| [Chapter 6 — Tokenization](../06-tokenization/) | [Syllabus](../../README.md#-syllabus) | [Chapter 8 — Need for Speed I: Device](../08-speed-device/) |
