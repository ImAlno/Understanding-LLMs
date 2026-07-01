# Chapter 15 — Finetuning II: RL — RLHF, PPO & DPO

> Chapter 14's SFT taught the model to *imitate* good answers. But imitation has a ceiling: a model
> trained to copy demonstrations has no notion that one answer is **better** than another — only that
> both are "the kind of thing a demonstration looks like." This chapter installs *preferences*: we
> teach the model that people like *this* reply more than *that* one. First the classic **RLHF**
> pipeline — a **reward model** that turns human comparisons into a score, then **reinforcement
> learning** (REINFORCE / PPO) to maximize it, held on a **KL leash** so the model doesn't cheat. Then
> the modern shortcut, **DPO**, which reaches the same place with a single supervised loss on
> preference pairs — no reward model, no RL loop. We build all of it from scratch and watch a curt
> little model learn to be warm.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ImAlno/Understanding-LLMs/blob/main/chapters/15-finetuning-rl/code/explore.ipynb)

> 💻 **No GPU needed.** Everything here — the reward model, the RL loop, DPO — trains on a handful of
> preference pairs on a CPU in seconds. The *mechanism* is exactly what aligns a real chat model; only
> the scale differs.

**You will be able to:**
- explain why **imitation (SFT) isn't enough** and what a **preference** signal adds;
- format **preference data** — `(prompt, chosen, rejected)` triples — and train a **reward model** with
  the **Bradley-Terry** loss so it scores chosen above rejected;
- run the **RLHF** loop: sample, score with the reward model, and update with the **REINFORCE** policy
  gradient, kept honest by a **KL penalty** — and explain what **PPO** adds on top;
- implement **DPO** from scratch (the implicit reward `β·log(π/π_ref)`, the preference loss) and say
  why it needs no reward model and no sampling;
- reason about **reward hacking**, over-optimization, and the other ways alignment goes wrong.

**Prerequisites:** SFT, the chat template, and loss masking (Chapter 14); the training loop and
`cross_entropy` (Chapters 3–5); `log_softmax` and sampling (Chapters 1, 5).

**Time:** ~3 hours. **Hardware:** any laptop CPU.

> ### 📓 Build it yourself in the notebook
> **[`code/explore.ipynb`](./code/explore.ipynb)**: watch the SFT model default to curt replies, train a
> reward model, run the RLHF policy gradient, then implement the DPO loss and warm the model up — all on
> a CPU. "✍️ Your turn", "▶️ Run this", "▶️ Check your work" cells. Watch for: ✏️ **In the notebook → Step N**.

---

## 0. Setup

Four small scripts, all CPU, all self-contained (each builds on the one before):

- [`code/gpt.py`](./code/gpt.py) — the compact GPT, the **reward model**, the preference data, and the
  helpers (`seq_logprob`, `generate`, `train_reference`). Import from here.
- [`code/reward_model.py`](./code/reward_model.py) — train a reward model with the Bradley-Terry loss.
- [`code/rlhf.py`](./code/rlhf.py) — the classic RLHF loop: REINFORCE against the reward model + a KL leash.
- [`code/dpo.py`](./code/dpo.py) — DPO: the same alignment with one supervised loss, no reward model.

```bash
python chapters/15-finetuning-rl/code/reward_model.py   # comparisons -> a score
python chapters/15-finetuning-rl/code/dpo.py            # curt -> warm, no RL loop
python chapters/15-finetuning-rl/code/rlhf.py           # curt -> warm, the classic way (~20s)
```

---

## A 2-minute primer: from imitation to preference

Chapter 14 gave us the three-stage recipe for a chat model: **pretrain** (learn language), **SFT**
(learn to follow instructions by imitating demonstrations), **align** (learn human *preferences*). We
did the first two. This chapter is the third.

Why isn't SFT enough? Because SFT is **imitation**, and imitation copies the *form* of good answers
without any sense of *ranking*. Show an SFT model a demonstration and its loss says "make this exact
text more likely." It can't say "this answer is helpful and that one is subtly worse"; it has never
seen a *worse* answer to push away from. Real preferences are comparative and fuzzy — "I'd rather read
*this* story," "*that* reply is more helpful," "*this* refusal is appropriate." You can't write those
down as demonstrations to copy. But you *can* collect **comparisons**: show a human two replies and ask
which is better. Alignment is the craft of turning those comparisons into a better model.

There are two ways to do it, and we build both:

1. **RLHF** (Reinforcement Learning from Human Feedback) — the original ChatGPT recipe. Train a
   **reward model** to predict which reply a human prefers, then use **reinforcement learning** to
   optimize the model against that reward.
2. **DPO** (Direct Preference Optimization) — a 2023 shortcut that skips the reward model *and* the RL,
   reaching the same objective with a single loss you train like any other. It's now the default for
   open models because it's so much simpler.

Our running example is deliberately tiny and legible. Five prompts, each with a **warm** reply we
prefer and a **curt** reply we don't — the same length, so the preference is about *tone*, not length:

```
  prompt "hi"      chosen "hello there"        rejected "what do you want"
  prompt "thanks"  chosen "my pleasure"        rejected "yeah sure fine"
  ...
```

We start from an SFT model that *leans curt* (it was trained on both, weighted toward the curt replies)
— so "before alignment" it greedily answers `"hi" → "what do you want"`. Every method below will warm
it up to `"hi" → "hello there"`.

---

## 1. Preference data: chosen vs. rejected

> ✏️ **In the notebook → Step 1.** Meet the curt reference.

SFT data was `(instruction, response)`. Preference data is a **triple**:

```
  (prompt, chosen, rejected)
```

a prompt, a response a human **preferred**, and one they **rejected**. That's it — no scores, no
labels, just "the first is better than the second." This is exactly how real preference datasets are
built: show a labeler (or a stronger model) two candidate replies and record which wins. Millions of
these comparisons, across every kind of prompt, are what teach a model to be helpful, honest, and
harmless in the way you actually want — not just to mimic the average of its training text.

Each response still lives in Chapter 14's chat template, `<user> prompt <assistant> response <end>`,
and we still only ever score the **response** tokens. The one new primitive we need is a function that
asks a model, *"how much probability do you place on producing this response?"* — the summed log-prob
of the response tokens:

```python
def seq_logprob(model, prompt, response, stoi):   # from gpt.py
    ...                                            # sum of log p(response token | everything before it)
    return tok_logp[a_pos:].sum()                  # over the response + <end>
```

`seq_logprob` is the workhorse of the whole chapter: both DPO and the RL policy gradient are built out
of it. Higher `seq_logprob(model, "hi", "hello there")` means the model is more likely to actually
*say* "hello there" when greeted.

---

## 2. The reward model: turn comparisons into a number

> ✏️ **In the notebook → Step 2.** Train it with the Bradley-Terry loss.

Reinforcement learning needs a **reward** — a number to go up. But humans gave us comparisons, not
numbers. The **reward model** bridges the gap: it's a network that reads a `(prompt, response)` and
outputs a single scalar "how good is this reply?" We train it so that, for every preference pair, it
scores the chosen reply **above** the rejected one. Why train a whole *model* rather than just keep the
comparisons in a table? Because RL will make the policy invent replies that **aren't in your dataset** — so
the reward model has to *generalize* your handful of comparisons to score whatever the policy dreams up.

Architecturally it's our GPT with the vocabulary head swapped for a **scalar head**: run the transformer,
take the **last token's** hidden state (it has attended to the whole prompt and response), and project
it to one number.

```python
class RewardModel(nn.Module):     # from gpt.py
    ...
    def forward(self, idx):
        return self.value(self.body(idx))[:, -1, 0]   # one scalar reward per sequence
```

How do we train it with only "chosen > rejected"? The **Bradley-Terry** model — the classic recipe for
turning pairwise "A beats B" comparisons into scores. If the reward model
gives the chosen reply score `s_c` and the rejected reply score `s_r`, we *define* the probability that
it agrees with the human as `sigmoid(s_c − s_r)`. Maximizing that agreement is minimizing:

```
  loss = −log sigmoid(s_c − s_r)
```

Look closely: this is the same `−log sigmoid(difference)` shape as logistic regression. It doesn't care
about the *absolute* scores — only that the chosen score sits above the rejected one. The bigger the
gap, the smaller the loss. Run [`reward_model.py`](./code/reward_model.py) and the model obliges:

```
  'hi':      chosen 'hello there'      -> +5.38   |   rejected 'what do you want'  -> -5.30   (margin +10.68)
  'thanks':  chosen 'my pleasure'      -> +5.42   |   rejected 'yeah sure fine'    -> -5.33   (margin +10.75)
  ...
accuracy (chosen scored above rejected): 100%   |   mean margin: +10.74
```

The reward model **was never told those numbers** — `+5.38`, `−5.30`. It invented a scoring scale from
scratch whose only job is to rank chosen above rejected, and it separated them by a comfortable ~10.7
margin. Now RL has something to climb.

> **A quiet warning we'll cash in later.** This reward model is a *learned proxy* for human preference,
> not the real thing. It's accurate on replies that look like its training pairs — and unreliable on
> the weird stuff a policy might invent while exploring. Optimizing too hard against a proxy is how
> alignment breaks (see the pitfalls below).

---

## 3. RLHF: optimize the policy with reinforcement learning

> ✏️ **In the notebook → Step 3.** Run the policy gradient and watch the reward climb.

Now the "RL" in RLHF — and a genuine change of paradigm. Every loss so far was **supervised**: here is the
exact target, make it more likely. Reinforcement learning is different — *try* something, receive a scalar
**score**, and do more of what scored well. You're rewarded for a *behaviour*, not handed a target to copy.

We have a **policy** (our SFT model) and a **reward model**, and we want the policy's replies to score
higher. The catch that forces us out of supervised learning: the reward grades *generated* text, and
generation is a chain of **discrete** picks — each token is an `argmax`/`multinomial` choice, which has no
gradient — and the reward model then scores the finished *string*, a black box. There's no smooth path from
the weights to the reward to differentiate, so you can't just ask "nudge the weights to raise this score."
The classic tool for exactly this situation is the **policy gradient**.

The core idea (**REINFORCE**) is beautifully simple. Sample a reply, score it, then:

```
  push logp_policy(reply) UP in proportion to the reply's reward
```

Good replies (high reward) get their probability increased; bad replies get decreased. In one line, the
per-sample loss to *minimize* is `−reward × logp_policy(reply)`. That's it — you're doing weighted
maximum likelihood, where the weights are rewards. (The sleight of hand: we never differentiate the
*sampling* step — we sample a reply first, treat it as fixed, then differentiate its **log-probability**,
which is perfectly smooth. That's why `seq_logprob` is differentiable even though generation is not.) Two
refinements make it actually work, and both are load-bearing:

- **A baseline.** Reinforcing raw reward is noisy — every reply gets pushed up, just by different
  amounts. Instead we subtract a **baseline** (the running average reward) and reinforce the
  *advantage* = reward − baseline. Now above-average replies go up and below-average ones go down. Same
  average behaviour, far less variance.
- **A KL penalty.** Left alone, the policy will chase reward straight off a cliff — drifting into weird
  text that happens to score high on the reward model, which is only a proxy (recall the warning above). So we add a
  penalty for straying from the frozen reference: the **KL divergence** `KL(policy ‖ reference)`
  (Kullback–Leibler — a standard measure of how far one probability distribution has drifted from another).
  It's a **leash**. On each
  sampled reply the penalty is just `logp_policy(reply) − logp_reference(reply)` — how much more the
  policy likes this reply than the original model did.

Put together, for each sampled reply the loss is:

```python
advantage = reward - baseline                 # reinforce reward ABOVE average
kl        = logp_policy - logp_reference       # how far the policy has drifted on this reply
loss      = -advantage * logp_policy + KL_COEF * kl
```

Run [`rlhf.py`](./code/rlhf.py). The policy samples replies, the reward model grades them, and the mean
reward climbs from negative to the reward model's ceiling:

```
  step   0:  mean reward -2.43
  step  20:  mean reward +5.44
  step  40:  mean reward +5.44
  step  60:  mean reward +4.03
  step  80:  mean reward +4.73
  step  99:  mean reward +5.44

Before -> after RLHF (greedy):
  'hi':   'what do you want'  -> 'hello there'       ✓
  'bye':  'finally you leave' -> 'take good care'    ✓
  ...
5/5 replies became warm
```

(The climb is a little jittery — RL is noisy, and you'll see the mean reward wobble before it settles.
That's normal.) The curt model became a warm one *without ever being shown which reply to give* — it
discovered the warm replies by sampling, and the reward model's thumbs-up did the rest.

**What about PPO?** REINFORCE is the *idea*; **PPO** (Proximal Policy Optimization) is the industrial
version, and it's what InstructGPT and ChatGPT actually used. PPO keeps everything above and adds two
things: (1) a **clipped objective** that refuses to move the policy too far in a single update (big
updates on a noisy gradient are how RL training explodes), and (2) a learned **value network** as a
smarter, per-token baseline. Same skeleton — sample, score, policy-gradient, KL leash — with guardrails
that make it stable at scale. Our REINFORCE-plus-baseline is that skeleton, and it's plenty to see the
mechanism.

---

## 4. DPO: preference alignment without a reward model or RL

> ✏️ **In the notebook → Step 4.** Implement the DPO loss — the star of the chapter.

Step back and count the moving parts in RLHF: a reward model (a whole second network, with its own
training and its own failure modes), a sampling loop, a policy gradient, a baseline, a KL controller.
It works, but it's a lot of machinery to keep balanced. In 2023, **DPO** showed you can throw almost
all of it away.

Here's the trick. RLHF's real objective is "maximize reward while staying close (in KL) to the
reference." That objective has a *known* optimal policy — one can show it is:

```
  π*(reply | prompt)  ∝  π_ref(reply | prompt) · exp( reward(prompt, reply) / β )
```

That `exp` shape is the **softmax/temperature** form from Chapter 5's sampling in an RL costume: the best
policy makes a reply *exponentially* more likely the higher its reward — but tethered to `π_ref`, with `β`
a temperature-like knob (we unpack it fully below) setting how sharply reward bends the policy off the
reference. Rearrange that for the reward, and something remarkable falls out — the reward is **hidden inside the
policy itself**:

```
  reward(prompt, reply)  =  β · log( π*(reply|prompt) / π_ref(reply|prompt) )  +  (a constant)
```

So we don't need a separate reward model at all. Any policy *implies* a reward: its **implicit reward**
is `β · (logp_policy − logp_reference)` — how much more probability it puts on a reply than the
reference does. DPO says: take that implicit reward and train it with the **Bradley-Terry loss from the
reward model** — the exact same `−log sigmoid(chosen − rejected)`, but on implicit rewards instead of a reward
model's scores (the pesky constant cancels in the subtraction):

```python
def dpo_loss(policy, ref, prompt, chosen, rejected, stoi):   # from dpo.py
    pol_c, pol_r = seq_logprob(policy, prompt, chosen, ...), seq_logprob(policy, prompt, rejected, ...)
    with torch.no_grad():                                     # reference is frozen
        ref_c, ref_r = seq_logprob(ref, prompt, chosen, ...), seq_logprob(ref, prompt, rejected, ...)
    chosen_reward   = BETA * (pol_c - ref_c)                  # implicit reward, chosen
    rejected_reward = BETA * (pol_r - ref_r)                  # implicit reward, rejected
    return -F.logsigmoid(chosen_reward - rejected_reward)
```

That's the whole method. **No reward model. No sampling. No RL loop.** It's a plain supervised loss over
your preference pairs — you `.backward()` it like the cross-entropy from Chapter 3. Under the hood it
does exactly what RLHF wanted (raise chosen, lower rejected, stay near the reference), but in closed
form. Run [`dpo.py`](./code/dpo.py):

```
DPO loss after training: 0.0010   (beta = 0.3)
Before -> after alignment (greedy):
  'hi':      'what do you want'  -> 'hello there'        ✓
  'bye':     'finally you leave' -> 'take good care'     ✓
  'thanks':  'yeah sure fine'    -> 'my pleasure'        ✓
  ...
preference accuracy: 100%   |   mean implicit-reward margin: +7.90
5/5 replies became the warm (chosen) one
```

Same result as RLHF — curt to warm, 5 for 5 — from a loss with a fraction of the moving parts.

**What is `β`?** The same leash as RLHF's KL penalty, wearing a different hat. It scales the implicit
reward, which controls how far DPO will let the policy drift from the reference to satisfy the
preferences. High `β` (we use `0.3`) = stay close, move gently. Low `β` = optimize aggressively — and,
just like a too-loose KL leash in RLHF, push it too far and the model degrades (below). RLHF's `KL_COEF`
and DPO's `β` are the *same idea*: how much you trust the preference signal versus the model you
started with.

---

## 5. RLHF or DPO? 

Both align a model to preferences; they trade off differently.

| | **RLHF (reward model + PPO)** | **DPO** |
|---|---|---|
| Moving parts | reward model + sampling + PPO | one supervised loss |
| Training signal | reward model's score on *fresh* samples | fixed `(chosen, rejected)` pairs |
| Stability | finicky; many knobs | trains like SFT |
| Can improve past the data? | yes — explores new replies and scores them | no — only reweights the pairs it's given |
| Used by | InstructGPT, ChatGPT, Claude | most open models (Llama, Mistral, Zephyr…) |

The key structural difference: **RLHF explores.** Because it samples fresh replies and scores them with
a reward model, it can discover and reinforce a *good reply it was never shown*. DPO only ever sees your
`(chosen, rejected)` pairs, so it can only reweight those — it can't invent a better answer than the
best one in your data. That extra reach is exactly why frontier labs still invest in RL pipelines
(often with newer estimators like **GRPO**, which drops the value network and normalizes rewards across
a group of samples). But for most fine-tuning, DPO's simplicity wins: no reward model to train and
babysit, no RL instability, and it reliably gets you most of the way there.

---

## 6. Alignment and its pitfalls

Getting preference optimization to *run* is the easy part. Getting it to produce a genuinely better
model — without new problems — is the hard part, and it's where most of the real research lives.

- **Reward hacking.** The reward model is a *proxy*. Optimize it hard enough and the policy stops
  producing good replies and starts producing whatever the proxy scores highly — text that games the
  reward model without being good. It's [Goodhart's law](https://en.wikipedia.org/wiki/Goodhart%27s_law):
  when a measure becomes a target, it stops being a good measure. The **KL penalty is the main defense** —
  it keeps the policy near the reference, out of the weird regions where the proxy is unreliable. (Our toy
  is actually too *clean* to hack: with only five short prompts the warm replies genuinely *are* the reward
  model's maximum, so `rlhf.py` converges to them even with `KL_COEF = 0`. Hacking bites when the reward
  model is an imperfect proxy over a vast output space — every real one — where high-scoring garbage exists
  to be found.)
- **Over-optimization (yes, DPO too).** DPO has no reward model to hack, but it has the same disease. Its
  loss can drive the chosen-vs-rejected *margin* up by crushing the rejected reply's probability so hard
  it distorts the whole distribution — the margin looks great while generations turn to mush. (Drop
  `dpo.py`'s `β` to `0.05` and raise its learning rate, and `'hello there'` over-optimizes into mush like
  `'hello the t   t te'` — margin still climbing, fluency gone; Exercise 3 makes this concrete.) A higher
  `β` — staying closer to the reference — is the fix, the same leash by
  another name.
- **The alignment tax.** Aligning a model can make it a little worse at raw capabilities (a few points on
  benchmarks). Keeping the KL penalty honest keeps the tax small.
- **Sycophancy.** If humans prefer replies that *sound* agreeable and confident, the reward model learns
  to reward agreeableness and confidence — so the model learns to tell you what you want to hear. Your
  preferences become the model's, warts and all.
- **Where the labels come from.** Human labeling is slow and expensive, so a lot of modern alignment uses
  **AI feedback**: a strong model provides the comparisons (**RLAIF**), sometimes guided by a written set
  of principles (**Constitutional AI**). The pipeline is identical — only the source of "chosen > rejected"
  changes.

None of these are solved. Alignment is the frontier where "make the number go up" meets "but is the
model actually *better*?" — and the gap between those two questions is the whole game.

---

## 🔌 How this plugs into the Storyteller

Your Storyteller can already write stories (pretraining) and take requests (SFT, Chapter 14). But
"technically a story" isn't "a story someone *enjoys*." Alignment closes that gap. Collect preference
pairs over its output — generate two stories for a prompt, keep the one you like better as `chosen`, the
other as `rejected` — and run **DPO**. With a few hundred such pairs the model learns your taste: more
vivid, better-paced, the kind of ending you prefer. The mini-project does exactly this in miniature —
DPO-aligns the model from curt to warm — and it's the same procedure, pair-for-pair, that turns a
capable base model into one people actually want to talk to.

---

## 🐛 Building it yourself: what trips people up

- **Optimizing the reward model too hard.** The reward model is a proxy, not the truth. At real scale,
  without a KL penalty the policy drifts into text that hacks the proxy — the leash is what keeps RL
  honest. (Our tiny toy is the exception — *Alignment and its pitfalls* explains why.)
- **Forgetting to freeze the reference.** Both RLHF's KL term and DPO's loss compare the policy to a
  *frozen* reference. If the reference trains too, there's nothing to measure drift against and the
  math falls apart. Freeze it (`requires_grad = False`) and wrap its `seq_logprob` in `torch.no_grad()`.
- **Starting the policy somewhere other than the reference.** The policy begins as an exact copy of the
  reference, so at step 0 the implicit reward / KL is exactly zero — alignment nudges gently from there.
  Start it elsewhere and you're fighting a needless initial gap.
- **Reading the DPO margin as quality.** A rising preference margin means chosen is beating rejected —
  *not* that generations are good. Over-optimized DPO shows a beautiful margin and mushy samples. Always
  eyeball the actual generations, not just the loss.
- **`β` / `KL_COEF` too low.** The single most common failure. Too small a leash and either method drifts
  into degenerate text. When alignment "breaks," turn the leash *up* first.

---

## 🤔 Common questions

- **Why not just SFT on the chosen replies?** You can (it's called rejection-sampling fine-tuning, and
  it helps), but it throws away half the signal — the *rejected* replies. Preference methods learn from
  the contrast "chosen **over** rejected," which is a stronger, more specific signal than "chosen is
  good." It teaches the model what to move *away* from, not just toward.
- **Is DPO really RL?** Not in the sampling-and-rewarding sense — there's no rollout, no reward model,
  no exploration. It's a supervised loss. But it provably optimizes the *same objective* RLHF does, which
  is why it belongs in this chapter. Think of it as "the RLHF objective, solved in closed form."
- **Where does the reward model's scale come from?** Nowhere in particular — it's arbitrary. Bradley-Terry
  only constrains *differences* (chosen minus rejected), so the model is free to center its scores
  anywhere. Only the gap is meaningful.
- **Does the reward model see the whole reply?** Yes — it reads the full `(prompt, response)` and scores
  from the last token's hidden state, which has attended to everything. That's why it can judge a reply
  as a whole rather than token by token.
- **Why is my RL reward so jumpy?** Because it's an average over *sampled* replies, and sampling is
  random. The baseline tames the variance but doesn't remove it. Watch the trend over many steps, not
  the step-to-step wiggle.
- **Is this how Claude / ChatGPT are made?** Yes, at the level of mechanism: SFT, then preference
  optimization (RLHF with PPO, or DPO-style losses, increasingly with AI feedback). What differs at scale
  is the *quantity* and *quality* of preferences and a great deal of engineering — not the core ideas
  you just built.

## ✅ Check your understanding

<details>
<summary>1. Why can't SFT alone install preferences?</summary>

SFT is imitation — it maximizes the likelihood of demonstrations. It only ever sees examples of *good*
answers to copy, never *worse* answers to push away from, so it has no notion of one answer being
**better** than another. Preferences are comparative ("A over B"), which imitation can't express.
</details>

<details>
<summary>2. What does the Bradley-Terry loss optimize, and what does it not care about?</summary>

It maximizes `sigmoid(s_chosen − s_rejected)` — i.e. minimizes `−log sigmoid(s_chosen − s_rejected)` —
so the chosen reply scores above the rejected one. It cares only about the **difference** in scores,
not their absolute values (the scale is arbitrary).
</details>

<details>
<summary>3. In the RLHF policy gradient, what are the baseline and the KL penalty each for?</summary>

The **baseline** (running-average reward) reduces variance: we reinforce reward *above average*
(the advantage), so good replies go up and bad ones go down. The **KL penalty** keeps the policy near
the frozen reference so it can't drift into text that **hacks** the reward-model proxy.
</details>

<details>
<summary>4. Write DPO's implicit reward, and say what DPO does with it.</summary>

The implicit reward for a reply is `β · (logp_policy(reply) − logp_reference(reply))`. DPO trains it
with the Bradley-Terry loss over preference pairs — `−log sigmoid(r̂(chosen) − r̂(rejected))` — which
optimizes the RLHF objective directly, with no reward model and no sampling.
</details>

<details>
<summary>5. RLHF and DPO both align to preferences — what can RLHF do that DPO structurally can't?</summary>

**Explore.** RLHF samples fresh replies and scores them with the reward model, so it can discover and
reinforce a good reply that was never in the data. DPO only reweights the `(chosen, rejected)` pairs it's
given, so it can't exceed the best reply in its dataset.
</details>

## 🎓 Key takeaways

- **SFT imitates; alignment ranks.** Preferences come as **comparisons** — `(prompt, chosen, rejected)` —
  because "which is better?" is answerable when "score this reply" is not.
- **RLHF = reward model + RL.** A **Bradley-Terry** reward model turns comparisons into a score; a
  **policy gradient** (REINFORCE, or **PPO** at scale) maximizes it; a **baseline** cuts variance and a
  **KL penalty** stops **reward hacking**.
- **DPO** skips both: its **implicit reward** `β·(logp_policy − logp_ref)` plugged into the Bradley-Terry
  loss optimizes the same objective as a **single supervised loss** — no reward model, no RL loop.
- **`β` (DPO) and `KL_COEF` (RLHF) are the same knob**: how far to let the policy drift from the
  reference. Too loose and both methods degrade.
- **Alignment's hard part is the pitfalls** — reward hacking, over-optimization, sycophancy, the
  alignment tax — not getting the training to run.

## 📖 New vocabulary

`preference data` · `(prompt, chosen, rejected)` · `reward model` · `Bradley-Terry loss` · `RLHF` ·
`policy` · `policy gradient` / `REINFORCE` · `baseline` / `advantage` · `KL penalty` · `PPO` ·
`reward hacking` · `DPO` · `implicit reward` · `β / KL coefficient` · `reference model` · `RLAIF` ·
`Constitutional AI` · `GRPO`.

## 🧪 Practice & build

1. **The notebook** ([Colab](https://colab.research.google.com/github/ImAlno/Understanding-LLMs/blob/main/chapters/15-finetuning-rl/code/explore.ipynb))
   — meet the curt model, train a reward model, run the RL policy gradient, implement DPO. Do this first.
2. **Exercises** — [`exercises/`](./exercises/): implement the Bradley-Terry loss, implement the DPO
   loss, watch a too-small β make DPO over-optimize, and confirm DPO leaves the reference frozen.
   Tiered hints + solutions; a starter for the first.
3. **Mini-project** — [`project/`](./project/): **"Align Your Storyteller with DPO"** — implement the
   DPO loss and turn the curt model warm, with the reference frozen the whole time.

## 🔗 Go deeper (optional)

- 📄 [Ouyang et al. (2022), *InstructGPT*](https://arxiv.org/abs/2203.02155) — the RLHF-with-PPO recipe
  behind ChatGPT.
- 📄 [Rafailov et al. (2023), *DPO*](https://arxiv.org/abs/2305.18290) — "Your Language Model is Secretly
  a Reward Model"; the implicit-reward derivation.
- 📄 [Christiano et al. (2017), *Deep RL from Human Preferences*](https://arxiv.org/abs/1706.03741) —
  where reward-model-from-comparisons began.
- 🎥 [Karpathy — Deep Dive into LLMs like ChatGPT](https://www.youtube.com/watch?v=7xTGNNLPyMI) — the
  SFT → RLHF arc in context.

---

| ⬅️ Previous | Up | Next ➡️ |
|:--|:-:|--:|
| [Chapter 14 — Finetuning: SFT](../14-finetuning-sft/) | [Syllabus](../../README.md#-syllabus) | [Chapter 16 — Deployment](../16-deployment/) |
