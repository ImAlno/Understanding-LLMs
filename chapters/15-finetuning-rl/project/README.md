# 🎓 Mini-Project — Align Your Storyteller with DPO

The capstone of the finetuning arc. Take an SFT model that defaults to **curt** replies and align it to
a *preference* — warmth — with **DPO**: one supervised loss on `(prompt, chosen, rejected)` triples,
the reference frozen the whole time. No reward model, no RL loop.

> 💻 **No GPU needed** — the whole alignment runs on a CPU in seconds.

> **How this works:** [`starter/align_storyteller.py`](./starter/align_storyteller.py) trains the curt
> reference (Chapter 14's SFT), freezes it, runs the DPO loop, and prints the before/after. It has **one
> TODO** — the `dpo_loss`. A reference is in [`solution/align_storyteller.py`](./solution/align_storyteller.py).

## 🎯 What it does

```bash
python starter/align_storyteller.py
```

Trains a reference that leans curt (`'hi' → 'what do you want'`), copies it into a policy, freezes the
reference, and runs DPO on the preference pairs — turning every reply warm (`'hi' → 'hello there'`)
while the base model never moves.

## 🛠️ Your TODO

Implement `dpo_loss` — the Bradley-Terry loss on each reply's **implicit reward** `β·(logp_policy − logp_ref)`:

```python
return -F.logsigmoid(BETA * ((pol_c - ref_c) - (pol_r - ref_r)))
```

`pol_c`/`pol_r` are the policy's log-probs of the chosen/rejected reply; `ref_c`/`ref_r` are the frozen
reference's. Until you fill it in, `dpo_loss` returns `None` and the script stops.

## ✅ Checking your work

- **Before → after:** the curt replies (`'what do you want'`) become the warm ones (`'hello there'`),
  reaching **5/5**.
- **The reference never moves** — DPO measures the policy's *drift* against it, so it must stay frozen
  (`requires_grad = False` + `torch.no_grad()`). Exercise 4 checks this explicitly.
- Compare against [`solution/align_storyteller.py`](./solution/align_storyteller.py).

## 🚀 Stretch

- **Your own preference.** Rewrite `PREFS` in [`../code/gpt.py`](../code/gpt.py) for a different tone —
  concise vs. rambling, formal vs. casual — and re-align. The mechanism doesn't care what "better" means.
- **Do it the RLHF way.** Align the same model with [`../code/rlhf.py`](../code/rlhf.py)'s reward model +
  policy gradient and confirm you land in the same place — the long way round.
- **Feel the β leash.** Drop `BETA` to `0.05` and raise the learning rate: the preference margin keeps
  climbing while the replies decay into mush (this is Exercise 3). Then turn β back up and watch it heal.
- **A real preference set.** Generate two stories per prompt with your Storyteller, hand-pick the one you
  prefer as `chosen`, and DPO on a few dozen such pairs — the exact recipe used on real models.

Next: [Chapter 16 — Deployment](../../16-deployment/), where the aligned Storyteller becomes a web app
you can actually talk to. 🎓
