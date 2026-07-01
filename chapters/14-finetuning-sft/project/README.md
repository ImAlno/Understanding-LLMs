# 🎓 Mini-Project — Instruction-Tune Your Storyteller

You've built SFT and LoRA separately; now **combine them**. Freeze a GPT, bolt **LoRA adapters** onto
its Linear layers, and **SFT** it (with loss masking) into an instruction-follower — training only a
sliver of the parameters while the base never moves.

> 💻 **No GPU needed** — a tiny GPT learns the instruction task on a CPU in seconds.

> **How this works:** [`starter/instruction_tune.py`](./starter/instruction_tune.py) freezes the
> model, runs the loss-masked SFT loop, and prints the before/after. It has **one TODO** — `apply_lora`,
> which swaps the model's `nn.Linear` layers for `LoRALinear` adapters. A reference is in
> [`solution/instruction_tune.py`](./solution/instruction_tune.py).

## 🎯 What it does

```bash
python starter/instruction_tune.py
```

Freezes the whole GPT, applies LoRA (so only the tiny adapters are trainable), SFTs it on the
instruction pairs with the loss masked to the response, and generates before and after — turning
babble into correct answers, with the base model untouched.

## 🛠️ Your TODO

Implement `apply_lora` — replace every `nn.Linear` with a `LoRALinear`:

```python
to_wrap = [(parent, name, child)
           for parent in list(model.modules())
           for name, child in parent.named_children()
           if isinstance(child, nn.Linear)]
for parent, name, child in to_wrap:
    setattr(parent, name, LoRALinear(child, r))
```

**Note the `list(...)` and collect-then-replace**: mutating the module tree *while* iterating it
recurses forever. Until you fill this in, the model has no trainable parameters and the script stops.

## ✅ Checking your work

- **Trainable count:** with LoRA applied, only the adapters train (~15% *on this tiny model*; well
  under 1% on a real one — the savings grow with size).
- **Before → after:** the responses go from babble to the correct answers (`'hi' -> 'hello!'` …),
  reaching 5/5, with the base weights frozen the whole time.
- Compare against [`solution/instruction_tune.py`](./solution/instruction_tune.py).

## 🚀 Stretch

- **Add your own instructions.** Extend `PAIRS` in [`../code/gpt.py`](../code/gpt.py) with new
  (instruction, response) pairs and re-tune — watch the adapter pick them up.
- **Merge the adapter.** After training, fold `B·A` back into each frozen weight (`W + B·A`) so the
  tuned model is a plain GPT again — zero inference overhead.
- **Compare to full fine-tuning.** Train the same task with all weights trainable and compare final
  loss and parameter counts — LoRA should match it at a fraction of the trainable params.
- **QLoRA in spirit.** Quantize the frozen base (Chapter 13's per-channel int8) before applying LoRA,
  and confirm the adapters still learn on top of the quantized weights.

Next: [Chapter 15 — Finetuning II: RL](../../15-finetuning-rl/), where we go past *imitating*
demonstrations to *aligning* the model with human preferences (RLHF, DPO). 🎓
