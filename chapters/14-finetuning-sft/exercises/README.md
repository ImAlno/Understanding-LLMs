# Chapter 14 — Exercises

Build the pieces of SFT and LoRA. **Try first**; solutions are in [`solutions/`](./solutions/).
Everything runs on a CPU in a second, importing from [`../code/`](../code/).

```bash
python chapters/14-finetuning-sft/exercises/solutions/e01_loss_mask.py
```

---

### E01 — Build the loss mask 🥇
The heart of SFT: turn a chat example into a target that trains only the response (prompt tokens →
`-100`).

- 🧩 **Scaffold:** [`starter/e01_loss_mask.py`](./starter/e01_loss_mask.py).
- ✅ **Solution:** [`solutions/e01_loss_mask.py`](./solutions/e01_loss_mask.py).

  <details><summary>💡 Hint</summary>

  `[(t if p >= a_pos else -100) for p, t in enumerate(y)]`, where `a_pos = seq.index(<assistant>)`.
  </details>

### E02 — LoRALinear: frozen base, no-op start
Verify two properties of `LoRALinear`: at init it's a **no-op** (output equals the frozen base, since
`B=0`), and **only** `A` and `B` are trainable.

- ✅ Solution: [`solutions/e02_lora_module.py`](./solutions/e02_lora_module.py).

  <details><summary>💡 Hint</summary>

  `torch.allclose(lora(x), base(x))` at init; `[n for n, p in lora.named_parameters() if p.requires_grad]`
  should be `['A', 'B']`.
  </details>

### E03 — LoRA's parameter savings
Tabulate `full = d·k` vs `LoRA = r·(d+k)` across layer sizes, and estimate a whole 7B model — the
savings grow as the layers get bigger (well under 1%).

- ✅ Solution: [`solutions/e03_param_savings.py`](./solutions/e03_param_savings.py).

  <details><summary>💡 Hint</summary>

  A `4096×4096` layer at rank 8 is `0.39%`; across a 7B model, LoRA trains ~27M of 7B params.
  </details>

### E04 — LoRA fits a low-rank target (not a full-rank one)
Show the low-rank hypothesis directly: LoRA nails a target that differs from its base by a **rank-8**
update (loss → 0), but can't fully fit a **full-rank** difference.

- ✅ Solution: [`solutions/e04_lora_fit.py`](./solutions/e04_lora_fit.py).

  <details><summary>💡 Hint</summary>

  Build the target from LoRA's *own* frozen base (`base.weight + change`), so `target - base` is
  exactly the `change` — rank-8 fits to ~0, full-rank stays high.
  </details>

> 🧠 **Takeaway:** SFT masks the loss to the response; LoRA freezes `W` and trains a tiny low-rank
> `B·A` that starts as a no-op — and it works because the change fine-tuning needs really is low-rank.
