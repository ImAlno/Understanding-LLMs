# Chapter 7 — Exercises

Feel each knob of training. **Try first**; solutions are in [`solutions/`](./solutions/). All
import the optimizers and the two-moons task from [`../code/optimizers.py`](../code/optimizers.py),
so run from anywhere:

```bash
python chapters/07-optimization/exercises/solutions/e01_fix_init.py
```

Each trains tiny networks — a second or two.

---

### E01 — Fix a broken initialization 🥇
A model whose output layer is scaled up starts **confidently wrong** (huge loss at init). Pick a
sane scale so it starts near −log(1/2) ≈ 0.69, and watch it both *start* and *end* better.

- 🧩 **Scaffold:** [`starter/e01_fix_init.py`](./starter/e01_fix_init.py).
- ✅ **Solution:** [`solutions/e01_fix_init.py`](./solutions/e01_fix_init.py).

  <details><summary>💡 Hint</summary>

  The fix is to **not** blow up the output layer — leave PyTorch's default (scale ≈ 1.0). The
  bad model multiplied its last layer's weights by 20; a sane init keeps them small.
  </details>

### E02 — Sweep the learning rate
Train Adam across learning rates from `1e-4` to `10` and watch the band where it works: too small
crawls, too big diverges.

- ✅ Solution: [`solutions/e02_lr_sweep.py`](./solutions/e02_lr_sweep.py) — the sweet spot is ~0.01–0.1.

### E03 — Momentum's `beta`
Sweep momentum's `beta` (0 → 0.99) and see how much past velocity to keep. Higher descends
faster here; on harder problems too-high overshoots.

- ✅ Solution: [`solutions/e03_momentum_beta.py`](./solutions/e03_momentum_beta.py).

### E04 — Add a warmup + cosine schedule
Implement `lr_at(step)` — linear warmup then cosine decay — and apply it during training.

- 🧩 **Scaffold:** [`starter/e04_schedule.py`](./starter/e04_schedule.py) (fill in `lr_at`).
- **Heads up — the result is the lesson:** on this easy task the schedule *doesn't* beat a
  well-tuned constant LR (constant already nails it). Its real payoff is **stability on big
  models** (warmup stops early divergence; decay fine-tunes late). The LR *curve* is the thing to
  see — and you'll wire exactly this into your GPT.
- ✅ Solution: [`solutions/e04_schedule.py`](./solutions/e04_schedule.py).

> 🧠 **Takeaway:** init sets the starting point, the optimizer sets the step, the schedule shapes
> the step over time — and the learning rate is the knob that matters most.
