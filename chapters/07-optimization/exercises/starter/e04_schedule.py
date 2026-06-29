"""
E04 — Warmup + cosine schedule (STARTER scaffold).
==================================================
Fill in `lr_at` — linear warmup, then cosine decay to ~0. Reference: ../solutions/e04_schedule.py.

Run:  python e04_schedule.py
"""
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))
import torch.nn.functional as F
from optimizers import make_moons, fresh_model, Adam

X, y = make_moons()

def lr_at(step, base, warmup, total):
    # ✍️ TODO: for the first `warmup` steps, ramp the LR linearly 0 -> base.
    #          after that, cosine-decay it from base down to 0 (the formula is decoded in lesson §7;
    #          progress = how far you are through the post-warmup phase, from 0 to 1).
    return base      # replace

# sanity: a correct schedule starts at ~0 (warmup) and ends at ~0 (cosine)
if lr_at(0, 0.05, 30, 300) > 1e-3 or lr_at(299, 0.05, 30, 300) > 1e-3:
    raise SystemExit("lr_at should be ~0 at step 0 and ~0 at the last step — fill the TODO, then run again.")

def train(use_schedule, base_lr=0.05, steps=300, warmup=30):
    model = fresh_model(); params = list(model.parameters()); opt = Adam(lr=base_lr)
    for s in range(steps):
        if use_schedule:
            opt.lr = lr_at(s, base_lr, warmup, steps)
        loss = F.cross_entropy(model(X), y); model.zero_grad(); loss.backward()
        opt.step(params, [p.grad for p in params])
    return loss.item()

print(f"constant lr     final {train(False):.4f}")
print(f"warmup+cosine   final {train(True):.4f}")
print("\nyour LR curve (base 0.05, warmup 30 of 300):")
for s in [0, 15, 30, 100, 200, 299]:
    print(f"  step {s:3}: lr {lr_at(s, 0.05, 30, 300):.4f}")
