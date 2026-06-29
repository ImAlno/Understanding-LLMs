"""
E04 — Warmup + cosine learning-rate schedule.
=============================================
Ramp the LR up over a short warmup, then anneal it to ~0 along a cosine curve. Big steps early,
tiny steps late.

Run:  python e04_schedule.py
"""
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))
import torch.nn.functional as F
from optimizers import make_moons, fresh_model, Adam

X, y = make_moons()

def lr_at(step, base, warmup, total):
    if step < warmup:
        return base * step / warmup                          # linear warmup
    progress = (step - warmup) / (total - warmup)
    return 0.5 * base * (1 + math.cos(math.pi * progress))   # cosine decay to 0

def train(use_schedule, base_lr=0.05, steps=300, warmup=30):
    model = fresh_model(); params = list(model.parameters()); opt = Adam(lr=base_lr)
    for s in range(steps):
        if use_schedule:
            opt.lr = lr_at(s, base_lr, warmup, steps)
        loss = F.cross_entropy(model(X), y); model.zero_grad(); loss.backward()
        opt.step(params, [p.grad for p in params])
    return loss.item()

print("Adam, 300 steps, constant LR vs warmup+cosine:")
print(f"  constant lr     final {train(False):.4f}")
print(f"  warmup+cosine   final {train(True):.4f}")
print("\nThe LR curve (base 0.05, warmup 30 of 300 steps):")
for s in [0, 15, 30, 100, 200, 299]:
    print(f"  step {s:3}: lr {lr_at(s, 0.05, 30, 300):.4f}")
print("\nNote the schedule doesn't beat a well-tuned constant LR here — this task is too easy, so")
print("constant already nails it. The schedule's real payoff is STABILITY on big models: warmup")
print("stops early divergence, and cosine decay fine-tunes late where a constant LR overshoots.")
