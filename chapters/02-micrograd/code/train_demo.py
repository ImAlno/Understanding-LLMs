"""
Train a neural network with our from-scratch autograd engine.
=============================================================
This is the exact same loop as Chapter 1 — forward, backward, update — except now
NOTHING is magic: `loss.backward()` is the code we wrote in engine.py, and we can
see every gradient. Four tiny examples, a 3→4→4→1 network, and a target of +1/-1.

Run:  python train_demo.py
"""

import random
from nn import MLP

random.seed(1337)   # reproducible weights

# A tiny toy dataset: 4 inputs (3 numbers each) and their desired outputs (+1 / -1).
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]

model = MLP(3, [4, 4, 1])    # 3 inputs → 4 → 4 → 1 output
print(f"network has {len(model.parameters())} parameters\n")

for step in range(100):
    # ---- forward: predict, then measure the loss (sum of squared errors) ----
    ypred = [model(x) for x in xs]
    loss = sum((yp - yt) ** 2 for yt, yp in zip(ys, ypred))

    # ---- backward: our engine fills in every parameter's .grad ----
    model.zero_grad()
    loss.backward()

    # ---- update: nudge every parameter downhill ----
    for p in model.parameters():
        p.data += -0.05 * p.grad

    if step % 10 == 0 or step == 99:
        print(f"step {step:3d} | loss {loss.data:.6f}")

print("\ntargets:     ", ys)
print("predictions: ", [round(model(x).data, 3) for x in xs])
print("\nThe predictions should be close to the targets — your hand-built")
print("autograd engine just trained a neural network. That is all backprop is.")
