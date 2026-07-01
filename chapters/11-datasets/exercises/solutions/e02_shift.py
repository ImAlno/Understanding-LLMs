"""
E02 — x and y are next-token pairs.
===================================
The whole training signal is "predict the next token." This makes the x->y shift visible: decode a
window and line up x above y so you can see every x character points at the next one, which is y.

Run:  python e02_shift.py
"""
import sys
import os
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))
import numpy as np
import torch
from synth_stories import generate
from prepare import prepare

D = tempfile.mkdtemp()
prepare(generate(500, seed=0), D)
import json
itos = {int(i): c for i, c in json.load(open(os.path.join(D, "meta.json")))["itos"].items()}
decode = lambda t: "".join(itos[int(i)] for i in t)

data = np.memmap(os.path.join(D, "train.bin"), dtype=np.uint16, mode="r")
i, block = 100, 24
x = torch.from_numpy(data[i:i + block].astype(np.int64))
y = torch.from_numpy(data[i + 1:i + 1 + block].astype(np.int64))

print("x:  " + " ".join(decode([t]) for t in x))
print("y:  " + " ".join(decode([t]) for t in y))
print("\nEach column: the model sees the x char and must predict the y char below it.")
print(f'e.g. after "{decode(x[:5].tolist())}" it should predict "{decode([y[4]])}"')
assert decode(x[1:].tolist()) == decode(y[:-1].tolist())    # y is x shifted by one
print("✓ y is exactly x shifted one token right — verified.")
