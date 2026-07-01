"""
E01 — Implement get_batch (STARTER scaffold).
=============================================
Fill the TODO: build x and y from the memory-mapped tokens. Reference: ../solutions/e01_get_batch.py.

Run:  python e01_get_batch.py
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


def get_batch(split, block_size=32, batch_size=8):
    data = np.memmap(os.path.join(D, f"{split}.bin"), dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # ✍️ TODO: x = the windows data[i : i+block_size]; y = the same shifted one right
    #          data[i+1 : i+1+block_size]. Cast each to int64 and torch.stack them.
    x = None      # replace
    y = None      # replace
    return x, y


torch.manual_seed(0)
x, y = get_batch("train")
if x is None or y is None:
    raise SystemExit("Fill the TODO (build x and y), then run again.")
assert x.shape == (8, 32) and x.dtype == torch.int64, "x should be (8, 32) int64"
assert torch.equal(x[:, 1:], y[:, :-1]), "y must be x shifted one token right"
print("✓ Correct — a valid (x, y) training batch.")
