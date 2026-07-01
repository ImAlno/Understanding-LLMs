"""
E01 — Implement get_batch.
==========================
The random-window sampler that feeds every GPT training run: memory-map the .bin, pick random start
offsets, and return (x, y) where y is x shifted one token right.

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
prepare(generate(500, seed=0), D)                      # a small corpus, packed to D/train.bin


def get_batch(split, block_size=32, batch_size=8):
    data = np.memmap(os.path.join(D, f"{split}.bin"), dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    return x, y


torch.manual_seed(0)
x, y = get_batch("train")
print(f"x {tuple(x.shape)}, y {tuple(y.shape)}, dtype {x.dtype}")
assert x.shape == (8, 32) and y.dtype == torch.int64
assert torch.equal(x[:, 1:], y[:, :-1])                # y is x shifted by one
print("✓ x and y are windows offset by one, int64 — a valid training batch.")
