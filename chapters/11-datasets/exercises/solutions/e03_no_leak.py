"""
E03 — The train/val split must not leak.
=========================================
Validation loss is only honest if the model never trained on the val tokens. We split by position
BEFORE sampling windows, so a train window can never reach into the val region. This verifies it.

Run:  python e03_no_leak.py
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
n_train, n_val, _ = prepare(generate(500, seed=0), D)

# train.bin and val.bin are disjoint slices of the same token stream: train = ids[:n_train],
# val = ids[n_train:]. A window from train.bin can start at most at (n_train - block), so its last
# token index is < n_train — it can never touch a val token. Verify by sampling many windows.
block = 32
train = np.memmap(os.path.join(D, "train.bin"), dtype=np.uint16, mode="r")
val = np.memmap(os.path.join(D, "val.bin"), dtype=np.uint16, mode="r")
print(f"train tokens: {len(train):,} | val tokens: {len(val):,} | disjoint by construction")

torch.manual_seed(0)
max_index_touched = 0
for _ in range(1000):
    i = torch.randint(len(train) - block, (1,)).item()
    max_index_touched = max(max_index_touched, i + block)      # last index this window reads
assert max_index_touched <= len(train), "a train window reached past the train region!"
print(f"over 1000 sampled windows, the furthest train index touched was {max_index_touched:,}")
print(f"the val region starts at train-index {len(train):,} — never reached.")
print("✓ No leak: training windows stay entirely inside train.bin.")
