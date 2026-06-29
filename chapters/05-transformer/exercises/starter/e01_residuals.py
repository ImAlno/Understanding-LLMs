"""
E01 — Ablate residual connections (STARTER scaffold).
=====================================================
Fill the two TODOs to run the experiment. Reference: ../solutions/e01_residuals.py.

Run:  python e01_residuals.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "solutions"))

from mini_gpt import train          # train(...) returns (val_loss, model)

cfg = dict(n_layer=4, n_head=4, n_embd=64, block_size=32)
print("training a 4-layer GPT with and without residuals (~40s)...")

# ✍️ TODO #1: train WITH residual connections, capture the val loss (train returns a tuple)
val_with = None       # replace

# ✍️ TODO #2: train WITHOUT residual connections (the same call, residuals turned off)
val_without = None    # replace

if val_with is None or val_without is None:
    raise SystemExit("Fill in the two train(...) calls (TODO #1 and #2), then run again.")
print(f"  WITH residuals    ->  val {val_with:.4f}")
print(f"  WITHOUT residuals ->  val {val_without:.4f}")
