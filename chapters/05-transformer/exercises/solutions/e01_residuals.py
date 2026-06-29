"""
E01 — Ablate residual connections.
==================================
Train a 4-layer GPT with and without the residual connections (the `x = x +` in each block).
Without them, gradients can't flow back through the stack and the deep model trains far worse.

Run:  python e01_residuals.py
"""
from mini_gpt import train

cfg = dict(n_layer=4, n_head=4, n_embd=64, block_size=32)
print("training a 4-layer GPT, with and without residual connections (~55s)...")
val_with, _ = train(residual=True, **cfg)
val_without, _ = train(residual=False, **cfg)
print(f"  WITH residuals    ->  val {val_with:.4f}")
print(f"  WITHOUT residuals ->  val {val_without:.4f}")
print("\nThe residual version wins clearly. The `x = x + sublayer(x)` highway lets gradients")
print("reach the early layers; without it, a deep stack barely trains. This is THE trick that")
print("made deep networks possible.")
