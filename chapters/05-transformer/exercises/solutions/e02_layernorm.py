"""
E02 — Ablate LayerNorm.
=======================
Train the same GPT with and without LayerNorm. The result is instructive: at this toy scale,
with residual connections already in place, removing LayerNorm barely changes the loss (it can
even land a hair lower). LayerNorm's job is *stability* as models get deeper and train longer —
not a lower loss on a tiny model.

Run:  python e02_layernorm.py
"""
from mini_gpt import train

cfg = dict(n_layer=4, n_head=4, n_embd=64, block_size=32)
print("training with and without LayerNorm (~55s)...")
val_with, _ = train(norm=True, **cfg)
val_without, _ = train(norm=False, **cfg)
print(f"  WITH LayerNorm    ->  val {val_with:.4f}")
print(f"  WITHOUT LayerNorm ->  val {val_without:.4f}")
print("\nBasically a wash here — even a touch better without it. With only 4 layers and the")
print("residual highway already in place, training is stable enough without LayerNorm at this")
print("scale. Its payoff is stability as depth and training time grow; at GPT scale it's")
print("essential. (Ablations don't always show the textbook effect on a toy — a real lesson.)")
