"""
E03 — Does depth help? Scale n_layer.
=====================================
Train GPTs with 1, 2, and 4 Transformer blocks and compare. More layers can build up more
structure — as long as residuals + LayerNorm keep them trainable (they do).

Run:  python e03_depth.py
"""
from mini_gpt import train

print("training at depths 1, 2, 4 (~80s)...")
for n_layer in [1, 2, 4]:
    val, _ = train(n_layer=n_layer, n_head=4, n_embd=64, block_size=32)
    print(f"  n_layer={n_layer}  ->  val {val:.4f}")
print("\nDeeper goes lower here — each block adds a round of 'communicate, then think'. The only")
print("reason stacking this deep even trains is the residual + LayerNorm scaffolding from E01/E02.")
