"""
E04 — ReLU vs GELU in the feed-forward net.
===========================================
GPT-2 uses GELU (a smooth ReLU) instead of plain ReLU. Swap the activation and compare — the
difference is usually small at this scale, which is itself worth seeing.

Run:  python e04_activation.py
"""
from mini_gpt import train

cfg = dict(n_layer=4, n_head=4, n_embd=64, block_size=32)
print("training with ReLU vs GELU feed-forward (~55s)...")
for act in ["relu", "gelu"]:
    val, _ = train(activation=act, **cfg)
    print(f"  {act.upper():4} ->  val {val:.4f}")
print("\nUsually close. GELU (what GPT-2 uses) is a smoother nonlinearity; at small scale the")
print("choice barely moves the loss, but details like this compound in big models.")
