"""
E01 — Vocabulary size vs. compression.
======================================
Train BPE at several vocab sizes and measure how much it compresses held-out text. More vocab =
more compression, with diminishing returns.

Run:  python e01_compression.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))
from bpe import BasicTokenizer

text = (Path(__file__).resolve().parents[2] / "data" / "input.txt").read_text()
train_text, held = text[:100_000], text[200_000:220_000]
held_bytes = len(held.encode("utf-8"))

sizes, ratios = [], []
print("training BPE at several vocab sizes (~30s)...")
for vs in [256, 512, 1024, 2048]:
    tok = BasicTokenizer()
    tok.train(train_text, vs)
    ratio = held_bytes / len(tok.encode(held))
    sizes.append(vs); ratios.append(ratio)
    print(f"  vocab {vs:5} -> {ratio:.2f}x compression")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.plot(sizes, ratios, "o-")
    plt.xlabel("vocabulary size"); plt.ylabel("compression (bytes / token)")
    plt.title("BPE: bigger vocab compresses more (with diminishing returns)")
    plt.grid(True, alpha=0.3); plt.tight_layout(); plt.savefig("compression.png", dpi=120)
    print("saved compression.png")
except ImportError:
    print("(install matplotlib to see the curve)")
print("\nMore vocab compresses more — but the gains shrink, and a bigger vocab means a bigger")
print("embedding table and rarer tokens. Real models pick a sweet spot (GPT-2: ~50k).")
