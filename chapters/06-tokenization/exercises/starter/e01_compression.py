"""
E01 — Vocabulary size vs. compression (STARTER scaffold).
=========================================================
Fill the TODO: train a tokenizer at each vocab size and measure its compression.
Reference: ../solutions/e01_compression.py.

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
print("fill the TODO below — then this trains BPE at several vocab sizes (~25s).")
for vs in [256, 512, 1024, 2048]:
    # ✍️ TODO: train a BasicTokenizer to vocab size `vs`, then set `ratio` to the compression =
    #          held_bytes divided by the number of tokens when you encode `held`.
    ratio = None      # replace
    if ratio is None:
        raise SystemExit("Fill in the TODO (train a tokenizer and measure compression), then run again.")
    sizes.append(vs); ratios.append(ratio)
    print(f"  vocab {vs:5} -> {ratio:.2f}x compression")

print("done — now try plotting `sizes` vs `ratios` with matplotlib (see the solution).")
