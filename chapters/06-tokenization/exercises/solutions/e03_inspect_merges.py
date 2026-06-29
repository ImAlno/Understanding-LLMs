"""
E03 — Inspect what BPE learned.
===============================
Look at the first merges (common letter pairs) and the longest tokens (whole words).

Run:  python e03_inspect_merges.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))
from bpe import BasicTokenizer

text = (Path(__file__).resolve().parents[2] / "data" / "input.txt").read_text()
tok = BasicTokenizer()
tok.train(text[:200_000], 512)

print("first 10 merges (the most common adjacent pairs):")
for (a, b), idx in list(tok.merges.items())[:10]:
    print(f"  {tok.vocab[a]!r} + {tok.vocab[b]!r}  ->  {tok.vocab[idx]!r}")

print("\nthe 10 longest tokens BPE built:")
longest = sorted(tok.vocab.items(), key=lambda kv: len(kv[1]), reverse=True)[:10]
for idx, b in longest:
    print(f"  token {idx}: {b!r}")
print("\nBPE rediscovers the structure of English on its own — first common letter pairs")
print("('th', 'ou'), then frequent word-chunks — purely by counting.")
