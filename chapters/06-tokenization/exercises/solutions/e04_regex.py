"""
E04 — Regex splitting (the GPT-2 way).
======================================
Compare the Basic tokenizer (merges across spaces) with the Regex tokenizer (splits into
word-ish chunks first, keeping a LEADING space with each word — like GPT-2/GPT-4).

Run:  python e04_regex.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))
from bpe import BasicTokenizer, RegexTokenizer

train_text = (Path(__file__).resolve().parents[2] / "data" / "input.txt").read_text()[:100_000]
basic = BasicTokenizer(); basic.train(train_text, 512)
regex = RegexTokenizer(); regex.train(train_text, 512)

print("first 8 merges learned:")
for name, tok in [("Basic", basic), ("Regex", regex)]:
    merges = [bytes(tok.vocab[a] + tok.vocab[b]) for (a, b) in list(tok.merges)[:8]]
    print(f"  {name}:", merges)

sample = "First Citizen we are resolved"
print(f"\nhow each splits {sample!r} into tokens:")
for name, tok in [("Basic", basic), ("Regex", regex)]:
    print(f"  {name}:", [tok.decode([t]) for t in tok.encode(sample)])
print("\nThe Basic tokenizer merges trailing spaces ('e '); the Regex tokenizer keeps a LEADING")
print("space with each word (' Citizen'), never gluing a word to the next — exactly how real")
print("GPT tokenizers behave.")
