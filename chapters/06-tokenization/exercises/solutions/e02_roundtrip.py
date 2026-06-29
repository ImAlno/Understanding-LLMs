"""
E02 — Round-trip stress test.
=============================
decode(encode(text)) must return the original for ANY text — because BPE works on raw bytes.

Run:  python e02_roundtrip.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))
from bpe import BasicTokenizer

text = (Path(__file__).resolve().parents[2] / "data" / "input.txt").read_text()
tok = BasicTokenizer()
tok.train(text[:100_000], 512)

tests = [
    "plain ASCII text",
    "café résumé naïve Zoë",          # accents
    "emoji party 🧙🔥🎉",              # multi-byte emoji
    "中文字符也可以",                   # CJK (never seen in training!)
    "mixed: Ω ≈ ç √ ∫ 🤖",
    "",                                # empty
    "tabs\tand\nnewlines",
    "numbers 123,456.789",
]
print("round-trip on tricky strings:")
all_ok = True
for s in tests:
    ok = tok.decode(tok.encode(s)) == s
    all_ok &= ok
    print(f"  {'✓' if ok else '✗'} {s!r}")
assert all_ok, "a round-trip failed!"
print("\nEvery string survives — including Chinese characters the tokenizer NEVER saw in")
print("training. Working on bytes means nothing is ever out-of-vocabulary; unseen text just")
print("falls back to single-byte tokens.")
