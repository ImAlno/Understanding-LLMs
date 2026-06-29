"""
Train Your Own Tokenizer — REFERENCE SOLUTION.
==============================================
Train a real (regex-splitting) BPE tokenizer on a text of your choice, then print a report:
round-trip, compression, a sample tokenization, and the longest tokens it discovered.

    python tokenizer_report.py        (~5s on a CPU)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))
from bpe import RegexTokenizer

# ---- pick your text and vocab size (try your own .txt here!) ----
TEXT_PATH = Path(__file__).resolve().parents[2] / "data" / "input.txt"
VOCAB_SIZE = 512

text = TEXT_PATH.read_text()
train_text, held = text[:150_000], text[300_000:330_000]

print(f"training a RegexTokenizer on {TEXT_PATH.name} ({len(train_text)} chars) to vocab {VOCAB_SIZE}...")
tok = RegexTokenizer()
tok.train(train_text, VOCAB_SIZE)

assert tok.decode(tok.encode(held)) == held, "round-trip failed!"
nbytes, ntoks = len(held.encode("utf-8")), len(tok.encode(held))

print("\n=== your tokenizer ===")
print(f"  round-trip:  ✓")
print(f"  compression: {nbytes} bytes -> {ntoks} tokens  ({nbytes/ntoks:.2f}x)")
sample = "First Citizen: we are all resolved rather to die than to famish."
toks = tok.encode(sample)
print(f"  sample:      {sample!r}")
print(f"               -> {len(toks)} tokens: {[tok.decode([t]) for t in toks]}")
longest = sorted(tok.vocab.values(), key=len, reverse=True)[:8]
print(f"  longest learned tokens: {longest}")
