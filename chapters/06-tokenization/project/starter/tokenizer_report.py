"""
Train Your Own Tokenizer — STARTER scaffold.
============================================
Fill the two TODOs (train the tokenizer; measure compression). The report is written for you.
Reference: ../solution/tokenizer_report.py.

    python tokenizer_report.py
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

tok = RegexTokenizer()
# ✍️ TODO #1: train `tok` on `train_text` to VOCAB_SIZE  (see RegexTokenizer.train)
if not tok.merges:
    raise SystemExit("Fill in TODO #1 (train the tokenizer), then run again.")
print(f"trained a RegexTokenizer on {TEXT_PATH.name} to vocab {VOCAB_SIZE}.")

assert tok.decode(tok.encode(held)) == held, "round-trip failed!"
# ✍️ TODO #2: set nbytes = number of UTF-8 bytes in `held`, ntoks = number of tokens tok.encode(held)
nbytes, ntoks = None, None      # replace both
if nbytes is None or ntoks is None:
    raise SystemExit("Fill in TODO #2 (the compression numbers), then run again.")

print("\n=== your tokenizer ===")
print(f"  round-trip:  ✓")
print(f"  compression: {nbytes} bytes -> {ntoks} tokens  ({nbytes/ntoks:.2f}x)")
sample = "First Citizen: we are all resolved rather to die than to famish."
toks = tok.encode(sample)
print(f"  sample:      {sample!r}")
print(f"               -> {len(toks)} tokens: {[tok.decode([t]) for t in toks]}")
longest = sorted(tok.vocab.values(), key=len, reverse=True)[:8]
print(f"  longest learned tokens: {longest}")
