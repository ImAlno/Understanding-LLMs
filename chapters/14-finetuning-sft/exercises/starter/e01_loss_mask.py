"""
E01 — Build the SFT loss mask (STARTER scaffold).
=================================================
Fill the TODO: mask the prompt tokens so loss trains only on the response.
Reference: ../solutions/e01_loss_mask.py.

Run:  python e01_loss_mask.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))
from gpt import PAIRS, USER, ASSISTANT, END, build_vocab, make_example

stoi, itos, V, block = build_vocab()
instr, resp = PAIRS[0]

seq = USER + instr + ASSISTANT + resp + END
ids = [stoi[c] for c in seq]
x, y = ids[:-1], ids[1:]
a_pos = seq.index(ASSISTANT)                              # response tokens come after <assistant>

# ✍️ TODO: build y_masked — target is -100 for positions before the response (p < a_pos),
#          and the real token y[p] otherwise.  hint: [(t if p >= a_pos else -100) for p, t in enumerate(y)]
y_masked = None      # replace

if y_masked is None:
    raise SystemExit("Fill the TODO (build y_masked), then run again.")

assert y_masked == make_example(instr, resp, stoi)[1], "check the mask: -100 before a_pos, real token after"
print("✓ Correct — only the response is trained; the prompt tokens are -100.")
