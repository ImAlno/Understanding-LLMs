"""
E01 — Build the SFT loss mask.
==============================
Each SFT example is the whole chat (<user> instr <assistant> resp <end>), but we train ONLY on the
response. Build the target with the prompt positions set to -100 (ignored by cross_entropy).

Run:  python e01_loss_mask.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))
from gpt import PAIRS, USER, ASSISTANT, END, build_vocab, make_example

stoi, itos, V, block = build_vocab()
instr, resp = PAIRS[0]                                    # "hi" -> "hello!"

seq = USER + instr + ASSISTANT + resp + END
ids = [stoi[c] for c in seq]
x, y = ids[:-1], ids[1:]
a_pos = seq.index(ASSISTANT)                              # response tokens come after <assistant>

# mask the prompt: -100 for positions before the response, the real token otherwise
y_masked = [(t if p >= a_pos else -100) for p, t in enumerate(y)]

shown = "".join("_" if t == -100 else ("<end>" if itos[t] == END else itos[t]) for t in y_masked)
print(f"chat  : <user>{instr}<assistant>{resp}<end>")
print(f"target: {shown}   ('_' = masked prompt, trained only on the response)")

assert y_masked == make_example(instr, resp, stoi)[1]
assert all(t == -100 for t in y_masked[:a_pos]) and all(t != -100 for t in y_masked[a_pos:])
print("✓ Only the response (and <end>) is trained; every prompt token is -100.")
