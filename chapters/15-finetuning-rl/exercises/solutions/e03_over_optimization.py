"""
E03 — The β leash: watch DPO over-optimize.
============================================
β controls how far DPO lets the policy drift from the reference. At a healthy β it warms the model up
cleanly. Too small a β (with a hot learning rate) and DPO *over-optimizes*: the chosen-vs-rejected
margin keeps climbing while the actual generations decay into mush. Bigger margin, worse text — the
same reward-hacking disease, with no reward model in sight.

Run:  python e03_over_optimization.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))
import torch
import torch.nn.functional as F
from gpt import PREFS, GPT, seq_logprob, generate, train_reference, build_vocab

stoi, itos, V, block = build_vocab()
ref = train_reference(stoi, V, block)


def run_dpo(beta, lr, steps):
    torch.manual_seed(2)
    pol = GPT(V, block); pol.load_state_dict(ref.state_dict())
    for p in ref.parameters():
        p.requires_grad = False
    opt = torch.optim.AdamW(pol.parameters(), lr=lr)
    for _ in range(steps):
        loss = 0.0
        for p, c, r in PREFS:
            pc, pr = seq_logprob(pol, p, c, stoi), seq_logprob(pol, p, r, stoi)
            with torch.no_grad():
                rc, rr = seq_logprob(ref, p, c, stoi), seq_logprob(ref, p, r, stoi)
            loss = loss - F.logsigmoid(beta * ((pc - rc) - (pr - rr)))
        loss = loss / len(PREFS)
        opt.zero_grad(); loss.backward(); opt.step()
    pol.eval()
    with torch.no_grad():
        margin = sum((seq_logprob(pol, p, c, stoi) - seq_logprob(pol, p, r, stoi)).item() for p, c, r in PREFS) / len(PREFS)
    return {p: generate(pol, p, stoi, itos) for p, c, r in PREFS}, margin


print("healthy  β=0.30 (stays near the reference):")
gens, m = run_dpo(0.30, 2e-4, 100)
for p, c, r in PREFS:
    print(f"    '{p}' -> {gens[p]!r:22} (chosen {c!r})")
print(f"    mean logp margin (chosen-rejected): {m:+.1f}\n")

print("too small β=0.05 + hot lr (over-optimizes):")
gens, m = run_dpo(0.05, 1e-3, 300)
for p, c, r in PREFS:
    print(f"    '{p}' -> {gens[p]!r:22} (chosen {c!r})")
print(f"    mean logp margin (chosen-rejected): {m:+.1f}   <- LARGER margin, mush text: that's over-optimization.")
print("\n✓ The margin is not quality. A higher β (a tighter leash to the reference) is the fix.")
