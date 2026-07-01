"""
E02 — Align a model with the DPO loss.
======================================
Implement DPO's loss — Bradley-Terry on the *implicit reward* β·(logp_policy − logp_ref) — and turn the
curt reference warm, with no reward model and no RL loop.

Run:  python e02_dpo_loss.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))
import torch
import torch.nn.functional as F
from gpt import PREFS, GPT, seq_logprob, generate, train_reference, build_vocab

BETA = 0.3
stoi, itos, V, block = build_vocab()


def dpo_loss(policy, ref, prompt, chosen, rejected):
    pc, pr = seq_logprob(policy, prompt, chosen, stoi), seq_logprob(policy, prompt, rejected, stoi)
    with torch.no_grad():                                        # reference is frozen
        rc, rr = seq_logprob(ref, prompt, chosen, stoi), seq_logprob(ref, prompt, rejected, stoi)
    return -F.logsigmoid(BETA * ((pc - rc) - (pr - rr)))          # implicit-reward Bradley-Terry


ref = train_reference(stoi, V, block)
policy = GPT(V, block); policy.load_state_dict(ref.state_dict())
for p in ref.parameters():
    p.requires_grad = False
opt = torch.optim.AdamW(policy.parameters(), lr=2e-4)
for _ in range(100):
    loss = sum(dpo_loss(policy, ref, p, c, r) for p, c, r in PREFS) / len(PREFS)
    opt.zero_grad(); loss.backward(); opt.step()
policy.eval()

correct = 0
for p, c, r in PREFS:
    got = generate(policy, p, stoi, itos)
    correct += (got == c)
    print(f"  '{p}' -> {got!r:20} (chosen {c!r})")
print(f"\n✓ {correct}/{len(PREFS)} replies became warm — DPO aligned the model with one supervised loss."
      if correct == len(PREFS) else f"\n✗ Only {correct}/{len(PREFS)} — check the DPO loss sign and β.")
