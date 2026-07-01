"""
Align Your Storyteller with DPO — SOLUTION.
===========================================
The capstone of the finetuning arc: take an SFT model that defaults to curt replies and align it to a
*preference* (warmth) with **DPO** — one supervised loss on (prompt, chosen, rejected) triples, the
reference frozen the whole time. No reward model, no RL loop.

Run:  python align_storyteller.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))
import torch
import torch.nn.functional as F
from gpt import PREFS, GPT, seq_logprob, generate, train_reference, build_vocab

BETA = 0.3          # the KL leash: how far the aligned policy may drift from the reference


def dpo_loss(policy, ref, prompt, chosen, rejected):
    """DPO: Bradley-Terry on the implicit reward β·(logp_policy − logp_ref)."""
    pol_c, pol_r = seq_logprob(policy, prompt, chosen, stoi), seq_logprob(policy, prompt, rejected, stoi)
    with torch.no_grad():                                        # the reference is frozen
        ref_c, ref_r = seq_logprob(ref, prompt, chosen, stoi), seq_logprob(ref, prompt, rejected, stoi)
    return -F.logsigmoid(BETA * ((pol_c - ref_c) - (pol_r - ref_r)))


def main():
    global stoi
    stoi, itos, V, block = build_vocab()

    ref = train_reference(stoi, V, block)                        # the SFT model (leans curt)
    policy = GPT(V, block); policy.load_state_dict(ref.state_dict())
    for p in ref.parameters():
        p.requires_grad = False                                  # freeze the reference

    before = {p: generate(policy, p, stoi, itos) for p, c, r in PREFS}

    opt = torch.optim.AdamW(policy.parameters(), lr=2e-4)
    for _ in range(100):
        loss = sum(dpo_loss(policy, ref, p, c, r) for p, c, r in PREFS) / len(PREFS)
        opt.zero_grad(); loss.backward(); opt.step()
    policy.eval()

    print(f"DPO alignment (β={BETA}). Curt -> warm, reference frozen:\n")
    correct = 0
    for p, c, r in PREFS:
        after = generate(policy, p, stoi, itos)
        correct += (after == c)
        print(f"  '{p}':  {before[p]!r:20} -> {after!r:20}  {'✓' if after == c else '✗'}  (chosen {c!r})")
    print(f"\n✅ {correct}/{len(PREFS)} replies aligned to the warm preference — with the DPO loss alone.")


if __name__ == "__main__":
    main()
