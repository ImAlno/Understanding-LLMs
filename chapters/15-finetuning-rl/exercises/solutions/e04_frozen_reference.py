"""
E04 — The reference must stay frozen.
=====================================
Both DPO and RLHF measure the policy's *drift* against a frozen reference. If the reference trains too,
there's nothing to measure against and the method falls apart. Confirm that DPO moves the policy while
leaving the reference's weights (and its log-probs) exactly unchanged.

Run:  python e04_frozen_reference.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))
import torch
import torch.nn.functional as F
from gpt import PREFS, GPT, seq_logprob, train_reference, build_vocab

BETA = 0.3
stoi, itos, V, block = build_vocab()
ref = train_reference(stoi, V, block)


def ref_logps():
    with torch.no_grad():
        return [seq_logprob(ref, p, c, stoi).item() for p, c, r in PREFS]


ref_sum_before = sum(p.sum().item() for p in ref.parameters())
ref_lp_before = ref_logps()

policy = GPT(V, block); policy.load_state_dict(ref.state_dict())
for p in ref.parameters():
    p.requires_grad = False                                      # FREEZE the reference
opt = torch.optim.AdamW(policy.parameters(), lr=2e-4)
for _ in range(100):
    loss = 0.0
    for p, c, r in PREFS:
        pc, pr = seq_logprob(policy, p, c, stoi), seq_logprob(policy, p, r, stoi)
        with torch.no_grad():
            rc, rr = seq_logprob(ref, p, c, stoi), seq_logprob(ref, p, r, stoi)
        loss = loss - F.logsigmoid(BETA * ((pc - rc) - (pr - rr)))
    loss = loss / len(PREFS)
    opt.zero_grad(); loss.backward(); opt.step()

ref_sum_after = sum(p.sum().item() for p in ref.parameters())
ref_lp_after = ref_logps()
with torch.no_grad():
    pol_lp = [seq_logprob(policy, p, c, stoi).item() for p, c, r in PREFS]

ref_frozen = abs(ref_sum_after - ref_sum_before) < 1e-6 and all(abs(a - b) < 1e-6 for a, b in zip(ref_lp_before, ref_lp_after))
policy_moved = any(abs(a - b) > 0.5 for a, b in zip(ref_lp_before, pol_lp))

print(f"reference param-sum: before {ref_sum_before:.4f}  ->  after {ref_sum_after:.4f}")
print(f"reference logp(chosen) unchanged: {all(abs(a - b) < 1e-6 for a, b in zip(ref_lp_before, ref_lp_after))}")
print(f"policy   logp(chosen) moved:      {policy_moved}  (it should — that's the training)")
print("\n✓ The reference stayed frozen while the policy learned — exactly what DPO/RLHF need."
      if ref_frozen and policy_moved else "\n✗ Something drifted — freeze the reference (requires_grad=False + no_grad).")
