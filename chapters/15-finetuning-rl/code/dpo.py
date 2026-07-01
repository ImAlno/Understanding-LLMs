"""
DPO: preference alignment without a reward model or RL.
======================================================
RLHF is a pipeline: train a reward model, then run reinforcement learning against it. **DPO** (Direct
Preference Optimization) collapses both stages into a single supervised-style loss on the preference
pairs — no reward model, no sampling, no RL loop. It has become the default for open models because it
is so much simpler, and it trains just like the losses you already know.

The DPO loss. Keep a **frozen reference** (the SFT model). For a preference pair (chosen, rejected),
define each reply's **implicit reward** as how much the policy raised its log-prob above the reference:
    r_hat(y) = beta * ( logp_policy(y) - logp_reference(y) )
Then apply the same Bradley-Terry idea as the reward model, but to these implicit rewards:
    loss = -log sigmoid( r_hat(chosen) - r_hat(rejected) )
Minimizing it pushes the policy to prefer chosen over rejected — while `beta` (the KL strength) keeps
it from drifting too far from the reference. That's the whole method.

Run:  python dpo.py
"""
import torch
import torch.nn.functional as F
from gpt import PREFS, GPT, seq_logprob, generate, train_reference, build_vocab

BETA = 0.3          # KL strength: higher = stay closer to the reference (less drift, safer)


def dpo_loss(policy, ref, prompt, chosen, rejected, stoi):
    """The DPO loss for one preference pair."""
    pol_c, pol_r = seq_logprob(policy, prompt, chosen, stoi), seq_logprob(policy, prompt, rejected, stoi)
    with torch.no_grad():                                          # the reference is frozen
        ref_c, ref_r = seq_logprob(ref, prompt, chosen, stoi), seq_logprob(ref, prompt, rejected, stoi)
    chosen_reward   = BETA * (pol_c - ref_c)                       # implicit reward for the chosen reply
    rejected_reward = BETA * (pol_r - ref_r)                       # implicit reward for the rejected reply
    return -F.logsigmoid(chosen_reward - rejected_reward)          # Bradley-Terry on the implicit rewards


def main():
    stoi, itos, V, block = build_vocab()

    # the reference = the SFT model (Chapter 14). It leans curt; alignment will warm it up.
    ref = train_reference(stoi, V, block)
    policy = GPT(V, block)
    policy.load_state_dict(ref.state_dict())                       # start the policy AT the reference
    for p in ref.parameters():
        p.requires_grad = False                                   # freeze the reference

    before = {p: generate(policy, p, stoi, itos) for p, c, r in PREFS}

    # DPO: plain supervised training on the preference pairs
    opt = torch.optim.AdamW(policy.parameters(), lr=2e-4)
    for _ in range(100):
        loss = sum(dpo_loss(policy, ref, p, c, r, stoi) for p, c, r in PREFS) / len(PREFS)
        opt.zero_grad(); loss.backward(); opt.step()
    policy.eval()

    print(f"DPO loss after training: {loss.item():.4f}   (beta = {BETA})\n")
    print("Before -> after alignment (greedy). The reference leaned curt; DPO warmed it up:")
    correct = 0
    with torch.no_grad():
        for p, c, r in PREFS:
            after = generate(policy, p, stoi, itos)
            correct += (after == c)
            print(f"  '{p}':  {before[p]!r:20} -> {after!r:20}  {'✓' if after == c else '✗'}  (chosen {c!r})")

        # the implicit-reward margin: how much more the policy prefers chosen over rejected than the reference did
        margins = [(BETA * ((seq_logprob(policy, p, c, stoi) - seq_logprob(ref, p, c, stoi))
                            - (seq_logprob(policy, p, r, stoi) - seq_logprob(ref, p, r, stoi)))).item()
                   for p, c, r in PREFS]
    acc = sum(m > 0 for m in margins) / len(margins)
    print(f"\npreference accuracy: {acc * 100:.0f}%   |   mean implicit-reward margin: {sum(margins) / len(margins):+.2f}")
    print(f"{correct}/{len(PREFS)} replies became the warm (chosen) one — no reward model, no RL loop, just the DPO loss.")


if __name__ == "__main__":
    main()
