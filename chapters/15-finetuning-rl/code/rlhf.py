"""
RLHF: optimize the policy against the reward model with reinforcement learning.
==============================================================================
The classic pipeline (InstructGPT / ChatGPT). We already have a **reward model** (reward_model.py).
Now we do the "RL" in RLHF: let the policy **generate** replies, **score** them with the reward model,
and nudge the policy to make high-scoring replies more likely.

The core estimator is **REINFORCE** (the policy gradient): for a sampled reply `y`,
    push logp_policy(y) UP in proportion to its (baseline-subtracted) reward.
So good replies get reinforced, bad ones get suppressed. Two stabilizers, both essential:
  * a **baseline** (the running average reward) — we reinforce reward *above average*, which cuts variance;
  * a **KL penalty** to the frozen reference — at real scale, without it the policy drifts off and **hacks**
    the reward model (finds gibberish it scores highly); `kl_coef` is the leash. (Our toy is too clean to
    hack — it converges even at kl_coef=0; the KL term earns its keep once the reward model is an imperfect
    proxy over a huge output space, i.e. every real one.)

Real RLHF uses **PPO**, which is REINFORCE plus a *clipped* objective (don't move too far per step) and a
learned value baseline. We use plain REINFORCE + a moving-average baseline: the same idea, minus the
clipping. `python rlhf.py` — takes ~20s (it generates thousands of sample replies).
"""
import torch
from gpt import PREFS, GPT, seq_logprob, reward, generate, train_reference, build_vocab
from reward_model import train_reward_model

KL_COEF = 0.3               # the leash: how hard to keep the policy near the reference (cf. DPO's beta)


def main():
    stoi, itos, V, block = build_vocab()
    ref = train_reference(stoi, V, block)                         # the SFT model (leans curt)
    rm, _ = train_reward_model(stoi, V, block)                    # the reward model (warm > curt)

    policy = GPT(V, block)
    policy.load_state_dict(ref.state_dict())                      # start the policy at the reference
    before = {p: generate(policy, p, stoi, itos) for p, c, r in PREFS}

    torch.manual_seed(3)
    opt = torch.optim.AdamW(policy.parameters(), lr=3e-4)
    baseline = 0.0
    print("REINFORCE against the reward model (mean reward should climb):")
    for step in range(100):
        loss, rewards = 0.0, []
        for p, c, r in PREFS:
            for _ in range(3):                                   # a few samples per prompt
                with torch.no_grad():
                    resp = generate(policy, p, stoi, itos, greedy=False)   # SAMPLE a reply (explore)
                    rew = reward(rm, p, resp, stoi).item()                 # score it with the reward model
                    ref_lp = seq_logprob(ref, p, resp, stoi)               # reference's log-prob of it
                rewards.append(rew)
                logp = seq_logprob(policy, p, resp, stoi)                  # policy's log-prob (differentiable)
                advantage = rew - baseline                                 # reinforce reward ABOVE average
                kl = logp - ref_lp                                         # KL(policy||ref) proxy on this sample
                loss = loss - advantage * logp + KL_COEF * kl             # maximize reward, stay near reference
        loss = loss / (len(PREFS) * 3)
        opt.zero_grad(); loss.backward(); opt.step()
        mean_r = sum(rewards) / len(rewards)
        baseline = 0.9 * baseline + 0.1 * mean_r                          # update the moving-average baseline
        if step % 20 == 0 or step == 99:
            print(f"  step {step:3d}:  mean reward {mean_r:+.2f}")

    policy.eval()
    print("\nBefore -> after RLHF (greedy). The reward model's preference pulled the policy warm:")
    correct = 0
    for p, c, r in PREFS:
        after = generate(policy, p, stoi, itos)
        correct += (after == c)
        print(f"  '{p}':  {before[p]!r:20} -> {after!r:20}  {'✓' if after == c else '✗'}  (chosen {c!r})")
    print(f"\n{correct}/{len(PREFS)} replies became warm — the policy learned to maximize the reward model.")
    print("(At real scale the KL leash is what stops reward hacking; our toy is too clean to hack — it converges even at KL_COEF=0.)")


if __name__ == "__main__":
    main()
