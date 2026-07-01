"""
The reward model: turn preferences into a number.
=================================================
RLHF's first stage. Humans can't score a reply on an absolute scale ("is this a 7.3/10 story?"), but
they *can* compare two: "A is better than B." A **reward model** learns from those comparisons to
output a scalar score, so the RL stage has something to optimize.

We train it with the **Bradley-Terry** loss. If the model scores the chosen reply `s_c` and the
rejected reply `s_r`, the probability it "agrees" with the human is  sigmoid(s_c - s_r).  We maximize
that, i.e. minimize  -log sigmoid(s_c - s_r).  The model is free to pick any scores it likes, as long
as **chosen > rejected**.

Run:  python reward_model.py
"""
import torch
import torch.nn.functional as F
from gpt import PREFS, RewardModel, reward, build_vocab


def train_reward_model(stoi, V, block, steps=300, seed=1):
    """Train a RewardModel on the preference pairs with the Bradley-Terry ranking loss."""
    torch.manual_seed(seed)
    rm = RewardModel(V, block)
    opt = torch.optim.AdamW(rm.parameters(), lr=3e-3)
    for _ in range(steps):
        # -log sigmoid(score(chosen) - score(rejected)), averaged over the preference pairs
        loss = sum(-F.logsigmoid(reward(rm, p, c, stoi) - reward(rm, p, r, stoi)) for p, c, r in PREFS) / len(PREFS)
        opt.zero_grad(); loss.backward(); opt.step()
    rm.eval()
    return rm, loss.item()


def main():
    stoi, itos, V, block = build_vocab()
    rm, final_loss = train_reward_model(stoi, V, block)

    print(f"Bradley-Terry loss after training: {final_loss:.4f}\n")
    print("The reward model scores each reply (it never saw these numbers — it learned them from the pairs):")
    with torch.no_grad():
        margins = []
        for p, c, r in PREFS:
            sc, sr = reward(rm, p, c, stoi).item(), reward(rm, p, r, stoi).item()
            margins.append(sc - sr)
            print(f"  '{p}':  chosen {c!r:20} -> {sc:+6.2f}   |   rejected {r!r:20} -> {sr:+6.2f}   (margin {sc - sr:+.2f})")

    acc = sum(m > 0 for m in margins) / len(margins)
    print(f"\naccuracy (chosen scored above rejected): {acc * 100:.0f}%   |   mean margin: {sum(margins) / len(margins):+.2f}")
    print("The reward model has turned 'A is better than B' comparisons into a score. Next: optimize a")
    print("policy to maximize it (rlhf.py) — or skip the reward model entirely with DPO (dpo.py).")


if __name__ == "__main__":
    main()
