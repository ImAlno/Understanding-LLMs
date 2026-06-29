"""
E02 & E03 — Train/dev/test split, and tuning smoothing the honest way.
======================================================================
E02: split the names 80/10/10, train counts on TRAIN only, and measure the loss on
     all three splits. Test loss is a touch higher than train loss — that gap is
     the honest measure of how well the model generalizes to names it never saw.

E03: the "+1" smoothing was a guess. Sweep several values, pick the one with the
     lowest DEV loss, and only THEN report TEST loss. You tune on dev and touch
     test once — the core discipline of all model development.

Run:  python e02_03_split_and_smoothing.py
"""
from pathlib import Path
import random
import torch

DATA = Path(__file__).resolve().parents[2] / "data" / "names.txt"


def build_vocab(words):
    chars = sorted(set("".join(words)))
    stoi = {c: i + 1 for i, c in enumerate(chars)}
    stoi["."] = 0
    return stoi


def bigram_counts(words, stoi, V):
    N = torch.zeros((V, V), dtype=torch.float32)
    for w in words:
        chs = ["."] + list(w) + ["."]
        for a, b in zip(chs, chs[1:]):
            N[stoi[a], stoi[b]] += 1
    return N


def probs_from_counts(N, smoothing):
    P = N + smoothing
    P /= P.sum(dim=1, keepdim=True)
    return P


def eval_loss(words, P, stoi):
    log_likelihood = 0.0
    n = 0
    for w in words:
        chs = ["."] + list(w) + ["."]
        for a, b in zip(chs, chs[1:]):
            log_likelihood += torch.log(P[stoi[a], stoi[b]])
            n += 1
    return (-log_likelihood / n).item()


def main():
    words = [w.strip() for w in DATA.read_text().splitlines() if w.strip()]
    stoi = build_vocab(words)          # vocab from all names (all are a-z anyway)
    V = len(stoi)

    # --- 80/10/10 split (shuffle with a fixed seed for reproducibility) ---
    random.seed(42)
    random.shuffle(words)
    n1, n2 = int(0.8 * len(words)), int(0.9 * len(words))
    train, dev, test = words[:n1], words[n1:n2], words[n2:]
    print(f"split: {len(train)} train / {len(dev)} dev / {len(test)} test\n")

    N_train = bigram_counts(train, stoi, V)

    # --- E02: one model, loss on every split ---
    P = probs_from_counts(N_train, smoothing=1.0)
    print("[E02] smoothing=1.0")
    print(f"      train {eval_loss(train, P, stoi):.4f} | "
          f"dev {eval_loss(dev, P, stoi):.4f} | "
          f"test {eval_loss(test, P, stoi):.4f}")
    print("      (nearly identical here: a bigram has so few parameters it can")
    print("       barely overfit — the train/test gap grows with bigger models, Ch 3+)\n")

    # --- E03: sweep smoothing, choose on dev, report test ---
    print("[E03] tuning smoothing on the DEV set:")
    best = None
    for s in [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]:
        P = probs_from_counts(N_train, smoothing=s)
        d = eval_loss(dev, P, stoi)
        print(f"      smoothing={s:>6} -> dev loss {d:.4f}")
        if best is None or d < best[1]:
            best = (s, d)

    P_best = probs_from_counts(N_train, smoothing=best[0])
    print(f"\n      best smoothing = {best[0]} (dev {best[1]:.4f})")
    print(f"      --> final TEST loss = {eval_loss(test, P_best, stoi):.4f}")


if __name__ == "__main__":
    main()
