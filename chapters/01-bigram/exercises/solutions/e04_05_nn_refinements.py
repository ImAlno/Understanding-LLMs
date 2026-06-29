"""
E04 & E05 — Two refinements that every real model uses.
=======================================================
E04: show that `F.one_hot(xs, V).float() @ W` is EXACTLY `W[xs]` (row indexing).
     "Look up a row" is what an embedding layer does — no wasteful matmul needed.

E05: show that the hand-written `softmax -> log -> mean` equals `F.cross_entropy`,
     then train with cross_entropy. It's faster and numerically stable.

Run:  python e04_05_nn_refinements.py
"""
from pathlib import Path
import torch
import torch.nn.functional as F

DATA = Path(__file__).resolve().parents[2] / "data" / "names.txt"


def main():
    words = [w.strip() for w in DATA.read_text().splitlines() if w.strip()]
    chars = sorted(set("".join(words)))
    stoi = {c: i + 1 for i, c in enumerate(chars)}
    stoi["."] = 0
    V = len(stoi)

    # build the bigram (input, target) dataset
    xs, ys = [], []
    for w in words:
        chs = ["."] + list(w) + ["."]
        for a, b in zip(chs, chs[1:]):
            xs.append(stoi[a])
            ys.append(stoi[b])
    xs, ys = torch.tensor(xs), torch.tensor(ys)

    # ---------- E04: one-hot @ W == W[xs] ----------
    g = torch.Generator().manual_seed(2147483647)
    W = torch.randn((V, V), generator=g)
    via_onehot = F.one_hot(xs, num_classes=V).float() @ W
    via_index = W[xs]
    same = torch.allclose(via_onehot, via_index, atol=1e-5)
    print(f"[E04] one_hot(xs) @ W  ==  W[xs] ?  {same}")
    print("      -> 'multiply by a one-hot' literally just selects a row. That row")
    print("         lookup is exactly what an Embedding layer does (see Chapter 3).\n")

    # ---------- E05: cross_entropy matches the manual loss ----------
    logits = W[xs]
    probs = F.softmax(logits, dim=1)
    manual = -probs[torch.arange(len(ys)), ys].log().mean()
    builtin = F.cross_entropy(logits, ys)
    print(f"[E05] manual NLL = {manual.item():.4f} | F.cross_entropy = {builtin.item():.4f} "
          f"| match: {torch.allclose(manual, builtin, atol=1e-5)}\n")

    # ---------- E05: now train with cross_entropy ----------
    W = torch.randn((V, V), generator=g, requires_grad=True)
    for k in range(200):
        logits = W[xs]                              # fast indexing (from E04)
        loss = F.cross_entropy(logits, ys) + 0.01 * (W ** 2).mean()
        W.grad = None
        loss.backward()
        W.data += -50 * W.grad
    print(f"[E05] trained with cross_entropy -> final loss {loss.item():.4f} "
          f"(same ~2.45 as the lesson, but cleaner & stabler).")


if __name__ == "__main__":
    main()
