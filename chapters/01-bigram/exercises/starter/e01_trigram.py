"""
E01 — Trigram model (STARTER scaffold).
=======================================
This bridges the gap between the notebook (fill one line) and a blank page. The
boring setup is done; fill in the 3 TODOs. A full reference is in
../solutions/e01_trigram.py — try the hints in the exercises README first!

Run:  python e01_trigram.py
"""
from pathlib import Path
import torch

DATA = Path(__file__).resolve().parents[2] / "data" / "names.txt"


def main():
    words = [w.strip() for w in DATA.read_text().splitlines() if w.strip()]
    chars = sorted(set("".join(words)))
    stoi = {c: i + 1 for i, c in enumerate(chars)}
    stoi["."] = 0
    itos = {i: c for c, i in stoi.items()}
    V = len(stoi)

    # ---- TODO #1: count every trigram into the 3-D table N (shape V x V x V) ----
    #   A trigram is three consecutive characters. The padding and the 3-way walk are
    #   set up for you; your job is the one line in the loop: add 1 to the cell that
    #   this (a, b, c) triple indexes.
    N = torch.zeros((V, V, V), dtype=torch.int32)
    for w in words:
        chs = ["."] * 2 + list(w) + ["."]      # two start tokens so the 1st letter has context
        for a, b, c in zip(chs, chs[1:], chs[2:]):
            pass  # <-- your one line here

    # ---- TODO #2: turn counts into probabilities P ----
    #   Add 1 to every cell (smoothing), then normalize along the LAST dimension so
    #   each (a, b) row of next-character probabilities sums to 1.
    P = None  # <-- replace

    # ---- TODO #3: average negative log-likelihood (the loss) ----
    #   Accumulate the log of P at each real trigram across all words, divide by the
    #   number of trigrams, and negate.
    loss = None  # <-- replace
    print(f"trigram loss = {loss}   (bigram was ~2.45; a trigram should reach ~2.1)")

    # ---- (Optional) sample names: like the lesson's Step 6, but slide a 2-char
    #      window — start i, j = 0, 0; sample k from P[i, j]; then i, j = j, k. ----


if __name__ == "__main__":
    main()
