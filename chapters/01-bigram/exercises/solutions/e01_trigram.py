"""
E01 (and E06) — Trigram model by counting.
==========================================
A bigram predicts the next character from ONE previous character. A trigram uses
TWO. We build a 27x27x27 table of counts N[i, j, k] = "how often did k follow the
pair (i, j)?", normalize, score it, and sample from it.

Expected: loss ~2.09, noticeably better than the bigram's ~2.45 — and the sampled
names look more pronounceable, because the model now remembers two characters.

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

    # --- count every trigram. Pad with TWO start tokens so the first real
    #     character is predicted from ('.', '.'). ---
    N = torch.zeros((V, V, V), dtype=torch.int32)
    for w in words:
        chs = ["."] * 2 + list(w) + ["."]
        for a, b, c in zip(chs, chs[1:], chs[2:]):
            N[stoi[a], stoi[b], stoi[c]] += 1

    # --- normalize the last dimension into probabilities (with +1 smoothing) ---
    P = (N + 1).float()
    P /= P.sum(dim=2, keepdim=True)

    # --- loss: average negative log-likelihood over every trigram ---
    log_likelihood = 0.0
    n = 0
    for w in words:
        chs = ["."] * 2 + list(w) + ["."]
        for a, b, c in zip(chs, chs[1:], chs[2:]):
            log_likelihood += torch.log(P[stoi[a], stoi[b], stoi[c]])
            n += 1
    print(f"trigram loss = {(-log_likelihood / n).item():.4f}   (bigram was ~2.45)")

    # --- sample names (slide a 2-character window forward) ---
    g = torch.Generator().manual_seed(2147483647)
    print("\nsampled names:")
    for _ in range(15):
        i, j = 0, 0          # start context = ('.', '.')
        out = []
        while True:
            k = torch.multinomial(P[i, j], num_samples=1, generator=g).item()
            if k == 0:
                break
            out.append(itos[k])
            i, j = j, k       # slide the window
        print("  ", "".join(out))


if __name__ == "__main__":
    main()
