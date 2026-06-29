"""
The Name Forge — REFERENCE SOLUTION.
====================================
A character-level bigram name generator with a small command-line interface.
Compare this to your own version in ../starter/name_forge.py.

Examples:
    python name_forge.py
    python name_forge.py --n 5 --start k
    python name_forge.py --temperature 1.6
    python name_forge.py --temperature 0.5 --smoothing 3
"""
from pathlib import Path
import argparse
import torch

# names.txt lives two folders up (chapters/01-bigram/data/names.txt)
DATA = Path(__file__).resolve().parents[2] / "data" / "names.txt"


def load_words(path=DATA):
    return [w.strip() for w in path.read_text().splitlines() if w.strip()]


def build_vocab(words):
    chars = sorted(set("".join(words)))
    stoi = {c: i + 1 for i, c in enumerate(chars)}
    stoi["."] = 0
    itos = {i: c for c, i in stoi.items()}
    return stoi, itos


# ---- TODO #1 in the starter ----
def build_probs(words, stoi, V, smoothing=1.0):
    """Count bigrams, smooth, and normalize each row into a distribution."""
    N = torch.zeros((V, V), dtype=torch.float32)
    for w in words:
        chs = ["."] + list(w) + ["."]
        for a, b in zip(chs, chs[1:]):
            N[stoi[a], stoi[b]] += 1
    P = N + smoothing
    P /= P.sum(dim=1, keepdim=True)
    return P


# ---- TODO #2 in the starter ----
def apply_temperature(p, temperature):
    """Re-weight a probability vector by temperature (1.0 = unchanged)."""
    if temperature == 1.0:
        return p
    q = p.clamp_min(1e-9) ** (1.0 / temperature)
    return q / q.sum()


# ---- TODO #3 in the starter ----
def sample_name(P, stoi, itos, start=None, temperature=1.0, g=None):
    """Generate a single name; if `start` is given, force the first letter."""
    out = []
    if start:
        out.append(start)
        ix = stoi[start]
    else:
        ix = 0  # the '.' start token
    while True:
        p = apply_temperature(P[ix], temperature)
        ix = torch.multinomial(p, num_samples=1, generator=g).item()
        if ix == 0:
            break
        out.append(itos[ix])
    return "".join(out)


# ---- TODO #4 in the starter ----
def model_loss(words, P, stoi):
    """Average negative log-likelihood over all bigrams."""
    log_likelihood = 0.0
    n = 0
    for w in words:
        chs = ["."] + list(w) + ["."]
        for a, b in zip(chs, chs[1:]):
            log_likelihood += torch.log(P[stoi[a], stoi[b]])
            n += 1
    return (-log_likelihood / n).item()


def main():
    parser = argparse.ArgumentParser(description="The Name Forge — conjure new names.")
    parser.add_argument("--n", type=int, default=10, help="how many names to forge")
    parser.add_argument("--start", type=str, default=None, help="force a starting letter")
    parser.add_argument("--temperature", type=float, default=1.0, help="creativity (>1 wilder)")
    parser.add_argument("--smoothing", type=float, default=1.0, help="add-k smoothing")
    parser.add_argument("--seed", type=int, default=2147483647, help="random seed")
    args = parser.parse_args()

    words = load_words()
    stoi, itos = build_vocab(words)
    V = len(stoi)

    if args.start is not None:
        args.start = args.start.lower()
        if len(args.start) != 1 or args.start not in stoi or args.start == ".":
            raise SystemExit("--start must be a single letter a-z")

    P = build_probs(words, stoi, V, smoothing=args.smoothing)
    g = torch.Generator().manual_seed(args.seed)

    print(f"🔥 The Name Forge — {args.n} names "
          f"(temp={args.temperature}, smoothing={args.smoothing})")
    for _ in range(args.n):
        print("  ", sample_name(P, stoi, itos, args.start, args.temperature, g))
    print(f"\nmodel loss (NLL) = {model_loss(words, P, stoi):.4f}")


if __name__ == "__main__":
    main()
