"""
The Name Forge — STARTER SCAFFOLD.
==================================
Fill in the four functions marked `# TODO`. The data loading and the command-line
interface are already wired up for you — once your four functions work, the whole
program works.

Run it any time to see which TODO is next:
    python name_forge.py
    python name_forge.py --n 5 --start k
    python name_forge.py --temperature 1.6

Stuck? Each TODO mirrors something from the lesson (chapters/01-bigram/README.md).
A full reference is in ../solution/name_forge.py — peek only after trying!
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


def build_probs(words, stoi, V, smoothing=1.0):
    """TODO #1 — Build the bigram probability matrix P (shape V x V).

    Steps:
      1. make a zeros tensor  N = torch.zeros((V, V))
      2. for each word, wrap it as ['.'] + list(word) + ['.'] and add 1 to
         N[stoi[a], stoi[b]] for each adjacent pair (a, b)
      3. P = N + smoothing
      4. divide each ROW of P by its sum so the row adds up to 1
             P /= P.sum(dim=1, keepdim=True)
      5. return P
    """
    raise NotImplementedError("TODO #1: build and return the bigram matrix P")


def apply_temperature(p, temperature):
    """TODO #2 — Re-weight a probability vector `p` by `temperature`.

      - if temperature == 1.0: return p unchanged
      - otherwise: q = p ** (1.0 / temperature), then return q / q.sum()
        (tip: use p.clamp_min(1e-9) before the power to avoid 0**big issues)
    """
    raise NotImplementedError("TODO #2: apply temperature and return a distribution")


def sample_name(P, stoi, itos, start=None, temperature=1.0, g=None):
    """TODO #3 — Generate ONE name as a string.

    Steps:
      - if `start` is given: out = [start]; ix = stoi[start]
        else:                out = [];      ix = 0          (the '.' token)
      - loop forever:
          p  = apply_temperature(P[ix], temperature)
          ix = torch.multinomial(p, num_samples=1, generator=g).item()
          if ix == 0: break                 # hit the end token
          out.append(itos[ix])
      - return "".join(out)
    """
    raise NotImplementedError("TODO #3: sample and return one generated name")


def model_loss(words, P, stoi):
    """TODO #4 — Average negative log-likelihood over all bigrams (a float).

    Accumulate `torch.log(P[stoi[a], stoi[b]])` over every adjacent pair in every
    (padded) word, divide by the count, and return the NEGATIVE of that mean.
    """
    raise NotImplementedError("TODO #4: compute and return the NLL loss")


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
