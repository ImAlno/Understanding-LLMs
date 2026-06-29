"""
Bigram Language Model — the COUNTING approach.
================================================

This is the simplest possible language model. It looks at pairs of adjacent
characters ("bigrams") in a big list of names, counts how often each pair
occurs, and turns those counts into probabilities. To generate a new name we
just keep sampling "what character tends to come next?" until we hit the end.

Run it:
    python bigram_counts.py
    python bigram_counts.py --num 20 --smoothing 1
    python bigram_counts.py --plot        # also saves a heatmap (needs matplotlib)

Everything here is plain counting — there is NO training and NO neural network.
We build the exact same model with a neural network in `bigram_nn.py`, and the
two end up agreeing almost perfectly. That is the whole point of Chapter 1.
"""

from pathlib import Path
import argparse
import torch

# Locate data/names.txt relative to THIS file, so the script works no matter
# which directory you run it from.
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "names.txt"


def load_words(path=DATA_PATH):
    """Read the dataset into a list of lowercase names (one per line)."""
    lines = path.read_text().splitlines()
    return [w.strip() for w in lines if w.strip()]


def build_vocab(words):
    """Map every character to an integer id and back.

    We add one special token, '.', at index 0. It marks both the START and the
    END of a name, which lets a single bigram model learn which letters tend to
    start a name and which tend to end one.
    """
    chars = sorted(set("".join(words)))        # the 26 letters a..z
    stoi = {c: i + 1 for i, c in enumerate(chars)}  # 'a'->1, 'b'->2, ...
    stoi["."] = 0                                   # boundary token -> 0
    itos = {i: c for c, i in stoi.items()}          # reverse mapping
    return stoi, itos


def build_counts(words, stoi, vocab_size):
    """Count every bigram. N[i, j] = how often character j follows character i."""
    N = torch.zeros((vocab_size, vocab_size), dtype=torch.int32)
    for w in words:
        chs = ["."] + list(w) + ["."]          # e.g. emma -> . e m m a .
        for ch1, ch2 in zip(chs, chs[1:]):     # walk over adjacent pairs
            N[stoi[ch1], stoi[ch2]] += 1
    return N


def counts_to_probs(N, smoothing=1.0):
    """Turn the count matrix into a probability matrix (each row sums to 1).

    `smoothing` adds a small fake count to every cell. This is "model smoothing":
    it stops any bigram from having probability exactly 0 (which would make the
    model assign probability 0 — and infinite loss — to any name containing it).
    """
    P = (N + smoothing).float()
    P /= P.sum(dim=1, keepdim=True)            # normalize each row to a distribution
    return P


@torch.no_grad()
def sample_names(P, itos, n=10, seed=2147483647):
    """Generate `n` names by repeatedly sampling the next character."""
    g = torch.Generator().manual_seed(seed)   # fixed seed -> reproducible names
    out = []
    for _ in range(n):
        ix = 0                                 # always start at '.'
        chars = []
        while True:
            p = P[ix]                          # distribution over the next char
            ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
            if ix == 0:                        # sampled the '.' end token
                break
            chars.append(itos[ix])
        out.append("".join(chars))
    return out


def nll_loss(words, P, stoi):
    """Average negative log-likelihood (NLL) — our quality score (lower is better).

    The likelihood of the data is the product of P(next | current) over every
    bigram. Probabilities multiply to tiny numbers, so we work in log space and
    sum instead. NLL = -(average log probability). A perfect model scores 0; a
    uniform random guess over 27 tokens scores log(27) ≈ 3.30.
    """
    log_likelihood = 0.0
    n = 0
    for w in words:
        chs = ["."] + list(w) + ["."]
        for ch1, ch2 in zip(chs, chs[1:]):
            prob = P[stoi[ch1], stoi[ch2]]
            log_likelihood += torch.log(prob)
            n += 1
    return (-log_likelihood / n).item()


def save_heatmap(N, itos, path="bigram_counts.png"):
    """Optional: visualize the 27x27 count grid. Needs matplotlib."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed — skipping --plot)")
        return
    vocab_size = N.shape[0]
    plt.figure(figsize=(16, 16))
    plt.imshow(N, cmap="Blues")
    for i in range(vocab_size):
        for j in range(vocab_size):
            chstr = itos[i] + itos[j]
            plt.text(j, i, chstr, ha="center", va="bottom", color="gray", fontsize=7)
            plt.text(j, i, N[i, j].item(), ha="center", va="top", color="gray", fontsize=7)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    print(f"saved heatmap -> {path}")


def main():
    parser = argparse.ArgumentParser(description="Bigram language model (counting).")
    parser.add_argument("--num", type=int, default=10, help="how many names to sample")
    parser.add_argument("--smoothing", type=float, default=1.0, help="add-k smoothing")
    parser.add_argument("--seed", type=int, default=2147483647, help="sampling seed")
    parser.add_argument("--plot", action="store_true", help="save a bigram heatmap")
    args = parser.parse_args()

    words = load_words()
    stoi, itos = build_vocab(words)
    vocab_size = len(stoi)
    print(f"loaded {len(words)} names | vocab size = {vocab_size}")

    N = build_counts(words, stoi, vocab_size)
    P = counts_to_probs(N, smoothing=args.smoothing)

    print("\nsampled names:")
    for name in sample_names(P, itos, n=args.num, seed=args.seed):
        print("  ", name)

    loss = nll_loss(words, P, stoi)
    print(f"\naverage negative log-likelihood (loss) = {loss:.4f}")
    print("(uniform-random baseline over 27 tokens would be log(27) = 3.2958)")

    if args.plot:
        save_heatmap(N, itos)


if __name__ == "__main__":
    main()
