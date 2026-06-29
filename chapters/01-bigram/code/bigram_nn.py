"""
Bigram Language Model — the NEURAL NETWORK approach.
====================================================

Here is the big idea of Chapter 1: the counting model from `bigram_counts.py`
can ALSO be expressed as a tiny neural network — a single 27x27 weight matrix
with no bias and no hidden layer — that we *train* with gradient descent instead
of *counting*. Both methods end up at essentially the same model and the same
loss (~2.45). Counting is the shortcut; gradient descent is the general method
that will scale all the way up to a real Transformer later in the course.

The network, end to end, is just:
    x (a character id)
      -> one-hot vector of length 27
      -> matrix multiply by W (27x27)         # these are interpreted as log-counts
      -> exp() then normalize  = softmax      # turns scores into probabilities
      -> p (a distribution over the next character)

We use PyTorch's autograd (loss.backward()) to get the gradients for us. Treat
it as magic for now — in Chapter 2 we build that exact machinery from scratch.

Run it:
    python bigram_nn.py
    python bigram_nn.py --steps 300 --lr 50 --reg 0.01
"""

from pathlib import Path
import argparse
import torch
import torch.nn.functional as F

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "names.txt"


def load_words(path=DATA_PATH):
    lines = path.read_text().splitlines()
    return [w.strip() for w in lines if w.strip()]


def build_vocab(words):
    chars = sorted(set("".join(words)))
    stoi = {c: i + 1 for i, c in enumerate(chars)}
    stoi["."] = 0
    itos = {i: c for c, i in stoi.items()}
    return stoi, itos


def build_bigram_dataset(words, stoi):
    """Flatten every name into (input char, target char) pairs.

    For "emma" we get the examples: (., e) (e, m) (m, m) (m, a) (a, .)
    xs holds the inputs, ys holds the correct "next character" for each input.
    """
    xs, ys = [], []
    for w in words:
        chs = ["."] + list(w) + ["."]
        for ch1, ch2 in zip(chs, chs[1:]):
            xs.append(stoi[ch1])
            ys.append(stoi[ch2])
    return torch.tensor(xs), torch.tensor(ys)


def train(xs, ys, vocab_size, steps=200, lr=50.0, reg=0.01,
          seed=2147483647, log_every=20):
    """Train the single-layer network with full-batch gradient descent."""
    g = torch.Generator().manual_seed(seed)
    # W[i, j] is the score for "character j follows character i". After training,
    # W roughly equals log(counts), so exp(W) recovers the counts — which is why
    # this matches the counting model.
    W = torch.randn((vocab_size, vocab_size), generator=g, requires_grad=True)
    num = xs.nelement()
    print(f"training on {num} bigrams for {steps} steps (lr={lr}, reg={reg})")

    for k in range(steps):
        # ---- forward pass ----
        xenc = F.one_hot(xs, num_classes=vocab_size).float()  # (num, 27)
        logits = xenc @ W                                     # (num, 27) raw scores
        # softmax: exponentiate the scores ("counts") and normalize each row
        counts = logits.exp()
        probs = counts / counts.sum(1, keepdim=True)          # (num, 27)
        # negative log-likelihood of the correct next character, plus a small
        # regularization term that pulls W toward 0 (this is exactly smoothing:
        # it nudges the predicted distribution toward uniform).
        loss = -probs[torch.arange(num), ys].log().mean() + reg * (W ** 2).mean()

        # ---- backward pass ----
        W.grad = None       # reset gradients from the previous step
        loss.backward()     # autograd fills in W.grad (Chapter 2 demystifies this)

        # ---- update ----
        W.data += -lr * W.grad   # step downhill: nudge W to reduce the loss

        if k % log_every == 0 or k == steps - 1:
            print(f"  step {k:4d} | loss {loss.item():.4f}")

    return W


@torch.no_grad()
def sample_names(W, itos, vocab_size, n=10, seed=2147483647):
    """Generate names from the trained network (same loop, network instead of P)."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        ix = 0
        chars = []
        while True:
            xenc = F.one_hot(torch.tensor([ix]), num_classes=vocab_size).float()
            logits = xenc @ W
            probs = F.softmax(logits, dim=1)   # exp-and-normalize, done cleanly
            ix = torch.multinomial(probs, num_samples=1, generator=g).item()
            if ix == 0:
                break
            chars.append(itos[ix])
        out.append("".join(chars))
    return out


def main():
    parser = argparse.ArgumentParser(description="Bigram language model (neural net).")
    parser.add_argument("--steps", type=int, default=200, help="gradient-descent steps")
    parser.add_argument("--lr", type=float, default=50.0, help="learning rate")
    parser.add_argument("--reg", type=float, default=0.01, help="regularization (smoothing)")
    parser.add_argument("--num", type=int, default=10, help="how many names to sample")
    parser.add_argument("--seed", type=int, default=2147483647, help="random seed")
    args = parser.parse_args()

    words = load_words()
    stoi, itos = build_vocab(words)
    vocab_size = len(stoi)
    xs, ys = build_bigram_dataset(words, stoi)
    print(f"loaded {len(words)} names | vocab size = {vocab_size}")

    W = train(xs, ys, vocab_size, steps=args.steps, lr=args.lr,
              reg=args.reg, seed=args.seed)

    print("\nsampled names:")
    for name in sample_names(W, itos, vocab_size, n=args.num, seed=args.seed):
        print("  ", name)

    print("\nCompare the final loss to bigram_counts.py — they should be")
    print("almost identical (~2.45). Same model, two different ways to find it.")


if __name__ == "__main__":
    main()
