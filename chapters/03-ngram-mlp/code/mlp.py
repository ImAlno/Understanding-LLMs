"""
The N-gram MLP — a neural language model with embeddings.
=========================================================
Chapter 1's bigram model saw only ONE previous character. This model sees the previous
`block_size` characters, turns each into a learned **embedding** vector, and feeds them
through a small **multi-layer perceptron** (MLP) to predict the next character. It's the
architecture from Bengio et al. (2003), and Karpathy's "makemore Part 2".

Everything is now a PyTorch **tensor** (a whole array at once) instead of one scalar at a
time like micrograd — the same autograd, just fast. The model trains to a markedly lower
loss than the bigram, and the names it dreams up are noticeably more name-like.

Run:  python mlp.py
"""
from pathlib import Path
import random
import torch
import torch.nn.functional as F

DATA = Path(__file__).resolve().parent.parent / "data" / "names.txt"

# ---- config (the knobs you'll tune in the exercises) ----
block_size = 3       # how many previous characters feed each prediction (the context)
n_embd = 10          # size of each character's embedding vector
n_hidden = 200       # neurons in the hidden layer
max_steps = 30000    # training steps (minibatches)
batch_size = 32


def load_words():
    return [w.strip() for w in DATA.read_text().splitlines() if w.strip()]


def build_vocab(words):
    chars = sorted(set("".join(words)))
    stoi = {c: i + 1 for i, c in enumerate(chars)}
    stoi["."] = 0
    itos = {i: c for c, i in stoi.items()}
    return stoi, itos


def build_dataset(words, stoi, block_size):
    """Turn names into (context, next-char) pairs.

    For "emma" with block_size=3 we slide a 3-char window, padded with the '.' token (id
    0) at the start:  [...] -> 'e',  [..e] -> 'm',  [.em] -> 'm',  [emm] -> 'a',  ...
    X holds the context ids (N, block_size); Y holds the next-char id (N,).
    """
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + ".":
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]      # slide the window forward
    return torch.tensor(X), torch.tensor(Y)


def main():
    words = load_words()
    stoi, itos = build_vocab(words)
    vocab_size = len(stoi)

    # 80/10/10 split (the discipline from Chapter 1's exercises)
    random.seed(42)
    random.shuffle(words)
    n1, n2 = int(0.8 * len(words)), int(0.9 * len(words))
    Xtr, Ytr = build_dataset(words[:n1], stoi, block_size)
    Xdev, Ydev = build_dataset(words[n1:n2], stoi, block_size)
    Xte, Yte = build_dataset(words[n2:], stoi, block_size)
    print(f"{len(words)} names | train {Xtr.shape[0]} / dev {Xdev.shape[0]} examples")

    # ---- parameters ----
    g = torch.Generator().manual_seed(2147483647)
    C = torch.randn((vocab_size, n_embd), generator=g)                 # the embedding table
    W1 = torch.randn((block_size * n_embd, n_hidden), generator=g) * 0.2
    b1 = torch.randn(n_hidden, generator=g) * 0.01
    # small output weights → first predictions are near-uniform, so the loss starts around
    # log(27) ≈ 3.3 instead of exploding. (Chapter 7 is all about initialization.)
    W2 = torch.randn((n_hidden, vocab_size), generator=g) * 0.01
    b2 = torch.zeros(vocab_size)
    params = [C, W1, b1, W2, b2]
    for p in params:
        p.requires_grad = True
    print(f"{sum(p.nelement() for p in params)} parameters\n")

    def forward(X):
        emb = C[X]                                   # (N, block_size, n_embd) — look up embeddings
        h = torch.tanh(emb.view(emb.shape[0], -1) @ W1 + b1)   # flatten context, hidden layer
        return h @ W2 + b2                           # (N, vocab_size) logits

    # ---- train ----
    for step in range(max_steps):
        ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)   # a random minibatch
        logits = forward(Xtr[ix])
        loss = F.cross_entropy(logits, Ytr[ix])
        for p in params:
            p.grad = None
        loss.backward()
        lr = 0.1 if step < 0.6 * max_steps else 0.01    # simple step decay
        for p in params:
            p.data += -lr * p.grad
        if step % 3000 == 0 or step == max_steps - 1:
            print(f"step {step:5d}/{max_steps} | minibatch loss {loss.item():.4f}")

    # ---- evaluate on full splits ----
    @torch.no_grad()
    def split_loss(X, Y):
        return F.cross_entropy(forward(X), Y).item()
    print(f"\ntrain loss {split_loss(Xtr, Ytr):.4f} | dev loss {split_loss(Xdev, Ydev):.4f}")
    print("(bigram was ~2.45; this MLP should land around ~2.1–2.2)")

    # ---- sample some names ----
    print("\nsampled names:")
    gen = torch.Generator().manual_seed(2147483647 + 10)
    for _ in range(15):
        out, context = [], [0] * block_size
        while True:
            logits = forward(torch.tensor([context]))
            probs = F.softmax(logits, dim=1)
            ix = torch.multinomial(probs, num_samples=1, generator=gen).item()
            context = context[1:] + [ix]
            if ix == 0:
                break
            out.append(itos[ix])
        print("  ", "".join(out))


if __name__ == "__main__":
    main()
