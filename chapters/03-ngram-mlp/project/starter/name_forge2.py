"""
The Name Forge MK II — STARTER scaffold.
========================================
Fill in the two TODOs (the training loop and the sampling step). Everything else — data,
vocab, parameters, the CLI, temperature, and the --start handling — is done for you.
A full reference is in ../solution/name_forge2.py.

    python name_forge2.py --n 10 --start k --temperature 0.8
"""
import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))

import random
import torch
import torch.nn.functional as F
from mlp import load_words, build_vocab, build_dataset

block_size, n_embd, n_hidden = 3, 10, 200


def main():
    ap = argparse.ArgumentParser(description="The Name Forge MK II — MLP name generator.")
    ap.add_argument("--n", type=int, default=10, help="how many names to forge")
    ap.add_argument("--start", type=str, default=None, help="force a starting letter")
    ap.add_argument("--temperature", type=float, default=1.0, help="creativity (>1 wilder)")
    ap.add_argument("--steps", type=int, default=20000, help="training steps")
    ap.add_argument("--seed", type=int, default=2147483647, help="random seed")
    args = ap.parse_args()

    words = load_words()
    stoi, itos = build_vocab(words)
    V = len(stoi)
    if args.start is not None:
        args.start = args.start.lower()
        if len(args.start) != 1 or args.start not in stoi or args.start == ".":
            raise SystemExit("--start must be a single letter a-z")

    Xtr, Ytr = build_dataset(words, stoi, block_size)

    g = torch.Generator().manual_seed(args.seed)
    C = torch.randn((V, n_embd), generator=g)
    W1 = torch.randn((block_size * n_embd, n_hidden), generator=g) * 0.2
    b1 = torch.randn(n_hidden, generator=g) * 0.01
    W2 = torch.randn((n_hidden, V), generator=g) * 0.01
    b2 = torch.zeros(V)
    params = [C, W1, b1, W2, b2]
    for p in params:
        p.requires_grad = True

    def forward(X):
        emb = C[X]
        h = torch.tanh(emb.view(emb.shape[0], -1) @ W1 + b1)
        return h @ W2 + b2

    print(f"forging the model ({args.steps} steps)...")
    loss = None
    for step in range(args.steps):
        ix = torch.randint(0, Xtr.shape[0], (32,), generator=g)
        lr = 0.1 if step < 0.6 * args.steps else 0.01
        # ✍️ TODO #1: the standard training step you know by heart — measure the minibatch
        #             loss, backprop it, and step every parameter downhill (use lr).
        if loss is None:
            raise SystemExit("Fill in TODO #1 (the training step) first, then run again.")
    print(f"done (final minibatch loss {loss.item():.3f})\n")

    @torch.no_grad()
    def sample():
        if args.start:
            context = [0] * (block_size - 1) + [stoi[args.start]]
            out = [args.start]
        else:
            context = [0] * block_size
            out = []
        while True:
            probs = F.softmax(forward(torch.tensor([context])), dim=1)[0]
            if args.temperature != 1.0:
                probs = probs.clamp_min(1e-9) ** (1.0 / args.temperature)
                probs = probs / probs.sum()
            # ✍️ TODO #2: sample the next id from probs (a torch.multinomial call, then
            #             .item()), then slide the context window (drop oldest, append ix).
            ix = 0                # replace this
            context = context     # replace this
            if ix == 0:
                break
            out.append(itos[ix])
        return "".join(out)

    print(f"🔥 The Name Forge MK II — {args.n} names (temp={args.temperature})")
    names = [sample() for _ in range(args.n)]
    if all(nm == "" for nm in names):
        print("  (all names came out empty — fill in TODO #2, the sampling step)")
    for nm in names:
        print("  ", nm)


if __name__ == "__main__":
    main()
