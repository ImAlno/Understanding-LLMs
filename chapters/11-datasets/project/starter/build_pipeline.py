"""
The Storyteller's Data Pipeline — STARTER scaffold.
===================================================
Fill the one TODO: implement the batch sampler in get_batch. The generate/pack and the
training-readiness check are done for you. Reference: ../solution/build_pipeline.py.

Run:  python build_pipeline.py
"""
import sys
import os
import json
import math
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from synth_stories import generate
from prepare import prepare

BLOCK, BATCH = 64, 16


def get_batch(D, split):
    data = np.memmap(os.path.join(D, f"{split}.bin"), dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - BLOCK, (BATCH,))
    # ✍️ TODO: build x (windows of BLOCK tokens) and y (each shifted one right), int64, torch.stack'd.
    #          x = data[i : i+BLOCK] for each i in ix;  y = data[i+1 : i+1+BLOCK].
    x = None      # replace
    y = None      # replace
    return x, y


def main():
    D = tempfile.mkdtemp()
    n_train, n_val, vocab = prepare(generate(2000, seed=0), D)
    print(f"1-2) packed {n_train:,} train + {n_val:,} val tokens, vocab {vocab}")

    torch.manual_seed(0)
    x, y = get_batch(D, "train")
    if x is None or y is None:
        raise SystemExit("Fill the ✍️ TODO in get_batch (build x and y), then run again.")
    print(f"3)   sampled a batch: x {tuple(x.shape)}, y {tuple(y.shape)}, dtype {x.dtype}")

    itos = {int(i): c for i, c in json.load(open(os.path.join(D, "meta.json")))["itos"].items()}
    decode = lambda t: "".join(itos[int(i)] for i in t)
    print(f"4)   x[0] decodes to: {decode(x[0][:48].tolist())!r}")

    torch.manual_seed(0)
    model = nn.Sequential(nn.Embedding(vocab, 32), nn.Linear(32, vocab))
    loss = F.cross_entropy(model(x).reshape(-1, vocab), y.reshape(-1))
    print(f"5)   a fresh model's loss on this batch: {loss.item():.3f}  (~ln(vocab) = {math.log(vocab):.3f})")
    assert abs(loss.item() - math.log(vocab)) < 0.5, "untrained loss should be near ln(vocab)"
    print("\n✅ Pipeline works end to end — point a GPT's training loop at get_batch and it trains.")


if __name__ == "__main__":
    main()
