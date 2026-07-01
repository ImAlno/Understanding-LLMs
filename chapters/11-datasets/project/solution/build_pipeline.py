"""
The Storyteller's Data Pipeline — SOLUTION.
===========================================
Assemble the whole pipeline end to end: generate a corpus, pack it into .bin shards, implement the
batch sampler, and prove the batches are training-ready by feeding one through a fresh embedding and
checking the loss is ~ln(vocab) (an untrained model's baseline). This is the fuel line for the GPT.

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
    """Sample a batch of (x, y) windows from the memory-mapped split; y is x shifted one right."""
    data = np.memmap(os.path.join(D, f"{split}.bin"), dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - BLOCK, (BATCH,))
    x = torch.stack([torch.from_numpy(data[i:i + BLOCK].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + BLOCK].astype(np.int64)) for i in ix])
    return x, y


def main():
    D = tempfile.mkdtemp()                                        # a real run would use a persistent dir

    # 1) generate + 2) pack
    n_train, n_val, vocab = prepare(generate(2000, seed=0), D)
    print(f"1-2) packed {n_train:,} train + {n_val:,} val tokens, vocab {vocab}")

    # 3) sample a batch
    torch.manual_seed(0)
    x, y = get_batch(D, "train")
    print(f"3)   sampled a batch: x {tuple(x.shape)}, y {tuple(y.shape)}, dtype {x.dtype}")

    # 4) decode it back to text (sanity: it's real stories)
    itos = {int(i): c for i, c in json.load(open(os.path.join(D, "meta.json")))["itos"].items()}
    decode = lambda t: "".join(itos[int(i)] for i in t)
    print(f"4)   x[0] decodes to: {decode(x[0][:48].tolist())!r}")

    # 5) training-readiness: feed the batch through a fresh model; loss should be ~ln(vocab)
    torch.manual_seed(0)
    model = nn.Sequential(nn.Embedding(vocab, 32), nn.Linear(32, vocab))
    logits = model(x)
    loss = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1))
    print(f"5)   a fresh model's loss on this batch: {loss.item():.3f}  (~ln(vocab) = {math.log(vocab):.3f})")

    assert abs(loss.item() - math.log(vocab)) < 0.5, "untrained loss should be near ln(vocab)"
    print("\n✅ Pipeline works end to end — batches decode to real stories AND feed a model cleanly.")
    print("   Point a Chapter 5 GPT's training loop at get_batch and it will train.")


if __name__ == "__main__":
    main()
