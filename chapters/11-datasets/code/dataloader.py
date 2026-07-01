"""
Stream batches from disk: memory-mapping + the get_batch pattern.
================================================================
Big corpora don't fit in RAM (FineWeb is ~15 trillion tokens = ~30 TB). The trick: **memory-map**
the .bin file — the OS pages in only the bytes you actually read, so you can sample from a file far
larger than memory as if it were an array. Each step we grab `batch_size` random windows of
`block_size+1` tokens; `x` is the window, `y` is the same window shifted one token right (the
next-token targets, exactly as since Chapter 1).

Run:  python dataloader.py
"""
import os
import json
import numpy as np
import torch

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def ensure_data():
    """Build the .bin shards if they're not there yet (so this runs from a fresh clone)."""
    if not os.path.exists(os.path.join(DATA_DIR, "train.bin")):
        from prepare import prepare, generate
        prepare(generate(2000, seed=0), DATA_DIR)


def get_batch(split, block_size=64, batch_size=16):
    """Sample a batch of (x, y) windows from the memory-mapped split. y is x shifted by one."""
    # re-open the memmap each call (recommended for memmap + many workers; it's cheap)
    data = np.memmap(os.path.join(DATA_DIR, f"{split}.bin"), dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - block_size, (batch_size,))              # random start offsets
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    return x, y


def load_decoder():
    meta = json.load(open(os.path.join(DATA_DIR, "meta.json")))
    itos = {int(i): c for i, c in meta["itos"].items()}
    return lambda ids: "".join(itos[int(i)] for i in ids)


if __name__ == "__main__":
    ensure_data()
    decode = load_decoder()

    train = np.memmap(os.path.join(DATA_DIR, "train.bin"), dtype=np.uint16, mode="r")
    print(f"train.bin is {train.nbytes / 1e6:.2f} MB on disk; memmap loads only what we read.\n")

    torch.manual_seed(0)
    x, y = get_batch("train", block_size=64, batch_size=16)
    print(f"one batch: x {tuple(x.shape)}, y {tuple(y.shape)}, dtype {x.dtype}")
    print(f"x[0] decodes to: {decode(x[0][:50].tolist())!r}")
    print(f"y[0] decodes to: {decode(y[0][:50].tolist())!r}")
    print("\nNote y is x shifted one character left — every position predicts the next token.")
