"""
Tokenize once, pack to disk: the .bin shard pipeline (nanoGPT-style).
====================================================================
You never re-tokenize text on every training run — you tokenize the whole corpus ONCE, store the
token ids as a flat binary file of `uint16`, and later memory-map it to sample batches (see
dataloader.py). This script builds that file: text -> ids -> train.bin / val.bin + meta.json.

We use a character-level tokenizer here to stay self-contained (a real run would use your Chapter 6
BPE). `uint16` holds ids 0..65535 — plenty for a char or BPE vocab, at 2 bytes per token.

Run:  python prepare.py
"""
import os
import json
import numpy as np
from synth_stories import generate

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def prepare(text, out_dir, val_frac=0.1):
    """Char-tokenize `text`, split train/val, and write .bin shards + meta.json. Returns stats."""
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    ids = np.array([stoi[c] for c in text], dtype=np.uint16)     # the whole corpus as token ids

    n_train = int(len(ids) * (1 - val_frac))
    train, val = ids[:n_train], ids[n_train:]                    # split BEFORE shuffling windows (no leak)

    os.makedirs(out_dir, exist_ok=True)
    train.tofile(os.path.join(out_dir, "train.bin"))             # raw uint16 bytes on disk
    val.tofile(os.path.join(out_dir, "val.bin"))
    meta = {"vocab_size": len(chars), "stoi": stoi, "itos": {i: c for c, i in stoi.items()}}
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"))
    return len(train), len(val), len(chars)


if __name__ == "__main__":
    text = generate(2000, seed=0)                                # our synthetic corpus
    n_train, n_val, vocab = prepare(text, DATA_DIR)
    mb = (n_train + n_val) * 2 / 1e6                             # 2 bytes/token
    print(f"corpus: {len(text):,} chars -> {n_train + n_val:,} tokens, vocab {vocab}")
    print(f"wrote train.bin ({n_train:,}) + val.bin ({n_val:,}) = {mb:.2f} MB of uint16 to {DATA_DIR}/")
    print("Tokenized once; every training run now just memory-maps these (see dataloader.py).")
