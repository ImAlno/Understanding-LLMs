"""
E04 — The .bin packing round-trips exactly.
===========================================
Packing writes token ids to disk as raw uint16 (2 bytes each, no header) and memmap reads them back.
This checks the round-trip is lossless and the file size is exactly what uint16 predicts.

Run:  python e04_packing.py
"""
import sys
import os
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))
import numpy as np
from synth_stories import generate

# tokenize a small corpus ourselves so we have the original ids to compare against
text = generate(500, seed=0)
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
ids = np.array([stoi[c] for c in text], dtype=np.uint16)

path = os.path.join(tempfile.mkdtemp(), "tokens.bin")
ids.tofile(path)                                          # write raw uint16

file_bytes = os.path.getsize(path)
print(f"{len(ids):,} tokens, vocab {len(chars)}")
print(f"file size: {file_bytes:,} bytes = {file_bytes / len(ids):.0f} bytes/token (uint16 -> 2)")
assert file_bytes == 2 * len(ids), "uint16 must be exactly 2 bytes/token"

back = np.memmap(path, dtype=np.uint16, mode="r")        # read it back
assert np.array_equal(np.asarray(back), ids), "round-trip changed the ids!"
print("✓ Round-trip is lossless, and the file is exactly 2 bytes/token.")

# what if you read it with the WRONG dtype? every value is garbage — there's no header to warn you.
wrong = np.memmap(path, dtype=np.uint8, mode="r")
print(f"\n(as a warning) reading the same bytes as uint8 gives {len(wrong):,} values, first few "
      f"{wrong[:5].tolist()} — nonsense. The dtype on read MUST match the dtype on write.")
