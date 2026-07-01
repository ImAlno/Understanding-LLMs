# Chapter 11 — Exercises

Build the data pipeline's moving parts. **Try first**; solutions are in [`solutions/`](./solutions/).
Everything runs on a plain CPU in a second — each exercise generates a small synthetic corpus, packs
it to a temp `.bin`, and works from there (no downloads).

```bash
python chapters/11-datasets/exercises/solutions/e01_get_batch.py
```

---

### E01 — Implement `get_batch` 🥇
The heart of the pipeline: memory-map the `.bin`, sample random windows, and return `(x, y)` with `y`
shifted one token right.

- 🧩 **Scaffold:** [`starter/e01_get_batch.py`](./starter/e01_get_batch.py).
- ✅ **Solution:** [`solutions/e01_get_batch.py`](./solutions/e01_get_batch.py).

  <details><summary>💡 Hint</summary>

  `x = torch.stack([torch.from_numpy(data[i:i+block].astype(np.int64)) for i in ix])`, and `y` the
  same with `data[i+1:i+1+block]`.
  </details>

### E02 — See the x → y shift
Decode a window and print `x` above `y` so you can *see* that every `x` token points at the token
below it in `y` — the "predict the next token" signal made visible.

- ✅ Solution: [`solutions/e02_shift.py`](./solutions/e02_shift.py).

  <details><summary>💡 Hint</summary>

  `y` starts one token later than `x`, so `decode(x[1:])` should equal `decode(y[:-1])`.
  </details>

### E03 — Prove the split doesn't leak
Show that training windows sampled from `train.bin` can never reach a validation token — because the
split happens *before* windowing.

- ✅ Solution: [`solutions/e03_no_leak.py`](./solutions/e03_no_leak.py).

  <details><summary>💡 Hint</summary>

  A train window starts at most at `len(train) - block`, so its last index is `< len(train)` — the
  val region (which begins at `len(train)`) is unreachable.
  </details>

### E04 — The packing round-trips
Write ids to a `.bin` as `uint16`, read them back with `memmap`, and confirm nothing changed — and
that the file is exactly 2 bytes/token. Then read the same bytes as `uint8` and watch it turn to
garbage.

- ✅ Solution: [`solutions/e04_packing.py`](./solutions/e04_packing.py).

  <details><summary>💡 Hint</summary>

  `os.path.getsize(path) == 2 * len(ids)`; `np.array_equal(np.asarray(memmap), ids)`. The dtype on
  read must match the dtype on write — there's no header.
  </details>

> 🧠 **Takeaway:** `get_batch` samples random windows from a memory-mapped `.bin`; `y` is `x` shifted
> by one; the split is leak-free by construction; and `uint16` packing is a lossless 2-bytes/token
> round-trip.
