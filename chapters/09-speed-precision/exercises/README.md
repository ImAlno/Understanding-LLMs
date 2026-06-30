# Chapter 9 — Exercises

Feel the formats. **Try first**; solutions are in [`solutions/`](./solutions/). Unlike Chapter 8,
**these all run fully on a CPU** — number formats behave identically everywhere. (Only the *speedup*
needs a GPU, and that's the mini-project.)

```bash
python chapters/09-speed-precision/exercises/solutions/e01_memory.py
```

---

### E01 — Half the bits, half the memory 🥇
A 16-bit number is 2 bytes; fp32 is 4. Compute the storage cost of a 100M-parameter model in fp32
vs fp16 vs bf16, and confirm the 16-bit formats use half.

- 🧩 **Scaffold:** [`starter/e01_memory.py`](./starter/e01_memory.py).
- ✅ **Solution:** [`solutions/e01_memory.py`](./solutions/e01_memory.py).

  <details><summary>💡 Hint</summary>

  Bytes per number is `torch.zeros(1, dtype=dt).element_size()` (4 for fp32, 2 for fp16/bf16); the
  tensor's cost is that times the number of elements.
  </details>

### E02 — Find fp16's overflow cliff
fp16's max is 65504. Probe powers of 2 to find where each format overflows to `inf`, and confirm
bf16 (fp32's range) survives values that kill fp16.

- ✅ Solution: [`solutions/e02_overflow.py`](./solutions/e02_overflow.py).

  <details><summary>💡 Hint</summary>

  Loop `e` upward while `torch.isfinite(torch.tensor(2.0**e, dtype=dt))`. fp16 tops out around
  2¹⁶; bf16 around 2¹²⁸ (same as fp32).
  </details>

### E03 — How big a loss scale do you need?
A `1e-8` gradient underflows fp16 to 0. Scale it up before storing, unscale after, and find which
scales rescue it — too small doesn't help, big enough does.

- ✅ Solution: [`solutions/e03_loss_scaling.py`](./solutions/e03_loss_scaling.py).

  <details><summary>💡 Hint</summary>

  For each scale `s`: `recovered = torch.tensor(g * s, dtype=torch.float16).item() / s`. It stays 0
  until `s` lifts `g` into fp16's representable range (around `s = 16` for `g = 1e-8`).
  </details>

### E04 — What autocast does to each op
`autocast` isn't a blanket cast. Run a matmul, a softmax, a layernorm, and a sum inside an autocast
block and print each output dtype — the matmul goes 16-bit, the sensitive ops stay fp32.

- ✅ Solution: [`solutions/e04_autocast_dtypes.py`](./solutions/e04_autocast_dtypes.py).

  <details><summary>💡 Hint</summary>

  Inside `with torch.autocast(device_type=device, dtype=torch.bfloat16):`, check each result's
  `.dtype`. Matmul → `bfloat16`; softmax / layer_norm / sum → `float32`.
  </details>

> 🧠 **Takeaway:** fp16 = fine precision but a short ruler (overflow/underflow); bf16 = long ruler,
> coarse steps (swamping). Loss scaling rescues fp16's small gradients; autocast picks the safe
> precision per op so you don't have to.
