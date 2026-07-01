# Chapter 13 — Exercises

Build quantization and feel the trade-offs. **Try first**; solutions are in [`solutions/`](./solutions/).
Everything runs on a CPU in a second. E04 imports the GPT from [`../code/gpt.py`](../code/gpt.py).

```bash
python chapters/13-inference-quantization/exercises/solutions/e01_quantize.py
```

---

### E01 — Implement quantize/dequantize 🥇
The core map: pick a scale, round to int8, multiply back — and land within a rounding step of the
originals.

- 🧩 **Scaffold:** [`starter/e01_quantize.py`](./starter/e01_quantize.py).
- ✅ **Solution:** [`solutions/e01_quantize.py`](./solutions/e01_quantize.py).

  <details><summary>💡 Hint</summary>

  `scale = w.abs().max()/127`; `q = torch.clamp(torch.round(w/scale), -128, 127)`; `w_hat = q*scale`.
  </details>

### E02 — Per-channel beats per-tensor
Add outlier rows to a weight matrix and measure the reconstruction error two ways: one scale for the
whole tensor vs one per row. Per-channel wins by a lot.

- ✅ Solution: [`solutions/e02_per_channel.py`](./solutions/e02_per_channel.py).

  <details><summary>💡 Hint</summary>

  Per-channel scale is `w.abs().amax(dim=1, keepdim=True) / qmax` — one per row. It's ~7× less error
  at int8 here.
  </details>

### E03 — Bit width vs error and size
Sweep int8 → int2 and watch the error climb as the size falls. Below int4 the quality cost gets steep.

- ✅ Solution: [`solutions/e03_bits.py`](./solutions/e03_bits.py).

  <details><summary>💡 Hint</summary>

  `levels = 2**bits`; a 7B model is `7e9 * bits/8` bytes. int8 ≈ 0.007 error; int2 ≈ 0.8.
  </details>

### E04 — Quantize a model, measure the drift
Quantize every `nn.Linear` weight of the GPT to int8, then int4, and measure how far the output
logits move from fp32 (the real quality signal) alongside the size drop.

- ✅ Solution: [`solutions/e04_quantize_model.py`](./solutions/e04_quantize_model.py).

  <details><summary>💡 Hint</summary>

  Loop over `model.modules()`, and for each `nn.Linear` replace `mod.weight.data` with its
  per-channel quantized-dequantized version. int8 drift ≈ 0.004; int4 ≈ 0.07.
  </details>

> 🧠 **Takeaway:** quantization is `q = round(w/scale)` back to `q·scale`; **per-channel** handles
> outliers; **fewer bits** trade quality for size; and what matters is the **output drift**, not the
> weight error — int8 is near-free, int4 is the aggressive-but-workable floor.
