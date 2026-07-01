# 🗜️ Mini-Project — Quantize Your Storyteller

You've built the quantization map; now **apply it to a whole GPT** and measure the trade-off that
decides how a model ships: how much **smaller** it gets, and how much its **output drifts** from the
fp32 original.

> 💻 **No GPU needed** — quantization is arithmetic on weight tensors. (The *size* and *quality* run
> honestly on a CPU; the *speed* win of quantization needs int8/int4 matmul kernels or a GPU, so we
> measure the two we can here.)

> **How this works:** [`starter/quantize_storyteller.py`](./starter/quantize_storyteller.py) walks a
> GPT's `nn.Linear` layers and measures size + logit drift for you. It has **one TODO** — the
> per-channel `quantize_dequantize` of a weight. A reference is in
> [`solution/quantize_storyteller.py`](./solution/quantize_storyteller.py).

## 🎯 What it does

```bash
python starter/quantize_storyteller.py
```

Quantizes every Linear weight of the GPT ([`../code/gpt.py`](../code/gpt.py)) to int8, then int4, and
prints a three-row table: **format, size, and logit drift** vs the fp32 model. With the map filled in
you'll see int8 halve the size for a ~0.4% drift, and int4 quarter it for a few percent.

## 🛠️ Your TODO

Implement `quantize_dequantize(w, bits)` — per-channel symmetric:

```python
scale = w.abs().amax(dim=1, keepdim=True) / qmax
return torch.clamp(torch.round(w / scale), -qmax - 1, qmax) * scale
```

Until you do, it returns the weight unchanged, the output doesn't move, and the script detects that
and asks you to fill it.

## ✅ Checking your work

- **Size** halves at int8 and quarters at int4 (pure arithmetic — exact).
- **Logit drift** should be tiny at int8 (~0.004) and a few percent at int4 (~0.07) — small enough
  that int8 text is indistinguishable from fp16, which is the whole point.
- **Unfilled → refusal:** if the weights aren't actually quantized, the output is identical and the
  script stops and tells you (no false success).
- Compare against [`solution/quantize_storyteller.py`](./solution/quantize_storyteller.py).

## 🚀 Stretch

- **Quantize your real Chapter 5 Storyteller.** Load your trained GPT, quantize it, and *generate* —
  read the int8 and int4 stories side by side with the fp16 ones and judge the quality yourself.
- **Per-tensor vs per-channel on the model.** Switch the quant to per-tensor and watch the int4 drift
  jump — the outlier problem, on a real model.
- **Sweep the bit width.** Add int6, int3, int2 rows and find where the drift stops being acceptable.
- **Weight-only vs also-activations.** Try quantizing the activations too (harder — they have wilder
  outliers) and see the drift grow, which is why LLM inference is usually weight-only.

Next: [Chapter 14 — Finetuning I: SFT](../../14-finetuning-sft/), where we stop making the model
*cheaper* and start making it *do what we want*. 🗜️
