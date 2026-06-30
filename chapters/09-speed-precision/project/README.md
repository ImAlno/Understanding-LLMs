# ⚡ Mini-Project — Mixed-Precision Your GPT

You've seen what 16-bit formats can do; now **turn mixed precision on for a GPT** and measure the
payoff — the speedup, the memory drop, and (the part that makes it safe) a loss curve that's
*unchanged* from fp32.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ImAlno/Understanding-LLMs/blob/main/chapters/09-speed-precision/code/explore.ipynb)

> **Best run on Colab** (Runtime → Change runtime type → GPU) for the real speed and memory numbers
> — but it also runs on an Apple GPU (MPS) or a CPU, where you'll still see the **loss parity** (and
> a reminder that the *speedup* needs a CUDA GPU's Tensor Cores).

> **How this works:** [`starter/mp_gpt.py`](./starter/mp_gpt.py) trains a compact GPT (the Chapter 5
> architecture, from [`../code/gpt_small.py`](../code/gpt_small.py)) and has **one TODO** — wrap the
> forward pass in `autocast`. A reference is in [`solution/mp_gpt.py`](./solution/mp_gpt.py).

## 🎯 What it does

```bash
python starter/mp_gpt.py
```

Trains the GPT for 40 steps in **fp32**, then in **mixed precision** (bf16 autocast), from the *same
initialization*. It prints the two loss curves (they should track), the **ms/step** for each, and —
on a CUDA GPU — the **peak memory** for each.

## 🛠️ Your TODO

In `train()`, make the forward pass run under `autocast` when `mixed` is set:

```python
ctx = torch.autocast(device_type=device, dtype=dtype) if mixed else contextlib.nullcontext()
with ctx:
    logits, loss = model(xb, yb)
```

The `GradScaler` is already wired for you (it's a no-op for bf16; it does the loss scaling for
fp16). Until you fill the TODO, the script detects that mixed precision isn't on and asks you to.

## ✅ Checking your work

- **Loss parity** (visible everywhere): the fp32 and mixed curves should match to a few decimals —
  the whole point is that 16-bit math doesn't change the result. The script prints the max gap
  (~0.0005 on this model).
- **Speedup** (CUDA only): on a Colab T4 the mixed run is meaningfully faster; on a CPU/Apple GPU
  it's ~1× (no Tensor Cores) — and on a CPU it can even be *slower*, which is honest and expected.
- **Memory** (CUDA only): the script prints peak VRAM for each — mixed precision uses noticeably
  less.
- Compare against [`solution/mp_gpt.py`](./solution/mp_gpt.py).

## 🚀 Stretch

- **Try fp16 instead of bf16.** Pass `dtype=torch.float16` and *enable* the `GradScaler` (set its
  `enabled=...` true on CUDA). Watch the scaler's loss-scale adjust — and notice bf16 needed none of
  this.
- **Mixed-precision your *real* GPT.** Drop the same three-line change into your Chapter 5
  [`gpt.py`](../../05-transformer/code/gpt.py) training loop and confirm the Shakespeare loss curve
  is unchanged while each step is faster on a GPU.
- **Measure the memory yourself.** On Colab, print `torch.cuda.max_memory_allocated()/1e9` before and
  after switching to mixed precision and watch it drop.

## Other extensions

- **Push the batch size.** Mixed precision frees VRAM — see how much bigger a batch you can fit.
- **Time just the matmuls.** Confirm that the speedup lives in the linear layers (Tensor Cores),
  not the LayerNorms or softmax (which stay fp32).

Next: [Chapter 10 — Need for Speed III: Distributed](../../10-speed-distributed/), where we scale
across *many* GPUs at once. 🏎️
