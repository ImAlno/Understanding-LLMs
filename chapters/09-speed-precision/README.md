# Chapter 09 — Need for Speed II: Precision — fp16, bf16, fp8

> Same model, half the bits, twice the speed. We train in **mixed precision**.

> 🚧 **Status: planned.** This chapter is scaffolded — the lesson, code,
> exercises, and project will be built out following the Chapter 1 template.
> The outline, concepts, and curated resources below are ready to learn from now.

## What you'll learn

We explore numeric formats (**fp32, fp16, bf16, fp8**), why lower precision is faster and lighter, and the tricks that keep it stable: **loss scaling** and an fp32 master copy. Then we add automatic mixed precision (AMP) to our training loop.

## Key concepts

- floating-point formats (fp32/fp16/bf16/fp8)
- why low precision is faster
- loss scaling
- fp32 master weights
- torch.amp / autocast

## 🔧 Mini-project (planned)

Add mixed-precision (bf16/fp16) training to your GPT with `torch.amp`. Measure the speedup and memory savings, and confirm the loss curve is unchanged.

## 📚 Resources

- [📄 Micikevicius et al. 2017 — Mixed Precision Training](https://arxiv.org/abs/1710.03740)
- [📄 PyTorch — What Every User Should Know About Mixed Precision](https://pytorch.org/blog/what-every-user-should-know-about-mixed-precision-training-in-pytorch/)


---

| ⬅️ Previous | Up | Next ➡️ |
|:--|:-:|--:|
| [Ch 08](../08-speed-device/) | [Syllabus](../../README.md#-syllabus) | [Ch 10](../10-speed-distributed/) |
