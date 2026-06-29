# Chapter 13 — Inference II: Quantization

> Shrink the model so it runs on modest hardware — with almost no quality loss.

> 🚧 **Status: planned.** This chapter is scaffolded — the lesson, code,
> exercises, and project will be built out following the Chapter 1 template.
> The outline, concepts, and curated resources below are ready to learn from now.

## What you'll learn

We compress trained weights from 16/32-bit floats down to **8-bit (or 4-bit) integers**. We cover the core idea (scale/zero-point), outlier handling (**LLM.int8()**), and one-shot methods (**GPTQ, AWQ**). *Note:* this is an add-on optimization beyond the nanochat baseline, but essential for real deployment.

## Key concepts

- int8/int4 quantization basics
- scales & zero-points
- outlier features (LLM.int8())
- post-training quantization (GPTQ, AWQ)
- quality vs. size trade-offs

## 🔧 Mini-project (planned)

Quantize your Storyteller to int8 and measure the three-way trade-off: model size, tokens/second, and loss/sample quality versus the fp16 baseline.

## 📚 Resources

- [📄 Dettmers et al. 2022 — LLM.int8()](https://arxiv.org/abs/2208.07339)
- [📄 Frantar et al. 2022 — GPTQ](https://arxiv.org/abs/2210.17323)
- [📄 HuggingFace — A Gentle Intro to 8-bit Matmul](https://huggingface.co/blog/hf-bitsandbytes-integration)


---

| ⬅️ Previous | Up | Next ➡️ |
|:--|:-:|--:|
| [Ch 12](../12-inference-kv-cache/) | [Syllabus](../../README.md#-syllabus) | [Ch 14](../14-finetuning-sft/) |
