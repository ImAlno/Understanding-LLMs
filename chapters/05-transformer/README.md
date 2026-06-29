# Chapter 05 — The Transformer — Building a GPT

> We assemble attention, MLPs, residuals, and LayerNorm into a real **GPT** — and write our first *sentences*.

> 🚧 **Status: planned.** This chapter is scaffolded — the lesson, code,
> exercises, and project will be built out following the Chapter 1 template.
> The outline, concepts, and curated resources below are ready to learn from now.

## What you'll learn

This is the heart of the course. We stack **Transformer blocks** (multi-head attention + MLP, each wrapped in **residual connections** and **LayerNorm**) into a decoder-only **GPT-2-style** model, and train it on text. The Storyteller graduates from names to sentences.

## Key concepts

- the Transformer block
- residual (skip) connections
- LayerNorm
- decoder-only GPT architecture
- training & sampling a GPT

## 🔧 Mini-project (planned)

Build a character-level GPT (nanoGPT-style) and train it on **tiny shakespeare** (and a slice of **TinyStories**). Generate a paragraph — your model's first real prose.

## 📚 Resources

- [🎥 Karpathy — Let's build GPT, from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- [📄 Vaswani et al. 2017 — Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [💻 karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)


---

| ⬅️ Previous | Up | Next ➡️ |
|:--|:-:|--:|
| [Ch 04](../04-attention/) | [Syllabus](../../README.md#-syllabus) | [Ch 06](../06-tokenization/) |
