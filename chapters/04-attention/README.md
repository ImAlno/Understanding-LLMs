# Chapter 04 — Attention — Letting the Model Look Back

> The single idea that made modern LLMs possible: every position can *attend* to every earlier position.

> 🚧 **Status: planned.** This chapter is scaffolded — the lesson, code,
> exercises, and project will be built out following the Chapter 1 template.
> The outline, concepts, and curated resources below are ready to learn from now.

## What you'll learn

We build **self-attention** from scratch: queries, keys, and values; the scaled dot-product; the causal mask that stops the model from peeking at the future; and **softmax** as the attention weighting. We also add **positional encodings** so the model knows *where* each token is.

## Key concepts

- queries, keys, values
- scaled dot-product attention
- the causal mask
- softmax as weighting
- positional encoding
- multi-head attention

## 🔧 Mini-project (planned)

Implement a single self-attention head and visualize its attention weights on a short string — literally see which earlier characters the model looks at when predicting the next one.

## 📚 Resources

- [🎥 Karpathy — Let's build GPT (attention section)](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- [📄 Bahdanau et al. 2014 — attention is born](https://arxiv.org/abs/1409.0473)
- [📄 Alammar — The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)


---

| ⬅️ Previous | Up | Next ➡️ |
|:--|:-:|--:|
| [Ch 03](../03-ngram-mlp/) | [Syllabus](../../README.md#-syllabus) | [Ch 05](../05-transformer/) |
