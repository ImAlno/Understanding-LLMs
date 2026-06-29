# Chapter 06 — Tokenization — Byte-Pair Encoding

> Characters are wasteful. Real LLMs work in **tokens** — and we build the tokenizer that makes them.

> 🚧 **Status: planned.** This chapter is scaffolded — the lesson, code,
> exercises, and project will be built out following the Chapter 1 template.
> The outline, concepts, and curated resources below are ready to learn from now.

## What you'll learn

We implement **byte-pair encoding (BPE)** from scratch (à la `minbpe`): start from raw bytes, repeatedly merge the most frequent pair, and build the vocabulary GPT-2/GPT-4 style. Then we re-tokenize our dataset and feel the efficiency gain.

## Key concepts

- bytes vs. characters vs. tokens (UTF-8)
- the BPE merge algorithm
- training a tokenizer
- encoding & decoding
- vocabulary size trade-offs

## 🔧 Mini-project (planned)

Build `minbpe`: train a BPE tokenizer on **TinyStories**, then swap it into your Chapter 5 GPT. Compare sequence lengths and loss: char-level vs. your BPE tokenizer.

## 📚 Resources

- [🎥 Karpathy — Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE)
- [📄 Sennrich et al. 2016 — Subword Units (BPE)](https://arxiv.org/abs/1508.07909)
- [💻 karpathy/minbpe](https://github.com/karpathy/minbpe)


---

| ⬅️ Previous | Up | Next ➡️ |
|:--|:-:|--:|
| [Ch 05](../05-transformer/) | [Syllabus](../../README.md#-syllabus) | [Ch 07](../07-optimization/) |
