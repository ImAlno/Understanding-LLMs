# Chapter 12 — Inference I: The KV-Cache

> Generation is slow if you recompute everything every step. The KV-cache fixes that.

> 🚧 **Status: planned.** This chapter is scaffolded — the lesson, code,
> exercises, and project will be built out following the Chapter 1 template.
> The outline, concepts, and curated resources below are ready to learn from now.

## What you'll learn

When generating token by token, the keys and values of past tokens never change — so we **cache** them. We add a KV-cache to our GPT's `generate()` and turn quadratic re-computation into a fast incremental decode, the core trick behind every chat UI's snappy responses.

## Key concepts

- the prefill vs. decode phases
- caching keys & values
- incremental attention
- memory cost of the cache
- latency vs. throughput

## 🔧 Mini-project (planned)

Add a KV-cache to your Chapter 5 GPT's sampling loop. Benchmark tokens/second with and without it, and reason about the memory the cache consumes.

## 📚 Resources

- [📄 HuggingFace — KV Caching Explained](https://huggingface.co/blog/not-lain/kv-caching)
- [📄 Pope et al. 2022 — Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102)


---

| ⬅️ Previous | Up | Next ➡️ |
|:--|:-:|--:|
| [Ch 11](../11-datasets/) | [Syllabus](../../README.md#-syllabus) | [Ch 13](../13-inference-quantization/) |
