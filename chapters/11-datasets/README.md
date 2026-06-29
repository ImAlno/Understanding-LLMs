# Chapter 11 — Datasets — Feeding the Storyteller

> A model is what it eats. We build the data pipeline that turns raw text into training batches.

> 🚧 **Status: planned.** This chapter is scaffolded — the lesson, code,
> exercises, and project will be built out following the Chapter 1 template.
> The outline, concepts, and curated resources below are ready to learn from now.

## What you'll learn

We assemble the **TinyStories** pipeline end-to-end: download, clean, tokenize (Chapter 6), pack into efficient binary shards, and stream **batches** to the GPU without bottlenecking. We also touch synthetic data generation — how the Storyteller's stories were made in the first place.

## Key concepts

- data loading & batching
- tokenize-and-pack pipelines
- streaming large datasets
- TinyStories & FineWeb
- synthetic data generation

## 🔧 Mini-project (planned)

Build the Storyteller's data loader: download TinyStories, tokenize with your BPE tokenizer, write `.bin` shards, and implement a fast batch sampler. This is the pretraining fuel for the capstone.

## 📚 Resources

- [📄 Eldan & Li 2023 — TinyStories](https://arxiv.org/abs/2305.07759)
- [📄 HuggingFace — FineWeb](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1)
- [📄 Gao et al. 2020 — The Pile](https://arxiv.org/abs/2101.00027)


---

| ⬅️ Previous | Up | Next ➡️ |
|:--|:-:|--:|
| [Ch 10](../10-speed-distributed/) | [Syllabus](../../README.md#-syllabus) | [Ch 12](../12-inference-kv-cache/) |
