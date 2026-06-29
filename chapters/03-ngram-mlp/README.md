# Chapter 03 — The N-gram MLP — More Context, Real Embeddings

> Our Storyteller learns to remember more than one character — and discovers *embeddings*.

> 🚧 **Status: planned.** This chapter is scaffolded — the lesson, code,
> exercises, and project will be built out following the Chapter 1 template.
> The outline, concepts, and curated resources below are ready to learn from now.

## What you'll learn

We rebuild the language model as a **multi-layer perceptron** (Bengio 2003): look up a learned **embedding** for each of the previous few characters, concatenate them, pass through a hidden layer with a **GELU** nonlinearity, and predict the next character. Names get dramatically better.

## Key concepts

- word/character embeddings
- the MLP: linear layers + nonlinearity
- matrix multiplication (matmul)
- GELU activation
- minibatches & the train/dev/test discipline

## 🔧 Mini-project (planned)

Upgrade the Name Forge into an MLP that uses a context of N previous characters. Tune the embedding size and context length on a dev set; watch the loss drop below the trigram's ~2.1.

## 📚 Resources

- [🎥 Karpathy — makemore Part 2: MLP](https://www.youtube.com/watch?v=TCH_1BHY58I)
- [📄 Bengio et al. 2003 — A Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
- [📄 Hendrycks & Gimpel 2016 — GELU](https://arxiv.org/abs/1606.08415)


---

| ⬅️ Previous | Up | Next ➡️ |
|:--|:-:|--:|
| [Ch 02](../02-micrograd/) | [Syllabus](../../README.md#-syllabus) | [Ch 04](../04-attention/) |
