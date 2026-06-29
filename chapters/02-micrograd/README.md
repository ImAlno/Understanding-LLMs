# Chapter 02 — Micrograd — Backpropagation from Scratch

> We open the black box. In Chapter 1 we said `loss.backward()` was magic — now we *build* that magic ourselves.

> 🚧 **Status: planned.** This chapter is scaffolded — the lesson, code,
> exercises, and project will be built out following the Chapter 1 template.
> The outline, concepts, and curated resources below are ready to learn from now.

## What you'll learn

You'll build a tiny **autograd engine**: a `Value` object that remembers how it was computed and can compute its own gradient by the **chain rule**. Then you'll train a small neural net with it — the exact mechanism PyTorch uses, in ~100 lines you wrote.

## Key concepts

- the chain rule & computational graphs
- forward vs. backward pass
- building a `Value` autograd class
- manual gradient descent
- what a neuron / MLP really is

## 🔧 Mini-project (planned)

Build your own `micrograd`: a scalar autograd engine + a small MLP, and train it to fit a toy dataset. Then re-derive Chapter 1's bigram gradients by hand to prove you understand `.backward()`.

## 📚 Resources

- [🎥 Karpathy — building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0)
- [💻 karpathy/micrograd](https://github.com/karpathy/micrograd)
- [📄 Olah — Calculus on Computational Graphs](https://colah.github.io/posts/2015-08-Backprop/)


---

| ⬅️ Previous | Up | Next ➡️ |
|:--|:-:|--:|
| [Ch 1 — Bigram](../01-bigram/) | [Syllabus](../../README.md#-syllabus) | [Ch 03](../03-ngram-mlp/) |
