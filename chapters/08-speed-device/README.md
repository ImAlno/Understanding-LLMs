# Chapter 08 — Need for Speed I: Device — CPU, GPU, CUDA

> Where does the compute actually happen? We move onto the GPU and write our first CUDA.

> 🚧 **Status: planned.** This chapter is scaffolded — the lesson, code,
> exercises, and project will be built out following the Chapter 1 template.
> The outline, concepts, and curated resources below are ready to learn from now.

## What you'll learn

We learn how tensors live in memory (**shapes, strides, contiguity**), what makes a GPU fast, and the basics of **CUDA** kernels (threads, blocks, the memory hierarchy). This is the first of three chapters chasing speed — and our first step toward the C/CUDA in `llm.c`.

## Key concepts

- CPU vs. GPU execution model
- tensor memory: shapes, strides, contiguous
- CUDA threads, blocks, grids
- the GPU memory hierarchy
- moving training to a GPU

## 🔧 Mini-project (planned)

Profile your GPT's training step, move it to a GPU (Colab is fine), and write a simple CUDA (or Triton) kernel for an operation like GELU. Measure the speedup.

## 📚 Resources

- [📄 NVIDIA — An Even Easier Introduction to CUDA](https://developer.nvidia.com/blog/even-easier-introduction-cuda/)
- [📄 Horace He — Making Deep Learning Go Brrrr](https://horace.io/brrr_intro.html)
- [💻 karpathy/llm.c](https://github.com/karpathy/llm.c)


---

| ⬅️ Previous | Up | Next ➡️ |
|:--|:-:|--:|
| [Ch 07](../07-optimization/) | [Syllabus](../../README.md#-syllabus) | [Ch 09](../09-speed-precision/) |
