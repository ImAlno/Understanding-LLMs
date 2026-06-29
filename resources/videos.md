# 🎥 Video & Code Resources

The single best companion to this course is Andrej Karpathy's free
**[Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)**
video series, plus his open-source repositories. Our chapters are built in the
same spirit — and where a chapter mirrors one of his lectures, we link it.

> All URLs below were verified to resolve in 2026. Runtimes are approximate.

## Neural Networks: Zero to Hero — the spine of the course

| Video | Length | Maps to |
|-------|--------|---------|
| [The spelled-out intro to neural networks and backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0) | ~2h25m | **Ch 2** — Micrograd |
| [The spelled-out intro to language modeling: building makemore](https://www.youtube.com/watch?v=PaCmpygFfXo) | ~1h57m | **Ch 1** — Bigram |
| [Building makemore Part 2: MLP](https://www.youtube.com/watch?v=TCH_1BHY58I) | ~1h15m | **Ch 3** — N-gram MLP |
| [Building makemore Part 3: Activations & Gradients, BatchNorm](https://www.youtube.com/watch?v=P6sfmUTpUmc) | ~1h55m | **Ch 7** — Optimization |
| [Building makemore Part 4: Becoming a Backprop Ninja](https://www.youtube.com/watch?v=q8SA3rM6ckI) | ~1h56m | **Ch 2 / 7** — backprop depth |
| [Building makemore Part 5: Building a WaveNet](https://www.youtube.com/watch?v=t3YJ5hKiMQ0) | ~56m | **Ch 3 / 5** — deeper nets |
| [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY) | ~1h56m | **Ch 4 / 5** — Attention & Transformer |
| [Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE) | ~2h13m | **Ch 6** — Tokenization |

## Standalone follow-ups (same teaching arc)

| Video | Length | Maps to |
|-------|--------|---------|
| [Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU) | ~4h01m | **Ch 5–10** — real pretraining + speed |
| [Deep Dive into LLMs like ChatGPT](https://www.youtube.com/watch?v=7xTGNNLPyMI) | ~3h31m | **Ch 11–15** — full training stack |
| [How I use LLMs](https://www.youtube.com/watch?v=EWvNQjAaOHw) | ~2h11m | **Ch 16–17** — usage & ecosystem |
| [Software Is Changing (Again) — "Software 3.0"](https://www.youtube.com/watch?v=LCEmiRjPEtQ) | ~39m | Framing / motivation |

## Karpathy's repositories we draw on

| Repo | What it is | Used around |
|------|------------|-------------|
| [micrograd](https://github.com/karpathy/micrograd) | ~100-line scalar autograd engine | Ch 2 |
| [makemore](https://github.com/karpathy/makemore) | Character-level LM: bigram → MLP → Transformer | Ch 1, 3 |
| [nn-zero-to-hero](https://github.com/karpathy/nn-zero-to-hero) | Notebooks for the whole video series | Ch 1–7 |
| [ng-video-lecture](https://github.com/karpathy/ng-video-lecture) | Minimal GPT built live in "Let's build GPT" | Ch 5 |
| [nanoGPT](https://github.com/karpathy/nanoGPT) | The clean, fast GPT training repo | Ch 5, 7–11 |
| [build-nanogpt](https://github.com/karpathy/build-nanogpt) | Step-by-step GPT-2 reproduction (git history = the lecture) | Ch 5, 9 |
| [minbpe](https://github.com/karpathy/minbpe) | Minimal byte-pair-encoding tokenizer | Ch 6 |
| [llama2.c](https://github.com/karpathy/llama2.c) | Llama-2 inference in one file of pure C | Ch 8, 12 |
| [llm.c](https://github.com/karpathy/llm.c) | GPT-2 training in raw C/CUDA | Ch 8–10 |
| [nanochat](https://github.com/karpathy/nanochat) | **The capstone reference** — full ChatGPT clone (tokenizer → pretrain → SFT → RL → KV-cache inference → web UI) | Ch 11–16 |

## Two things worth knowing

- **`karpathy/LLM101n` is archived and contains only a syllabus** — no course
  content. The full course is being developed separately by
  [Eureka Labs](https://eurekalabs.ai). This repo is our from-scratch take on
  that syllabus, so we lean on the *Zero to Hero* videos and the code repos
  above rather than LLM101n itself.
- **nanochat has no official video.** Karpathy's walkthrough is written: see the
  repo's **[Discussions](https://github.com/karpathy/nanochat/discussions)**
  ("Introducing nanochat", Oct 2025).

➡️ For papers and written tutorials per chapter, see **[papers.md](./papers.md)**.
