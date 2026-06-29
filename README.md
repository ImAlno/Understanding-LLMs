# 🪄 Understanding LLMs — *Let's Build a Storyteller*

> Build a **Storyteller AI** — a Large Language Model that writes (and eventually
> illustrates) little stories — completely from scratch, end to end, until you
> have a working ChatGPT-style web app. Along the way you'll gain a genuinely
> deep understanding of AI, LLMs, and deep learning.

This is a hands-on, build-it-yourself course. We start from a model so simple you
could do it on paper, and finish with a small but real chat application. Every
line is something you write and understand — no magic, no hand-waving.

---

## ✨ What makes this course different

- **From scratch, for real.** We build the autograd engine, the tokenizer, the
  Transformer, the training loop, and the web app ourselves — in Python first,
  then dropping into C and CUDA when we chase speed.
- **Learn by building.** Every chapter has a written lesson, runnable code, an
  interactive notebook, exercises, and a **mini-project you build yourself**
  (with a reference solution to check against).
- **One story, told all the way through.** The "Storyteller" theme connects the
  chapters: a model that first invents *character names*, then writes *sentences*,
  then *paragraphs*, then *finished illustrated stories*.

## 🎯 Who this is for

You're comfortable with **basic Python** (variables, functions, loops, lists) and
curious about how ChatGPT-like systems actually work. You do **not** need prior
machine-learning experience, and you don't need to remember your calculus — we
build the math intuition as we go. This is roughly the level of Karpathy's
*Neural Networks: Zero to Hero* series, which pairs beautifully with these notes.

## 🗺️ How each chapter is organized

Every chapter folder is **self-contained** and follows the same shape:

```
chapters/NN-topic/
├── README.md          ← the lesson: read this first
├── code/              ← runnable, heavily-commented reference scripts
│   └── explore.ipynb  ← an interactive notebook to poke at the ideas
├── exercises/         ← short exercises to cement the ideas
│   └── solutions/     ← worked solutions (peek only after you try!)
└── project/           ← the mini-project YOU build
    ├── starter/       ← scaffold with TODOs to fill in
    └── solution/      ← a reference implementation to check against
```

**Suggested loop per chapter:** read the lesson → run/poke the notebook → do the
exercises → build the project (starter → your code → compare to solution) →
optionally watch the matching Karpathy video.

---

## 📚 Syllabus

> Status legend: ✅ ready · 🚧 in progress · 🗓️ planned

| # | Chapter | You'll learn | Status |
|---|---------|--------------|--------|
| 01 | [Bigram Language Model](./chapters/01-bigram/) | language modeling, sampling, the loss function, the count↔neural-net equivalence | ✅ |
| 02 | [Micrograd](./chapters/02-micrograd/) | machine learning, backpropagation, building autograd from scratch | ✅ |
| 03 | [N-gram Model](./chapters/03-ngram-mlp/) | multi-layer perceptron, embeddings, matmul, GELU | ✅ |
| 04 | [Attention](./chapters/04-attention/) | attention, softmax, positional encoding | ✅ |
| 05 | [Transformer](./chapters/05-transformer/) | the Transformer, residuals, LayerNorm, GPT-2 | ✅ |
| 06 | [Tokenization](./chapters/06-tokenization/) | minBPE, byte-pair encoding | ✅ |
| 07 | [Optimization](./chapters/07-optimization/) | initialization, optimization, AdamW | ✅ |
| 08 | [Need for Speed I: Device](./chapters/08-speed-device/) | CPU, GPU, CUDA, devices | 🗓️ |
| 09 | [Need for Speed II: Precision](./chapters/09-speed-precision/) | mixed precision, fp16, bf16, fp8 | 🗓️ |
| 10 | [Need for Speed III: Distributed](./chapters/10-speed-distributed/) | distributed training, DDP, ZeRO | 🗓️ |
| 11 | [Datasets](./chapters/11-datasets/) | data loading, synthetic data, TinyStories | 🗓️ |
| 12 | [Inference I: KV-cache](./chapters/12-inference-kv-cache/) | the KV cache | 🗓️ |
| 13 | [Inference II: Quantization](./chapters/13-inference-quantization/) | quantization (int8, GPTQ, AWQ) | 🗓️ |
| 14 | [Finetuning I: SFT](./chapters/14-finetuning-sft/) | supervised finetuning, PEFT, LoRA, chat | 🗓️ |
| 15 | [Finetuning II: RL](./chapters/15-finetuning-rl/) | RLHF, PPO, DPO | 🗓️ |
| 16 | [Deployment](./chapters/16-deployment/) | API, web app | 🗓️ |
| 17 | [Multimodal](./chapters/17-multimodal/) | VQVAE, diffusion transformer, illustrating stories | 🗓️ |
| — | [Appendix](./appendix/) | tensors, data types, C & assembly primers, math refreshers | 🗓️ |

The **end goal** mirrors Karpathy's [nanochat](https://github.com/karpathy/nanochat)
("the best ChatGPT that $100 can buy"): tokenizer → pretraining → finetuning →
RL → KV-cache inference → web UI. Our capstone trains on
[TinyStories](https://arxiv.org/abs/2305.07759), where even sub-10M-parameter
models learn to write coherent stories — so you can train yours on modest hardware.

---

## 🚀 Setup (5 minutes)

You need **Python 3.10+** (3.13 recommended) and `git`.

```bash
# 1. Clone the repo
git clone https://github.com/ImAlno/Understanding-LLMs.git
cd Understanding-LLMs

# 2. Create a virtual environment + install dependencies
#    Option A — uv (fast, recommended):  https://docs.astral.sh/uv/
uv venv --python 3.13 .venv
uv pip install -r requirements.txt

#    Option B — plain pip:
# python3 -m venv .venv
# source .venv/bin/activate        # Windows: .venv\Scripts\activate
# pip install -r requirements.txt

# 3. Sanity check — train a bigram model on 32k names
.venv/bin/python chapters/01-bigram/code/bigram_counts.py
```

If that prints some sampled names and a loss around **2.45**, you're ready. 🎉

> **No GPU?** No problem. Chapters 1–7 run fine on a laptop CPU. GPUs only become
> important in the "Need for Speed" chapters, where we also show free options
> (Google Colab, etc.).

---

## 🧭 Learning resources

- **[resources/videos.md](./resources/videos.md)** — Karpathy's *Zero to Hero*
  videos and code repos, mapped to our chapters.
- **[resources/papers.md](./resources/papers.md)** — the key paper + one gentle
  explainer for each chapter.

## 🙏 Acknowledgements

This course stands on the shoulders of **Andrej Karpathy**'s open educational
work — *Neural Networks: Zero to Hero*, `makemore`, `micrograd`, `nanoGPT`,
`minbpe`, and `nanochat` — and his archived
[LLM101n "Let's build a Storyteller"](https://github.com/karpathy/LLM101n)
syllabus, which this course is a from-scratch, fully-built-out interpretation of.
It also draws on the foundational papers linked throughout. Huge thanks to the
researchers and educators who made all of this freely available.

## 📜 License

Course text and code are provided for learning. See individual linked resources
for their own licenses (e.g., datasets and papers retain theirs).

---

*Ready? Start with **[Chapter 1: The Bigram Language Model →](./chapters/01-bigram/)***
