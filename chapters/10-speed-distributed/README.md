# Chapter 10 — Need for Speed III: Distributed (DDP & ZeRO)

> One GPU isn't enough. Big models train on *many* GPUs at once — dozens, thousands — and the way
> they cooperate is simpler than it sounds: give every GPU a **copy of the model** and a **different
> slice of the batch**, then **average their gradients** each step so the copies stay identical.
> That's **data parallelism** (DDP). When the model itself is too big to fit on one GPU, **ZeRO**
> shards the weights and optimizer state across the GPUs so it fits. This chapter builds both — and,
> because the whole mechanism is just message-passing, you run **real distributed training on your
> own laptop** with multiple CPU processes (no GPU required).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ImAlno/Understanding-LLMs/blob/main/chapters/10-speed-distributed/code/explore.ipynb)

> 💻 **You do not need a GPU for this chapter.** `torch.distributed` runs N processes on a plain CPU
> with the **gloo** backend, so every demo here — gradient all-reduce, DDP, ZeRO's memory math —
> runs on your laptop. On real hardware you swap the backend to **nccl** and put each rank on its
> own GPU; *the code is otherwise identical.* The one thing a laptop can't show is the actual
> **throughput win** (that needs many real GPUs) — but the *mechanism* and its *correctness* are
> fully visible locally.

**You will be able to:**
- explain **data parallelism** — replicate, shard the batch, **all-reduce** the gradients — and why
  it gives the *exact* same result as single-GPU training;
- name the core **collective operations** (`all-reduce`, `broadcast`) and what each is for;
- write a **DDP** training loop (`init_process_group`, `DistributedDataParallel`,
  `DistributedSampler`, `torchrun`) and reason about **effective batch size** and **LR scaling**;
- explain **ZeRO** stages 1/2/3 (shard optimizer state → gradients → parameters) and **FSDP**, and
  compute the per-GPU memory they save;
- place **tensor / pipeline parallelism** as the tools for when even a layer won't fit.

**Prerequisites:** the training loop and optimizer state (Chapter 7), Chapter 8's device model, and
Chapter 9's memory accounting (the "16 bytes/param" for mixed-precision Adam comes straight from
there). No new math — just processes passing tensors.

**Time:** ~2.5 hours. **Hardware:** a laptop CPU is enough to *run everything*; real multi-GPU only
to *feel the speedup*.

> ### 📓 Build it yourself in the notebook
> **[`code/explore.ipynb`](./code/explore.ipynb)**: simulate an all-reduce, **prove** that averaging
> shard-gradients equals the full-batch gradient, launch a real multi-process DDP run, and compute
> ZeRO's memory savings. "✍️ Your turn", "▶️ Run this", "▶️ Check your work" cells. Watch for:
> ✏️ **In the notebook → Step N**.

---

## 0. Setup

Four reference scripts. Two are **single-process** (instant, the conceptual core); two launch **real
multi-process** distributed jobs on your CPU:

- [`code/data_parallel_sim.py`](./code/data_parallel_sim.py) — *proves* averaging shard-gradients =
  the full-batch gradient (one process, no gloo).
- [`code/zero_memory.py`](./code/zero_memory.py) — the ZeRO per-GPU memory math.
- [`code/allreduce_demo.py`](./code/allreduce_demo.py) — real `all-reduce` / `broadcast` across N CPU
  processes (gloo).
- [`code/ddp_train.py`](./code/ddp_train.py) — a real **DDP** training run; proves all ranks end
  identical *and* match single-process training.

```bash
python chapters/10-speed-distributed/code/data_parallel_sim.py   # instant
python chapters/10-speed-distributed/code/ddp_train.py           # launches 4 processes (~5s to start)
```

---

## A 2-minute primer: why more than one GPU?

Chapters 8–9 made a *single* GPU fast. But two walls remain:

1. **Too slow.** Training a real model on all of the internet takes *months* on one GPU. You want to
   split the work across many and finish in days.
2. **Too big.** A large model's weights + gradients + optimizer state can need *hundreds* of GB —
   far more than one GPU's memory (Chapter 9's accounting: ~16 bytes × parameters for mixed-precision
   Adam, so a 10B model needs ~160 GB just for training state).

There are a few ways to split, answering different walls:

- **Data parallelism** — every GPU holds the *whole model* but processes a *different slice of the
  batch*, and they average gradients. Fixes "too slow." This is **DDP**, and it's most of what you'll
  ever use.
- **Shard the state (ZeRO / FSDP)** — still data-parallel *compute* (every GPU runs the whole model,
  one layer at a time), but stop storing `N` redundant copies of the weights/gradients/optimizer
  state — each GPU keeps only its slice. Fixes "too big" *without* splitting the computation.
- **Model parallelism** — split the *computation itself*: different layers, or even pieces of one
  matmul, on different GPUs (tensor / pipeline parallelism). The tool of last resort, for when even a
  single layer won't fit.

We spend most of the chapter on data parallelism (§1–§6) because it's the workhorse, then turn to
fitting big models with sharding and model parallelism (§7–§8).

---

## 1. Data parallelism: replicate, shard, synchronize

> ✏️ **In the notebook → Step 1.**

The idea in three moves:

1. **Replicate.** Put an identical copy of the model on each of `N` workers (called **ranks**,
   numbered `0 … N-1`; the total count `N` is the **world size**). A `broadcast` from rank 0
   guarantees they *start* identical.
2. **Shard.** Split each batch into `N` pieces; rank `r` processes only its piece. So with `N = 4`
   and a batch of 16, each rank does a forward/backward on 4 examples — a quarter of the work.
3. **Synchronize.** After `loss.backward()`, **average the gradients across all ranks** (an
   `all-reduce`). Now every rank has the *same* averaged gradient, so `optimizer.step()` moves every
   replica identically — they stay perfect copies, step after step.

In a picture, with `N = 4` and a batch of 16:

```
        batch of 16
    ┌────┬────┬────┬────┐
    │ 4  │ 4  │ 4  │ 4  │        split into 4 equal shards (DistributedSampler, §5)
    └─┬──┴─┬──┴─┬──┴─┬──┘
      ▼    ▼    ▼    ▼
    [M0] [M1] [M2] [M3]          identical model replicas — ranks 0..3
      │    │    │    │           each: forward + backward on ITS shard
      g0   g1   g2   g3          → four different gradients
      └────┴──┬─┴────┘
          all-reduce             average them: ḡ = (g0+g1+g2+g3) / 4
             ▼
      ḡ  on every rank           every replica steps with the SAME gradient → still identical
```

The payoff: `N` workers chew through `N` shards at once, so you process an **effective batch of
`N × per-rank-batch`** in the time one rank handles its slice — ideally an `N×` speedup. And because
the gradients are averaged (§3), the result is *mathematically identical* to training one model on
the whole batch. Faster, same answer.

---

## 2. The collective operations

> ✏️ **In the notebook → Step 1.**

Ranks cooperate through **collective operations** — a handful of primitives where all workers
participate. Two carry the whole chapter:

- **`all-reduce`** — every rank contributes a tensor; every rank gets back the *combined* result
  (a SUM, or an AVERAGE). This is how gradients are synchronized.
- **`broadcast`** — rank 0's tensor is copied to every other rank. This is how the replicas start
  from the same weights.
- **`barrier`** — no data moves; every rank simply *waits* here until all of them arrive. An
  "everyone catch up" used to keep processes in step (this chapter's scripts call one before shutting
  down, so no rank tears down while another still needs it).

Real output from [`allreduce_demo.py`](./code/allreduce_demo.py) (4 CPU processes, gloo):

```
all_reduce SUM of [1,2,3,4]  -> every rank now has 10.0   (= 1+2+3+4)
all_reduce AVG of [1,2,3,4]  -> every rank now has 2.5    (= the mean)
broadcast from rank 0 (99)   -> every rank now has 99.0
```

"AVG" is just "SUM, then divide by `N`" — and that average is exactly what DDP does to the
gradients. (Under the hood these are implemented cleverly so they don't bottleneck on one machine —
that's the **ring all-reduce** of §6 — but the *result* is simply the combined tensor on every rank.)

---

## 3. Why averaging gradients is correct

> ✏️ **In the notebook → Step 2.**

It's worth pausing on *why* you're allowed to average gradients from different shards and call it
training. The reason is that **differentiation is linear** and a batch loss is a **mean** over its
examples: the gradient of a mean is the mean of the per-example gradients. So the whole-batch
gradient *is* the average of the per-example gradients — and splitting the batch into equal shards,
taking each shard's gradient, and averaging those, reconstructs that exact full-batch gradient.

[`data_parallel_sim.py`](./code/data_parallel_sim.py) proves it on one CPU — no processes, just the
arithmetic:

```
full-batch gradient    (first 4 numbers): [0.0068, 0.2927, -0.3715, -0.6513]
avg of 4 shard grads   (first 4 numbers): [0.0068, 0.2927, -0.3715, -0.6513]
max difference: 5.96e-08
```

Identical to floating-point rounding. That's the guarantee underneath DDP: **it changes the speed,
never the answer.** (This is also why the shards must be *equal-sized* — unequal shards would weight
some examples more than others, and the average would no longer equal the full-batch gradient. DDP's
`DistributedSampler`, §5, is what keeps them equal.)

---

## 4. DDP in PyTorch

> ✏️ **In the notebook → Step 3.**

PyTorch's **`DistributedDataParallel`** wraps your model and does the gradient all-reduce for you —
*automatically*, inside `loss.backward()`. The skeleton:

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group("gloo")          # "gloo" on CPU (what we run); "nccl" on real GPUs. Joins the N processes.
model = DDP(MyModel().to(device))        # broadcasts rank 0's weights (start in sync); backward() now all-reduces grads

for xb, yb in loader:                     # each rank's loader yields a DIFFERENT shard (§5)
    opt.zero_grad()
    loss = loss_fn(model(xb), yb)
    loss.backward()                       # <-- gradients are averaged across ranks HERE
    opt.step()                            # every replica takes the identical step
```

You launch the `N` processes with **`torchrun`** (it sets each process's rank and the address of
rank 0):

```bash
torchrun --nproc_per_node=4 train.py     # 4 ranks on this machine (one per GPU)
```

One reason DDP is *efficient*, not just correct: it **overlaps communication with computation**.
Gradients become ready back-to-front during `backward()` (the last layer first), and DDP starts
all-reducing each layer's gradient the moment it's ready — while the earlier layers are *still*
computing theirs. So most of the network traffic happens *during* the backward pass, hidden behind
compute, rather than as a serial step after it. That overlap is a big part of why adding GPUs
actually pays off instead of drowning in communication.

[`ddp_train.py`](./code/ddp_train.py) runs exactly this on 4 CPU processes and checks the two things
that must be true — real output:

```
rank 0/4: final weight[0,0] = 0.243267     ← all four ranks print
rank 1/4: final weight[0,0] = 0.243267        the IDENTICAL weight
rank 2/4: final weight[0,0] = 0.243267        (the all-reduce kept them in sync)
rank 3/4: final weight[0,0] = 0.243267
max weight difference (DDP vs single-process) = 2.98e-08   ← and it matches full-batch training
```

Both guarantees confirmed: the replicas stayed in lockstep, and the DDP result equals a single
process trained on the whole batch. Swap `"gloo"→"nccl"` and `.to("cpu")→.to("cuda")` and this same
script runs on 4 GPUs, `~4×` faster.

---

## 5. What actually changes: batch size, the sampler, the LR

Turning on DDP has three practical consequences a beginner trips over:

- **The effective batch size multiplies by `N`.** Each rank uses its own `batch_size`, and DDP
  averages their gradients — so each step's update is the mean over `N × batch_size` examples, i.e.
  an effective batch of `N × batch_size`. Four ranks of 32 = an effective batch of 128.
- **You usually scale the learning rate up** to match (the "linear scaling rule": `N×` batch →
  roughly `N×` learning rate, with a warmup — Chapter 7). A bigger batch gives a less noisy gradient,
  so it can take a bigger step. *Concretely:* if a single GPU trained well at batch 32, LR `3e-4`,
  then 4 ranks of 32 (effective batch 128) want LR ≈ `1.2e-3` (4×), ramped up with warmup. Skip the
  scaling and the big-batch run creeps (tiny steps for a huge batch); overdo it and it diverges —
  Chapter 7's learning-rate lesson, now at scale.
- **Each rank must see *different* data.** If all ranks read the same batches, you've just computed
  the same gradient `N` times — no speedup, and a silently wrong effective batch. PyTorch's
  **`DistributedSampler`** partitions the dataset so rank `r` gets a disjoint, equal-sized shard (and
  you call `sampler.set_epoch(e)` each epoch so the shuffle differs). Forgetting the sampler is the
  #1 DDP bug.

---

## 6. Ring all-reduce: why it scales

A naïve all-reduce would send every rank's gradient to one "master" rank to sum and send back — but
then that master's network link is a bottleneck that gets *worse* with more GPUs. The trick that
makes data parallelism scale to thousands of GPUs is the **ring all-reduce**: arrange the ranks in a
ring, and pass partial sums around it in chunks, so every GPU is sending and receiving at the same
time and no single link is the bottleneck.

```
   GPU0 ──► GPU1 ──► GPU2 ──► GPU3 ──┐     each GPU sends a chunk to its neighbour and
    ▲                                │     receives one from the other side, adding as it
    └─────────────────────────────◄─┘     goes — every link busy at once, no central master
```

In numbers: each GPU sends about **2× its gradient's
size** in total (one "scatter-reduce" pass around the ring, then one "all-gather" pass) — and that's
true whether `N` is 8 or 8,000. Contrast the naïve "send everything to one master to sum": that
master receives `N×` the data, so it gets *slower* as you add GPUs. The ring's near-constant cost is
the whole reason you can add GPUs and keep getting speedup. You don't implement it (NCCL does), but
knowing it exists explains why "just average the gradients" doesn't choke at scale.

---

## 7. When the model won't fit: ZeRO

> ✏️ **In the notebook → Step 4.**

DDP replicates *everything* on every GPU — parameters, gradients, and optimizer state, all identical
across the `N` ranks. That's enormously **redundant**: you're storing `N` copies of state that's the
same everywhere. For a model whose training state exceeds one GPU, that redundancy is fatal.

**ZeRO** (Zero Redundancy Optimizer) fixes it by **sharding** that state across the ranks — each GPU
holds only `1/N` of it, and workers exchange the pieces they need on the fly. It comes in three
cumulative stages:

- **ZeRO-1** — shard the **optimizer state** (Adam's `m` and `v` + the fp32 master weights). This is
  the biggest piece, so it's the biggest win.
- **ZeRO-2** — also shard the **gradients**.
- **ZeRO-3** — also shard the **parameters** themselves. (This is PyTorch's **FSDP**, Fully Sharded
  Data Parallel.)

Using Chapter 9's accounting (16 bytes/param for mixed-precision Adam: 2 fp16 params + 2 fp16 grads +
4 fp32 master + 4 `m` + 4 `v`), [`zero_memory.py`](./code/zero_memory.py) shows the per-GPU memory
for a **7.5B** model on **8** GPUs:

```
       plain DDP (no sharding):  120.0 GB   ← impossible on a 40 GB or 80 GB card
      ZeRO-1 (optimizer state):   41.2 GB
          ZeRO-2 (+ gradients):   28.1 GB
  ZeRO-3 / FSDP (+ parameters):   15.0 GB   ← now it fits, comfortably
```

Plain DDP needs 120 GB/GPU — it simply won't run. ZeRO-3 brings the *same* model to 15 GB/GPU by
never storing a redundant copy. The cost is more **communication** (ranks fetch each other's shards
when needed), which is why you reach for the lowest stage that fits: ZeRO-1 if it's enough, ZeRO-3
when you must. This is how models with tens or hundreds of billions of parameters are trained at all.

A useful instinct for *which* stage: the optimizer state is **12 of the 16** bytes per parameter (the
fp32 master + `m` + `v`), so **ZeRO-1 alone removes most of the redundancy**. ZeRO-1 and ZeRO-2 keep
essentially DDP's communication volume (a reduce-scatter + all-gather, ≈ 2× the gradient size), so
they're nearly free; **ZeRO-3** adds an extra round each layer to gather the sharded parameters, so
it's the one that clearly costs more bandwidth. Reach for the lowest stage that fits. In PyTorch,
ZeRO-1 is `ZeroRedundancyOptimizer` and ZeRO-3 is **FSDP** (`FullyShardedDataParallel`) — you wrap the
model much like `DDP(model)`, and FSDP is the default path for training large models today.

---

## 8. When even a layer won't fit: tensor & pipeline parallelism

ZeRO shards *data-parallel* state, but every GPU still runs the whole model one layer at a time. If a
*single layer* is too big, or you want to cut latency further, you split the **model's computation**:

- **Pipeline parallelism** — put different *layers* on different GPUs (GPU 0 runs layers 1–12, GPU 1
  runs 13–24, …) and stream micro-batches through like an assembly line.
- **Tensor parallelism** — split a *single* matmul across GPUs (each holds a slice of the weight
  matrix and they combine partial results). This is Megatron-LM's approach for the giant attention
  and MLP matrices.

Real frontier training combines all of them — data + pipeline + tensor parallelism, "**3D
parallelism**" — plus ZeRO. Models like GPT-3 were trained exactly this way: **tensor** parallelism
*within* a **node** (one machine of ~8 GPUs on a fast interconnect — the giant attention/MLP matmuls
are split across them), **pipeline** parallelism *across* nodes (different layers on different), and
**data** parallelism *on top* (many such model-copies, each on a batch shard). No single axis is
enough past a few billion parameters. You won't wire these by hand in this course, but the mental
model is worth having: *data parallelism splits the batch; model parallelism splits the model; the
biggest runs split both.*

---

## 9. Why bother: a word on scaling

The reason all this engineering exists is the **scaling laws** finding: across many orders of
magnitude, a Transformer's loss falls *predictably* as you grow three things together — parameters,
data, and compute. Bigger-trained-longer reliably means better, with no plateau yet in sight. That
turned "train a good model" into "train the *biggest* model you can afford," which turned distributed
training from a nicety into the whole game. Everything in this chapter is the machinery that makes
"the biggest model you can afford" a larger number.

---

## 🔌 How this plugs into the Storyteller

Your Chapter 5 GPT trains on one device today. Wrapping it in DDP — `init_process_group`, `DDP(model)`,
a `DistributedSampler`, launched with `torchrun` — lets it train on `N` GPUs at `~N×` the speed, with
a loss curve identical to the single-GPU run (you'll prove exactly this in the mini-project). Combined
with Chapters 8–9 (device + precision), this is the full "train it fast" toolkit: the same model, run
on real hardware at real scale. From here (Chapter 11 on) we turn from *speed* to *data* — what to
feed a model this capable.

---

## 🐛 Building it yourself: what trips people up

Distributed bugs are their own genre. The classics:

- **Forgetting the `DistributedSampler`.** Without it, every rank trains on the *same* data — you get
  no speedup and a wrong effective batch, but nothing errors. The single most common DDP mistake.
- **Mismatched collectives → hang.** Every rank must call the *same* collectives in the *same order*.
  If one rank takes an `if` branch that skips an `all-reduce` (or does an extra one), the others wait
  for it forever — a silent **deadlock**, not a crash. (This chapter's own `ddp_train.py` uses a
  `barrier()` before teardown for exactly this reason.)
- **Not scaling the learning rate.** `N×` the effective batch with the *same* LR often trains
  noticeably worse — scale the LR up (with warmup).
- **Printing/saving from every rank.** All `N` ranks run the whole script, so a naïve `print` or
  `torch.save` fires `N` times. Guard rank-0-only work with `if rank == 0:`.
- **Expecting a speedup from `N` processes on one CPU.** Locally the ranks share the same cores, so
  it's *not* faster — it's for *learning the mechanism*. The speedup needs `N` real GPUs.

---

## 🤔 Common questions

- **Does DDP change the result vs one GPU?** No — averaging equal-shard gradients is exactly the
  full-batch gradient (§3). The mini-project reproduces the single-GPU loss curve on `N` ranks.
- **Data parallelism vs model parallelism — which do I need?** Data parallelism (DDP) if the model
  *fits* on one GPU and you just want it faster — that's the common case. Model parallelism (ZeRO-3 /
  FSDP, tensor/pipeline) only when the model is *too big to fit*.
- **What's a "rank" and a "world size"?** A **rank** is one worker process (usually one GPU),
  numbered `0 … N-1`; the **world size** is `N`, the total number of workers.
- **Why `gloo` here instead of `nccl`?** `nccl` is NVIDIA's fast GPU-to-GPU backend; `gloo` runs on
  CPU, so we can run *real* distributed code on a laptop. Swap one string on real GPUs.
- **Is ZeRO the same as DDP?** ZeRO is a *memory optimization on top of* data parallelism: same
  replicate-and-average idea, but it stops storing `N` redundant copies of the training state.
- **How do thousands of GPUs not choke on communication?** The **ring all-reduce** (§6) keeps every
  link busy and its cost nearly independent of `N`, and ZeRO/overlap hide comms behind compute.
- **How does this relate to gradient accumulation (Chapter 7)?** Same idea, *across time on one GPU*:
  run several small batches, sum their gradients, then step — a big effective batch without the
  memory of one. DDP does it *across GPUs in parallel*. They compose: `N` GPUs × `k` accumulation
  steps × `batch` = effective batch `N·k·batch`, which is how the biggest runs reach batch sizes in
  the millions.

## ✅ Check your understanding

<details>
<summary>1. In one sentence each: replicate, shard, synchronize — what does data parallelism do?</summary>

**Replicate** the model onto every rank (broadcast so they start identical); **shard** each batch so
every rank processes a different slice; **synchronize** by all-reducing (averaging) the gradients
after backward so every replica takes the identical step.
</details>

<details>
<summary>2. Why is averaging the shard-gradients mathematically identical to full-batch training?</summary>

The gradient of a mean loss is linear in the examples, so the full-batch gradient *is* the average
of the per-example gradients. Averaging equal-shard gradients reconstructs that exact average — DDP
changes speed, not the answer. (It requires *equal-sized* shards, which `DistributedSampler` ensures.)
</details>

<details>
<summary>3. What do <code>all-reduce</code> and <code>broadcast</code> each do, and where does DDP use them?</summary>

`all-reduce`: every rank contributes a tensor and every rank gets the combined (summed/averaged)
result — DDP uses it to average gradients each step. `broadcast`: copy rank 0's tensor to all ranks
— DDP uses it once to start every replica from the same weights.
</details>

<details>
<summary>4. What does each ZeRO stage shard, and what does it cost?</summary>

ZeRO-1 shards the optimizer state, ZeRO-2 also the gradients, ZeRO-3 (FSDP) also the parameters —
each cumulatively cutting per-GPU memory toward `1/N`, at the cost of more communication (ranks fetch
each other's shards). Use the lowest stage that fits.
</details>

<details>
<summary>5. You turn on DDP with 4 ranks and forget the <code>DistributedSampler</code>. What happens?</summary>

Every rank trains on the *same* data, so you compute the identical gradient four times and average it
back to itself — no speedup, and the effective batch is wrong. Nothing errors; the model just trains
as if on one rank (or worse). The sampler is what gives each rank a disjoint shard.
</details>

## 🎓 Key takeaways

- **Data parallelism (DDP)**: replicate the model on `N` ranks, give each a batch shard, and
  **all-reduce (average) the gradients** each step — an `N×` speedup with a *mathematically
  identical* result to one GPU.
- The mechanism is **collective operations** — `broadcast` to start in sync, `all-reduce` to average
  gradients — and it's exact because a mean-loss gradient is linear.
- Practical consequences: effective batch **×N**, scale the **LR** up, and give each rank different
  data with a **`DistributedSampler`**.
- **ZeRO** shards the redundant training state (optimizer → gradients → parameters; ZeRO-3 = **FSDP**)
  so a model too big for one GPU fits — trading communication for memory.
- **Tensor/pipeline parallelism** split the *model* when even a layer won't fit; the biggest runs
  combine all of it (**3D parallelism**). It all exists because **scaling laws** make bigger reliably
  better.

## 📖 New vocabulary

`data parallelism` · `model parallelism` · `DDP` · `rank` / `world size` · `collective operation` ·
`all-reduce` · `broadcast` · `ring all-reduce` · `DistributedSampler` · `effective batch size` ·
`linear scaling rule` · `gloo` / `nccl` · `torchrun` · `ZeRO` (stages 1/2/3) · `FSDP` ·
`tensor parallelism` · `pipeline parallelism` · `3D parallelism` · `scaling laws`.

## 🧪 Practice & build

1. **The notebook** ([Colab](https://colab.research.google.com/github/ImAlno/Understanding-LLMs/blob/main/chapters/10-speed-distributed/code/explore.ipynb))
   — simulate an all-reduce, prove the shard-gradient equivalence, launch a real DDP run, compute
   ZeRO's savings. Do this first.
2. **Exercises** — [`exercises/`](./exercises/): implement all-reduce-average, prove shard-grads sum
   to the full gradient, catch a `DistributedSampler` overlap bug, and compute a ZeRO memory table.
   Tiered hints + solutions; a starter for the first.
3. **Mini-project** — [`project/`](./project/): **"DDP Your GPT"** — wrap a GPT training loop in DDP,
   run it across processes, and prove the multi-rank loss curve matches the single-process one.

## 🔗 Go deeper (optional)

- 📄 [PyTorch — Getting Started with DDP](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html)
  — the official DDP + `torchrun` walkthrough.
- 📄 [Rajbhandari et al. (2019), *ZeRO*](https://arxiv.org/abs/1910.02054) — the memory-sharding paper
  behind DeepSpeed and FSDP.
- 📄 [Shoeybi et al. (2019), *Megatron-LM*](https://arxiv.org/abs/1909.08053) — tensor parallelism for
  the giant matmuls.

---

| ⬅️ Previous | Up | Next ➡️ |
|:--|:-:|--:|
| [Chapter 9 — Precision](../09-speed-precision/) | [Syllabus](../../README.md#-syllabus) | [Chapter 11 — Datasets](../11-datasets/) |
