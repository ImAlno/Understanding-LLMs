# Chapter 15 — Exercises

Four exercises on preference alignment. Do them **after the notebook**, and **try before peeking** at
[`solutions/`](./solutions/). A starter scaffold for the first is in [`starter/`](./starter/).

| # | Exercise | You implement / observe |
|---|----------|-------------------------|
| **E01** | Bradley-Terry reward model | the ranking loss `−log σ(chosen − rejected)` (starter in [`starter/`](./starter/)) |
| **E02** | The DPO loss | the implicit-reward Bradley-Terry loss; align curt → warm |
| **E03** | The β leash | too-small β over-optimizes: margin climbs while generations turn to mush |
| **E04** | The frozen reference | DPO moves the policy but leaves the reference byte-for-byte unchanged |

```bash
# start here — fill in the one TODO, then run:
python chapters/15-finetuning-rl/exercises/starter/e01_bradley_terry.py
```

Each script is self-contained and CPU-only, importing the model, reward model, and helpers from
[`../code/`](../code/). Stuck? The worked solutions (with the numbers to expect) are in
[`solutions/`](./solutions/).
