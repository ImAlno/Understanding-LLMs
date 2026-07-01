"""
A compact GPT plus the preference-data helpers for RLHF & DPO.
==============================================================
Shared foundation for the alignment scripts. Chapter 14 taught SFT (imitate demonstrations); here we
go past imitation to *preferences* — teaching the model that one reply is **better** than another.

The data is no longer (instruction -> response). It is a **preference triple**:
    (prompt, chosen, rejected)
where a human preferred `chosen` over `rejected`. Every response is wrapped in Chapter 14's chat
template  <user> prompt <assistant> response <end>  and, as before, we only ever score the response.

To keep the signal legible, the latent preference is **warmth**: the `chosen` replies are warm, the
`rejected` ones are curt — and they are the *same length*, so the preference is about tone, not length.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# preference triples: (prompt, chosen = warm, rejected = curt). Length-balanced on purpose.
PREFS = [
    ("hi",      "hello there",      "what do you want"),
    ("bye",     "take good care",   "finally you leave"),
    ("thanks",  "my pleasure",      "yeah sure fine"),
    ("hey",     "great to see you", "oh no its you"),
    ("help",    "glad to help",     "figure it out"),
]

USER, ASSISTANT, END = "\x01", "\x02", "\x03"      # single-char role / end markers (as in Chapter 14)


def build_vocab():
    """Vocabulary over every character that appears in any templated chosen OR rejected sequence."""
    texts = [USER + p + ASSISTANT + c + END for p, c, r in PREFS] + \
            [USER + p + ASSISTANT + r + END for p, c, r in PREFS]
    chars = sorted(set("".join(texts)))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    block = max(len(t) for t in texts) + 16          # + margin so generation has room for positions
    return stoi, itos, len(chars), block


def make_example(prompt, response, stoi):
    """(x, y) for one templated turn, with the PROMPT masked to -100 so loss trains only the response."""
    seq = USER + prompt + ASSISTANT + response + END
    ids = [stoi[c] for c in seq]
    x, y = ids[:-1], ids[1:]
    a_pos = seq.index(ASSISTANT)
    y = [(t if p >= a_pos else -100) for p, t in enumerate(y)]
    return x, y


def seq_logprob(model, prompt, response, stoi):
    """Sum of log p(response tokens incl <end> | prompt) under `model`. Differentiable.

    This is the one quantity DPO and the RL policy gradient are built from: "how much probability does
    this model place on producing THIS response to THIS prompt?"
    """
    seq = USER + prompt + ASSISTANT + response + END
    ids = [stoi[c] for c in seq]
    x = torch.tensor(ids[:-1]).unsqueeze(0)
    y = torch.tensor(ids[1:])
    logp = torch.log_softmax(model(x)[0], dim=-1)         # (T, vocab) log-probabilities
    a_pos = seq.index(ASSISTANT)
    tok_logp = logp[torch.arange(len(y)), y]              # log-prob of each actual next token
    return tok_logp[a_pos:].sum()                         # positions p >= a_pos are the response + <end>


# ----------------------------------------------------------------------------------------------------
# The transformer backbone (identical to Chapter 14's GPT) — shared by the policy and the reward model.
# ----------------------------------------------------------------------------------------------------
class _Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(n_embd), nn.LayerNorm(n_embd)
        self.n_head, self.head_dim = n_head, n_embd // n_head
        self.q, self.k, self.v, self.proj = (nn.Linear(n_embd, n_embd) for _ in range(4))
        self.ff = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.GELU(), nn.Linear(4 * n_embd, n_embd))

    def _attn(self, x):
        B, T, C = x.shape
        q, k, v = (l(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2) for l in (self.q, self.k, self.v))
        mask = torch.triu(torch.ones(T, T) * float("-inf"), diagonal=1)
        att = torch.softmax((q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim) + mask, dim=-1)
        return self.proj((att @ v).transpose(1, 2).reshape(B, T, C))

    def forward(self, x):
        x = x + self._attn(self.ln1(x))
        return x + self.ff(self.ln2(x))


class _Backbone(nn.Module):
    """Token + position embeddings and the transformer blocks — everything up to the final hidden state."""
    def __init__(self, vocab_size, block_size, n_embd, n_head, n_layer):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, n_embd)
        self.pos = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([_Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)

    def forward(self, idx):
        x = self.tok(idx) + self.pos(torch.arange(idx.size(1)))
        for block in self.blocks:
            x = block(x)
        return self.ln_f(x)                              # (B, T, n_embd) final hidden states


class GPT(nn.Module):
    """The policy: a language model. Final hidden state -> a distribution over the next token."""
    def __init__(self, vocab_size, block_size, n_embd=64, n_head=4, n_layer=2):
        super().__init__()
        self.body = _Backbone(vocab_size, block_size, n_embd, n_head, n_layer)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        return self.head(self.body(idx))                 # (B, T, vocab) logits


class RewardModel(nn.Module):
    """A reward model: same backbone, but a **scalar** head. It reads the LAST token's hidden state
    (which has attended to the whole prompt+response) and outputs one number = "how good is this reply?"."""
    def __init__(self, vocab_size, block_size, n_embd=64, n_head=4, n_layer=2):
        super().__init__()
        self.body = _Backbone(vocab_size, block_size, n_embd, n_head, n_layer)
        self.value = nn.Linear(n_embd, 1)

    def forward(self, idx):
        return self.value(self.body(idx))[:, -1, 0]      # (B,) one scalar reward per sequence


def reward(rm, prompt, response, stoi):
    """The reward model's score for one (prompt, response), always evaluated on the full template."""
    ids = [stoi[c] for c in USER + prompt + ASSISTANT + response + END]
    return rm(torch.tensor(ids).unsqueeze(0))[0]


@torch.no_grad()
def generate(model, prompt, stoi, itos, greedy=True, max_new=18):
    """Generate `model`'s reply to `prompt` (greedy by default; sampled if greedy=False). Stops at <end>."""
    ids = [stoi[c] for c in USER + prompt + ASSISTANT]
    for _ in range(max_new):
        logits = model(torch.tensor(ids).unsqueeze(0))[0, -1]
        nxt = torch.argmax(logits).item() if greedy else torch.multinomial(torch.softmax(logits, -1), 1).item()
        if itos[nxt] == END:
            break
        ids.append(nxt)
    return "".join(itos[t] for t in ids[ids.index(stoi[ASSISTANT]) + 1:])


def train_reference(stoi, V, block, steps=400, seed=0):
    """SFT a reference/policy model (this is just Chapter 14's SFT). We train on the chosen replies AND
    the curt ones, but weight the curt ones 2x, so the *starting* model defaults to terse answers —
    that is our honest 'before alignment' baseline. Alignment (DPO / RLHF) will warm it up."""
    torch.manual_seed(seed)
    model = GPT(V, block)
    examples = [make_example(p, c, stoi) for p, c, r in PREFS] + \
               2 * [make_example(p, r, stoi) for p, c, r in PREFS]
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(steps):
        loss = sum(F.cross_entropy(model(torch.tensor(x).unsqueeze(0)).view(-1, V),
                                   torch.tensor(y), ignore_index=-100) for x, y in examples) / len(examples)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model
