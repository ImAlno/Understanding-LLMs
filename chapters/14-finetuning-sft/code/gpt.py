"""
A compact GPT plus the SFT chat-data helpers.
=============================================
Shared foundation for the SFT scripts: a small GPT, a tiny instruction dataset, the chat template,
and the loss-masking that trains only the response. Import from here.

The chat template wraps each (instruction, response) pair with role markers:
    <user> instruction <assistant> response <end>
and the loss mask makes the model learn *only the response* tokens — never the user's words.
"""
import math
import torch
import torch.nn as nn

# a tiny instruction-tuning dataset (instruction -> response)
PAIRS = [("hi", "hello!"), ("bye", "goodbye!"), ("thanks", "welcome!"),
         ("yes", "great!"), ("no", "okay.")]

USER, ASSISTANT, END = "\x01", "\x02", "\x03"      # single-char role / end markers


def build_vocab():
    texts = [USER + i + ASSISTANT + r + END for i, r in PAIRS]
    chars = sorted(set("".join(texts)))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    block = max(len(t) for t in texts) + 16          # + margin so generation has room for positions
    return stoi, itos, len(chars), block


def make_example(instr, resp, stoi):
    """Return (x, y) for one chat pair, with the PROMPT masked to -100 so loss trains only the response."""
    seq = USER + instr + ASSISTANT + resp + END
    ids = [stoi[c] for c in seq]
    x, y = ids[:-1], ids[1:]
    a_pos = seq.index(ASSISTANT)                    # response tokens are the ones after the <assistant> marker
    y = [(t if p >= a_pos else -100) for p, t in enumerate(y)]   # -100 = "ignore" in cross_entropy
    return x, y


class GPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd=64, n_head=4, n_layer=2):
        super().__init__()
        self.n_head, self.head_dim = n_head, n_embd // n_head
        self.tok = nn.Embedding(vocab_size, n_embd)
        self.pos = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([_Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        T = idx.size(1)
        x = self.tok(idx) + self.pos(torch.arange(T))
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))


class _Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(n_embd), nn.LayerNorm(n_embd)
        self.attn = _Attention(n_embd, n_head)
        self.ff = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.GELU(), nn.Linear(4 * n_embd, n_embd))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ff(self.ln2(x))


class _Attention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.n_head, self.head_dim = n_head, n_embd // n_head
        self.q, self.k, self.v, self.proj = (nn.Linear(n_embd, n_embd) for _ in range(4))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = (lin(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2) for lin in (self.q, self.k, self.v))
        mask = torch.triu(torch.ones(T, T) * float("-inf"), diagonal=1)
        att = torch.softmax((q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim) + mask, dim=-1)
        return self.proj((att @ v).transpose(1, 2).reshape(B, T, C))


@torch.no_grad()
def respond(model, instr, stoi, itos, max_new=12):
    """Greedily generate the model's response to an instruction (stops at <end>)."""
    ids = [stoi[c] for c in USER + instr + ASSISTANT]
    for _ in range(max_new):
        nxt = model(torch.tensor(ids).unsqueeze(0))[0, -1].argmax().item()
        if itos[nxt] == END:
            break
        ids.append(nxt)
    start = ids.index(stoi[ASSISTANT]) + 1
    return "".join(itos[t] for t in ids[start:])
