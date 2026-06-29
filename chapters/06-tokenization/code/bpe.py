"""
A byte-pair encoding (BPE) tokenizer, from scratch.
===================================================
So far the Storyteller has read *characters* (a 65-symbol vocab for Shakespeare). Real LLMs
read *subword tokens* built by **byte-pair encoding**: start from raw bytes, then repeatedly
merge the most common adjacent pair into a new token. Common chunks like " the", "ing", or
"Citizen" become single tokens, so text turns into far fewer tokens — and the model sees
word-like units instead of letters.

This file builds two tokenizers (closely following Karpathy's minbpe):
  • BasicTokenizer — BPE straight on the bytes.
  • RegexTokenizer — first split text into word-ish chunks (GPT-2's pattern) so merges never
    cross word/space boundaries, exactly like GPT-2/GPT-4.

Run:  python bpe.py
"""
import regex as re

# GPT-2's text-splitting pattern: contractions, runs of letters, runs of digits, punctuation,
# and whitespace — each a separate chunk that BPE is then applied to *within*.
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def get_stats(ids, counts=None):
    """Count how often each adjacent pair appears.  [1,2,1,2] -> {(1,2):2, (2,1):1}"""
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, idx):
    """Replace every occurrence of `pair` in `ids` with the single token `idx`."""
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids


class BasicTokenizer:
    """BPE applied directly to the UTF-8 bytes of the text."""

    def __init__(self):
        self.merges = {}                                          # (int, int) -> int
        self.vocab = {idx: bytes([idx]) for idx in range(256)}    # int -> bytes

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256
        ids = list(text.encode("utf-8"))                          # bytes 0..255
        merges, vocab = {}, {idx: bytes([idx]) for idx in range(256)}
        for i in range(num_merges):
            stats = get_stats(ids)
            pair = max(stats, key=stats.get)                      # most common adjacent pair
            idx = 256 + i
            ids = merge(ids, pair, idx)
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            if verbose:
                print(f"  merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]!r}) | {stats[pair]} occurrences")
        self.merges, self.vocab = merges, vocab

    def encode(self, text):
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            stats = get_stats(ids)
            # the pair whose merge we learned EARLIEST (lowest new-token id) goes first
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break                                             # nothing left to merge
            ids = merge(ids, pair, self.merges[pair])
        return ids

    def decode(self, ids):
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        return text_bytes.decode("utf-8", errors="replace")


class RegexTokenizer:
    """BPE within regex-split chunks, so merges respect word/space boundaries (GPT-2 style)."""

    def __init__(self, pattern=None):
        self.pattern = GPT2_SPLIT_PATTERN if pattern is None else pattern
        self.compiled = re.compile(self.pattern)
        self.merges = {}
        self.vocab = {idx: bytes([idx]) for idx in range(256)}

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256
        chunks = re.findall(self.compiled, text)                  # ["First", " Citizen", ":", ...]
        ids = [list(ch.encode("utf-8")) for ch in chunks]
        merges, vocab = {}, {idx: bytes([idx]) for idx in range(256)}
        for i in range(num_merges):
            stats = {}
            for chunk_ids in ids:                                 # count pairs WITHIN chunks only
                get_stats(chunk_ids, stats)
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
        self.merges, self.vocab = merges, vocab

    def _encode_chunk(self, text_bytes):
        ids = list(text_bytes)
        while len(ids) >= 2:
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = merge(ids, pair, self.merges[pair])
        return ids

    def encode(self, text):
        ids = []
        for chunk in re.findall(self.compiled, text):
            ids.extend(self._encode_chunk(chunk.encode("utf-8")))
        return ids

    def decode(self, ids):
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        return text_bytes.decode("utf-8", errors="replace")


def main():
    from pathlib import Path
    text = (Path(__file__).resolve().parent.parent / "data" / "input.txt").read_text()
    train_text = text[:200_000]
    held_out = text[500_000:550_000]

    for name, Tok in [("Basic", BasicTokenizer), ("Regex", RegexTokenizer)]:
        tok = Tok()
        tok.train(train_text, vocab_size=512)
        s = "First Citizen: we are accounted poor — 🧙 the Storyteller!"
        assert tok.decode(tok.encode(s)) == s, "roundtrip failed"
        nbytes = len(held_out.encode("utf-8"))
        ntoks = len(tok.encode(held_out))
        print(f"{name:5}Tokenizer | vocab 512 | roundtrip ✓ | "
              f"compression: {nbytes} bytes -> {ntoks} tokens ({nbytes/ntoks:.2f}x)")

    print("\nFirst 6 merges the Basic tokenizer learned:")
    tok = BasicTokenizer(); tok.train(train_text, vocab_size=262)
    for (a, b), idx in list(tok.merges.items()):
        print(f"  {tok.vocab[a]!r} + {tok.vocab[b]!r}  ->  {tok.vocab[idx]!r}")


if __name__ == "__main__":
    main()
