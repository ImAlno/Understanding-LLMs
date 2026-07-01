"""
Supervised fine-tuning (SFT): teach a base model to follow instructions.
========================================================================
A pretrained model *continues text*. SFT teaches it to *respond*, using the same next-token loss but
different data: (instruction, response) pairs in a chat template, with the loss **masked** so it
trains only on the response — never on the user's words.

This shows the loss mask, generates BEFORE training (the base model babbles), then does the SFT and
generates AFTER (it answers). Full fine-tuning here; LoRA is the mini-project.

Run:  python sft.py
"""
import torch
import torch.nn.functional as F
from gpt import GPT, PAIRS, USER, ASSISTANT, END, build_vocab, make_example, respond


def main():
    torch.manual_seed(0)
    stoi, itos, vocab_size, block = build_vocab()
    examples = [make_example(i, r, stoi) for i, r in PAIRS]

    # show the loss mask on one example: only the response tokens (and <end>) are trained
    x, y = examples[0]
    seq = USER + PAIRS[0][0] + ASSISTANT + PAIRS[0][1] + END
    pretty = seq.replace(USER, "<user>").replace(ASSISTANT, "<assistant>").replace(END, "<end>")
    shown = "".join("_" if t == -100 else ("<end>" if itos[t] == END else itos[t]) for t in y)   # "_" = masked
    print(f"chat template : {pretty!r}")
    print(f"trained target: {shown!r}   ('_' = masked prompt, trained only on the response)\n")

    model = GPT(vocab_size, block)

    # BEFORE: the untrained model doesn't answer
    print("before SFT:")
    for i, r in PAIRS[:3]:
        print(f"  '{i}' -> {respond(model, i, stoi, itos)!r}")

    # SFT: minimize cross-entropy on the (masked) chat examples
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(400):
        loss = sum(F.cross_entropy(model(torch.tensor(x).unsqueeze(0)).view(-1, vocab_size),
                                   torch.tensor(y), ignore_index=-100) for x, y in examples) / len(examples)
        opt.zero_grad(); loss.backward(); opt.step()

    # AFTER: it follows the instructions
    model.eval()
    print(f"\nafter SFT (loss {loss.item():.4f}):")
    correct = 0
    for i, r in PAIRS:
        got = respond(model, i, stoi, itos)
        correct += (got == r)
        print(f"  '{i}' -> {got!r}   {'✓' if got == r else '✗ (want ' + repr(r) + ')'}")
    print(f"\n{correct}/{len(PAIRS)} correct — the base model became an instruction-follower.")


if __name__ == "__main__":
    main()
