"""
Floating-point formats, up close.
=================================
A number like 3.14159 is stored in a fixed budget of bits, split into a SIGN, an EXPONENT (how
big — the range), and a MANTISSA (the significant digits — the precision). Spend fewer bits and
the number gets cheaper to store and move, but coarser. The whole of mixed-precision training is
choosing *which* coarseness you can afford where.

This file shows, on the CPU (no GPU needed), exactly what each format can and can't represent:
  • fp32  — 1 sign / 8 exponent / 23 mantissa : the default. Full range, full precision.
  • fp16  — 1 sign / 5 exponent / 10 mantissa : half the bits. Good precision, but TINY range.
  • bf16  — 1 sign / 8 exponent /  7 mantissa : half the bits. fp32's range, COARSE precision.

Run:  python formats.py
"""
import torch

FORMATS = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def budget_table():
    """Bytes, range, and smallest-positive for each format — the bit budget, made concrete."""
    print("format   bytes        max value     smallest > 0     mantissa bits (~precision)")
    print("-" * 78)
    mant = {"fp32": 23, "fp16": 10, "bf16": 7}
    for name, dt in FORMATS.items():
        info = torch.finfo(dt)
        print(f"{name:>5}   {info.bits // 8:>3}      {info.max:>12.3g}     {info.tiny:>12.3g}"
              f"        {mant[name]:>3}")
    print("\nfp16 and bf16 are BOTH 2 bytes — they split the same budget differently:")
    print("  fp16 spends more bits on the mantissa (finer steps) but few on the exponent (tiny range).")
    print("  bf16 keeps fp32's exponent (full range) and pays for it with a coarse mantissa.")


def round_trip(value):
    """Store one number in each format and print what it actually becomes."""
    print(f"\nstoring {value!r} in each format (what it rounds to):")
    for name, dt in FORMATS.items():
        stored = torch.tensor(value, dtype=dt).item()
        print(f"  {name}: {stored!r}")


def range_failure():
    """fp16 overflows to inf around 65504; bf16 (fp32's range) shrugs it off."""
    print("\nRANGE: store 70000 in each format")
    for name, dt in FORMATS.items():
        stored = torch.tensor(70000.0, dtype=dt).item()
        flag = "  <- overflowed to infinity!" if stored == float("inf") else ""
        print(f"  {name}: {stored!r}{flag}")


def underflow_failure():
    """A small gradient (1e-8) vanishes to 0 in fp16; bf16 keeps it. This is why fp16 needs
    loss scaling and bf16 does not."""
    print("\nUNDERFLOW: store a tiny gradient 1e-8 in each format")
    for name, dt in FORMATS.items():
        stored = torch.tensor(1e-8, dtype=dt).item()
        flag = "  <- underflowed to ZERO (gradient lost!)" if stored == 0.0 else ""
        print(f"  {name}: {stored!r}{flag}")


def swamping_failure():
    """PRECISION: add a small number to a large one. With bf16's 7 mantissa bits, the small one
    falls off the bottom and is lost — 'swamped'."""
    print("\nPRECISION: compute 256 + 0.5 in each format")
    for name, dt in FORMATS.items():
        a = torch.tensor(256.0, dtype=dt)
        b = torch.tensor(0.5, dtype=dt)
        stored = (a + b).item()
        flag = "  <- the 0.5 was swamped (lost)!" if stored == 256.0 else ""
        print(f"  {name}: 256 + 0.5 = {stored!r}{flag}")


def loss_scaling_demo():
    """The fix for fp16 underflow (preview of §5): multiply the tiny value UP into fp16's range
    before storing it, then divide back out in fp32. Nothing is lost."""
    g = 1e-8                                            # a gradient too small for fp16
    print("\nLOSS SCALING: rescue the underflowing 1e-8 gradient")
    naive = torch.tensor(g, dtype=torch.float16).item()
    print(f"  fp16(1e-8)            = {naive!r}   (lost)")
    scaled = torch.tensor(g * 1024, dtype=torch.float16)       # scale UP by 1024 first
    recovered = scaled.float().item() / 1024                   # unscale in fp32 afterwards
    print(f"  fp16(1e-8 * 1024)/1024 = {recovered!r}   (survived!)")
    print("  Scale the loss up before backward, unscale before the optimizer step — that's all")
    print("  loss scaling is. (bf16 keeps the range, so it never needs this.)")


def memory_savings():
    """Same tensor, three formats — the storage cost is just bytes-per-number x count."""
    n = 1_000_000
    print(f"\nMEMORY: one tensor of {n:,} numbers")
    for name, dt in FORMATS.items():
        t = torch.zeros(n, dtype=dt)
        mb = t.element_size() * t.numel() / 1e6
        print(f"  {name}: {t.element_size()} bytes each -> {mb:.1f} MB")
    print("Half the bytes per number means half the memory AND half the data to move — which,")
    print("on a GPU where moving data is the bottleneck, is most of where the speedup comes from.")


def main():
    print("=" * 78)
    print("FLOATING-POINT FORMATS: the same number, three budgets")
    print("=" * 78)
    budget_table()
    round_trip(3.14159265)          # pi: watch the digits get truncated
    round_trip(0.1)                 # 0.1 isn't exact in binary even in fp32
    range_failure()
    underflow_failure()
    swamping_failure()
    loss_scaling_demo()
    memory_savings()
    print("\n" + "=" * 78)
    print("Lesson: fp16 = fine precision but tiny range (overflow/underflow risk).")
    print("        bf16 = full range but coarse precision. Modern training picks bf16.")
    print("=" * 78)


if __name__ == "__main__":
    main()
