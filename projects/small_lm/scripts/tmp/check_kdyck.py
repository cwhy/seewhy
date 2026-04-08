"""
Sanity check for lib/kdyck.py: verify sequences are valid, check token ranges.

Usage:
    uv run python projects/small_lm/scripts/tmp/check_kdyck.py
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from lib.kdyck import KDyck


def verify_sequence(tokens, k, offset):
    """Check that a token sequence is a valid k-Dyck word."""
    stack = []
    for tok in tokens:
        if offset <= tok < offset + k:
            stack.append(tok - offset)
        elif offset + k <= tok < offset + 2 * k:
            bracket_type = tok - offset - k
            if not stack or stack[-1] != bracket_type:
                return False, f"mismatched close {tok}"
            stack.pop()
        else:
            return False, f"unexpected token {tok}"
    if stack:
        return False, f"unclosed: {stack}"
    return True, "ok"


def check(k, seq_len, n=200):
    print(f"\nk={k}, seq_len={seq_len}")
    kdyck = KDyck(k=k, seq_len=seq_len)

    batch = kdyck.generate_batch(n, seed=42)
    assert batch.shape == (n, seq_len), f"wrong shape: {batch.shape}"
    assert batch.dtype == np.int32

    # Token range
    assert batch.min() >= kdyck.token_offset, f"token below offset: {batch.min()}"
    assert batch.max() <= kdyck.token_offset + 2 * k - 1, f"token above range: {batch.max()}"

    # Validity
    failures = 0
    for i, seq in enumerate(batch):
        ok, msg = verify_sequence(seq.tolist(), k, kdyck.token_offset)
        if not ok:
            failures += 1
            print(f"  [FAIL] seq {i}: {msg}")
    if failures == 0:
        print(f"  [✓] All {n} sequences valid")
    else:
        print(f"  [✗] {failures}/{n} sequences invalid")

    # Show one example
    seq = batch[0].tolist()
    opens  = [t for t in seq if t < kdyck.token_offset + k]
    closes = [t for t in seq if t >= kdyck.token_offset + k]
    print(f"  Token range: {batch.min()}..{batch.max()}  "
          f"(opens={len(opens)}, closes={len(closes)})")
    print(f"  First 20 tokens: {seq[:20]}")

    # Epoch iteration
    batches = list(kdyck.generate_epoch(n_samples=1000, batch_size=32, epoch=0))
    print(f"  Epoch (1000 samples, bs=32): {len(batches)} batches")
    # Determinism: same epoch same data
    b0a = kdyck.generate_batch(32, seed=0)
    b0b = kdyck.generate_batch(32, seed=0)
    assert np.array_equal(b0a, b0b), "not deterministic!"
    print(f"  [✓] Deterministic")
    # Different seeds differ
    b1 = kdyck.generate_batch(32, seed=1)
    assert not np.array_equal(b0a, b1), "seeds don't differ!"
    print(f"  [✓] Different seeds produce different data")


if __name__ == "__main__":
    check(k=4,  seq_len=16)
    check(k=16, seq_len=128)
    check(k=64, seq_len=128)
    print("\nAll checks passed.")
