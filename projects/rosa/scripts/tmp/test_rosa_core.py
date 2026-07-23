"""
Smoke test for lib/rosa_core.py.

Checks:
  T1  SAM-deterministic recall on a simple repeat: feed [1,2,3,1,2,3], walk to
      after [1,2], expect deterministic '3'.
  T2  Argmax-along-stream returns 'sam' source whenever it can.
  T3  Boundary blocking: two examples [1,2,3] and [4,5,6]; after example
      boundaries the SAM should not deterministically suggest cross-boundary
      tokens.
  T4  Fallback LM Witten-Bell builds without errors and returns a valid prob
      dist.
  T5  Save/load round-trip preserves SAM topology and predictions.
  T6  Vocabulary scale: feed a 10K-token random stream over V=1024 and confirm
      the SAM grows to a sensible state count, predictions return SOMETHING
      from {sam, lm, unigram, empty}.

Run:
    uv run python projects/rosa/scripts/tmp/test_rosa_core.py
"""
from __future__ import annotations

import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from lib.rosa_core import ROSACore  # noqa: E402


def _check(name: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  — {detail}" if detail else ""))
    if not cond:
        sys.exit(1)


def t1_sam_deterministic_recall() -> None:
    print("T1: SAM-deterministic recall")
    rosa = ROSACore(vocab_size=16)
    rosa.commit_burst([1, 2, 3, 1, 2, 3])

    # Walk along [1, 2] and predict the next token.
    preds = rosa.predict_argmax_along_stream([1, 2])
    _check("preds length", len(preds) == 2, f"got {len(preds)}")
    pred_id, source = preds[-1]
    _check("after [1,2] → 3", pred_id == 3, f"got {pred_id}")
    _check("source=sam", source == "sam", f"got {source!r}")


def t2_argmax_along_stream_uses_sam() -> None:
    print("T2: stream-walk uses SAM whenever possible")
    rosa = ROSACore(vocab_size=16)
    rosa.commit_burst([1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
    preds = rosa.predict_argmax_along_stream([1, 2, 3, 4])
    sources = [src for _, src in preds]
    _check("all sources sam", all(s == "sam" for s in sources),
           f"got {sources}")
    ids = [tok for tok, _ in preds]
    _check("predictions match training", ids == [2, 3, 4, 5],
           f"got {ids}")


def t3_boundary_blocks_cross_example_recall() -> None:
    print("T3: example boundaries block cross-burst recall")
    rosa = ROSACore(vocab_size=16)
    rosa.commit_burst([7, 8, 9])
    rosa.commit_burst([7, 8, 10])

    # After [7,8] the SAM has TWO matches: end of burst-1 (-> 9) and middle of
    # burst-2 (-> 10). The deterministic rule should refuse to cross the
    # burst-1 boundary at position 2 (where 9 sits at the end of burst-1) and
    # so should return 10 (from burst-2's interior).
    preds = rosa.predict_argmax_along_stream([7, 8])
    _check("walk has 2 entries", len(preds) == 2)
    pred_id, source = preds[-1]
    _check("after [7,8] suggests 10 (boundary-respecting)",
           pred_id == 10 and source == "sam",
           f"got ({pred_id}, {source!r})")


def t4_fallback_lm_builds() -> None:
    print("T4: Witten-Bell fallback LM builds")
    rosa = ROSACore(vocab_size=16, max_order=64)
    examples = [
        [1, 2, 3, 4, 5],
        [2, 3, 5, 7, 9],
        [1, 1, 2, 3, 5],
    ]
    for ex in examples:
        rosa.commit_burst(ex)
    rosa.build_lm(examples)

    _check("lm exists", rosa.lm is not None)
    # Walk to after [2, 3]; force fallback by querying a state with
    # deterministic answer first, then deliberately query a novel suffix.
    # Fastest novel: query from root with token 99 (never seen)
    from lib.rosa_core import _advance_state  # noqa: E402

    sam = rosa.sam
    v = _advance_state(sam.b, sam.c, 0, 99)  # unknown token → root
    probs = rosa.lm.probs_for_state(v)
    _check("probs is dict", isinstance(probs, dict))
    s = sum(probs.values())
    _check("probs sum ≈ 1", abs(s - 1.0) < 1e-6, f"sum={s}")


def t5_save_load_roundtrip() -> None:
    print("T5: save/load round-trip")
    rosa = ROSACore(vocab_size=32, max_order=64)
    examples = [
        [0, 5, 10, 5, 10, 1, 7],
        [0, 6, 11, 6, 11, 1, 8],
    ]
    for ex in examples:
        rosa.commit_burst(ex)
    rosa.build_lm(examples)

    n_states_before = rosa.n_states()
    pred_before = rosa.predict_argmax_along_stream([0, 5, 10])

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    rosa.save(path)
    rosa2 = ROSACore.load(path)

    _check("state count preserved", rosa2.n_states() == n_states_before,
           f"{rosa2.n_states()} vs {n_states_before}")
    _check("token count preserved", rosa2.n_tokens() == rosa.n_tokens())
    pred_after = rosa2.predict_argmax_along_stream([0, 5, 10])
    _check("predictions preserved", pred_after == pred_before,
           f"{pred_after} vs {pred_before}")
    Path(path).unlink()


def t6_scale_smoke() -> None:
    print("T6: 10K-token random stream over V=1024")
    rng = random.Random(42)
    vocab = 1024
    rosa = ROSACore(vocab_size=vocab, max_order=64)

    n_bursts = 100
    burst_len = 100
    examples: list[list[int]] = []
    for _ in range(n_bursts):
        burst = [rng.randrange(2, vocab) for _ in range(burst_len)]
        examples.append(burst)
        rosa.commit_burst(burst)

    rosa.build_lm(examples)

    _check("sam states grew", rosa.n_states() > burst_len * n_bursts // 2,
           f"got {rosa.n_states()}")
    _check("tokens recorded", rosa.n_tokens() == n_bursts * burst_len)

    # Pick a random suffix from training and check we get a 'sam' hit.
    pick = examples[3][:7]
    preds = rosa.predict_argmax_along_stream(pick)
    sources = [src for _, src in preds]
    _check("at least one sam-hit on training suffix", "sam" in sources,
           f"got {sources}")

    # Truly novel tokens should fall to lm or unigram (no sam hit at root).
    preds = rosa.predict_argmax_along_stream([0])  # <SAMPLE>; never seen mid-burst
    _, src = preds[-1]
    _check("novel context falls back to non-sam", src in {"lm", "unigram"},
           f"got {src!r}")


def main() -> None:
    t1_sam_deterministic_recall()
    t2_argmax_along_stream_uses_sam()
    t3_boundary_blocks_cross_example_recall()
    t4_fallback_lm_builds()
    t5_save_load_roundtrip()
    t6_scale_smoke()
    print("\nAll T1–T6 passed.")


if __name__ == "__main__":
    main()
