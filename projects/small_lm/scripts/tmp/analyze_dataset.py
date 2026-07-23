"""
Analyze the small_lm dataset: size, category distribution, diversity, and
indicators of AI/template generation.

Usage:
    uv run python projects/small_lm/scripts/tmp/analyze_dataset.py
"""

import json
from collections import Counter
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent / "data"


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_text(text):
    """Extract user input and assistant output from chat-ml format."""
    try:
        user_part = text.split("<|im_start|>user\n")[1].split("<|im_end|>")[0].strip()
        asst_part = text.split("<|im_start|>assistant\n")[1].split("<|im_end|>")[0].strip()
        return user_part, asst_part
    except (IndexError, KeyError):
        return "", ""


def analyze(rows, label):
    print(f"\n{'='*60}")
    print(f"{label}  ({len(rows):,} samples)")
    print('='*60)

    parsed  = [parse_text(r["text"]) for r in rows]
    inputs  = [p[0] for p in parsed]
    outputs = [p[1] for p in parsed]
    cats    = [r.get("category", "?") for r in rows]

    # ── Basic counts ──────────────────────────────────────────────────────────
    unique_inputs  = len(set(inputs))
    unique_outputs = len(set(outputs))
    print(f"\nUnique inputs:  {unique_inputs:,} / {len(inputs):,}  ({unique_inputs/len(inputs)*100:.1f}%)")
    print(f"Unique outputs: {unique_outputs:,} / {len(outputs):,}  ({unique_outputs/len(outputs)*100:.1f}%)")

    # ── Category breakdown ────────────────────────────────────────────────────
    cat_counts = Counter(cats)
    print(f"\nCategories: {len(cat_counts)}")
    for cat, n in sorted(cat_counts.items(), key=lambda x: -x[1])[:25]:
        bar = "█" * (n * 40 // max(cat_counts.values()))
        print(f"  {cat:<30} {n:>5}  {bar}")
    if len(cat_counts) > 25:
        print(f"  ... and {len(cat_counts)-25} more")

    # ── Output length distribution ────────────────────────────────────────────
    lens = [len(o.split()) for o in outputs]
    print(f"\nOutput length (words): min={min(lens)}  max={max(lens)}  "
          f"mean={sum(lens)/len(lens):.1f}  median={sorted(lens)[len(lens)//2]}")

    buckets = Counter(l // 5 * 5 for l in lens)
    print("  Distribution (word buckets):")
    for b in sorted(buckets)[:15]:
        bar = "█" * (buckets[b] * 30 // max(buckets.values()))
        print(f"    {b:>3}-{b+4:<3} words: {buckets[b]:>5}  {bar}")

    # ── Most common outputs (template repetition indicator) ───────────────────
    print(f"\nTop 10 most repeated outputs:")
    for out, n in Counter(outputs).most_common(10):
        print(f"  {n:>4}×  {out[:80]}")

    # ── Most common input patterns ────────────────────────────────────────────
    print(f"\nTop 10 most repeated inputs:")
    for inp, n in Counter(inputs).most_common(10):
        print(f"  {n:>4}×  {inp[:80]}")

    # ── Vocabulary size ───────────────────────────────────────────────────────
    all_words = " ".join(outputs).lower().split()
    vocab = set(all_words)
    print(f"\nOutput vocabulary: {len(vocab):,} unique words from {len(all_words):,} total")
    word_freq = Counter(all_words)
    print(f"Top 20 output words: {[w for w,_ in word_freq.most_common(20)]}")

    # ── Template indicator: how many outputs share a prefix ──────────────────
    prefix_counts = Counter(o[:20] for o in outputs)
    repeated_prefixes = {p: n for p, n in prefix_counts.items() if n > 1}
    print(f"\nOutputs sharing a common 20-char prefix: "
          f"{sum(repeated_prefixes.values()):,} ({sum(repeated_prefixes.values())/len(outputs)*100:.1f}%)")
    print("Top prefix patterns:")
    for p, n in sorted(repeated_prefixes.items(), key=lambda x: -x[1])[:10]:
        print(f"  {n:>5}×  '{p}'")

    # ── Sample outputs per category ───────────────────────────────────────────
    print(f"\nSample output per category (first occurrence):")
    seen_cats = set()
    for r, (inp, out) in zip(rows, parsed):
        c = r.get("category", "?")
        if c not in seen_cats:
            seen_cats.add(c)
            print(f"  [{c}]")
            print(f"    in:  {inp[:70]}")
            print(f"    out: {out[:100]}")
        if len(seen_cats) >= 15:
            break


def main():
    train = load_jsonl(DATA_DIR / "train.jsonl")
    eval_ = load_jsonl(DATA_DIR / "eval.jsonl")

    analyze(train, "TRAIN SET")
    analyze(eval_,  "EVAL SET")

    print(f"\n{'='*60}")
    print(f"COMBINED TOTALS")
    print('='*60)
    all_rows = train + eval_
    all_cats = Counter(r.get("category","?") for r in all_rows)
    print(f"Total samples:    {len(all_rows):,}")
    print(f"Total categories: {len(all_cats)}")
    all_parsed  = [parse_text(r["text"]) for r in all_rows]
    all_inputs  = [p[0] for p in all_parsed]
    all_outputs = [p[1] for p in all_parsed]
    print(f"Unique outputs:   {len(set(all_outputs)):,} "
          f"({len(set(all_outputs))/len(all_rows)*100:.1f}%)")
    print(f"Unique inputs:    {len(set(all_inputs)):,} "
          f"({len(set(all_inputs))/len(all_rows)*100:.1f}%)")


if __name__ == "__main__":
    main()
