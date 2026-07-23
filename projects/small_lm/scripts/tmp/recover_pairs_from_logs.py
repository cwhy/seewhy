"""
Recover (question, exact_sentence) pairs from kylo training logs.
Questions are logged at [iter N] q: "..." and labels at [interp] → "..."
Due to async design, interp for iter N is logged at top of iter N+1.
Note: both fields may be truncated in logs (questions at 70 chars, labels at 70 chars).
"""
import json
import re
import sys
from pathlib import Path

SMALL_LM = Path(__file__).parent.parent.parent

LOG_FILES = [
    SMALL_LM / "scripts" / "tmp" / "kylo_run.log",
    SMALL_LM / "scripts" / "tmp" / "kylo_run2.log",
    SMALL_LM / "scripts" / "tmp" / "kylo_run3.log",
]

OUT_FILE = SMALL_LM / "babystep" / "experiment-2-kylo" / "recovered_pairs.jsonl"

RE_QUESTION = re.compile(r'\[iter (\d+)\] q: "(.+)"$')
RE_INTERP   = re.compile(r'\[interp\] → "(.+)"$')

def parse_log(path: Path):
    """Extract ordered list of (question, iter) and interp labels from log."""
    questions = []   # list of (iter_idx, question_text)
    labels    = []   # list of label_text (in order they appear in log)

    with open(path) as f:
        for line in f:
            line = line.rstrip()
            m = RE_QUESTION.search(line)
            if m:
                questions.append((int(m.group(1)), m.group(2)))
                continue
            m = RE_INTERP.search(line)
            if m:
                labels.append(m.group(1))

    return questions, labels


all_pairs = []
for log_path in LOG_FILES:
    if not log_path.exists():
        print(f"  SKIP (not found): {log_path.name}")
        continue

    questions, labels = parse_log(log_path)
    print(f"{log_path.name}: {len(questions)} questions, {len(labels)} labels")

    # async offset: interp for question[i] is logged as label[i]
    # (label[0] would be from iter 0's bootstrap question, skip it)
    # In practice labels[0] corresponds to questions[0], labels[1] to questions[1], etc.
    # — the bootstrap question has no matching q: line, so labels lead by 1.
    # Heuristic: zip(questions, labels[1:]) drops the first label (bootstrap).
    pairs_this = []
    label_offset = 1 if len(labels) > len(questions) else 0
    for (iter_idx, q), label in zip(questions, labels[label_offset:]):
        pairs_this.append({
            "question": q,
            "exact_sentence": label,
            "source_log": log_path.name,
            "iter": iter_idx,
            "truncated": len(q) >= 69 or len(label) >= 69,
        })

    print(f"  → {len(pairs_this)} pairs recovered")
    all_pairs.extend(pairs_this)

# Append to existing pairs file (don't overwrite)
existing = 0
if OUT_FILE.exists():
    with open(OUT_FILE) as f:
        existing = sum(1 for _ in f)
    print(f"\nExisting recovered_pairs.jsonl: {existing} lines — appending")

with open(OUT_FILE, "a") as f:
    for p in all_pairs:
        f.write(json.dumps(p) + "\n")

total = existing + len(all_pairs)
truncated = sum(1 for p in all_pairs if p["truncated"])
print(f"\nTotal pairs written: {len(all_pairs)}  (truncated: {truncated})")
print(f"recovered_pairs.jsonl now has {total} entries")
print(f"\nSample:")
for p in all_pairs[:3]:
    print(f"  Q: {p['question']}")
    print(f"  A: {p['exact_sentence']}")
    print()
