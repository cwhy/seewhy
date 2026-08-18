"""Appending to `results.jsonl` from more than one process at a time.

Two experiment processes run concurrently, one per GPU. A result row carries
its full per-eval history, so it is several kilobytes — comfortably over the
`PIPE_BUF` limit below which an `O_APPEND` write is atomic. Without a lock, two
overlapping appends can interleave mid-line and produce a row that no longer
parses, which is only discovered much later when the figures are built.

`flock` is advisory, so every writer has to go through this function.
"""

from __future__ import annotations

import fcntl
import json
from pathlib import Path


def append_result(jsonl: Path, row: dict) -> None:
    line = json.dumps(row) + "\n"
    with open(jsonl, "a") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(line)
            f.flush()
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def read_rows(jsonl: Path) -> list[dict]:
    """Every parseable row, newest last. Unparseable lines are reported, not hidden."""
    if not jsonl.exists():
        return []
    rows, bad = [], 0
    for line in jsonl.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            bad += 1
    if bad:
        print(f"WARNING: {bad} unparseable line(s) in {jsonl}")
    return rows
