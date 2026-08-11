"""
Reading `results.jsonl`.

Every project writes one: append-only, one JSON object per experiment, with all
hyperparameters, `n_params`, `time_s` and the per-epoch curves. Every project
then needs the same two things to plot from it, and both existing projects grew
their own copy — this is that code, once.

    from shared_lib.results import load_results, run_order

    rows = load_results(PROJECT / "results.jsonl")
    for name in run_order(rows):
        ...
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def load_results(path: str | Path, *, key: str = "experiment") -> dict[str, dict]:
    """Read `results.jsonl` into {experiment: row}, first write wins.

    Duplicates are normal rather than exceptional: two runners that both read
    the skip-if-done set before either had written will each append the same
    experiment, and a post-crash rerun does the same. The first row is the one
    whose run finished cleanly, so it is the one kept.

    Malformed lines are skipped rather than raising — a results file truncated
    mid-write by a killed job should still yield the runs that completed.
    """
    rows: dict[str, dict] = {}
    src = Path(path)
    if not src.exists():
        return rows
    for line in src.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.setdefault(str(row.get(key, "?")), row)
    return rows


def run_order(names: Iterable[str]) -> list[str]:
    """Experiment names in numeric order.

    Plain `sorted()` puts exp10 between exp1 and exp2, which silently scrambles
    every axis and legend built from it.
    """
    def sort_key(name: str) -> tuple[int, str]:
        digits = "".join(c for c in name if c.isdigit())
        return (int(digits) if digits else 0, name)

    return sorted(names, key=sort_key)


def latest(rows: dict[str, dict], field: str, default: Any = None) -> Any:
    """The value of `field` from the highest-numbered run that has it.

    For figures that need one representative setting (image size, chance level)
    rather than a per-run value.
    """
    for name in reversed(run_order(rows)):
        if field in rows[name]:
            return rows[name][field]
    return default
