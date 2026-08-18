"""Run a grid of training cells, resumably.

exp1-exp4 are all the same shape: a set of (task, mode, width, seed) cells, one
result row each, appended as it finishes so an interrupted run resumes instead
of restarting. The only thing that differs is which cells are in the grid.

Cells are identified by a string key that goes into the row as `cell`, and the
skip-if-done check reads those keys back out of `results.jsonl`. Two processes
running disjoint slices of a grid is the normal case here — one per GPU — so
appends go through the locking writer in `results_io`.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Callable, Iterable

from .results_io import append_result, read_rows
from .train import strip_params, train_task


def done_cells(jsonl: Path, prefix: str) -> set[str]:
    return {r["cell"] for r in read_rows(jsonl)
            if r.get("experiment", "").startswith(prefix) and "cell" in r}


def run_grid(
    cells: Iterable[dict],
    *,
    jsonl: Path,
    exp: str,
    task_for: Callable[[str], object],
    save_dir: Path | None = None,
    save_when: Callable[[dict], bool] = lambda c: False,
    extra_row: Callable[[dict], dict] = lambda c: {},
):
    """Train every cell not already in ``jsonl``.

    Each cell is a dict with ``key`` plus the keyword arguments for
    :func:`~lib.train.train_task` (``task`` given as a name in ``task_name``).
    Tasks are built lazily and cached, because building the parenthesis pool
    costs seconds and the memorization map costs memory.
    """
    done = done_cells(jsonl, exp)
    cells = list(cells)
    todo = [c for c in cells if c["key"] not in done]
    logging.info(f"{exp}: {len(cells)} cells, {len(cells) - len(todo)} already done, "
                 f"{len(todo)} to run")

    cache: dict[str, object] = {}

    for i, cell in enumerate(todo, 1):
        name = cell["task_name"]
        if name not in cache:
            cache.clear()          # only ever one task resident
            cache[name] = task_for(name)
        task = cache[name]

        logging.info(f"[{i}/{len(todo)}] {cell['key']}")
        kwargs = {k: v for k, v in cell.items() if k not in ("key", "task_name")}
        row = train_task(task, **kwargs)

        if save_dir is not None and save_when(cell):
            path = save_dir / f"params_{exp}_{cell['key'].replace('/', '_')}.pkl"
            with open(path, "wb") as f:
                pickle.dump(row["_params"], f)

        row = strip_params(row)
        row.update({"experiment": exp, "cell": cell["key"]})
        row.update(extra_row(cell))
        row.setdefault("name", f"{exp} — {cell['key']}")
        append_result(jsonl, row)
        logging.info(f"    seq_acc={row['test_seq_acc']:.4f} "
                     f"tok_acc={row['test_tok_acc']:.4f} "
                     f"chance={row['chance']:.4f} ({row['time_s']:.0f}s)")

    logging.info(f"{exp} complete")
