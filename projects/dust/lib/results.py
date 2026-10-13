"""Read results.jsonl. Plain stdlib, no JAX, so reports and figures import it anywhere.

Unlike a one-row-per-experiment project, many rows here share an `experiment` (exp3 is six
runs), so `shared_lib.results.load_results` -- which keys by experiment -- is the wrong
reader. A row is identified by its `name`; if two share one, the later wins (a rerun).
"""

import json
import statistics
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results.jsonl"


def load(path: Path = RESULTS) -> list[dict]:
    rows: dict[str, dict] = {}
    for line in Path(path).read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        rows[r["name"]] = r
    return list(rows.values())


# Experiments whose rows are *ladder* runs: the final settings, three seeds (one at 10M and 4,096). Everything else
# in results.jsonl is tuning, ablation or a checkpoint, and must never be averaged into a ladder cell: exp4 alone
# holds twelve different DUST 64-draw configurations, which would otherwise pool into one "64 draws" number.
LADDER_EXPS = ("exp1", "exp3", "exp9", "exp10", "exp11", "exp13", "exp17")


def train_rows(rows: list[dict], exps: tuple[str, ...] | None = None, **where) -> list[dict]:
    """Training runs (not cosine / parity / constants rows) matching every key=value in `where`,
    optionally restricted to the experiments in `exps`."""
    out = []
    for r in rows:
        if r.get("kind") in ("cosine", "parity", "constants", "theory", "extrapolation"):
            continue
        if exps is not None and r["experiment"] not in exps:
            continue
        if all(r.get(k) == v for k, v in where.items()):
            out.append(r)
    return out


def mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.fmean(values), (statistics.stdev(values) if len(values) > 1 else float("nan"))


def cell(rows: list[dict], key: str = "test_at_best_val", exps: tuple[str, ...] | None = LADDER_EXPS, **where) -> dict | None:
    """Mean / sd / n of `key` over the ladder runs matching `where`; None if there are none.
    Pass exps=None to pool every experiment (almost never what you want)."""
    rs = train_rows(rows, exps=exps, **where)
    if not rs:
        return None
    m, sd = mean_sd([r[key] for r in rs])
    return dict(mean=m, sd=sd, n=len(rs), seeds=sorted(r["seed"] for r in rs),
                step_s=statistics.fmean(r["step_s"] for r in rs))
