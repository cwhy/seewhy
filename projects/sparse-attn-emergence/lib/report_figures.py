"""
Report figures for sparse-attn-emergence, built on `shared_lib.typst_plot`.

Each function takes rows from results.jsonl and returns a `Figure` — data plus a
declarative gribouille spec — rather than drawing anything. `scripts/gen_report.py`
collects them and calls `write_figures()` once, so regeneration is idempotent and only
the figures that moved get rewritten.
"""

import json
from pathlib import Path

from shared_lib.typst_plot import bar_chart, line_chart

RESULTS = Path(__file__).parent.parent / "results.jsonl"
ARM_LABEL = {"transformer": "transformer", "mixer": "static mixer", "kda": "KDA"}
ARM_ORDER = ["transformer", "static mixer", "KDA"]


def load_results(path: Path | None = None) -> list[dict]:
    """Every row, first-write-wins on duplicates (concurrent shards and re-runs both
    duplicate; the first is the one that finished cleanly)."""
    rows, seen = [], set()
    src = path or RESULTS
    if not src.exists():
        return rows
    for line in src.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        name = r.get("experiment")
        if name and name not in seen and not str(name).startswith("smoke_"):
            seen.add(name)
            rows.append(r)
    return rows


def _best(rows, key_fn, rank_fn):
    """Keep the best row per key — used to collapse a learning-rate sweep."""
    out = {}
    for r in rows:
        k = key_fn(r)
        if k not in out or rank_fn(r) > rank_fn(out[k]):
            out[k] = r
    return out


# ─────────────────────────── positional task: the crossover ───────────────────

def crossover(rows):
    """Solve rate against sparsity for all three architectures on the linear map."""
    src = [r for r in rows if r["experiment"].startswith(("exp8_", "exp9_"))]
    best = _best(src, lambda r: (r["s"], r["arch"]),
                 lambda r: (r["solve_rate"], r.get("exact_rate", 0)))
    series = {}
    for (s, arm), r in sorted(best.items()):
        series.setdefault(ARM_LABEL[arm], {})[s] = r["solve_rate"]

    xs, ys, cs = [], [], []
    for arm in ARM_ORDER:
        for s, v in sorted(series.get(arm, {}).items()):
            xs.append(s)
            ys.append(v)
            cs.append(arm)
    return line_chart(
        "crossover", {"s": xs, "solve": ys, "architecture": cs},
        x="s", y="solve", colour="architecture", points=True,
        x_label="row sparsity s (positions the pattern must select)",
        y_label="fraction of 16 seeds solving", colour_label="architecture",
        y_limits=(-0.04, 1.04),
        alt="Solve rate against sparsity for transformer, static mixer and KDA.",
    )


# ─────────────────────────── content task: induction ──────────────────────────

def induction(rows):
    """Final recall accuracy on associative recall, best learning rate per arm."""
    src = [r for r in rows if r["experiment"].startswith("exp11_")
           and r.get("steps") == 30_000 and r.get("n_layers") == 2]
    best = _best(src, lambda r: r["arch"],
                 lambda r: (r["solve_rate"], -min(r["final_recall_loss"])))
    arms, acc = [], []
    for arm in ARM_ORDER:
        key = [k for k, v in ARM_LABEL.items() if v == arm][0]
        if key in best:
            r = best[key]
            arms.append(arm)
            acc.append(sorted(r["final_recall_acc"])[len(r["final_recall_acc"]) // 2])
    return bar_chart(
        "induction", {"architecture": arms, "recall": acc},
        x="architecture", y="recall", x_order=arms,
        x_label="", y_label="median recall accuracy", y_limits=(0, 1.05),
        hlines=[(1 / 32, "chance")],
        alt="Median recall accuracy on induction for each architecture.",
    )


# ─────────────────────────── memorisation vs in-context ───────────────────────

def pool_curve(rows):
    """In-context gain against rule-pool size, including the unmemorisable case."""
    src = [r for r in rows if r["experiment"].startswith("exp12_")]
    best = _best(src, lambda r: r["n_rules"],
                 lambda r: (r["solve_rate"], -(r.get("median_t_star") or 1e9)))
    labels, gains, tstar = [], [], []
    for k in sorted(best, key=lambda x: (x is None, x or 0)):
        labels.append("fresh" if k is None else str(k))
        gains.append(best[k]["in_context_gain"])
        tstar.append(best[k].get("median_t_star") or 0)
    return bar_chart(
        "pool_gain", {"pool": labels, "gain": gains},
        x="pool", y="gain", x_order=labels,
        x_label="rule pool size N (fresh = a new rule every sequence)",
        y_label="in-context gain (per-state loss, first − last)",
        alt="In-context gain against rule pool size.",
    ), bar_chart(
        "pool_tstar", {"pool": labels, "tstar": tstar},
        x="pool", y="tstar", x_order=labels,
        x_label="rule pool size N", y_label="median time-to-emergence (steps)",
        alt="Time to emergence against rule pool size.",
    )


# ─────────────────────────── background: H1 ───────────────────────────────────

def kofm_difficulty(rows):
    """Solve rate against k for the content-keyed task, both task variants.

    The ambiguous variant is kept because it is the control: if the curve were an artifact
    of low-k retrieval being ill-posed, fixing that would have flattened it.
    """
    xs, ys, cs = [], [], []
    for prefix, label in (("exp13_", "ambiguous match"), ("exp13u_", "unique match")):
        src = [r for r in rows if r["experiment"].startswith(prefix)]
        best = _best(src, lambda r: r["k"], lambda r: r["solve_rate"])
        for k, r in sorted(best.items()):
            xs.append(k)
            ys.append(r["solve_rate"])
            cs.append(label)
    return line_chart(
        "kofm_k", {"k": xs, "solve": ys, "variant": cs},
        x="k", y="solve", colour="variant", points=True,
        x_label="k — relevant attributes the match depends on (of m = 8)",
        y_label="fraction of 16 seeds solving", colour_label="task variant",
        y_limits=(-0.04, 1.04),
        alt="Solve rate rising with k for both variants of the k-of-m task.",
    )


def kofm_candidates(rows):
    """The same solve rates plotted against C(m,k) — the quantity that governs the
    positional task. Two cells share C = 28 and land at opposite ends."""
    src = [r for r in rows if r["experiment"].startswith("exp13u_")]
    best = _best(src, lambda r: r["k"], lambda r: r["solve_rate"])
    ks = sorted(best)
    return bar_chart(
        "kofm_candidates",
        {"cell": [f"k={k}, C={best[k]['candidates']}" for k in ks],
         "solve": [best[k]["solve_rate"] for k in ks]},
        x="cell", y="solve",
        x_order=[f"k={k}, C={best[k]['candidates']}" for k in
                 sorted(ks, key=lambda k: best[k]["candidates"])],
        x_label="ordered by candidate count C(m, k)", y_label="fraction of seeds solving",
        y_limits=(0, 1.05),
        alt="Solve rate ordered by candidate count, showing no relationship.",
    )


def emergence_spread(rows):
    """Per-seed time-to-emergence for the two independent 16-seed samples."""
    xs, ys, cs = [], [], []
    for name, label in (("exp1", "exp1"), ("exp4", "exp4")):
        r = next((x for x in rows if x["experiment"] == name), None)
        if not r:
            continue
        ts = r["t_star"]
        ts = ts.get("0.95") if isinstance(ts, dict) else ts
        for i, v in enumerate(sorted(t for t in ts if t)):
            xs.append(i + 1)
            ys.append(v)
            cs.append(label)
    return line_chart(
        "emergence_spread", {"rank": xs, "step": ys, "run": cs},
        x="rank", y="step", colour="run", points=True,
        x_label="seed, ordered by emergence time", y_label="step at which it emerged",
        colour_label="run",
        alt="Sorted time-to-emergence for two independent 16-seed samples.",
    )
