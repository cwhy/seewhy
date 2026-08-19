"""Report figures, built on `shared_lib.typst_plot`.

Each function takes result rows and returns a `Figure` — data plus a
declarative spec — rather than drawing anything, so the same definition feeds
the markdown reports and the paper:

    write_figures(PROJECT / "paper", [fig])   # into the paper tree
    url = save_figure(fig)                     # standalone image for a .md report

Every accuracy figure carries its chance level, as a drawn reference line where
the geometry allows one and as an explicit series where it does not. Seed
spread is not drawn — the figures show medians, and the min-max range across
seeds goes in the accompanying table, which is where a reader can actually read
numbers off it.
"""

from __future__ import annotations

import statistics
from collections import defaultdict

from shared_lib.typst_plot import Figure, bar_chart, cm, line_chart, long_form

TASK_LABEL = {
    "mod_add": "modular addition",
    "needle": "needle-in-a-haystack",
    "decimal": "decimal addition",
    "parens": "parenthesis balancing",
}
TASK_ORDER = ["mod_add", "needle", "decimal", "parens"]

# Short forms for grouped bar charts. Four groups of four bars leaves too little
# width per category for the full names, and adjacent labels collide.
SHORT_LABEL = {
    "mod_add": "mod. add",
    "needle": "needle",
    "decimal": "decimal",
    "parens": "parens",
    "memorization": "memorize",
}


def _median(xs):
    return statistics.median(xs) if xs else float("nan")


def paper_metric(row: dict) -> float:
    """The accuracy the paper's tables report, for one run.

    The authors score *per token* over the positions their `compute_metrics`
    marks as wanted. For modular addition, needle-in-a-haystack and parenthesis
    balancing there is exactly one scored token per sequence, so token and
    sequence accuracy are the same number. Decimal addition is the exception:
    it scores ten or eleven digits plus a terminator, and the strict
    all-or-nothing reading is a substantially harder metric than the paper's.

    Reporting sequence accuracy there would have understated every model,
    including ours, against the published numbers.
    """
    return row["test_tok_acc"] if row.get("task") == "decimal" else row["test_seq_acc"]


def group(rows, *keys):
    """Bucket rows by a tuple of fields, dropping rows missing any of them."""
    out = defaultdict(list)
    for r in rows:
        if all(k in r for k in keys):
            out[tuple(r[k] for k in keys)].append(r)
    return out


def spread(rows, field=None):
    """``(median, min, max)`` across seeds — what the tables report.

    With no field, uses :func:`paper_metric`, which is the comparable number.
    """
    vals = [r[field] for r in rows] if field else [paper_metric(r) for r in rows]
    return _median(vals), min(vals), max(vals)


# ── exp1: the main table ──────────────────────────────────────────────────────

def _best_over_lr(rows, task, mode, d):
    """Best median accuracy for a cell, over every learning rate we ran for it.

    exp1 runs one learning rate. exp8 adds two more, but only for the width-16
    and LSTM cells — the baselines. Reporting a baseline at anything other than
    its best would inflate the comparison the paper is actually making, so those
    cells are taken at their best and the width-1024 condition under test is not.
    The asymmetry is deliberate and is stated in the limitations section.
    """
    groups = defaultdict(list)
    for r in rows:
        if r.get("task") != task or r.get("mode") != mode or r.get("d") != d:
            continue
        exp = r.get("experiment", "")
        if exp.startswith("exp1_") or exp == "exp8":
            groups[r.get("lr")].append(paper_metric(r))
    if not groups:
        return None
    return max(_median(v) for v in groups.values())


def main_table(rows, name: str = "main_table") -> Figure:
    """Sequence accuracy per task and condition, with chance drawn as its own bar.

    Chance is a series rather than a reference line because it differs per task,
    and a single dashed line across a grouped bar chart would be wrong for three
    of the four groups.
    """
    exp1 = [r for r in rows if r.get("experiment", "").startswith("exp1_")]

    conditions = [("random", 1024, "random 1024"), ("random", 16, "random 16"),
                  ("full", 1024, "trained 1024"), ("full", 16, "trained 16"),
                  ("lstm", 1024, "LSTM 1024")]

    task_col, cond_col, acc_col = [], [], []
    for task in TASK_ORDER:
        for mode, d, label in conditions:
            best = _best_over_lr(rows, task, mode, d)
            if best is None:
                continue
            task_col.append(SHORT_LABEL[task]); cond_col.append(label)
            acc_col.append(round(best, 4))
        chance = next((r["chance"] for r in exp1 if r["task"] == task), None)
        if chance is not None:
            task_col.append(SHORT_LABEL[task]); cond_col.append("chance")
            acc_col.append(round(chance, 4))

    return bar_chart(
        name, {"task": task_col, "condition": cond_col, "accuracy": acc_col},
        x="task", y="accuracy", fill="condition",
        x_order=[SHORT_LABEL[t] for t in TASK_ORDER], position="dodge",
        x_label="", y_label="sequence accuracy", fill_label="model",
        y_limits=(0.0, 1.0), width=cm(16), height=cm(7.5),
        alt="Grouped bars of sequence accuracy on four algorithmic tasks for "
            "random and fully trained transformers at widths 1024 and 16, a "
            "fully trained LSTM, and the chance level for each task.",
    )


# ── exp1 + exp3: the width sweep ──────────────────────────────────────────────

def width_sweep(rows, task: str, name: str | None = None) -> Figure:
    """Accuracy against width for one task, random vs fully trained.

    exp1 supplies the endpoints (16 and 1024) and exp3 the interior, so both are
    read here rather than in either experiment.
    """
    rs = [r for r in rows
          if r.get("task") == task
          and (r.get("experiment", "").startswith("exp1_") or r.get("experiment") == "exp3")
          and r.get("mode") in ("random", "full")]
    by = group(rs, "mode", "d")
    widths = sorted({d for _, d in by})

    series = {}
    for mode, label in (("random", "random (embeddings only)"), ("full", "fully trained")):
        series[label] = [round(_median([paper_metric(r) for r in by[(mode, w)]]), 4)
                         if (mode, w) in by else None for w in widths]
    series = {k: v for k, v in series.items() if any(x is not None for x in v)}

    chance = next((r["chance"] for r in rs), 0.0)
    data = long_form(widths, series, x_name="width", y_name="accuracy", series_name="training")

    return line_chart(
        name or f"width_{task}", data,
        x="width", y="accuracy", colour="training", points=True, log_x=True,
        x_label="hidden width", y_label="sequence accuracy", colour_label="",
        y_limits=(0.0, 1.0), x_limits=(min(widths), max(widths)),
        hlines=[(round(chance, 4), "chance")],
        title=TASK_LABEL.get(task, task),
        width=cm(13), height=cm(7),
        alt=f"Sequence accuracy on {TASK_LABEL.get(task, task)} against hidden "
            "width on a log scale, for embedding-only and fully trained models, "
            "with a dashed chance line.",
    )


# ── exp2: which matrices must be trained ──────────────────────────────────────

def ablation(rows, name: str = "ablation") -> Figure:
    exp2 = [r for r in rows if r.get("experiment") == "exp2"]
    exp1 = [r for r in rows if r.get("experiment", "").startswith("exp1_")
            and r.get("mode") == "random" and r.get("d") == 1024]
    by = group(exp2 + exp1, "task", "mode")

    labels = [("random", "all three (E_token, E_pos, U)"), ("etoken_u", "E_token & U"),
              ("e_only", "E_token & E_pos"), ("u_only", "U only")]
    task_col, var_col, acc_col = [], [], []
    for task in TASK_ORDER:
        for mode, label in labels:
            rs = by.get((task, mode))
            if not rs:
                continue
            task_col.append(SHORT_LABEL[task]); var_col.append(label)
            acc_col.append(round(_median([paper_metric(r) for r in rs]), 4))

    return bar_chart(
        name, {"task": task_col, "trained": var_col, "accuracy": acc_col},
        x="task", y="accuracy", fill="trained",
        x_order=[SHORT_LABEL[t] for t in TASK_ORDER], position="dodge",
        x_label="", y_label="sequence accuracy", fill_label="optimised",
        y_limits=(0.0, 1.0), width=cm(16), height=cm(7.5),
        alt="Grouped bars showing sequence accuracy at width 1024 when different "
            "subsets of the embedding matrices are optimised.",
    )


# ── exp4: memorization capacity ───────────────────────────────────────────────

def memorization_bits(rows, name: str = "memorization_bits") -> Figure:
    exp4 = [r for r in rows if r.get("experiment") == "exp4"]
    by = group(exp4, "mode")
    mode_col, val_col = [], []
    for mode, label in (("full", "fully trained"), ("random", "random")):
        if (mode,) in by:
            mode_col.append(label)
            val_col.append(round(_median([r["bits_per_trainable_param"] for r in by[(mode,)]]), 3))
    return bar_chart(
        name, {"model": mode_col, "bits": val_col},
        x="model", y="bits", x_label="", y_label="bits per trainable parameter",
        width=cm(10), height=cm(7),
        alt="Bits of arbitrary association stored per trainable parameter, for "
            "fully trained and random transformers on the memorization task.",
    )


# ── exp5: subspace selection vs sparsification ────────────────────────────────

def subspace(rows, layer: str = "L1", name: str | None = None) -> Figure:
    """Top-10 explained variance in the principal-component basis and the neuron
    basis, at one layer. The gap between the two bars *is* the claim."""
    exp5 = [r for r in rows if r.get("experiment") == "exp5"]
    by = group(exp5, "task", "mode")

    task_col, basis_col, val_col = [], [], []
    order = TASK_ORDER + ["memorization"]
    for task in order:
        for mode, mlabel in (("random", "random"), ("full", "trained")):
            rs = by.get((task, mode))
            if not rs or f"pc_{layer}" not in rs[0]:
                continue
            for key, blabel in ((f"pc_{layer}", "top 10 components"),
                                (f"neuron_{layer}", "top 10 neurons")):
                task_col.append(SHORT_LABEL.get(task, task))
                basis_col.append(f"{mlabel}: {blabel}")
                val_col.append(round(_median([r[key] for r in rs]), 4))

    return bar_chart(
        name or f"subspace_{layer}",
        {"task": task_col, "basis": basis_col, "variance": val_col},
        x="task", y="variance", fill="basis",
        x_order=[SHORT_LABEL.get(t, t) for t in order], position="dodge",
        x_label="", y_label="fraction of variance explained", fill_label="",
        y_limits=(0.0, 1.0), width=cm(17), height=cm(7.5),
        title=f"activations after layer {layer[-1]}" if layer.startswith("L") else "embeddings",
        alt="Grouped bars comparing the fraction of activation variance explained "
            "by the top ten principal components against the top ten individual "
            "neurons, per task and training regime.",
    )


# ── exp6: circuit imitation ───────────────────────────────────────────────────

def circuit_imitation(rows, name: str = "circuit_imitation") -> Figure:
    exp6 = [r for r in rows if r.get("experiment") == "exp6"]
    by = group(exp6, "mode", "target_width")
    widths = sorted({w for _, w in by})
    series = {}
    for mode, label in (("random", "random student"), ("full", "fully trained student")):
        vals = [round(_median([r["final_kl"] for r in by[(mode, w)]]), 4)
                if (mode, w) in by else None for w in widths]
        if any(v is not None for v in vals):
            series[label] = vals
    data = long_form(widths, series, x_name="target_width", y_name="kl", series_name="student")
    return line_chart(
        name, data, x="target_width", y="kl", colour="student", points=True, log_x=True,
        x_label="width of the target circuit", y_label="held-out KL divergence",
        colour_label="", width=cm(13), height=cm(7),
        alt="KL divergence from a random target transformer against the target's "
            "hidden width, on a log scale, for random and fully trained students.",
    )


# ── exp7: language modeling ───────────────────────────────────────────────────

def lm_scaling(rows, name: str = "lm_scaling") -> Figure:
    exp7 = [r for r in rows if r.get("experiment") == "exp7"]
    by = group(exp7, "mode", "n_layer", "d")
    series, xs = {}, sorted({d for _, _, d in by})
    for mode, mlabel in (("random", "random"), ("full", "fully trained")):
        for depth in sorted({l for _, l, _ in by}):
            vals = [round(_median([r["eval_loss"] for r in by[(mode, depth, d)]]), 4)
                    if (mode, depth, d) in by else None for d in xs]
            if any(v is not None for v in vals):
                series[f"{mlabel}, {depth} layers"] = vals
    data = long_form(xs, series, x_name="width", y_name="loss", series_name="model")
    return line_chart(
        name, data, x="width", y="loss", colour="model", points=True, log_x=True,
        x_label="hidden width", y_label="validation cross-entropy (nats/token)",
        colour_label="", width=cm(13), height=cm(7.5),
        alt="Validation cross-entropy on TinyStories against hidden width on a "
            "log scale, for random and fully trained transformers at two depths.",
    )


# ── the set gen_report.py writes ──────────────────────────────────────────────

def build_figures(rows) -> list[Figure]:
    """Every figure the results currently support.

    Figures whose experiment has not run yet are skipped rather than emitted
    empty, so the paper never picks up a plot of nothing.
    """
    have = {r.get("experiment", "") for r in rows}
    figs: list[Figure] = []

    if any(e.startswith("exp1_") for e in have):
        figs.append(main_table(rows))
        for task in TASK_ORDER:
            if any(r.get("task") == task for r in rows):
                figs.append(width_sweep(rows, task))
    if "exp2" in have:
        figs.append(ablation(rows))
    if "exp4" in have:
        figs.append(memorization_bits(rows))
    if "exp5" in have:
        for layer in ("emb", "L1", "L2"):
            figs.append(subspace(rows, layer))
    if "exp6" in have:
        figs.append(circuit_imitation(rows))
    if "exp7" in have:
        figs.append(lm_scaling(rows))

    return figs
