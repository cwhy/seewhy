"""Report figures, built on `shared_lib.typst_plot`.

Each function takes the rows of `results.jsonl` and returns a `Figure` — data
plus a declarative spec — so the same definition serves the paper tree and a
standalone markdown report.

Rows are one per cell: one (method, architecture, seed) run, tagged with the
experiment it belongs to.
"""

import statistics as stats

from shared_lib.typst_plot import Figure, bar_chart, cm, line_chart, long_form

# The tweet being replicated, for the side-by-side bars.
TWEET = {
    "backprop 32x32 res": {"acc": 0.8739, "params": 56128, "train_s": 0.324, "infer_s": 0.0229},
    "PC-ALM 32x32 res": {"acc": 0.8628, "params": 56128, "train_s": 13.485, "infer_s": 0.0241},
    "Forward-Forward 2x66": {"acc": 0.8883, "params": 56232, "train_s": 0.549, "infer_s": 0.0781},
}

# Display order and labels for the five arms of the one-epoch comparison.
ARMS = [
    ("exp1", "bp", "", "backprop 32x32 res"),
    ("exp1", "pc", "", "PC 32x32 res"),
    ("exp1", "pcalm", "", "PC-ALM 32x32 res"),
    ("exp2", "ff", "", "Forward-Forward 2x66"),
    ("exp2", "bp_plain", "", "backprop 2x66"),
    ("exp2", "ff", "deep", "Forward-Forward 31x32"),
]

METHOD_LABEL = {"bp": "backprop", "pc": "PC", "pcalm": "PC-ALM",
                "ff": "Forward-Forward", "bp_plain": "backprop (plain)"}


# ── row selection ────────────────────────────────────────────────────────────

def cells(rows: dict[str, dict], exp: str, **match) -> list[dict]:
    out = []
    for row in rows.values():
        if row.get("exp") != exp:
            continue
        if all(row.get(k) == v for k, v in match.items()):
            out.append(row)
    return out


def mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def spread(values) -> float:
    values = list(values)
    return stats.stdev(values) if len(values) > 1 else 0.0


# ── figures ──────────────────────────────────────────────────────────────────

def accuracy_vs_tweet(rows: dict[str, dict], name: str = "accuracy_vs_tweet") -> Figure:
    """Our five arms against the three numbers the tweet reports."""
    labels, accs, source = [], [], []
    for exp, method, tag, label in ARMS:
        got = cells(rows, exp, method=method, tag=tag)
        if not got:
            continue
        labels.append(label)
        accs.append(mean(c["test_acc"] for c in got))
        source.append("this replication")
        if label in TWEET:
            labels.append(label)
            accs.append(TWEET[label]["acc"])
            source.append("the tweet")
    return bar_chart(
        name, {"arm": labels, "accuracy": accs, "source": source},
        x="arm", y="accuracy", fill="source", position="dodge", horizontal=True,
        x_order=[label for _, _, _, label in ARMS][::-1],
        x_label="", y_label="test accuracy after one epoch",
        fill_label="", y_limits=(0.0, 1.0), hlines=[(0.1, "chance")],
        width=cm(14), height=cm(8),
        alt="Bar chart of one-epoch MNIST test accuracy for six arms, each of "
            "this replication's means shown beside the tweet's number where the "
            "tweet reports one, with a dashed line at 10% chance.",
    )


def cost_by_method(rows: dict[str, dict], field: str, name: str, y_label: str) -> Figure:
    labels, values = [], []
    for exp, method, tag, label in ARMS:
        got = cells(rows, exp, method=method, tag=tag)
        if got:
            labels.append(label)
            values.append(mean(c[field] for c in got))
    return bar_chart(
        name, {"arm": labels, y_label: values},
        x="arm", y=y_label, horizontal=True,
        x_order=labels[::-1], x_label="", y_label=y_label,
        width=cm(14), height=cm(6.5),
        alt=f"Bar chart of {y_label} for each arm of the one-epoch comparison.",
    )


def accuracy_vs_walltime(rows: dict[str, dict], name: str = "accuracy_vs_walltime") -> Figure:
    """Test accuracy against cumulative training seconds, ten epochs per method."""
    series: dict[str, tuple[list[float], list[float]]] = {}
    for exp, method, tag, label in ARMS:
        got = cells(rows, "exp4", method=method, tag=tag)
        if not got:
            continue
        n_epochs = min(len(c["history"]) for c in got)
        xs = [mean(c["history"][e]["train_s"] for c in got) for e in range(n_epochs)]
        ys = [mean(c["history"][e]["test_acc"] for c in got) for e in range(n_epochs)]
        series[label] = (xs, ys)

    data = {"train_s": [], "accuracy": [], "method": []}
    for label, (xs, ys) in series.items():
        data["train_s"] += xs
        data["accuracy"] += ys
        data["method"] += [label] * len(xs)
    return line_chart(
        name, data, x="train_s", y="accuracy", colour="method", points=True,
        log_x=True, x_label="cumulative training time (s, log scale)",
        y_label="test accuracy", colour_label="",
        y_limits=(0.0, 1.0), hlines=[(0.1, "chance")],
        width=cm(14), height=cm(8),
        alt="Test accuracy against cumulative training wall-clock on a log axis, "
            "one line per learning rule, ten epochs each.",
    )


def accuracy_vs_depth(rows: dict[str, dict], name: str = "accuracy_vs_depth") -> Figure:
    got = cells(rows, "exp5")
    depths = sorted({r["depth"] if r["method"] in ("bp", "pc", "pcalm")
                     else len(r["hidden"]) + 1 for r in got})
    series: dict[str, list[float]] = {}
    for method in ("bp", "pc", "pcalm", "ff", "bp_plain"):
        ys = []
        for L in depths:
            sel = [r for r in got if r["method"] == method and r.get("tag") == f"L{L}"]
            ys.append(mean(r["test_acc"] for r in sel) if sel else float("nan"))
        if any(y == y for y in ys):
            series[METHOD_LABEL[method]] = ys
    data = long_form(depths, series, x_name="depth", y_name="accuracy", series_name="method")
    data = {k: [v for v, y in zip(col, data["accuracy"]) if y == y]
            for k, col in data.items()}
    return line_chart(
        name, data, x="depth", y="accuracy", colour="method", points=True,
        log_x=True, x_label="layers", y_label="test accuracy after one epoch",
        colour_label="", y_limits=(0.0, 1.0), hlines=[(0.1, "chance")],
        width=cm(14), height=cm(8),
        alt="Test accuracy against network depth on a log axis, one line per "
            "learning rule, width held at 32 units.",
    )


def grad_cosine_vs_depth(rows: dict[str, dict], name: str = "grad_cosine_vs_depth") -> Figure:
    got = cells(rows, "exp5")
    depths = sorted({r["depth"] for r in got if r["method"] in ("pc", "pcalm")})
    series = {}
    for method in ("pc", "pcalm"):
        ys = []
        for L in depths:
            sel = [r for r in got if r["method"] == method and r["depth"] == L
                   and r.get("grad_cos_to_bp") is not None]
            ys.append(mean(r["grad_cos_to_bp"] for r in sel) if sel else float("nan"))
        series[METHOD_LABEL[method]] = ys
    data = long_form(depths, series, x_name="depth", y_name="cosine", series_name="method")
    return line_chart(
        name, data, x="depth", y="cosine", colour="method", points=True,
        log_x=True, x_label="layers", y_label="cosine to the backprop gradient",
        colour_label="", y_limits=(0.0, 1.05), hlines=[(1.0, "backprop")],
        width=cm(14), height=cm(7),
        alt="Cosine similarity between each local rule's weight update and the "
            "backprop gradient, against depth.",
    )


def learning_rate_grid(rows: dict[str, dict], name: str = "learning_rate_grid") -> Figure:
    got = cells(rows, "exp3")
    rates = sorted({r["lr"] for r in got})
    series = {}
    for method in ("bp", "pc", "pcalm", "ff", "bp_plain"):
        ys = []
        for lr in rates:
            sel = [r for r in got if r["method"] == method and abs(r["lr"] - lr) < 1e-12]
            ys.append(mean(r["test_acc"] for r in sel) if sel else float("nan"))
        if any(y == y for y in ys):
            series[METHOD_LABEL[method]] = ys
    data = long_form(rates, series, x_name="lr", y_name="accuracy", series_name="method")
    return line_chart(
        name, data, x="lr", y="accuracy", colour="method", points=True,
        log_x=True, x_label="Adam learning rate (log scale)",
        y_label="test accuracy after one epoch", colour_label="",
        y_limits=(0.0, 1.0), hlines=[(0.1, "chance")],
        width=cm(14), height=cm(8),
        alt="Test accuracy against Adam learning rate for each learning rule, "
            "five rates spanning two orders of magnitude.",
    )


def tuned_vs_default(rows: dict[str, dict], name: str = "tuned_vs_default") -> Figure:
    """Each arm at the shared default rate, and at its own best rate."""
    tuned_arms = [
        ("bp", "tuned", "backprop 32x32 res"),
        ("pc", "tuned", "PC 32x32 res"),
        ("pcalm", "tuned", "PC-ALM 32x32 res"),
        ("ff", "tuned", "Forward-Forward 2x66"),
        ("bp_plain", "tuned", "backprop 2x66"),
        ("ff", "tuned_deep", "Forward-Forward 31x32"),
    ]
    labels, accs, which = [], [], []
    for (exp, method, tag, label), (_, ttag, _) in zip(ARMS, tuned_arms):
        default = cells(rows, exp, method=method, tag=tag)
        tuned = cells(rows, "exp9", method=method, tag=ttag)
        if not default or not tuned:
            continue
        labels += [label, label]
        accs += [mean(c["test_acc"] for c in default), mean(c["test_acc"] for c in tuned)]
        which += ["shared default rate", "its own best rate"]
    return bar_chart(
        name, {"arm": labels, "accuracy": accs, "rate": which},
        x="arm", y="accuracy", fill="rate", position="dodge", horizontal=True,
        x_order=[label for _, _, _, label in ARMS][::-1],
        x_label="", y_label="test accuracy after one epoch", fill_label="",
        y_limits=(0.0, 1.0), hlines=[(0.1, "chance")],
        width=cm(14), height=cm(8),
        alt="Bar chart comparing each arm's one-epoch accuracy at the shared "
            "default learning rate against its accuracy at the best rate found "
            "for it by a five-point sweep.",
    )


def ff_readouts(rows: dict[str, dict], name: str = "ff_readouts") -> Figure:
    """Forward-Forward's two ways of reading an answer out, shallow and deep."""
    labels, accs, readout = [], [], []
    for exp, tag, label in (("exp2", "", "Forward-Forward 2x66"),
                            ("exp2", "deep", "Forward-Forward 31x32")):
        got = cells(rows, exp, method="ff", tag=tag)
        if not got:
            continue
        labels += [label, label]
        accs += [mean(c["test_acc"] for c in got),
                 mean(c["test_acc_probe"] for c in got)]
        readout += ["goodness sweep (10 passes)", "linear probe on the features"]
    return bar_chart(
        name, {"network": labels, "accuracy": accs, "readout": readout},
        x="network", y="accuracy", fill="readout", position="dodge", horizontal=True,
        x_order=["Forward-Forward 31x32", "Forward-Forward 2x66"],
        x_label="", y_label="test accuracy after one epoch", fill_label="",
        y_limits=(0.0, 1.0), hlines=[(0.1, "chance")],
        width=cm(13), height=cm(6.5),
        alt="Bar chart of Forward-Forward accuracy under its own goodness "
            "readout and under a linear probe on the same features, for the "
            "two-layer and 31-layer networks.",
    )


def build_figures(rows: dict[str, dict]) -> list[Figure]:
    """Every figure the paper needs, guarded on what results exist so far."""
    figures: list[Figure] = []
    if cells(rows, "exp1") and cells(rows, "exp2"):
        figures.append(accuracy_vs_tweet(rows))
        figures.append(cost_by_method(rows, "train_s", "train_time_by_method",
                                      "training seconds (one epoch)"))
        figures.append(cost_by_method(rows, "infer_s", "infer_time_by_method",
                                      "seconds to classify 10,000 images"))
    if cells(rows, "exp2"):
        figures.append(ff_readouts(rows))
    if cells(rows, "exp3"):
        figures.append(learning_rate_grid(rows))
    if cells(rows, "exp9"):
        figures.append(tuned_vs_default(rows))
    if cells(rows, "exp4"):
        figures.append(accuracy_vs_walltime(rows))
    if cells(rows, "exp5"):
        figures.append(accuracy_vs_depth(rows))
        figures.append(grad_cosine_vs_depth(rows))
    return figures
