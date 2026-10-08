"""
Report figures for Recall-GAN, built on `shared_lib.typst_plot`.

Each function takes the result rows (keyed by `run`) and returns a `Figure` —
data plus a declarative gribouille spec. `scripts/gen_report.py` writes them
into the paper tree. Pixel figures (episodes, samples) are PNGs written by
`scripts/eval_report.py`; the method diagram is hand-drawn in
`paper/diagrams/game.typ`.

The method's runs: exp15 tag "adam" (context-aware generator) and exp18/19
tag "adam" (blind generator), seeds 50-61. Evaluation rows: experiment
"eval_report".
"""

import numpy as np

from shared_lib.typst_plot import Figure, bar_chart, cm, line_chart, long_form

CTX_C, BLIND_C, BASE_C = "#e8590c", "#2a78d6", "#8c8a85"


def method_runs(rows: dict[str, dict]) -> dict[str, list[dict]]:
    """Training rows of the final recipe, by generator."""
    out = {"context-aware": [], "blind": []}
    for r in rows.values():
        if r.get("tag") != "adam":
            continue
        if r["experiment"] == "exp15" and r["arm"] == "ctx":
            out["context-aware"].append(r)
        elif r["experiment"] in ("exp18", "exp19") and r["arm"] == "blind":
            out["blind"].append(r)
    return out


def eval_rows(rows: dict[str, dict]) -> tuple[dict[str, list[dict]], dict]:
    ev = {"context-aware": [], "blind": []}
    base = None
    for r in rows.values():
        if r["experiment"] != "eval_report":
            continue
        if r["run"] == "eval_report_baselines":
            base = r
        else:
            ev[r["generator"]].append(r)
    return ev, base


def _median_curve(runs: list[dict], key: str) -> tuple[list, list]:
    steps = runs[0]["history"]["step"]
    return steps, [float(np.median([r["history"][key][i] for r in runs])) for i in range(len(steps))]


def class_probe_curve(rows) -> Figure:
    runs = method_runs(rows)
    steps, ctx = _median_curve(runs["context-aware"], "class_acc")
    _, blind = _median_curve(runs["blind"], "class_acc")
    data = long_form(steps, {"context-aware generator": ctx, "blind generator": blind},
                     x_name="step", y_name="accuracy", series_name="generator")
    return line_chart(
        "class_probe_curve", data, x="step", y="accuracy", colour="generator",
        x_label="training step", y_label="digit classifier accuracy",
        colour_label="generator", y_limits=(0.0, 1.0), hlines=[(0.1, "chance")],
        palette=[CTX_C, BLIND_C], width=cm(12.5), height=cm(6.2),
        alt="Median over 12 runs of the discriminator's 10-way digit-classifier accuracy "
            "against training step, for the two generators, with chance at 0.1.")


def swap_curve(rows) -> Figure:
    runs = method_runs(rows)
    steps, own = _median_curve(runs["context-aware"], "auroc_gen")
    _, sw = _median_curve(runs["context-aware"], "auroc_gen_swap")
    _, b_own = _median_curve(runs["blind"], "auroc_gen")
    _, b_sw = _median_curve(runs["blind"], "auroc_gen_swap")
    data = long_form(steps, {"context-aware, own context": own, "context-aware, swapped context": sw,
                             "blind, own context": b_own, "blind, swapped context": b_sw},
                     x_name="step", y_name="AUROC", series_name="setting")
    return line_chart(
        "swap_curve", data, x="step", y="AUROC", colour="setting",
        x_label="training step", y_label="AUROC: context images vs generated",
        colour_label="generator, context used for scoring", y_limits=(0.4, 1.0),
        hlines=[(0.5, "chance")], palette=[CTX_C, "#f5b48a", BLIND_C, "#9cc2ee"],
        width=cm(12.5), height=cm(6.2),
        alt="Median AUROC of context images against generated images, scored with the episode's "
            "own context and with another episode's, over training, for both generators.")


def gen_context_use_curve(rows) -> Figure:
    runs = method_runs(rows)["context-aware"]
    steps = runs[0]["history"]["step"]
    series = {f"seed {r['seed']}": r["history"]["g_ctx_dep"] for r in sorted(runs, key=lambda r: r["seed"])}
    data = long_form(steps, series, x_name="step", y_name="context use", series_name="run")
    return line_chart(
        "gen_context_use", data, x="step", y="context use", colour="run",
        x_label="training step", y_label="generator context use",
        colour_label="run", y_limits=(0.0, 1.1), hlines=[(0.0, "ignores context")],
        palette=["#e8590c"] * len(series), width=cm(12.5), height=cm(5.6),
        alt="For each of the 12 context-aware runs, the generator's context use against training step.")


def classifier_bars(rows) -> Figure:
    ev, base = eval_rows(rows)
    med = lambda g: float(np.median([r["cls_acc"] for r in ev[g]]))
    names = ["pixel nearest neighbour", "pixel RBF density (median bandwidth)",
             "pixel RBF density (best bandwidth)", "discriminator, blind generator",
             "discriminator, context-aware generator"]
    vals = [base["pixnn_cls_acc"], base["rbf_med_cls_acc"], base["rbf_best_cls_acc"],
            med("blind"), med("context-aware")]
    return bar_chart(
        "classifier_bars", {"method": names, "accuracy": vals}, x="method", y="accuracy",
        x_order=names[::-1], horizontal=True, x_label=None, y_label="10-way accuracy (chance 0.1)",
        y_limits=(0.0, 1.0), hlines=[(0.1, "chance")], width=cm(12.5), height=cm(5.2),
        alt="10-way digit accuracy from 16 examples per digit: three pixel-space baselines and the "
            "discriminators of both generators (median of 12 runs).")


def per_digit_bars(rows) -> Figure:
    ev, base = eval_rows(rows)
    ctx = np.median([r["cls_auroc"] for r in ev["context-aware"]], axis=0)
    digits, method, val = [], [], []
    for k in range(10):
        for name, v in (("pixel RBF (best bandwidth)", base["rbf_best_cls_auroc"][k]),
                        ("discriminator, context-aware", float(ctx[k]))):
            digits.append(str(k)); method.append(name); val.append(v)
    return bar_chart(
        "per_digit_bars", {"digit": digits, "method": method, "AUROC": val}, x="digit", y="AUROC",
        fill="method", position="dodge", x_order=[str(k) for k in range(10)],
        x_label="digit in the context", y_label="AUROC: that digit vs the rest",
        fill_label="", y_limits=(0.5, 1.0), palette=[BASE_C, CTX_C], width=cm(12.5), height=cm(5.6),
        alt="Per-digit AUROC of separating that digit from the other nine, pixel RBF baseline against "
            "the context-aware discriminator.")


def density_hist(rows) -> Figure:
    ev, _ = eval_rows(rows)
    runs = ev["context-aware"]
    centres = [round(0.025 + 0.05 * i, 3) for i in range(20)]
    mean = lambda k: [float(v) for v in np.mean([r[k] for r in runs], axis=0)]
    data = long_form(centres, {"the 16 context images": mean("dens_hist_ctx"),
                               "unseen, digit present in context": mean("dens_hist_present"),
                               "unseen, digit absent from context": mean("dens_hist_absent")},
                     x_name="score", y_name="share", series_name="images")
    return line_chart(
        "density_hist", data, x="score", y="share", colour="images", points=True,
        x_label="discriminator's P(in context)", y_label="share of images",
        colour_label="", palette=[CTX_C, "#f5b48a", BLIND_C], width=cm(12.5), height=cm(6.0),
        alt="Distribution of the discriminator's score for context images and for unseen real test "
            "images whose digit is or is not present in the context.")


def build_figures(rows: dict[str, dict]) -> list[Figure]:
    """Guarded on what exists, so a partial results.jsonl still builds."""
    figs: list[Figure] = []
    runs = method_runs(rows)
    if runs["context-aware"] and runs["blind"]:
        figs += [class_probe_curve(rows), swap_curve(rows), gen_context_use_curve(rows)]
    ev, base = eval_rows(rows)
    if base and ev["context-aware"] and ev["blind"]:
        figs += [classifier_bars(rows), per_digit_bars(rows), density_hist(rows)]
    return figs
