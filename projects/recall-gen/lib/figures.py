"""
Report figures for Recall-Gen, built on `shared_lib.typst_plot`.

Each function takes result rows and returns a `Figure` — data plus a declarative
gribouille spec — rather than drawing anything. `scripts/gen_report.py` collects
them and calls `write_figures()` once, so regenerating is idempotent.

    write_figures(PROJECT / "paper", [fig])   # into the paper tree
    url = save_figure(fig)                    # standalone image for a .md report

Genuinely-pixel figures — the completion grids — have no grammar-of-graphics
spec and stay in `lib/viz.py` on matplotlib.

Every figure here plots *normalised* MSE, so 1.0 (drawn as a reference line) is
"no better than predicting the average training image" and the baselines are
directly comparable across conditions.
"""

from shared_lib.typst_plot import Figure, bar_chart, cm, line_chart, long_form

COND_LABEL = {
    "A_seen_present":  "A  seen ctx, target present",
    "B_novel_present": "B  novel ctx, target present",
    "C_seen_absent":   "C  seen ctx, target absent",
    "D_novel_absent":  "D  novel ctx, target absent",
    "E_same_present":  "E  novel img seen class, present",
    "F_same_absent":   "F  novel img seen class, absent",
}


def divergence(row: dict, name: str = "divergence") -> Figure:
    """The headline curve: recall improves while completion decays.

    One run, four conditions, normalised MSE against training step. The two
    target-present conditions fall towards zero; the two target-absent
    conditions climb towards 1.0. That opposition is the finding, and it only
    reads if both are on the same axes with the 1.0 line drawn.
    """
    h = row["history"]
    conds = list(h["nmse"].keys())
    data = long_form(
        h["step"], {COND_LABEL.get(c, c): h["nmse"][c] for c in conds},
        x_name="step", y_name="nmse", series_name="condition",
    )
    return line_chart(
        name, data,
        x="step", y="nmse", colour="condition", points=False,
        x_label="training step", y_label="normalised MSE",
        colour_label="condition",
        hlines=[(1.0, "predict the mean image")],
        width=cm(14), height=cm(8),
        alt="Normalised MSE against training step for four evaluation "
            "conditions. The two conditions whose target is present in the "
            "context fall towards zero; the two whose target is absent rise "
            "towards the mean-image reference line at 1.0.",
    )


def context_size(rows: list[dict], baselines: dict, name: str = "context_size") -> Figure:
    """Completion quality against context size, model against look-up ceiling.

    `rows` are the recall-trained runs, one per M; `baselines` maps M to the
    soft-look-up normalised MSE for condition D. The question the figure exists
    to answer is whether the model tracks a ceiling that is itself moving.
    """
    Ms = sorted({r["M"] for r in rows})
    by_M = {r["M"]: r for r in rows}
    series = {
        "recall-trained model": [by_M[m]["final"]["D_novel_absent"]["nmse"] for m in Ms],
        "best soft look-up from context": [baselines[m] for m in Ms],
    }
    data = long_form(Ms, series, x_name="M", y_name="nmse", series_name="what")
    return line_chart(
        name, data,
        x="M", y="nmse", colour="what", points=True,
        x_label="context images per episode (M)", y_label="normalised MSE, condition D",
        colour_label="",
        hlines=[(1.0, "predict the mean image")],
        width=cm(13), height=cm(7.5),
        alt="Normalised MSE on the novel-context, absent-target condition "
            "against the number of context images, for the recall-trained "
            "model and for the best soft look-up from the context.",
    )


def condition_bars(rows: dict[str, dict], name: str = "condition_bars") -> Figure:
    """Final normalised MSE per condition for each training mode."""
    bars_run, bars_cond, bars_val = [], [], []
    for label, r in rows.items():
        for c, v in r["final"].items():
            bars_run.append(label)
            bars_cond.append(COND_LABEL.get(c, c).split()[0])
            bars_val.append(v["nmse"])
    return bar_chart(
        name,
        {"condition": bars_cond, "nmse": bars_val, "trained on": bars_run},
        x="condition", y="nmse", fill="trained on",
        x_order=["A", "B", "C", "D"],
        x_label="condition", y_label="final normalised MSE",
        fill_label="trained on",
        hlines=[(1.0, "predict the mean image")],
        width=cm(13), height=cm(7),
        alt="Final normalised MSE for each evaluation condition, grouped by "
            "what the model was trained on.",
    )


def _recall_runs(rows: dict) -> list[dict]:
    """Recall-trained runs at the default state size, one per context size."""
    return sorted((r for r in rows.values()
                   if r.get("train_mode") == "recall" and r.get("seed") == 0
                   and r.get("cfg", {}).get("dk") == 64
                   and r.get("train_digits") is None),
                  key=lambda r: r["M"])


def _state_runs(rows: dict) -> list[dict]:
    """Recall-trained runs at M=16, one per state size."""
    rs = [r for r in rows.values()
          if r.get("train_mode") == "recall" and r.get("M") == 16
          and r.get("seed") == 0 and r.get("train_digits") is None]
    return sorted(rs, key=lambda r: -r["state_floats"])


def state_size(rows: list[dict], name: str = "state_size") -> Figure:
    """The control for the M-sweep: shrink the memory, hold the context fixed.

    Retrieval quality (condition A) and completion quality (condition D) against
    the size of the recurrent state, at M=16 throughout. If the two move in
    opposite directions here — where the context never changes — then what the
    M-sweep measured was capacity, not information.
    """
    xs = [r["state_floats"] for r in rows]
    series = {
        "A  retrieval (target present)": [r["final"]["A_seen_present"]["nmse"] for r in rows],
        "D  completion (target absent)": [r["final"]["D_novel_absent"]["nmse"] for r in rows],
    }
    data = long_form(xs, series, x_name="state", y_name="nmse", series_name="condition")
    return line_chart(
        name, data,
        x="state", y="nmse", colour="condition", points=True,
        x_label="numbers in the recurrent state", y_label="normalised MSE",
        colour_label="", hlines=[(1.0, "predict the mean image")],
        width=cm(13), height=cm(7.5),
        alt="Normalised MSE on the retrieval and completion conditions against "
            "the size of the recurrent state, with the context size held at 16.",
    )


def build_figures(rows: dict[str, dict]) -> list[Figure]:
    """Every figure the paper uses, from the results rows."""
    figs = []
    base = {k: v for k, v in rows.items() if k.startswith("baselines")}

    if "exp1" in rows:
        figs.append(divergence(rows["exp1"]))

    sweep = _recall_runs(rows)
    knn = {}
    for r in sweep:
        b = base.get(f"baselines_M{r['M']}_r{r['mask_rows']}")
        if b:
            knn[r["M"]] = b["baselines"]["D_novel_absent"]["n_knn"]
    if len(sweep) > 1 and all(r["M"] in knn for r in sweep):
        figs.append(context_size(sweep, knn))

    st = _state_runs(rows)
    if len(st) > 1:
        figs.append(state_size(st))

    modes = {lbl: rows[e] for lbl, e in
             (("recall only", "exp1"), ("completion only", "exp2"), ("mixed", "exp3"))
             if e in rows}
    if len(modes) > 1:
        figs.append(condition_bars(modes))

    return figs
