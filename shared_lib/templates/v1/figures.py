"""
Report figures for {PROJECT_NAME}, built on `shared_lib.typst_plot`.

Each function takes result rows and returns a `Figure` — data plus a
declarative gribouille spec — rather than drawing anything. Nothing here writes
to disk; `scripts/gen_report.py` collects them and calls `write_figures()` once,
so regenerating is idempotent and only figures that actually moved get written.

Keeping figures as objects rather than drawing code is what lets one definition
serve both tiers:

    write_figures(PROJECT / "paper", [fig])   # into the paper tree
    url = save_figure(fig)                     # standalone image for a .md report

Genuinely-pixel figures — sample grids, reconstructions, attention maps — have
no grammar-of-graphics spec and stay in `lib/viz.py` on matplotlib.
"""

from shared_lib.results import run_order
from shared_lib.typst_plot import Figure, bar_chart, cm, line_chart, long_form


def learning_curves(row: dict, name: str = "learning_curves") -> Figure:
    """Accuracy over training, with the chance floor marked.

    Delete or rewrite this — it is here to show the shape, not because every
    project wants it. Note what it does with `hlines`: a metric without its
    baseline drawn beside it is not readable, and the paper checklist requires
    the baseline anyway.
    """
    h = row["history"]
    data = long_form(
        h["step"],
        {"train": h["train_acc"], "eval": h["eval_acc"]},
        x_name="step", y_name="accuracy", series_name="split",
    )
    return line_chart(
        name, data,
        x="step", y="accuracy", colour="split", points=True,
        x_label="training step", y_label="accuracy",
        colour_label="split",
        y_limits=(0.0, 1.0),
        hlines=[(row["chance"], "chance")],
        width=cm(13), height=cm(7.5),
        alt="Accuracy against training step for the train and eval splits, "
            "with a dashed chance reference line.",
    )


def final_comparison(rows: dict[str, dict], name: str = "final_comparison") -> Figure:
    """Final metric per run, in numeric run order."""
    names = run_order(rows)
    return bar_chart(
        name,
        {"run": names, "accuracy": [rows[n]["eval_acc"] for n in names]},
        x="run", y="accuracy",
        x_label="run", y_label="final accuracy",
        width=cm(13), height=cm(6.5),
        alt="Final evaluation accuracy for each run.",
    )


def build_figures(rows: dict[str, dict]) -> list[Figure]:
    """Every figure the report tree needs. Called by scripts/gen_report.py.

    Guard on what exists: this runs against a partially-filled results.jsonl
    for most of a project's life, and a KeyError here blocks publishing a paper
    that only wanted the figures it already has.
    """
    figures: list[Figure] = []
    for run in run_order(rows):
        if "history" in rows[run]:
            figures.append(learning_curves(rows[run], name=f"learning_curves_{run}"))
    if len(rows) > 1:
        figures.append(final_comparison(rows))
    return figures
