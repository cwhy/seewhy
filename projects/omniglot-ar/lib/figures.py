"""
Report figures for omniglot-ar, built on `shared_lib.typst_plot`.

Each function takes a results row and returns a `Figure` — data plus a
declarative gribouille spec — rather than drawing anything. Nothing here writes
to disk; `scripts/gen_report.py` collects the figures and calls
`write_figures()` once, so regenerating is idempotent and only the figures that
actually moved get rewritten.

The one exception is `episode_grid()`, which rasterises real drawings through
matplotlib: a picture of the input is not a chart, and there is no
grammar-of-graphics spec for "show me the pixels".
"""

import json
from pathlib import Path

from shared_lib.typst_plot import (
    Figure, bar_chart, cm, line_chart, long_form,
)

RESULTS = Path(__file__).parent.parent / "results.jsonl"


def run_order(rows) -> list[str]:
    """Experiment names in numeric order — plain sorted() puts exp10 after exp1."""
    def key(name: str):
        digits = "".join(c for c in name if c.isdigit())
        return (int(digits) if digits else 0, name)
    return sorted(rows, key=key)


def load_results(path: Path | None = None) -> dict[str, dict]:
    """Read `results.jsonl`, keyed by experiment, keeping the FIRST of any
    duplicates — concurrent runners and post-crash re-runs both duplicate rows
    (see workflow.md), and the first write is the one that finished cleanly.
    """
    rows: dict[str, dict] = {}
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
        rows.setdefault(r.get("experiment", "?"), r)
    return rows


def learning_curves(row: dict, name: str = "learning_curves") -> Figure:
    """Accuracy over training, with the chance and 1-NN floors marked.

    The three series answer three different questions: `train` is whether it is
    fitting at all, `seen` is how much of that is memorisation, and `unseen` is
    the actual claim.
    """
    h = row["history"]
    data = long_form(
        h["step"],
        {
            "train episodes": h["train_acc"],
            "seen characters": h["acc_bg"],
            "unseen characters": h["acc_ev"],
        },
        x_name="step", y_name="accuracy", series_name="episodes",
    )
    return line_chart(
        name, data,
        x="step", y="accuracy", colour="episodes", points=True,
        title=f"{row['spec_n_way']}-way {row['spec_k_shot']}-shot accuracy",
        subtitle="trained on background characters only",
        x_label="training step", y_label="N-way accuracy",
        colour_label="episodes drawn from",
        y_limits=(0.0, 1.0),
        hlines=[(row["chance"], "chance"), (row["nn_ev"], "pixel 1-NN")],
        width=cm(13), height=cm(7.5),
        alt="Accuracy against training step for train, seen and unseen episodes, "
            "with dashed chance and nearest-neighbour reference lines.",
    )


def loss_curve(row: dict, name: str = "loss_curve") -> Figure:
    h = row["history"]
    return line_chart(
        name, {"step": h["step"], "loss": h["loss"]},
        x="step", y="loss",
        title="training loss", x_label="training step",
        y_label="cross-entropy on query labels (nats)",
        width=cm(13), height=cm(5.5),
        alt="Training cross-entropy falling with step.",
    )


def _run_label(exp: str, r: dict) -> str:
    """A short tag naming what makes this run different, not just its number."""
    bits = [f"{r['spec_n_way']}-way"]
    if r.get("spec_label_field"):
        bits.append("+label field")
    if r.get("spec_ink_pool"):
        bits.append("+ink pool")
    if r.get("spec_n_bins") not in (None, 8):
        bits.append(f"{r['spec_n_bins']} bins")
    if r["spec_n_ctx"] != 196:
        bits.append(f"{r['spec_n_ctx']}px")
    return f"{exp} ({' '.join(bits)})"


def excess_over_chance(rows: dict[str, dict], name: str = "excess") -> Figure:
    """Every run on one axis as accuracy *minus its own chance*.

    Runs differ in `n_way`, so raw accuracies are not comparable across them —
    0.53 at 2-way is worse than 0.25 at 5-way. Subtracting each run's own chance
    puts them on a common scale where zero means "learned nothing" regardless of
    episode shape.
    """
    # Built row by row rather than through long_form(): runs no longer share a
    # step axis (exp8/exp9 go to 25 000 where exp1-exp7 stop at 12 000), so
    # there is no single x to broadcast against.
    data: dict[str, list] = {"step": [], "excess": [], "run": []}
    for exp in run_order(rows):
        r = rows[exp]
        label = _run_label(exp, r)
        for s, a in zip(r["history"]["step"], r["history"]["acc_ev"]):
            data["step"].append(s)
            data["excess"].append(a - r["chance"])
            data["run"].append(label)
    return line_chart(
        name, data,
        x="step", y="excess", colour="run",
        title="accuracy above chance, all runs",
        subtitle="unseen (evaluation-split) characters; zero means nothing learned",
        x_label="training step", y_label="accuracy − chance",
        colour_label="", y_limits=(-0.12, 0.30),
        hlines=[(0.0, "chance")],
        width=cm(14), height=cm(7.5),
        alt="Accuracy above chance against training step, one line per run, all "
            "hovering around zero.",
    )


def floor_comparison(rows: dict[str, dict], name: str = "floor_comparison") -> Figure:
    """Every run against its own floors, side by side.

    Chance differs between runs (0.200 for 5-way, 0.500 for 2-way), so the bars
    are grouped per run rather than pooled — the question is never "which run
    scored higher" but "did each one clear the floor it had".
    """
    label, kind, acc = [], [], []
    for exp in run_order(rows):
        r = rows[exp]
        for k, v in (("chance", r["chance"]), ("pixel 1-NN", r["nn_ev"]),
                     ("model", r["acc_ev"])):
            label.append(f"{exp} · {k}")
            kind.append(k)
            acc.append(v)
    return bar_chart(
        name, {"condition": label, "accuracy": acc, "kind": kind},
        x="condition", y="accuracy", fill="kind", x_order=label[::-1],
        horizontal=True,
        title="each run against its own floors",
        subtitle="unseen (evaluation-split) characters",
        x_label="", y_label="accuracy", fill_label="",
        y_limits=(0.0, 1.0),
        width=cm(13), height=cm(15),
        alt="Grouped bars comparing chance, nearest neighbour and the model "
            "for each run.",
    )


def episode_grid(
    report_dir: Path,
    X,
    cls_idx,
    spec,
    seed: int = 0,
    name: str = "episode",
) -> str:
    """Rasterise one episode's drawings to `assets/<name>.png`.

    Shows both the full 28×28 drawings and what the model actually receives —
    the `n_ctx` observed pixels of the shared pool, with everything else blank.
    Returns the root-relative path for `#image(...)`.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    rng = np.random.default_rng(seed)
    classes = rng.choice(len(cls_idx), spec.n_way, replace=False)
    pool = rng.permutation(spec.img_size ** 2)[: spec.n_ctx]
    mask = np.zeros(spec.img_size ** 2, bool)
    mask[pool] = True

    per = spec.k_shot + spec.n_query
    fig, axes = plt.subplots(
        2 * per, spec.n_way,
        figsize=(1.05 * spec.n_way, 1.05 * 2 * per),
        squeeze=False,
    )
    for c, cls in enumerate(classes):
        picked = rng.choice(cls_idx[int(cls)], per, replace=False)
        for j, row in enumerate(picked):
            img = np.asarray(X[int(row)], float).reshape(spec.img_size, spec.img_size)
            axes[2 * j][c].imshow(img, cmap="gray_r", vmin=0, vmax=255)
            seen = np.where(mask, np.asarray(X[int(row)], float), np.nan)
            axes[2 * j + 1][c].imshow(
                seen.reshape(spec.img_size, spec.img_size),
                cmap="gray_r", vmin=0, vmax=255,
            )
            if c == 0:
                role = "support" if j < spec.k_shot else "query"
                axes[2 * j][c].set_ylabel(role, fontsize=6)
                axes[2 * j + 1][c].set_ylabel(f"{role}\nobserved", fontsize=6)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        f"one {spec.n_way}-way {spec.k_shot}-shot episode "
        f"({spec.n_ctx}/{spec.img_size ** 2} pixels observed)",
        fontsize=8,
    )
    fig.tight_layout()

    assets = Path(report_dir) / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    fig.savefig(assets / f"{name}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return f"/assets/{name}.png"
