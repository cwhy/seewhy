"""The synthetic-prior figures, drawn with the project's Typst/gribouille stack.

The matplotlib versions in `lib/synthfig.py` are what these replace. Two things
recur and are worth stating once here rather than in every builder.

**Merging an accuracy with an error.** Several of these figures used to be two
panels side by side over the same x — one for finding, one for predicting —
because the two quantities have different units and opposite polarity. Putting
them on one axis needs a common scale, and the only honest one is each
quantity's advantage over the trivial answer:

    finding      (accuracy - 1/16) / (1 - 1/16)    0 = chance, 1 = perfect
    predicting   1 - error                         0 = average real item, 1 = perfect

Both then read 0 when the network has bought nothing and 1 when it is perfect,
so one panel carries both and the x is not drawn twice.

**Sizing.** Typst pages here are `width: auto`, so a figure's intrinsic size is
whatever `width`/`height` say. `shared_lib.report` scales images to the column,
but a figure much narrower than the column still arrives with its text sized for
17cm and then gets enlarged, so set the width close to the width it will be read
at rather than relying on the browser.
"""

import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT.parents[1]))

from shared_lib.typst_plot import (GRIBOUILLE, PALETTE, bar_chart, cm, line_chart,
                                   long_form, scatter_chart, typ)
from shared_lib.typst_report import compose, panels, save_figure
from shared_lib.media import array_to_png

CHANCE = 1.0 / 16
W, H = cm(17), cm(9.5)


def advantage_id(v):
    """Identification, as a fraction of the way from chance to perfect."""
    return (v - CHANCE) / (1.0 - CHANCE)


def advantage_err(v):
    """Normalised error, as a fraction of the way from the trivial answer to perfect."""
    return 1.0 - v


def _pub(fig, name, fmt="svg"):
    return save_figure(fig, name=name, fmt=fmt)


def tile_grid(name, spec, headline, sub, legend, tile="1.5cm", fmt="svg"):
    """The merged completion grid, typeset by Typst instead of drawn by matplotlib.

    Image domains only: a tile is a greyscale array and becomes a PNG. A board
    domain would need its squares and glyphs drawn, which matplotlib already
    does in `lib/domains.draw`, so chess keeps the matplotlib grid.
    """
    from . import domains
    dom = domains.get(spec["domain"])
    if dom.kind != "image":
        raise ValueError(f"tile_grid draws image tiles; {spec['domain']!r} is a board")
    side = dom.shape[0]

    assets, cells = {}, []
    cells.append(f"#text(size: 12pt, weight: \"bold\")[{headline}]")
    for line, colour in legend:
        cells.append(f"#text(size: 8.4pt, fill: rgb({typ(colour)}))[{line}]")
    for line in sub.split("\n"):
        cells.append(f"#text(size: 8.2pt, fill: luma(90))[{line}]")

    blocks = []
    for b, band in enumerate(spec["bands"]):
        rows_src = []
        for r, row in enumerate(band["rows"]):
            lab = row["label"].replace("\n", "\\ ")
            agg = (f"\n#v(1pt)\n#text(size: 7.2pt, fill: rgb(\"#a03030\"))[{row['agg']}]"
                   if row["agg"] else "")
            rows_src.append(
                f"align(right + horizon)[#text(size: 8.2pt, fill: rgb({typ(row['colour'])}))[{lab}]{agg}]")
            for c, t in enumerate(row["tiles"]):
                key = f"t{b}_{r}_{c}.png"
                flat = np.asarray(t["img"], np.float32).reshape(-1)[:side * side]
                assets[key] = array_to_png(flat.reshape(side, side), upscale=8)
                num = (f"#text(size: 7pt)[{t['num']}]" if t["num"] else "")
                rows_src.append(
                    f"align(center)[#image({typ(key)}, width: {tile})"
                    f"{'#v(-3pt)' if num else ''}{num}]")
        ncol = 1 + len(band["rows"][0]["tiles"])
        # Built as markup, so each block already starts with its own `#`.
        blocks.append(
            f"#text(size: 10.5pt, weight: \"bold\")[{band['label'].replace(chr(10), ' — ')}]"
            f"\n#v(4pt)\n#grid(columns: {ncol}, column-gutter: 4pt, row-gutter: 6pt,\n"
            + ",\n".join(rows_src) + ")")

    body = ("#align(center)[" + "\n#v(3pt)\n".join(cells) + "]\n#v(10pt)\n"
            + "\n#v(14pt)\n".join(blocks))
    return compose(name, body, assets=assets, fmt=fmt, margin="10pt")


def arms_chart(name, arms, ident, comp, refs, pretty):
    """Three training objectives, finding and predicting, on one axis.

    `arms` is [(label, exp, colour)]; `ident` is {(exp, band): accuracy} over
    bands "prior" and "real"; `comp` is {exp: error against the average real
    item}.
    """
    labels = [lab for lab, _, _ in arms]
    data = long_form(
        labels,
        {"finding — fresh worlds from the prior":
            [advantage_id(ident[(e, "prior")]) for _, e, _ in arms],
         f"finding — real {pretty}":
            [advantage_id(ident[(e, "real")]) for _, e, _ in arms],
         f"predicting — real {pretty}":
            [advantage_err(comp[e]) for _, e, _ in arms]},
        x_name="arm", y_name="advantage", series_name="ability")
    return _pub(bar_chart(
        name, data, x="arm", y="advantage", fill="ability", position="dodge",
        x_order=labels,
        title=f"What each training objective buys on {pretty}",
        subtitle=("Advantage over the trivial answer: 0 is chance for finding and "
                  "the average real item for predicting; 1 is perfect."),
        x_label="", y_label="advantage over the trivial answer",
        hlines=[(0.0, "no better than doing nothing")],
        caption=(f"Soft look-up, which needs no training, reaches "
                 f"{advantage_err(refs['soft']):+.2f} on predicting; ridge, which never "
                 f"sees the context, {advantage_err(refs['ridge']):+.2f}."),
        width=W, height=H), name)


def capacity_two_panel(name, runs, refs, pretty="MNIST"):
    """The capacity sweep as TWO panels, finding and predicting, side by side.

    Restores the split layout. Merging them needed a shared "advantage" scale,
    which put both on one axis at the cost of hiding the raw numbers behind a
    transform; kept separate, each panel can use the units the report quotes.
    """
    compute = [st * pr / 1e12 for _, st, pr, _, _ in runs]
    find = long_form(
        compute,
        {"worlds seen in training": [b["A"] for *_, b, _ in runs],
         "worlds never seen": [b["E"] for *_, b, _ in runs],
         f"real {pretty}": [b["B"] for *_, b, _ in runs]},
        x_name="compute", y_name="accuracy", series_name="pool")
    pred = long_form(compute, {f"real {pretty}": [e for *_, e in runs]},
                     x_name="compute", y_name="error", series_name="pool")
    left = line_chart(
        name + "-find", find, x="compute", y="accuracy", colour="pool",
        points=True, log_x=True, title="Finding: climbs with capacity",
        x_label="compute (steps x params, 1e12)", y_label="identification accuracy",
        y_limits=(0.0, 1.0), hlines=[(CHANCE, "chance (1 of 16)")],
        width=cm(9), height=cm(8))
    right = line_chart(
        name + "-pred", pred, x="compute", y="error", colour="pool",
        points=True, log_x=True, title="Predicting: does not",
        x_label="compute (steps x params, 1e12)",
        y_label="error / error of the average real item",
        # Bare hlines, named in the caption instead. On MNIST the soft look-up
        # lands at 0.993 and the do-nothing line at 1.000, so on-plot labels sit
        # on top of each other.
        hlines=[1.0, refs["soft"], refs["ridge"]],
        caption=(f"Dashed lines, low to high: soft look-up {refs['soft']:.2f}, "
                 f"the average real item 1.00, ridge {refs['ridge']:.2f}."),
        palette=("#c1121f",), width=cm(9), height=cm(8))
    return panels(name, (left, right))


def capacity_chart(name, runs, pretty="MNIST"):
    """Finding and predicting against training compute, one panel.

    `runs` is [(label, steps, params, {band: accuracy}, error)].
    """
    compute = [st * pr / 1e12 for _, st, pr, _, _ in runs]
    data = long_form(
        compute,
        {"finding — worlds seen in training": [advantage_id(b["A"]) for *_, b, _ in runs],
         "finding — worlds never seen": [advantage_id(b["E"]) for *_, b, _ in runs],
         f"finding — real {pretty}": [advantage_id(b["B"]) for *_, b, _ in runs],
         f"predicting — real {pretty}": [advantage_err(e) for *_, e in runs]},
        x_name="compute", y_name="advantage", series_name="ability")
    return _pub(line_chart(
        name, data, x="compute", y="advantage", colour="ability", points=True,
        log_x=True,
        title="Finding and predicting, against training compute",
        subtitle=("Three d=256 runs at 12k, 48k and 192k steps, then the d=512 model "
                  "at the same compute as the 192k run."),
        x_label="training compute (steps x parameters, units of 1e12)",
        y_label="advantage over the trivial answer",
        hlines=[(0.0, "no better than doing nothing")],
        caption=("Finding is (accuracy - 1/16)/(1 - 1/16); predicting is 1 - error, "
                 "error taken against the average real item."),
        width=W, height=H), name)


def context_chart(name, per_ctx, margins, pretty="MNIST"):
    """One network, two kinds of context. `per_ctx` is {ctx: (accuracy, error)}."""
    names = {"iid": "sixteen unrelated items",
             "knn": "the query's nearest neighbours"}
    order = [names[c] for c in ("iid", "knn")]
    data = long_form(
        order,
        {"finding": [advantage_id(per_ctx[c][0]) for c in ("iid", "knn")],
         "predicting": [advantage_err(per_ctx[c][1]) for c in ("iid", "knn")]},
        x_name="context", y_name="advantage", series_name="ability")
    return _pub(bar_chart(
        name, data, x="context", y="advantage", fill="ability", position="dodge",
        x_order=order,
        title="A closer context is harder to name and easier to reconstruct",
        subtitle=(f"Median distance from a target to its nearest rival falls from "
                  f"{margins['iid']:.2f} to {margins['knn']:.2f} between the two."),
        x_label="", y_label="advantage over the trivial answer",
        hlines=[(0.0, "no better than doing nothing")],
        caption=("A neighbour context is assembled from the query's closest matches, "
                 "so it is by construction a low-margin context."),
        width=cm(15), height=cm(8.5)), name)


def resemblance_chart(name, points, pretty):
    """Simplex advantage against how binary a dataset is. Three points."""
    data = {"binary": [x for _, x, _ in points],
            "gain": [y for _, _, y in points],
            "dataset": [lab for lab, _, _ in points]}
    return _pub(scatter_chart(
        name, data, x="binary", y="gain", colour="dataset",
        title="The prior transfers as far as it resembles the target",
        subtitle=(f"{pretty} is the one this report is about."),
        x_label="share of the dataset's coordinates within 0.1 of 0 or 1",
        y_label="gain in identification from the prior's simplex mode",
        x_limits=(0.5, 1.05),
        caption=("A simplex world is exactly 1.00 on this axis and a continuous world "
                 "0.20. Three datasets is an ordering, not a fit."),
        width=cm(15), height=cm(8.5)), name)


def scaling_chart(name, runs):
    """Identification on the prior's own worlds, per run. `runs` is
    [(label, steps, d_model, id_train, id_held)]."""
    labels = [lab for lab, *_ in runs]
    data = long_form(
        labels,
        {"worlds seen in training": [tr for *_, tr, _ in runs],
         "worlds never seen": [hd for *_, hd in runs]},
        x_name="run", y_name="accuracy", series_name="pool")
    return _pub(bar_chart(
        name, data, x="run", y="accuracy", fill="pool", position="dodge",
        x_order=labels,
        title="Identification against training budget and width",
        subtitle=("Endpoints of completed cosine schedules. Every run flattens over its "
                  "own last decile whatever it converged to, so only endpoints compare."),
        x_label="", y_label="identification accuracy",
        y_limits=(0.0, 1.0),
        hlines=[(CHANCE, "chance (1 of 16)")],
        width=W, height=cm(9)), name)
