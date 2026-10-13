"""Figures for the curve-vs-ball probe.

Two questions, two figures:

  `order_sweep`  -- does bend pay? Distortion against order at fixed budget,
                    with the N=1 (k-means) column as the reference line.
  `fitted_curves` -- *why*? The actual curves drawn over the data, at the best
                    order and at a deliberately too-high one. Aggregate
                    distortion cannot distinguish "curves fit badly" from
                    "curves tangled", and a picture can.

Matplotlib rather than `shared_lib.typst_plot`: this is a probe, and one of the
figures is a genuinely-pixel overlay of geometry on a scatter, which has no
grammar-of-graphics spec.
"""

from __future__ import annotations

import collections
import math
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lib import toy
from lib.curves import CurveClusters

PALETTE = {
    "spiral2d": "#4C72B0", "moons2d": "#DD8452", "square2d": "#55A868",
    "swissroll": "#C44E52", "blobs10d": "#8172B3", "gauss10d": "#937860",
    "checkerboard": "#DA8BC3",
}
LOW_DIM = ("spiral2d", "moons2d", "square2d", "checkerboard", "swissroll")


def from_exp(rows, *prefixes):
    """Rows from specific experiment files, by `experiment` prefix.

    Filtering on incidental fields does not work and has produced two wrong
    figures already: exp4 rows also carry budget=128 and n_passes=1, so a
    predicate built from those silently averages exp4's coarse-grid sweep into
    exp1's order sweep. Provenance is the only unambiguous filter.
    """
    tags = tuple(f"{p}_" for p in prefixes)
    return [r for r in rows if str(r.get("experiment", "")).startswith(tags)]


def _agg(rows, n_passes, budget):
    """{(dataset, order): (mean, stderr)} of held-out distortion.

    Filters on `budget` as well as `n_passes`. Not optional: exp2 sweeps the
    budget at n_passes=1, so a filter on passes alone silently mixes rows from
    five other budgets into this panel and the curve stops meaning anything.
    """
    buckets = collections.defaultdict(list)
    for r in from_exp(rows, "exp1", "exp2"):
        if r.get("n_passes") == n_passes and r.get("budget") == budget:
            buckets[(r["dataset"], r["order"])].append(r["distortion_test"])
    out = {}
    for key, vals in buckets.items():
        se = statistics.stdev(vals) / len(vals) ** 0.5 if len(vals) > 1 else 0.0
        out[key] = (statistics.mean(vals), se)
    return out


def order_sweep(rows, n_passes=1, budget=128):
    stats = _agg(rows, n_passes, budget)
    orders = sorted({o for (_, o) in stats})
    fig, (ax_abs, ax_rel) = plt.subplots(1, 2, figsize=(12.5, 4.8))

    for ds in toy.DATASETS:
        m = np.array([stats[(ds, o)][0] for o in orders])
        se = np.array([stats[(ds, o)][1] for o in orders])
        c = PALETTE[ds]
        ax_abs.errorbar(orders, m, yerr=se, marker="o", ms=4, lw=1.6, color=c, label=ds)
        # ratio to the N=1 k-means column; errors added in quadrature
        rel = m / m[0]
        rel_se = rel * np.sqrt((se / m) ** 2 + (se[0] / m[0]) ** 2)
        ax_rel.errorbar(orders, rel, yerr=rel_se, marker="o", ms=4, lw=1.6, color=c, label=ds)

    for ax in (ax_abs, ax_rel):
        ax.set_xscale("log", base=2)
        ax.set_xticks(orders)
        ax.set_xticklabels([str(o) for o in orders])
        ax.set_xlabel("curve order N   (1 = point/k-means, 2 = straight, ≥3 = bent)")
        ax.grid(alpha=0.25, lw=0.6)
    ax_abs.set_yscale("log")
    ax_abs.set_ylabel("held-out distortion (fraction of variance unexplained)")
    ax_abs.set_title(f"absolute — budget K·N = {budget} held fixed, {n_passes} pass(es)")
    ax_rel.axhline(1.0, color="black", lw=1.2, ls="--")
    ax_rel.set_ylabel("distortion ÷ distortion at N=1")
    ax_rel.set_title("relative to k-means (dashed = no gain)")
    ax_rel.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    return fig


def fitted_curves(orders=(3, 32), n_passes=1, seed=0):
    """Curves drawn over the data. Rows = order, columns = dataset."""
    fig, axes = plt.subplots(len(orders), len(LOW_DIM),
                             figsize=(4.0 * len(LOW_DIM), 3.7 * len(orders)))
    axes = np.atleast_2d(axes)
    for i, order in enumerate(orders):
        for j, ds in enumerate(LOW_DIM):
            ax = axes[i, j]
            train, _ = toy.load(ds, seed=seed)
            k = max(1, 128 // order)
            model = CurveClusters(k, order, train.shape[1], max(64, 16 * order))
            rng = np.random.default_rng(1000 + seed)
            for _ in range(n_passes):
                model.partial_fit(train[rng.permutation(len(train))])
            ax.scatter(train[:, 0], train[:, 1], s=2, c="#BBBBBB", zorder=1)
            for jj in range(model.n_seeded):
                v = model.values[jj]
                ax.plot(v[:, 0], v[:, 1], lw=1.4, zorder=2)
            ax.set_title(f"{ds}  N={order}, K={k}", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    fig.suptitle("Fitted curves over the data (first two dimensions). "
                 "Top: best order. Bottom: order too high for the budget.",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def bend_vs_budget(rows, orders=(2, 3)):
    """Paired per-seed ratio N=orders[1] over N=orders[0], against budget.

    This is the asymptotic question. Every smooth manifold is locally straight,
    so a bigger budget means shorter pieces and less curvature per piece. If
    this ratio climbs towards 1.0 as the budget grows, bend stops paying in the
    limit that matters and the idea does not scale.
    """
    a, b = orders
    by = collections.defaultdict(dict)
    for r in from_exp(rows, "exp1", "exp2"):
        if r.get("n_passes") == 1 and "budget" in r:
            by[(r["dataset"], r["budget"], r["seed"])][r["order"]] = r["distortion_test"]
    budgets = sorted({bg for (_, bg, _) in by})
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    for ds in toy.DATASETS:
        xs, ys, es = [], [], []
        for bg in budgets:
            ratios = [v[b] / v[a] for (d, g, _), v in by.items()
                      if d == ds and g == bg and a in v and b in v]
            if len(ratios) < 2:
                continue
            xs.append(bg)
            ys.append(statistics.mean(ratios))
            es.append(statistics.stdev(ratios) / len(ratios) ** 0.5)
        ax.errorbar(xs, ys, yerr=es, marker="o", ms=5, lw=1.8,
                    color=PALETTE[ds], label=ds)
    ax.axhline(1.0, color="black", ls="--", lw=1.2)
    ax.set_xscale("log", base=2)
    ax.set_xticks(budgets)
    ax.set_xticklabels([str(bg) for bg in budgets])
    ax.set_xlabel("budget K·N")
    ax.set_ylabel(f"distortion at N={b}  ÷  distortion at N={a}   (paired per seed)")
    ax.set_title(f"Does bend keep paying? N={b} vs N={a} against budget\n"
                 "below the dashed line = bend wins; rising towards it = bend stops paying")
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def _fit(dataset, order, budget=128, seed=0, n_passes=1):
    train, test = toy.load(dataset, seed=seed)
    k = max(1, budget // order)
    model = CurveClusters(k, order, train.shape[1], max(64, 16 * order))
    rng = np.random.default_rng(1000 + seed)
    for _ in range(n_passes):
        model.partial_fit(train[rng.permutation(len(train))])
    return model, train, model.distortion(test)


def comparison_grid(orders=(1, 2, 3), budget=128, seed=0):
    """Balls vs sticks vs curves, drawn over the data, with distortion labelled.

    The point of the figure is that the three rows are the *same algorithm* at
    three settings of one dial -- row 1 is provably online k-means -- so the
    visual difference between the rows is exactly what the distortion column of
    the results table is measuring.
    """
    fig, axes = plt.subplots(len(orders), len(LOW_DIM),
                             figsize=(3.5 * len(LOW_DIM), 3.6 * len(orders)))
    axes = np.atleast_2d(axes)
    labels = {1: "N=1  points (k-means)", 2: "N=2  straight sticks", 3: "N=3  bent curves"}
    for i, order in enumerate(orders):
        for j, ds in enumerate(LOW_DIM):
            ax = axes[i, j]
            model, train, dist = _fit(ds, order, budget, seed)
            ax.scatter(train[:, 0], train[:, 1], s=2.5, c="#D5D5D5", zorder=1, linewidths=0)
            for jj in range(model.n_seeded):
                v = model.values[jj]
                if order == 1:                      # a constant curve is one point
                    ax.plot(v[0, 0], v[0, 1], marker="o", ms=4.5, zorder=3)
                else:
                    ax.plot(v[:, 0], v[:, 1], lw=1.5, zorder=2)
            ax.set_title(f"{ds}   K={model.n_seeded}\ndistortion {dist:.4f}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
            if j == 0:
                ax.set_ylabel(labels.get(order, f"N={order}"), fontsize=11)
    fig.suptitle(f"One dial, three settings — budget K·N = {budget}, single pass, "
                 "same algorithm throughout", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    return fig


def rate_distortion(rows, m_points=2000):
    """Distortion against TOTAL description length, both terms charged.

        L = K*N*D * (1/2)log2(M)      model
          + M * log2(addresses)       which curve, where on it

    This is the comparison exp1-3 could not make. Holding K*N fixed holds only
    the model term, which is ~10x smaller than the address term, and exp1-3 let
    the address term float upward in the curve's favour (G was tied to N).

    Two subtleties the earlier version of this figure got wrong:

    * An order-1 curve is constant, so its G grid points are all the SAME
      position. Its address count is K, not K*G. Charging it K*G drew k-means as
      a flat line across a bit range it cannot actually reach.
    * k-means has no free rate knob at all. To use R address bits it must carry
      2^R centroids, so its frontier is traced by growing K -- and every step
      right costs model size. A curve buys rate from G at zero model cost. That
      asymmetry is the whole point, and it is only visible on a total-bits axis.
    """
    bcoef = 0.5 * math.log2(m_points)
    pts = collections.defaultdict(list)
    for r in from_exp(rows, "exp1", "exp2", "exp4"):
        if r.get("n_passes") != 1 or "grid" not in r or "n_curves" not in r:
            continue
        k, n, g, dim = r["n_curves"], r["order"], r["grid"], r["dim"]
        addr = k if n == 1 else k * g
        total = k * n * dim * bcoef + m_points * math.log2(addr)
        pts[(r["dataset"], n == 1, k, n, g)].append((total, r["distortion_test"]))

    grouped = collections.defaultdict(list)
    for (ds, is_km, _, _, _), vals in pts.items():
        grouped[(ds, is_km)].append((vals[0][0], statistics.mean(d for _, d in vals)))

    def frontier(points):
        """Lower-left staircase: cheapest cost achieving each distortion."""
        best, out = float("inf"), []
        for cost, dist in sorted(points):
            if dist < best:
                best = dist
                out.append((cost / 8 / 1024, dist))
        return out

    names = [d for d in toy.DATASETS if (d, True) in grouped]
    ncol = 4
    nrow = -(-len(names) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.8 * nrow), squeeze=False)
    for ax in axes.ravel():
        ax.set_visible(False)
    for i, ds in enumerate(names):
        ax = axes[i // ncol][i % ncol]
        ax.set_visible(True)
        for is_km, colour, lab in ((True, "black", "k-means (N=1), rate from K"),
                                   (False, "#4C72B0", "curves (N≥2), rate from G")):
            f = frontier(grouped.get((ds, is_km), []))
            if f:
                ax.plot([c for c, _ in f], [d for _, d in f], marker="o", ms=4,
                        lw=1.8, color=colour, label=lab)
        ax.set_yscale("log")
        ax.set_title(ds, fontsize=10)
        ax.set_xlabel("total description length (KB)")
        ax.set_ylabel("held-out distortion")
        ax.grid(alpha=0.25, lw=0.6)
        if i == 0:
            ax.legend(fontsize=8)
    fig.suptitle("Rate-distortion frontier with BOTH terms charged — model + addresses\n"
                 "lower-left is better; where the blue frontier drops below black, "
                 "curves are cheaper at that fidelity",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def low_order_zoom(rows, budget=128, orders=(1, 2, 3, 4, 5, 6, 8)):
    """The low-order region on a linear x-axis, with standard errors.

    Exists because a contaminated render of `order_sweep` appeared to show a
    dip at N=5. It did not: exp2 sweeps the budget at n_passes=1, so before
    `_agg` filtered on budget, rows from five other budgets were averaged into
    the same point. The underlying numbers are monotone above N=3. This plots
    the region large enough, and with error bars, that the question does not
    have to be taken on trust.

    Everything interesting happens between N=1 and N=4, which `order_sweep`
    compresses into the left third of a log2 axis running to N=32.
    """
    stats = _agg(rows, 1, budget)
    orders = [o for o in orders if any((ds, o) in stats for ds in toy.DATASETS)]
    fig, (ax_lo, ax_hi) = plt.subplots(1, 2, figsize=(12.6, 4.9))

    for ds in toy.DATASETS:
        got = [(o, *stats[(ds, o)]) for o in orders if (ds, o) in stats]
        if not got:
            continue
        xs = [o for o, _, _ in got]
        m = np.array([v for _, v, _ in got])
        se = np.array([e for _, _, e in got])
        ax = ax_hi if m[0] > 0.05 else ax_lo      # gauss10d is 100x the others
        ax.errorbar(xs, m, yerr=se, marker="o", ms=5, lw=1.7, capsize=3,
                    color=PALETTE[ds], label=ds)

    for ax, title in ((ax_lo, "low-distortion datasets"),
                      (ax_hi, "gauss10d (own scale — distortion ≈ 0.44)")):
        ax.set_xticks(orders)
        ax.set_xlabel("curve order N")
        ax.set_ylabel("held-out distortion")
        ax.grid(alpha=0.25, lw=0.6)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
    ax_lo.set_yscale("log")
    fig.suptitle(f"Low-order region, linear axis, ±1 s.e. over 8 seeds "
                 f"(budget K·N = {budget}, 1 pass)\n"
                 "monotone above N=3 — the apparent dip at N=5 was a "
                 "contaminated render, not data",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


def dimension_flip(rows):
    """Curve-vs-k-means ratio against ambient dimension, four panels.

    exp4 found the one result pointing against all the others: on gauss10d,
    curves at matched rate BEAT k-means. Two candidate explanations, which exp4
    could not separate because that dataset has ambient D = intrinsic d = 10:

      * ambient-dimension story -- the continuity constraint removes one
        direction out of D, so it gets cheaper as D grows
      * intrinsic-dimension story -- something about structurelessness

    The `spiral` rows settle it. They carry intrinsic d = 1 into ambient D up to
    40, so if the flip tracked ambient dimension they would flip too. They do
    not: they converge to 1.0 from above and never cross. The flip tracks
    INTRINSIC dimension, which is the opposite of the ambient story.

    The two rate framings also trend in opposite directions with dimension,
    which is the reason to draw them side by side rather than pick one.
    """
    per = collections.defaultdict(dict)
    for r in rows:
        if r.get("kind") not in ("gauss", "spiral"):
            continue
        per[(r["kind"], r["dim"], r["grid_mode"], r["seed"])][r["order"]] = r["distortion_test"]
    dims = sorted({d for (_, d, _, _) in per})
    orders = [2, 3, 4, 6, 8]
    kinds = (("gauss", "gauss: intrinsic d = ambient D"),
             ("spiral", "spiral: intrinsic d = 1, rotated into D"))
    modes = (("matched", "rate matched (addresses charged)"),
             ("generous", "rate generous (addresses free)"))
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.4))
    cmap = plt.get_cmap("viridis")
    for i, (kind, ktitle) in enumerate(kinds):
        for j, (mode, mtitle) in enumerate(modes):
            ax = axes[i][j]
            for oi, o in enumerate(orders):
                xs, ys, es = [], [], []
                for d in dims:
                    seeds = [s for (k, dd, m, s) in per
                             if k == kind and dd == d and m == mode
                             and 1 in per[(k, dd, m, s)] and o in per[(k, dd, m, s)]]
                    if len(seeds) < 2:
                        continue
                    rs = [per[(kind, d, mode, s)][o] / per[(kind, d, mode, s)][1] for s in seeds]
                    xs.append(d)
                    ys.append(statistics.mean(rs))
                    es.append(statistics.stdev(rs) / len(rs) ** 0.5)
                ax.errorbar(xs, ys, yerr=es, marker="o", ms=4.5, lw=1.7, capsize=2.5,
                            color=cmap(oi / max(1, len(orders) - 1)), label=f"N={o}")
            ax.axhline(1.0, color="black", ls="--", lw=1.3)
            ax.set_xscale("log", base=2)
            ax.set_xticks(dims)
            ax.set_xticklabels([str(d) for d in dims])
            ax.set_xlabel("ambient dimension D")
            ax.set_ylabel("distortion ÷ distortion at N=1 (k-means)")
            ax.set_title(f"{ktitle}\n{mtitle}", fontsize=9.5)
            ax.grid(alpha=0.25, lw=0.6)
            if i == 0 and j == 0:
                ax.legend(fontsize=8)
    fig.suptitle("Below the dashed line curves beat k-means. "
                 "The flip tracks INTRINSIC dimension, not ambient:\n"
                 "gauss (top) crosses below 1 by D=20; spiral (bottom) carries d=1 "
                 "to D=40 and never crosses.",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def intrinsic_dimension(rows):
    """Curve advantage against INTRINSIC dimension, ambient dimension pinned.

    This is the figure that answers the question exp5 got wrong. exp5 varied
    ambient and intrinsic dimension together (`gauss_D`) or held intrinsic at 1
    (`spiral_D`), so it could show the advantage moving but not what moved it,
    and its headline -- curves win in high dimension -- came from a 40-D
    isotropic Gaussian where k-means scores 0.94 and curves 0.92, i.e. two
    methods failing.

    Here ambient dimension is pinned at 40 and only intrinsic dimension varies,
    so the trend is attributable. Real datasets are overlaid as horizontal
    lines: where they cut the curve is the intrinsic dimension they behave like.
    """
    per = collections.defaultdict(dict)
    for r in rows:
        if r.get("kind") not in ("subspace", "real"):
            continue
        per[(r["dataset"], r.get("kind"), r.get("intrinsic"), r["grid_mode"], r["seed"])][r["order"]] = \
            r["distortion_test"]

    def ratios(tag, kind, intr, mode, order):
        rs = [v[order] / v[1] for (t, k, i, m, _), v in per.items()
              if t == tag and m == mode and 1 in v and order in v]
        if len(rs) < 2:
            return None
        return statistics.mean(rs), statistics.stdev(rs) / len(rs) ** 0.5

    dims = sorted({i for (_, k, i, _, _) in per if k == "subspace" and i is not None})
    reals = sorted({t for (t, k, _, _, _) in per if k == "real"})
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2))
    for ax, mode, title in ((axes[0], "matched", "rate matched (addresses charged)"),
                            (axes[1], "generous", "rate generous (addresses free)")):
        for oi, order in enumerate((2, 3, 4)):
            xy = [(d, *ratios(f"sub{d}_in40", "subspace", d, mode, order))
                  for d in dims if ratios(f"sub{d}_in40", "subspace", d, mode, order)]
            if not xy:
                continue
            ax.errorbar([d for d, _, _ in xy], [m for _, m, _ in xy],
                        yerr=[e for _, _, e in xy], marker="o", ms=5, lw=1.8, capsize=3,
                        color=plt.get_cmap("viridis")(oi / 2), label=f"N={order}")
        for ri, tag in enumerate(reals):
            r = ratios(tag, "real", None, mode, 3)
            if r:
                ax.axhline(r[0], ls=":", lw=1.6,
                           color=["#C44E52", "#DD8452", "#8172B3", "#55A868"][ri % 4],
                           label=f"{tag} (N=3)")
        ax.axhline(1.0, color="black", ls="--", lw=1.3)
        ax.set_xscale("log", base=2)
        ax.set_xticks(dims)
        ax.set_xticklabels([str(d) for d in dims])
        ax.set_xlabel("intrinsic dimension d   (ambient D = 40 throughout)")
        ax.set_ylabel("distortion ÷ distortion at N=1 (k-means)")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25, lw=0.6)
        ax.legend(fontsize=8)
    fig.suptitle("The curve advantage is governed by INTRINSIC dimension, and ambient "
                 "dimension is irrelevant\n"
                 "free addressing: 3.6× better at d=1, nothing by d=32. "
                 "charged addressing: nothing anywhere. Real image data sits at the dead end.",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


HIGH_DIM = [("mnist_raw", 784), ("mnist_pca32", 32),
            ("fashion_mnist_raw", 784), ("fashion_mnist_pca32", 32),
            ("sub8_in40", 40), ("sub16_in40", 40), ("sub32_in40", 40),
            ("gauss_10d", 10), ("gauss_20d", 20), ("gauss_40d", 40)]


def does_bend_help(rows):
    """The plain version: same stored floats, fine grid, distortion against N.

    Every other figure here compares under some framing -- matched rate, free
    addressing, description length. Those exist to answer whether the shape or
    the address budget is doing the work, but they bury the plain question:
    holding the stored floats fixed, does raising N help on high-dimensional
    data? A fine grid is the natural choice, since grid resolution costs nothing
    to store, so that is what this uses and there is nothing else to hold.

    One line per dataset, percent improvement over N=1. Flat after N=2 is the
    whole finding.
    """
    a = collections.defaultdict(list)
    for r in rows:
        if r.get("grid_mode") == "generous":
            a[(r["dataset"], r["order"])].append(r["distortion_test"])
    orders = [1, 2, 3, 4, 6, 8]
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    cmap = plt.get_cmap("tab10")
    for i, (name, dim) in enumerate(HIGH_DIM):
        v = {o: statistics.mean(a[(name, o)]) for o in orders if a[(name, o)]}
        if 1 not in v:
            continue
        xs = [o for o in orders if o in v]
        ys = [(1 - v[o] / v[1]) * 100 for o in xs]
        real = not name.startswith(("sub", "gauss"))
        ax.plot(xs, ys, marker="o", ms=5.5, lw=2.2 if real else 1.4,
                ls="-" if real else "--", color=cmap(i % 10),
                label=f"{name} (D={dim})", zorder=3 if real else 2)
    ax.axhline(0.0, color="black", lw=1.3)
    ax.set_xticks(orders)
    ax.set_xlabel("curve order N      (1 = a point, 2 = a straight stick, ≥3 = bent)")
    ax.set_ylabel("% better than N=1 (k-means), same stored floats")
    ax.set_title("Does raising N help on high-dimensional data?\n"
                 "Yes — but N=2 gets all of it. Solid = real data, dashed = synthetic.",
                 fontsize=11)
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    return fig


def k_and_n(rows):
    """K and N swept independently: distortion against stored floats, line per N.

    Every other figure fixes K*N and sweeps N, which hides K -- at budget 128,
    N=1 runs with K=128 while N=2 runs with K=64, so "N=2 wins" is really "64
    sticks beat 128 points" and the shape effect is inseparable from the cluster
    count.

    Here K is traced ALONG each line instead of being eliminated. A vertical
    slice is an equal-cost comparison across N, and the lowest line at a given
    cost is the best order to pick. If the N=1 line is lowest everywhere, shape
    buys nothing and the earlier results were about K all along.
    """
    pts = collections.defaultdict(list)
    for r in rows:
        if r.get("grid_mode") == "indep":
            pts[(r["dataset"], r["order"])].append((r["stored_floats"], r["distortion_test"]))
    names = sorted({d for (d, _) in pts},
                   key=lambda d: (not d.startswith(("mnist", "fashion")), d))
    orders = sorted({o for (_, o) in pts})
    ncol = min(3, len(names))
    nrow = -(-len(names) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 4.3 * nrow), squeeze=False)
    for ax in axes.ravel():
        ax.set_visible(False)
    cmap = plt.get_cmap("viridis")
    for i, ds in enumerate(names):
        ax = axes[i // ncol][i % ncol]
        ax.set_visible(True)
        for oi, o in enumerate(orders):
            by_cost = collections.defaultdict(list)
            for cost, dist in pts[(ds, o)]:
                by_cost[cost].append(dist)
            xy = sorted((c, statistics.mean(v)) for c, v in by_cost.items())
            if not xy:
                continue
            ax.plot([c for c, _ in xy], [d for _, d in xy], marker="o", ms=4.5,
                    lw=2.4 if o == 1 else 1.7,
                    color="black" if o == 1 else cmap((oi - 1) / max(1, len(orders) - 2)),
                    label="N=1 (k-means)" if o == 1 else f"N={o}",
                    zorder=4 if o == 1 else 3)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("stored floats  K·N·D")
        ax.set_ylabel("held-out distortion")
        ax.set_title(ds, fontsize=10)
        ax.grid(alpha=0.25, lw=0.6)
        if i == 0:
            ax.legend(fontsize=8)
    fig.suptitle("K and N swept independently — K runs along each line, not hidden\n"
                 "a vertical slice is equal cost; lowest line wins. "
                 "Black is k-means.",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


TRIM_SETS = ["spiral2d", "square2d", "sub8_in40", "mnist_pca32"]


def trimmed_updates(rows, order=1):
    """Trimmed centre updates against contamination, two metrics.

    Top row is plain mean-squared distortion, where trimming is guaranteed to
    lose on clean data -- for a fixed partition the mean is the minimiser of
    squared distance, so any trimmed centre is worse by construction. Showing
    only that row would be a rigged test, so the bottom row is trimmed-50, the
    objective a trimmed update actually descends.

    x is the fraction of TRAINING points replaced by uniform junk; the test
    split is always clean, so this asks how well real data is represented after
    the model was shown garbage.
    """
    per = collections.defaultdict(dict)
    for r in from_exp(rows, "exp8"):
        if r["order"] != order:
            continue
        per[(r["dataset"], r["eps"], r["seed"])][r["trim_q"]] = (
            r["distortion_test"], r["trimmed50_test"])
    epss = sorted({e for (_, e, _) in per})
    qs = [0.25, 0.5, 0.75]
    fig, axes = plt.subplots(2, len(TRIM_SETS),
                             figsize=(3.7 * len(TRIM_SETS), 7.4), squeeze=False)
    cmap = plt.get_cmap("plasma")
    for mi, mlabel in enumerate(("plain distortion", "trimmed-50 distortion")):
        for di, ds in enumerate(TRIM_SETS):
            ax = axes[mi][di]
            for qi, q in enumerate(qs):
                xs, ys, es = [], [], []
                for e in epss:
                    rs = [v[q][mi] / v[1.0][mi] for (d, ee, _), v in per.items()
                          if d == ds and ee == e and q in v and 1.0 in v]
                    if len(rs) < 2:
                        continue
                    xs.append(e * 100)
                    ys.append(statistics.mean(rs))
                    es.append(statistics.stdev(rs) / len(rs) ** 0.5)
                ax.errorbar(xs, ys, yerr=es, marker="o", ms=5, lw=1.8, capsize=3,
                            color=cmap(qi / (len(qs) + 0.4)), label=f"keep q={q}")
            ax.axhline(1.0, color="black", ls="--", lw=1.3)
            ax.set_xlabel("% of training data replaced by junk")
            ax.set_ylabel(f"{mlabel} ÷ untrimmed")
            ax.set_title(f"{ds} — {mlabel}", fontsize=9.5)
            ax.grid(alpha=0.25, lw=0.6)
            if mi == 0 and di == 0:
                ax.legend(fontsize=8)
    fig.suptitle(f"Trimmed centre updates (N={order}). Below the dashed line trimming wins.\n"
                 "Top: the metric trimming cannot win on by construction. "
                 "Bottom: the metric it actually optimises.",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def trimmed_picture(panels=(("spiral2d", 0.0), ("spiral2d", 0.2), ("square2d", 0.2)),
                    qs=(1.0, 0.75, 0.5, 0.25), k=16, seed=0, zoom=1.8):
    """What trimmed updates actually do, drawn on 2-D data.

    K is 16 here rather than exp8's 64, purely so the centres and their
    acceptance discs are legible. The mechanism is the same, so the numbers in
    the panel titles are not comparable to exp8's tables.

    The view is clipped to `zoom` times the extent of the CLEAN data. Framing on
    the contaminated data instead is useless: junk is spread over a box three
    times the data's scale, so it sets the limits and squashes the actual
    dataset into a few pixels. Junk therefore extends past the frame, and each
    panel reports how many centres were dragged outside it -- which is the
    effect being looked for, counted rather than eyeballed.

    Grey: clean training points. Red: injected junk. Black: centres. Arrows run
    from each untrimmed centre to where trimming moved it. The dashed circle is
    the widest learned acceptance radius, i.e. the threshold itself made visible.
    """
    fig, axes = plt.subplots(len(panels), len(qs),
                             figsize=(4.2 * len(qs), 4.4 * len(panels)), squeeze=False)
    for row, (ds, eps) in enumerate(panels):
        base_tr, te = toy.load(ds, seed=seed)
        tr, bad = toy.contaminate(base_tr, eps, seed)
        order = np.random.default_rng(1000 + seed).permutation(len(tr))
        fits = {}
        for q in qs:
            m = CurveClusters(k, 1, tr.shape[1], 32, trim_q=q)
            m.partial_fit(tr[order])
            fits[q] = (m.values[:m.n_seeded, 0, :].copy(), m.distortion(te),
                       m.thresh[:m.n_seeded].copy())
        ref = fits[1.0][0]
        lim = float(np.abs(base_tr).max()) * zoom
        for col, q in enumerate(qs):
            ax = axes[row][col]
            cen, dist, thr = fits[q]
            ax.scatter(tr[~bad, 0], tr[~bad, 1], s=4, c="#C8C8C8", linewidths=0, zorder=1)
            if bad.any():
                ax.scatter(tr[bad, 0], tr[bad, 1], s=11, c="#D62728", linewidths=0,
                           alpha=0.7, zorder=2, label="injected junk")
            if q < 1.0:
                for a, b in zip(ref, cen):
                    if max(np.abs(a).max(), np.abs(b).max()) < lim * 1.4:
                        ax.annotate("", xy=b, xytext=a, zorder=3,
                                    arrowprops=dict(arrowstyle="->", lw=1.1,
                                                    color="#1f77b4", alpha=0.9))
                finite = np.isfinite(thr)
                if finite.any():
                    i = int(np.argmax(np.where(finite, thr, -np.inf)))
                    ax.add_patch(plt.Circle(cen[i], thr[i], fill=False, ls="--",
                                            lw=1.3, color="#1f77b4", zorder=4))
            ax.scatter(cen[:, 0], cen[:, 1], s=55, c="black", edgecolors="white",
                       linewidths=1.0, zorder=5)
            outside = int((np.abs(cen).max(1) > lim).sum())
            lab = "untrimmed (k-means)" if q == 1.0 else f"keep closest q={q}"
            ax.set_title(f"{ds}, {eps:.0%} junk — {lab}\n"
                         f"clean distortion {dist:.4f}   ·   "
                         f"{outside}/{len(cen)} centres dragged off-frame",
                         fontsize=9)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
            if row == 0 and col == 1:
                ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("Trimmed centre updates on 2-D data, K=16 centres, single pass.\n"
                 "Arrows: where trimming moved each centre. Dashed circle: the widest "
                 "learned acceptance radius. View clipped to the clean data.",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    return fig


def trim_shape_grid(dataset="spiral2d", orders=(1, 2, 3, 6),
                    qs=(1.0, 0.75, 0.5, 0.25), k=16, seed=0):
    """What trimming does to the SHAPE of a curve, on the clean dataset.

    K is held fixed across orders rather than fixing the K*N budget, on purpose:
    the question here is what trimming does to a curve of a given order, so
    everything except the order and the trim has to stay put. Distortion in the
    titles therefore falls with N simply because higher N buys more parameters
    -- see Q5 for the budget-matched comparison.

    Mean arc length is printed alongside, because the visible effect is
    quantitative: trimming discards the far assigned points, and those are the
    ones that stretch a curve to its full extent, so trimmed curves should
    shrink toward the dense core of their cluster.
    """
    fig, axes = plt.subplots(len(orders), len(qs),
                             figsize=(4.1 * len(qs), 4.3 * len(orders)), squeeze=False)
    train, test = toy.load(dataset, seed=seed)
    order_ix = np.random.default_rng(1000 + seed).permutation(len(train))
    for row, n in enumerate(orders):
        for col, q in enumerate(qs):
            ax = axes[row][col]
            m = CurveClusters(k, n, train.shape[1], max(32, 8 * n), trim_q=q)
            m.partial_fit(train[order_ix])
            ax.scatter(train[:, 0], train[:, 1], s=4, c="#D0D0D0", linewidths=0, zorder=1)
            arc = []
            for j in range(m.n_seeded):
                v = m.values[j]
                arc.append(float(np.linalg.norm(np.diff(v, axis=0), axis=1).sum()))
                if n == 1:
                    ax.plot(v[0, 0], v[0, 1], marker="o", ms=6, zorder=3)
                else:
                    ax.plot(v[:, 0], v[:, 1], lw=1.9, zorder=2)
                    ax.plot(v[[0, -1], 0], v[[0, -1], 1], ls="none", marker="o",
                            ms=3.0, color="black", zorder=3)
            lab = "untrimmed (k-means rule)" if q == 1.0 else f"keep closest q={q}"
            ax.set_title(f"N={n}  ·  {lab}\n"
                         f"distortion {m.distortion(test):.4f}   ·   "
                         f"mean arc length {statistics.mean(arc):.3f}",
                         fontsize=9)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
            lim = float(np.abs(train).max()) * 1.12
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    fig.suptitle(f"Trimmed updates on clean {dataset}, K={k} fixed, single pass.\n"
                 "Rows: curve order. Columns: how much of each cluster's data the "
                 "update is allowed to use.",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    return fig


def trim_shape_datasets(datasets=("spiral2d", "moons2d", "square2d", "checkerboard"),
                        qs=(1.0, 0.75, 0.5, 0.25), order=3, k=16, seed=0):
    """The same view across clean datasets at one order."""
    fig, axes = plt.subplots(len(datasets), len(qs),
                             figsize=(4.1 * len(qs), 4.3 * len(datasets)), squeeze=False)
    for row, ds in enumerate(datasets):
        train, test = toy.load(ds, seed=seed)
        order_ix = np.random.default_rng(1000 + seed).permutation(len(train))
        for col, q in enumerate(qs):
            ax = axes[row][col]
            m = CurveClusters(k, order, train.shape[1], 8 * order, trim_q=q)
            m.partial_fit(train[order_ix])
            ax.scatter(train[:, 0], train[:, 1], s=4, c="#D0D0D0", linewidths=0, zorder=1)
            arc = []
            for j in range(m.n_seeded):
                v = m.values[j]
                arc.append(float(np.linalg.norm(np.diff(v, axis=0), axis=1).sum()))
                ax.plot(v[:, 0], v[:, 1], lw=1.9, zorder=2)
            lab = "untrimmed" if q == 1.0 else f"q={q}"
            ax.set_title(f"{ds}  ·  {lab}\n"
                         f"distortion {m.distortion(test):.4f}   ·   "
                         f"arc {statistics.mean(arc):.3f}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
            lim = float(np.abs(train).max()) * 1.12
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    fig.suptitle(f"Trimmed updates on clean data, order N={order}, K={k} fixed.",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    return fig
