"""
Fashion-MNIST architecture comparison plot.

X-axis: total parameters.
Y-axis: test accuracy.

Series (per-family Pareto staircases):
  1. gram_f3    — red    diamonds  (Gram bilinear, f3(4,4,4) emb)
  2. gram_joint — orange squares   (Gram bilinear, joint(d=16) emb)
  3. reshape    — purple hexagons  (no_gram_reshape_1h)
  4. pure_mlp   — green  triangles (1-layer MLP on raw pixels)
  5. pure_mlp2  — teal   triangles (2-layer MLP on raw pixels)

Usage:
    uv run python projects/rnn/plot_pareto_fashion.py
"""

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared_lib.media import save_matplotlib_figure

JSONL     = Path(__file__).parent / "results_fashion_mnist.jsonl"
ACC_FLOOR = 0.80


def load_rows():
    with open(JSONL) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    return [r for r in rows if r["test_acc"] >= ACC_FLOOR]


def pareto_front(pts):
    """Return indices of Pareto-optimal (min params, max acc) points."""
    sorted_pts = sorted(enumerate(pts), key=lambda iv: iv[1][0])
    front, best = [], -1.0
    for idx, (p, a) in sorted_pts:
        if a > best:
            best = a
            front.append(idx)
    return set(front)


# ── series config ─────────────────────────────────────────────────────────────

SERIES = [
    # (family_key,   label,                      color,     marker, lw,  ms, zo)
    ("gram_f3",    "Gram bilinear f3(4,4,4)",  "#e84646", "D",   2.0,  8,  6),
    ("flat_gram",  "Gram flat (full M)",        "#e74c3c", "*",   1.8,  9,  5),
    ("gram_joint", "Gram bilinear joint(d=16)","#e67e22", "s",   1.5,  7,  4),
    ("reshape",    "no_gram reshape_1h",        "#9b59b6", "h",   1.5,  8,  4),
    ("pure_mlp",   "pure MLP 1-layer",          "#27ae60", "^",   1.5,  8,  4),
    ("pure_mlp2",  "pure MLP 2-layer",          "#1abc9c", "v",   1.5,  8,  4),
]


def _short(name):
    return (name
            .replace("+noise30", "").replace("+noise15", "n15")
            .replace("flat_gram ", "").replace("gram_f3 ", "")
            .replace("gram_joint ", "").replace("reshape_1h ", "")
            .replace("pure_mlp 1L ", "").replace("pure_mlp 2L ", "")
            .strip())


def make_plot(rows):
    # bucket by family
    buckets = {key: [] for key, *_ in SERIES}
    for r in rows:
        fam = r.get("family", "")
        if fam in buckets:
            buckets[fam].append(r)

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xscale("log")

    for key, label, color, marker, lw, ms, zo in SERIES:
        fam_rows = buckets[key]
        if not fam_rows:
            continue
        all_pts = sorted(
            (r["n_total_params"], r["test_acc"], r["name"]) for r in fam_rows)

        front_idx = pareto_front([(p, a) for p, a, _ in all_pts])
        fp_pts    = sorted([(all_pts[i][0], all_pts[i][1]) for i in front_idx])

        # all points (small, semi-transparent)
        ax.scatter([p for p,_,_ in all_pts], [a for _,a,_ in all_pts],
                   color=color, marker=marker, s=(ms*0.7)**2,
                   edgecolors="white", lw=0.4, zorder=zo, alpha=0.4)

        # Pareto staircase
        if len(fp_pts) > 1:
            sx, sy = [fp_pts[0][0]], [fp_pts[0][1]]
            for (x2, y2), (x1, y1) in zip(fp_pts[1:], fp_pts[:-1]):
                sx += [x2, x2]; sy += [y1, y2]
            ax.plot(sx, sy, color=color, lw=lw, ls="-", zorder=zo+1, alpha=0.85)

        # Pareto points (full size)
        ax.scatter([p for p,_ in fp_pts], [a for _,a in fp_pts],
                   color=color, marker=marker, s=ms**2,
                   edgecolors="white", lw=0.5, zorder=zo+2, label=label)

        # labels on Pareto points
        fp_set = set(fp_pts)
        for p, a, nm in all_pts:
            if (p, a) in fp_set:
                ax.annotate(_short(nm), xy=(p, a), xytext=(4, 3),
                            textcoords="offset points", fontsize=5.5, color=color,
                            zorder=zo+3,
                            path_effects=[pe.withStroke(linewidth=1.5,
                                                        foreground="white")])

    # reference lines
    for tgt in [0.82, 0.84, 0.86, 0.88, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95]:
        if tgt < ACC_FLOOR:
            continue
        ax.axhline(tgt, color="#cccccc", lw=0.7, ls=":", zorder=1)
        ax.text(ax.get_xlim()[0] if ax.get_xlim()[0] > 1 else 200,
                tgt + 0.001, f"{tgt:.0%}", fontsize=7.5, color="#999999")

    ax.set_xlabel("Total parameters (non-emb + emb)", fontsize=12)
    ax.set_ylabel("Test accuracy", fontsize=12)
    ax.set_title("Fashion-MNIST pixel-by-pixel — architecture family comparison\n"
                 "(total params vs test accuracy, noise aug, 40 epochs)", fontsize=13)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, which="both", alpha=0.18)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1%}"))
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    rows = load_rows()
    print(f"Loaded {len(rows)} rows")
    by_family = {}
    for r in rows:
        by_family.setdefault(r.get("family","?"), []).append(r)
    for fam, rs in sorted(by_family.items()):
        best = max(r["test_acc"] for r in rs)
        print(f"  {fam:<14}: {len(rs):2d} runs, best={best:.4f}")

    fig = make_plot(rows)
    path = save_matplotlib_figure("pareto_fashion", fig, format="png", dpi=150)
    print(f"Saved: {path}")

    plt.rcParams["svg.fonttype"] = "none"
    fig2 = make_plot(rows)
    svg  = save_matplotlib_figure("pareto_fashion", fig2, format="svg")
    print(f"Saved: {svg}")
    plt.close("all")
