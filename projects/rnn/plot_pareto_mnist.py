"""
Plot Pareto-optimal front: non-embedding params vs test accuracy.

Usage:
    uv run python projects/rnn/plot_pareto_mnist.py

Reads results_mnist.jsonl and saves pareto_mnist.png via shared_lib.media.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np

# allow running from repo root or from this directory
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared_lib.media import save_matplotlib_figure

JSONL = Path(__file__).parent / "results_mnist.jsonl"


def load_results(path=JSONL):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def pareto_front(points):
    """Return indices of Pareto-optimal points (minimize params, maximize acc)."""
    pts = sorted(enumerate(points), key=lambda iv: iv[1][0])  # sort by params asc
    front = []
    best_acc = -1.0
    for idx, (params, acc) in pts:
        if acc > best_acc:
            best_acc = acc
            front.append(idx)
    return set(front)


ACC_FLOOR = 0.94  # hide all points below this threshold


def make_plot(rows):
    rows    = [r for r in rows if r["test_acc"] >= ACC_FLOOR]
    params  = [r["n_nonemb_params"] for r in rows]
    acc     = [r["test_acc"] for r in rows]
    names   = [r["name"] for r in rows]
    has_tr  = [r.get("train_acc") is not None for r in rows]

    front_idx = pareto_front(list(zip(params, acc)))

    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xscale("log")

    # ── all points ────────────────────────────────────────────────────────────
    for i, (p, a, nm) in enumerate(zip(params, acc, names)):
        color  = "#e84646" if i in front_idx else "#6baed6"
        zorder = 4 if i in front_idx else 2
        marker = "D" if i in front_idx else "o"
        ms     = 9 if i in front_idx else 6
        ax.scatter(p, a, color=color, zorder=zorder, s=ms**2, marker=marker,
                   edgecolors="white", linewidths=0.5)

    # ── Pareto staircase ──────────────────────────────────────────────────────
    front_pts = sorted([(params[i], acc[i]) for i in front_idx])
    fx = [p for p, _ in front_pts]
    fy = [a for _, a in front_pts]
    # draw as step-function staircase
    sx, sy = [fx[0]], [fy[0]]
    for (x2, y2), (x1, y1) in zip(front_pts[1:], front_pts[:-1]):
        sx += [x2, x2]
        sy += [y1, y2]
    ax.plot(sx, sy, color="#e84646", linewidth=1.5, linestyle="--",
            zorder=3, alpha=0.7, label="Pareto front")

    # ── labels for Pareto points ──────────────────────────────────────────────
    for i in front_idx:
        p, a, nm = params[i], acc[i], names[i]
        # shorten label
        short = nm.replace("outer+asym", "asym").replace("outer", "out").replace(" mlp=", "/")
        ax.annotate(
            f"{short}\n{a:.3f}",
            xy=(p, a), xytext=(6, 4), textcoords="offset points",
            fontsize=7.5, color="#c0392b", zorder=5,
            path_effects=[pe.withStroke(linewidth=2, foreground="white")],
        )

    # ── labels for non-Pareto points (smaller, grey) ─────────────────────────
    for i, (p, a, nm) in enumerate(zip(params, acc, names)):
        if i in front_idx:
            continue
        short = nm.replace("outer+asym", "asym").replace("outer", "out").replace(" mlp=", "/")
        ax.annotate(
            short, xy=(p, a), xytext=(4, 3), textcoords="offset points",
            fontsize=6.5, color="#555555", zorder=3,
            path_effects=[pe.withStroke(linewidth=1.5, foreground="white")],
        )

    # ── train accuracy markers (×) ────────────────────────────────────────────
    for i, r in enumerate(rows):
        if r.get("train_acc") is not None:
            ax.scatter(params[i], r["train_acc"], marker="x", color="#888888",
                       s=40, zorder=3, linewidths=1.2)
    # legend proxy
    ax.scatter([], [], marker="x", color="#888888", s=40, linewidths=1.2,
               label="train acc (×)")
    ax.scatter([], [], color="#e84646", marker="D", s=81, edgecolors="white",
               linewidths=0.5, label="Pareto optimal")
    ax.scatter([], [], color="#6baed6", marker="o", s=36, edgecolors="white",
               linewidths=0.5, label="other")

    ax.set_xlabel("Non-embedding parameters", fontsize=12)
    ax.set_ylabel("Test accuracy", fontsize=12)
    ax.set_title("MNIST pixel-by-pixel linear RNN — Pareto front\n"
                 "(non-embedding params vs test accuracy)", fontsize=13)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, which="both", alpha=0.25)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1%}"))

    # ── target lines ─────────────────────────────────────────────────────────
    for tgt, lbl in [(0.95, "95%"), (0.96, "96%"), (0.97, "97%"),
                     (0.98, "98%"), (0.99, "99%")]:
        if tgt < ACC_FLOOR:
            continue
        ax.axhline(tgt, color="#aaaaaa", linewidth=0.8, linestyle=":", zorder=1)
        ax.text(ax.get_xlim()[0] if ax.get_xlim()[0] > 0 else 1e4,
                tgt + 0.001, lbl, fontsize=8, color="#888888")

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    rows = load_results()
    print(f"Loaded {len(rows)} results")

    # print current Pareto front
    params  = [r["n_nonemb_params"] for r in rows]
    acc     = [r["test_acc"] for r in rows]
    names   = [r["name"] for r in rows]
    front   = pareto_front(list(zip(params, acc)))
    print("\nPareto front:")
    for i in sorted(front, key=lambda i: params[i]):
        print(f"  {params[i]:>10,} params  {acc[i]:.4f}  {names[i]}")

    fig = make_plot(rows)
    path = save_matplotlib_figure("pareto_mnist", fig, format="png", dpi=150)
    print(f"\nSaved: {path}")

    # Re-render with text as actual SVG <text> elements (not path outlines)
    plt.rcParams["svg.fonttype"] = "none"
    fig2 = make_plot(rows)
    svg_path = save_matplotlib_figure("pareto_mnist", fig2, format="svg")
    print(f"Saved: {svg_path}")
    plt.close("all")
