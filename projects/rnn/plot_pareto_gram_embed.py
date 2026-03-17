"""
MNIST Pareto plot: f3(4,4,4) d=64 baseline vs f3 small-d (d=8, d=16).

X-axis: TOTAL parameters (non-emb + emb).
Y-axis: test accuracy.

Series:
  1. f3_d64   — existing f3(4,4,4) results, gray background + red Pareto staircase
  2. f3_small_d — exp26 f3(2,2,2)/f3(4,4,1) results, orange + orange Pareto staircase

At the same TOTAL param budget, smaller d leaves room for larger K/mlp,
potentially achieving higher accuracy per param.

Usage:
    uv run python projects/rnn/plot_pareto_gram_embed.py
"""

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared_lib.media import save_matplotlib_figure

JSONL     = Path(__file__).parent / "results_mnist.jsonl"
ACC_FLOOR = 0.80


def load_rows():
    rows = [json.loads(l) for l in open(JSONL) if l.strip()]
    rows = [r for r in rows if r["test_acc"] >= ACC_FLOOR]
    # Compute total params: use stored n_total_params if available,
    # else fall back to n_nonemb_params (old rows without emb tracking).
    for r in rows:
        if "n_total_params" not in r:
            r["n_total_params"] = r["n_nonemb_params"]
    return rows


def pareto_front(pts):
    """Return indices of Pareto-optimal (min params, max acc) points."""
    s = sorted(enumerate(pts), key=lambda iv: iv[1][0])
    front, best = [], -1.0
    for idx, (p, a) in s:
        if a > best:
            best = a
            front.append(idx)
    return set(front)


def staircase(front_pts):
    """Convert sorted Pareto points into staircase (sx, sy) lists."""
    sx, sy = [front_pts[0][0]], [front_pts[0][1]]
    for (x2, y2), (x1, y1) in zip(front_pts[1:], front_pts[:-1]):
        sx += [x2, x2]; sy += [y1, y2]
    return sx, sy


def _short(name):
    return (name
            .replace("gram_embed", "ge")
            .replace("+noise30", "")
            .replace("no_gram_reshape_1h f3(4,4,4) ", "rsh1h ")
            .replace("no_gram_reshape_2h f3(4,4,4) ", "rsh2h ")
            .replace("f3(4,4,4) bilinear ", "f3 ")
            .replace("joint(d=16) ", "jt16 ")
            .replace("(4→", "(")
            .strip())


def make_plot(rows):
    # f3(4,4,4) d=64 baseline: existing runs that used the large-d f3 embedding
    d64  = [r for r in rows if "f3(4,4,4)" in r["name"] and r.get("family") != "f3_small_d"]
    # new small-d models
    sd   = [r for r in rows if r.get("family") == "f3_small_d"]

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xscale("log")

    # ── d=64 baseline: background ──────────────────────────────────────────────
    d64_pts = [(r["n_total_params"], r["test_acc"]) for r in d64]
    d64_fi  = pareto_front(d64_pts)
    for i, (p, a) in enumerate(d64_pts):
        on_front = i in d64_fi
        ax.scatter(p, a,
                   color="#e84646" if on_front else "#aec7e8",
                   marker="D" if on_front else "o",
                   s=(7 if on_front else 5)**2,
                   edgecolors="white", linewidths=0.4,
                   zorder=3 if on_front else 2, alpha=0.7)

    d64_front_pts = sorted([d64_pts[i] for i in d64_fi])
    if len(d64_front_pts) > 1:
        sx, sy = staircase(d64_front_pts)
        ax.plot(sx, sy, color="#e84646", lw=1.5, ls="--", zorder=4, alpha=0.7,
                label="f3(4,4,4) d=64 Pareto front")

    for i in d64_fi:
        p, a = d64_pts[i]
        nm = _short(d64[i]["name"])
        ax.annotate(f"{nm}\n{a:.3f}", xy=(p, a), xytext=(-4, 5),
                    textcoords="offset points", fontsize=5.5, color="#c0392b",
                    zorder=5,
                    path_effects=[pe.withStroke(linewidth=1.5, foreground="white")])

    # ── small-d: foreground ────────────────────────────────────────────────────
    if sd:
        sd_pts = [(r["n_total_params"], r["test_acc"]) for r in sd]
        sd_fi  = pareto_front(sd_pts)

        ax.scatter([p for p, _ in sd_pts], [a for _, a in sd_pts],
                   color="#e67e22", marker="s", s=6**2,
                   edgecolors="white", linewidths=0.4, zorder=5, alpha=0.4)

        sd_front_pts = sorted([sd_pts[i] for i in sd_fi])
        if len(sd_front_pts) > 1:
            sx, sy = staircase(sd_front_pts)
            ax.plot(sx, sy, color="#e67e22", lw=2.0, ls="-", zorder=7, alpha=0.9,
                    label="f3 small-d (d=8/16) Pareto front")

        ax.scatter([sd_pts[i][0] for i in sd_fi],
                   [sd_pts[i][1] for i in sd_fi],
                   color="#e67e22", marker="s", s=9**2,
                   edgecolors="white", linewidths=0.5, zorder=8)

        sd_front_set = set(sd_front_pts)
        for i, (p, a) in enumerate(sd_pts):
            if (p, a) in sd_front_set:
                nm = _short(sd[i]["name"])
                ax.annotate(f"{nm}\n{a:.3f}", xy=(p, a), xytext=(5, 3),
                            textcoords="offset points", fontsize=6, color="#d35400",
                            zorder=9,
                            path_effects=[pe.withStroke(linewidth=1.5,
                                                        foreground="white")])

    # ── reference lines ────────────────────────────────────────────────────────
    for tgt in [0.95, 0.96, 0.97, 0.975, 0.98, 0.985, 0.99]:
        if tgt < ACC_FLOOR:
            continue
        ax.axhline(tgt, color="#cccccc", lw=0.7, ls=":", zorder=1)
        ax.text(ax.get_xlim()[0] if ax.get_xlim()[0] > 1 else 100,
                tgt + 0.0008, f"{tgt:.1%}", fontsize=7.5, color="#999999")

    ax.set_xlabel("Total parameters (non-emb + emb)", fontsize=12)
    ax.set_ylabel("Test accuracy", fontsize=12)
    ax.set_title(
        "MNIST pixel-by-pixel — f3 small-d (d=8/16) vs d=64 baseline\n"
        "(total params — smaller d allows larger K/mlp at same budget)", fontsize=13)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, which="both", alpha=0.18)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1%}"))
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    rows = load_rows()
    sd = [r for r in rows if r.get("family") == "f3_small_d"]
    print(f"Total rows: {len(rows)}   f3_small_d: {len(sd)}")

    if sd:
        sd_pts = [(r["n_total_params"], r["test_acc"]) for r in sd]
        sd_fi  = pareto_front(sd_pts)
        print("\nf3_small_d Pareto front (total params):")
        for i in sorted(sd_fi, key=lambda i: sd_pts[i][0]):
            print(f"  {sd_pts[i][0]:>8,} total  {sd_pts[i][1]:.4f}  {sd[i]['name']}")

    fig = make_plot(rows)
    path = save_matplotlib_figure("pareto_gram_embed", fig, format="png", dpi=150)
    print(f"\nSaved: {path}")

    plt.rcParams["svg.fonttype"] = "none"
    fig2 = make_plot(rows)
    svg  = save_matplotlib_figure("pareto_gram_embed", fig2, format="svg")
    print(f"Saved: {svg}")
    plt.close("all")
