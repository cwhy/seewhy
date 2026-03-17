"""
Ablation plot: effect of the Gram matrix M, with improved baselines.

X-axis: total parameters (non-emb + emb).
Y-axis: test accuracy.

Series:
  1. Pareto front        — red  staircase  (Gram bilinear, all prior exps)
  2. no_gram_fullH       — orange squares  (H @ h → 784 features; exp21)
  3. no_gram_mean        — gold  circles   (mean(H) → K dot products; exp22)
  4. no_gram_bin4        — brown diamonds  (4×4 spatial bins → 16*K features; exp22)
  5. pure_mlp 1-layer    — green triangles (conv_mlp exp21 + mlp=2/4/8 exp22)
  6. pure_mlp 2-layer    — teal  triangles (exp22)

Usage:
    uv run python projects/rnn/plot_ablation_mnist.py
"""

import json, re, sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared_lib.media import save_matplotlib_figure

JSONL     = Path(__file__).parent / "results_mnist.jsonl"
ACC_FLOOR = 0.88


# ── infer total params for pre-exp21 results (no n_total_params field) ─────────

def infer_emb_params(name: str) -> int:
    m = re.search(r'f(?:actored)?3\((\d+),(\d+),(\d+)\)', name)
    if m:
        d_r, d_c, d_v = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return 28 * d_r + 28 * d_c + 32 * d_v
    m = re.search(r'joint\(d=(\d+)\)', name)
    if m:
        return 784 * 32 * int(m.group(1))
    m = re.search(r'outer.*?\((\d+),(\d+)\)', name)
    if m:
        return 784 * int(m.group(1)) + 32 * int(m.group(2))
    return 0

def get_total(row: dict) -> int:
    if "n_total_params" in row:
        return row["n_total_params"]
    return row["n_nonemb_params"] + infer_emb_params(row["name"])


# ── load and bucket results ─────────────────────────────────────────────────────

def load_and_bucket():
    with open(JSONL) as f:
        rows = [json.loads(l) for l in f if l.strip()]

    pareto_rows  = []   # all non-ablation experiments
    fullH_rows   = []   # no_gram_fullH    (exp21)
    mean_rows    = []   # no_gram_mean     (exp22)
    bin4_rows    = []   # no_gram_bin4     (exp22)
    mlp1_rows    = []   # pure_mlp 1-layer (exp21 conv_mlp + exp22)
    mlp2_rows    = []   # pure_mlp 2-layer (exp22)
    reshape_rows = []   # no_gram_reshape  (exp23)

    for r in rows:
        name = r["name"]
        exp  = r.get("experiment", "")
        acc  = r["test_acc"]
        if acc < ACC_FLOOR:
            continue

        if name.startswith("no_gram_reshape"):
            reshape_rows.append(r)
        elif name.startswith("no_gram_mean"):
            mean_rows.append(r)
        elif name.startswith("no_gram_bin4"):
            bin4_rows.append(r)
        elif name.startswith("no_gram"):
            fullH_rows.append(r)
        elif name.startswith("pure_mlp 2layer") or name.startswith("pure_mlp 2-layer"):
            mlp2_rows.append(r)
        elif name.startswith("pure_mlp") or name.startswith("conv_mlp"):
            mlp1_rows.append(r)
        elif exp not in ("exp21", "exp22", "exp23", "exp24"):
            pareto_rows.append(r)

    return pareto_rows, fullH_rows, mean_rows, bin4_rows, mlp1_rows, mlp2_rows, reshape_rows


# ── Pareto front ───────────────────────────────────────────────────────────────

def pareto_front(pts):
    sorted_pts = sorted(enumerate(pts), key=lambda iv: iv[1][0])
    front, best = [], -1.0
    for idx, (p, a) in sorted_pts:
        if a > best:
            best = a
            front.append(idx)
    return set(front)


# ── plot ────────────────────────────────────────────────────────────────────────

SERIES = [
    # (key,      label,                        color,     marker, lw,  ms, zo, ann)
    ("fullH",   "no_gram H@h (784 feat)",      "#e67e22", "s",  1.5,  7,  3, True),
    ("mean",    "no_gram mean pool",           "#f1c40f", "o",  1.5,  7,  3, True),
    ("bin4",    "no_gram 4×4 bins (hardcoded)","#c0392b", "P",  1.5,  8,  3, True),
    ("reshape", "no_gram reshape+h_helper",    "#9b59b6", "h",  1.8,  8,  4, True),
    ("mlp1",    "pure MLP 1-layer",            "#27ae60", "^",  1.5,  8,  4, True),
    ("mlp2",    "pure MLP 2-layer",            "#1abc9c", "v",  1.5,  8,  4, True),
]

def _short(name):
    return (name
            .replace("+noise30", "").replace("no_gram_mean ", "mean ")
            .replace("no_gram_bin4 ", "bin4 ").replace("no_gram ", "")
            .replace("f3(4,4,4) ", "").replace("pure_mlp ", "")
            .replace("conv_mlp ", "").strip())

def make_plot(pareto_rows, fullH_rows, mean_rows, bin4_rows, mlp1_rows, mlp2_rows,
              reshape_rows=None):
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xscale("log")

    # ── 1. Pareto front ────────────────────────────────────────────────────────
    pts      = [(get_total(r), r["test_acc"]) for r in pareto_rows]
    front    = pareto_front(pts)
    fp       = sorted([pts[i] for i in front])
    fx, fy   = [p for p,_ in fp], [a for _,a in fp]
    # staircase
    sx, sy = [fx[0]], [fy[0]]
    for (x2, y2), (x1, y1) in zip(fp[1:], fp[:-1]):
        sx += [x2, x2]; sy += [y1, y2]
    ax.plot(sx, sy, color="#e84646", lw=2, ls="--", zorder=3, alpha=0.6)
    ax.scatter(fx, fy, color="#e84646", marker="D", s=80,
               edgecolors="white", lw=0.6, zorder=5,
               label="Pareto front (Gram bilinear)")
    # labels on a few key Pareto points (every other to avoid crowding)
    for i, (p, a) in enumerate(fp):
        if i % 2 == 0 or a > 0.982:
            ax.annotate(f"{a:.3f}", xy=(p, a), xytext=(4, 3),
                        textcoords="offset points", fontsize=6, color="#c0392b",
                        zorder=6,
                        path_effects=[pe.withStroke(linewidth=2, foreground="white")])

    # ── 2–6. Ablation series (per-series Pareto staircase) ────────────────────
    buckets = {"fullH": fullH_rows, "mean": mean_rows, "bin4": bin4_rows,
               "reshape": reshape_rows or [], "mlp1": mlp1_rows, "mlp2": mlp2_rows}

    for key, label, color, marker, lw, ms, zo, ann in SERIES:
        rows = buckets[key]
        if not rows:
            continue
        all_pts = sorted((get_total(r), r["test_acc"], r["name"]) for r in rows)

        # compute per-series Pareto front
        front_idx = pareto_front([(p, a) for p, a, _ in all_pts])
        fp_pts    = sorted([(all_pts[i][0], all_pts[i][1]) for i in front_idx])

        # draw all points (small, semi-transparent)
        ax.scatter([p for p,_,_ in all_pts], [a for _,a,_ in all_pts],
                   color=color, marker=marker, s=(ms*0.7)**2,
                   edgecolors="white", lw=0.4, zorder=zo, alpha=0.45)

        # draw Pareto staircase
        if len(fp_pts) > 1:
            sx, sy = [fp_pts[0][0]], [fp_pts[0][1]]
            for (x2, y2), (x1, y1) in zip(fp_pts[1:], fp_pts[:-1]):
                sx += [x2, x2]; sy += [y1, y2]
            ax.plot(sx, sy, color=color, lw=lw, ls="-", zorder=zo+1, alpha=0.85)

        # draw Pareto points (full size)
        ax.scatter([p for p,_ in fp_pts], [a for _,a in fp_pts],
                   color=color, marker=marker, s=ms**2,
                   edgecolors="white", lw=0.5, zorder=zo+2, label=label)

        # labels only on Pareto points
        if ann:
            fp_set = set(fp_pts)
            for p, a, nm in all_pts:
                if (p, a) in fp_set:
                    ax.annotate(_short(nm), xy=(p, a), xytext=(4, 3),
                                textcoords="offset points", fontsize=5.5, color=color,
                                zorder=zo+3,
                                path_effects=[pe.withStroke(linewidth=1.5,
                                                            foreground="white")])

    # ── reference lines ────────────────────────────────────────────────────────
    for tgt in [0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]:
        if tgt < ACC_FLOOR:
            continue
        ax.axhline(tgt, color="#cccccc", lw=0.7, ls=":", zorder=1)
        ax.text(ax.get_xlim()[0] if ax.get_xlim()[0] > 1 else 100,
                tgt + 0.001, f"{tgt:.0%}", fontsize=7.5, color="#999999")

    ax.set_xlabel("Total parameters (non-emb + emb)", fontsize=12)
    ax.set_ylabel("Test accuracy", fontsize=12)
    ax.set_title("MNIST pixel-by-pixel — ablation: effect of Gram matrix M\n"
                 "(total params vs test accuracy, noise30 aug)", fontsize=13)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, which="both", alpha=0.18)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1%}"))
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    buckets = load_and_bucket()
    pareto_rows, fullH_rows, mean_rows, bin4_rows, mlp1_rows, mlp2_rows, reshape_rows = buckets

    print(f"Loaded rows: pareto={len(pareto_rows)}, fullH={len(fullH_rows)}, "
          f"mean={len(mean_rows)}, bin4={len(bin4_rows)}, "
          f"reshape={len(reshape_rows)}, mlp1={len(mlp1_rows)}, mlp2={len(mlp2_rows)}")

    fig = make_plot(*buckets)
    path = save_matplotlib_figure("ablation_mnist", fig, format="png", dpi=150)
    print(f"Saved: {path}")

    plt.rcParams["svg.fonttype"] = "none"
    fig2 = make_plot(*buckets)
    svg  = save_matplotlib_figure("ablation_mnist", fig2, format="svg")
    print(f"Saved: {svg}")
    plt.close("all")
