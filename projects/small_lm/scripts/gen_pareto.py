"""
Pareto-front plot: training time vs eval perplexity across all experiments.

A point is Pareto-optimal if no other experiment achieves both lower PPL
and shorter training time simultaneously.

Usage (standalone):
    uv run python projects/small_lm/scripts/gen_pareto.py

As a library:
    from scripts.gen_pareto import plot_pareto
    url = plot_pareto()   # returns R2 URL or local path
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

from shared_lib.media import save_matplotlib_figure

JSONL = Path(__file__).parent.parent / "results.jsonl"

# Label overrides: exp_name → display label
LABELS = {
    "exp_baseline": "baseline\n(AdamW, learned pos, weight-tied)",
    "exp_muon":     "muon\n(Muon optimizer)",
    "exp_rope":     "rope\n(RoPE, no learned pos embed)",
    "exp_no_tie":   "no_tie\n(separate LM head)",
}

COLORS = {
    "exp_baseline": "#4C72B0",
    "exp_muon":     "#DD8452",
    "exp_rope":     "#55A868",
    "exp_no_tie":   "#C44E52",
}


def load_results() -> list[dict]:
    rows = []
    if not JSONL.exists():
        return rows
    with open(JSONL) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    # De-duplicate: keep last result per experiment name
    seen, unique = {}, []
    for r in reversed(rows):
        k = r.get("experiment")
        if k and k not in seen:
            seen[k] = True
            unique.append(r)
    return unique


def pareto_front(points: list[tuple]) -> list[int]:
    """Return indices of Pareto-optimal points (min time AND min ppl)."""
    optimal = []
    for i, (ti, pi) in enumerate(points):
        dominated = any(
            tj <= ti and pj <= pi and (tj < ti or pj < pi)
            for j, (tj, pj) in enumerate(points) if j != i
        )
        if not dominated:
            optimal.append(i)
    return optimal


def plot_pareto(save_name: str = "small_lm_pareto") -> str:
    """Generate Pareto plot, upload to R2, return URL."""
    rows = load_results()
    if not rows:
        print("No results in results.jsonl — skipping Pareto plot.")
        return ""

    names  = [r["experiment"] for r in rows]
    times  = [r["time_s"] for r in rows]
    ppls   = [r["final_eval_ppl"] for r in rows]
    points = list(zip(times, ppls))

    pareto_idx = set(pareto_front(points))

    fig, ax = plt.subplots(figsize=(8, 5))

    # All points
    for i, (name, t, p) in enumerate(zip(names, times, ppls)):
        color  = COLORS.get(name, "#888888")
        marker = "*" if i in pareto_idx else "o"
        ms     = 14 if i in pareto_idx else 9
        zorder = 5 if i in pareto_idx else 4
        ax.scatter(t, p, color=color, s=ms**2, marker=marker, zorder=zorder,
                   edgecolors="white" if i in pareto_idx else "none", linewidths=1.2)
        label = LABELS.get(name, name)
        ax.annotate(label, (t, p),
                    textcoords="offset points", xytext=(8, 4),
                    fontsize=7.5, color=color)

    # Draw Pareto front line (sorted by time)
    if len(pareto_idx) > 1:
        px = [(times[i], ppls[i]) for i in sorted(pareto_idx, key=lambda i: times[i])]
        px_t, px_p = zip(*px)
        ax.step(px_t, px_p, where="post", color="#888888", lw=1.2,
                ls="--", alpha=0.6, zorder=3)

    ax.set_xlabel("Training time (s)")
    ax.set_ylabel("Eval perplexity (↓ better)")
    ax.set_title("small_lm — Pareto front: time vs perplexity")
    ax.grid(True, alpha=0.3)

    star_patch = mpatches.Patch(color="gray", label="★ Pareto-optimal")
    ax.legend(handles=[star_patch], fontsize=8, loc="upper right")

    fig.tight_layout()
    url = save_matplotlib_figure(save_name, fig, format="png", dpi=150)
    plt.close(fig)
    print(f"  pareto → {url}")
    return url


if __name__ == "__main__":
    url = plot_pareto()
    print(url)
