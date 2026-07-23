"""
Generate the SVG plots for projects/universal-ar/report-progress.md, from
results.jsonl `history` curves. Run on the remote GPU server (has the venv +
R2 creds); this script itself is CPU-only.

Usage:
    uv run python projects/universal-ar/scripts/gen_report_progress.py
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shared_lib.media import save_matplotlib_figure

JSONL = Path(__file__).parent.parent / "results.jsonl"


def load_rows():
    rows = []
    with open(JSONL) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def find(rows, exp):
    for r in rows:
        if r.get("experiment") == exp:
            return r
    raise KeyError(exp)


def main():
    rows = load_rows()
    exp2 = find(rows, "exp2")
    exp3 = find(rows, "exp3")
    exp4 = find(rows, "exp4")
    exp5 = find(rows, "exp5")

    # ---- Plot 1 (PRIORITY): loss curves, exp2/exp3/exp4/exp5 --------------
    h2, h3, h4, h5 = (e["history"] for e in (exp2, exp3, exp4, exp5))
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(h3["step"], h3["loss"], label="exp3 (variant A, S=32)", color="#1f77b4")
    ax.plot(h4["step"], h4["loss"], label="exp4 (variant A+ref, S=48)", color="#2ca02c")
    ax.plot(h5["step"], h5["loss"], label="exp5 (variant A+ref, S=256)", color="#d62728")
    ax.plot(h2["step"], h2["loss"], label="exp2 (variant B, cross-completion, S=32)",
            color="#7f7f7f", linestyle="--")
    ax.set_yscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel("loss (log scale)")
    ax.set_title("Loss curves, exp2–exp5\nall converge/plateau by step ~1000–2000")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    url_loss = save_matplotlib_figure("universal-ar_loss_curves", fig, format="svg")
    plt.close(fig)
    print("loss curves:", url_loss)

    # ---- Plot 2: context-length trend (label_attn, ink_attn vs S) ---------
    S_vals = [32, 48, 256]
    label_attn_vals = [exp3["label_attn"], exp4["label_attn"], exp5["label_attn"]]
    ink_attn_vals = [exp3["ink_attn"], exp4["ink_attn"], exp5["ink_attn"]]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(S_vals, label_attn_vals, marker="o", color="#ff7f0e", label="label_attn (final)")
    ax.plot(S_vals, ink_attn_vals, marker="o", color="#2ca02c", label="ink_attn (final)")
    ax.axhline(0.10, color="#ff7f0e", linestyle=":", linewidth=1, label="label chance (0.10)")
    ax.axhline(0.031, color="#2ca02c", linestyle=":", linewidth=1, label="ink chance (0.031)")
    for x, y in zip(S_vals, label_attn_vals):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 8), fontsize=8, ha="center")
    for x, y in zip(S_vals, ink_attn_vals):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, -14), fontsize=8, ha="center")
    ax.set_xscale("log")
    ax.set_xticks(S_vals)
    ax.set_xticklabels([str(s) for s in S_vals])
    ax.set_xlabel("episode size S (samples/episode)")
    ax.set_ylabel("cross-completion accuracy (final)")
    ax.set_title("Longer in-context length hurts variant A's label generalization\n(exp3 S=32 → exp4 S=48 → exp5 S=256; self-recall-only training)")
    ax.set_ylim(0, 0.5)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    url_ctx = save_matplotlib_figure("universal-ar_context_length_trend", fig, format="svg")
    plt.close(fig)
    print("context-length trend:", url_ctx)

    # ---- Plot 3 (optional): exp4 representative metric curves -------------
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(h4["step"], h4["recall"], label="recall (self, test)", color="#1f77b4")
    ax.plot(h4["step"], h4["ref_recall"], label="ref_recall (self, test)", color="#9467bd")
    ax.plot(h4["step"], h4["label_attn"], label="label_attn (cross, test)", color="#ff7f0e")
    ax.plot(h4["step"], h4["ink_attn"], label="ink_attn (cross, test)", color="#2ca02c")
    ax.axhline(h4["bg"][-1], color="black", linestyle=":", linewidth=1, label=f"bg baseline ({h4['bg'][-1]:.2f})")
    ax.axhline(0.10, color="#ff7f0e", linestyle=":", linewidth=1, alpha=0.6)
    ax.axhline(0.031, color="#2ca02c", linestyle=":", linewidth=1, alpha=0.6)
    ax.set_xlabel("training step")
    ax.set_ylabel("accuracy")
    ax.set_title("exp4 (variant A+ref, S=48): metric curves, test set")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=8, loc="center right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    url_exp4 = save_matplotlib_figure("universal-ar_exp4_curves", fig, format="svg")
    plt.close(fig)
    print("exp4 curves:", url_exp4)

    print()
    print("PLOT_URL_LOSS=", url_loss)
    print("PLOT_URL_CTX=", url_ctx)
    print("PLOT_URL_EXP4=", url_exp4)


if __name__ == "__main__":
    main()
