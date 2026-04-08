"""Generate chain distillation report: direct vs 2-hop comparison."""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from shared_lib.media import save_matplotlib_figure

SSL          = Path(__file__).parent.parent
CHAIN_JSONL  = SSL / "results_distill_chain.jsonl"
DAE_JSONL    = SSL / "results_distill_heatmap_dae.jsonl"
REPORT       = SSL / "report-distill-chain.md"

ARCHS = ["mlp", "mlp2", "conv", "conv2", "vit", "vit_patch", "gram_dual", "gram_single", "gram_glu"]
ARCH_LABELS = {
    "mlp": "MLP", "mlp2": "MLP-2", "conv": "Conv", "conv2": "Conv-2",
    "vit": "ViT", "vit_patch": "ViT-patch",
    "gram_dual": "Gram-dual", "gram_single": "Gram-single", "gram_glu": "Gram-GLU",
}

# ── Load results ───────────────────────────────────────────────────────────────

chain_rows = [json.loads(l) for l in open(CHAIN_JSONL)]
dae_rows   = [json.loads(l) for l in open(DAE_JSONL)]

# L2 MLP-2 result (hop=1)
l2_row  = next(r for r in chain_rows if r.get("hop") == 1)
l2_probe = l2_row["probe_acc"] * 100
l2_knn   = l2_row["knn_1nn_acc"] * 100

# L3 sub-students (hop=2)
chain = {r["sub_student_arch"]: r for r in chain_rows if r.get("hop") == 2}

# Direct: DAE Conv-2 -> sub-student
direct = {r["student_arch"]: r for r in dae_rows if r["teacher_arch"] == "conv2"}

# ── Bar chart comparison ───────────────────────────────────────────────────────

arch_order = [a for a in ARCHS if a in chain and a in direct]
labels     = [ARCH_LABELS[a] for a in arch_order]

direct_probe = [direct[a]["probe_acc"] * 100 for a in arch_order]
chain_probe  = [chain[a]["probe_acc"]  * 100 for a in arch_order]
direct_knn   = [direct[a]["knn_1nn_acc"] * 100 for a in arch_order]
chain_knn    = [chain[a]["knn_1nn_acc"]  * 100 for a in arch_order]

x = np.arange(len(arch_order))
w = 0.35

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, d_vals, c_vals, metric in zip(
        axes,
        [direct_probe, direct_knn],
        [chain_probe, chain_knn],
        ["Linear Probe Accuracy (%)", "1-NN KNN Accuracy (%)"]):
    bars_d = ax.bar(x - w/2, d_vals, w, label="Direct (DAE Conv-2 → sub-student)",
                    color="#4C72B0", alpha=0.85)
    bars_c = ax.bar(x + w/2, c_vals, w, label="Chain (DAE Conv-2 → MLP-2 → sub-student)",
                    color="#DD8452", alpha=0.85)
    for bar, val in zip(bars_d, d_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{val:.1f}", ha="center", va="bottom", fontsize=7, color="#4C72B0")
    for bar, val in zip(bars_c, c_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{val:.1f}", ha="center", va="bottom", fontsize=7, color="#DD8452")
    ymin = min(min(d_vals), min(c_vals)) - 2
    ax.set_ylim(ymin, 100)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(metric, fontsize=10)
    ax.set_title(metric, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

fig.suptitle("Chain Distillation vs Direct: DAE Conv-2 teacher, MNIST", fontsize=12, fontweight="bold")
plt.tight_layout()
url_bar = save_matplotlib_figure("ssl_distill_chain_bar", fig, dpi=150)
plt.close(fig)
print(f"Bar chart → {url_bar}")

# ── Delta chart (chain - direct) ───────────────────────────────────────────────

delta_probe = [c - d for c, d in zip(chain_probe, direct_probe)]
delta_knn   = [c - d for c, d in zip(chain_knn, direct_knn)]

fig, ax = plt.subplots(figsize=(10, 4))
colors = ["#2ca02c" if d >= 0 else "#d62728" for d in delta_probe]
bars = ax.bar(x, delta_probe, color=colors, alpha=0.85, label="Probe Δ")
ax.bar(x, delta_knn, width=0.3, color=["#98df8a" if d >= 0 else "#ff9896" for d in delta_knn],
       alpha=0.7, label="KNN Δ")
for bar, dp, dk in zip(bars, delta_probe, delta_knn):
    ax.text(bar.get_x() + bar.get_width()/2,
            max(dp, dk) + 0.05 if max(dp, dk) >= 0 else min(dp, dk) - 0.2,
            f"{dp:+.1f}", ha="center", va="bottom" if dp >= 0 else "top",
            fontsize=8, fontweight="bold")
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
ax.set_ylabel("Δ accuracy (pp) — chain minus direct", fontsize=10)
ax.set_title("Chain − Direct: positive = chain wins", fontsize=11, fontweight="bold")
ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
url_delta = save_matplotlib_figure("ssl_distill_chain_delta", fig, dpi=150)
plt.close(fig)
print(f"Delta chart → {url_delta}")

# ── Write local report ─────────────────────────────────────────────────────────

lines = [
    "# Chain Distillation: DAE Conv-2 → MLP-2 → Sub-student",
    "",
    "**Hypothesis**: Can a sub-student learn better from an already-distilled MLP-2",
    "than directly from the original DAE Conv-2 teacher?",
    "",
    "**Chain**:",
    "- L1 teacher: DAE Conv-2 (frozen, 98.0% probe baseline)",
    "- L2 student: MLP-2 trained on L1 targets (best pair from heatmap)",
    f"- L2 result: probe={l2_probe:.1f}%  knn={l2_knn:.1f}%",
    "- L3 sub-students: all 9 archs trained on L2 (MLP-2) targets",
    "",
    "**Baseline**: Direct DAE Conv-2 → sub-student (from `results_distill_heatmap_dae.jsonl`).",
    "",
    "---",
    "",
    "## Results",
    "",
    f"![Bar chart]({url_bar})",
    "",
    f"![Delta chart]({url_delta})",
    "",
    "---",
    "",
    "## Comparison Table",
    "",
    "| Sub-student | Direct probe | Chain probe | Δ probe | Direct KNN | Chain KNN | Δ KNN |",
    "|-------------|-------------|-------------|---------|------------|-----------|-------|",
]
for a in arch_order:
    dp = direct[a]["probe_acc"] * 100
    cp = chain[a]["probe_acc"]  * 100
    dk = direct[a]["knn_1nn_acc"] * 100
    ck = chain[a]["knn_1nn_acc"]  * 100
    lines.append(
        f"| {ARCH_LABELS[a]} | {dp:.1f}% | {cp:.1f}% | {cp-dp:+.1f} pp "
        f"| {dk:.1f}% | {ck:.1f}% | {ck-dk:+.1f} pp |"
    )

gains   = [(a, chain[a]["probe_acc"]*100 - direct[a]["probe_acc"]*100) for a in arch_order]
winners = [(a, d) for a, d in gains if d > 0]
losers  = [(a, d) for a, d in gains if d <= 0]

lines += [
    "",
    "## Key Findings",
    "",
    f"**Chain wins on {len(winners)}/{len(arch_order)} sub-students** (probe accuracy):",
    "",
]
for a, d in sorted(winners, key=lambda x: -x[1]):
    lines.append(f"- {ARCH_LABELS[a]}: {d:+.1f} pp")

if losers:
    lines += ["", "**Direct wins on**:"]
    for a, d in sorted(losers, key=lambda x: x[1]):
        lines.append(f"- {ARCH_LABELS[a]}: {d:+.1f} pp")

avg_delta = np.mean([chain[a]["probe_acc"] - direct[a]["probe_acc"] for a in arch_order]) * 100
lines += [
    "",
    f"**Average Δ probe (chain − direct)**: {avg_delta:+.2f} pp",
    "",
    "### Interpretation",
    "",
]
if avg_delta > 0.2:
    lines.append("The chain path generally **improves** over direct distillation. "
                 "The MLP-2 intermediate acts as a useful 'translator' — "
                 "compressing Conv-2's spatial features into a dense MLP representation "
                 "that sub-students find easier to imitate.")
elif avg_delta < -0.2:
    lines.append("The chain path generally **hurts** vs direct distillation. "
                 "Each hop introduces noise/distortion; sub-students learn a degraded "
                 "copy of the original teacher signal.")
else:
    lines.append("The chain path is roughly **neutral** — the MLP-2 intermediate "
                 "neither helps nor hurts significantly. Information passes through "
                 "the chain without major loss or gain.")

REPORT.write_text("\n".join(lines) + "\n")
print(f"\nLocal report → {REPORT}")
