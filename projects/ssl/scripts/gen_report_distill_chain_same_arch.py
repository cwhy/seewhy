"""Generate same-arch chain distillation report: hop 0-3 per architecture."""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from shared_lib.media import save_matplotlib_figure

SSL    = Path(__file__).parent.parent
JSONL  = SSL / "results_distill_chain_same_arch.jsonl"
REPORT = SSL / "report-distill-chain-same-arch.md"

ARCHS = ["mlp2", "vit_patch", "gram_dual", "gram_single", "gram_glu"]
ARCH_LABELS = {
    "mlp2": "MLP-2", "vit_patch": "ViT-patch",
    "gram_dual": "Gram-dual", "gram_single": "Gram-single", "gram_glu": "Gram-GLU",
}
HOPS = list(range(11))
HOP_LABELS = ["Hop 0\n(random)"] + [f"Hop {h}" for h in range(1, 11)]
import matplotlib.cm as cm
COLORS = [cm.tab10(i/10) for i in range(11)]

rows = [json.loads(l) for l in open(JSONL) if json.loads(l).get("arch") in ARCHS]

# Build matrices: probe[arch_idx][hop], knn[arch_idx][hop]
probe = {a: {} for a in ARCHS}
knn   = {a: {} for a in ARCHS}
for r in rows:
    probe[r["arch"]][r["hop"]] = r["probe_acc"] * 100
    knn[r["arch"]][r["hop"]]   = r["knn_1nn_acc"] * 100

# ── Line plot: probe accuracy per arch across hops ────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, metric, data, ylabel in zip(
        axes,
        ["Probe", "KNN"],
        [probe, knn],
        ["Linear Probe Accuracy (%)", "1-NN KNN Accuracy (%)"]):
    for arch in ARCHS:
        vals = [data[arch].get(h, np.nan) for h in HOPS]
        ax.plot(HOPS, vals, marker="o", linewidth=2, markersize=7, label=ARCH_LABELS[arch])
        # annotate final value
        ax.annotate(f"{vals[-1]:.1f}", (HOPS[-1], vals[-1]),
                    textcoords="offset points", xytext=(6, 0),
                    fontsize=8, va="center")
    ax.set_xticks(HOPS)
    ax.set_xticklabels(HOP_LABELS, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(f"{metric} Accuracy per Hop", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    ymin = min(data[a].get(0, 100) for a in ARCHS) - 1
    ax.set_ylim(ymin, 100)

fig.suptitle("Same-Arch Chain Distillation: Random → Hop1 → Hop2 → Hop3 — MNIST",
             fontsize=12, fontweight="bold")
plt.tight_layout()
url_line = save_matplotlib_figure("ssl_distill_chain_same_line", fig, dpi=150)
plt.close(fig)
print(f"Line plot → {url_line}")

# ── Convergence plot: zoom in on hops 1-10 ────────────────────────────────────

fig, ax = plt.subplots(figsize=(11, 5))
for arch in ARCHS:
    vals = [probe[arch].get(h, np.nan) for h in range(1, 11)]
    ax.plot(range(1, 11), vals, marker="o", linewidth=2, markersize=6, label=ARCH_LABELS[arch])
    ax.annotate(f"{vals[-1]:.1f}", (10, vals[-1]),
                textcoords="offset points", xytext=(5, 0), fontsize=8, va="center")
ax.set_xticks(range(1, 11))
ax.set_xlabel("Hop", fontsize=10)
ax.set_ylabel("Linear Probe Accuracy (%)", fontsize=10)
ax.set_title("Probe Accuracy Hops 1–10 (same-arch chain)", fontsize=11, fontweight="bold")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
ymin = min(probe[a].get(1, 100) for a in ARCHS) - 1
ax.set_ylim(ymin, 100)
plt.tight_layout()
url_bar = save_matplotlib_figure("ssl_distill_chain_same_convergence", fig, dpi=150)
plt.close(fig)
print(f"Convergence plot → {url_bar}")

# ── Delta plot: each hop vs hop 1 ─────────────────────────────────────────────

x = np.arange(len(ARCHS))
w = 0.07
fig, ax = plt.subplots(figsize=(12, 4))
for hop in range(2, 11):
    vals = [probe[a].get(hop, np.nan) - probe[a].get(1, np.nan) for a in ARCHS]
    ax.bar(x + (hop - 6) * w, vals, w, label=f"Hop {hop}",
           color=cm.RdYlGn((hop - 1) / 10), alpha=0.85)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(x); ax.set_xticklabels([ARCH_LABELS[a] for a in ARCHS], fontsize=10)
ax.set_ylabel("Δ probe vs hop 1 (pp)", fontsize=10)
ax.set_title("Per-Hop Gain Over Hop 1 (does chaining help?)", fontsize=11, fontweight="bold")
ax.legend(fontsize=7, ncol=3); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
url_delta = save_matplotlib_figure("ssl_distill_chain_same_delta", fig, dpi=150)
plt.close(fig)
print(f"Delta chart → {url_delta}")

# ── Write local report ─────────────────────────────────────────────────────────

lines = [
    "# Same-Architecture Chain Distillation (10 Hops)",
    "",
    "**Setup**: For each architecture, train 10 successive distillation hops where each",
    "student learns from the previous hop's params as a frozen teacher (same arch).",
    "",
    "- **Hop 0**: Random init — evaluated directly, no training",
    "- **Hops 1–10**: Each distills from the previous hop's frozen params",
    "",
    "Architectures: " + ", ".join(ARCH_LABELS[a] for a in ARCHS),
    "(MLP, ViT, Conv, Conv-2 excluded — MLP/ViT weaker than counterparts;",
    "Conv/Conv-2 stuck at ~21% because random conv features are near-chance,",
    "making the chain unable to bootstrap from noise)",
    "",
    "---",
    "",
    "## Results",
    "",
    f"![Line plot]({url_line})",
    "",
    f"![Bar chart]({url_bar})",
    "",
    f"![Delta chart]({url_delta})",
    "",
    "---",
    "",
    "## Full Results Table (Probe Accuracy)",
    "",
    "| Architecture | H0 (rand) | H1 | H2 | H3 | H4 | H5 | H6 | H7 | H8 | H9 | H10 | Δ H1→H10 |",
    "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
]
for a in ARCHS:
    h0  = probe[a].get(0, float("nan"))
    vals = [probe[a].get(h, float("nan")) for h in range(1, 11)]
    best = max(vals)
    delta = vals[-1] - vals[0]
    row = f"| {ARCH_LABELS[a]} | {h0:.1f}% | "
    row += " | ".join(f"**{v:.1f}%**" if v == best else f"{v:.1f}%" for v in vals)
    row += f" | {delta:+.1f} pp |"
    lines.append(row)

lines += [
    "",
    "*(bold = peak value; Δ H1→H10 = whether chaining beyond hop 1 helps)*",
    "",
    "## KNN Results",
    "",
    "| Architecture | H0 | H1 | H2 | H3 | H4 | H5 | H6 | H7 | H8 | H9 | H10 |",
    "|---|---|---|---|---|---|---|---|---|---|---|---|",
]
for a in ARCHS:
    vals = [knn[a].get(h, float("nan")) for h in range(11)]
    lines.append(f"| {ARCH_LABELS[a]} | " + " | ".join(f"{v:.1f}%" for v in vals) + " |")

# Findings
lines += ["", "## Key Findings", ""]

# Does chaining help (hop10 vs hop1)?
LAST_HOP = 10
improvers = [(a, probe[a].get(LAST_HOP,0) - probe[a].get(1,0)) for a in ARCHS if probe[a].get(LAST_HOP,0) > probe[a].get(1,0)]
degraders = [(a, probe[a].get(LAST_HOP,0) - probe[a].get(1,0)) for a in ARCHS if probe[a].get(LAST_HOP,0) <= probe[a].get(1,0)]

lines.append(f"**Does extended chaining (hop 2–10) improve over hop 1?** "
             f"{len(improvers)}/{len(ARCHS)} architectures gain at hop 10:")
lines.append("")
for a, d in sorted(improvers, key=lambda x: -x[1]):
    lines.append(f"- {ARCH_LABELS[a]}: {d:+.1f} pp")
if degraders:
    lines.append("")
    lines.append("Hurt or flat by hop 10:")
    for a, d in sorted(degraders, key=lambda x: x[1]):
        lines.append(f"- {ARCH_LABELS[a]}: {d:+.1f} pp")

# Peak hop per arch
lines += ["", "**Peak hop per architecture**:"]
for a in ARCHS:
    vals = [(h, probe[a].get(h, float("nan"))) for h in range(1, LAST_HOP + 1)]
    best_h, best_v = max(vals, key=lambda x: x[1])
    lines.append(f"- {ARCH_LABELS[a]}: hop {best_h} ({best_v:.1f}%)")

# Big jump: hop0→hop1
lines += ["", "**Biggest jump from random → hop 1**:"]
jumps = sorted(ARCHS, key=lambda a: probe[a].get(1,0) - probe[a].get(0,0), reverse=True)
for a in jumps[:3]:
    d = probe[a].get(1,0) - probe[a].get(0,0)
    lines.append(f"- {ARCH_LABELS[a]}: {d:+.1f} pp  (random={probe[a].get(0,0):.1f}% → hop1={probe[a].get(1,0):.1f}%)")

REPORT.write_text("\n".join(lines) + "\n")
print(f"\nLocal report → {REPORT}")
