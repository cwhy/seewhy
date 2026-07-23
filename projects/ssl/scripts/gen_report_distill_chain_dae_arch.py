"""Generate DAE-seeded same-arch chain distillation report: hop 0-10 per architecture."""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from shared_lib.media import save_matplotlib_figure

SSL    = Path(__file__).parent.parent
JSONL  = SSL / "results_distill_chain_dae_arch.jsonl"
JSONL_RAND = SSL / "results_distill_chain_same_arch.jsonl"
REPORT = SSL / "report-distill-chain-dae-arch.md"

ARCHS = ["mlp2", "vit_patch", "gram_dual", "gram_single", "gram_glu"]
ARCH_LABELS = {
    "mlp2": "MLP-2", "vit_patch": "ViT-patch",
    "gram_dual": "Gram-dual", "gram_single": "Gram-single", "gram_glu": "Gram-GLU",
}
HOPS = list(range(11))

rows_dae  = [json.loads(l) for l in open(JSONL)  if json.loads(l).get("arch") in ARCHS]
rows_rand = [json.loads(l) for l in open(JSONL_RAND) if json.loads(l).get("arch") in ARCHS]

probe_dae  = {a: {} for a in ARCHS}
knn_dae    = {a: {} for a in ARCHS}
probe_rand = {a: {} for a in ARCHS}

for r in rows_dae:
    probe_dae[r["arch"]][r["hop"]] = r["probe_acc"] * 100
    knn_dae[r["arch"]][r["hop"]]   = r["knn_1nn_acc"] * 100
for r in rows_rand:
    probe_rand[r["arch"]][r["hop"]] = r["probe_acc"] * 100

# ── Line plot: DAE vs Random starting point across hops ───────────────────────

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

colors = plt.cm.tab10(np.linspace(0, 0.5, len(ARCHS)))

for ax, (probe_a, probe_b, title) in zip(axes, [
    (probe_dae, probe_rand, "DAE-seeded vs Random-seeded (Probe Accuracy)"),
    (knn_dae, None, "KNN Accuracy — DAE-seeded chain"),
]):
    for i, arch in enumerate(ARCHS):
        vals_a = [probe_a[arch].get(h, np.nan) for h in HOPS]
        ax.plot(HOPS, vals_a, marker="o", linewidth=2, markersize=6,
                color=colors[i], label=ARCH_LABELS[arch])
        ax.annotate(f"{vals_a[-1]:.1f}", (HOPS[-1], vals_a[-1]),
                    textcoords="offset points", xytext=(6, 0),
                    fontsize=8, va="center", color=colors[i])
        if probe_b is not None:
            vals_b = [probe_b[arch].get(h, np.nan) for h in HOPS]
            ax.plot(HOPS, vals_b, marker="s", linewidth=1.5, markersize=4,
                    color=colors[i], linestyle="--", alpha=0.5)

    ax.set_xticks(HOPS)
    ax.set_xticklabels(["H0\n(teacher)"] + [f"H{h}" for h in range(1, 11)], fontsize=8)
    ax.set_ylabel("Accuracy (%)", fontsize=10)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.grid(alpha=0.3)
    ymin = min(probe_a[a].get(0, 100) for a in ARCHS) - 1
    ax.set_ylim(ymin, 100)

axes[0].legend(fontsize=8, title="solid=DAE, dashed=random", loc="lower right")
axes[1].legend(fontsize=8, loc="lower right")

fig.suptitle("Same-Arch Chain Distillation: DAE-seeded vs Random-seeded — MNIST",
             fontsize=12, fontweight="bold")
plt.tight_layout()
url_line = save_matplotlib_figure("ssl_distill_chain_dae_line", fig, dpi=150)
plt.close(fig)
print(f"Line plot → {url_line}")

# ── Delta plot: DAE-seeded gain over DAE teacher (hop 0) ──────────────────────

fig, ax = plt.subplots(figsize=(13, 5))
for i, arch in enumerate(ARCHS):
    vals = [probe_dae[arch].get(h, np.nan) - probe_dae[arch].get(0, np.nan) for h in range(1, 11)]
    ax.plot(range(1, 11), vals, marker="o", linewidth=2, markersize=6,
            color=colors[i], label=ARCH_LABELS[arch])
    ax.annotate(f"{vals[-1]:+.1f}", (10, vals[-1]),
                textcoords="offset points", xytext=(5, 0), fontsize=8,
                va="center", color=colors[i])

ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(range(1, 11))
ax.set_xlabel("Hop", fontsize=10)
ax.set_ylabel("Δ probe vs DAE teacher / hop 0 (pp)", fontsize=10)
ax.set_title("Gain Over DAE Teacher (does distillation chain improve on its starting point?)",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout()
url_delta = save_matplotlib_figure("ssl_distill_chain_dae_delta", fig, dpi=150)
plt.close(fig)
print(f"Delta plot → {url_delta}")

# ── DAE vs Random comparison at each hop ──────────────────────────────────────

fig, ax = plt.subplots(figsize=(13, 5))
x = np.arange(len(ARCHS))
w = 0.35
for i, arch in enumerate(ARCHS):
    dae_h1  = probe_dae[arch].get(1, np.nan)
    rand_h1 = probe_rand[arch].get(1, np.nan)
    dae_h10  = probe_dae[arch].get(10, np.nan)
    rand_h10 = probe_rand[arch].get(10, np.nan)

bars_dae_h1  = [probe_dae[a].get(1, np.nan)  for a in ARCHS]
bars_rand_h1 = [probe_rand[a].get(1, np.nan) for a in ARCHS]
bars_dae_h10  = [probe_dae[a].get(10, np.nan)  for a in ARCHS]
bars_rand_h10 = [probe_rand[a].get(10, np.nan) for a in ARCHS]

ax.bar(x - w*1.5, bars_dae_h1,  w, label="DAE-seed hop 1",   color="#2196F3", alpha=0.85)
ax.bar(x - w*0.5, bars_rand_h1, w, label="Rand-seed hop 1",  color="#90CAF9", alpha=0.85)
ax.bar(x + w*0.5, bars_dae_h10,  w, label="DAE-seed hop 10",  color="#F44336", alpha=0.85)
ax.bar(x + w*1.5, bars_rand_h10, w, label="Rand-seed hop 10", color="#FFCDD2", alpha=0.85)

ax.set_xticks(x); ax.set_xticklabels([ARCH_LABELS[a] for a in ARCHS], fontsize=10)
ax.set_ylabel("Linear Probe Accuracy (%)", fontsize=10)
ax.set_title("DAE-seeded vs Random-seeded: Hop 1 and Hop 10", fontsize=11, fontweight="bold")
ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
ymin = min(min(bars_dae_h1), min(bars_rand_h1)) - 1
ax.set_ylim(ymin, 100)
for i, (dv, rv) in enumerate(zip(bars_dae_h1, bars_rand_h1)):
    ax.text(x[i] - w*1.5, dv + 0.1, f"{dv:.1f}", ha="center", fontsize=7)
    ax.text(x[i] - w*0.5, rv + 0.1, f"{rv:.1f}", ha="center", fontsize=7)
for i, (dv, rv) in enumerate(zip(bars_dae_h10, bars_rand_h10)):
    ax.text(x[i] + w*0.5, dv + 0.1, f"{dv:.1f}", ha="center", fontsize=7)
    ax.text(x[i] + w*1.5, rv + 0.1, f"{rv:.1f}", ha="center", fontsize=7)
plt.tight_layout()
url_bar = save_matplotlib_figure("ssl_distill_chain_dae_vs_rand", fig, dpi=150)
plt.close(fig)
print(f"Comparison bar → {url_bar}")

# ── Write local report ─────────────────────────────────────────────────────────

lines = [
    "# DAE-Seeded Same-Architecture Chain Distillation (10 Hops)",
    "",
    "**Setup**: Same as random-seeded chain, but hop 0 starts from a DAE-trained teacher",
    "instead of random init. Each hop's student distills from the previous hop's frozen params.",
    "",
    "- **Hop 0**: DAE-trained teacher — evaluated directly (no training in this script)",
    "- **Hops 1–10**: Each distills from the previous hop's frozen params (same arch)",
    "",
    "Architectures: " + ", ".join(ARCH_LABELS[a] for a in ARCHS),
    "",
    "---",
    "",
    "## Results",
    "",
    f"![Line plot]({url_line})",
    "*(solid lines = DAE-seeded, dashed lines = random-seeded for comparison)*",
    "",
    f"![Delta vs DAE teacher]({url_delta})",
    "",
    f"![DAE vs Random comparison]({url_bar})",
    "",
    "---",
    "",
    "## Full Results Table (Probe Accuracy %)",
    "",
    "| Architecture | H0 (DAE) | H1 | H2 | H3 | H4 | H5 | H6 | H7 | H8 | H9 | H10 | Δ H0→H10 |",
    "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
]
for a in ARCHS:
    h0   = probe_dae[a].get(0, float("nan"))
    vals = [probe_dae[a].get(h, float("nan")) for h in range(1, 11)]
    best = max(vals)
    delta = vals[-1] - h0
    row = f"| {ARCH_LABELS[a]} | {h0:.1f}% | "
    row += " | ".join(f"**{v:.1f}%**" if v == best else f"{v:.1f}%" for v in vals)
    row += f" | {delta:+.1f} pp |"
    lines.append(row)

lines += [
    "",
    "*(bold = peak value; Δ H0→H10 = net change from DAE teacher to final student)*",
    "",
    "## KNN Results",
    "",
    "| Architecture | H0 | H1 | H2 | H3 | H4 | H5 | H6 | H7 | H8 | H9 | H10 |",
    "|---|---|---|---|---|---|---|---|---|---|---|---|",
]
for a in ARCHS:
    vals = [knn_dae[a].get(h, float("nan")) for h in range(11)]
    lines.append(f"| {ARCH_LABELS[a]} | " + " | ".join(f"{v:.1f}%" for v in vals) + " |")

lines += [
    "",
    "## DAE-seeded vs Random-seeded Comparison (Probe %)",
    "",
    "| Architecture | DAE H0 | Rand H0 | DAE H1 | Rand H1 | DAE H10 | Rand H10 |",
    "|---|---|---|---|---|---|---|",
]
for a in ARCHS:
    d0  = probe_dae[a].get(0, float("nan"))
    r0  = probe_rand[a].get(0, float("nan"))
    d1  = probe_dae[a].get(1, float("nan"))
    r1  = probe_rand[a].get(1, float("nan"))
    d10 = probe_dae[a].get(10, float("nan"))
    r10 = probe_rand[a].get(10, float("nan"))
    lines.append(f"| {ARCH_LABELS[a]} | {d0:.1f}% | {r0:.1f}% | {d1:.1f}% | {r1:.1f}% | {d10:.1f}% | {r10:.1f}% |")

# Key findings
lines += ["", "## Key Findings", ""]

# Does chaining improve on DAE teacher?
LAST_HOP = 10
improvers = [(a, probe_dae[a].get(LAST_HOP,0) - probe_dae[a].get(0,0))
             for a in ARCHS if probe_dae[a].get(LAST_HOP,0) > probe_dae[a].get(0,0)]
degraders = [(a, probe_dae[a].get(LAST_HOP,0) - probe_dae[a].get(0,0))
             for a in ARCHS if probe_dae[a].get(LAST_HOP,0) <= probe_dae[a].get(0,0)]

lines.append(f"**Does the chain improve on the DAE teacher (hop 0)?** "
             f"{len(improvers)}/{len(ARCHS)} architectures gain at hop 10:")
lines.append("")
for a, d in sorted(improvers, key=lambda x: -x[1]):
    lines.append(f"- {ARCH_LABELS[a]}: {d:+.1f} pp  "
                 f"(DAE={probe_dae[a].get(0,0):.1f}% → hop10={probe_dae[a].get(LAST_HOP,0):.1f}%)")
if degraders:
    lines.append("")
    lines.append("Worse than DAE teacher at hop 10:")
    for a, d in sorted(degraders, key=lambda x: x[1]):
        lines.append(f"- {ARCH_LABELS[a]}: {d:+.1f} pp  "
                     f"(DAE={probe_dae[a].get(0,0):.1f}% → hop10={probe_dae[a].get(LAST_HOP,0):.1f}%)")

# DAE vs Random at hop 1
lines += ["", "**DAE-seeded vs Random-seeded at hop 1** (how much does a better start help?):"]
for a in ARCHS:
    d1 = probe_dae[a].get(1,0)
    r1 = probe_rand[a].get(1,0)
    diff = d1 - r1
    lines.append(f"- {ARCH_LABELS[a]}: DAE {d1:.1f}% vs Random {r1:.1f}% ({diff:+.1f} pp)")

# Does convergence happen faster?
lines += ["", "**Peak hop per architecture (DAE-seeded)**:"]
for a in ARCHS:
    vals = [(h, probe_dae[a].get(h, float("nan"))) for h in range(1, LAST_HOP + 1)]
    best_h, best_v = max(vals, key=lambda x: x[1])
    lines.append(f"- {ARCH_LABELS[a]}: hop {best_h} ({best_v:.1f}%)")

REPORT.write_text("\n".join(lines) + "\n")
print(f"\nLocal report → {REPORT}")
