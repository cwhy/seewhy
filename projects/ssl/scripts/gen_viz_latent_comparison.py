"""
Visualization: probe accuracy vs LATENT_DIM for augmented vs no-augmentation models.

Reads results.jsonl and plots two lines:
  - "with augmentation": best Hebbian+Hadamard result per latent size
  - "no augmentation":   exp17b variants (no input corruption, SIGReg only)

Usage:
    uv run python projects/ssl/scripts/gen_viz_latent_comparison.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from shared_lib.media import save_matplotlib_figure

JSONL = Path(__file__).parent.parent / "results.jsonl"

# ── Load results ───────────────────────────────────────────────────────────────

rows = []
with open(JSONL) as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))

# ── Collect data ───────────────────────────────────────────────────────────────

# "with augmentation": Hebbian+Hadamard experiments (encoder=hebbian-gram+hadamard-readout)
# Best probe_acc per latent_dim, keeping only the canonical line:
# exp10a(512), exp11a(1024), exp13a(2048), exp16a(4096) — single/multi noise+mask

aug_exps = {
    "exp10a": 512,
    "exp11a": 1024,
    "exp13a": 2048,
    "exp16a": 4096,
}

# "no augmentation": exp_noaug_* variants
noaug_prefix = "exp_noaug"

aug_data   = {}   # latent_dim → probe_acc
noaug_data = {}   # latent_dim → probe_acc

for r in rows:
    exp = r.get("experiment", "")
    acc = r.get("final_probe_acc")
    latent = r.get("latent_dim")
    if acc is None or latent is None:
        continue

    if exp in aug_exps:
        # keep highest if multiple entries
        if latent not in aug_data or acc > aug_data[latent]:
            aug_data[latent] = acc

    if exp.startswith(noaug_prefix) or exp == "exp17b":
        if latent not in noaug_data or acc > noaug_data[latent]:
            noaug_data[latent] = acc

print("With augmentation:")
for k in sorted(aug_data):
    print(f"  LATENT={k:5d}  acc={aug_data[k]:.1%}")

print("No augmentation:")
for k in sorted(noaug_data):
    print(f"  LATENT={k:5d}  acc={noaug_data[k]:.1%}")

# ── Plot ───────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(7, 4.5))

def plot_line(data, label, color, marker):
    if not data:
        return
    xs = sorted(data)
    ys = [data[x] * 100 for x in xs]
    ax.plot(xs, ys, color=color, marker=marker, linewidth=2,
            markersize=7, label=label)
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8.5, color=color)

plot_line(aug_data,   "With augmentation (noise σ=75 + masking)", "#2563eb", "o")
plot_line(noaug_data, "No augmentation (identity context)",        "#dc2626", "s")

ax.set_xlabel("Latent dimension", fontsize=11)
ax.set_ylabel("Linear probe accuracy (%)", fontsize=11)
ax.set_title("Effect of augmentation across latent sizes\n(Hebbian gram + Hadamard readout, SIGReg, MNIST)", fontsize=11)
ax.set_xscale("log", base=2)
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax.set_xticks(sorted(set(list(aug_data.keys()) + list(noaug_data.keys()))))

all_accs = [v * 100 for v in list(aug_data.values()) + list(noaug_data.values())]
if all_accs:
    ymin = max(50, min(all_accs) - 3)
    ymax = min(100, max(all_accs) + 3)
    ax.set_ylim(ymin, ymax)

ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
fig.tight_layout()

url = save_matplotlib_figure("ssl_latent_comparison", fig, format="svg")
print(f"Saved → {url}")
plt.close(fig)
