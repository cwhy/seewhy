"""
Generate full SSL experiment report.

Plots:
  1. Probe accuracy vs LATENT_DIM: with-aug (SIGReg), no-aug (SIGReg), EMA
  2. SIGReg diagnostic loss vs LATENT_DIM for EMA encoder
  3. SIGReg diagnostic loss vs epoch for each EMA latent size

Uploads all figures and assembles an HTML report via shared_lib.report.

Usage:
    uv run python projects/ssl/scripts/gen_report.py
"""

import json, sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from shared_lib.media import save_matplotlib_figure
from shared_lib.report import save_report

JSONL = Path(__file__).parent.parent / "results.jsonl"

# ── Load results ───────────────────────────────────────────────────────────────

rows = []
with open(JSONL) as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))

# ── Helpers ────────────────────────────────────────────────────────────────────

def best_per_latent(exps):
    """Return {latent_dim: best_probe_acc} for given experiment names."""
    data = {}
    for r in rows:
        if r.get("experiment") in exps:
            lat = r.get("latent_dim")
            acc = r.get("final_probe_acc")
            if lat and acc:
                if lat not in data or acc > data[lat]:
                    data[lat] = acc
    return data

def sigreg_diag_by_latent(prefix):
    """Return {latent_dim: final_sigreg_diag} for exp names starting with prefix."""
    data = {}
    for r in rows:
        if r.get("experiment", "").startswith(prefix) or r.get("experiment") == prefix.rstrip("_"):
            lat = r.get("latent_dim")
            curve = r.get("sigreg_diag_curve", [])
            if lat and curve:
                final = curve[-1][1] if isinstance(curve[-1], list) else curve[-1][1]
                if lat not in data or final < data[lat]:
                    data[lat] = final
    return data

def sigreg_curves_by_latent(prefix):
    """Return {latent_dim: [(epoch, sigreg), ...]} for plotting training curves."""
    data = {}
    for r in rows:
        exp = r.get("experiment", "")
        if exp.startswith(prefix) or exp == prefix.rstrip("_"):
            lat = r.get("latent_dim")
            curve = r.get("sigreg_diag_curve", [])
            if lat and curve:
                data[lat] = [(c[0], c[1]) for c in curve]
    return data

# ── Data ───────────────────────────────────────────────────────────────────────

# With augmentation: canonical Hadamard+FiLM series
aug_exps = {"exp10a", "exp11a", "exp13a", "exp15b", "exp16a"}
aug_data = best_per_latent(aug_exps)

# No augmentation
noaug_exps = {"exp17b", "exp_noaug_512", "exp_noaug_1024", "exp_noaug_2048"}
noaug_data = best_per_latent(noaug_exps)

# EMA
ema_exps = {"exp_ema", "exp_ema_512", "exp_ema_1024", "exp_ema_2048"}
ema_data = best_per_latent(ema_exps)

# SIGReg diagnostic for EMA
ema_sreg_final  = sigreg_diag_by_latent("exp_ema")
ema_sreg_curves = sigreg_curves_by_latent("exp_ema")

print("With aug:  ", {k: f"{v:.3f}" for k, v in sorted(aug_data.items())})
print("No aug:    ", {k: f"{v:.3f}" for k, v in sorted(noaug_data.items())})
print("EMA:       ", {k: f"{v:.3f}" for k, v in sorted(ema_data.items())})
print("SIGReg diag:", {k: f"{v:.3f}" for k, v in sorted(ema_sreg_final.items())})

# ── Plot 1: Accuracy vs LATENT_DIM ────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(7.5, 4.5))

def plot_acc_line(data, label, color, marker, ls="-"):
    if not data:
        return
    xs = sorted(data)
    ys = [data[x] * 100 for x in xs]
    ax.plot(xs, ys, color=color, marker=marker, linewidth=2,
            markersize=7, label=label, linestyle=ls)
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8, color=color)

plot_acc_line(aug_data,   "With augmentation + SIGReg", "#2563eb", "o")
plot_acc_line(noaug_data, "No augmentation + SIGReg",   "#dc2626", "s")
plot_acc_line(ema_data,   "With augmentation + EMA",    "#16a34a", "^")

ax.set_xlabel("Latent dimension", fontsize=11)
ax.set_ylabel("Linear probe accuracy (%)", fontsize=11)
ax.set_title("Probe accuracy vs latent size\n(Hebbian gram + Hadamard readout, MNIST)", fontsize=11)
ax.set_xscale("log", base=2)
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
all_latents = sorted(set(list(aug_data) + list(noaug_data) + list(ema_data)))
ax.set_xticks(all_latents)
all_accs = [v * 100 for v in list(aug_data.values()) + list(noaug_data.values()) + list(ema_data.values())]
if all_accs:
    ax.set_ylim(max(50, min(all_accs) - 3), min(100, max(all_accs) + 2))
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
fig.tight_layout()
url_acc = save_matplotlib_figure("ssl_accuracy_vs_latent", fig, format="svg")
plt.close(fig)
print(f"Plot 1 → {url_acc}")

# ── Plot 2: SIGReg diagnostic vs LATENT_DIM for EMA ──────────────────────────

fig, ax = plt.subplots(figsize=(6, 4))

if ema_sreg_final:
    xs = sorted(ema_sreg_final)
    ys = [ema_sreg_final[x] for x in xs]
    ax.plot(xs, ys, color="#16a34a", marker="^", linewidth=2, markersize=7)
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9, color="#16a34a")
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.set_xticks(xs)

ax.set_xlabel("Latent dimension", fontsize=11)
ax.set_ylabel("SIGReg diagnostic loss", fontsize=11)
ax.set_title("SIGReg loss (diagnostic) for EMA encoder vs latent size\n"
             "(lower = embeddings closer to N(0,I))", fontsize=10)
ax.grid(True, alpha=0.3)
fig.tight_layout()
url_sreg_lat = save_matplotlib_figure("ssl_sigreg_diag_vs_latent", fig, format="svg")
plt.close(fig)
print(f"Plot 2 → {url_sreg_lat}")

# ── Plot 3: SIGReg diagnostic curves per EMA latent size ─────────────────────

fig, ax = plt.subplots(figsize=(7.5, 4.5))
colors = {512: "#f59e0b", 1024: "#3b82f6", 2048: "#8b5cf6", 4096: "#16a34a"}

for lat, curve in sorted(ema_sreg_curves.items()):
    epochs = [c[0] for c in curve]
    vals   = [c[1] for c in curve]
    ax.plot(epochs, vals, color=colors.get(lat, "gray"), linewidth=1.8,
            label=f"LATENT={lat}")

ax.set_xlabel("Epoch", fontsize=11)
ax.set_ylabel("SIGReg diagnostic loss", fontsize=11)
ax.set_title("SIGReg diagnostic over training — EMA encoder\n"
             "(not in loss; measures how Gaussian embeddings are)", fontsize=10)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
fig.tight_layout()
url_sreg_curve = save_matplotlib_figure("ssl_sigreg_diag_curves", fig, format="svg")
plt.close(fig)
print(f"Plot 3 → {url_sreg_curve}")

# ── Assemble report ────────────────────────────────────────────────────────────

def fmt_table(data_dict, label):
    if not data_dict:
        return f"*No {label} results yet.*\n"
    rows_md = "| LATENT_DIM | probe_acc |\n|---|---|\n"
    for k in sorted(data_dict):
        rows_md += f"| {k} | {data_dict[k]:.1%} |\n"
    return rows_md

md = f"""
# SSL Experiment Report — Hebbian Gram + Hadamard Readout on MNIST

## Summary

Experiments tuning a JEPA-style self-supervised model on MNIST using a
Hebbian gram encoder with Hadamard readout. Two collapse-prevention strategies
compared: **SIGReg** (Epps-Pulley characteristic function regularizer) and
**EMA** (momentum target encoder).

---

## 1 · Probe Accuracy vs Latent Size

![]({url_acc})

### With augmentation + SIGReg

{fmt_table(aug_data, "with-aug")}

### No augmentation + SIGReg

{fmt_table(noaug_data, "no-aug")}

### With augmentation + EMA

{fmt_table(ema_data, "EMA")}

**Key finding:** EMA consistently outperforms SIGReg at the same latent size.
No augmentation degrades quality substantially (~6 pp gap at LATENT=4096).

---

## 2 · SIGReg Diagnostic for EMA Encoder

![]({url_sreg_lat})

Even though EMA training uses no SIGReg loss, we compute the SIGReg statistic
as a diagnostic to measure how Gaussian the learned embeddings are.
Lower = closer to N(0, I).

![]({url_sreg_curve})

---

## 3 · Best Results

| Experiment | Config | probe_acc |
|---|---|---|
| exp16a | SIGReg, 4 actions, LATENT=4096 | {aug_data.get(4096, float('nan')):.1%} |
| exp_ema | EMA, 4 actions, LATENT=4096 | {ema_data.get(4096, float('nan')):.1%} |
| exp17b | No aug, SIGReg, LATENT=4096 | {noaug_data.get(4096, float('nan')):.1%} |

---

*Generated by `projects/ssl/scripts/gen_report.py`*
"""

url_report = save_report("ssl_report", md, title="SSL Experiments — MNIST")
print(f"\nReport → {url_report}")
