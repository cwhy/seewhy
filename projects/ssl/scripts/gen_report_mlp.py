"""
Generate MLP encoder comparison report.

Compares encoder architectures under EMA-JEPA training on MNIST:
  - Naive MLP (784→512→LATENT)
  - Deeper MLP (784→1024→1024→LATENT)
  - ConvNet (Conv32→Conv64→Conv128→GAP→LATENT)
  - Hebbian Gram + Hadamard readout (reference)

Plots:
  1. Architecture diagram (MLP vs MLP2 vs Conv vs Gram)
  2. Probe accuracy vs LATENT_DIM for all aug encoders
  3. MLP aug vs noaug (shallow vs deep)

Usage:
    uv run python projects/ssl/scripts/gen_report_mlp.py
"""

import json, sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch

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

def best_per_latent(exp_set):
    data = {}
    for r in rows:
        if r.get("experiment") in exp_set:
            lat, acc = r.get("latent_dim"), r.get("final_probe_acc")
            if lat and acc:
                if lat not in data or acc > data[lat]:
                    data[lat] = acc
    return data

mlp_aug_data    = best_per_latent({
    "exp_mlp_ema", "exp_mlp_ema_512", "exp_mlp_ema_1024",
    "exp_mlp_ema_2048", "exp_mlp_ema_4096"
})
mlp_noaug_data  = best_per_latent({
    "exp_mlp_ema_noaug_512", "exp_mlp_ema_noaug_1024",
    "exp_mlp_ema_noaug_2048", "exp_mlp_ema_noaug_4096"
})
mlp2_aug_data   = best_per_latent({
    "exp_mlp2_ema", "exp_mlp2_ema_512", "exp_mlp2_ema_1024",
    "exp_mlp2_ema_2048", "exp_mlp2_ema_4096"
})
conv_aug_data   = best_per_latent({
    "exp_conv_ema", "exp_conv_ema_512", "exp_conv_ema_1024",
    "exp_conv_ema_2048", "exp_conv_ema_4096"
})
gram_aug_data   = best_per_latent({
    "exp_ema", "exp_ema_512", "exp_ema_1024", "exp_ema_2048"
})
gram_noaug_data = best_per_latent({
    "exp_ema_noaug_512", "exp_ema_noaug_1024",
    "exp_ema_noaug_2048", "exp_ema_noaug_4096"
})

print("MLP   aug:  ", {k: f"{v:.3f}" for k, v in sorted(mlp_aug_data.items())})
print("MLP   noaug:", {k: f"{v:.3f}" for k, v in sorted(mlp_noaug_data.items())})
print("MLP2  aug:  ", {k: f"{v:.3f}" for k, v in sorted(mlp2_aug_data.items())})
print("Conv  aug:  ", {k: f"{v:.3f}" for k, v in sorted(conv_aug_data.items())})
print("Gram  aug:  ", {k: f"{v:.3f}" for k, v in sorted(gram_aug_data.items())})
print("Gram  noaug:", {k: f"{v:.3f}" for k, v in sorted(gram_noaug_data.items())})

# ── Diagram: Encoder architecture comparison ──────────────────────────────────

fig, axes = plt.subplots(1, 4, figsize=(18, 5.5))

def box(ax, x, y, w, h, label, color, fontsize=8.5, text_color="white"):
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="round,pad=0.03", linewidth=1.3,
                          edgecolor="#555", facecolor=color)
    ax.add_patch(rect)
    ax.text(x, y, label, ha="center", va="center", fontsize=fontsize,
            color=text_color, fontweight="bold", multialignment="center")

def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="#555", lw=1.4))

# ── Shallow MLP ───────────────────────────────────────────────────────────────
ax = axes[0]
ax.set_xlim(0, 1); ax.set_ylim(-0.05, 1.05); ax.axis("off")
ax.set_title("Shallow MLP", fontsize=11, fontweight="bold", pad=8)
box(ax, 0.5, 0.90, 0.55, 0.10, "Input x  (784,)", "#6b7280")
arrow(ax, 0.5, 0.85, 0.5, 0.75)
box(ax, 0.5, 0.70, 0.65, 0.10, "Linear(784→512) + GeLU", "#3b82f6")
arrow(ax, 0.5, 0.65, 0.5, 0.55)
box(ax, 0.5, 0.50, 0.65, 0.10, "Linear(512→LATENT)", "#3b82f6")
arrow(ax, 0.5, 0.45, 0.5, 0.35)
box(ax, 0.5, 0.30, 0.55, 0.10, "Embedding  (LATENT,)", "#1d4ed8")
ax.text(0.5, 0.13, "1 hidden layer\n~400K–4M params", ha="center",
        fontsize=8, color="#6b7280", style="italic")

# ── Deeper MLP ────────────────────────────────────────────────────────────────
ax = axes[1]
ax.set_xlim(0, 1); ax.set_ylim(-0.05, 1.05); ax.axis("off")
ax.set_title("Deeper MLP", fontsize=11, fontweight="bold", pad=8)
box(ax, 0.5, 0.90, 0.55, 0.10, "Input x  (784,)", "#6b7280")
arrow(ax, 0.5, 0.85, 0.5, 0.75)
box(ax, 0.5, 0.70, 0.68, 0.10, "Linear(784→1024) + GeLU", "#f59e0b", text_color="white")
arrow(ax, 0.5, 0.65, 0.5, 0.55)
box(ax, 0.5, 0.50, 0.68, 0.10, "Linear(1024→1024) + GeLU", "#f59e0b", text_color="white")
arrow(ax, 0.5, 0.45, 0.5, 0.35)
box(ax, 0.5, 0.30, 0.65, 0.10, "Linear(1024→LATENT)", "#d97706", text_color="white")
arrow(ax, 0.5, 0.25, 0.5, 0.15)
box(ax, 0.5, 0.10, 0.55, 0.10, "Embedding  (LATENT,)", "#1d4ed8")
ax.text(0.5, -0.02, "2 hidden layers\n~3M–10M params", ha="center",
        fontsize=8, color="#6b7280", style="italic")

# ── ConvNet ───────────────────────────────────────────────────────────────────
ax = axes[2]
ax.set_xlim(0, 1); ax.set_ylim(-0.05, 1.05); ax.axis("off")
ax.set_title("ConvNet", fontsize=11, fontweight="bold", pad=8)
box(ax, 0.5, 0.92, 0.60, 0.09, "Input x  (B,28,28,1)", "#6b7280")
arrow(ax, 0.5, 0.88, 0.5, 0.80)
box(ax, 0.5, 0.75, 0.68, 0.09, "Conv(3×3, 32ch) + GeLU", "#16a34a")
arrow(ax, 0.5, 0.71, 0.5, 0.63)
box(ax, 0.5, 0.58, 0.58, 0.09, "MaxPool(2×2)  →14×14", "#15803d")
arrow(ax, 0.5, 0.54, 0.5, 0.46)
box(ax, 0.5, 0.41, 0.68, 0.09, "Conv(3×3, 64ch) + GeLU", "#16a34a")
arrow(ax, 0.5, 0.37, 0.5, 0.29)
box(ax, 0.5, 0.24, 0.58, 0.09, "MaxPool(2×2)  →7×7", "#15803d")
arrow(ax, 0.5, 0.20, 0.5, 0.12)
box(ax, 0.5, 0.07, 0.70, 0.09, "Conv(3×3,128ch)+GeLU+GAP", "#16a34a")
ax.text(0.5, -0.02, "→ Linear(128→LATENT)", ha="center",
        fontsize=8, color="#555", fontweight="bold")

# ── Gram + Hadamard ───────────────────────────────────────────────────────────
ax = axes[3]
ax.set_xlim(0, 1); ax.set_ylim(-0.05, 1.05); ax.axis("off")
ax.set_title("Gram + Hadamard", fontsize=11, fontweight="bold", pad=8)
box(ax, 0.5, 0.92, 0.55, 0.09, "Input x  (784,)", "#6b7280")
arrow(ax, 0.5, 0.88, 0.5, 0.80)
box(ax, 0.5, 0.75, 0.72, 0.09, "x_norm[p]×W_enc[p]→F  (784×32)", "#7c3aed")
arrow(ax, 0.5, 0.71, 0.5, 0.63)
box(ax, 0.5, 0.58, 0.65, 0.09, "H = FᵀF  →  (1024,)", "#7c3aed")
arrow(ax, 0.5, 0.54, 0.5, 0.46)
box(ax, 0.28, 0.38, 0.40, 0.09, "H @ W_A", "#ec4899")
box(ax, 0.72, 0.38, 0.40, 0.09, "H @ W_B", "#ec4899")
ax.annotate("", xy=(0.28, 0.34), xytext=(0.5, 0.42),
            arrowprops=dict(arrowstyle="->", color="#555", lw=1.2))
ax.annotate("", xy=(0.72, 0.34), xytext=(0.5, 0.42),
            arrowprops=dict(arrowstyle="->", color="#555", lw=1.2))
ax.annotate("", xy=(0.5, 0.22), xytext=(0.28, 0.34),
            arrowprops=dict(arrowstyle="->", color="#555", lw=1.2))
ax.annotate("", xy=(0.5, 0.22), xytext=(0.72, 0.34),
            arrowprops=dict(arrowstyle="->", color="#555", lw=1.2))
ax.text(0.5, 0.27, "⊙", ha="center", va="center", fontsize=15, color="#d97706")
box(ax, 0.5, 0.15, 0.60, 0.09, "Embedding  a⊙b  (LATENT,)", "#1d4ed8")
ax.text(0.5, -0.02, "Co-activation prior", ha="center",
        fontsize=8, color="#6b7280", style="italic")

fig.suptitle("Encoder Architectures — EMA-JEPA on MNIST (same predictor + training)",
             fontsize=12, fontweight="bold", y=1.01)
fig.tight_layout()
url_arch = save_matplotlib_figure("mlp_vs_gram_architecture", fig, format="svg")
plt.close(fig)
print(f"Architecture diagram → {url_arch}")

# ── Plot 1: Accuracy vs LATENT_DIM (all aug) ──────────────────────────────────

fig, ax = plt.subplots(figsize=(9, 5.5))

def plot_line(data, label, color, marker, ls="-"):
    if not data:
        return
    xs = sorted(data)
    ys = [data[x] * 100 for x in xs]
    ax.plot(xs, ys, color=color, marker=marker, linewidth=2,
            markersize=7, label=label, linestyle=ls)
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=7.5, color=color)

plot_line(gram_aug_data,  "Gram+Hadamard + EMA",  "#7c3aed", "D")
plot_line(conv_aug_data,  "ConvNet + EMA",         "#16a34a", "^")
plot_line(mlp2_aug_data,  "Deeper MLP + EMA",      "#f59e0b", "s")
plot_line(mlp_aug_data,   "Shallow MLP + EMA",     "#3b82f6", "o")

ax.set_xlabel("Latent dimension", fontsize=11)
ax.set_ylabel("Linear probe accuracy (%)", fontsize=11)
ax.set_title("Probe accuracy vs latent size — encoder comparison\n(EMA training, 4 augmentation actions, MNIST)", fontsize=11)
ax.set_xscale("log", base=2)
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
all_latents = sorted(set(
    list(gram_aug_data) + list(conv_aug_data) +
    list(mlp2_aug_data) + list(mlp_aug_data)
))
if all_latents:
    ax.set_xticks(all_latents)
all_accs = [v * 100 for v in (
    list(gram_aug_data.values()) + list(conv_aug_data.values()) +
    list(mlp2_aug_data.values()) + list(mlp_aug_data.values())
)]
if all_accs:
    ax.set_ylim(max(50, min(all_accs) - 3), min(100, max(all_accs) + 3))
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
fig.tight_layout()
url_acc = save_matplotlib_figure("mlp_vs_gram_accuracy", fig, format="svg")
plt.close(fig)
print(f"Accuracy plot → {url_acc}")

# ── Plot 2: Shallow MLP and Deeper MLP aug vs noaug ───────────────────────────

fig, ax = plt.subplots(figsize=(8, 5))

plot_line(mlp_aug_data,   "Shallow MLP + aug",   "#3b82f6", "o")
plot_line(mlp_noaug_data, "Shallow MLP + no-aug","#93c5fd", "o", ls="--")
plot_line(mlp2_aug_data,  "Deeper MLP + aug",    "#f59e0b", "s")
plot_line(gram_aug_data,  "Gram+Hadamard + aug (ref)", "#7c3aed", "D", ls=":")
plot_line(gram_noaug_data,"Gram+Hadamard + no-aug (ref)", "#c4b5fd", "D", ls=":")

ax.set_xlabel("Latent dimension", fontsize=11)
ax.set_ylabel("Linear probe accuracy (%)", fontsize=11)
ax.set_title("MLP aug vs no-aug, and Gram reference\n(EMA training, MNIST)", fontsize=11)
ax.set_xscale("log", base=2)
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
all_latents2 = sorted(set(
    list(mlp_aug_data) + list(mlp_noaug_data) + list(mlp2_aug_data) +
    list(gram_aug_data) + list(gram_noaug_data)
))
if all_latents2:
    ax.set_xticks(all_latents2)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
fig.tight_layout()
url_aug_gap = save_matplotlib_figure("mlp_aug_gap", fig, format="svg")
plt.close(fig)
print(f"Aug gap plot → {url_aug_gap}")

# ── Assemble report ────────────────────────────────────────────────────────────

def fmt_table(data_dict):
    if not data_dict:
        return "*No results yet.*\n"
    md = "| LATENT_DIM | probe_acc |\n|---|---|\n"
    for k in sorted(data_dict):
        md += f"| {k} | {data_dict[k]:.1%} |\n"
    return md

def best_row(d):
    if not d:
        return "—", "—"
    k = max(d, key=d.get)
    return k, f"{d[k]:.1%}"

gram_best_lat,  gram_best_acc  = best_row(gram_aug_data)
conv_best_lat,  conv_best_acc  = best_row(conv_aug_data)
mlp2_best_lat,  mlp2_best_acc  = best_row(mlp2_aug_data)
mlp_best_lat,   mlp_best_acc   = best_row(mlp_aug_data)

md = f"""
# Encoder Architecture Comparison — EMA JEPA on MNIST

## Overview

Four encoder architectures compared under identical EMA-JEPA training
(τ=0.996, FiLM predictor, 4 augmentation actions, 200 epochs, MNIST).

---

## Architecture Diagrams

![]({url_arch})

---

## 1 · Probe Accuracy vs Latent Size (with augmentation)

![]({url_acc})

### Gram + Hadamard readout (reference)

{fmt_table(gram_aug_data)}

### ConvNet encoder

{fmt_table(conv_aug_data)}

### Deeper MLP (784→1024→1024→LATENT)

{fmt_table(mlp2_aug_data)}

### Shallow MLP (784→512→LATENT)

{fmt_table(mlp_aug_data)}

---

## 2 · Augmentation Sensitivity (MLP variants + Gram reference)

![]({url_aug_gap})

---

## 3 · Best Results Summary

| Encoder | Best LATENT | probe_acc | Notes |
|---|---|---|---|
| Gram+Hadamard | {gram_best_lat} | {gram_best_acc} | Co-activation prior |
| ConvNet | {conv_best_lat} | {conv_best_acc} | Spatial inductive bias |
| Deeper MLP | {mlp2_best_lat} | {mlp2_best_acc} | 2×1024 hidden, scales with latent |
| Shallow MLP | {mlp_best_lat} | {mlp_best_acc} | 512 hidden, saturates early |

---

*Generated by `projects/ssl/scripts/gen_report_mlp.py`*
"""

url_report = save_report("ssl_report_mlp", md, title="Encoder Comparison — EMA JEPA MNIST")
print(f"\nReport → {url_report}")
