"""
Generate SSL report v2.

New in v2:
  - Architecture diagram (encoder structure, no training details)
  - EMA training schematic
  - SIGReg training schematic
  - EMA + no-aug line added to comparison plot

Usage:
    uv run python projects/ssl/scripts/gen_report_v2.py
"""

import json, sys
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

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

aug_data       = best_per_latent({"exp10a", "exp11a", "exp13a", "exp15b", "exp16a"})
noaug_data     = best_per_latent({"exp17b", "exp_noaug_512", "exp_noaug_1024", "exp_noaug_2048"})
ema_data       = best_per_latent({"exp_ema", "exp_ema_512", "exp_ema_1024", "exp_ema_2048", "exp_ema_diag"})
ema_noaug_data = best_per_latent({"exp_ema_noaug_512", "exp_ema_noaug_1024",
                                   "exp_ema_noaug_2048", "exp_ema_noaug_4096"})

# ── Diagram helpers ────────────────────────────────────────────────────────────

def box(ax, x, y, w, h, label, color="#dbeafe", textcolor="#1e3a5f",
        fontsize=9, bold=False, radius=0.04):
    patch = FancyBboxPatch((x - w/2, y - h/2), w, h,
                           boxstyle=f"round,pad=0,rounding_size={radius}",
                           facecolor=color, edgecolor="#94a3b8", linewidth=1.2)
    ax.add_patch(patch)
    weight = "bold" if bold else "normal"
    ax.text(x, y, label, ha="center", va="center", fontsize=fontsize,
            color=textcolor, weight=weight, wrap=True,
            multialignment="center")

def arrow(ax, x0, y0, x1, y1, color="#475569", lw=1.5, style="->"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, connectionstyle="arc3,rad=0"))

def label_edge(ax, x, y, text, fontsize=7.5, color="#64748b"):
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            color=color, style="italic")

# ── Diagram 1: Architecture ────────────────────────────────────────────────────

def make_arch_diagram():
    fig, ax = plt.subplots(figsize=(12, 3.6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 3.6); ax.axis("off")

    BH = 0.72; BW = 1.5; MID = 1.8; STEP = 1.75

    xs = [0.9, 0.9 + STEP, 0.9 + 2*STEP, 0.9 + 3*STEP,
          0.9 + 4*STEP, 0.9 + 5*STEP, 0.9 + 5.9*STEP]
    y  = MID

    # Boxes
    box(ax, xs[0], y, BW, BH, "Input\nimage\n(28×28)", "#f0f9ff")
    box(ax, xs[1], y, BW, BH, "Positional\nfeatures\n(784, D=144)", "#e0f2fe")
    box(ax, xs[2], y, BW, BH, "Pixel\nfeatures f\n(784, DEC_RANK)", "#e0f2fe")
    box(ax, xs[3], y, BW, BH, "Hebbian gram\nH = FᵀF\n(DEC_RANK²)", "#ede9fe", "#3b0764")
    box(ax, xs[4], y, BW, BH, "Hadamard\na⊙b\n(LATENT_DIM)", "#ede9fe", "#3b0764")
    box(ax, xs[5], y, BW, BH, "Embedding\nh\n(LATENT_DIM)", "#dcfce7", "#14532d")
    box(ax, xs[6], y, BW*0.95, BH, "Predictor\n(MLP + FiLM)\n(LATENT_DIM)", "#fef9c3", "#713f12")

    # Arrows between boxes
    for i in range(len(xs)-1):
        x0 = xs[i] + BW/2; x1 = xs[i+1] - BW/2
        arrow(ax, x0, y, x1, y)

    # Sub-labels
    label_edge(ax, (xs[0]+xs[1])/2, y + 0.52, "row/col embed\n→ outer product")
    label_edge(ax, (xs[1]+xs[2])/2, y + 0.52, "W_pos_enc\n+ W_enc")
    label_edge(ax, (xs[2]+xs[3])/2, y + 0.52, "F[p]=x_norm[p]·f[p]\neinsum")
    label_edge(ax, (xs[3]+xs[4])/2, y + 0.52, "W_enc_A, W_enc_B\n(parallel)")
    label_edge(ax, (xs[5]+xs[6])/2, y + 0.52, "action emb\n(FiLM)")

    ax.set_title("Encoder + Predictor Architecture", fontsize=11, pad=6, weight="bold")
    fig.tight_layout(pad=0.4)
    return fig

# ── Diagram 2: EMA training schematic ─────────────────────────────────────────

def make_ema_diagram():
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.set_xlim(0, 9); ax.set_ylim(0, 4.2); ax.axis("off")

    BW, BH = 1.55, 0.65

    # Row positions
    TOP, BOT = 3.1, 1.5

    # Input boxes
    box(ax, 1.0, TOP, 1.2, BH, "x_aug\n(corrupted)", "#fef3c7")
    box(ax, 1.0, BOT, 1.2, BH, "x\n(clean)", "#fef3c7")

    # Encoder boxes
    box(ax, 3.1, TOP, BW, BH, "Online\nEncoder", "#dbeafe", bold=True)
    box(ax, 3.1, BOT, BW, BH, "Target\nEncoder\n(stop grad)", "#e2e8f0",
        textcolor="#475569", fontsize=8)

    # Predictor
    box(ax, 5.5, TOP, BW, BH, "Predictor\n(FiLM)", "#fef9c3", bold=True)

    # Embeddings / loss
    box(ax, 7.4, TOP, 1.1, BH, "ê_pred", "#dcfce7")
    box(ax, 7.4, BOT, 1.1, BH, "e_target", "#e2e8f0", textcolor="#475569")

    # MSE loss box
    box(ax, 8.4, (TOP+BOT)/2, 0.9, 0.5, "MSE\nloss", "#fee2e2", textcolor="#7f1d1d",
        fontsize=8, radius=0.06)

    # Main forward arrows
    arrow(ax, 1.6, TOP, 2.32, TOP)
    arrow(ax, 1.6, BOT, 2.32, BOT)
    arrow(ax, 3.88, TOP, 4.72, TOP)
    arrow(ax, 6.28, TOP, 6.84, TOP)
    arrow(ax, 3.88, BOT, 6.84, BOT)
    arrow(ax, 6.99, TOP, 7.94, TOP)
    arrow(ax, 6.99, BOT, 7.94, BOT)
    arrow(ax, 7.95, TOP, 7.95, (TOP+BOT)/2 + 0.25)
    arrow(ax, 7.95, BOT, 7.95, (TOP+BOT)/2 - 0.25)

    # EMA update arrow (curved, bottom)
    ax.annotate("", xy=(3.1, BOT - BH/2 - 0.08),
                xytext=(3.1, TOP - BH/2 - 0.08),
                arrowprops=dict(arrowstyle="->", color="#7c3aed", lw=1.8,
                                connectionstyle="arc3,rad=0.5"))
    ax.text(1.55, (TOP+BOT)/2 - 0.05,
            "EMA update\nτ·target + (1-τ)·online",
            ha="center", va="center", fontsize=7.5, color="#7c3aed", style="italic")

    ax.set_title("EMA Training Schematic", fontsize=11, pad=6, weight="bold")
    fig.tight_layout(pad=0.4)
    return fig

# ── Diagram 3: SIGReg training schematic ──────────────────────────────────────

def make_sigreg_diagram():
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.set_xlim(0, 9); ax.set_ylim(0, 4.2); ax.axis("off")

    BW, BH = 1.55, 0.65
    TOP, BOT = 3.1, 1.5

    # Input boxes
    box(ax, 1.0, TOP, 1.2, BH, "x_aug\n(corrupted)", "#fef3c7")
    box(ax, 1.0, BOT, 1.2, BH, "x\n(clean)", "#fef3c7")

    # Shared encoder (same weights, called twice)
    box(ax, 3.1, TOP, BW, BH, "Encoder\n(shared weights)", "#dbeafe", bold=True)
    box(ax, 3.1, BOT, BW, BH, "Encoder\n(shared weights)", "#dbeafe", bold=True)

    # Predictor
    box(ax, 5.5, TOP, BW, BH, "Predictor\n(FiLM)", "#fef9c3", bold=True)

    # Embeddings
    box(ax, 7.4, TOP, 1.1, BH, "ê_pred", "#dcfce7")
    box(ax, 7.4, BOT, 1.1, BH, "e_full", "#dcfce7")

    # MSE loss
    box(ax, 8.4, (TOP+BOT)/2, 0.9, 0.5, "MSE\nloss", "#fee2e2", textcolor="#7f1d1d",
        fontsize=8, radius=0.06)

    # SIGReg box
    box(ax, 5.0, BOT - 0.95, BW, BH*0.9,
        "SIGReg\n(match N(0,I))", "#ede9fe", textcolor="#3b0764", fontsize=8)

    # Regularization loss
    box(ax, 7.0, BOT - 0.95, 1.2, BH*0.9, "Reg\nloss", "#fee2e2",
        textcolor="#7f1d1d", fontsize=8, radius=0.06)

    # Main forward arrows
    arrow(ax, 1.6, TOP, 2.32, TOP)
    arrow(ax, 1.6, BOT, 2.32, BOT)
    arrow(ax, 3.88, TOP, 4.72, TOP)
    arrow(ax, 6.28, TOP, 6.84, TOP)
    arrow(ax, 3.88, BOT, 6.84, BOT)
    arrow(ax, 6.99, TOP, 7.94, TOP)
    arrow(ax, 6.99, BOT, 7.94, BOT)
    arrow(ax, 7.95, TOP, 7.95, (TOP+BOT)/2 + 0.25)
    arrow(ax, 7.95, BOT, 7.95, (TOP+BOT)/2 - 0.25)

    # SIGReg branch from e_full
    arrow(ax, 6.84, BOT, 5.78, BOT - 0.62)
    arrow(ax, 5.78, BOT - 0.95, 6.4, BOT - 0.95)

    ax.set_title("SIGReg Training Schematic", fontsize=11, pad=6, weight="bold")
    fig.tight_layout(pad=0.4)
    return fig

# ── Plot 4: Accuracy vs LATENT_DIM (all 4 lines) ──────────────────────────────

def make_acc_plot():
    fig, ax = plt.subplots(figsize=(8, 4.8))

    series = [
        (aug_data,       "With aug + SIGReg",    "#2563eb", "o", "-"),
        (noaug_data,     "No aug + SIGReg",       "#dc2626", "s", "--"),
        (ema_data,       "With aug + EMA",         "#16a34a", "^", "-"),
        (ema_noaug_data, "No aug + EMA",            "#d97706", "D", "--"),
    ]

    for data, label, color, marker, ls in series:
        if not data:
            continue
        xs = sorted(data)
        ys = [data[x] * 100 for x in xs]
        ax.plot(xs, ys, color=color, marker=marker, linewidth=2,
                markersize=7, label=label, linestyle=ls)
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=7.5, color=color)

    ax.set_xlabel("Latent dimension", fontsize=11)
    ax.set_ylabel("Linear probe accuracy (%)", fontsize=11)
    ax.set_title("Probe accuracy vs latent size\n(Hebbian gram + Hadamard readout, MNIST)", fontsize=11)
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    all_latents = sorted({k for d in [aug_data, noaug_data, ema_data, ema_noaug_data]
                          for k in d})
    ax.set_xticks(all_latents)
    all_accs = [v * 100 for d in [aug_data, noaug_data, ema_data, ema_noaug_data]
                for v in d.values()]
    if all_accs:
        ax.set_ylim(max(50, min(all_accs) - 3), min(100, max(all_accs) + 2))
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig

# ── SIGReg diagnostic plots (reuse from v1) ───────────────────────────────────

def sigreg_diag_by_latent():
    data = {}
    for r in rows:
        exp = r.get("experiment", "")
        if "ema" in exp and not "noaug" in exp:
            lat = r.get("latent_dim")
            curve = r.get("sigreg_diag_curve", [])
            if lat and curve:
                final = curve[-1][1]
                if lat not in data or final < data[lat]:
                    data[lat] = final
    return data

def make_sigreg_lat_plot():
    data = sigreg_diag_by_latent()
    fig, ax = plt.subplots(figsize=(6, 4))
    if data:
        xs = sorted(data); ys = [data[x] for x in xs]
        ax.plot(xs, ys, color="#16a34a", marker="^", linewidth=2, markersize=7)
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=9, color="#16a34a")
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.set_xticks(xs)
    ax.set_xlabel("Latent dimension", fontsize=11)
    ax.set_ylabel("SIGReg diagnostic loss", fontsize=11)
    ax.set_title("SIGReg loss (diagnostic) for EMA encoder vs latent size\n"
                 "(lower = closer to N(0,I))", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig

# ── Upload all figures ─────────────────────────────────────────────────────────

print("Generating figures...")
url_arch    = save_matplotlib_figure("ssl_arch_diagram",    make_arch_diagram(),   format="svg")
url_ema_sch = save_matplotlib_figure("ssl_ema_schematic",   make_ema_diagram(),    format="svg")
url_sig_sch = save_matplotlib_figure("ssl_sigreg_schematic",make_sigreg_diagram(), format="svg")
url_acc     = save_matplotlib_figure("ssl_accuracy_vs_latent_v2", make_acc_plot(), format="svg")
url_sreg    = save_matplotlib_figure("ssl_sigreg_diag_vs_latent_v2", make_sigreg_lat_plot(), format="svg")

for name, url in [("Architecture", url_arch), ("EMA schematic", url_ema_sch),
                  ("SIGReg schematic", url_sig_sch), ("Accuracy plot", url_acc),
                  ("SIGReg diag", url_sreg)]:
    print(f"  {name} → {url}")

# ── Format result tables ───────────────────────────────────────────────────────

def fmt_row(data, latent):
    v = data.get(latent)
    return f"{v:.1%}" if v else "—"

latents = [512, 1024, 2048, 4096]

table_rows = "| LATENT_DIM |" + " With aug +<br>SIGReg |" + \
             " No aug +<br>SIGReg |" + " With aug +<br>EMA |" + \
             " No aug +<br>EMA |\n"
table_rows += "|---" * 5 + "|\n"
for lat in latents:
    table_rows += (f"| {lat} | {fmt_row(aug_data, lat)} | {fmt_row(noaug_data, lat)}"
                   f" | {fmt_row(ema_data, lat)} | {fmt_row(ema_noaug_data, lat)} |\n")

# ── Assemble report ────────────────────────────────────────────────────────────

md = f"""
# SSL Experiments — Hebbian Gram + Hadamard Readout on MNIST

## Architecture

![]({url_arch})

The encoder maps a raw 28×28 image to a dense embedding through four stages:
positional features (row/col outer products), per-pixel weighted features,
a Hebbian gram matrix (H = FᵀF), and a Hadamard readout (a ⊙ b where
a, b are two linear projections of H). A FiLM-conditioned MLP predictor
maps the context embedding to the target embedding.

---

## Training Methods

### EMA (Exponential Moving Average)

![]({url_ema_sch})

The online encoder is trained by gradient descent. A target encoder, updated
as an EMA of the online encoder (τ = 0.996), provides stable regression targets
without collapse. No regularization loss is needed.

### SIGReg (Sketch Isotropic Gaussian Regularizer)

![]({url_sig_sch})

A single encoder (shared weights) is used for both context and target.
The Epps-Pulley test statistic penalises deviation of the full-image embeddings
from N(0, I) via the characteristic function, preventing representation collapse.

---

## Results: Probe Accuracy vs Latent Size

![]({url_acc})

{table_rows}

**Key findings:**
- EMA consistently outperforms SIGReg by ~0.4 pp at every latent size
- No augmentation degrades quality by ~6 pp (SIGReg) and ~1–2 pp (EMA) at large latents
- EMA is more robust to missing augmentation than SIGReg
- Both methods scale smoothly with LATENT_DIM — no saturation up to 4096

---

## SIGReg Diagnostic for EMA Encoder

![]({url_sreg})

The SIGReg statistic computed on EMA embeddings (not used in training) measures
how Gaussian the learned representations are. EMA at LATENT=512 produces highly
non-Gaussian embeddings (score ~188) while larger latents are better-behaved
(~50–60), suggesting EMA naturally regularises representations with capacity.

---

*Generated by `projects/ssl/scripts/gen_report_v2.py`*
"""

url_report = save_report("ssl_report_v2", md, title="SSL Experiments v2 — MNIST")
print(f"\nReport → {url_report}")
