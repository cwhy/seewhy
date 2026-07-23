"""
Architecture comparison report: ConvNet, mini-ViT, Gram-single vs Gram+Hadamard.

Sections:
  1. Architecture diagrams (4 panels)
  2. Pseudocode block per encoder
  3. Accuracy vs latent size: all architectures + EMA Gram reference
  4. Best results table

Usage:
    uv run python projects/ssl/scripts/gen_report_arch.py
"""

import json, sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from shared_lib.media import save_matplotlib_figure
from shared_lib.report import save_report

JSONL = Path(__file__).parent.parent / "results.jsonl"

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

conv1_data  = best_per_latent({
    "exp_conv_ema", "exp_conv_ema_512", "exp_conv_ema_1024",
    "exp_conv_ema_2048", "exp_conv_ema_4096"
})
conv2_data  = best_per_latent({
    "exp_conv2_ema", "exp_conv2_ema_512", "exp_conv2_ema_1024",
    "exp_conv2_ema_2048", "exp_conv2_ema_4096"
})
vit_data    = best_per_latent({
    "exp_vit_ema", "exp_vit_ema_512", "exp_vit_ema_1024",
    "exp_vit_ema_2048", "exp_vit_ema_4096"
})
gram_data   = best_per_latent({
    "exp_ema", "exp_ema_512", "exp_ema_1024", "exp_ema_2048"
})
gramS_data  = best_per_latent({
    "exp_gram_single_ema", "exp_gram_single_ema_256",
    "exp_gram_single_ema_1024", "exp_gram_single_ema_4096"
})

print("ConvNet v1:  ", {k: f"{v:.3f}" for k, v in sorted(conv1_data.items())})
print("ConvNet v2:  ", {k: f"{v:.3f}" for k, v in sorted(conv2_data.items())})
print("ViT:         ", {k: f"{v:.3f}" for k, v in sorted(vit_data.items())})
print("Gram+Had:    ", {k: f"{v:.3f}" for k, v in sorted(gram_data.items())})
print("Gram-Single: ", {k: f"{v:.3f}" for k, v in sorted(gramS_data.items())})

# ── Diagram helpers ───────────────────────────────────────────────────────────

def box(ax, x, y, w, h, label, fc, ec="#444", fs=8, tc="white", bold=True):
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="round,pad=0.025", lw=1.3, ec=ec, fc=fc)
    ax.add_patch(rect)
    ax.text(x, y, label, ha="center", va="center", fontsize=fs,
            color=tc, fontweight="bold" if bold else "normal",
            multialignment="center")

def arr(ax, x1, y1, x2, y2, color="#555"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.4))

def note(ax, x, y, text):
    ax.text(x, y, text, ha="center", fontsize=7.5, color="#6b7280", style="italic")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1 — ConvNet architecture diagram
# ═══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(13, 6))
fig.suptitle("ConvNet Encoder — Architecture Comparison", fontsize=12, fontweight="bold")

C_GRAY  = "#6b7280"; C_BLUE  = "#2563eb"; C_GREEN = "#16a34a"
C_DARK  = "#1d4ed8"; C_POOL  = "#0f766e"; C_FC    = "#7c3aed"

for col, (ax, title, ch, fc_note) in enumerate(zip(axes,
    ["v1: channels [32, 64, 128]", "v2: channels [64, 128, 256] + FC"],
    [[32, 64, 128], [64, 128, 256]],
    ["Direct 128→LATENT (bottleneck)", "128→FC(512)→GeLU→LATENT (fixed)"]
)):
    ax.set_xlim(0, 1); ax.set_ylim(-0.05, 1.05); ax.axis("off")
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)

    c1, c2, c3 = ch
    ys = [0.93, 0.83, 0.73, 0.63, 0.53, 0.43, 0.33, 0.18]

    box(ax, 0.5, ys[0], 0.60, 0.09, "x  (B, 784)  → reshape (B,28,28,1)", C_GRAY)
    arr(ax, 0.5, ys[0]-0.045, 0.5, ys[1]+0.045)
    box(ax, 0.5, ys[1], 0.70, 0.09, f"Conv(3×3, {c1}ch, SAME) + GeLU  →  28×28×{c1}", C_BLUE)
    arr(ax, 0.5, ys[1]-0.045, 0.5, ys[2]+0.045)
    box(ax, 0.5, ys[2], 0.55, 0.09, f"MaxPool(2×2)  →  14×14×{c1}", C_POOL)
    arr(ax, 0.5, ys[2]-0.045, 0.5, ys[3]+0.045)
    box(ax, 0.5, ys[3], 0.70, 0.09, f"Conv(3×3, {c2}ch, SAME) + GeLU  →  14×14×{c2}", C_BLUE)
    arr(ax, 0.5, ys[3]-0.045, 0.5, ys[4]+0.045)
    box(ax, 0.5, ys[4], 0.55, 0.09, f"MaxPool(2×2)  →  7×7×{c2}", C_POOL)
    arr(ax, 0.5, ys[4]-0.045, 0.5, ys[5]+0.045)
    box(ax, 0.5, ys[5], 0.70, 0.09, f"Conv(3×3, {c3}ch, SAME) + GeLU  →  7×7×{c3}", C_BLUE)
    arr(ax, 0.5, ys[5]-0.045, 0.5, ys[6]+0.045)
    box(ax, 0.5, ys[6], 0.60, 0.09, f"Global Avg Pool  →  ({c3},)", C_GREEN)
    arr(ax, 0.5, ys[6]-0.045, 0.5, ys[7]+0.045)
    if col == 0:
        box(ax, 0.5, ys[7], 0.55, 0.09, "Linear  →  (LATENT,)", C_DARK)
        note(ax, 0.5, 0.07, fc_note)
    else:
        box(ax, 0.5, ys[7]+0.055, 0.60, 0.09, "Linear(FC_HIDDEN=512) + GeLU", C_FC)
        arr(ax, 0.5, ys[7]+0.01, 0.5, ys[7]-0.055)
        box(ax, 0.5, ys[7]-0.10, 0.55, 0.09, "Linear  →  (LATENT,)", C_DARK)
        note(ax, 0.5, 0.01, fc_note)

fig.tight_layout()
url_conv_diag = save_matplotlib_figure("arch_convnet_diagram", fig, format="svg")
plt.close(fig)
print(f"ConvNet diagram → {url_conv_diag}")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2 — mini ViT architecture diagram
# ═══════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(8, 9))
ax.set_xlim(0, 1); ax.set_ylim(-0.02, 1.02); ax.axis("off")
ax.set_title("Mini Vision Transformer Encoder\n(3 layers, D_MODEL=128, 4 heads, 49 patches)",
             fontsize=11, fontweight="bold", pad=10)

C_PATCH = "#0369a1"; C_ATTN = "#7c3aed"; C_FFN = "#15803d"
C_LN = "#b45309"; C_HEAD = "#1d4ed8"

ys = [0.96, 0.88, 0.81, 0.63, 0.45, 0.35, 0.22, 0.12]

box(ax, 0.5, ys[0], 0.62, 0.07, "x  (B, 784)  →  reshape (B, 28, 28)", C_GRAY)
arr(ax, 0.5, ys[0]-0.035, 0.5, ys[1]+0.035)
box(ax, 0.5, ys[1], 0.72, 0.07, "Split into 49 non-overlapping 4×4 patches  →  (B, 49, 16)", C_PATCH)
arr(ax, 0.5, ys[1]-0.035, 0.5, ys[2]+0.035)
box(ax, 0.5, ys[2], 0.72, 0.07, "PatchEmbed: Linear(16→128) + learned pos_emb(49×128)", C_PATCH)
arr(ax, 0.5, ys[2]-0.035, 0.5, ys[3]+0.035)

# Transformer block (show one, label ×3)
bx, by, bw, bh = 0.5, 0.63, 0.80, 0.18
rect = FancyBboxPatch((bx - bw/2, by - bh/2), bw, bh,
                      boxstyle="round,pad=0.02", lw=1.8, ec="#9333ea",
                      fc="#faf5ff", zorder=0)
ax.add_patch(rect)
ax.text(0.93, by + 0.02, "×3", fontsize=12, color="#9333ea", fontweight="bold", ha="center")

inner_ys = [by + 0.065, by + 0.015, by - 0.04, by - 0.09]
box(ax, 0.5, inner_ys[0], 0.65, 0.055, "LayerNorm  (pre-norm)", C_LN, fs=7.5)
arr(ax, 0.5, inner_ys[0]-0.027, 0.5, inner_ys[1]+0.027)
box(ax, 0.5, inner_ys[1], 0.65, 0.055,
    "Multi-Head Self-Attention  (H=4, DH=32)\nQ=K=V=x@W  →  softmax(QKᵀ/√32)V@Wo", C_ATTN, fs=7)
arr(ax, 0.5, inner_ys[1]-0.027, 0.5, inner_ys[2]+0.027)
box(ax, 0.5, inner_ys[2], 0.65, 0.055, "LayerNorm  + Residual", C_LN, fs=7.5)
arr(ax, 0.5, inner_ys[2]-0.027, 0.5, inner_ys[3]+0.027)
box(ax, 0.5, inner_ys[3], 0.65, 0.055,
    "FFN: Linear(128→256)+GeLU→Linear(256→128) + Residual", C_FFN, fs=7)

arr(ax, 0.5, by - bh/2 - 0.01, 0.5, ys[4]+0.035)
box(ax, 0.5, ys[4], 0.62, 0.07, "Final LayerNorm  →  mean over 49 tokens  →  (B, 128)", C_LN)
arr(ax, 0.5, ys[4]-0.035, 0.5, ys[5]+0.035)
box(ax, 0.5, ys[5], 0.55, 0.07, "Linear(128→LATENT)", C_HEAD)
arr(ax, 0.5, ys[5]-0.035, 0.5, ys[6]+0.035)
box(ax, 0.5, ys[6], 0.50, 0.07, "(B, LATENT_DIM)", C_HEAD)

note(ax, 0.5, 0.04, "Total transformer params (excl. head): ~392K  |  Latent-agnostic backbone")

fig.tight_layout()
url_vit_diag = save_matplotlib_figure("arch_vit_diagram", fig, format="svg")
plt.close(fig)
print(f"ViT diagram → {url_vit_diag}")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Gram Hadamard comparison: dual vs single projection
# ═══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(13, 7))
fig.suptitle("Gram Encoder: Dual Projection (original) vs Single Projection Hadamard",
             fontsize=11, fontweight="bold")

C_POS = "#0369a1"; C_GRAM = "#7c3aed"; C_HAD = "#ec4899"; C_OUT = "#1d4ed8"

for ax, (title, variant) in zip(axes, [
    ("Original: a = H@W_A,  b = H@W_B,  out = a⊙b", "dual"),
    ("Ablation: z = H@W_proj,  out = z⊙H  (LATENT=DEC_RANK²)", "single"),
]):
    ax.set_xlim(0, 1); ax.set_ylim(-0.04, 1.04); ax.axis("off")
    ax.set_title(title, fontsize=9, fontweight="bold", pad=8)

    # Shared top: positional encoding + gram
    box(ax, 0.5, 0.96, 0.55, 0.07, "x  (B, 784)", C_GRAY)
    arr(ax, 0.5, 0.925, 0.5, 0.875)
    box(ax, 0.5, 0.84, 0.72, 0.07,
        "row_emb[p] ⊗ col_emb[p]  →  pos_feat  @  W_pos_enc  @  W_enc\n→  f  (784, DEC_RANK)", C_POS, fs=7.5)
    arr(ax, 0.5, 0.805, 0.5, 0.755)
    box(ax, 0.5, 0.72, 0.68, 0.07,
        "F[b,p,i] = x_norm[b,p] × f[p,i]\nH = einsum(F,F,bpi,bpj→bij)  →  (B, DEC_RANK²)", C_GRAM, fs=7.5)
    arr(ax, 0.5, 0.685, 0.5, 0.635)

    if variant == "dual":
        box(ax, 0.27, 0.60, 0.43, 0.07, "a = H @ W_enc_A\n(B, LATENT)", C_HAD, fs=7.5)
        box(ax, 0.73, 0.60, 0.43, 0.07, "b = H @ W_enc_B\n(B, LATENT)", C_HAD, fs=7.5)
        ax.annotate("", xy=(0.27, 0.565), xytext=(0.5, 0.635),
                    arrowprops=dict(arrowstyle="->", lw=1.3, color="#555"))
        ax.annotate("", xy=(0.73, 0.565), xytext=(0.5, 0.635),
                    arrowprops=dict(arrowstyle="->", lw=1.3, color="#555"))
        ax.annotate("", xy=(0.5, 0.49), xytext=(0.27, 0.565),
                    arrowprops=dict(arrowstyle="->", lw=1.3, color="#555"))
        ax.annotate("", xy=(0.5, 0.49), xytext=(0.73, 0.565),
                    arrowprops=dict(arrowstyle="->", lw=1.3, color="#555"))
        ax.text(0.5, 0.515, "⊙", ha="center", va="center", fontsize=18, color="#d97706")
        box(ax, 0.5, 0.44, 0.60, 0.07, "output = a ⊙ b  →  (B, LATENT)", C_OUT)
        note(ax, 0.5, 0.34, "Two projection matrices W_enc_A, W_enc_B")
        note(ax, 0.5, 0.29, "Params: 2 × DEC_RANK² × LATENT")
        note(ax, 0.5, 0.24, "Learns independent 'what' and 'how much' projections")
        note(ax, 0.5, 0.16, "DEC_RANK=32 independent of LATENT_DIM")
    else:
        arr(ax, 0.5, 0.635, 0.5, 0.585)
        box(ax, 0.5, 0.55, 0.62, 0.07, "z = H @ W_proj  →  (B, LATENT)", C_HAD)
        ax.annotate("", xy=(0.35, 0.455), xytext=(0.5, 0.515),
                    arrowprops=dict(arrowstyle="->", lw=1.3, color="#555"))
        ax.annotate("", xy=(0.65, 0.455), xytext=(0.5, 0.515),
                    arrowprops=dict(arrowstyle="->", lw=1.3, color="#555"))
        box(ax, 0.22, 0.42, 0.35, 0.065, "z\n(B, LATENT)", C_HAD, fs=7.5)
        box(ax, 0.78, 0.42, 0.35, 0.065, "H  (reused)\n(B, LATENT)", C_GRAM, fs=7.5)
        ax.annotate("", xy=(0.5, 0.345), xytext=(0.22, 0.387),
                    arrowprops=dict(arrowstyle="->", lw=1.3, color="#555"))
        ax.annotate("", xy=(0.5, 0.345), xytext=(0.78, 0.387),
                    arrowprops=dict(arrowstyle="->", lw=1.3, color="#555"))
        ax.text(0.5, 0.37, "⊙", ha="center", va="center", fontsize=18, color="#d97706")
        box(ax, 0.5, 0.30, 0.60, 0.07, "output = z ⊙ H  →  (B, LATENT)", C_OUT)
        note(ax, 0.5, 0.20, "Single projection matrix W_proj")
        note(ax, 0.5, 0.15, "Params: DEC_RANK² × LATENT  (half the readout)")
        note(ax, 0.5, 0.10, "Constraint: LATENT_DIM = DEC_RANK²")
        note(ax, 0.5, 0.04, "Run at LATENT ∈ {256 (rank=16), 1024 (rank=32), 4096 (rank=64)}")

fig.tight_layout()
url_gram_diag = save_matplotlib_figure("arch_gram_single_diagram", fig, format="svg")
plt.close(fig)
print(f"Gram-single diagram → {url_gram_diag}")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Accuracy comparison plot
# ═══════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(9, 5.5))

def plot_line(data, label, color, marker, ls="-", zorder=2):
    if not data:
        return
    xs = sorted(data)
    ys = [data[x] * 100 for x in xs]
    ax.plot(xs, ys, color=color, marker=marker, linewidth=2, markersize=7,
            label=label, linestyle=ls, zorder=zorder)
    off = 8
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                    xytext=(0, off), ha="center", fontsize=7.5, color=color)
        off = -off

plot_line(gram_data,  "Gram+Hadamard (dual, reference)", "#7c3aed", "D", zorder=3)
plot_line(gramS_data, "Gram+Hadamard (single W_proj)",   "#c084fc", "d", ls="--")
plot_line(conv2_data, "ConvNet v2 [64,128,256]+FC",       "#16a34a", "^")
plot_line(conv1_data, "ConvNet v1 [32,64,128]",           "#86efac", "^", ls="--")
plot_line(vit_data,   "Mini ViT (3L, D=128, 4H)",         "#f59e0b", "s")

ax.set_xlabel("Latent dimension", fontsize=11)
ax.set_ylabel("Linear probe accuracy (%)", fontsize=11)
ax.set_title("Probe accuracy vs latent size\n(EMA training, 4 augmentation actions, MNIST)", fontsize=11)
ax.set_xscale("log", base=2)
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
all_latents = sorted(set(
    list(gram_data) + list(gramS_data) + list(conv2_data) +
    list(conv1_data) + list(vit_data)
))
if all_latents:
    ax.set_xticks(all_latents)
all_accs = [v * 100 for v in (
    list(gram_data.values()) + list(gramS_data.values()) + list(conv2_data.values()) +
    list(conv1_data.values()) + list(vit_data.values())
)]
if all_accs:
    ax.set_ylim(max(50, min(all_accs) - 3), min(100, max(all_accs) + 3))
ax.legend(fontsize=9, loc="lower right")
ax.grid(True, alpha=0.3)
fig.tight_layout()
url_acc = save_matplotlib_figure("arch_accuracy_comparison", fig, format="svg")
plt.close(fig)
print(f"Accuracy plot → {url_acc}")

# ═══════════════════════════════════════════════════════════════════════════════
# Assemble report
# ═══════════════════════════════════════════════════════════════════════════════

def fmt_table(data_dict):
    if not data_dict:
        return "*No results yet.*\n"
    md = "| LATENT_DIM | probe_acc |\n|---|---|\n"
    for k in sorted(data_dict):
        md += f"| {k} | {data_dict[k]:.1%} |\n"
    return md

def best_of(d):
    if not d: return "—", "—"
    k = max(d, key=d.get); return k, f"{d[k]:.1%}"

md = f"""
# Architecture Study — EMA JEPA on MNIST

New encoder architectures compared under identical EMA-JEPA training
(τ=0.996, FiLM predictor, 4 augmentation actions, 200 epochs, cosine LR).

---

## 1 · ConvNet Encoder

### Architecture

![]({url_conv_diag})

### Why v1 was underwhelming

The original ConvNet (v1) had channels [32, 64, 128]. After global average pooling
the representation is only 128-dimensional before projecting to LATENT_DIM.
At LATENT_DIM=4096 this means a direct 128→4096 expansion, which has limited
capacity and can train poorly. v2 fixes this with wider channels [64, 128, 256]
and an intermediate FC layer (256→512→GeLU→LATENT_DIM).

### Pseudocode

```python
def encode(x):                    # x: (B, 784)
    h = x.reshape(B, 28, 28, 1) / 255.0
    h = gelu(conv(h, W_conv1, 32ch, 3×3, SAME))   # → (B, 28, 28, 64)  [v2: 64ch]
    h = maxpool2x2(h)                              # → (B, 14, 14, 64)
    h = gelu(conv(h, W_conv2, 64ch, 3×3, SAME))   # → (B, 14, 14, 128)
    h = maxpool2x2(h)                              # → (B,  7,  7, 128)
    h = gelu(conv(h, W_conv3, 128ch, 3×3, SAME))  # → (B,  7,  7, 256) [v2: 256ch]
    h = h.mean(axis=(H,W))                         # global avg pool → (B, 256)
    # v1: return h @ W_fc                          # → (B, LATENT)  [bottleneck]
    # v2:
    h = gelu(h @ W_fc1)                            # → (B, 512)
    return h @ W_fc2                               # → (B, LATENT)
```

### Results

#### ConvNet v2 [64, 128, 256] + FC(512)
{fmt_table(conv2_data)}

#### ConvNet v1 [32, 64, 128] — baseline
{fmt_table(conv1_data)}

---

## 2 · Mini Vision Transformer

### Architecture

![]({url_vit_diag})

### Design

- **Tokenisation**: 28×28 image split into 49 non-overlapping 4×4 patches (16 dims each)
- **Patch embedding**: Linear(16→128) + learned positional embeddings (49×128)
- **3 transformer layers** with pre-norm (LayerNorm before attention/FFN)
- **Multi-head attention**: H=4 heads, DH=32 per head
- **FFN**: Linear(128→256)+GeLU→Linear(256→128) per layer
- **Readout**: final LayerNorm → mean pool over 49 tokens → Linear(128→LATENT)

### Pseudocode

```python
def encode(x):                              # x: (B, 784)
    patches = extract_4x4_patches(x/255)   # (B, 49, 16)
    h = patches @ W_patch + pos_emb        # (B, 49, 128)  patch embed + pos

    for i in range(3):
        # Pre-norm self-attention
        h_ln = layer_norm(h, g1_i, b1_i)
        Q = h_ln @ W_q_i;  K = h_ln @ W_k_i;  V = h_ln @ W_v_i
        # Reshape to (B, H, S, DH) for multi-head
        attn = softmax(Q @ Kᵀ / sqrt(32))
        h = h + (attn @ V) @ W_o_i         # residual + proj

        # Pre-norm FFN
        h_ln = layer_norm(h, g2_i, b2_i)
        h = h + gelu(h_ln @ W_ff1_i) @ W_ff2_i  # residual

    h = layer_norm(h, g_final, b_final).mean(axis=1)   # (B, 128)
    return h @ W_head                                   # (B, LATENT)
```

### Results

{fmt_table(vit_data)}

---

## 3 · Gram + Single-Projection Hadamard

### Architecture

![]({url_gram_diag})

### Design

The original Gram+Hadamard uses two independent projection matrices (W_enc_A, W_enc_B)
to produce two views `a` and `b` of the gram matrix, then combines them via `a ⊙ b`.

This ablation uses a **single** projection W_proj, and computes the Hadamard product
between the projection output `z` and the raw flattened gram `H`:

```
output = (H @ W_proj)  ⊙  H
```

This requires LATENT_DIM = DEC_RANK² so the shapes match.
Run at LATENT ∈ {{256 (DEC_RANK=16), 1024 (DEC_RANK=32), 4096 (DEC_RANK=64)}}.

### Pseudocode

```python
def encode(x):                         # x: (B, 784)
    # --- same Hebbian gram as original ---
    pos_feat = pos_emb(x)              # (784, D=144) via outer product
    f = pos_feat @ W_enc               # (784, DEC_RANK)
    F = (x/255)[:, :, None] * f        # (B, 784, DEC_RANK)  pixel-weighted
    H = einsum("bpi,bpj->bij", F, F)   # (B, DEC_RANK, DEC_RANK)  gram matrix
    H = H.reshape(B, DEC_RANK**2)      # (B, LATENT)  since LATENT = DEC_RANK²

    # --- single-projection Hadamard (replaces W_enc_A + W_enc_B) ---
    z = H @ W_proj                     # (B, LATENT)
    return z * H                       # (B, LATENT)  self-gating
```

### Results

{fmt_table(gramS_data)}

---

## 4 · Accuracy Comparison

![]({url_acc})

## 5 · Summary Table

| Encoder | Best LATENT | probe_acc | Notes |
|---|---|---|---|
| Gram+Hadamard (dual) | {best_of(gram_data)[0]} | {best_of(gram_data)[1]} | Reference |
| Gram+Hadamard (single) | {best_of(gramS_data)[0]} | {best_of(gramS_data)[1]} | Half the readout params |
| ConvNet v2 | {best_of(conv2_data)[0]} | {best_of(conv2_data)[1]} | Fixed bottleneck |
| ConvNet v1 | {best_of(conv1_data)[0]} | {best_of(conv1_data)[1]} | Original |
| Mini ViT | {best_of(vit_data)[0]} | {best_of(vit_data)[1]} | 3L, D=128, 4H |

---

*Generated by `projects/ssl/scripts/gen_report_arch.py`*
"""

url_report = save_report("ssl_report_arch", md, title="Architecture Study — EMA JEPA MNIST")
print(f"\nReport → {url_report}")
