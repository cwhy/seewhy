"""Generate arch×arch masked distillation heatmap report."""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from shared_lib.media import save_matplotlib_figure

JSONL        = Path(__file__).parent.parent / "results_distill_heatmap.jsonl"
RESULTS_JSONL = Path(__file__).parent.parent / "results.jsonl"
REPORT       = Path(__file__).parent.parent / "report-distill-heatmap.md"

# ── Load results ─────────────────────────────────────────────────────────────

rows = [json.loads(l) for l in open(JSONL)]
print(f"Loaded {len(rows)} distill heatmap results")

ARCHS = ["mlp", "mlp2", "conv", "conv2", "vit", "vit_patch",
         "gram_dual", "gram_single", "gram_glu"]

ARCH_LABELS = {
    "mlp": "MLP", "mlp2": "MLP-2", "conv": "Conv", "conv2": "Conv-2",
    "vit": "ViT", "vit_patch": "ViT-patch",
    "gram_dual": "Gram-dual", "gram_single": "Gram-single", "gram_glu": "Gram-GLU",
}
LABELS = [ARCH_LABELS[a] for a in ARCHS]

probe_mat = np.full((len(ARCHS), len(ARCHS)), np.nan)
knn_mat   = np.full((len(ARCHS), len(ARCHS)), np.nan)

for r in rows:
    i = ARCHS.index(r["teacher_arch"])
    j = ARCHS.index(r["student_arch"])
    probe_mat[i, j] = r["probe_acc"] * 100
    knn_mat[i, j]   = r["knn_1nn_acc"] * 100

# ── EMA-JEPA baselines ────────────────────────────────────────────────────────

ARCH_EXP = {
    "mlp":         "exp_mlp_ema_1024",
    "mlp2":        "exp_mlp2_ema_1024",
    "conv":        "exp_conv_ema_1024",
    "conv2":       "exp_conv2_ema_1024",
    "vit":         "exp_vit_ema_1024",
    "vit_patch":   "exp_vit_patch_ema",
    "gram_dual":   "exp_ema_1024",
    "gram_single": "exp_gram_single_ema_1024",
    "gram_glu":    "exp_gram_glu_ema",
}
all_rows = [json.loads(l) for l in open(RESULTS_JSONL)]
baseline = {}
for r in all_rows:
    for arch, exp in ARCH_EXP.items():
        if r["experiment"] == exp:
            baseline[arch] = r["final_probe_acc"] * 100

# Delta matrix: distill probe - teacher's EMA-JEPA baseline (per teacher row)
delta_mat = np.full((len(ARCHS), len(ARCHS)), np.nan)
for i, arch in enumerate(ARCHS):
    if arch in baseline:
        delta_mat[i, :] = probe_mat[i, :] - baseline[arch]

# ── Heatmap helper ────────────────────────────────────────────────────────────

def plot_heatmap(mat, title, vmin, vmax, cbar_label, fmt=".1f", cmap="RdYlGn"):
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(mat, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")
    plt.colorbar(im, ax=ax, label=cbar_label, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(ARCHS))); ax.set_xticklabels(LABELS, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(ARCHS))); ax.set_yticklabels(LABELS, fontsize=9)
    ax.set_xlabel("Student encoder", fontsize=11)
    ax.set_ylabel("Teacher encoder", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    for i in range(len(ARCHS)):
        for j in range(len(ARCHS)):
            v = mat[i, j]
            if not np.isnan(v):
                norm_v = (v - vmin) / (vmax - vmin)
                txt_col = "white" if norm_v < 0.25 or norm_v > 0.85 else "black"
                prefix = "+" if fmt != ".1f" or v > 0 else ""
                label = f"{prefix}{v:{fmt}}" if "+" in prefix else f"{v:{fmt}}"
                ax.text(j, i, label, ha="center", va="center",
                        fontsize=7, color=txt_col, fontweight="bold")
    plt.tight_layout()
    return fig

# ── Plot probe heatmap ────────────────────────────────────────────────────────

vmin_p = max(85, np.nanmin(probe_mat) - 1)
vmax_p = min(100, np.nanmax(probe_mat) + 0.5)
fig_probe = plot_heatmap(probe_mat,
    "Masked Distillation: Linear Probe Accuracy (%) — MNIST",
    vmin=vmin_p, vmax=vmax_p, cbar_label="Probe accuracy (%)")
url_probe = save_matplotlib_figure("ssl_distill_heatmap_probe", fig_probe, dpi=150)
plt.close(fig_probe)
print(f"Probe heatmap → {url_probe}")

# ── Plot KNN heatmap ─────────────────────────────────────────────────────────

vmin_k = max(80, np.nanmin(knn_mat) - 1)
vmax_k = min(100, np.nanmax(knn_mat) + 0.5)
fig_knn = plot_heatmap(knn_mat,
    "Masked Distillation: 1-NN KNN Accuracy (%) — MNIST",
    vmin=vmin_k, vmax=vmax_k, cbar_label="1-NN KNN accuracy (%)")
url_knn = save_matplotlib_figure("ssl_distill_heatmap_knn", fig_knn, dpi=150)
plt.close(fig_knn)
print(f"KNN heatmap → {url_knn}")

# ── Plot delta heatmap ────────────────────────────────────────────────────────

abs_max = np.nanmax(np.abs(delta_mat))
fig_delta = plot_heatmap(delta_mat,
    "Δ Probe vs Teacher EMA-JEPA Baseline (student distill − teacher baseline) — MNIST",
    vmin=-abs_max, vmax=abs_max, cbar_label="Δ accuracy (pp)",
    fmt="+.1f", cmap="RdYlGn")
url_delta = save_matplotlib_figure("ssl_distill_heatmap_delta", fig_delta, dpi=150)
plt.close(fig_delta)
print(f"Delta heatmap → {url_delta}")

# ── Summary stats ─────────────────────────────────────────────────────────────

top5 = sorted(rows, key=lambda r: r["probe_acc"], reverse=True)[:5]
bot5 = sorted(rows, key=lambda r: r["probe_acc"])[:5]

teacher_avg = {a: np.nanmean(probe_mat[i]) for i, a in enumerate(ARCHS)}
student_avg = {a: np.nanmean(probe_mat[:, j]) for j, a in enumerate(ARCHS)}
best_teacher = max(teacher_avg, key=teacher_avg.get)
best_student = max(student_avg, key=student_avg.get)

diag_probe = {ARCHS[i]: probe_mat[i, i] for i in range(len(ARCHS))}

# Per-student best distill vs baseline
best_distill = {}
for r in rows:
    s = r["student_arch"]
    if s not in best_distill or r["probe_acc"] > best_distill[s]["probe_acc"]:
        best_distill[s] = r

# ── Write local report ────────────────────────────────────────────────────────

lines = [
    "# Masked Distillation: Arch×Arch Heatmap",
    "",
    "**Method**: Masked distillation — frozen EMA-JEPA teacher encodes clean images;",
    "student learns to predict teacher representations from augmented views (Gaussian noise,",
    "25/50/75% pixel masks) via a FiLM predictor. 200 epochs, LATENT_DIM=1024.",
    "",
    "**Grid**: 9 teacher architectures × 9 student architectures = 81 experiments.",
    "All teachers are pre-trained EMA-JEPA encoders at LATENT_DIM=1024.",
    "",
    "---",
    "",
    "## Linear Probe Accuracy",
    "",
    f"![Probe heatmap]({url_probe})",
    "",
    "## 1-NN KNN Accuracy",
    "",
    f"![KNN heatmap]({url_knn})",
    "",
    "## Δ vs EMA-JEPA Baseline",
    "",
    "Each cell shows how much the student's probe accuracy differs from the **teacher's** own",
    "EMA-JEPA baseline. Positive = student surpasses the teacher; negative = student falls short.",
    "",
    f"![Delta heatmap]({url_delta})",
    "",
    "---",
    "",
    "## Key Findings",
    "",
    f"**Best teacher** (avg probe across all students): **{ARCH_LABELS[best_teacher]}** "
    f"({teacher_avg[best_teacher]:.1f}%)",
    "",
    f"**Best student** (avg probe across all teachers): **{ARCH_LABELS[best_student]}** "
    f"({student_avg[best_student]:.1f}%)",
    "",
    "### Top 5 combinations (linear probe)",
    "",
    "| Teacher | Student | Probe | KNN |",
    "|---------|---------|-------|-----|",
]
for r in top5:
    lines.append(
        f"| {ARCH_LABELS[r['teacher_arch']]} | {ARCH_LABELS[r['student_arch']]} "
        f"| {r['probe_acc']:.1%} | {r['knn_1nn_acc']:.1%} |"
    )

lines += [
    "",
    "### Bottom 5 combinations (linear probe)",
    "",
    "| Teacher | Student | Probe | KNN |",
    "|---------|---------|-------|-----|",
]
for r in bot5:
    lines.append(
        f"| {ARCH_LABELS[r['teacher_arch']]} | {ARCH_LABELS[r['student_arch']]} "
        f"| {r['probe_acc']:.1%} | {r['knn_1nn_acc']:.1%} |"
    )

lines += [
    "",
    "### Best student per teacher vs teacher's own baseline",
    "",
    "| Teacher | Teacher baseline | Best student | Best student probe | Δ |",
    "|---------|-----------------|--------------|-------------------|---|",
]
best_student_per_teacher = {}
for r in rows:
    t = r["teacher_arch"]
    if t not in best_student_per_teacher or r["probe_acc"] > best_student_per_teacher[t]["probe_acc"]:
        best_student_per_teacher[t] = r
for arch in ARCHS:
    b = baseline.get(arch, float("nan"))
    d = best_student_per_teacher.get(arch, {})
    dp = d.get("probe_acc", float("nan")) * 100 if "probe_acc" in d else float("nan")
    delta = dp - b
    student = ARCH_LABELS.get(d.get("student_arch", ""), "?")
    lines.append(
        f"| {ARCH_LABELS[arch]} | {b:.1f}% | {student} | {dp:.1f}% | {delta:+.1f} pp |"
    )

lines += [
    "",
    "### Average probe accuracy by teacher",
    "",
    "| Teacher | Avg probe (all students) |",
    "|---------|--------------------------|",
]
for a in sorted(ARCHS, key=lambda x: teacher_avg[x], reverse=True):
    lines.append(f"| {ARCH_LABELS[a]} | {teacher_avg[a]:.1f}% |")

lines += [
    "",
    "### Average probe accuracy by student",
    "",
    "| Student | Avg probe (all teachers) |",
    "|---------|--------------------------|",
]
for a in sorted(ARCHS, key=lambda x: student_avg[x], reverse=True):
    lines.append(f"| {ARCH_LABELS[a]} | {student_avg[a]:.1f}% |")

lines += [
    "",
    "### Self-distillation diagonal (same arch as teacher and student)",
    "",
    "| Arch | Self-distill probe | EMA-JEPA baseline | Δ |",
    "|------|--------------------|--------------------|---|",
]
for a in ARCHS:
    self_p = diag_probe[a]
    base_p = baseline.get(a, float("nan"))
    delta  = self_p - base_p
    lines.append(f"| {ARCH_LABELS[a]} | {self_p:.1f}% | {base_p:.1f}% | {delta:+.1f} pp |")

REPORT.write_text("\n".join(lines) + "\n")
print(f"\nLocal report written to {REPORT}")
