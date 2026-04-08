"""Generate heatmap report for a given teacher type: sigreg, dae, or random.

Usage:
    uv run python projects/ssl/scripts/gen_report_distill_heatmap_typed.py sigreg
    uv run python projects/ssl/scripts/gen_report_distill_heatmap_typed.py dae
    uv run python projects/ssl/scripts/gen_report_distill_heatmap_typed.py random
"""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from shared_lib.media import save_matplotlib_figure

if len(sys.argv) < 2 or sys.argv[1] not in ("sigreg", "dae", "random"):
    print(__doc__)
    sys.exit(1)

TEACHER_TYPE = sys.argv[1]

SSL          = Path(__file__).parent.parent
JSONL        = SSL / f"results_distill_heatmap_{TEACHER_TYPE}.jsonl"
RESULTS_JSONL = SSL / "results.jsonl"
REPORT       = SSL / f"report-distill-heatmap-{TEACHER_TYPE}.md"

# ── Load results ──────────────────────────────────────────────────────────────

rows = [json.loads(l) for l in open(JSONL)]
print(f"Loaded {len(rows)} results for teacher_type={TEACHER_TYPE}")

ALL_ARCHS = ["mlp", "mlp2", "conv", "conv2", "vit", "vit_patch",
             "gram_dual", "gram_single", "gram_glu"]

TEACHER_ARCHS = sorted({r["teacher_arch"] for r in rows},
                       key=lambda a: ALL_ARCHS.index(a))
STUDENT_ARCHS = ALL_ARCHS

ARCH_LABELS = {
    "mlp": "MLP", "mlp2": "MLP-2", "conv": "Conv", "conv2": "Conv-2",
    "vit": "ViT", "vit_patch": "ViT-patch",
    "gram_dual": "Gram-dual", "gram_single": "Gram-single", "gram_glu": "Gram-GLU",
}

T_LABELS = [ARCH_LABELS[a] for a in TEACHER_ARCHS]
S_LABELS = [ARCH_LABELS[a] for a in STUDENT_ARCHS]

probe_mat = np.full((len(TEACHER_ARCHS), len(STUDENT_ARCHS)), np.nan)
knn_mat   = np.full((len(TEACHER_ARCHS), len(STUDENT_ARCHS)), np.nan)

for r in rows:
    i = TEACHER_ARCHS.index(r["teacher_arch"])
    j = STUDENT_ARCHS.index(r["student_arch"])
    probe_mat[i, j] = r["probe_acc"] * 100
    knn_mat[i, j]   = r["knn_1nn_acc"] * 100

# ── Teacher baselines (probe acc of teacher's own training method) ─────────────

TEACHER_EXP = {
    "sigreg": {
        "mlp": "exp_sigreg_mlp", "mlp2": "exp_sigreg_mlp2",
        "conv": "exp_sigreg_conv", "conv2": "exp_sigreg_conv2",
        "vit": "exp_sigreg_vit", "vit_patch": "exp_sigreg_vit_patch",
        "gram_single": "exp_sigreg_gram_single", "gram_glu": "exp_sigreg_gram_glu",
    },
    "dae": {
        "mlp": "exp_dae_mlp", "mlp2": "exp_dae_mlp2",
        "conv": "exp_dae_conv", "conv2": "exp_dae_conv2",
        "vit": "exp_dae_vit", "vit_patch": "exp_dae_vit_patch",
        "gram_dual": "exp_dae_gram_dual",
        "gram_single": "exp_dae_gram_single", "gram_glu": "exp_dae_gram_glu",
    },
    "random": {},   # no trained baseline for random
}

all_rows = [json.loads(l) for l in open(RESULTS_JSONL)]
teacher_baseline = {}
for r in all_rows:
    for arch, exp in TEACHER_EXP.get(TEACHER_TYPE, {}).items():
        if r["experiment"] == exp:
            teacher_baseline[arch] = r["final_probe_acc"] * 100

# Delta vs teacher baseline (per teacher row)
delta_mat = np.full((len(TEACHER_ARCHS), len(STUDENT_ARCHS)), np.nan)
for i, arch in enumerate(TEACHER_ARCHS):
    if arch in teacher_baseline:
        delta_mat[i, :] = probe_mat[i, :] - teacher_baseline[arch]

# ── Heatmap helper ─────────────────────────────────────────────────────────────

def plot_heatmap(mat, title, vmin, vmax, cbar_label, t_labels, s_labels,
                 fmt=".1f", cmap="RdYlGn", show_plus=False):
    fig, ax = plt.subplots(figsize=(9, max(5, len(t_labels) * 0.78 + 1.5)))
    im = ax.imshow(mat, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")
    plt.colorbar(im, ax=ax, label=cbar_label, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(s_labels)))
    ax.set_xticklabels(s_labels, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(t_labels)))
    ax.set_yticklabels(t_labels, fontsize=9)
    ax.set_xlabel("Student encoder", fontsize=11)
    ax.set_ylabel("Teacher encoder", fontsize=11)
    ax.set_title(title, fontsize=11, fontweight="bold")
    for i in range(len(t_labels)):
        for j in range(len(s_labels)):
            v = mat[i, j]
            if not np.isnan(v):
                norm_v = (v - vmin) / (vmax - vmin)
                txt_col = "white" if norm_v < 0.25 or norm_v > 0.85 else "black"
                label = f"{v:+{fmt}}" if show_plus else f"{v:{fmt}}"
                ax.text(j, i, label, ha="center", va="center",
                        fontsize=7, color=txt_col, fontweight="bold")
    plt.tight_layout()
    return fig

# ── Probe heatmap ──────────────────────────────────────────────────────────────

vmin_p = max(85, np.nanmin(probe_mat) - 1)
vmax_p = min(100, np.nanmax(probe_mat) + 0.5)
fig = plot_heatmap(probe_mat,
    f"Masked Distillation ({TEACHER_TYPE.upper()} teachers): Linear Probe % — MNIST",
    vmin_p, vmax_p, "Probe accuracy (%)", T_LABELS, S_LABELS)
url_probe = save_matplotlib_figure(f"ssl_distill_{TEACHER_TYPE}_probe", fig, dpi=150)
plt.close(fig)
print(f"Probe → {url_probe}")

# ── KNN heatmap ───────────────────────────────────────────────────────────────

vmin_k = max(80, np.nanmin(knn_mat) - 1)
vmax_k = min(100, np.nanmax(knn_mat) + 0.5)
fig = plot_heatmap(knn_mat,
    f"Masked Distillation ({TEACHER_TYPE.upper()} teachers): 1-NN KNN % — MNIST",
    vmin_k, vmax_k, "1-NN KNN accuracy (%)", T_LABELS, S_LABELS)
url_knn = save_matplotlib_figure(f"ssl_distill_{TEACHER_TYPE}_knn", fig, dpi=150)
plt.close(fig)
print(f"KNN  → {url_knn}")

# ── Delta heatmap (only if we have baselines) ──────────────────────────────────

url_delta = None
if teacher_baseline:
    abs_max = max(np.nanmax(np.abs(delta_mat)), 1.0)
    fig = plot_heatmap(delta_mat,
        f"Δ Probe vs {TEACHER_TYPE.upper()} Teacher Baseline — MNIST",
        -abs_max, abs_max, "Δ accuracy (pp)", T_LABELS, S_LABELS,
        fmt=".1f", cmap="RdYlGn", show_plus=True)
    url_delta = save_matplotlib_figure(f"ssl_distill_{TEACHER_TYPE}_delta", fig, dpi=150)
    plt.close(fig)
    print(f"Delta → {url_delta}")

# ── Summary stats ──────────────────────────────────────────────────────────────

top5 = sorted(rows, key=lambda r: r["probe_acc"], reverse=True)[:5]
bot5 = sorted(rows, key=lambda r: r["probe_acc"])[:5]
teacher_avg = {a: np.nanmean(probe_mat[i]) for i, a in enumerate(TEACHER_ARCHS)}
student_avg = {a: np.nanmean(probe_mat[:, j]) for j, a in enumerate(STUDENT_ARCHS)}
best_student_per_teacher = {}
for r in rows:
    t = r["teacher_arch"]
    if t not in best_student_per_teacher or r["probe_acc"] > best_student_per_teacher[t]["probe_acc"]:
        best_student_per_teacher[t] = r

# ── Write local report ─────────────────────────────────────────────────────────

TYPE_NAMES = {"sigreg": "SigReg", "dae": "DAE (Denoising Autoencoder)", "random": "Random (untrained)"}
type_name  = TYPE_NAMES[TEACHER_TYPE]

lines = [
    f"# Masked Distillation: Arch×Arch Heatmap — {type_name} Teachers",
    "",
    f"**Method**: Masked distillation — frozen {type_name} teacher encodes clean images;",
    "student learns to predict teacher representations from augmented views (Gaussian noise,",
    "25/50/75% pixel masks) via a FiLM predictor. 200 epochs, LATENT_DIM=1024.",
    "",
    f"**Grid**: {len(TEACHER_ARCHS)} teacher architectures × {len(STUDENT_ARCHS)} student architectures"
    f" = {len(rows)} experiments.",
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
]

if url_delta:
    lines += [
        "## Δ vs Teacher Baseline",
        "",
        f"Each cell: student probe − teacher's own {type_name} probe accuracy.",
        "Positive = student surpasses the teacher.",
        "",
        f"![Delta heatmap]({url_delta})",
        "",
    ]

lines += [
    "---",
    "",
    "## Key Findings",
    "",
    f"**Best teacher** (avg probe): **{ARCH_LABELS[max(teacher_avg, key=teacher_avg.get)]}** "
    f"({max(teacher_avg.values()):.1f}%)",
    "",
    f"**Best student** (avg probe): **{ARCH_LABELS[max(student_avg, key=student_avg.get)]}** "
    f"({max(student_avg.values()):.1f}%)",
    "",
    "### Top 5 combinations",
    "",
    "| Teacher | Student | Probe | KNN |",
    "|---------|---------|-------|-----|",
]
for r in top5:
    lines.append(f"| {ARCH_LABELS[r['teacher_arch']]} | {ARCH_LABELS[r['student_arch']]} "
                 f"| {r['probe_acc']:.1%} | {r['knn_1nn_acc']:.1%} |")

lines += [
    "",
    "### Bottom 5 combinations",
    "",
    "| Teacher | Student | Probe | KNN |",
    "|---------|---------|-------|-----|",
]
for r in bot5:
    lines.append(f"| {ARCH_LABELS[r['teacher_arch']]} | {ARCH_LABELS[r['student_arch']]} "
                 f"| {r['probe_acc']:.1%} | {r['knn_1nn_acc']:.1%} |")

if teacher_baseline:
    lines += [
        "",
        f"### Best student per teacher vs {type_name} baseline",
        "",
        f"| Teacher | {type_name} baseline | Best student | Best student probe | Δ |",
        "|---------|-----------------|--------------|-------------------|---|",
    ]
    for arch in TEACHER_ARCHS:
        b = teacher_baseline.get(arch, float("nan"))
        d = best_student_per_teacher.get(arch, {})
        dp = d.get("probe_acc", float("nan")) * 100 if "probe_acc" in d else float("nan")
        student = ARCH_LABELS.get(d.get("student_arch", ""), "?")
        lines.append(f"| {ARCH_LABELS[arch]} | {b:.1f}% | {student} | {dp:.1f}% | {dp-b:+.1f} pp |")

lines += [
    "",
    "### Average probe by teacher",
    "",
    "| Teacher | Avg probe |",
    "|---------|-----------|",
]
for a in sorted(TEACHER_ARCHS, key=lambda x: teacher_avg[x], reverse=True):
    lines.append(f"| {ARCH_LABELS[a]} | {teacher_avg[a]:.1f}% |")

lines += [
    "",
    "### Average probe by student",
    "",
    "| Student | Avg probe |",
    "|---------|-----------|",
]
for a in sorted(STUDENT_ARCHS, key=lambda x: student_avg[x], reverse=True):
    lines.append(f"| {ARCH_LABELS[a]} | {student_avg[a]:.1f}% |")

REPORT.write_text("\n".join(lines) + "\n")
print(f"\nLocal report → {REPORT}")
