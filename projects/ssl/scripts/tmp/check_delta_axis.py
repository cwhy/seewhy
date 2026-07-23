"""Check whether improvement is driven by teacher or student choice."""
import json
import numpy as np

ARCHS = ["mlp", "mlp2", "conv", "conv2", "vit", "vit_patch",
         "gram_dual", "gram_single", "gram_glu"]
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
LABELS = {"mlp":"MLP","mlp2":"MLP-2","conv":"Conv","conv2":"Conv-2",
          "vit":"ViT","vit_patch":"ViT-patch","gram_dual":"Gram-dual",
          "gram_single":"Gram-single","gram_glu":"Gram-GLU"}

rows = [json.loads(l) for l in open("projects/ssl/results.jsonl")]
baseline = {}
for r in rows:
    for arch, exp in ARCH_EXP.items():
        if r["experiment"] == exp:
            baseline[arch] = r["final_probe_acc"] * 100

heatmap = [json.loads(l) for l in open("projects/ssl/results_distill_heatmap.jsonl")]

probe_mat = np.full((len(ARCHS), len(ARCHS)), np.nan)
for r in heatmap:
    i = ARCHS.index(r["teacher_arch"])
    j = ARCHS.index(r["student_arch"])
    probe_mat[i, j] = r["probe_acc"] * 100

delta_mat = np.full((len(ARCHS), len(ARCHS)), np.nan)
for j, arch in enumerate(ARCHS):
    if arch in baseline:
        delta_mat[:, j] = probe_mat[:, j] - baseline[arch]

print("=== Avg Δ by TEACHER (rows) — how much lift each teacher provides ===")
for i, arch in enumerate(ARCHS):
    avg = np.nanmean(delta_mat[i])
    print(f"  {LABELS[arch]:<14} avg Δ = {avg:+.1f} pp")

print()
print("=== Avg Δ by STUDENT (cols) — how much each student benefits ===")
for j, arch in enumerate(ARCHS):
    avg = np.nanmean(delta_mat[:, j])
    print(f"  {LABELS[arch]:<14} avg Δ = {avg:+.1f} pp")

print()
print("=== Variance decomposition ===")
row_means = np.array([np.nanmean(delta_mat[i]) for i in range(len(ARCHS))])
col_means = np.array([np.nanmean(delta_mat[:, j]) for j in range(len(ARCHS))])
print(f"  Std of teacher avg deltas:  {row_means.std():.2f} pp")
print(f"  Std of student avg deltas:  {col_means.std():.2f} pp")
print(f"  -> Higher std = that axis drives more variance in improvement")
