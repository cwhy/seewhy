"""Delta: distill result vs teacher's EMA-JEPA baseline."""
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

# Delta vs teacher baseline (rows)
print("Δ = student_distill_probe − teacher_EMA_JEPA_baseline")
print(f"{'':14}", end="")
for a in ARCHS:
    print(f" {LABELS[a]:>11}", end="")
print()
print("-" * (14 + 12 * len(ARCHS)))
for i, ta in enumerate(ARCHS):
    print(f"{LABELS[ta]:<14}", end="")
    for j in range(len(ARCHS)):
        delta = probe_mat[i, j] - baseline[ta]
        print(f" {delta:>+10.1f}%", end="")
    print(f"  teacher={baseline[ta]:.1f}%")
