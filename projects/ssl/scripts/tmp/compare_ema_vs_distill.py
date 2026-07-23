"""Compare EMA-JEPA 1024 baselines vs best distillation result per student arch."""
import json
import numpy as np

rows = [json.loads(l) for l in open("projects/ssl/results.jsonl")]
heatmap = [json.loads(l) for l in open("projects/ssl/results_distill_heatmap.jsonl")]

# EMA-JEPA baselines at 1024 — map encoder arch -> probe_acc
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

baseline = {}
for r in rows:
    for arch, exp in ARCH_EXP.items():
        if r["experiment"] == exp:
            baseline[arch] = r["final_probe_acc"]

# Best distillation result per student arch
best_distill = {}
for r in heatmap:
    s = r["student_arch"]
    if s not in best_distill or r["probe_acc"] > best_distill[s]["probe_acc"]:
        best_distill[s] = r

ARCHS = ["mlp", "mlp2", "conv", "conv2", "vit", "vit_patch", "gram_dual", "gram_single", "gram_glu"]
LABELS = {"mlp": "MLP", "mlp2": "MLP-2", "conv": "Conv", "conv2": "Conv-2",
          "vit": "ViT", "vit_patch": "ViT-patch",
          "gram_dual": "Gram-dual", "gram_single": "Gram-single", "gram_glu": "Gram-GLU"}

print(f"{'Student':<14} {'EMA-JEPA':>10} {'Best distill':>13} {'Delta':>8} {'Best teacher'}")
print("-" * 65)
for arch in ARCHS:
    b = baseline.get(arch, float("nan"))
    d = best_distill.get(arch, {})
    dp = d.get("probe_acc", float("nan"))
    delta = dp - b
    teacher = d.get("teacher_arch", "?")
    print(f"{LABELS[arch]:<14} {b:>10.1%} {dp:>13.1%} {delta:>+7.1%}   {LABELS.get(teacher, teacher)}")
