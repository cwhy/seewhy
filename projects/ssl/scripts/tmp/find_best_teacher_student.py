"""Find best teacher-student pairs across all heatmap experiments."""
import json
import numpy as np
from pathlib import Path

SSL = Path("projects/ssl")

ARCH_LABELS = {
    "mlp": "MLP", "mlp2": "MLP-2", "conv": "Conv", "conv2": "Conv-2",
    "vit": "ViT", "vit_patch": "ViT-patch",
    "gram_dual": "Gram-dual", "gram_single": "Gram-single", "gram_glu": "Gram-GLU",
}

# Load all heatmap results
all_rows = []
for t in ["ema", "sigreg", "dae", "random"]:
    f = SSL / f"results_distill_heatmap{'_'+t if t != 'ema' else ''}.jsonl"
    if f.exists():
        for line in open(f):
            r = json.loads(line)
            r["teacher_type"] = t
            all_rows.append(r)

print(f"Total pairs: {len(all_rows)}")

# Top 20 by probe accuracy across all teacher types
print("\n=== Top 20 pairs by probe accuracy (all teacher types) ===")
top = sorted(all_rows, key=lambda r: r["probe_acc"], reverse=True)[:20]
for r in top:
    print(f"  {r['teacher_type']:8} {ARCH_LABELS[r['teacher_arch']]:14} -> "
          f"{ARCH_LABELS[r['student_arch']]:14}  probe={r['probe_acc']:.2%}  knn={r['knn_1nn_acc']:.2%}")

# Exclude random teachers (noisy signal) — best non-random pairs
print("\n=== Top 10 pairs (excluding random teacher) ===")
non_rand = [r for r in all_rows if r["teacher_type"] != "random"]
top_nr = sorted(non_rand, key=lambda r: r["probe_acc"], reverse=True)[:10]
for r in top_nr:
    print(f"  {r['teacher_type']:8} {ARCH_LABELS[r['teacher_arch']]:14} -> "
          f"{ARCH_LABELS[r['student_arch']]:14}  probe={r['probe_acc']:.2%}  knn={r['knn_1nn_acc']:.2%}")

# For chain distillation: we want a student with high probe but different arch than teacher
# so the sub-student has a meaningful "cross-arch" signal to learn from
print("\n=== Top cross-arch pairs (teacher != student arch, non-random) ===")
cross = [r for r in non_rand if r["teacher_arch"] != r["student_arch"]]
top_cross = sorted(cross, key=lambda r: r["probe_acc"], reverse=True)[:10]
for r in top_cross:
    print(f"  {r['teacher_type']:8} {ARCH_LABELS[r['teacher_arch']]:14} -> "
          f"{ARCH_LABELS[r['student_arch']]:14}  probe={r['probe_acc']:.2%}  knn={r['knn_1nn_acc']:.2%}")
