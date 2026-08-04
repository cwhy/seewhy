"""
exp4 figures from history_exp4.pkl (no re-training).

Usage:
    uv run --no-sync python projects/sparse-attn-emergence/scripts/gen_viz_exp4.py
"""

import pickle
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

from lib.viz import (save_ablation_panel, save_attention_maps,          # noqa: E402
                     save_mechanism_panel)

with open(PROJECT / "history_exp4.pkl", "rb") as f:
    h = pickle.load(f)

S = h["A"].shape[0]
sparsity = int(h["A"].sum(1)[0])
dstep = h["diag_step"]
loss_at_d = h["loss2"][:, dstep - 1]

url_mech = save_mechanism_panel("sparse_attn_emergence_exp4_mechanism", dstep,
                                h["iou_row"], h["ent_min"], loss_at_d, sparsity)
url_abl = save_ablation_panel("sparse_attn_emergence_exp4_ablation", h["base_loss"],
                              h["abl_best"], h["abl_worst"], float(np.log(2)))

# Representative seed: the one whose emergence is closest to the median.
t_end = h["loss2"][:, -1]
seed = int(np.argmin(np.abs(h["iou_row"][:, -1] - np.median(h["iou_row"][:, -1]))))
head = int(h["best_head"][seed])
url_maps = save_attention_maps("sparse_attn_emergence_exp4_attention", h["A"],
                               h["snap_early"][seed, head], h["snap_final"][seed, head],
                               seed, head, S)

print(f"\nseeds={h['loss2'].shape[0]}  final loss2 max={t_end.max():.2e}")
print(f"iou_row final: mean {h['iou_row'][:, -1].mean():.3f}  "
      f"min {h['iou_row'][:, -1].min():.3f}  max {h['iou_row'][:, -1].max():.3f}")
print(f"iou_head final: mean {h['iou_head'][:, -1].mean():.3f}  (exp1's aggregation)")
print(f"ablation  intact {h['base_loss'].mean():.4f}  "
      f"best-removed {h['abl_best'].mean():.4f}  worst-removed {h['abl_worst'].mean():.4f}")
print(f"per-head iou spread, seed {seed}: "
      + " ".join(f"{v:.2f}" for v in h["per_head_iou"][seed]))
print(f"\nmechanism  → {url_mech}\nablation   → {url_abl}\nattention  → {url_maps}")
