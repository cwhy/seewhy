"""
Visualise each codebook entry in isolation for exp9.
Grid: rows = free slots (1–15), cols = codes (0–15).
Each image has all slots = 0 except the one being varied.
Usage: uv run python projects/imle-gram/scripts/gen_codebook_viz.py
"""
import pickle, sys, numpy as np, jax, jax.numpy as jnp
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments9 import decode, N_VARS, N_CODES, BIN_CENTERS, hard_lookup
from shared_lib.media import save_matplotlib_figure

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open(Path(__file__).parent.parent / "params_exp9.pkl", "rb") as f:
    params = {k: jnp.array(v) for k, v in pickle.load(f).items()}

free_slots = N_VARS - 1   # slots 1..N_VARS-1
n_rows, n_cols = free_slots, N_CODES   # 15 × 16

# Build index matrix: (n_rows * n_cols, N_VARS), all zeros except varied slot
all_idx = np.zeros((n_rows * n_cols, N_VARS), dtype=np.int32)
for slot in range(free_slots):        # slot offset 1 in the latent vector
    for code in range(N_CODES):
        row = slot * N_CODES + code
        all_idx[row, slot + 1] = code  # slot 0 stays 0; vary slot+1

z      = hard_lookup(params, jnp.array(all_idx))
logits = decode(params, z)
probs  = jax.nn.softmax(logits, axis=-1)
imgs   = np.array((probs * BIN_CENTERS[None, None, :]).sum(-1)).reshape(-1, 28, 28)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.1, n_rows * 1.3))
for slot in range(n_rows):
    for code in range(n_cols):
        ax  = axes[slot, code]
        img = imgs[slot * N_CODES + code]
        ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.set_title(f"{code:x}", fontsize=6, fontfamily="monospace")
        ax.axis("off")
    axes[slot, 0].set_ylabel(f"s{slot+1}", fontsize=6, rotation=0, labelpad=14, va="center")

fig.suptitle("exp9 shared codebook — all slots=0 except one", fontsize=8)
fig.tight_layout()

url = save_matplotlib_figure("exp9_codebook_viz", fig, dpi=120)
print(f"url   {url}")
