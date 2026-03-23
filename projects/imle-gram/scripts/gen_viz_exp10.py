"""
Two visualisations for exp10 (P_ZERO=0.5 shared codebook):
  1. Random sample grid with hex codes as titles.
  2. Codebook isolation grid: all slots=0 except one varied through 0..15.
Usage: uv run python projects/imle-gram/scripts/gen_viz_exp10.py
"""
import pickle, sys, numpy as np, jax, jax.numpy as jnp
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments10 import decode, N_VARS, N_CODES, BIN_CENTERS, hard_lookup, sample_indices
from shared_lib.media import save_matplotlib_figure

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open(Path(__file__).parent.parent / "params_exp10.pkl", "rb") as f:
    params = {k: jnp.array(v) for k, v in pickle.load(f).items()}

# ── 1. Hex sample grid ────────────────────────────────────────────────────────

rows, cols = 5, 8
key = jax.random.PRNGKey(0)
idx = sample_indices(key, rows * cols)
z   = hard_lookup(params, idx)
probs = jax.nn.softmax(decode(params, z), axis=-1)
imgs  = np.array((probs * BIN_CENTERS[None, None, :]).sum(-1)).reshape(rows * cols, 28, 28)
idx_np = np.array(idx)

fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.8))
for i, ax in enumerate(axes.flat):
    ax.imshow(imgs[i], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    ax.set_title("".join(f"{v:x}" for v in idx_np[i]), fontsize=6, fontfamily="monospace")
    ax.axis("off")
fig.tight_layout()
url = save_matplotlib_figure("exp10_samples_hex", fig, dpi=120)
print(f"samples  {url}")
plt.close(fig)

# ── 2. Codebook isolation grid ────────────────────────────────────────────────

free_slots = N_VARS - 1   # slots 1..N_VARS-1
n_rows, n_cols = free_slots, N_CODES   # 15 × 16

all_idx = np.zeros((n_rows * n_cols, N_VARS), dtype=np.int32)
for slot in range(free_slots):
    for code in range(N_CODES):
        all_idx[slot * N_CODES + code, slot + 1] = code

z      = hard_lookup(params, jnp.array(all_idx))
probs  = jax.nn.softmax(decode(params, z), axis=-1)
imgs   = np.array((probs * BIN_CENTERS[None, None, :]).sum(-1)).reshape(-1, 28, 28)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.1, n_rows * 1.3))
for slot in range(n_rows):
    for code in range(n_cols):
        ax = axes[slot, code]
        ax.imshow(imgs[slot * N_CODES + code], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.set_title(f"{code:x}", fontsize=6, fontfamily="monospace")
        ax.axis("off")
    axes[slot, 0].set_ylabel(f"s{slot+1}", fontsize=6, rotation=0, labelpad=14, va="center")
fig.suptitle("exp10 shared codebook (P_ZERO=0.5) — all slots=0 except one", fontsize=8)
fig.tight_layout()
url = save_matplotlib_figure("exp10_codebook_viz", fig, dpi=120)
print(f"codebook {url}")
