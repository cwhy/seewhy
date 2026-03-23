"""
Visualisations for exp14 (Hebbian, n_active=1-2).
Usage: uv run python projects/imle-gram/scripts/gen_viz_exp14.py
"""
import pickle, sys, numpy as np, jax, jax.numpy as jnp
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments14 import decode, N_VARS, N_CODES, BIN_CENTERS, hard_lookup, sample_indices
from shared_lib.media import save_matplotlib_figure

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open(Path(__file__).parent.parent / "params_exp14.pkl", "rb") as f:
    params = {k: jnp.array(v) for k, v in pickle.load(f).items()}

# ── 1. Hex sample grid ────────────────────────────────────────────────────────

rows, cols = 5, 8
key  = jax.random.PRNGKey(0)
idx  = sample_indices(key, rows * cols)
z    = hard_lookup(params, idx)
probs = jax.nn.softmax(decode(params, z), axis=-1)
imgs  = np.array((probs * BIN_CENTERS[None, None, :]).sum(-1)).reshape(rows * cols, 28, 28)
idx_np = np.array(idx)

fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.8))
for i, ax in enumerate(axes.flat):
    ax.imshow(imgs[i], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    ax.set_title("".join(f"{v:x}" for v in idx_np[i]), fontsize=5, fontfamily="monospace")
    ax.axis("off")
fig.tight_layout()
url = save_matplotlib_figure("exp14_samples_hex", fig, dpi=120)
print(f"samples   {url}")
plt.close(fig)

# ── 2. Codebook isolation (1 active slot) ─────────────────────────────────────
# For Hebbian with 1 active, M = u⊗v — pure rank-1.
# Show one representative slot (slot 1) across all codes.

fig, axes = plt.subplots(1, N_CODES, figsize=(N_CODES * 1.0, 1.5))
all_idx = np.zeros((N_CODES, N_VARS), dtype=np.int32)
for code in range(N_CODES):
    all_idx[code, 1] = code   # slot 1, all other slots 0

z     = hard_lookup(params, jnp.array(all_idx))
probs = jax.nn.softmax(decode(params, z), axis=-1)
imgs  = np.array((probs * BIN_CENTERS[None, None, :]).sum(-1)).reshape(-1, 28, 28)

for code, ax in enumerate(axes):
    ax.imshow(imgs[code], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    ax.set_title(f"{code:x}", fontsize=6, fontfamily="monospace")
    ax.axis("off")
fig.suptitle("exp14 — single active slot (slot 1, codes 0–1f)", fontsize=8)
fig.tight_layout()
url = save_matplotlib_figure("exp14_codebook_single", fig, dpi=120)
print(f"codebook single  {url}")
plt.close(fig)

# ── 3. Pairwise combinations: slot 1 × slot 2 (code sweep) ───────────────────
# rows = code for slot 1, cols = code for slot 2 (first 8 of each)

n_show = 8
all_idx = np.zeros((n_show * n_show, N_VARS), dtype=np.int32)
for i in range(n_show):
    for j in range(n_show):
        all_idx[i * n_show + j, 1] = i + 1   # slot 1: codes 1..8
        all_idx[i * n_show + j, 2] = j + 1   # slot 2: codes 1..8

z     = hard_lookup(params, jnp.array(all_idx))
probs = jax.nn.softmax(decode(params, z), axis=-1)
imgs  = np.array((probs * BIN_CENTERS[None, None, :]).sum(-1)).reshape(-1, 28, 28)

fig, axes = plt.subplots(n_show, n_show, figsize=(n_show * 1.1, n_show * 1.1))
for i in range(n_show):
    for j in range(n_show):
        ax = axes[i, j]
        ax.imshow(imgs[i * n_show + j], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.axis("off")
    axes[i, 0].set_ylabel(f"s1={i+1:x}", fontsize=6, rotation=0, labelpad=18, va="center")
for j in range(n_show):
    axes[0, j].set_title(f"s2={j+1:x}", fontsize=6)
fig.suptitle("exp14 — rank-2: slot1 × slot2 combinations", fontsize=8)
fig.tight_layout()
url = save_matplotlib_figure("exp14_codebook_pairs", fig, dpi=120)
print(f"codebook pairs   {url}")
