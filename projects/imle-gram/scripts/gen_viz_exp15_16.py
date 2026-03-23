"""
Visualisations for exp15 (N_VARS=2) and exp16 (pure clustering).
Usage: uv run python projects/imle-gram/scripts/gen_viz_exp15_16.py
"""
import pickle, sys, numpy as np, jax, jax.numpy as jnp
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared_lib.media import save_matplotlib_figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import importlib.util

def load_exp(name):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).parent.parent / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    exp_name = name.replace("experiments", "exp")
    with open(Path(__file__).parent.parent / f"params_{exp_name}.pkl", "rb") as f:
        params = {k: jnp.array(v) for k, v in pickle.load(f).items()}
    return mod, params

# ── exp15: sample grid + pairwise combinations ────────────────────────────────

mod15, p15 = load_exp("experiments15")

rows, cols = 5, 8
key = jax.random.PRNGKey(0)
idx = mod15.sample_indices(key, rows * cols)
z   = mod15.hard_lookup(p15, idx)
probs = jax.nn.softmax(mod15.decode(p15, z), axis=-1)
imgs  = np.array((probs * mod15.BIN_CENTERS[None,None,:]).sum(-1)).reshape(rows*cols, 28, 28)
idx_np = np.array(idx)

fig, axes = plt.subplots(rows, cols, figsize=(cols*2.2, rows*2.8))
for i, ax in enumerate(axes.flat):
    ax.imshow(imgs[i], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    ax.set_title("".join(f"{v:x}" for v in idx_np[i]), fontsize=6, fontfamily="monospace")
    ax.axis("off")
fig.tight_layout()
url = save_matplotlib_figure("exp15_samples_hex", fig, dpi=120)
print(f"exp15 samples  {url}")
plt.close(fig)

# exp15: all pairs (N_CODES×N_CODES = 32×32 is too big, show 16×16 subset)
n_show = 16
all_idx = np.zeros((n_show * n_show, 2), dtype=np.int32)
for i in range(n_show):
    for j in range(n_show):
        all_idx[i * n_show + j] = [i + 1, j + 1]
z     = mod15.hard_lookup(p15, jnp.array(all_idx))
probs = jax.nn.softmax(mod15.decode(p15, z), axis=-1)
imgs  = np.array((probs * mod15.BIN_CENTERS[None,None,:]).sum(-1)).reshape(-1, 28, 28)

fig, axes = plt.subplots(n_show, n_show, figsize=(n_show*0.9, n_show*0.9))
for i in range(n_show):
    for j in range(n_show):
        ax = axes[i, j]
        ax.imshow(imgs[i*n_show+j], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.axis("off")
    axes[i, 0].set_ylabel(f"{i+1:x}", fontsize=5, rotation=0, labelpad=10, va="center")
for j in range(n_show):
    axes[0, j].set_title(f"{j+1:x}", fontsize=5)
fig.suptitle("exp15 rank-2: code_0 × code_1 (codes 1–f)", fontsize=8)
fig.tight_layout()
url = save_matplotlib_figure("exp15_pairs", fig, dpi=120)
print(f"exp15 pairs    {url}")
plt.close(fig)

# ── exp16: all 32 cluster prototypes ─────────────────────────────────────────

mod16, p16 = load_exp("experiments16")

all_idx = jnp.arange(mod16.N_CODES).reshape(-1, 1)
z     = mod16.hard_lookup(p16, all_idx)
probs = jax.nn.softmax(mod16.decode(p16, z), axis=-1)
imgs  = np.array((probs * mod16.BIN_CENTERS[None,None,:]).sum(-1)).reshape(-1, 28, 28)

rows16, cols16 = 4, 8   # 32 prototypes in a 4×8 grid
fig, axes = plt.subplots(rows16, cols16, figsize=(cols16*1.5, rows16*1.6))
for k, ax in enumerate(axes.flat):
    ax.imshow(imgs[k], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    ax.set_title(f"{k:x}", fontsize=7, fontfamily="monospace")
    ax.axis("off")
fig.suptitle(f"exp16 — all {mod16.N_CODES} cluster prototypes (M = u_k ⊗ v_k)", fontsize=9)
fig.tight_layout()
url = save_matplotlib_figure("exp16_prototypes", fig, dpi=120)
print(f"exp16 prototypes  {url}")
