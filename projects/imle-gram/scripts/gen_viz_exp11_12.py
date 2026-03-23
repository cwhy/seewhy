"""
Codebook isolation + hex sample grids for exp11 (P_ZERO=0.8) and exp12 (P_ZERO=0.9).
Usage: uv run python projects/imle-gram/scripts/gen_viz_exp11_12.py
"""
import pickle, sys, numpy as np, jax, jax.numpy as jnp
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared_lib.media import save_matplotlib_figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def run(exp_name, exp_mod):
    decode       = exp_mod.decode
    hard_lookup  = exp_mod.hard_lookup
    sample_indices = exp_mod.sample_indices
    N_VARS       = exp_mod.N_VARS
    N_CODES      = exp_mod.N_CODES
    BIN_CENTERS  = exp_mod.BIN_CENTERS
    P_ZERO       = exp_mod.P_ZERO

    with open(Path(__file__).parent.parent / f"params_{exp_name}.pkl", "rb") as f:
        params = {k: jnp.array(v) for k, v in pickle.load(f).items()}

    # ── hex sample grid ───────────────────────────────────────────────────────
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
        ax.set_title("".join(f"{v:x}" for v in idx_np[i]), fontsize=6, fontfamily="monospace")
        ax.axis("off")
    fig.tight_layout()
    url = save_matplotlib_figure(f"{exp_name}_samples_hex", fig, dpi=120)
    print(f"{exp_name} samples   {url}")
    plt.close(fig)

    # ── codebook isolation grid ───────────────────────────────────────────────
    free_slots = N_VARS - 1
    all_idx = np.zeros((free_slots * N_CODES, N_VARS), dtype=np.int32)
    for slot in range(free_slots):
        for code in range(N_CODES):
            all_idx[slot * N_CODES + code, slot + 1] = code

    z     = hard_lookup(params, jnp.array(all_idx))
    probs = jax.nn.softmax(decode(params, z), axis=-1)
    imgs  = np.array((probs * BIN_CENTERS[None, None, :]).sum(-1)).reshape(-1, 28, 28)

    fig, axes = plt.subplots(free_slots, N_CODES, figsize=(N_CODES * 1.1, free_slots * 1.3))
    for slot in range(free_slots):
        for code in range(N_CODES):
            ax = axes[slot, code]
            ax.imshow(imgs[slot * N_CODES + code], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            ax.set_title(f"{code:x}", fontsize=6, fontfamily="monospace")
            ax.axis("off")
        axes[slot, 0].set_ylabel(f"s{slot+1}", fontsize=6, rotation=0, labelpad=14, va="center")
    fig.suptitle(f"{exp_name} shared codebook (P_ZERO={P_ZERO}) — all slots=0 except one", fontsize=8)
    fig.tight_layout()
    url = save_matplotlib_figure(f"{exp_name}_codebook_viz", fig, dpi=120)
    print(f"{exp_name} codebook  {url}")
    plt.close(fig)


import importlib, importlib.util

for exp_name, fname in [("exp11", "experiments11"), ("exp12", "experiments12")]:
    spec = importlib.util.spec_from_file_location(fname, Path(__file__).parent.parent / f"{fname}.py")
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    run(exp_name, mod)
