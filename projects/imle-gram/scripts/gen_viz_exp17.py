"""
Visualisations for exp17 (load-balanced clustering, N_CODES=32).
Usage: uv run python projects/imle-gram/scripts/gen_viz_exp17.py
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

mod, params = load_exp("experiments17")

# ── All 32 cluster prototypes ─────────────────────────────────────────────────

imgs = mod.decode_prototypes(params)   # (32, 784)
imgs = imgs.reshape(-1, 28, 28)

rows, cols = 4, 8
fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.6))
for k, ax in enumerate(axes.flat):
    ax.imshow(imgs[k], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    ax.set_title(f"{k:x}", fontsize=7, fontfamily="monospace")
    ax.axis("off")
fig.suptitle(f"exp17 — all {mod.N_CODES} cluster prototypes (load-balanced, no collapse)", fontsize=9)
fig.tight_layout()
url = save_matplotlib_figure("exp17_prototypes", fig, dpi=120)
print(f"exp17 prototypes  {url}")
plt.close(fig)
