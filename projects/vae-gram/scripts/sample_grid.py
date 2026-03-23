"""
Generate a grid of samples from the prior (z ~ N(0,1)) using the best saved params.

Usage:
    uv run python projects/vae-gram/scripts/sample_grid.py
    uv run python projects/vae-gram/scripts/sample_grid.py --exp 11
    uv run python projects/vae-gram/scripts/sample_grid.py --rows 5 --cols 8 --seed 7
"""

import sys, argparse, pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
from shared_lib.media import save_matplotlib_figure

PROJ = Path(__file__).parent.parent


def load_exp(exp_name: str):
    """Import decode fn and hyperparams from an experiment module, load its params."""
    import importlib.util
    py = PROJ / f"experiments{exp_name.lstrip('exp')}.py"
    spec = importlib.util.spec_from_file_location(f"exp_{exp_name}", py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    pkl = PROJ / f"params_{exp_name}.pkl"
    with open(pkl, "rb") as f:
        raw = pickle.load(f)
    params = {k: jnp.array(v) for k, v in raw.items()}

    return mod, params


def make_grid(mod, params, rows: int, cols: int, seed: int):
    n = rows * cols
    z = jax.random.normal(jax.random.PRNGKey(seed), (n, mod.LATENT_DIM))

    logits = mod.decode(params, z)                                     # (n, 784, N_VAL_BINS)
    probs  = jax.nn.softmax(logits, axis=-1)                           # (n, 784, N_VAL_BINS)
    imgs   = np.array((probs * mod.BIN_CENTERS[None, None, :]).sum(-1))  # (n, 784)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.4, rows * 1.4))
    for i, ax in enumerate(axes.flat):
        ax.imshow(imgs[i].reshape(28, 28), cmap="gray", vmin=0, vmax=255)
        ax.axis("off")
    fig.suptitle(f"Prior samples — {mod.__name__} (seed={seed})", fontsize=10)
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp",  default="exp14", help="experiment name, e.g. exp14")
    parser.add_argument("--rows", type=int, default=5)
    parser.add_argument("--cols", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    exp_name = args.exp if args.exp.startswith("exp") else f"exp{args.exp}"
    print(f"Loading {exp_name}…")
    mod, params = load_exp(exp_name)
    print(f"  LATENT_DIM={mod.LATENT_DIM}, D={mod.D}")

    print(f"Generating {args.rows}×{args.cols} sample grid…")
    fig = make_grid(mod, params, args.rows, args.cols, args.seed)

    tag = f"{exp_name}_samples_{args.rows}x{args.cols}_seed{args.seed}"
    url = save_matplotlib_figure(tag, fig, format="png", dpi=150)
    plt.close(fig)
    print(f"  → {url}")
