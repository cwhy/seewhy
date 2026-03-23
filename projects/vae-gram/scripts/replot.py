"""
Reload saved params and regenerate visualisations without retraining.

Usage:
    uv run python projects/vae-gram/scripts/replot.py --exp 2
"""

import sys, argparse, pickle
from pathlib import Path
import importlib.util

import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

from shared_lib.datasets import load_supervised_image


def load_exp(n: int):
    """Import experiments<n>.py as a module."""
    path = Path(__file__).parent.parent / f"experiments{n}.py"
    spec = importlib.util.spec_from_file_location(f"exp{n}", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_params(n: int):
    path = Path(__file__).parent.parent / f"params_exp{n}.pkl"
    with open(path, "rb") as f:
        raw = pickle.load(f)
    return {k: jnp.array(v) for k, v in raw.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=int, default=2, help="Experiment number")
    args = parser.parse_args()

    mod    = load_exp(args.exp)
    params = load_params(args.exp)

    data   = load_supervised_image("mnist")
    X_test = data.X_test.reshape(data.n_test_samples, 784).astype(jnp.float32)
    y_test = data.y_test

    mod.save_reconstruction_grid(params, X_test, "replot")
    mod.save_pixel_entropy_grid(params, X_test, "replot")
    mod.save_latent_pca(params, X_test, y_test, "replot")
