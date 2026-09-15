"""MNIST loading, preprocessed exactly as the PC-ALM reference implementation does.

Two details matter for comparability with the numbers we are replicating:

* per-pixel standardisation with the torchvision MNIST constants
  (mean 0.1307, std 0.3081), not a plain [0, 1] rescale;
* class-balanced subsets, so a 1024-image smoke subset still has ~102 of each
  digit rather than whatever the shuffle happened to draw.

Forward-Forward additionally needs the label written *into* the image, which is
what `overlay_label` does: it replaces the first ten pixels with a one-hot
label, in the raw pixel domain, before standardisation is undone. In normalised
units that means 2.8215 for the hot pixel and -0.4242 for the rest.
"""

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import Array

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

# (mean, std) per dataset, as the PC-ALM reference implementation uses them
STATS = {"mnist": (0.1307, 0.3081), "fashion_mnist": (0.2860, 0.3530)}

PIXEL_HI = (1.0 - MNIST_MEAN) / MNIST_STD     # 2.8215 — a white pixel, normalised
PIXEL_LO = (0.0 - MNIST_MEAN) / MNIST_STD     # -0.4242 — a black pixel, normalised


class Data(NamedTuple):
    x_train: Array    # (n_train, 784) float32, standardised
    y_train: Array    # (n_train, 10)  float32 one-hot
    l_train: Array    # (n_train,)     int32 labels
    x_test: Array
    y_test: Array
    l_test: Array


def balanced_indices(labels: np.ndarray, n: int, seed: int) -> np.ndarray:
    """`n` indices with as equal a class split as `n` allows, then shuffled."""
    rng = np.random.default_rng(seed)
    if n >= labels.shape[0]:
        idx = np.arange(labels.shape[0], dtype=np.int64)
        rng.shuffle(idx)
        return idx
    base, remainder = divmod(n, 10)
    out = []
    for cls in range(10):
        take = base + (1 if cls < remainder else 0)
        cls_idx = np.flatnonzero(labels == cls)
        rng.shuffle(cls_idx)
        out.append(cls_idx[:take])
    idx = np.concatenate(out)
    rng.shuffle(idx)
    return idx.astype(np.int64)


def load_mnist(n_train: int = 60000, n_test: int = 10000, seed: int = 0,
               dataset: str = "mnist") -> Data:
    from shared_lib.datasets import load_supervised_image

    mean, std = STATS[dataset]
    ds = load_supervised_image(dataset)
    x_tr = np.asarray(ds.X).reshape(ds.n_samples, -1)
    l_tr = np.asarray(ds.y).astype(np.int32)
    x_te = np.asarray(ds.X_test).reshape(ds.n_test_samples, -1)
    l_te = np.asarray(ds.y_test).astype(np.int32)

    tr = balanced_indices(l_tr, n_train, seed)
    te = balanced_indices(l_te, n_test, seed + 17)

    def prep(x):
        return ((x.astype(np.float32) / 255.0 - mean) / std).astype(np.float32)

    def onehot(l):
        return np.eye(10, dtype=np.float32)[l]

    return Data(
        jnp.asarray(prep(x_tr[tr])), jnp.asarray(onehot(l_tr[tr])), jnp.asarray(l_tr[tr]),
        jnp.asarray(prep(x_te[te])), jnp.asarray(onehot(l_te[te])), jnp.asarray(l_te[te]),
    )


def overlay_label(x: Array, onehot: Array) -> Array:
    """Write a one-hot label into the first ten pixels (Hinton 2022, §3.3)."""
    hot = PIXEL_LO + (PIXEL_HI - PIXEL_LO) * onehot
    hot = jnp.broadcast_to(hot, x.shape[:-1] + (10,))
    return jnp.concatenate([hot, x[..., 10:]], axis=-1)
