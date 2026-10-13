"""Toy datasets for the curve-vs-ball probe.

Each returns (train, test) float64 arrays of shape (M, D), scaled so that the
mean squared distance to the global mean is exactly 1.0 on the train split.
That makes the reported distortion a *fraction of variance unexplained*, so
numbers are comparable across datasets of different dimension and scale, and a
value near 1.0 means "no better than predicting the global mean".

Datasets span intrinsic dimension, because that is the one thing the
curve-vs-ball trade is predicted to depend on:

    spiral2d    d=1  in D=2    one long 1-D manifold
    moons2d     d=1  in D=2    two separate 1-D manifolds
    square2d    d=2  in D=2    genuinely 2-D, the interesting middle case
    swissroll   d=2  in D=3    2-D manifold curved through 3-D
    checkerboard d=2 in D=2    2-D but disconnected -- continuity should cost
    blobs10d    d=0  in D=10   8 isotropic point clusters -- balls should win
    gauss10d    d=10 in D=10   no low-dim structure at all -- balls should win

Splits share their latent structure
-----------------------------------
Each dataset is a *sampler* built once per (name, seed); train and test are two
draws from that one sampler. This matters for `blobs10d` and would silently
ruin it otherwise: an earlier version passed the same live rng to two builder
calls, so the test split drew its own fresh cluster centres and landed nowhere
near the train centres. The symptom was distortion above 1.0 -- the model
scoring worse than the global mean -- which is the tell to watch for, since a
representation can never legitimately do that on held-out data from the same
distribution.
"""

from __future__ import annotations

import numpy as np

N_TRAIN = 2000
N_TEST = 1000


def _standardise(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = train.mean(0)
    train, test = train - mu, test - mu
    scale = np.sqrt((train ** 2).sum(1).mean())
    return train / scale, test / scale


def _spiral(rng, noise=0.02):
    def draw(n):
        t = rng.uniform(0.0, 1.0, n)
        r, a = 0.2 + 0.8 * t, 4.0 * np.pi * t
        return np.stack([r * np.cos(a), r * np.sin(a)], 1) + noise * rng.standard_normal((n, 2))
    return draw


def _moons(rng, noise=0.05):
    def draw(n):
        k = n // 2
        t1, t2 = rng.uniform(0, np.pi, k), rng.uniform(0, np.pi, n - k)
        a = np.stack([np.cos(t1), np.sin(t1)], 1)
        b = np.stack([1.0 - np.cos(t2), 0.5 - np.sin(t2)], 1)
        return np.concatenate([a, b], 0) + noise * rng.standard_normal((n, 2))
    return draw


def _square(rng):
    return lambda n: rng.uniform(0, 1, (n, 2))


def _swissroll(rng, noise=0.05):
    def draw(n):
        t = 1.5 * np.pi * (1 + 2 * rng.uniform(0, 1, n))
        h = rng.uniform(0, 21, n)
        return np.stack([t * np.cos(t), h, t * np.sin(t)], 1) + noise * rng.standard_normal((n, 3))
    return draw


def _blobs(rng, dim=10, k=8, spread=0.3):
    centres = rng.standard_normal((k, dim)) * 3.0      # drawn ONCE, shared by both splits
    def draw(n):
        return centres[rng.integers(0, k, n)] + spread * rng.standard_normal((n, dim))
    return draw


def _checkerboard(rng, tiles=4, dim=2):
    """2-D checkerboard: d=2 support, but *disconnected* into tiles.

    The other 2-D set (square2d) is one connected blob, so a curve can wander
    across all of it without ever leaving the data. A checkerboard cannot be
    traversed without crossing empty squares, which is the case where the
    curve's continuity constraint should actually cost something: any piece
    spanning two tiles pays for the gap between them.
    """
    def draw(n):
        out = np.empty((0, dim))
        while len(out) < n:
            cand = rng.uniform(0, tiles, (4 * n, dim))
            keep = (np.floor(cand).sum(1) % 2) == 0
            out = np.concatenate([out, cand[keep]], 0)
        return out[:n]
    return draw


def _gauss(rng, dim=10):
    return lambda n: rng.standard_normal((n, dim))


SAMPLERS = {
    "spiral2d": _spiral,
    "moons2d": _moons,
    "square2d": _square,
    "swissroll": _swissroll,
    "checkerboard": _checkerboard,
    "blobs10d": _blobs,
    "gauss10d": _gauss,
}

DATASETS = tuple(SAMPLERS)


def load(name: str, seed: int = 0, n_train: int | None = None,
         n_test: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return (train, test), variance-normalised, from one shared sampler.

    `n_train` is overridable because budget and points-per-cluster are otherwise
    confounded: at K=256 clusters over 2000 points each cluster sees 8 points,
    so a shrinking advantage at high budget could be estimator noise rather than
    geometry. exp3 separates them by holding the budget and varying n_train.
    """
    rng = np.random.default_rng(seed)
    draw = SAMPLERS[name](rng)
    return _standardise(draw(n_train or N_TRAIN), draw(n_test or N_TEST))


def contaminate(train: np.ndarray, eps: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Replace `eps` of the training points with uniform junk over a 3x box.

    Returns (contaminated_train, boolean mask of which points are junk). The
    test split is never touched: the point is to measure how well clean data is
    represented after junk was shown during fitting.

    Lives here rather than in the experiment file so the figures draw exactly
    the contamination the experiment fitted. An earlier duplicated helper in
    this project is what let two copies of a filter drift apart.
    """
    mask = np.zeros(len(train), dtype=bool)
    if eps <= 0:
        return train, mask
    rng = np.random.default_rng(9000 + seed)
    n = int(round(eps * len(train)))
    idx = rng.choice(len(train), n, replace=False)
    scale = 3.0 * np.abs(train).max(0)
    out = train.copy()
    out[idx] = rng.uniform(-scale, scale, (n, train.shape[1]))
    mask[idx] = True
    return out, mask
