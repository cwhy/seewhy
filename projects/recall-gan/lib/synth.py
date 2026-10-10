"""The synthetic prior, taken from recall-gen (projects/recall-gen/lib/domains.py,
`_synth_world` / `_synth_pool`), generated at MNIST's width so a network trained
on it can be scored on MNIST directly.

A *world* is a random latent-factor model: k latent dimensions (log-uniform on
[k_min, 64]), a random mixing matrix with a power-law spectrum, a random bias,
one of four pointwise nonlinearities, noise, then rescaling to [0, 1]. Each
world yields `per_world` items. At width 784 (not a multiple of 13) recall-gen's
one-hot "simplex" mode never fires, so every item is continuous.

Training episodes take all 16 context items from one world, as in recall-gen.
"""

import numpy as np

NONLIN = ("identity", "tanh", "relu", "sign")
WIDTH = 784
TRAIN_WORLDS, TEST_WORLDS, PER_WORLD = 1024, 128, 48


def synth_world(rng, width: int, n_items: int, k_min: float = 1.0) -> np.ndarray:
    """recall-gen's `_synth_world`, continuous mode only."""
    k = int(np.clip(np.exp(rng.uniform(np.log(k_min), np.log(64.0))), 1, 64))
    spec = (np.arange(1, k + 1) ** -rng.uniform(0.0, 2.0)).astype(np.float32)
    A = (rng.standard_normal((width, k)) * spec).astype(np.float32)
    b = (rng.standard_normal(width) * rng.uniform(0.0, 1.0)).astype(np.float32)
    h = rng.standard_normal((n_items, k)).astype(np.float32) @ A.T + b
    nl = NONLIN[rng.integers(len(NONLIN))]
    if nl == "tanh":
        h = np.tanh(h)
    elif nl == "relu":
        h = np.maximum(h, 0.0)
    elif nl == "sign":
        h = np.sign(h)
    h = h + rng.standard_normal(h.shape).astype(np.float32) * rng.uniform(0.0, 0.3)
    lo, hi = h.min(), h.max()
    return ((h - lo) / max(float(hi - lo), 1e-6)).astype(np.float32)


def synth_pools(k_min: float = 1.0):
    """(train (1024, 48, 784), test (128, 48, 784)) — worlds as the first axis.
    Seeds 7 and 8, as recall-gen."""
    out = []
    for n_w, seed in ((TRAIN_WORLDS, 7), (TEST_WORLDS, 8)):
        rng = np.random.default_rng(seed)
        out.append(np.stack([synth_world(rng, WIDTH, PER_WORLD, k_min) for _ in range(n_w)]))
    return tuple(out)
