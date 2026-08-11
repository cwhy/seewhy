"""
The contract between tasks and models.

Everything in this project passes through one sequence layout, so that a task
knows nothing about architectures and an architecture knows nothing about tasks:

    x_1  y_1  x_2  y_2  ...  x_K  y_K          L = 2K tokens

Each token is a vector of width `d_x + 2`:

    x-token   [ x        , 0 , 0 ]
    y-token   [ 0 ... 0  , y , 1 ]

The trailing flag is not decoration. A zero-padded layout (Garg et al.'s) leaves
x- and y-tokens distinguishable only by *where* they sit, which is free for a
transformer with position embeddings and free for a static positional mixer, but
a GRU or a linear-attention stack has no position channel at all and would have
to infer parity from the residual stream. Making the type explicit removes a
confound from the architecture comparison that has nothing to do with in-context
learning.

The model emits one scalar per token; only the outputs at x-positions are read.
The output above x_k has seen x_1,y_1,…,x_{k-1},y_{k-1},x_k — every pair before
k, and the query, and no answer. So `read_predictions(out)[:, k]` is a **k-shot**
prediction, k = 0 … K-1, and the loss curve over that index is the measurement
this whole project is built on.
"""

from __future__ import annotations

import jax.numpy as jnp


def d_in(d_x: int) -> int:
    """Token width for a task with `d_x`-dimensional inputs."""
    return d_x + 2


def to_sequence(xs: jnp.ndarray, ys: jnp.ndarray) -> jnp.ndarray:
    """(B, K, d_x), (B, K) -> (B, 2K, d_x + 2), interleaved x/y tokens."""
    B, K, d_x = xs.shape
    x_tok = jnp.concatenate([xs, jnp.zeros((B, K, 2))], axis=-1)
    y_tok = jnp.concatenate(
        [jnp.zeros((B, K, d_x)), ys[..., None], jnp.ones((B, K, 1))], axis=-1
    )
    # (B, K, 2, d_in) -> (B, 2K, d_in): [x_1, y_1, x_2, y_2, ...]
    return jnp.stack([x_tok, y_tok], axis=2).reshape(B, 2 * K, d_x + 2)


def read_predictions(out: jnp.ndarray) -> jnp.ndarray:
    """(B, 2K, 1) -> (B, K). Entry k is the k-shot prediction of y_{k+1}."""
    return out[:, ::2, 0]


def shot_counts(n_points: int) -> jnp.ndarray:
    """The x-axis of every ICL curve: 0, 1, …, K-1 in-context examples."""
    return jnp.arange(n_points)
