"""
Reference estimators — what a statistician who knew the function class would do.

Every one of these takes an episode's `(xs, ys)` and returns, for each k, the
prediction it would make for `y_k` after seeing only the first k pairs. Same
shape as the model's output, so "excess loss over reference" is a subtraction.

This is where the project's zero point comes from. A raw MSE of 0.31 means
nothing; 0.31 against a Bayes-optimal 0.29 means the model has essentially
solved the task, and against a Bayes-optimal 0.02 it means the model has barely
started. Half the claims in the ICL literature are unreadable because they never
draw this line.

All estimators are prefix-vectorised: the running Gram matrix and moment vector
are computed once by exclusive cumulative sum, and every prefix's fit is solved
in one batched `linalg.solve`. Cost is O(K d^3) per episode with no Python loop
over k.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def _exclusive_cumsum(a: jnp.ndarray) -> jnp.ndarray:
    """Cumulative sum over axis 1 that EXCLUDES the current element.

    Entry k aggregates elements 0…k-1, which is exactly the information a k-shot
    prediction is allowed to use. Entry 0 is all-zero — the empty prefix.
    """
    return jnp.concatenate(
        [jnp.zeros_like(a[:, :1]), jnp.cumsum(a, axis=1)[:, :-1]], axis=1
    )


def prefix_moments(xs: jnp.ndarray, ys: jnp.ndarray):
    """Running (Gram, moment) over strict prefixes.

    Returns A (B, K, d, d) with A[:, k] = sum_{i<k} x_i x_i^T
    and     b (B, K, d)    with b[:, k] = sum_{i<k} x_i y_i.
    """
    G = xs[..., :, None] * xs[..., None, :]      # (B, K, d, d)
    m = xs * ys[..., None]                       # (B, K, d)
    return _exclusive_cumsum(G), _exclusive_cumsum(m)


def zero(xs: jnp.ndarray, ys: jnp.ndarray) -> jnp.ndarray:
    """Predict the prior mean. Its loss is the scale every other loss is read against."""
    return jnp.zeros_like(ys)


def ridge(lam: float):
    """Ridge / Bayes posterior mean for a Gaussian-prior linear model.

    With w ~ N(0, tau^2 I) and y = x·w + eps, eps ~ N(0, sigma^2), the posterior
    mean is (X^T X + (sigma^2/tau^2) I)^{-1} X^T y, so `lam = sigma^2 / tau^2`
    makes this estimator *exactly* Bayes-optimal — not merely a good baseline.
    Noiseless tasks pass a small positive lam purely to keep the k < d prefixes
    (where the Gram matrix is singular) solvable; there it is the minimum-norm
    interpolant, which is also the posterior mean in the limit.
    """

    def estimate(xs, ys):
        A, b = prefix_moments(xs, ys)
        d = xs.shape[-1]
        w = jnp.linalg.solve(A + lam * jnp.eye(d), b)     # (B, K, d)
        return jnp.einsum("bkd,bkd->bk", xs, w)

    return estimate


def lasso(lam: float, iters: int = 200):
    """ISTA for the L1-penalised fit — the reference for sparse linear tasks.

    Approximate, and labelled as such wherever it is reported: a fixed iteration
    budget with a step size read off the Gram trace, not a converged solver and
    not Bayes-optimal for a spike-and-slab prior. It is here because the honest
    alternative — ridge — is *known* to be beatable on sparse problems, which
    would flatter the model for no reason.
    """

    def estimate(xs, ys):
        A, b = prefix_moments(xs, ys)
        # 1/L step size, L bounded by the Gram trace (>= its largest eigenvalue).
        eta = 1.0 / (jnp.trace(A, axis1=-2, axis2=-1) + 1e-6)[..., None]

        def body(_, w):
            grad = jnp.einsum("bkde,bke->bkd", A, w) - b
            z = w - eta * grad
            return jnp.sign(z) * jnp.maximum(jnp.abs(z) - eta * lam, 0.0)

        w = jax.lax.fori_loop(0, iters, body, jnp.zeros_like(b))
        return jnp.einsum("bkd,bkd->bk", xs, w)

    return estimate


def knn(n_neighbours: int = 3):
    """k-nearest-neighbour average over the prefix — the reference for tasks with
    no closed form (ReLU teachers, sinusoids).

    Deliberately weak. A model beating kNN has learned something about the
    function class; a model losing to kNN has not learned to use the context at
    all. Read it as a floor on competence, never as an optimum.
    """

    def estimate(xs, ys):
        B, K, _ = xs.shape
        d2 = jnp.sum((xs[:, :, None, :] - xs[:, None, :, :]) ** 2, axis=-1)  # (B,K,K)
        strict_prefix = jnp.tril(jnp.ones((K, K), bool), -1)                 # [k, i] = i < k
        d2 = jnp.where(strict_prefix, d2, jnp.inf)

        neg_d2, idx = jax.lax.top_k(-d2, min(n_neighbours, K))               # (B,K,n)
        valid = jnp.isfinite(neg_d2)
        neigh_y = jnp.take_along_axis(jnp.broadcast_to(ys[:, None, :], (B, K, K)), idx, axis=2)
        total = jnp.sum(jnp.where(valid, neigh_y, 0.0), axis=-1)
        count = jnp.sum(valid, axis=-1)
        return jnp.where(count > 0, total / jnp.maximum(count, 1), 0.0)      # 0-shot -> prior mean

    return estimate
