"""EGGROLL: evolution strategies with low-rank perturbations.

The fourth way of training the same networks, and the only one here that never
computes a derivative. Instead of asking the network how the loss would change,
it perturbs the weights, measures how the loss actually changed, and moves in
whatever direction did better:

    eps_i   ~ a low-rank random matrix, scaled by sigma
    f_i      = L(theta + eps_i) - L(theta - eps_i)          (antithetic pair)
    g-hat    = 1/(P sigma^2) sum_i f_i eps_i

`g-hat` is an unbiased estimate of the true gradient up to a constant, and is
fed to Adam exactly as a real gradient would be. P antithetic pairs cost 2P
forward passes per weight update and no backward pass at all.

The EGGROLL part @sarkar2026eggroll is the shape of the perturbation. A dense
perturbation of an m x n weight matrix needs m n random numbers; EGGROLL draws
A (m x r) and B (n x r) with r much smaller and uses A B^T, which costs
(m + n) r numbers to store and move. Each perturbation is rank r, but the update
is a weighted sum of P of them, so the aggregate has rank up to P r and is
effectively full-rank. Vectors (biases) get an ordinary Gaussian perturbation.

Ported from `projects/es/experiments1.py`, which validated the estimator on
MNIST; the mechanics are unchanged and the loss is whichever loss the arm under
test uses, so the gradient being estimated is the same one backprop computes.
"""

import jax
import jax.numpy as jnp
import optax
from jax import Array

from lib.model import Params


def perturbation(key: Array, params: Params, sigma: float, rank: int) -> Params:
    """One low-rank perturbation of the whole parameter tree."""
    keys = jax.random.split(key, len(params))
    eps = []
    for k, p in zip(keys, params):
        if p.ndim == 2:
            m, n = p.shape
            ab = jax.random.normal(k, (m + n, rank))
            eps.append((ab[:m] @ ab[m:].T) * sigma)
        else:
            eps.append(jax.random.normal(k, p.shape) * sigma)
    return eps


def es_gradient(key: Array, params: Params, loss_fn, *, sigma: float,
                n_pop: int, rank: int) -> Params:
    """The antithetic low-rank ES estimate of grad(loss) at `params`."""

    def pair(pair_key):
        eps = perturbation(pair_key, params, sigma, rank)
        pos = [p + e for p, e in zip(params, eps)]
        neg = [p - e for p, e in zip(params, eps)]
        return loss_fn(pos) - loss_fn(neg), eps

    fitness, eps = jax.vmap(pair)(jax.random.split(key, n_pop))
    return [jnp.einsum("i,i...->...", fitness, e) / (n_pop * sigma ** 2) for e in eps]


def make_es_epoch(loss_of, opt, *, sigma: float, n_pop: int, rank: int):
    """An epoch of ES training as one scanned loop, matching the other arms.

    `loss_of(params, x, y)` is the arm's own loss — the same function backprop
    differentiates, so the two methods are estimating the same quantity.
    """

    def step(carry, batch):
        params, opt_state, key = carry
        x, y, = batch
        key, sub = jax.random.split(key)
        g = es_gradient(sub, params, lambda p: loss_of(p, x, y),
                        sigma=sigma, n_pop=n_pop, rank=rank)
        updates, opt_state = opt.update(g, opt_state, params)
        return (optax.apply_updates(params, updates), opt_state, key), 0.0

    def epoch(params, opt_state, key, xb, yb):
        (params, opt_state, key), _ = jax.lax.scan(step, (params, opt_state, key), (xb, yb))
        return params, opt_state, key

    return epoch
