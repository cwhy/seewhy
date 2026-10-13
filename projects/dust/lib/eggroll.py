"""EGGROLL for the transformer, with the same setting as `projects/local-learning/lib/es.py`.

That file is this repo's EGGROLL (Sarkar et al. 2025, arXiv:2511.16652) and this
one changes nothing about the estimator, only the network it is applied to:

    eps_i  = sigma * a_i b_i^T        rank 1; a_i [out], b_i [in] standard normal,
                                      for every weight matrix (embeddings and the
                                      head included); vectors get a dense Gaussian
    f_i    = L(theta + eps_i) - L(theta - eps_i)          antithetic pair, raw loss
    g-hat  = 1/(P sigma^2) sum_i f_i eps_i                P = number of pairs
    update = Adam(g-hat)

`g-hat` is twice the gradient in expectation (an antithetic difference is 2 eps.g),
which Adam's per-coordinate normalisation absorbs, as `reports/04-pcalm-eggroll-audit`
in local-learning documents. Fitness is the raw loss difference: no rank shaping and
no z-scoring, exactly as there.

Population is counted in *forward passes of the batch*, as the DUST paper counts
EGGROLL: P pairs cost 2P forwards. Every member sees the same batch.

What is different in a transformer is only cost. A member perturbs ~60 matrices,
and the rank-1 factors let each be applied as a side branch (see `model.LR`), so a
member's forward is the ordinary forward plus two thin products per layer.
"""

from functools import partial

import jax
import jax.numpy as jnp
import optax
from jax import Array, lax

from lib.model import BF16, F32, LR, Cfg, cast, forward_loss, rope_tables


def _is_matrix(x) -> bool:
    return x.ndim == 2


def sample_factors(key: Array, params: dict, n: int) -> dict:
    """Perturbation factors for n pairs: (a [n,out], b [n,in]) per matrix, [n,L] per vector."""
    leaves, treedef = jax.tree_util.tree_flatten(params)
    keys = jax.random.split(key, len(leaves))
    out = []
    for k, x in zip(keys, leaves):
        if _is_matrix(x):
            ka, kb = jax.random.split(k)
            out.append((jax.random.normal(ka, (n, x.shape[0]), F32), jax.random.normal(kb, (n, x.shape[1]), F32)))
        else:
            out.append(jax.random.normal(k, (n,) + x.shape, F32))
    return jax.tree_util.tree_unflatten(treedef, out)


def _is_factor(x) -> bool:
    return isinstance(x, tuple) or hasattr(x, "shape")


def perturbed(pb: dict, fac: dict, s: Array) -> dict:
    """Parameter tree for one member: matrices become `LR`, vectors get s * noise."""
    def one(w, f):
        if isinstance(f, tuple):
            return LR(w, f[0], f[1], s)
        return w + (s * f).astype(w.dtype)
    return jax.tree_util.tree_map(one, pb, fac, is_leaf=lambda x: isinstance(x, tuple) or hasattr(x, "shape"))


def _contribution(fit: Array, f):
    """sum_i fit_i * a_i b_i^T for a matrix, sum_i fit_i * noise_i for a vector."""
    if isinstance(f, tuple):
        return jnp.einsum("p,po,pi->oi", fit, f[0], f[1])
    return jnp.einsum("p,p...->...", fit, f)


@partial(jax.jit, static_argnames=("cfg", "n_pairs", "chunk", "dtype"))
def es_gradient(params: dict, ids: Array, targets: Array, key: Array, sigma: Array, *,
                cfg: Cfg, n_pairs: int, chunk: int, dtype=BF16) -> tuple[dict, Array]:
    """The antithetic rank-1 ES estimate of grad(mean token loss) at `params`.

    Returns (g-hat, mean fitness magnitude). `chunk` pairs are evaluated together,
    each as two forwards of the whole batch; `n_pairs` must be a multiple of it.
    """
    assert n_pairs % chunk == 0
    rot = rope_tables(cfg)
    pb = cast(params, dtype)

    def member(fac, s):
        return forward_loss(perturbed(pb, fac, s), cfg, ids, targets, rot).mean()

    def body(i, carry):
        acc, mag = carry
        fac = sample_factors(jax.random.fold_in(key, i), params, chunk)
        lp = jax.vmap(lambda f: member(f, sigma))(fac)
        lm = jax.vmap(lambda f: member(f, -sigma))(fac)
        fit = (lp - lm).astype(F32)
        acc = jax.tree_util.tree_map(lambda a, f: a + _contribution(fit, f), acc, fac,
                                     is_leaf=lambda x: isinstance(x, tuple) or hasattr(x, "shape"))
        return acc, mag + jnp.abs(fit).sum()

    zero = jax.tree_util.tree_map(jnp.zeros_like, params)
    acc, mag = lax.fori_loop(0, n_pairs // chunk, body, (zero, jnp.float32(0)))
    # sum_i f_i (a_i b_i^T) / (P sigma): the sigma^2 in the denominator absorbs eps's own sigma.
    grads = jax.tree_util.tree_map(lambda a: a / (n_pairs * sigma), acc)
    return grads, mag / n_pairs


def centered_ranks(x: Array) -> Array:
    """Ranks of x scaled to [-0.5, 0.5], the usual evolution-strategies fitness transform."""
    r = jnp.argsort(jnp.argsort(x)).astype(F32)
    return r / (x.shape[0] - 1) - 0.5


@partial(jax.jit, static_argnames=("cfg", "n_pairs", "chunk", "dtype"))
def es_gradient_ranked(params: dict, ids: Array, targets: Array, key: Array, sigma: Array, *,
                       cfg: Cfg, n_pairs: int, chunk: int, dtype=BF16) -> tuple[dict, Array]:
    """`es_gradient` with rank-shaped fitness, in the OpenAI-ES form: the 2P losses are replaced by
    their centred ranks and a pair's weight is rank(L+) - rank(L-). Ranks need the whole population,
    so every pair's factors are drawn up front and kept (tens of MB at 256 pairs), the fitness of
    all pairs is computed, and only then are the weighted factors summed.
    """
    assert n_pairs % chunk == 0
    rot = rope_tables(cfg)
    pb = cast(params, dtype)
    fac = sample_factors(key, params, n_pairs)

    def member(f, s):
        return forward_loss(perturbed(pb, f, s), cfg, ids, targets, rot).mean()

    def one_chunk(f):
        return jax.vmap(lambda g: member(g, sigma))(f), jax.vmap(lambda g: member(g, -sigma))(f)

    chunked = jax.tree_util.tree_map(lambda x: x.reshape((n_pairs // chunk, chunk) + x.shape[1:]), fac)
    lp, lm = jax.lax.map(one_chunk, chunked)
    lp, lm = lp.reshape(-1).astype(F32), lm.reshape(-1).astype(F32)
    u = centered_ranks(jnp.concatenate([lp, lm]))
    w = u[:n_pairs] - u[n_pairs:]
    grads = jax.tree_util.tree_map(lambda f: _contribution(w, f) / (n_pairs * sigma), fac,
                                   is_leaf=lambda x: isinstance(x, tuple) or hasattr(x, "shape"))
    return grads, jnp.abs(lp - lm).mean()


def make_optimizer(lr: float, b1: float = 0.9, b2: float = 0.999) -> optax.GradientTransformation:
    """Adam, with optax's defaults for everything not named -- the local-learning setting."""
    return optax.adam(lr, b1=b1, b2=b2)
