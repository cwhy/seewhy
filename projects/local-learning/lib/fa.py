"""Feedback alignment: backprop with the backward weights replaced by noise.

Backpropagation sends error back down through the transpose of the same weight
matrix the forward pass used, W^T. That is the *weight transport problem*: a
synapse would have to know the strength of a different synapse, in another
neuron, pointing the other way, and nothing in cortex is known to arrange that.

Feedback alignment @lillicrap2016 asks what happens if the backward pass uses a
*fixed random matrix* B instead, of the same shape as W and never trained. The
surprising answer is that it still learns: the forward weights drift into
partial alignment with B — the network learns to make its own gradient useful
rather than being handed a useful one — and the update ends up within 90 degrees
of the true gradient, which is all descent needs.

Implementation: one `custom_vjp` per matmul. The forward value is unchanged, the
gradient with respect to the weights is the same outer product backprop uses,
and only the gradient flowing *down* to the previous layer is replaced:

    forward   y = x W^T
    backward  dW = g^T x        (unchanged)
              dx = g B          (backprop would use g W)

Everything else — the architecture, the loss, the optimiser — is untouched, so
the comparison against backprop isolates the backward weights and nothing else.
"""

import jax
import jax.numpy as jnp
from jax import Array

from lib.model import Params


@jax.custom_vjp
def fa_matmul(x: Array, W: Array, B: Array) -> Array:
    """x @ W.T on the way up; B, not W, carries the error on the way down."""
    return x @ W.T


def _fa_fwd(x, W, B):
    return x @ W.T, (x, B)


def _fa_bwd(res, g):
    x, B = res
    dx = g @ B                      # the feedback path: fixed, random, untrained
    dW = jnp.einsum("...o,...i->oi", g, x)
    return dx, dW, None             # B is a constant, so no gradient for it


fa_matmul.defvjp(_fa_fwd, _fa_bwd)


def init_feedback(key: Array, params: Params) -> Params:
    """One fixed random feedback matrix per weight matrix, matched in shape *and scale*.

    The scale matters more than it looks. Feedback alignment's claim is about
    direction — a random B still points the update within 90 degrees of the
    gradient — but the magnitude of B sets how large the error arriving at the
    layer below is. Drawing every B from a unit normal while the forward weights
    are He-initialised (standard deviation sqrt(2/fan_in), about 0.05 for a
    784-input layer) makes the feedback twenty times too strong, which is an
    enormous per-layer learning rate rather than a statement about feedback. So
    each B is scaled to the standard deviation of the W it stands in for.
    """
    keys = jax.random.split(key, len(params))
    return [jax.random.normal(k, p.shape, dtype=p.dtype) * jnp.std(p) if p.ndim == 2
            else jnp.zeros_like(p) for k, p in zip(keys, params)]


def alignment(params: Params, feedback: Params) -> Array:
    """Mean cosine between each B and the W it stands in for.

    This is the quantity the method is named after. It starts near zero — B is
    random — and rises if the forward weights are learning to accommodate the
    feedback they are given.
    """
    cosines = []
    for W, B in zip(params, feedback):
        if W.ndim != 2:
            continue
        w, b = W.reshape(-1), B.reshape(-1)
        cosines.append(jnp.sum(w * b) / jnp.maximum(
            jnp.linalg.norm(w) * jnp.linalg.norm(b), 1e-30))
    return jnp.mean(jnp.stack(cosines))


def residual_mm(feedback: Params):
    """Matmul override for the residual MLP: layer i uses feedback[i]."""
    return lambda i, inp, W: fa_matmul(inp, W, feedback[i])


def plain_mm(feedback: Params):
    """Matmul override for the plain MLP, whose params interleave W and b."""
    return lambda i, inp, W: fa_matmul(inp, W, feedback[2 * i])
