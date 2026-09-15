"""The two networks under test.

`ResidualMLP` is the architecture from the PC-ALM paper (Seely & Gould 2026):
depth L, width N, no biases, fixed per-layer scale factors, and a skip
connection around every hidden block. Written out, with z_0 = x:

    z_1     = s_1 W_1 x                                    (no skip)
    z_i     = z_{i-1} + s_h W_i phi(z_{i-1})   i = 2..L-1  (skip)
    logits  = s_L W_L phi(z_{L-1})                         (no skip)

    s_1 = 1/sqrt(d_in),  s_h = 1/sqrt(N L),  s_L = 1/N

The scales are the paper's `gamma0 = 1` parameterisation: weights are drawn from
a unit normal and the scale lives outside them, which keeps the same optimiser
step meaningful across depths.

`PlainMLP` is the ordinary thing: dense layers with biases and ReLU, no skips.
Forward-Forward runs on it, and so does the backprop control that isolates how
much of the comparison is architecture rather than learning rule.
"""

import math
from typing import Callable

import jax
import jax.numpy as jnp
from jax import Array

Params = list[Array]


def relu(x: Array) -> Array:
    return jax.nn.relu(x)


# ── Residual MLP (PC-ALM paper architecture) ─────────────────────────────────

def res_scales(width: int, depth: int, input_dim: int) -> list[float]:
    if depth < 2:
        raise ValueError("depth must be at least 2 (one input layer, one output layer)")
    return ([1.0 / math.sqrt(input_dim)]
            + [1.0 / math.sqrt(width * depth)] * (depth - 2)
            + [1.0 / width])


def res_skips(depth: int) -> tuple[bool, ...]:
    return tuple([False] + [True] * (depth - 2) + [False])


def res_init(key: Array, *, depth: int, width: int,
             input_dim: int = 784, output_dim: int = 10) -> Params:
    keys = jax.random.split(key, depth)
    params = []
    for i in range(depth):
        in_dim = input_dim if i == 0 else width
        out_dim = output_dim if i == depth - 1 else width
        params.append(jax.random.normal(keys[i], (out_dim, in_dim), dtype=jnp.float32))
    return params


def res_block(W: Array, scale: float, skip: bool, z_prev: Array,
              phi: Callable[[Array], Array], *, is_first: bool) -> Array:
    """One block's prediction of the next activation, given the previous one."""
    inp = z_prev if is_first else phi(z_prev)
    pred = scale * (inp @ W.T)
    return pred + z_prev if skip else pred


def res_forward(params: Params, scales, skips, x: Array, phi) -> list[Array]:
    acts, z = [], x
    for i, W in enumerate(params):
        z = res_block(W, scales[i], skips[i], z, phi, is_first=(i == 0))
        acts.append(z)
    return acts


def res_logits(params: Params, scales, skips, x: Array, phi) -> Array:
    return res_forward(params, scales, skips, x, phi)[-1]


# ── Plain MLP (Forward-Forward, and its backprop control) ────────────────────

def mlp_init(key: Array, sizes: list[int]) -> Params:
    """`sizes` is [in, h1, ..., out]; returns [W1, b1, W2, b2, ...].

    He-style init: weights scaled by sqrt(2/fan_in), biases zero. This is the
    initialisation Forward-Forward reproductions use, and the only sane default
    for a ReLU stack.
    """
    keys = jax.random.split(key, len(sizes) - 1)
    params = []
    for k, (fan_in, fan_out) in zip(keys, zip(sizes[:-1], sizes[1:])):
        W = jax.random.normal(k, (fan_out, fan_in), dtype=jnp.float32) * math.sqrt(2.0 / fan_in)
        params += [W, jnp.zeros((fan_out,), dtype=jnp.float32)]
    return params


def mlp_forward(params: Params, x: Array, *, final_relu: bool = False) -> Array:
    n_layers = len(params) // 2
    h = x
    for i in range(n_layers):
        h = h @ params[2 * i].T + params[2 * i + 1]
        if final_relu or i < n_layers - 1:
            h = relu(h)
    return h


# ── Bookkeeping ──────────────────────────────────────────────────────────────

def n_params(params: Params) -> int:
    return int(sum(p.size for p in jax.tree_util.tree_leaves(params)))
