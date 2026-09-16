"""Backprop, predictive coding, and PC-ALM as three ways to get one gradient.

All three produce a weight update for the same residual MLP and the same loss.
They differ only in how the credit for that loss reaches the weights:

* **BP** — one global backward pass; `jax.grad` of the loss.
* **PC** — the network is treated as a constrained optimisation problem,

      minimise  1/2 ||y - s_L W_L phi(z_{L-1})||^2
      s.t.      z_i = block_i(z_{i-1})   for every hidden i,

  relaxed into the free energy F = loss + 1/2 sum_i ||r_i||^2, with
  r_i = z_i - block_i(z_{i-1}) the layer's prediction error. Inference runs T
  gradient steps on F with respect to the activations z (each step needs only
  the neighbouring layers), then the weights take one step on the same F with
  the activations frozen.
* **PC-ALM** — same, on the augmented Lagrangian

      L = loss + sum_i lambda_i^T r_i + rho/2 sum_i ||r_i||^2,

  with a dual ascent step lambda_i <- lambda_i + alpha r_i interleaved into
  inference. Each layer accumulates its own prediction error; in a linear
  network the accumulated lambda_i converges to that layer's backprop credit
  signal. alpha = 0 recovers PC exactly.

Two implementation details are not in the paper's equations but are needed to
reproduce its numbers, and both follow the authors' released reference:

1. the energy is a mean over the batch, so an activity step of eta_h on the
   per-sample energy is a step of eta_h * batch_size on the batch mean; without
   that rescale, inference at finite T barely moves;
2. the weight update uses the duals from *before* the final dual ascent step
   (`pre_dual_energy` in the reference).
"""

import jax
import jax.numpy as jnp
from jax import Array

from lib.model import Params, res_block, res_forward


def supervised_loss(params: Params, scales, skips, y: Array, free: list[Array], phi) -> Array:
    y_pred = res_block(params[-1], scales[-1], skips[-1], free[-1], phi, is_first=False)
    return 0.5 * jnp.mean(jnp.sum((y_pred - y) ** 2, axis=-1))


def bp_loss(params: Params, scales, skips, x: Array, y: Array, phi, mm=None) -> Array:
    """`mm` overrides the matmul; feedback alignment uses it to change the
    backward pass without touching the forward value."""
    y_pred = res_forward(params, scales, skips, x, phi, mm)[-1]
    return 0.5 * jnp.mean(jnp.sum((y_pred - y) ** 2, axis=-1))


def free_init(params: Params, scales, skips, x: Array, phi) -> list[Array]:
    """Inference starts from the forward pass, as is standard in the PC literature."""
    return res_forward(params, scales, skips, x, phi)[:-1]


def residuals(params: Params, scales, skips, x: Array, free: list[Array], phi) -> list[Array]:
    """r_i = z_i - block_i(z_{i-1}); the output edge is not a constraint."""
    out = []
    for i, z in enumerate(free):
        z_prev = x if i == 0 else free[i - 1]
        out.append(z - res_block(params[i], scales[i], skips[i], z_prev, phi, is_first=(i == 0)))
    return out


def al_energy(params: Params, scales, skips, x, y, free, duals, rho: float, phi) -> Array:
    """L(h, theta, lambda), written in the completed-square form.

    lambda^T r + rho/2 ||r||^2 = rho/2 ||r + lambda/rho||^2 up to a constant in
    r, so shifting the target by lambda/rho is all the dual does to the energy.
    """
    total = supervised_loss(params, scales, skips, y, free, phi)
    batch = x.shape[0]
    for r, lam in zip(residuals(params, scales, skips, x, free, phi), duals):
        shifted = r + lam / rho
        total = total + 0.5 * rho * jnp.sum(shifted * shifted) / batch
    return total


def _inference_steps(params, scales, skips, x, y, free, duals, state_lr, rho, steps, phi):
    grad_free = jax.grad(lambda f: al_energy(params, scales, skips, x, y, f, duals, rho, phi))
    eff_lr = state_lr * x.shape[0]      # see module docstring, point 1

    def step(f, _):
        return [z - eff_lr * g for z, g in zip(f, grad_free(f))], None

    if steps <= 0:
        return free
    free, _ = jax.lax.scan(step, free, xs=None, length=steps)
    return free


def run_inference(params, scales, skips, x, y, *, alpha: float, rho: float,
                  state_lr: float, budget: int, phi):
    """T rounds of primal descent, each followed by a dual ascent step.

    Returns the settled activations and the duals the weight update should see.
    With alpha = 0 the duals stay zero and this is plain PC inference.
    """
    free = free_init(params, scales, skips, x, phi)
    duals = [jnp.zeros_like(r) for r in residuals(params, scales, skips, x, free, phi)]

    def outer(carry, _):
        f, lam = carry
        f = _inference_steps(params, scales, skips, x, y, f, lam, state_lr, rho, 1, phi)
        r = residuals(params, scales, skips, x, f, phi)
        return (f, [l + alpha * ri for l, ri in zip(lam, r)]), None

    if budget > 1:
        (free, duals), _ = jax.lax.scan(outer, (free, duals), xs=None, length=budget - 1)
    # the last round's primal step happens, but its dual step does not reach the
    # weight update (`pre_dual_energy`)
    free = _inference_steps(params, scales, skips, x, y, free, duals, state_lr, rho, 1, phi)
    return free, duals


def method_grad(params: Params, scales, skips, x, y, *, method: str, alpha: float,
                rho: float, state_lr: float, budget: int, phi):
    """The weight gradient each method hands to the optimiser."""
    if method == "bp":
        return jax.grad(lambda p: bp_loss(p, scales, skips, x, y, phi))(params)
    if method not in ("pc", "pcalm"):
        raise ValueError(f"unknown method: {method}")
    a = 0.0 if method == "pc" else alpha
    free, duals = run_inference(params, scales, skips, x, y,
                                alpha=a, rho=rho, state_lr=state_lr, budget=budget, phi=phi)
    free = jax.tree_util.tree_map(jax.lax.stop_gradient, free)
    duals = jax.tree_util.tree_map(jax.lax.stop_gradient, duals)
    return jax.grad(lambda p: al_energy(p, scales, skips, x, y, free, duals, rho, phi))(params)


def grad_cosine(a, b) -> Array:
    """Cosine between two gradient trees — how aligned a local rule is with BP."""
    leaves_a = jax.tree_util.tree_leaves(a)
    leaves_b = jax.tree_util.tree_leaves(b)
    dot = sum(jnp.sum(u * v) for u, v in zip(leaves_a, leaves_b))
    na = jnp.sqrt(sum(jnp.sum(u * u) for u in leaves_a))
    nb = jnp.sqrt(sum(jnp.sum(v * v) for v in leaves_b))
    return dot / jnp.maximum(na * nb, 1e-30)
