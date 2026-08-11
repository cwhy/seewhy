"""
Function-class tasks: the "what" axis of the task × architecture grid.

A task is a *prior over functions*, not a dataset. One episode is

    theta ~ prior(task)       xs ~ P(x)       ys = f_theta(xs) + noise

and the model sees only `(xs, ys)` interleaved (see `encoding.py`). Because
theta is redrawn every episode and never shown, anything the model gets right
beyond the prior mean it got from the context.

Two knobs turn the same task into different *regimes*, and they are the reason
this module separates `TaskSpec` (the function class) from `episode_sampler`
(how episodes are drawn from it):

  n_tasks   None draws a fresh theta per episode — an effectively infinite task
            distribution, where reading the context is the only strategy. An
            integer draws from a fixed pool of that many functions, where
            memorising the pool competes with reading the context. Sweeping this
            is how the project puts a number on "when does a model stop learning
            in context and start looking things up".

  pool_seed Which pool. Train on pool 0, evaluate on pool 1, and a model that
            memorised has nowhere to hide.

Adding a task means adding a `TaskSpec` factory and one line in `SPECS`. Nothing
else in the project needs to know it exists.
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp

from . import references

# Fixed root for finite task pools. Constant across processes and runs so that
# "pool 0" means the same 64 functions on the GPU box today and on a rerun in a
# month — otherwise the memorisation sweep is not reproducible.
POOL_ROOT = 20260811


class Episode(NamedTuple):
    xs: jnp.ndarray       # (B, K, d_x)
    ys: jnp.ndarray       # (B, K)
    theta: Any            # pytree, (B, …) — for diagnostics only, never shown to the model


class TaskSpec(NamedTuple):
    """A function class, split into the pieces the samplers need."""
    name: str
    d_x: int
    sample_theta: Callable[[jnp.ndarray], Any]                 # key -> theta (no batch axis)
    predict: Callable[[Any, jnp.ndarray], jnp.ndarray]         # theta, (K, d) -> (K,)
    sample_x: Callable[[jnp.ndarray, tuple], jnp.ndarray]      # key, shape -> xs
    reference: Callable[..., jnp.ndarray]
    reference_note: str
    noise_std: float = 0.0


# ── Samplers ──────────────────────────────────────────────────────────────────

def episode_sampler(
    spec: TaskSpec,
    *,
    n_tasks: int | None = None,
    pool_seed: int = 0,
    x_scale: float = 1.0,
    noise_std: float | None = None,
) -> Callable[[jnp.ndarray, int, int], Episode]:
    """Build `sample(key, batch, n_points) -> Episode` for one regime of a task.

    `x_scale` multiplies the inputs after sampling. At 1.0 it is the training
    prior; anything else is a covariate shift the model was never trained on,
    which is the cleanest test of whether it runs an estimator or a lookup.
    """
    sigma = spec.noise_std if noise_std is None else noise_std
    pool_key = jax.random.PRNGKey(POOL_ROOT + pool_seed)

    def draw_thetas(key, batch):
        if n_tasks is None:
            return jax.vmap(spec.sample_theta)(jax.random.split(key, batch))
        idx = jax.random.randint(key, (batch,), 0, n_tasks)
        return jax.vmap(lambda i: spec.sample_theta(jax.random.fold_in(pool_key, i)))(idx)

    def sample(key, batch: int, n_points: int) -> Episode:
        k_x, k_theta, k_noise = jax.random.split(key, 3)
        xs = spec.sample_x(k_x, (batch, n_points, spec.d_x)) * x_scale
        theta = draw_thetas(k_theta, batch)
        ys = jax.vmap(spec.predict)(theta, xs)
        if sigma > 0:
            ys = ys + sigma * jax.random.normal(k_noise, ys.shape)
        return Episode(xs, ys, theta)

    return sample


def _gaussian_x(key, shape):
    return jax.random.normal(key, shape)


# ── Function classes ──────────────────────────────────────────────────────────
#
# Each prior is scaled so that E[y^2] ~ 1, which keeps optimisation settings
# transferable across tasks. Cross-task comparisons never rely on that holding
# exactly: every reported loss is normalised by the *measured* loss of the zero
# predictor (see metrics.py).

def linear(d_x: int = 8, noise_std: float = 0.0) -> TaskSpec:
    """y = x·w, w ~ N(0, I/d). The one task with an exactly optimal reference.

    Ridge with lam = sigma^2 / tau^2 = sigma^2 * d IS the Bayes posterior mean
    here, so excess loss over it is a true regret and 0.0 is attainable. Every
    other task in this file is measured against something weaker; this one is
    the calibration anchor for the whole project.
    """
    tau2 = 1.0 / d_x
    lam = max(noise_std ** 2 / tau2, 1e-6)
    return TaskSpec(
        name=f"linear_d{d_x}" + (f"_n{noise_std}" if noise_std else ""),
        d_x=d_x,
        sample_theta=lambda k: {"w": jax.random.normal(k, (d_x,)) * jnp.sqrt(tau2)},
        predict=lambda th, xs: xs @ th["w"],
        sample_x=_gaussian_x,
        reference=references.ridge(lam),
        reference_note=f"ridge(lam={lam:.3g}) — exactly the Bayes posterior mean",
        noise_std=noise_std,
    )


def sparse_linear(d_x: int = 16, sparsity: int = 3, noise_std: float = 0.0) -> TaskSpec:
    """y = x·w with exactly `sparsity` nonzero coordinates.

    Separates two strategies that the dense linear task cannot tell apart:
    inverting the Gram matrix (ridge, needs k > d) versus selecting a support
    (lasso, needs k ~ s log d). A model whose curve tracks ridge is doing
    least squares; one that beats ridge in the k < d regime is doing selection.
    """
    def sample_theta(k):
        k_pos, k_val = jax.random.split(k)
        support = jax.random.permutation(k_pos, d_x)[:sparsity]
        vals = jax.random.normal(k_val, (sparsity,)) / jnp.sqrt(sparsity)
        return {"w": jnp.zeros(d_x).at[support].set(vals)}

    return TaskSpec(
        name=f"sparse_d{d_x}_s{sparsity}",
        d_x=d_x,
        sample_theta=sample_theta,
        predict=lambda th, xs: xs @ th["w"],
        sample_x=_gaussian_x,
        reference=references.lasso(lam=0.05),
        reference_note="ISTA lasso, 200 iters — approximate, not Bayes-optimal",
        noise_std=noise_std,
    )


def relu_nn(d_x: int = 8, hidden: int = 8, noise_std: float = 0.0) -> TaskSpec:
    """A random 2-layer ReLU teacher — the first genuinely nonlinear class here.

    No tractable posterior, so the reference is 3-NN: a floor, not an optimum.
    Beating it means the model exploits structure a local average cannot.
    """
    scale = jnp.sqrt(2.0 / hidden)

    def sample_theta(k):
        k_w, k_a = jax.random.split(k)
        return {
            "W": jax.random.normal(k_w, (d_x, hidden)) / jnp.sqrt(d_x),
            "a": jax.random.normal(k_a, (hidden,)),
        }

    return TaskSpec(
        name=f"relu_d{d_x}_h{hidden}",
        d_x=d_x,
        sample_theta=sample_theta,
        predict=lambda th, xs: scale * (jax.nn.relu(xs @ th["W"]) @ th["a"]),
        sample_x=_gaussian_x,
        reference=references.knn(3),
        reference_note="3-NN over the prefix — a competence floor, not an optimum",
        noise_std=noise_std,
    )


def sinusoid(noise_std: float = 0.0) -> TaskSpec:
    """y = A sin(f x + phi) in one dimension — the plottable task.

    Kept because every curve for it can be drawn on paper next to the model's
    prediction, which catches failure modes that a scalar MSE hides (predicting
    the prior mean everywhere, latching onto the last y, phase-flipping).
    """
    def sample_theta(k):
        k_a, k_f, k_p = jax.random.split(k, 3)
        return {
            "A": jax.random.uniform(k_a, (), minval=0.7, maxval=1.8),
            "f": jax.random.uniform(k_f, (), minval=0.5, maxval=2.0),
            "phi": jax.random.uniform(k_p, (), minval=0.0, maxval=2 * jnp.pi),
        }

    return TaskSpec(
        name="sinusoid",
        d_x=1,
        sample_theta=sample_theta,
        predict=lambda th, xs: th["A"] * jnp.sin(th["f"] * xs[:, 0] + th["phi"]),
        sample_x=lambda k, shape: jax.random.uniform(k, shape, minval=-3.0, maxval=3.0),
        reference=references.knn(3),
        reference_note="3-NN over the prefix — a competence floor, not an optimum",
        noise_std=noise_std,
    )


SPECS: dict[str, Callable[..., TaskSpec]] = {
    "linear": linear,
    "sparse_linear": sparse_linear,
    "relu_nn": relu_nn,
    "sinusoid": sinusoid,
}
