"""
Synthetic task generators.

Linear map (paper §3.1): A in {0,1}^{SxS} with exactly s ones per row, transition
f(x) = Ax mod 2. A sequence is concat(x0, x1) of S*T tokens with T=2, vocab C=2.

The point of the construction: predicting token S+i requires attending to exactly
the s positions where row i of A is 1. The ground-truth attention support is known
by construction, so "did the model find the pattern" is directly measurable
(see support_iou in lib/models.py callers).

The first half of every sequence is i.i.d. uniform, so its CE is exactly ln 2 and
carries no signal — all metrics use the second half only.
"""

import jax
import jax.numpy as jnp


def linear_map_matrix(key, S: int, s: int) -> jnp.ndarray:
    """(S, S) int32 matrix with exactly s ones per row, columns chosen uniformly."""
    # argsort of uniform noise gives an independent random permutation per row;
    # taking the first s columns picks s distinct positions without replacement.
    idx = jnp.argsort(jax.random.uniform(key, (S, S)), axis=1)[:, :s]
    return jnp.zeros((S, S), jnp.int32).at[jnp.arange(S)[:, None], idx].set(1)


def linear_map_batch(key, A: jnp.ndarray, batch: int) -> jnp.ndarray:
    """(batch, 2S) int32 sequences concat(x0, A x0 mod 2), x0 ~ U{0,1}^S."""
    S = A.shape[0]
    x0 = jax.random.bernoulli(key, 0.5, (batch, S)).astype(jnp.int32)
    x1 = (x0 @ A.T) % 2
    return jnp.concatenate([x0, x1], axis=1)


def linear_map_traj_batch(key, A: jnp.ndarray, batch: int, T: int) -> jnp.ndarray:
    """(batch, S*T) — T states of a trajectory x_{t+1} = A x_t mod 2, flattened.

    T=2 reduces to linear_map_batch. Larger T puts several applications of the SAME A in
    one sequence, i.e. more worked examples of the map per sequence. The paper fixes T=2
    for the linear map ("We always use C=2 and T=2") and only varies trajectory length on
    the cellular automata task, so this axis is untested for the linear map.
    """
    S = A.shape[0]
    x0 = jax.random.bernoulli(key, 0.5, (batch, S)).astype(jnp.int32)

    def step(x, _):
        nx = (x @ A.T) % 2
        return nx, nx

    _, rest = jax.lax.scan(step, x0, None, length=T - 1)          # (T-1, batch, S)
    return jnp.concatenate([x0[None], rest], 0).transpose(1, 0, 2).reshape(batch, S * T)


def ca_rule_pool(key, n_rules: int, C: int = 4, W: int = 3) -> jnp.ndarray:
    """(n_rules, C**W) int32 lookup tables. Sampled once per run; one rule per example.

    Per the paper's appendix: "N: Number of rules; one rule is sampled per training
    example". So unlike the linear map's single fixed A, the active rule changes every
    sequence — the model has to infer it IN CONTEXT before it can predict.
    """
    return jax.random.randint(key, (n_rules, C**W), 0, C)


def ca_batch(key, rules: jnp.ndarray, batch: int, S: int, T: int, k: int,
             C: int = 4) -> jnp.ndarray:
    """(batch, S*T) int32. Each row: a random rule from the pool applied to a random
    initial state, T states flattened. Boundaries wrap (the paper does not specify).

    k is the composition depth — the rule is applied k times per state transition, so the
    span of x_{t+1}[i] over x_t is 2k+1 wide. k is a Python int (static).
    """
    k_rule, k_state = jax.random.split(key)
    R = rules[jax.random.randint(k_rule, (batch,), 0, rules.shape[0])]    # (batch, C**W)
    x0 = jax.random.randint(k_state, (batch, S), 0, C)

    def apply_once(x):
        idx = jnp.roll(x, 1, axis=1) * C * C + x * C + jnp.roll(x, -1, axis=1)
        return jnp.take_along_axis(R, idx, axis=1)

    def step(x, _):
        for _ in range(k):
            x = apply_once(x)
        return x, x

    _, rest = jax.lax.scan(step, x0, None, length=T - 1)                  # (T-1, batch, S)
    return jnp.concatenate([x0[None], rest], 0).transpose(1, 0, 2).reshape(batch, S * T)
