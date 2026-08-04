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
