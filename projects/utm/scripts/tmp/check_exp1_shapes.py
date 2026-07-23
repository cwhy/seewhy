"""Sanity check: init params, verify shapes, single forward pass."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[4]))
sys.path.insert(0, str(Path(__file__).parents[2]))

import jax.numpy as jnp
from jax import random

from experiments1 import init_params, n_params, forward, N_CELLS

params = init_params(random.PRNGKey(0))
print(f"n_params: {n_params(params):,}")

dummy = jnp.zeros(N_CELLS, dtype=jnp.int32)
logits, ponder_cost = forward(params, dummy)
print(f"logits shape:  {logits.shape}")
print(f"ponder_cost:   {ponder_cost:.3f}")
print("OK")
