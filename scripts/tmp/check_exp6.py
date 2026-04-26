"""Sanity check for experiments6.py constants and shapes."""
import sys
sys.path.insert(0, ".")
from projects.variations.experiments6 import (
    N_CAT, VOCAB_SIZE, E_DIM, COND_DIM, IN_DIM, LATENT_DIM,
    EVAL_COND_CATS, K_NN, init_params, cats_to_cond, n_params,
)
import jax, jax.numpy as jnp
import numpy as np
from shared_lib.random_utils import infinite_safe_keys_from_key

print(f"N_CAT={N_CAT} VOCAB_SIZE={VOCAB_SIZE} E_DIM={E_DIM}")
print(f"COND_DIM={COND_DIM} IN_DIM={IN_DIM}")
print(f"Total slots={VOCAB_SIZE**N_CAT}")
print(f"Embedding params={N_CAT*VOCAB_SIZE*E_DIM}")
print(f"EVAL_COND_CATS shape={EVAL_COND_CATS.shape}")

key_gen = infinite_safe_keys_from_key(jax.random.PRNGKey(42))
params = init_params(key_gen)
print(f"recon_emb shape={params['recon_emb'].shape}")
print(f"cat_emb shape={params['cat_emb'].shape}")
print(f"n_params={n_params(params):,}")

B = 4
recon_cond = jnp.broadcast_to(params["recon_emb"][None], (B, COND_DIM))
print(f"recon_cond broadcast shape={recon_cond.shape}")

cats = jnp.array(np.random.randint(0, VOCAB_SIZE, (10, N_CAT)), dtype=jnp.int32)
cond = cats_to_cond(params, cats)
print(f"cats_to_cond output shape={cond.shape}")

print("All checks passed!")
