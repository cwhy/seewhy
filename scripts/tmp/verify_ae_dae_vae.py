"""Quick sanity check: encode shape + no predictor param leakage for AE/DAE/VAE."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import importlib.util
import jax
import jax.numpy as jnp
from shared_lib.random_utils import infinite_safe_keys_from_key

SSL = Path(__file__).parent.parent.parent / "projects" / "ssl"

checks = [
    ("exp_ae_mlp",          "ae"),
    ("exp_ae_gram_dual",    "ae"),
    ("exp_dae_vit_patch",   "dae"),
    ("exp_dae_gram_glu",    "dae"),
    ("exp_vae_mlp",         "vae"),
    ("exp_vae_gram_dual",   "vae"),
    ("exp_vae_vit_patch",   "vae"),
]

BAD_KEYS = ["W_pred1", "action_emb", "W_film_s", "W_film_t"]

for name, kind in checks:
    spec = importlib.util.spec_from_file_location("_m", SSL / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    kg = infinite_safe_keys_from_key(jax.random.PRNGKey(0))
    p = m.init_params(kg)
    x = jnp.ones((4, 784)) * 128.0
    e = m.encode(p, x)
    n_p = sum(v.size for v in jax.tree_util.tree_leaves(p))
    for bad in BAD_KEYS:
        assert bad not in p, f"{name} has unexpected param: {bad}"
    if kind == "vae":
        assert "W_vae" in p, f"{name} missing W_vae"
    print(f"OK  {name:<30}  encode={e.shape}  params={n_p:,}")

print("\nAll checks passed.")
