"""Check pixel-space L2 distance distribution between decoded prototypes."""
import pickle, numpy as np, sys
from pathlib import Path
import jax, jax.numpy as jnp

root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root.parent.parent))

exp = sys.argv[1] if len(sys.argv) > 1 else "exp29"

with open(root / f"params_{exp}.pkl", "rb") as f:
    raw = pickle.load(f)

# params must be jnp arrays so vmap indexing works on traced integers
p = jax.tree_util.tree_map(jnp.array, raw)

# Import decode_prototypes from the experiment module
import importlib.util
spec = importlib.util.spec_from_file_location("exp", root / f"experiments{exp[3:]}.py")
mod  = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

protos = mod.decode_prototypes(p)   # (K, 784)
K = protos.shape[0]

diff = protos[np.newaxis] - protos[:, np.newaxis]   # (K, K, 784)
r    = np.linalg.norm(diff, axis=-1)                 # (K, K)
np.fill_diagonal(r, np.inf)
r_off = r[r < np.inf]

print(f"{exp} prototype pixel-space L2 distances (off-diagonal):")
print(f"  min:   {r_off.min():.4f}")
print(f"  p1:    {np.percentile(r_off, 1):.4f}")
print(f"  p5:    {np.percentile(r_off, 5):.4f}")
print(f"  p10:   {np.percentile(r_off, 10):.4f}")
print(f"  mean:  {r_off.mean():.4f}")
print(f"  max:   {r_off.max():.4f}")
print(f"\nSuggested MERGE_DIST_PIX values:")
print(f"  aggressive (p1):  {np.percentile(r_off, 1):.2f}")
print(f"  moderate   (p5):  {np.percentile(r_off, 5):.2f}")
print(f"  conservative(p10):{np.percentile(r_off, 10):.2f}")
