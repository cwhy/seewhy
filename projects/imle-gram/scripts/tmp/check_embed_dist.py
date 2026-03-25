"""Check L2 distance distribution between codebook entries."""
import pickle, numpy as np, sys
from pathlib import Path

exp = sys.argv[1] if len(sys.argv) > 1 else "exp28"
with open(Path(__file__).parent.parent.parent / f"params_{exp}.pkl", "rb") as f:
    p = pickle.load(f)

Z = np.array(p["codebook"])  # (K, D_EMB)
K = Z.shape[0]

diff = Z[np.newaxis] - Z[:, np.newaxis]          # (K, K, D)
r    = np.linalg.norm(diff, axis=-1)             # (K, K)
np.fill_diagonal(r, np.inf)

print(f"{exp} codebook L2 distances (off-diagonal):")
print(f"  min:   {r.min():.4f}")
print(f"  p1:    {np.percentile(r[r < np.inf], 1):.4f}")
print(f"  p5:    {np.percentile(r[r < np.inf], 5):.4f}")
print(f"  mean:  {r[r < np.inf].mean():.4f}")
print(f"  norms: min={np.linalg.norm(Z, axis=1).min():.4f}  "
      f"mean={np.linalg.norm(Z, axis=1).mean():.4f}  "
      f"max={np.linalg.norm(Z, axis=1).max():.4f}")
print(f"\nFor grav=3e-5 per epoch → 100 epochs total disp ≈ {3e-5*100:.4f}")
print(f"Need G ≈ {r.min() / 2 / (3e-5 / 0.1):.1f} to move clusters to merge in 100 epochs")
