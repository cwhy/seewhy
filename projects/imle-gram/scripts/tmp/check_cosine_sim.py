"""Check codebook cosine similarity distribution for a given exp."""
import pickle, numpy as np, sys
from pathlib import Path

exp = sys.argv[1] if len(sys.argv) > 1 else "exp27"
with open(Path(__file__).parent.parent.parent / f"params_{exp}.pkl", "rb") as f:
    p = pickle.load(f)

Z = np.array(p["codebook"])
Z_n = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)
C = Z_n @ Z_n.T
np.fill_diagonal(C, 0)

print(f"{exp} codebook cosine similarity (off-diagonal):")
print(f"  max:          {C.max():.4f}")
print(f"  mean |sim|:   {np.abs(C).mean():.4f}")
print(f"  p90:          {np.percentile(C, 90):.4f}")
print(f"  p95:          {np.percentile(C, 95):.4f}")
print(f"  p99:          {np.percentile(C, 99):.4f}")
print(f"  pairs > 0.3:  {(C > 0.3).sum()}")
print(f"  pairs > 0.5:  {(C > 0.5).sum()}")
print(f"  pairs > 0.7:  {(C > 0.7).sum()}")
print(f"  pairs > 0.9:  {(C > 0.9).sum()}")
