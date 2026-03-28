"""
Smoke test: verify lib/pca.py and lib/cca.py imports and basic correctness.

Usage:
    uv run python projects/component-analysis/scripts/tmp/smoke_test.py
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from lib.pca import fit_pca, project, reconstruct, reconstruction_error
from lib.cca import fit_mcca, project_class, project_all, canonical_correlations

print("lib imports OK")

# ── PCA smoke test ────────────────────────────────────────────────────────────
X = np.random.default_rng(0).standard_normal((100, 20))
r = fit_pca(X, 5)
assert r.components.shape == (5, 20), r.components.shape
assert r.explained_variance_ratio.sum() <= 1.0 + 1e-9
s = project(X, r)
assert s.shape == (100, 5)
mse = reconstruction_error(X, r)
print(f"PCA OK  EVR.sum={r.explained_variance_ratio.sum():.4f}  recon_mse={mse:.4f}")

# ── MCCA smoke test ───────────────────────────────────────────────────────────
rng = np.random.default_rng(1)
Xs = [rng.standard_normal((80, 20)) for _ in range(4)]
mc = fit_mcca(Xs, n_components=3)
assert mc.components.shape == (3, 20)
assert mc.components_pixel.shape == (3, 20)
sc = project_class(Xs[0], 0, mc)
assert sc.shape == (80, 3)
all_sc = project_all(Xs, mc)
corrs = canonical_correlations(all_sc)
assert corrs.shape == (3,)
print(f"MCCA OK  eigenvalues={mc.eigenvalues.round(2).tolist()}  mean_corrs={corrs.round(3).tolist()}")

print("All OK")
