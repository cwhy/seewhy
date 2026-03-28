import numpy as np, time, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from lib.ksubspaces import _randomized_pca, _fit_once

rng = np.random.default_rng(42)

# Time randomized PCA (small cluster)
Xc = rng.standard_normal((300, 784)).astype(np.float64)
t = time.perf_counter()
for _ in range(100):
    _randomized_pca(Xc, 20, rng)
print(f"randomized_pca N=300  x100: {time.perf_counter()-t:.3f}s  ({(time.perf_counter()-t)/100*1000:.1f}ms each)")

# Time randomized PCA (large cluster)
Xc2 = rng.standard_normal((6000, 784)).astype(np.float64)
t = time.perf_counter()
for _ in range(10):
    _randomized_pca(Xc2, 20, rng)
print(f"randomized_pca N=6000 x10:  {time.perf_counter()-t:.3f}s  ({(time.perf_counter()-t)/10*1000:.1f}ms each)")

# Full fit_once timing
X = rng.standard_normal((5000, 784)).astype(np.float64)
t = time.perf_counter()
r = _fit_once(X, 20, 20, 5, 42)
print(f"_fit_once N=5000 K=20 5iters: {time.perf_counter()-t:.3f}s  n_iter={r.n_iter}")
