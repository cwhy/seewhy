"""
Shared K-means utilities for kmeansT.

  fit(X, K, n_iter, seed)         -> (assignments, centroids)
  extract_features(X, assignments) -> feature matrix (N, K)
"""

import numpy as np


# ---------------------------------------------------------------------------
# Core K-means
# ---------------------------------------------------------------------------

def _sq_dists(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Squared Euclidean distances. X: (n,d), C: (k,d) -> (n,k)."""
    X_sq = np.einsum('nd,nd->n', X, X)
    C_sq = np.einsum('kd,kd->k', C, C)
    cross = X @ C.T
    return np.maximum(X_sq[:, None] - 2.0 * cross + C_sq[None, :], 0.0)


def _init_plusplus(X: np.ndarray, K: int, rng: np.random.Generator) -> np.ndarray:
    n = X.shape[0]
    centroids = [X[rng.integers(n)].copy()]
    for _ in range(K - 1):
        C = np.stack(centroids)
        dists = _sq_dists(X, C).min(axis=1)
        total = dists.sum()
        probs = dists / total if total > 0 else np.ones(n) / n
        centroids.append(X[rng.choice(n, p=probs)].copy())
    return np.stack(centroids)


def fit(
    X: np.ndarray,
    K: int,
    n_iter: int = 50,
    seed: int = 42,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    K-means++ on X (n_points, n_dims).
    Returns (assignments (n_points,), centroids (K, n_dims)).
    """
    rng = np.random.default_rng(seed)
    centroids = _init_plusplus(X, K, rng)
    assignments = np.zeros(X.shape[0], dtype=np.int32)

    for it in range(n_iter):
        new_assignments = _sq_dists(X, centroids).argmin(axis=1).astype(np.int32)
        changed = (new_assignments != assignments).sum()
        assignments = new_assignments
        for k in range(K):
            mask = assignments == k
            if mask.sum() > 0:
                centroids[k] = X[mask].mean(axis=0)
        if verbose:
            print(f"  iter {it+1:3d}  changed={changed}")
        if changed == 0:
            break

    return assignments, centroids


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(X: np.ndarray, pixel_assignments: np.ndarray, K: int) -> np.ndarray:
    """
    Pool pixel values by cluster assignment.

    X:                 (N, 784)  — flattened images, float32 in [0,1]
    pixel_assignments: (784,)    — cluster id for each pixel (int, 0..K-1)
    K:                 number of clusters

    Returns (N, K) — mean pixel value per cluster per image.
    """
    N = X.shape[0]
    out = np.empty((N, K), dtype=np.float32)
    for k in range(K):
        mask = pixel_assignments == k
        if mask.sum() == 0:
            out[:, k] = 0.0
        else:
            out[:, k] = X[:, mask].mean(axis=1)
    return out
