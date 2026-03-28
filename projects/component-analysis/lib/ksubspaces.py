"""
K-Subspaces clustering (pure numpy, no I/O).

Alternates between:
  Fit:    PCA on each cluster → one subspace per cluster
  Assign: each point → cluster with lowest reconstruction error

Distance from x to subspace c:
    d(x, c) = ||x - mean_c||² - ||V_c (x - mean_c)||²
            = residual variance not captured by the subspace

The assign step is fully vectorised: one big matmul X @ V_stack^T
then per-cluster mean corrections.
"""

import numpy as np
from typing import NamedTuple


def _randomized_pca(Xc: np.ndarray, n_comp: int, rng, n_power: int = 2) -> np.ndarray:
    """
    Top n_comp right singular vectors of Xc via randomized range finder.

    Avoids np.linalg.svd (slow in this environment) by using eigh on the
    tiny (r×r) inner-product matrix B B^T instead:
      B = Q^T Xc  (r, D)
      M = B B^T   (r, r)  ← small
      eigh(M)     → left singular vectors U
      Vt = U^T B  → right singular vectors (unnormalised)

    Returns (n_comp, D), rows are orthonormal.
    """
    N, D = Xc.shape
    r = n_comp + min(10, n_comp)                            # slight oversampling

    Omega = rng.standard_normal((D, r)).astype(Xc.dtype)
    Y = Xc @ Omega                                          # (N, r)

    for _ in range(n_power):
        Y = Xc @ (Xc.T @ Y)                                # power iteration

    Q, _ = np.linalg.qr(Y)                                 # (N, r) — QR is fast
    B = Q.T @ Xc                                            # (r, D)

    # eigh on (r, r) instead of SVD on (r, D) — avoids slow LAPACK dgesvd
    M = B @ B.T                                             # (r, r)
    _, U = np.linalg.eigh(M)                               # (r, r)
    U_top = U[:, -n_comp:][:, ::-1]                        # (r, n_comp) descending

    Vt = U_top.T @ B                                        # (n_comp, D)
    norms = np.linalg.norm(Vt, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return Vt / norms                                       # (n_comp, D)


class KSubspacesResult(NamedTuple):
    components: np.ndarray   # (K, n_comp, D)  one PCA basis per cluster
    means: np.ndarray        # (K, D)           per-cluster means
    assignments: np.ndarray  # (N,)             cluster index for each training point
    errors: np.ndarray       # (N,)             reconstruction error per training point
    total_error: float
    n_iter: int


def _assign(X: np.ndarray, components: np.ndarray, means: np.ndarray) -> np.ndarray:
    """
    Vectorised assignment.  Returns (N,) cluster indices.

    All K distance computations are fused into two matmuls:
        scores_all = X @ V_stack^T         (N, K*n_comp)
        X_means    = X @ means^T           (N, K)
    """
    N, D   = X.shape
    K, n_comp, _ = components.shape

    x_sq     = (X ** 2).sum(axis=1)                     # (N,)
    means_sq = (means ** 2).sum(axis=1)                  # (K,)
    X_means  = X @ means.T                               # (N, K)

    # Stack all component matrices: (K*n_comp, D)
    V_stack     = components.reshape(K * n_comp, D)
    scores_all  = (X @ V_stack.T).reshape(N, K, n_comp)  # (N, K, n_comp)

    # Projection of each cluster mean onto its own subspace: (K, n_comp)
    mean_projs  = np.einsum("kd,knd->kn", means, components)

    # V_c (x - mean_c) = scores_all[:,c,:] - mean_projs[c]
    proj_sq = ((scores_all - mean_projs[np.newaxis]) ** 2).sum(axis=2)  # (N, K)

    # ||x - mean_c||²
    x_mean_sq = x_sq[:, np.newaxis] - 2 * X_means + means_sq[np.newaxis]  # (N, K)

    dist = x_mean_sq - proj_sq   # reconstruction error per (point, cluster)
    dist = np.maximum(dist, 0.0) # numerical guard
    return dist.argmin(axis=1), dist


def _fit_once(
    X: np.ndarray,
    K: int,
    n_components: int,
    max_iter: int,
    seed: int,
) -> KSubspacesResult:
    rng = np.random.default_rng(seed)
    N, D = X.shape

    # ── Init: vectorised K-means++ ────────────────────────────────────────────
    # d² to nearest chosen centre, updated incrementally using:
    #   ||x - c||² = ||x||² - 2 x·c + ||c||²
    x_sq = (X ** 2).sum(axis=1)                          # (N,)
    first = int(rng.integers(N))
    chosen = [first]
    min_d2 = x_sq - 2 * (X @ X[first]) + float((X[first] ** 2).sum())
    min_d2 = np.maximum(min_d2, 0.0)

    for _ in range(K - 1):
        probs = min_d2 / min_d2.sum()
        c_idx = int(rng.choice(N, p=probs))
        chosen.append(c_idx)
        new_d2 = x_sq - 2 * (X @ X[c_idx]) + float((X[c_idx] ** 2).sum())
        np.minimum(min_d2, np.maximum(new_d2, 0.0), out=min_d2)

    centres_arr = X[np.array(chosen)]                    # (K, D)
    # Initial assignments by nearest centre (vectorised)
    x_sq_col   = x_sq[:, np.newaxis]                     # (N, 1)
    c_sq_row   = (centres_arr ** 2).sum(axis=1)          # (K,)
    dists      = x_sq_col + c_sq_row - 2 * (X @ centres_arr.T)
    assignments = dists.argmin(axis=1)

    components = np.zeros((K, n_components, D))
    means      = np.zeros((K, D))

    for it in range(max_iter):
        # ── Fit: truncated PCA per cluster ────────────────────────────────────
        counts = np.bincount(assignments, minlength=K)
        for c in range(K):
            mask = assignments == c
            if counts[c] < n_components + 1:
                # Reinitialise from the largest cluster
                biggest = counts.argmax()
                idx = rng.choice(np.where(assignments == biggest)[0])
                means[c] = X[idx] + rng.standard_normal(D) * 1e-3
                rand, _ = np.linalg.qr(rng.standard_normal((D, n_components)))
                components[c] = rand[:, :n_components].T
                continue
            Xc = X[mask]
            means[c] = Xc.mean(axis=0)
            Xc = Xc - means[c]
            components[c] = _randomized_pca(Xc, n_components, rng)

        # ── Assign ─────────────────────────────────────────────────────────────
        new_asgn, dist_matrix = _assign(X, components, means)
        if np.all(new_asgn == assignments):
            assignments = new_asgn
            break
        assignments = new_asgn

    pt_errors = dist_matrix[np.arange(N), assignments]
    return KSubspacesResult(
        components=components,
        means=means,
        assignments=assignments,
        errors=pt_errors,
        total_error=float(pt_errors.sum()),
        n_iter=it + 1,
    )


def fit_ksubspaces(
    X: np.ndarray,
    K: int,
    n_components: int,
    max_iter: int = 40,
    n_restarts: int = 1,
    seed: int = 0,
) -> KSubspacesResult:
    """Fit K-subspaces; return the restart with lowest total reconstruction error."""
    best = None
    for r in range(n_restarts):
        result = _fit_once(X, K, n_components, max_iter, seed=seed + r * 9973)
        if best is None or result.total_error < best.total_error:
            best = result
    return best


def predict(X: np.ndarray, result: KSubspacesResult) -> tuple[np.ndarray, np.ndarray]:
    """
    Assign X to nearest subspace.
    Returns (cluster_indices (N,), per_point_errors (N,)).
    """
    asgn, dist_matrix = _assign(X, result.components, result.means)
    return asgn, dist_matrix[np.arange(len(X)), asgn]
