"""
Pure-function PCA via economy SVD.

All functions are side-effect free: they take numpy arrays and return numpy
arrays or NamedTuples. No JAX, no I/O.

Typical usage:
    result = fit_pca(X, n_components=50)
    scores  = project(X, result)          # (N, n_components)
    X_recon = reconstruct(scores, result)  # (N, D)
"""

import numpy as np
from typing import NamedTuple


class PCAResult(NamedTuple):
    components: np.ndarray              # (n_components, D)  row = eigenvector
    explained_variance: np.ndarray      # (n_components,)    variance along each PC
    explained_variance_ratio: np.ndarray  # (n_components,)  fraction of total
    singular_values: np.ndarray         # (n_components,)
    mean: np.ndarray                    # (D,)


def fit_pca(X: np.ndarray, n_components: int | None = None) -> PCAResult:
    """
    Fit PCA on (N, D) array X.

    Uses economy SVD of the centered data matrix, so it's efficient even when
    D > N (e.g. per-digit subsets with N≈6 000, D=784).

    n_components: number of top components to keep (None = keep all min(N,D)).
    """
    X = np.asarray(X, dtype=np.float64)
    mean = X.mean(axis=0)
    Xc = X - mean

    # Economy SVD: Xc = U S Vt, shapes (N,k) (k,) (k,D) where k=min(N,D)
    _, s, Vt = np.linalg.svd(Xc, full_matrices=False)

    # Eigenvalues of covariance matrix = s² / (N-1)
    N = X.shape[0]
    variance = s ** 2 / (N - 1)
    total_var = variance.sum()

    if n_components is not None:
        Vt = Vt[:n_components]
        s = s[:n_components]
        variance = variance[:n_components]

    return PCAResult(
        components=Vt,
        explained_variance=variance,
        explained_variance_ratio=variance / total_var,
        singular_values=s,
        mean=mean,
    )


def project(X: np.ndarray, result: PCAResult) -> np.ndarray:
    """
    Project (N, D) array X onto the PCA subspace.
    Returns (N, n_components) scores.
    """
    return (np.asarray(X, dtype=np.float64) - result.mean) @ result.components.T


def reconstruct(scores: np.ndarray, result: PCAResult) -> np.ndarray:
    """
    Reconstruct (N, D) array from (N, n_components) scores.
    Inverse of project (up to truncation).
    """
    return scores @ result.components + result.mean


def reconstruction_error(X: np.ndarray, result: PCAResult) -> float:
    """
    Mean squared reconstruction error per dimension when using result.components.
    """
    X = np.asarray(X, dtype=np.float64)
    scores = project(X, result)
    Xhat = reconstruct(scores, result)
    return float(((X - Xhat) ** 2).mean())
