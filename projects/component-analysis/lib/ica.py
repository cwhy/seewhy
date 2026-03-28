"""
Pure-function ICA via sklearn FastICA.

FastICA finds statistically independent components by maximising non-Gaussianity.
Unlike PCA, components are not ordered by variance and not orthogonal.

Two sets of vectors are useful:
  components  (n_components, D) — unmixing filters: project data onto sources
  mixing      (n_components, D) — mixing atoms: what each source looks like in
                                  pixel space (columns of the mixing matrix)

For visualization, show the mixing atoms — they are the pixel-space "images"
of each independent component, analogous to PCA eigenvectors.

Typical usage:
    result = fit_ica(X, n_components=50)
    sources = project(X, result)          # (N, n_components)
    X_hat   = reconstruct(sources, result) # (N, D)
"""

import numpy as np
from typing import NamedTuple
from sklearn.decomposition import FastICA


class ICAResult(NamedTuple):
    components: np.ndarray   # (n_components, D)  unmixing filters (rows)
    mixing: np.ndarray       # (n_components, D)  pixel-space atoms (rows of mixing.T)
    mean: np.ndarray         # (D,)
    n_iter: int              # iterations until convergence


def fit_ica(
    X: np.ndarray,
    n_components: int,
    max_iter: int = 1000,
    tol: float = 1e-4,
    seed: int = 0,
) -> ICAResult:
    """
    Fit FastICA on (N, D) array X.

    sklearn whitens internally (PCA pre-processing), so this is efficient
    even for D=784.  Raises a ConvergenceWarning if max_iter is hit.
    """
    X = np.asarray(X, dtype=np.float64)
    ica = FastICA(
        n_components=n_components,
        whiten="unit-variance",
        max_iter=max_iter,
        tol=tol,
        random_state=seed,
    )
    ica.fit(X)

    # components_: (n_components, D) — unmixing
    # mixing_:     (D, n_components) — mixing matrix; columns = atoms
    mixing_rows = ica.mixing_.T.copy()   # (n_components, D) — atoms as rows

    return ICAResult(
        components=ica.components_.copy(),
        mixing=mixing_rows,
        mean=ica.mean_.copy(),
        n_iter=ica.n_iter_,
    )


def project(X: np.ndarray, result: ICAResult) -> np.ndarray:
    """Project (N, D) array onto ICA components. Returns (N, n_components) sources."""
    return (np.asarray(X, dtype=np.float64) - result.mean) @ result.components.T


def reconstruct(sources: np.ndarray, result: ICAResult) -> np.ndarray:
    """Reconstruct (N, D) from (N, n_components) sources."""
    return sources @ result.mixing + result.mean


def reconstruction_error(X: np.ndarray, result: ICAResult) -> float:
    """Mean squared reconstruction error per dimension."""
    X = np.asarray(X, dtype=np.float64)
    return float(((X - reconstruct(project(X, result), result)) ** 2).mean())
