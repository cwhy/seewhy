"""
Pure-function CCA utilities: pairwise 2-set CCA and Multiset CCA (MCCA).

──────────────────────────────────────────────────────────────────────────────
Pairwise CCA  (fit_cca)
──────────────────────────────────────────────────────────────────────────────
Standard 2-set CCA between datasets A (N_a × D) and B (N_b × D).

Algorithm:
    1. Center:      Xc_a = X_a − mean_a,  Xc_b = X_b − mean_b
    2. Within-set covariances (regularised):
           C_aa = Xc_a.T @ Xc_a / N_a  +  reg·I
           C_bb = Xc_b.T @ Xc_b / N_b  +  reg·I
    3. Whitening matrices:  W_a = C_aa^{−1/2},  W_b = C_bb^{−1/2}
    4. Cross-covariance (balanced N):
           C_ab = (1/N_min) Xc_a[:N_min].T @ Xc_b[:N_min]
    5. Whitened cross-covariance:  M = W_a @ C_ab @ W_b.T
    6. SVD of M:  M = U S V.T
       - Canonical correlations = S[:n_components]
       - Pixel-space directions for A: W_a @ U[:, :n]   →  (D, n)
       - Pixel-space directions for B: W_b @ V[:, :n]   →  (D, n)

Projection of class A:  (X_a − mean_a) @ W_a @ U[:, :n]
Projection of class B:  (X_b − mean_b) @ W_b @ V[:, :n]

──────────────────────────────────────────────────────────────────────────────
Multiset CCA  (fit_mcca)  — MAXVAR criterion

Given K datasets X_0, …, X_{K-1} (each N_k × D), find n_components shared
directions that maximize explained variance across all whitened datasets.

Algorithm (pooled-whitening MAXVAR):
    1. Center each class:   Xc_k = X_k − mean_k
    2. Pooled within-class covariance:
           C_w = (1/K) Σ_k  Xc_k.T @ Xc_k / N_k
    3. Whitening matrix:    W = C_w^{−1/2}   (via eigendecomposition + reg)
    4. Whiten each class:   Xw_k = Xc_k @ W    (now I-covariance within each class)
    5. Subsample to N_min per class for balance; stack → Z  (K·N_min × D)
    6. Economy SVD of Z:    Z ≈ U S Vt
       Right singular vectors (rows of Vt) are the canonical directions in
       whitened space.

Projection onto canonical subspace for class k:
    scores_k = (X_k − mean_k) @ W @ Vt[:n].T

Canonical directions in pixel space (for visualization):
    v_pixel_i = W @ Vt[i]       (reshape to 28×28 for MNIST)

All functions are pure: numpy in, numpy/NamedTuple out — no I/O, no JAX.
"""

import numpy as np
from typing import NamedTuple


class CCAPairResult(NamedTuple):
    directions_a: np.ndarray    # (n_components, D)  canonical directions in pixel space for A
    directions_b: np.ndarray    # (n_components, D)  canonical directions in pixel space for B
    correlations: np.ndarray    # (n_components,)    canonical correlations (singular values of M)
    mean_a: np.ndarray          # (D,)
    mean_b: np.ndarray          # (D,)
    digit_a: int
    digit_b: int


def fit_cca(
    X_a: np.ndarray,
    X_b: np.ndarray,
    digit_a: int,
    digit_b: int,
    n_components: int,
    reg: float = 1e-4,
    seed: int = 0,
) -> CCAPairResult:
    """
    Standard 2-set CCA between datasets X_a (N_a, D) and X_b (N_b, D).

    Returns CCAPairResult — see module docstring for full algorithm.
    directions_a[i] and directions_b[i] are the paired pixel-space canonical
    directions for component i; correlations[i] is their canonical correlation.
    """
    X_a = np.asarray(X_a, dtype=np.float64)
    X_b = np.asarray(X_b, dtype=np.float64)
    D = X_a.shape[1]
    assert X_b.shape[1] == D

    mean_a, mean_b = X_a.mean(0), X_b.mean(0)
    Xc_a, Xc_b = X_a - mean_a, X_b - mean_b

    # Within-set covariances
    C_aa = Xc_a.T @ Xc_a / len(Xc_a) + reg * np.eye(D)
    C_bb = Xc_b.T @ Xc_b / len(Xc_b) + reg * np.eye(D)

    # Whitening matrices  W = C^{-1/2}
    W_a = _sym_matrix_power(C_aa, -0.5, reg)
    W_b = _sym_matrix_power(C_bb, -0.5, reg)

    # Cross-covariance on balanced subsample
    N_min = min(len(Xc_a), len(Xc_b))
    rng = np.random.default_rng(seed)
    idx_a = rng.choice(len(Xc_a), N_min, replace=False)
    idx_b = rng.choice(len(Xc_b), N_min, replace=False)
    C_ab = Xc_a[idx_a].T @ Xc_b[idx_b] / N_min

    # Whitened cross-covariance and SVD
    M = W_a @ C_ab @ W_b.T          # (D, D)
    U, s, Vt = np.linalg.svd(M, full_matrices=False)

    # Pixel-space directions: rows = components
    dirs_a = (W_a @ U[:, :n_components]).T    # (n_components, D)
    dirs_b = (W_b @ Vt[:n_components].T).T    # (n_components, D)

    return CCAPairResult(
        directions_a=dirs_a,
        directions_b=dirs_b,
        correlations=s[:n_components],
        mean_a=mean_a,
        mean_b=mean_b,
        digit_a=digit_a,
        digit_b=digit_b,
    )


def project_cca_a(X: np.ndarray, result: CCAPairResult) -> np.ndarray:
    """Project dataset A onto its canonical subspace. Returns (N, n_components)."""
    return (X - result.mean_a) @ result.directions_a.T


def project_cca_b(X: np.ndarray, result: CCAPairResult) -> np.ndarray:
    """Project dataset B onto its canonical subspace. Returns (N, n_components)."""
    return (X - result.mean_b) @ result.directions_b.T


class MCCAResult(NamedTuple):
    components: np.ndarray           # (n_components, D)  canonical directions in whitened space
    components_pixel: np.ndarray     # (n_components, D)  canonical directions in pixel space
    eigenvalues: np.ndarray          # (n_components,)    squared singular values from step 6
    means: list                      # K × (D,)           per-class means
    whitener: np.ndarray             # (D, D)             C_w^{-1/2}


def _sym_matrix_power(C: np.ndarray, power: float, reg: float) -> np.ndarray:
    """
    Compute C^power for symmetric PSD matrix C via eigendecomposition.
    reg is added to eigenvalues before raising to the power (avoids division by zero).
    """
    eigvals, eigvecs = np.linalg.eigh(C)
    eigvals = np.maximum(eigvals, reg)
    return eigvecs @ np.diag(eigvals ** power) @ eigvecs.T


def fit_mcca(
    Xs: list[np.ndarray],
    n_components: int,
    reg: float = 1e-4,
    n_subsample: int | None = None,
    seed: int = 0,
) -> MCCAResult:
    """
    Fit pooled-whitening MAXVAR MCCA.

    Parameters
    ----------
    Xs          : list of K arrays, each (N_k, D).  D must match across all.
    n_components: number of canonical components to return.
    reg         : regularisation added to eigenvalues of C_w before inverting.
    n_subsample : samples per class when stacking for the SVD.
                  None → use min(N_k) across classes (balanced).
    seed        : RNG seed for subsampling.

    Returns
    -------
    MCCAResult  — see module docstring for field descriptions.
    """
    K = len(Xs)
    D = Xs[0].shape[1]
    assert all(X.shape[1] == D for X in Xs), "All datasets must have the same D."

    # Step 1: center each class
    means = [X.mean(axis=0) for X in Xs]
    centered = [X - m for X, m in zip(Xs, means)]

    # Step 2: pooled within-class covariance
    C_w = sum(Xc.T @ Xc / len(Xc) for Xc in centered) / K   # (D, D)

    # Step 3: whitening matrix  W = C_w^{-1/2}
    W = _sym_matrix_power(C_w, -0.5, reg)   # (D, D)

    # Step 4: whiten
    whitened = [Xc @ W for Xc in centered]  # list of (N_k, D)

    # Step 5: subsample to N_min for balance
    N_min = min(len(X) for X in Xs)
    if n_subsample is None:
        n_subsample = N_min
    rng = np.random.default_rng(seed)
    Z_parts = []
    for Xw in whitened:
        idx = rng.choice(len(Xw), n_subsample, replace=(n_subsample > len(Xw)))
        Z_parts.append(Xw[idx])
    Z = np.vstack(Z_parts)   # (K * n_subsample, D)

    # Step 6: SVD → canonical directions
    _, s, Vt = np.linalg.svd(Z, full_matrices=False)  # Vt: (min(K*n_sub, D), D)

    components = Vt[:n_components]                        # (n_components, D) in whitened space
    components_pixel = (W @ components.T).T               # (n_components, D) in pixel space

    return MCCAResult(
        components=components,
        components_pixel=components_pixel,
        eigenvalues=s[:n_components] ** 2,
        means=means,
        whitener=W,
    )


def project_class(X: np.ndarray, class_idx: int, result: MCCAResult) -> np.ndarray:
    """
    Project class k data onto the MCCA canonical subspace.
    Returns (N, n_components) scores.
    """
    Xc = np.asarray(X, dtype=np.float64) - result.means[class_idx]
    return (Xc @ result.whitener) @ result.components.T


def project_all(Xs: list[np.ndarray], result: MCCAResult) -> list[np.ndarray]:
    """
    Project each class dataset onto the canonical subspace.
    Returns list of K arrays, each (N_k, n_components).
    """
    return [project_class(X, k, result) for k, X in enumerate(Xs)]


def canonical_correlations(scores_list: list[np.ndarray]) -> np.ndarray:
    """
    Compute mean pairwise Pearson correlation for each canonical component
    across K projected datasets.

    scores_list : K arrays, each (N_k, n_components).
    Returns     : (n_components,) mean pairwise correlation per component.
    """
    K = len(scores_list)
    n_comp = scores_list[0].shape[1]
    # Standardise each set's scores
    normed = [(s - s.mean(0)) / (s.std(0) + 1e-12) for s in scores_list]
    pairs = 0
    corr_sum = np.zeros(n_comp)
    for i in range(K):
        for j in range(i + 1, K):
            N = min(len(normed[i]), len(normed[j]))
            # Pearson r per component (using subsampled N for equal weighting)
            r = (normed[i][:N] * normed[j][:N]).mean(0)
            corr_sum += r
            pairs += 1
    return corr_sum / pairs
