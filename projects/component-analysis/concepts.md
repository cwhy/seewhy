# Concepts — component-analysis

## PCA (Principal Component Analysis)

Standard dimensionality reduction via economy SVD of the centered data matrix.

**Algorithm** (`lib/pca.py`):
1. Center: `Xc = X - mean(X, axis=0)`
2. Economy SVD: `Xc = U S Vt`  where shapes are `(N,k) (k,) (k,D)`, `k = min(N, D)`
3. Components = rows of `Vt` (eigenvectors of the covariance matrix `Xc.T @ Xc / (N-1)`)
4. Explained variance = `s² / (N-1)` per component

**Visualization**: reshape each component row to (28, 28), normalise to [0,1], display as a grayscale image. These are the "eigendigits" — pixel patterns that explain the most variance.

**Interpretation in this project**:
- **Global PCA**: directions of maximum variance across all 60 000 images. The first few components capture digit stroke width, global intensity, and broad digit shape.
- **Per-digit PCA**: variance structure within one class. Components capture within-digit style variation (slant, thickness, connectivity).

---

## Multiset CCA — MAXVAR Criterion (`lib/cca.py`)

Multiset CCA generalises pairwise CCA to K > 2 datasets. The MAXVAR criterion finds shared directions that maximise the sum of explained variances across all whitened datasets.

**Setup**: K = 10 datasets X₀, …, X₉, one per MNIST digit. Each Xₖ is (Nₖ, 784).

**Algorithm (pooled-whitening MAXVAR)**:
1. **Center** each class: `Xc_k = X_k - mean_k`
2. **Pooled within-class covariance**:
   `C_w = (1/K) Σ_k  Xc_k.T @ Xc_k / N_k`
   This is the average per-class covariance — it captures the within-class pixel variance structure shared across all digits.
3. **Whitening matrix**: `W = C_w^{-1/2}` (via eigendecomposition with regularisation `reg`)
   After whitening, each class dataset has identity covariance.
4. **Whiten**: `Xw_k = Xc_k @ W`
5. **Balance**: subsample each class to N_min = min(Nₖ) images; stack: `Z = [Xw_0; …; Xw_9]` shape `(K·N_min, 784)`
6. **SVD of Z**: `Z = U S Vt`
   Right singular vectors (rows of `Vt`) are canonical directions in whitened space.

**Projection** of class k onto canonical subspace:
```
scores_k = (X_k - mean_k) @ W @ Vt[:n].T
```

**Canonical images** (directions in pixel space):
```
v_pixel_i = W @ Vt[i]     ← reshape to (28, 28) for display
```
This follows from: if `x_w = (x - mean_k) @ W`, then `score = x_w @ v = (x - mean_k) @ W @ v`, so the pixel-space direction is `W @ v`.

**Interpretation**:
- Canonical components are shared across all digit classes after removing within-class variance.
- A high eigenvalue → this direction explains a lot of variance across the whitened (class-normalised) data.
- Compare to global PCA: PCA components are driven by which digits are more numerous or more variable; MCCA treats each digit equally via the balanced subsampling.

**Key hyperparameter**: `reg` — regularisation added to eigenvalues of C_w before inverting. Too small → numerical instability in low-variance pixel directions. Default: `1e-4`.

---

## Comparison: PCA vs MCCA

| | Global PCA | Per-digit PCA | MCCA |
|---|---|---|---|
| Training data | All 60k images | ~6k per digit | 10 balanced classes |
| What it finds | Max overall variance | Max within-digit variance | Max shared variance (after whitening) |
| First component | Roughly "mean digit" contrast | Dominant style for that digit | Cross-class shared structure |
| Use case | Compression, visualisation | Class-specific subspaces | Shared representation |

---

## Metrics

- **Explained variance ratio (EVR)**: fraction of total variance captured by first k PCA components. Logged at k = 1, 2, 5, 10, 20, 50.
- **Reconstruction MSE**: mean squared pixel error when projecting test images onto k components and reconstructing. Lower = better compression.
- **Eigenvalues**: squared singular values from the MCCA SVD step. Higher = more cross-class shared variance on this component.
- **Mean pairwise canonical correlation**: Pearson r between class-k and class-l projections onto each canonical component, averaged over all C(10,2)=45 pairs.
- **Score std ratio**: per-digit score std divided by global score std per component. Reveals which digits "use" each canonical direction more.
