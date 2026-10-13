"""Single-pass clustering where each cluster is a curve, not a point.

The whole idea in one line: a cluster is a smooth curve `M_j(p)` for p in [0,1],
written as N orthogonal-polynomial coefficients per input dimension. A point is
stored by finding the spot on some curve that already looks most like it, then
nudging the curve at that spot towards the point.

`N` is the dial that contains the baseline:

    N = 1   the curve is a constant, i.e. a single point  -> online k-means
    N = 2   a straight segment                            -> local PCA-1
    N >= 3  an actually bent curve                        -> the idea under test

So "balls vs curves" is not two systems to compare. It is one system with the
dial turned. At fixed budget `K * N * D`, sweeping N traces the trade directly,
and N=1 is a *provably exact* online k-means rather than a reimplementation of
one -- see the alpha derivation below.

No training, no epochs, no gradients on parameters. Every decision is made at
inference time, per point, in one pass.

Representation note
-------------------
State is held as grid *values* `Ghat[j] = Phi @ C_j`, shape (K, G, D), never as
the coefficients themselves. Every update adds `k(p*) e^T = Phi phi(p*) e^T`,
which lies in the column space of Phi, so Ghat stays inside an N-dimensional
subspace at all times and still faithfully represents an N-coefficient curve.
The stored budget is therefore `K * N * D`, not `K * G * D`; G is a scan
resolution, not memory. Working in values makes each write O(G*D) instead of
O(N*D) plus a re-evaluation, and turns the neighbourhood into a column lookup
in the precomputed kernel.
"""

from __future__ import annotations

import numpy as np


def legendre_basis(p: np.ndarray, n: int) -> np.ndarray:
    """Shifted Legendre basis on [0,1], orthonormal: int_0^1 phi_i phi_j = delta_ij.

    Returns (len(p), n). Column k is sqrt(2k+1) * P_k(2p-1).
    """
    x = 2.0 * p - 1.0
    out = np.zeros((len(p), n))
    out[:, 0] = 1.0
    if n > 1:
        out[:, 1] = x
    for k in range(1, n - 1):
        out[:, k + 1] = ((2 * k + 1) * x * out[:, k] - k * out[:, k - 1]) / (k + 1)
    return out * np.sqrt(2.0 * np.arange(n) + 1.0)


class CurveClusters:
    """K curves of order N over D dimensions, fitted by single-pass assignment.

    Parameters
    ----------
    n_curves, order, dim : the budget. Stored numbers = n_curves * order * dim.
    grid : scan resolution. Must comfortably exceed `order`; the curve has
        roughly `order` resolvable spots, so a grid near `order` would alias.
    """

    def __init__(self, n_curves: int, order: int, dim: int, grid: int,
                 trim_q: float = 1.0, warmup: int = 10) -> None:
        """`trim_q` < 1 moves each centre using only its closest `trim_q`
        fraction of assigned points; `trim_q = 1.0` is the untrimmed rule.

        Assignment is never trimmed -- every point still goes to its nearest
        spot, so distortion is measured over all data. Only the write is
        restricted.

        The acceptance threshold is a per-cluster stochastic quantile estimate,
        so the extra state is O(K), not O(K x window). A buffer of recent
        distances would estimate the quantile better but costs K*W floats,
        which for K=64, W=32 would roughly double the model's own memory on a
        32-dimensional input and make the comparison unfair.

        `warmup` points per cluster are accepted unconditionally, because the
        threshold is meaningless before the cluster has any shape and a bad
        early estimate can reject everything and freeze the cluster.
        """
        self.K, self.N, self.D, self.G = n_curves, order, dim, grid
        self.trim_q, self.warmup = float(trim_q), int(warmup)
        p = (np.arange(grid) + 0.5) / grid
        phi = legendre_basis(p, order)          # (G, N)
        self.kernel = phi @ phi.T               # (G, G) reproducing kernel
        self.kdiag = np.diag(self.kernel).copy()
        self.values = np.zeros((n_curves, grid, dim))
        self.counts = np.zeros(n_curves, dtype=np.int64)      # accepted writes
        self.seen = np.zeros(n_curves, dtype=np.int64)        # assignments, accepted or not
        self.thresh = np.full(n_curves, np.inf)               # acceptance distance
        self.mean_d = np.zeros(n_curves)                      # running mean distance, sets the step scale
        # Warmup acceptances are unconditional, so they must be counted apart
        # from gated ones. An overall accept fraction mixes the two and reads
        # as a mis-calibrated tracker when the tracker is fine: at K=64 over
        # 2000 points the warmup alone is a third of all assignments.
        self.warm_accepted = 0
        self.gated_seen = 0
        self.gated_accepted = 0
        self.n_seeded = 0
        self._flat = self.values.reshape(-1, dim)
        self._sqn = np.zeros(n_curves * grid)

    @property
    def n_params(self) -> int:
        return self.K * self.N * self.D

    @property
    def gated_accept_rate(self) -> float:
        """Acceptance rate among points the threshold actually judged.

        This is the number to compare against `trim_q`. The overall rate is not:
        it includes the unconditional warmup acceptances and so always reads
        high, by 1 - gated_seen/seen.
        """
        return self.gated_accepted / self.gated_seen if self.gated_seen else float("nan")

    def _seed(self, x: np.ndarray) -> None:
        """A new cluster is born as a constant curve sitting on its first point.

        Order-N structure is not imposed at birth; it accumulates only where
        residuals demand it, so a cluster that stays sparse stays a ball.
        """
        j = self.n_seeded
        self.values[j, :, :] = x
        self._sqn[j * self.G:(j + 1) * self.G] = (x ** 2).sum()
        self.n_seeded += 1

    def _assign(self, x: np.ndarray) -> tuple[int, int, np.ndarray]:
        """Exhaustive scan over (which curve, where on it). Global, not local.

        This is the step the 1-D latent buys: every spot on every curve is
        evaluated, so the returned optimum is the true one, not whatever a
        gradient-based inner loop or an amortised encoder happened to find.
        """
        live = self.n_seeded * self.G
        d2 = self._sqn[:live] - 2.0 * (self._flat[:live] @ x)
        best = int(np.argmin(d2))
        j, g = divmod(best, self.G)
        return j, g, x - self.values[j, g]

    def _write(self, j: int, g: int, e: np.ndarray) -> None:
        """Minimum-norm nudge: move the curve so it fits better at p*, changing
        the coefficients as little as possible.

        The step size is what makes N=1 exactly online k-means. A write at p*
        spreads over a kernel bump of width ~1/N, so each of the curve's ~N
        resolvable regions is hit about `counts/N` times; averaging within a
        region therefore wants a step of 1/(1 + counts/N). At N=1 the curve has
        one region, alpha = 1/(1+counts), and the update collapses to the
        running-mean centroid rule. Larger N decays more slowly, which is
        correct -- those writes are shared among more regions.
        """
        alpha = 1.0 / (1.0 + self.counts[j] / self.N)
        block = slice(j * self.G, (j + 1) * self.G)
        self.values[j] += (alpha / self.kdiag[g]) * np.outer(self.kernel[:, g], e)
        self._sqn[block] = (self._flat[block] ** 2).sum(1)
        self.counts[j] += 1

    def _accept(self, j: int, dist: float) -> bool:
        """Is this point in the closest `trim_q` fraction of cluster j so far?

        Updates the running mean distance and the stochastic quantile estimate
        on every assigned point, accepted or not -- the threshold has to track
        the distribution of everything assigned, not of what it let through.
        """
        self.seen[j] += 1
        n = self.seen[j]
        self.mean_d[j] += (dist - self.mean_d[j]) / n
        if self.trim_q >= 1.0:
            return True
        if n <= self.warmup:
            self.thresh[j] = self.mean_d[j]
            self.warm_accepted += 1
            return True
        below = 1.0 if dist <= self.thresh[j] else 0.0
        # Robbins-Monro on P(d <= t) = q; step scaled by the distance scale so
        # it is dimensionless, and decaying so the estimate settles.
        self.thresh[j] += 0.5 * self.mean_d[j] * (self.trim_q - below) / n ** 0.5
        self.thresh[j] = max(self.thresh[j], 1e-12)
        ok = dist <= self.thresh[j]
        self.gated_seen += 1
        self.gated_accepted += int(ok)
        return ok

    def partial_fit(self, data: np.ndarray) -> None:
        """One pass. Points are consumed in the order given and not revisited."""
        for x in data:
            if self.n_seeded < self.K:
                self._seed(x)
                continue
            j, g, e = self._assign(x)
            if self._accept(j, float(np.sqrt((e ** 2).sum()))):
                self._write(j, g, e)

    def squared_errors(self, data: np.ndarray) -> np.ndarray:
        """Per-point squared distance to the best spot on any curve."""
        live = self._flat[:self.n_seeded * self.G]
        d2 = self._sqn[:self.n_seeded * self.G][None, :] - 2.0 * (data @ live.T)
        return d2.min(1) + (data ** 2).sum(1)

    def distortion(self, data: np.ndarray) -> float:
        """Mean squared distance from each point to its best spot on any curve.

        Data is variance-normalised upstream, so this reads directly as the
        fraction of variance the representation fails to explain.
        """
        return float(self.squared_errors(data).mean())

    def trimmed_distortion(self, data: np.ndarray, keep: float) -> float:
        """Mean squared error over the best-fitting `keep` fraction of points.

        The metric a trimmed update rule is actually trying to optimise. Plain
        distortion cannot show trimming in a good light even in principle: for
        a fixed partition the mean is the minimiser of squared distance, so any
        trimmed centre is worse on it by construction.
        """
        e = np.sort(self.squared_errors(data))
        return float(e[:max(1, int(round(keep * len(e))))].mean())
