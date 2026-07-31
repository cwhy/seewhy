"""Is a POSITION-INDEPENDENT value statistic enough to solve 1-shot 0v1 but not 4v9?

Hypothesis for the one working case (anonymised 0-vs-1): it is solved through a channel
that ignores WHERE values are and uses only HOW MANY of each — which is exactly what
summing value embeddings computes. Two earlier results support this indirectly, both by
destroying that channel:

    MLP-combiner embedding (entangles pos into value):  0v1  0.992 -> 0.523
    PCA features (value semantics become per-position): 0v1  1.000 -> 0.508

This tests it directly by handing a 1-shot matcher strictly less information:

    full       pixel-by-pixel comparison on shared positions
    histogram  only the multiset of values — position discarded entirely
    ink-count  only ONE scalar: the fraction of non-background pixels

If 0v1 stays high as information is stripped away and 4v9 does not, the asymmetry
between the working and failing case is explained without reference to architecture.

Usage (server): uv run python projects/universal-ar/scripts/marginal_test.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from shared_lib.datasets import load_supervised_image

K = 32
N_OBS = 196          # matches OBS_FRAC=0.25 in the experiments
N_TRIALS = 3000

d = load_supervised_image("mnist")
X = np.asarray(d.X.reshape(d.n_samples, -1), np.float32)
y = np.asarray(d.y)
binv = lambda v: np.floor(v / 255.0 * (K - 1))


def run(pair, mode, rng):
    idx = {c: np.where(y == c)[0] for c in pair}
    ok = 0
    for _ in range(N_TRIALS):
        qc = int(rng.integers(2))
        q = X[idx[pair[qc]][rng.integers(len(idx[pair[qc]]))]]
        sup = [X[idx[c][rng.integers(len(idx[c]))]] for c in pair]
        oq = rng.permutation(784)[:N_OBS]
        osup = [rng.permutation(784)[:N_OBS] for _ in range(2)]
        sc = []
        for j in range(2):
            if mode == "full":                       # position-specific comparison
                sh = np.intersect1d(oq, osup[j])
                sc.append(float(np.mean((binv(q[sh]) - binv(sup[j][sh])) ** 2)) if len(sh) > 2 else 1e9)
            elif mode == "histogram":                # value multiset, position discarded
                hq = np.bincount(binv(q[oq]).astype(int), minlength=K) / N_OBS
                hs = np.bincount(binv(sup[j][osup[j]]).astype(int), minlength=K) / N_OBS
                sc.append(float(((hq - hs) ** 2).sum()))
            else:                                    # a single scalar
                sc.append(abs(float((binv(q[oq]) > 0).mean() - (binv(sup[j][osup[j]]) > 0).mean())))
        ok += int(int(np.argmin(sc)) == qc)
    return ok / N_TRIALS


rng = np.random.default_rng(0)
print(f"1-shot accuracy, {N_OBS} of 784 pixels observed, {N_TRIALS} trials (chance 0.500)\n")
print(f"{'pair':<10}{'full':>10}{'histogram':>12}{'ink-count':>12}")
for pair in [(0, 1), (4, 9), (3, 8), (5, 6)]:
    vals = [run(pair, m, rng) for m in ("full", "histogram", "ink")]
    print(f"{str(pair):<10}" + "".join(f"{v:>12.3f}" for v in vals))
print("\nfull      = position-specific comparison")
print("histogram = value multiset only, WHERE each value sits is discarded")
print("ink-count = one scalar per sample")
