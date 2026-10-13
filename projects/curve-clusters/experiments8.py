"""exp8 -- Trimmed centre updates: move the centre using only the closest half.

The change
----------
Standard online k-means moves a centre toward every point assigned to it. Here
the centre moves toward only the closest `q` fraction of its assigned points and
ignores the rest. Assignment is untouched -- every point still goes to its
nearest centre, so distortion is measured over all data; only the write is
restricted. q = 1.0 is the standard rule, so the baseline is inside the method.

Stated before running: on plain mean-squared distortion this MUST lose
-------------------------------------------------------------------------
For a fixed partition the mean is the minimiser of the sum of squared
distances -- that is what a mean is. So a trimmed centre is strictly worse on
that metric by construction, and a result saying "distortion got worse" carries
no information. Reporting only that would be a rigged test.

Trimming is a robustness device. Trimmed centres sit in the dense core of a
cluster instead of being dragged outward by outliers and boundary points. So it
should win on:

  * clean-data distortion when the TRAINING data is contaminated -- the honest
    robustness test, since a trimmed centre should ignore the junk
  * trimmed distortion (mean squared error over the best-fitting 50% / 90% of
    points), which is the objective a trimmed update actually descends

and it should lose on plain distortion over uncontaminated data. Predictions:

    eps=0,    plain distortion       q<1 worse    (certain, by the above)
    eps=0,    trimmed distortion     q<1 better
    eps>0,    clean-test distortion  q<1 better   (the interesting one)

If the third does not hold, trimming has no use here at all.

Design
------
Contamination replaces `eps` of the TRAINING points with uniform draws over a
box three times the data's scale. The test split stays clean, so "clean-test
distortion" asks how well the model represents real data after being shown junk.

N=2 is run alongside N=1 because the trimmed write costs nothing extra for
curves and it is worth knowing whether the effect carries over.

Usage:
    python projects/curve-clusters/experiments8.py
    python projects/curve-clusters/experiments8.py --selftest   # q=1 == untrimmed
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib.curves import CurveClusters
from lib import toy
from lib.toy import contaminate
from experiments6 import real_data, subspace_data

K = 64
ORDERS = (1, 2)
QS = (0.25, 0.5, 0.75, 1.0)
EPS = (0.0, 0.05, 0.20)
SEEDS = tuple(range(6))
GRID = 32

DATASETS = {
    "spiral2d": lambda s: toy.load("spiral2d", seed=s),
    "square2d": lambda s: toy.load("square2d", seed=s),
    "sub8_in40": lambda s: subspace_data(8, s),
    "mnist_pca32": lambda s: real_data("mnist", 32, s),
}

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results.jsonl"


def done_names() -> set[str]:
    if not RESULTS.exists():
        return set()
    out = set()
    for line in RESULTS.read_text(encoding="utf-8").splitlines():
        try:
            out.add(json.loads(line)["experiment"])
        except Exception:
            continue
    return out


def selftest() -> None:
    """q=1.0 must reproduce the untrimmed model exactly, not approximately."""
    train, _ = toy.load("square2d", seed=0)
    a = CurveClusters(16, 1, train.shape[1], 32)
    b = CurveClusters(16, 1, train.shape[1], 32, trim_q=1.0)
    rng = lambda: np.random.default_rng(7).permutation(len(train))
    a.partial_fit(train[rng()])
    b.partial_fit(train[rng()])
    gap = float(np.abs(a.values - b.values).max())
    print(f"max |untrimmed - trim_q=1.0| = {gap:.3e}")
    assert gap == 0.0, "trim_q=1.0 is not the untrimmed rule"
    # and trimming must actually reject something
    c = CurveClusters(16, 1, train.shape[1], 32, trim_q=0.5)
    c.partial_fit(train[rng()])
    print(f"trim_q=0.5: overall accepted {c.counts.sum() / max(1, c.seen.sum()):.3f} "
          f"(inflated by {c.warm_accepted} unconditional warmup acceptances), "
          f"gated rate {c.gated_accept_rate:.3f} (target 0.50)")
    assert 0.42 < c.gated_accept_rate < 0.58, \
        f"quantile tracker off target: {c.gated_accept_rate:.3f}"
    print("selftest passed")


def main() -> None:
    todo = done_names()
    for ds, load in DATASETS.items():
        for seed in SEEDS:
            base_tr, te = load(seed)
            for eps in EPS:
                tr, bad_mask = contaminate(base_tr, eps, seed)
                n_bad = int(bad_mask.sum())
                for order in ORDERS:
                    for q in QS:
                        name = f"exp8_{ds}_N{order}_q{q}_e{eps}_s{seed}"
                        if name in todo:
                            continue
                        m = CurveClusters(K, order, tr.shape[1], GRID, trim_q=q)
                        rng = np.random.default_rng(1000 + seed)
                        t0 = time.time()
                        m.partial_fit(tr[rng.permutation(len(tr))])
                        row = {
                            "experiment": name, "dataset": ds, "trim_q": q,
                            "eps": eps, "n_contaminated": n_bad, "order": order,
                            "n_curves": K, "grid": GRID, "n_passes": 1,
                            "seed": seed, "dim": int(tr.shape[1]),
                            "n_params": m.n_params,
                            "accept_frac": round(float(m.counts.sum() / max(1, m.seen.sum())), 4),
                            "gated_accept_rate": round(float(m.gated_accept_rate), 4)
                            if m.gated_seen else None,
                            "warm_accepted": int(m.warm_accepted),
                            "gated_seen": int(m.gated_seen),
                            "distortion_test": m.distortion(te),
                            "trimmed50_test": m.trimmed_distortion(te, 0.5),
                            "trimmed90_test": m.trimmed_distortion(te, 0.9),
                            "time_s": round(time.time() - t0, 3),
                        }
                        with RESULTS.open("a", encoding="utf-8") as fh:
                            fh.write(json.dumps(row) + "\n")
            print(f"{ds:<13s} seed {seed} done", flush=True)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        main()
