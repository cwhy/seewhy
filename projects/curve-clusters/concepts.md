# Curve Clusters — Concepts

## The idea in plain words

A pile of data points with no natural order. Store it as a set of smooth
**curves** rather than a set of **points**.

To store a point: find the spot on some curve that already looks most like it,
then nudge the curve at that spot towards the point. To read back: go to a spot
and read off the curve. If a new point already matches what the curve says,
the nudge is tiny and nothing gets stored — that is where compression comes
from.

## The trade being tested

Two ways to spend the same memory:

- **K free points.** Representative points, anywhere, unconnected. Online k-means.
- **K curves.** Roughly the same number of resolvable spots, but forced to lie
  along connected lines — in exchange, everything *along* each line is also
  representable, for free.

The curve pays a constraint (its spots must be threadable in one order) and
gets a freebie (the segments between them). **The idea is worth pursuing iff
the freebie outweighs the constraint.**

## Why the dial contains its own baselines

A curve is N orthogonal-polynomial (shifted Legendre) coefficients per input
dimension. The order N is the dial:

| N | shape | equivalent to |
|---|-------|---------------|
| 1 | constant | a point — **online k-means** |
| 2 | straight segment | local PCA with 1 component |
| ≥3 | bent curve | the idea under test |

So balls-vs-curves is not two systems to compare. It is one system with the
dial turned. Holding the budget `K·N·D` fixed and sweeping N traces the trade
on one axis, and the reading is unambiguous:

- best N = 1 → balls win, spend the budget on more clusters, idea is dead
- best N = 2 → straight sticks win, use local PCA and drop the polynomials
- best N ≥ 3 → **bend pays for itself**

`lib/curves.py` proves the N=1 equivalence rather than asserting it:
`experiments1.py --selftest` runs the reference centroid rule on the same
stream and requires agreement to floating point (observed 5.6e-17). The
baseline column is therefore a genuine k-means, not a lookalike.

## Regime

No training, no epochs, no gradients on parameters. One pass over the data,
every decision made at inference time, per point. Look-back at old data is
permitted but not yet used (see Open questions).

**Assignment is exhaustive.** For each point, every spot on every curve is
scanned, so the chosen spot is the true global optimum — not what a
gradient-based inner loop or an amortised encoder happened to find. This is
what the 1-D latent buys, and it is only affordable because the latent is 1-D:
the scan is `K·G·D` per point, the same kernel shape as a codebook lookup.

**The write is closed-form.** Minimum-norm nudge along `φ(p*)`, which spreads
over the reproducing kernel `K(p*,·) = φ(p*)ᵀφ(·)` — a bump of width ~1/N, not
a discrete slot. Step size `α = 1/(1 + count/N)`; at N=1 that collapses exactly
to the running-mean centroid rule, which is why the selftest passes.

**State is held as grid values**, `Ĝ[j] = Φ C_j`, never as coefficients. Every
update adds `Φφ(p*)eᵀ`, which stays in the N-dimensional column space of Φ, so
the represented budget is still `K·N·D`; G is scan resolution, not memory.

## Metrics

`distortion` — mean squared distance from a point to its nearest spot on any
curve. Data is variance-normalised per dataset, so this reads directly as
**fraction of variance unexplained**, comparable across dimensions.

A value ≥ 1.0 on held-out data from the same distribution is impossible for a
working model — it means worse than predicting the global mean. Treat it as a
bug signal. It caught one (see below).

## Findings

*(appended as they land)*

- **N=1 is exactly online k-means**, to 5.6e-17, with identical assignment
  counts. The sweep is anchored on a real baseline.
- **A train/test distribution leak, caught by the ≥1.0 tell.** The first
  version of `toy.py` passed the same live rng to two builder calls, so
  `blobs10d` drew *fresh cluster centres for the test split*. Held-out
  distortion came out at 1.17 — worse than the global mean — which is the
  symptom to watch for. Fixed by building each dataset as a sampler once per
  (name, seed) and drawing both splits from it.
- **The optimum order is N=3, everywhere.** All five low-dimensional datasets
  put their minimum at N=3 at budget 128; N≥12 is worse than k-means on
  several. HiPPO's own function-approximation experiment uses N=256. The order
  that makes HiPPO work is two orders of magnitude from the order that helps
  here.
- **Balls → sticks is the whole story: 30–70% distortion reduction, 8/8 seeds,
  every dataset.** Sticks → bend adds 10–15% on low-dimensional data and ~1%
  on the high-dimensional controls. The polynomial machinery earns the last
  10%; the first 60% is local PCA.
- **`checkerboard` confirmed the connectedness prediction.** Disconnected
  support means a curve must cross empty squares, and bend duly pays less
  there (0.88–0.98) than on the connected square of the same intrinsic
  dimension (0.81–0.88).
- **High order fails by tangling, visibly.** At N=32, K=4 the curves make long
  excursions and loops through empty space — a few long curves need a global
  ordering they cannot find in one pass. Three passes barely helps, so it is
  not undertraining.
- **A confound that inverted a conclusion.** The budget sweep appeared to show
  the sticks advantage decaying as budget grew, i.e. "does not scale". But
  budget and points-per-cluster are confounded — at B=512, N=2 each of 256
  clusters sees 8 points. Holding budget fixed and raising n_train, the
  advantage *grows* (spiral B=512: 0.701 → 0.609 → 0.548). exp2's decay was
  estimator noise; the geometric prediction stands.

**Reports** — one per research question, each standalone with full background,
all built by `scripts/gen_reports.py` with every table computed from
`results.jsonl` at build time:

| | |
|---|---|
| [Overview](https://media.tanh.xyz/seewhy/26-09-09/curve_clusters_probe.html) | entry point |
| [Q1 · curve vs ball](https://media.tanh.xyz/seewhy/26-09-09/curve_clusters_q1_shape.html) | exp1 |
| [Q2 · budget and data](https://media.tanh.xyz/seewhy/26-09-09/curve_clusters_q2_scaling.html) | exp2, exp3 |
| [Q3 · addresses / MDL](https://media.tanh.xyz/seewhy/26-09-09/curve_clusters_q3_addresses.html) | exp4 |
| [Q4 · which dimension](https://media.tanh.xyz/seewhy/26-09-09/curve_clusters_q4_dimension.html) | exp5, exp6 |
| [Q5 · K or N](https://media.tanh.xyz/seewhy/26-09-09/curve_clusters_q5_k_or_n.html) | exp7 |

**Filter rows by `experiment` prefix, never by incidental fields.** exp4 rows
also carry `budget=128` and `n_passes=1`, so a predicate built from those
silently averages exp4's coarse-grid sweep into exp1's order sweep. That
produced two wrong figures and one wrong table before it was caught; both
`lib/figures.py` and `scripts/gen_reports.py` now have a `from_exp()` helper
and use it everywhere.

## Open questions

- ~~Does the N=2 → N=3 margin survive seed noise?~~ Yes, but it is only
  10–15%, and only on low-dimensional data.
- ~~High orders do badly — tangling or step-size artefact?~~ Tangling, and it
  is visible in `curve_clusters_fitted.png`. Whether an annealing schedule
  (start at low order, reveal coefficients as residuals demand) recovers the
  high-order regime is still open — and it is the one change that could make
  order >3 useful.
- Look-back is allowed by the regime but unused. The cheap version is local
  repair: when one cluster's residuals stay high, re-fit *that curve* from a
  small buffer of its own members. Buffer is one cluster's worth, not the
  dataset's.
- Cluster birth-and-growth (start as a point, grow order as residuals demand)
  is implemented in spirit — `_seed` makes a constant curve — but K is fixed
  rather than spawned, to keep the budget matched for this sweep.

## Prior art to check before claiming novelty

Clusters-as-flat-pieces is well established (k-flats, mixtures of PPCA).
Clusters-as-curves exists as principal-curve clustering / local principal
curves. The single-curve global version is principal curves (Hastie &
Stuetzle) and, as an algorithm, a 1-D self-organising map. What plausibly is
not covered: the **single-pass** version with exact 1-D search, closed-form
minimum-norm writes, and clusters that grow their own order.

## Layout

| file | purpose |
|---|---|
| `lib/curves.py` | the algorithm — `CurveClusters`, Legendre basis |
| `lib/toy.py` | six toy datasets spanning intrinsic dimension |
| `lib/figures.py` | distortion-vs-order plot |
| `experiments1.py` | the budget-matched order sweep, plus `--selftest` |
| `results.jsonl` | one row per (dataset, order, passes, seed) |

Deviation from the v3 start guide: no `paper/`, no runner scripts. This is a
single go/no-go probe that runs in minutes on CPU; scaffold the full tiers only
if the answer comes back positive.
