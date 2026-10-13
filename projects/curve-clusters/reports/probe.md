**Verdict: it depends entirely on whether you charge for addresses — and the
first version of this report got that wrong.**

At a fixed *model* size, curves beat balls by 30–70%. But that comparison
quietly gave curves more address bits, and at matched **rate** k-means wins on
every dataset tested. Under a full description-length accounting the picture is
different again: k-means is cheaper at loose fidelity, and curves become cheaper
— then the only option at all — as the distortion target tightens.

So this is a **high-fidelity-regime method**, not a general improvement on
k-means. The mechanism is an asymmetry in how the two buy rate: k-means needs
K = 2^R centroids to use R address bits, so its model grows exponentially in the
rate, while a curve buys rate from grid resolution at no model cost.

## The plain version

Fixing `K·N` and sweeping N — which is what every experiment above does — hides
K. At budget 128, N=1 runs with K=128 while N=2 runs with K=64, so "N=2 beats
N=1" is really "64 sticks beat 128 points", and the shape effect cannot be
separated from the cluster count.

So sweep K and N independently and plot distortion against stored floats, with
K running along each line. A vertical slice is then an exact equal-cost
comparison, nothing is held fixed by fiat, and there are no ratios.

![K and N independently](https://media.tanh.xyz/seewhy/26-09-09/curve_clusters_k_and_n.png)

Two things are visible immediately, and they are the whole result:

**1. N=1 is the only order that is different.** The black line sits above the
others on every dataset. **The N=2 through N=8 lines lie on top of each other.**
Once a cluster is allowed any extension at all, the order barely matters.

**2. K is the real lever.** Moving right along any single line — more clusters —
buys far more than any change of N at fixed cost.

Best curve versus k-means at exactly equal stored floats:

| dataset | K·N=16 | 32 | 64 | 128 | 256 | best order |
|---|---|---|---|---|---|---|
| mnist_pca32 | +1.1% | +2.1% | +3.5% | +5.0% | +4.4% | N=2 |
| fashion_mnist_pca32 | +7.1% | +7.8% | +6.8% | +7.6% | +8.7% | N=2–4 |
| mnist_raw (784-D) | −0.0% | +0.8% | +2.6% | — | — | N=2 |
| sub8_in40 | +10.1% | +15.1% | +15.8% | +15.1% | +14.6% | N=2–4 |
| gauss_40d | +2.4% | +4.1% | +5.0% | +4.7% | +5.4% | N=2 |

**So: yes, N>1 helps in high dimension — 1–16% depending on the data — and
essentially all of it is the step from N=1 to N=2.** Bending beyond a straight
stick is free to try and worth nothing.

This also corrects an earlier claim in this report. "The optimum is N=3" was an
artefact of pinning `K·N = 128`: with K free, N=2 through N=8 are
indistinguishable, and where a best order does show up it drifts with budget
rather than sitting at 3.


---

## The setup

A cluster is a smooth curve `M_j(p)`, p in [0,1], stored as N shifted-Legendre
coefficients per input dimension. To store a point: scan every spot on every
curve, take the best, and nudge the curve there towards the point. No training,
no epochs, no gradients on parameters — one pass, every decision at inference
time.

The order N is a dial that **contains its own baselines**:

| N | shape | equivalent to |
|---|-------|---------------|
| 1 | constant | a point — online k-means |
| 2 | straight segment | local PCA, 1 component |
| ≥3 | bent curve | the idea under test |

Holding the budget `K·N·D` fixed and sweeping N therefore traces the whole
trade on one axis. Balls-vs-curves is not two systems to compare; it is one
system with the dial turned.

**The N=1 equivalence is proved, not asserted.** `experiments1.py --selftest`
runs the textbook centroid rule on the same stream and compares:

```
max |curve_N1 - kmeans| = 5.551e-17   counts match: True
```

So the N=1 column below is a genuine online k-means, and every ratio against it
is a real baseline comparison.

## What it looks like

![One dial, three settings](https://media.tanh.xyz/seewhy/26-09-09/curve_clusters_comparison.png)

Same algorithm in all three rows, same budget `K·N = 128`, single pass.
Distortion falls down every column.

## What it measures

Data is variance-normalised per dataset, so distortion reads directly as
**fraction of variance unexplained** and is comparable across dimensions.

Order sweep at `K·N = 128`, one pass, 8 seeds:

| dataset | N=1 | N=2 | N=3 | N=4 | N=6 | N=8 | N=16 | N=32 | best |
|---|---|---|---|---|---|---|---|---|---|
| spiral2d | 0.0024 | 0.0011 | **0.0011** | 0.0018 | 0.0031 | 0.0036 | 0.0041 | 0.0118 | N=3 |
| moons2d | 0.0021 | 0.0014 | **0.0012** | 0.0015 | 0.0019 | 0.0024 | 0.0038 | 0.0069 | N=3 |
| square2d | 0.0091 | 0.0048 | **0.0042** | 0.0049 | 0.0053 | 0.0068 | 0.0099 | 0.0136 | N=3 |
| checkerboard | 0.0050 | 0.0029 | **0.0028** | 0.0031 | 0.0046 | 0.0054 | 0.0113 | 0.0169 | N=3 |
| swissroll | 0.0226 | 0.0137 | **0.0125** | 0.0158 | 0.0165 | 0.0185 | 0.0274 | 0.0409 | N=3 |
| blobs10d *(control)* | 0.0089 | **0.0082** | 0.0082 | 0.0083 | 0.0086 | 0.0088 | 0.0096 | 0.0236 | N=2 |
| gauss10d *(control)* | 0.4948 | 0.4389 | 0.4335 | **0.4315** | 0.4339 | 0.4377 | 0.4682 | 0.5033 | N=4 |

![Order sweep](https://media.tanh.xyz/seewhy/26-09-09/curve_clusters_order_sweep_p1.png)

Paired per-seed ratios, so the two orders see identical data draws. Values
below 1 mean the higher order wins; the fraction is how many of 8 seeds agree.

**Balls → sticks (N=2 ÷ N=1):**

| dataset | B=32 | B=128 | B=512 |
|---|---|---|---|
| spiral2d | 0.325 (8/8) | 0.456 (8/8) | 0.614 (8/8) |
| moons2d | 0.446 (8/8) | 0.647 (8/8) | 0.727 (8/8) |
| square2d | 0.515 (8/8) | 0.487 (8/8) | 0.561 (8/8) |
| checkerboard | 0.578 (8/8) | 0.582 (8/8) | 0.671 (8/8) |
| swissroll | 0.633 (8/8) | 0.560 (8/8) | 0.599 (8/8) |
| blobs10d *(control)* | 0.930 (8/8) | 0.915 (8/8) | 0.897 (8/8) |
| gauss10d *(control)* | 0.912 (8/8) | 0.887 (8/8) | 0.887 (8/8) |

**Sticks → bend (N=3 ÷ N=2):**

| dataset | B=32 | B=128 | B=512 |
|---|---|---|---|
| spiral2d | 1.625 (1/8) | 0.888 (6/8) | 0.896 (7/8) |
| moons2d | 1.352 (2/8) | 0.852 (8/8) | 0.926 (8/8) |
| square2d | 0.876 (5/8) | 0.855 (8/8) | 0.880 (8/8) |
| checkerboard | 0.898 (7/8) | 0.979 (6/8) | 0.943 (6/8) |
| swissroll | 1.008 (5/8) | 0.906 (7/8) | 0.890 (8/8) |
| blobs10d *(control)* | 1.358 (2/8) | 1.005 (2/8) | 0.988 (8/8) |
| gauss10d *(control)* | 0.988 (6/8) | 0.988 (7/8) | 0.990 (7/8) |

Both controls behave as they should. Isotropic point clusters and a featureless
10-d Gaussian get ~1% from bend and ~10% from sticks — consistent with the
theoretical rate, where a curve beats a codebook by one dimension's worth, which
is nearly nothing at d=10.

`checkerboard` was added to test one specific prediction: its support is
disconnected, so a curve cannot traverse it without crossing empty squares, and
continuity should cost something. It does — bend buys 0.88–0.98 there against
0.81–0.88 on the connected square of the same intrinsic dimension.

## Why the optimum is N=3, and why that matters

Every smooth manifold is locally straight. At fixed budget, raising N means
fewer, longer pieces, each spanning more curvature; lowering it means more,
shorter pieces, each spanning a stretch that is nearly flat. Bend only pays
while pieces are long relative to the radius of curvature — and at any sensible
budget they are not.

**HiPPO's own experiments use N = 256** (the paper's long-range function
approximation task; elsewhere it "tie[s] the memory size N to the hidden state
dimension d"). Our optimum is N = 3, and N ≥ 12 is worse than k-means on
several datasets. That is not a contradiction — HiPPO compresses one long
ordered signal, where high order is the whole point — but it does say the
polynomial order that makes HiPPO work is two orders of magnitude away from the
order that helps here.

![Fitted curves](https://media.tanh.xyz/seewhy/26-09-09/curve_clusters_fitted.png)

The failure at high order is visible rather than inferred. Bottom row, N=32,
K=4: long excursions and loops through empty space. A few long curves must find
a global ordering of their members, and in one pass they cannot.

## Two things that were nearly wrong

**A train/test distribution leak.** The first version of the data module passed
the same live rng to two builder calls, so `blobs10d` drew *fresh cluster
centres for the test split*. Held-out distortion came back at 1.17 — worse than
predicting the global mean, which is impossible for a working model on
in-distribution data. That impossibility is what caught it. Datasets are now
samplers built once per (name, seed), with both splits drawn from the one
sampler.

**A confound that inverted a conclusion.** The budget sweep appeared to show
the sticks advantage decaying as budget grew (spiral: 0.325 at B=32 → 0.614 at
B=512), which reads as "the idea does not scale". But budget and
points-per-cluster are confounded: at B=512, N=2 there are K=256 clusters
sharing 2000 points — eight points each. Holding the budget fixed and varying
the data instead:

| dataset | budget | M=2000 | M=8000 | M=32000 |
|---|---|---|---|---|
| spiral2d | 512 | 0.701 | 0.609 | 0.548 |
| square2d | 512 | 0.688 | 0.556 | 0.473 |
| swissroll | 128 | 0.644 | 0.540 | 0.478 |

The advantage **grows** with data. The apparent decay was estimator noise, and
the geometric prediction stands. Reported because the wrong reading was the
natural one.

## The description-length correction

The budget `K·N` above counts the model's degrees of freedom: `K·N·D` stored
reals. It does **not** count addresses, and the original version of this report
justified it by claiming a curve has "~N resolvable spots" so `K·N` matches the
address count too. That is wrong. A curve's address `p` is a real number and can
be resolved arbitrarily finely; `N` bounds how many *wiggles* the curve can
have, not how finely it can be indexed. The real address count is `K·G`, and
exp1–3 used `G = max(64, 16N)`.

The full two-part code has three terms, and they are not the same size:

```
L  =  K·N·D · ½log₂M          model      ≈  1,400 bits
   +  M · log₂(addresses)     per point  ≈ 14,000–22,800 bits
   +  M · residual bits       the lossy part being traded away
```

**The address term dominates the model term by 10–16×.** exp1–3 pinned the term
that barely matters and let the dominant one float — upward, in the curve's
favour, since `G` was tied to `N`.

### At matched rate, k-means wins

exp4 sweeps `G` at fixed `N`. Order 1 is the control: a constant curve is the
same at every `p`, so its distortion must be *exactly* flat in `G`. Largest
spread across all (dataset, seed) pairs: **5.2e-18**. The address axis is being
measured correctly.

Distortion at ~7 address bits, model held at `K·N = 128`. Order-1 addresses are
counted as `K`, not `K·G`, since a constant curve's grid points are all the same
position:

| dataset | N=1 (k-means) | N=2 | N=3 | N=6 |
|---|---|---|---|---|
| spiral2d | **0.00238** | 0.00362 | 0.00385 | 0.00492 |
| moons2d | **0.00215** | 0.00244 | 0.00244 | 0.00384 |
| square2d | **0.00908** | 0.00992 | 0.00948 | 0.01081 |
| checkerboard | **0.00508** | 0.00569 | 0.00583 | 0.00692 |
| swissroll | **0.02321** | 0.02596 | 0.02543 | 0.02756 |
| blobs10d *(control)* | 0.00839 | 0.00841 | **0.00831** | 0.00864 |
| gauss10d *(control)* | 0.49893 | 0.49948 | 0.48875 | **0.48681** |

k-means wins on all five low-dimensional sets. The mechanism is visible at
`G=2`: "64 segments sampled twice" is just 128 points constrained to lie in
collinear pairs, and constrained points lose to free points. **The curve's
continuum is only an asset if addressing it is free.**

### But k-means cannot reach the higher rates at all

At ~11 address bits there is no k-means row, because reaching 11 bits requires
K = 2048 centroids — a 16× larger model. That is the asymmetry, and it is what
the total-description-length frontier picks up.

Minimum total bits to reach a distortion target, over all 497 operating points
measured (`K`, `N`, `G` combinations from exp1/exp2/exp4):

| dataset | ≤ 0.02 | ≤ 0.01 | ≤ 0.005 | ≤ 0.002 |
|---|---|---|---|---|
| spiral2d | k-means 1.6KB | k-means 1.6KB | k-means 1.9KB | **curve 2.1KB** (k-means 2.3KB) |
| moons2d | k-means 1.3KB | k-means 1.3KB | k-means 1.6KB | **curve 2.1KB** (k-means 2.3KB) |
| square2d | k-means 1.6KB | k-means 1.9KB | k-means 2.3KB | **curve 3.9KB** — k-means unreachable |
| checkerboard | k-means 1.6KB | k-means 1.6KB | **curve 2.1KB** | k-means 2.9KB |
| swissroll | **curve 2.2KB** | k-means 3.2KB | **curve 4.3KB** — k-means unreachable | — |
| blobs10d *(control)* | k-means 1.4KB | k-means 1.9KB | — | — |
| gauss10d *(control)* | unreachable | — | — | — |

![Rate-distortion frontier](https://media.tanh.xyz/seewhy/26-09-09/curve_clusters_rate_distortion.png)

Black is k-means, tracing its frontier by growing K — every step right costs
model size. Blue is curves, buying rate from grid resolution at no model cost.
k-means dominates the cheap end and then stops; the curve frontier continues to
lower distortion.

The crossover sits around distortion 0.002–0.005 on the low-dimensional sets.
Below it, curves are cheaper or are the only thing that gets there. Above it,
k-means is cheaper. Both controls behave: `blobs10d` prefers k-means throughout,
and nothing reaches even 0.02 on `gauss10d`.

Two caveats on this table. The operating-point grid is coarse and irregular —
only the `(K, N, G)` combinations that exp1/exp2/exp4 happened to run — which is
why `checkerboard` reads non-monotonically across the row; that is grid
coarseness, not an effect. And `½log₂M` bits per coefficient is a convention;
a different coefficient cost moves the crossover.


## The low-order region, and the N=5 dip that wasn't

An earlier render of the order sweep appeared to dip at N=5. It was a
contaminated figure, not data: exp2 sweeps the *budget* at `n_passes=1`, so
before `_agg` filtered on budget as well as passes, rows from five other budgets
were averaged into the same point. The underlying numbers are monotone above
N=3 — spiral2d goes 0.00111, 0.00175, 0.00236, 0.00308 across N=3,4,5,6.

![Low-order region](https://media.tanh.xyz/seewhy/26-09-09/curve_clusters_low_order.png)

Linear axis, ±1 s.e. over 8 seeds. The minimum is at N=3 with no feature at
N=5. `gauss10d` is on its own scale because its distortion is ~0.44 against
~0.005 for the rest, and it *does* wiggle non-monotonically between N=4 and
N=8 — a small separate effect, not the same one.

## Where curves actually beat k-means: high intrinsic dimension

The matched-rate table has one row pointing against all the others. On
`gauss10d`, curves *beat* k-means: N=3 gives 0.980 ± 0.001 and N=6 gives
0.976 ± 0.003, on 4 of 4 seeds, improving with order. Everything low-dimensional
is 1.09–2.06× worse.

`gauss10d` has ambient D = intrinsic d = 10, so exp4 cannot say which mattered.
exp5 separates them: `gauss_D` has intrinsic = ambient, while `spiral_D` carries
a 1-D spiral into ambient D by random rotation.

![Dimension flip](https://media.tanh.xyz/seewhy/26-09-09/curve_clusters_dimension_flip.png)

Ratio to k-means at matched rate, paired per seed:

| ambient D | gauss, N=3 | spiral, N=3 |
|---|---|---|
| 2 | 1.095 (1/6) | 1.545 (0/6) |
| 5 | 1.016 (1/6) | 1.308 (0/6) |
| 10 | 1.003 (2/6) | 1.092 (0/6) |
| 20 | **0.989 (6/6)** | 1.041 (0/6) |
| 40 | **0.982 (6/6)** | 1.000 (3/6) |

**The flip tracks intrinsic dimension, not ambient.** `gauss` crosses below 1.0
by D=20 and holds there on 6 of 6 seeds. `spiral` carries intrinsic d=1 all the
way to ambient D=40 and converges to 1.000 from above without ever crossing.

That refutes the obvious mechanism. I expected the continuity constraint to cost
one direction out of D and so become cheap as ambient D grows — which would have
made `spiral_40d` flip too. It doesn't. What actually happens is about how far
each method is from resolving the data: where 128 representation points are
hopelessly few, the only thing left to win on is cell *shape*, and elongated
cells tile high-dimensional space marginally better than round ones.

**The win is real and almost worthless.** Here is the context the ratio hides:

| gauss, matched rate | D=2 | D=10 | D=20 | D=40 |
|---|---|---|---|---|
| k-means (N=1) | 0.0227 | 0.4960 | 0.7423 | 0.9395 |
| curves (N=3) | 0.0249 | 0.4975 | **0.7342** | **0.9229** |

Curves win at D=40 — where 92% of the variance is unexplained and neither
method represents the data at all. Where the problem is actually solvable
(`spiral`, distortion 0.4–4%), curves tie at best.

And the two rate framings trend in **opposite** directions with dimension. With
addresses free, the curve advantage is largest in low dimension (0.45–0.55× at
D=2) and decays to ~0.95 by D=40. With addresses charged, it is negative in low
dimension and turns slightly positive at high intrinsic dimension. So "are
curves better in high dimension?" has no answer until the rate framing is fixed
— and under the framing that favours curves overall, high dimension is where
they help *least*.


## Real data, and why the high-dimensional result above does not count

The `gauss_40d` win is not evidence. An isotropic Gaussian in 40-D has no
structure to exploit — k-means scores 0.9395 there and curves 0.9229, which is
two methods failing and one failing marginally less. And the other synthetic
family went to the opposite extreme, intrinsic d=1. Neither resembles real data,
which is high ambient dimension with *low* intrinsic dimension.

exp6 fixes the design: ambient dimension **pinned at 40**, intrinsic dimension
swept from 1 to 40 by drawing a d-dimensional Gaussian in a random d-subspace.
Plus MNIST and Fashion-MNIST, raw pixels and PCA-32.

![Intrinsic dimension](https://media.tanh.xyz/seewhy/26-09-09/curve_clusters_intrinsic_dim.png)

Ratio to k-means, N=3, paired per seed, ambient D=40 throughout:

| intrinsic d | k-means abs. distortion | rate charged | rate free |
|---|---|---|---|
| 1 | 0.0012 | 1.138 (3/6) | **0.303 (6/6)** |
| 2 | 0.0216 | 1.067 (2/6) | **0.507 (6/6)** |
| 4 | 0.1500 | 1.034 (0/6) | **0.706 (6/6)** |
| 8 | 0.4041 | 1.006 (1/6) | 0.855 (6/6) |
| 16 | 0.6772 | 0.996 (5/6) | 0.925 (6/6) |
| 32 | 0.8904 | 0.981 (6/6) | 0.949 (6/6) |
| 40 | 0.9366 | 0.988 (5/6) | 0.956 (6/6) |

**The advantage is governed entirely by intrinsic dimension; ambient dimension
is irrelevant.** With addresses free it decays monotonically from 3.3× better at
d=1 to 5% by d=32. With addresses charged there is nothing anywhere — the
apparent win at d≥32 is the same artefact as `gauss_40d`, appearing exactly
where absolute distortion passes 0.89.

### Real data sits at the dead end of that curve

| dataset | k-means abs. | rate charged, N=3 | rate free, N=3 |
|---|---|---|---|
| mnist_raw (784-D) | 0.5916 | 1.003 (2/3) | 0.971 (3/3) |
| mnist_pca32 | 0.4269 | 1.014 (0/3) | 0.951 (3/3) |
| fashion_mnist_raw | 0.3460 | 1.015 (1/3) | 0.957 (3/3) |
| fashion_mnist_pca32 | 0.1775 | 1.042 (0/3) | **0.930 (3/3)** |

Under the framing most favourable to curves, real image data gets **3–7%**.
Under matched rate it gets **nothing, or slightly worse**. Every dotted line in
the figure lands on the flat right-hand end of the synthetic curve.

One internal consistency check worth noting: the four real datasets order by
curve-gain almost exactly as they order by compressibility (`fashion_pca32`,
the most compressible at 0.1775, gets the largest gain at 0.930;
`mnist_raw`, the least at 0.5916, gets the smallest at 0.971). That is what the
intrinsic-dimension story predicts, and it is not something the design forced.

**So the answer to the question this project started from — can a dataset be
compressed into one of these — is no for real image data.** The method is for
genuinely low-intrinsic-dimensional data, d ≤ 4, and nothing else.


## Recommendation

Worth pursuing, but only with the claim stated precisely:

1. **At matched rate, curves lose to k-means.** Do not claim otherwise. The
   30–70% figure in the model-only accounting is bought with address bits.
2. **The real claim is regime-specific**: because k-means's model grows as 2^R
   in the rate and a curve's does not, curves win on total description length
   at tight distortion targets, and are eventually the only option. That is a
   defensible and falsifiable claim, and it needs a proper rate-distortion
   sweep — a dense grid of `(K, N, G)` — rather than the coarse one here.
3. **Only N=1 versus N>1 matters.** With K free, orders 2 through 8 are
   indistinguishable at equal cost, and K is the lever that actually moves
   distortion. The earlier "optimum is N=3" was an artefact of pinning K·N.
   HiPPO's own function-approximation experiment uses N=256, which this says
   nothing supports here.
4. **The usable range is intrinsic dimension d ≤ 4.** With addresses free the
   advantage runs 3.3× at d=1, 2.0× at d=2, 1.4× at d=4, and is gone by d=32.
   Ambient dimension does not matter. Real image data (MNIST, Fashion-MNIST,
   raw and PCA-32) gets 3–7% with addresses free and nothing at matched rate.
   Do not pitch this as a high-dimensional or an image method.
5. **The single-pass algorithm is the part that might be novel** — exact 1-D
   search for the assignment, closed-form minimum-norm write, clusters born as
   points that grow their own order. Prior art to clear first: principal-curve
   clustering, local principal curves, and the 1-D self-organising map, which
   this algorithm closely resembles.

What it should not be sold as: a use for high-order polynomial memory, or a
general improvement on k-means.

## Reproducing

```
projects/curve-clusters/
  lib/curves.py       CurveClusters, Legendre basis
  lib/toy.py          seven toy datasets spanning intrinsic dimension
  lib/figures.py      the figures above
  experiments1.py     order sweep at fixed budget  (--selftest for the N=1 proof)
  experiments2.py     budget sweep
  experiments3.py     the starvation control
  experiments4.py     the grid sweep — the address-bit accounting
  experiments5.py     dimension sweep — ambient vs intrinsic
  experiments6.py     intrinsic-d sweep at pinned ambient D, plus real data
  experiments7.py     K and N independent — the plain version
  results.jsonl       6572 rows

The plain table above is the one to read first; the framing-dependent
comparisons after it explain where its few percent come from.
```

Runs in minutes on CPU. 8 seeds everywhere except exp3 (4).
