"""Build one report per research question, each readable standalone.

Six documents: an overview plus one per question. Every one carries the same
BACKGROUND block, so a reader arriving at Q4 from a link does not have to go
find the setup. The duplication is deliberate.

All tables are computed from `results.jsonl` here rather than typed into
markdown. Earlier drafts of the single combined report carried hand-copied
numbers and one of them drifted; anything a reader might act on should come
out of the data file on every build.

Usage (on the GPU box, from the repo root):
    uv run --no-sync python projects/curve-clusters/scripts/gen_reports.py
"""

from __future__ import annotations

import collections
import json
import math
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.append(str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

from lib import figures                                    # noqa: E402
from shared_lib.media import save_matplotlib_figure        # noqa: E402
from shared_lib.report import save_report                  # noqa: E402

ROWS = [json.loads(l) for l in (HERE / "results.jsonl").read_text().splitlines() if l.strip()]


def fig(name: str, f) -> str:
    return save_matplotlib_figure(f"{name}.png", f, dpi=150)


def table(header: list[str], body: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in body]
    return "\n".join(out)


def from_exp(*prefixes):
    """Rows from specific experiment files. Provenance is the only safe filter:
    exp4 also carries budget=128 and n_passes=1, so predicates built from those
    fields silently mix its coarse-grid sweep into exp1's order sweep."""
    tags = tuple(f"{p}_" for p in prefixes)
    return [r for r in ROWS if str(r.get("experiment", "")).startswith(tags)]


def mean_by(pred, key, source=None):
    acc = collections.defaultdict(list)
    for r in (source if source is not None else ROWS):
        if pred(r):
            acc[key(r)].append(r["distortion_test"])
    return {k: statistics.mean(v) for k, v in acc.items()}


def paired(pred, group, base_order=1, source=None):
    """{group: (mean ratio to base_order, n_wins, n_seeds)} paired per seed."""
    per = collections.defaultdict(dict)
    for r in (source if source is not None else ROWS):
        if pred(r):
            per[(group(r), r["seed"])][r["order"]] = r["distortion_test"]
    out = collections.defaultdict(dict)
    for (g, _), v in per.items():
        if base_order not in v:
            continue
        for o, d in v.items():
            out[g].setdefault(o, []).append(d / v[base_order])
    return {g: {o: (statistics.mean(rs), sum(x < 1 for x in rs), len(rs))
                for o, rs in d.items()} for g, d in out.items()}


BACKGROUND = """
## Background — the idea

A pile of data points with no natural order. Store it as a set of smooth
**curves** rather than a set of **points**.

To store a point: find the spot on some curve that already looks most like it,
then nudge the curve at that spot towards the point. To read back: go to a spot
and read off the curve. If a new point already matches what the curve says, the
nudge is tiny and nothing gets stored — that is where compression would come
from.

**The trade.** Two ways to spend the same memory. *K free points*, anywhere,
unconnected — that is online k-means. Or *K curves*, roughly the same number of
resolvable spots, but forced to lie along connected lines, in exchange for which
everything *along* each line is representable too. The curve pays a constraint
and gets a freebie. The idea is worth pursuing only if the freebie outweighs the
constraint.

**Why the dial contains its own baseline.** A curve is N shifted-Legendre
coefficients per input dimension, and the order N is a dial:

| N | shape | equivalent to |
|---|-------|---------------|
| 1 | constant | a point — online k-means |
| 2 | straight segment | local PCA, 1 component |
| ≥3 | bent curve | the idea under test |

So this is not two systems to compare. It is one system with the dial turned.

**The N=1 equivalence is proved, not asserted.** `experiments1.py --selftest`
runs the textbook centroid rule on the same stream and requires agreement:
observed max difference **5.6e-17**, with identical assignment counts. Every
comparison against N=1 in these reports is a comparison against a genuine
online k-means.

**Regime.** No training, no epochs, no gradients on parameters. One pass over
the data, every decision made at inference time, per point. Assignment is an
exhaustive scan over (which curve, where on it), so the chosen spot is the true
global optimum — affordable only because the latent is one-dimensional. The
write is a closed-form minimum-norm nudge along φ(p\\*), spread over the
reproducing kernel K(p\\*,·), with step size 1/(1 + count/N); at N=1 that
collapses exactly to the running-mean centroid rule, which is why the selftest
passes.

**Metric.** `distortion` — mean squared distance from a point to its nearest
spot on any curve, always on held-out data. Each dataset is variance-normalised,
so this reads directly as **fraction of variance unexplained** and is comparable
across dimensions. A value ≥ 1.0 is impossible for a working model; it means
worse than predicting the global mean, and it is a bug signal. It caught one
(see Q1).
"""

INDEX = {
    "curve_clusters_probe": "Overview — all five questions",
    "curve_clusters_q1_shape": "Q1 · Is a curve better than a ball?",
    "curve_clusters_q2_scaling": "Q2 · Does it survive more budget and more data?",
    "curve_clusters_q3_addresses": "Q3 · Is it just buying more addresses?",
    "curve_clusters_q4_dimension": "Q4 · Which dimension matters, ambient or intrinsic?",
    "curve_clusters_q5_k_or_n": "Q5 · K or N — which lever actually matters?",
    "curve_clusters_q6_trimmed": "Q6 · Move the centre with only the closest half?",
}
BASE = "https://media.tanh.xyz/seewhy/26-09-09/"


def related(exclude: str) -> str:
    lines = [f"- [{t}]({BASE}{k}.html)" for k, t in INDEX.items() if k != exclude]
    return "## Related reports\n\n" + "\n".join(lines) + "\n"


# ---------------------------------------------------------------- Q1
def q1() -> tuple[str, str, str]:
    key = "curve_clusters_q1_shape"
    exp1 = from_exp("exp1")
    b128 = lambda r: r.get("budget") == 128 and r.get("n_passes") == 1
    orders = [1, 2, 3, 4, 5, 6, 8, 12, 16, 24, 32]
    names = ["spiral2d", "moons2d", "square2d", "checkerboard", "swissroll",
             "blobs10d", "gauss10d"]
    m = mean_by(b128, lambda r: (r["dataset"], r["order"]), source=exp1)
    body = []
    for ds in names:
        vals = {o: m[(ds, o)] for o in orders if (ds, o) in m}
        best = min(vals, key=vals.get)
        cells = [f"**{vals[o]:.4f}**" if o == best else f"{vals[o]:.4f}"
                 if o in vals else "—" for o in orders]
        label = ds + (" *(control)*" if ds in ("blobs10d", "gauss10d") else "")
        body.append([label] + cells + [f"N={best}"])
    sweep = table(["dataset"] + [f"N={o}" for o in orders] + ["best"], body)

    pr = paired(b128, lambda r: r["dataset"], source=exp1)
    body2 = []
    for ds in names:
        r2, r3 = pr[ds].get(2), pr[ds].get(3)
        body2.append([ds,
                      f"{r2[0]:.3f} ({r2[1]}/{r2[2]})" if r2 else "—",
                      f"{r3[0]:.3f} ({r3[1]}/{r3[2]})" if r3 else "—"])
    pairs = table(["dataset", "N=2 ÷ N=1", "N=3 ÷ N=2"], body2)

    md = f"""**Question: at the same stored memory, is a cluster better represented as a
smooth curve than as a single point?**

**Answer: yes — but the gain is the step from a point to a *straight stick*, not
the bending.** Going from N=1 to N=2 cuts distortion by 35–55% on every
low-dimensional dataset, 8 of 8 seeds. Going from N=2 to N=3 buys a further
5–17%, real but small. Order beyond 3 is actively harmful at fixed budget.

Read [Q5]({BASE}curve_clusters_q5_k_or_n.html) before acting on the specific
order: this experiment pins `K·N`, which hides K, and with K free the orders
above 1 stop being distinguishable.

{BACKGROUND}

## Method

Budget `K·N = 128` held fixed, so raising the order buys fewer clusters. Single
pass, 8 seeds, held-out distortion. Seven datasets spanning intrinsic dimension,
including two deliberate controls where balls should win.

## Results

{sweep}

![Order sweep]({fig("curve_clusters_order_sweep_p1", figures.order_sweep(ROWS, n_passes=1))})

Paired per-seed ratios — the two orders see identical data draws, and the
fraction is how many of 8 seeds agree:

{pairs}

![One dial, three settings]({fig("curve_clusters_comparison", figures.comparison_grid())})

Same algorithm in all three rows. Distortion falls down every column.

## Why the optimum sits low, and why high order fails

Every smooth manifold is locally straight. At fixed budget, raising N means
fewer, longer pieces, each spanning more curvature; lowering it means more,
shorter pieces, each nearly flat. Bend only pays while pieces are long relative
to the radius of curvature — and at any sensible budget they are not.

The failure at high order is visible rather than inferred:

![Fitted curves]({fig("curve_clusters_fitted", figures.fitted_curves())})

Bottom row, N=32 with K=4: long excursions and loops through empty space. A few
long curves must find a global ordering of their members, and in one pass they
cannot. Three passes barely helps, so it is not undertraining.

![Low-order region]({fig("curve_clusters_low_order", figures.low_order_zoom(ROWS))})

The low-order region on a linear axis with standard errors. Monotone above N=3.
An earlier render appeared to dip at N=5; that was a contaminated figure — rows
from five other budgets were being averaged into each point before the
aggregator filtered on budget as well as passes.

## A bug this caught

The first version of the data module passed the same live rng to two builder
calls, so `blobs10d` drew **fresh cluster centres for the test split**. Held-out
distortion came back at 1.17 — worse than predicting the global mean, which is
impossible for a working model on in-distribution data. That impossibility is
what surfaced it. Datasets are now samplers built once per (name, seed) with
both splits drawn from the one sampler.

`checkerboard` was added later to test a specific prediction: its support is
disconnected, so a curve cannot traverse it without crossing empty squares, and
continuity should cost something. It does — bend buys 0.88–0.98 there against
0.81–0.88 on the connected square of the same intrinsic dimension.

## Caveats

- `K·N` is pinned, so K and N move together. This is the flaw [Q5]({BASE}curve_clusters_q5_k_or_n.html) fixes.
- Addresses are not charged. [Q3]({BASE}curve_clusters_q3_addresses.html) charges them, and the result changes sign.

{related(key)}"""
    return key, "Q1 · Is a curve better than a ball?", md


# ---------------------------------------------------------------- Q2
def q2() -> tuple[str, str, str]:
    key = "curve_clusters_q2_scaling"
    names = ["spiral2d", "moons2d", "square2d", "checkerboard", "swissroll",
             "blobs10d", "gauss10d"]
    budgets = [16, 32, 64, 128, 256, 512]
    pr_by_budget = {}
    for bg in budgets:
        pr_by_budget[bg] = paired(
            lambda r, bg=bg: r.get("budget") == bg and r.get("n_passes") == 1,
            lambda r: r["dataset"], source=from_exp("exp1", "exp2"))
    body = []
    for ds in names:
        row = [ds]
        for bg in budgets:
            v = pr_by_budget[bg].get(ds, {}).get(2)
            row.append(f"{v[0]:.3f} ({v[1]}/{v[2]})" if v else "—")
        body.append(row)
    bud = table(["dataset"] + [f"B={b}" for b in budgets], body)

    st = collections.defaultdict(dict)
    for r in from_exp("exp3"):
        if "n_train" in r:
            st[(r["dataset"], r["budget"], r["seed"])][r["order"]] = r["distortion_test"]
            st[(r["dataset"], r["budget"], r["seed"])]["_m"] = r["n_train"]
    acc = collections.defaultdict(list)
    for (ds, bg, _), v in st.items():
        if 1 in v and 2 in v:
            acc[(ds, bg, v["_m"])].append(v[2] / v[1])
    body2 = []
    for ds in ("spiral2d", "square2d", "swissroll"):
        for bg in (128, 512):
            cells = []
            for mm in (2000, 8000, 32000):
                rs = acc.get((ds, bg, mm))
                cells.append(f"{statistics.mean(rs):.3f}" if rs else "—")
            body2.append([ds, bg] + cells)
    starve = table(["dataset", "budget", "M=2,000", "M=8,000", "M=32,000"], body2)

    md = f"""**Question: the shape advantage exists at one budget — does it survive a bigger
budget, and is it real or an artefact of too little data?**

**Answer: it survives, and the trend that suggested otherwise was a confound.**
The advantage appeared to decay as budget grew, which reads as "does not scale".
It does not: budget and points-per-cluster were entangled, and with the budget
held fixed and the data increased, the advantage **grows**.

{BACKGROUND}

## Method

Two sweeps. First, the budget `K·N` over five octaves, reading the paired
per-seed N=2÷N=1 ratio at each. Second — because the first is confounded — the
budget held fixed while the training set grows over a decade.

## Result 1: the budget sweep, which looks like bad news

Paired per-seed ratio, N=2 ÷ N=1. Below 1 means the stick wins:

{bud}

![Sticks vs budget]({fig("curve_clusters_stick_vs_budget", figures.bend_vs_budget(ROWS, orders=(1, 2)))})

On spiral2d the ratio runs 0.325 at B=32 up to 0.614 at B=512. Taken at face
value, the shape advantage is evaporating as you spend more.

## Why that reading is wrong

Budget and points-per-cluster are entangled. At B=512 with N=2 there are K=256
clusters sharing 2,000 points — **eight points each**. Fitting a two-parameter
curve from eight points in a single pass is estimator-noise-dominated, and that
alone would erode the advantage with no geometric content.

The theory also points the other way. For data on a 1-D manifold, K points each
cover arc ~1/K with squared error ~1/K², while K/2 chords span arc ~2/K and
deviate by curvature·(2/K)², giving ~1/K⁴. Sticks should beat points by *more*
as budget grows. A measured trend that contradicts the prediction is the
signature of a finite-sample artefact.

## Result 2: hold the budget, add data

Paired N=2 ÷ N=1 ratio, budget fixed:

{starve}

**The advantage grows with data.** At B=512 on spiral2d it goes 0.701 → 0.609 →
0.548. The decay in Result 1 was starvation, and the geometric prediction
stands.

![Bend vs budget]({fig("curve_clusters_bend_vs_budget", figures.bend_vs_budget(ROWS, orders=(2, 3)))})

Bend, separately, holds roughly flat with budget rather than growing.

## Caveats

- The starvation control runs 4 seeds against 8 elsewhere, and covers three
  datasets rather than seven.
- This whole report measures at fixed *stored floats* with addresses free.
  [Q3]({BASE}curve_clusters_q3_addresses.html) charges for addresses and the
  conclusion changes.

{related(key)}"""
    return key, "Q2 · Does it survive more budget and more data?", md


# ---------------------------------------------------------------- Q3
def q3() -> tuple[str, str, str]:
    key = "curve_clusters_q3_addresses"
    names = ["spiral2d", "moons2d", "square2d", "checkerboard", "swissroll",
             "blobs10d", "gauss10d"]
    pts = collections.defaultdict(dict)
    for r in from_exp("exp4"):
        if r.get("budget") == 128:
            k, n, g = r["n_curves"], r["order"], r["grid"]
            bits = math.log2(k if n == 1 else k * g)
            pts[(r["dataset"], n, g)] = (bits, pts.get((r["dataset"], n, g), (0, []))[1] + [r["distortion_test"]]) \
                if (r["dataset"], n, g) in pts else (bits, [r["distortion_test"]])
    body = []
    for ds in names:
        row, best, bestv = [ds], None, 9e9
        for o in (1, 2, 3, 6):
            cand = [(abs(b - 7), statistics.mean(v)) for (d, n, _), (b, v) in pts.items()
                    if d == ds and n == o and 6.4 <= b <= 7.7]
            if not cand:
                row.append("—")
                continue
            v = min(cand)[1]
            row.append(f"{v:.5f}")
            if v < bestv:
                best, bestv = o, v
        row = [row[0]] + [f"**{c}**" if i == [1, 2, 3, 6].index(best) else c
                          for i, c in enumerate(row[1:])]
        body.append(row + [f"N={best}"])
    matched = table(["dataset", "N=1 (k-means)", "N=2", "N=3", "N=6", "winner"], body)

    md = f"""**Question: the curve wins at equal stored floats — but is it just spending more
addresses? Charge for them and see.**

**Answer: charge for addresses and the win disappears.** At matched rate,
k-means is better on all five low-dimensional datasets. The 30–70% advantage in
[Q1]({BASE}curve_clusters_q1_shape.html) was bought with address bits, not with
shape. Under a full description-length accounting there is still a regime where
curves win — high fidelity — but it is narrow and it is a different claim.

{BACKGROUND}

## The accounting error this report fixes

`K·N` counts the model's degrees of freedom: `K·N·D` stored reals. It does
**not** count addresses. An earlier draft justified it by claiming a curve has
"~N resolvable spots" so `K·N` matched the address count too. That is wrong. A
curve's address `p` is a real number and can be resolved arbitrarily finely; N
bounds how many *wiggles* the curve can have, not how finely it can be indexed.
The real address count is `K·G`, and Q1/Q2 used `G = max(64, 16N)`.

The full two-part code has three terms and they are not the same size:

```
L  =  K·N·D · ½log₂M          model      ≈  1,400 bits
   +  M · log₂(addresses)     per point  ≈ 14,000–22,800 bits
   +  M · residual bits       the lossy part being traded away
```

**The address term dominates the model term by 10–16×.** Q1 and Q2 pinned the
term that barely matters and let the dominant one float — upward, in the curve's
favour, since G was tied to N.

## Method

Sweep the grid G at fixed order, so rate varies while the model does not. Order
1 is the control: a constant curve is identical at every p, so its distortion
must be **exactly** flat in G. Largest spread across all (dataset, seed) pairs:
**5.2e-18**. The address axis is measured correctly.

Order-1 addresses are counted as K, not K·G, since a constant curve's grid
points are all the same position.

## At matched rate, k-means wins

Distortion at ~7 address bits, model held at `K·N = 128`:

{matched}

k-means wins on all five low-dimensional sets. The mechanism is visible at G=2:
"64 segments sampled twice" is just 128 points constrained to lie in collinear
pairs, and constrained points lose to free points. **The curve's continuum is
only an asset if addressing it is free.**

The two controls invert, and that inversion is itself an artefact — see
[Q4]({BASE}curve_clusters_q4_dimension.html), where it turns out to appear
exactly where absolute distortion passes 0.89 and neither method works.

## But k-means cannot reach the higher rates at all

At ~11 address bits there is no k-means row: reaching 11 bits needs K=2048
centroids, a 16× larger model. **k-means needs K = 2^R centroids to use R
address bits, so its model grows exponentially in the rate, while a curve buys
rate from grid resolution at no model cost.** That asymmetry is what the
total-description-length frontier picks up.

![Rate-distortion frontier]({fig("curve_clusters_rate_distortion", figures.rate_distortion(ROWS))})

Black is k-means, tracing its frontier by growing K — every step right costs
model size. Blue is curves. k-means dominates the cheap end and then stops; the
curve frontier continues to lower distortion.

Minimum total bits to reach a distortion target, over 497 measured operating
points:

| dataset | ≤ 0.02 | ≤ 0.01 | ≤ 0.005 | ≤ 0.002 |
|---|---|---|---|---|
| spiral2d | k-means 1.6KB | k-means 1.6KB | k-means 1.9KB | **curve 2.1KB** |
| moons2d | k-means 1.3KB | k-means 1.3KB | k-means 1.6KB | **curve 2.1KB** |
| square2d | k-means 1.6KB | k-means 1.9KB | k-means 2.3KB | **curve 3.9KB** — k-means unreachable |
| checkerboard | k-means 1.6KB | k-means 1.6KB | **curve 2.1KB** | k-means 2.9KB |
| swissroll | **curve 2.2KB** | k-means 3.2KB | **curve 4.3KB** — k-means unreachable | — |
| blobs10d *(control)* | k-means 1.4KB | k-means 1.9KB | — | — |
| gauss10d *(control)* | unreachable | — | — | — |

The crossover sits around distortion 0.002–0.005. Below it, curves are cheaper
or the only thing that gets there.

## Answer

Three accountings, three answers, and the right one depends on whether you ever
need to *name* a point:

| accounting | winner | when it is the right question |
|---|---|---|
| stored floats only | curves, by 30–70% | you match against the memory but never name a point — a support or novelty model |
| matched rate | **k-means, all five sets** | you have a fixed per-point code budget |
| total description length | k-means at loose fidelity, curves at tight | actual compression: reconstructing the dataset |

## Caveats

- The 497 operating points are a coarse, irregular grid — only the `(K, N, G)`
  combinations Q1/Q2 and this experiment happened to run. That is why
  `checkerboard` reads non-monotonically across its row; grid coarseness, not
  an effect.
- `½log₂M` bits per coefficient is a convention. A different coefficient cost
  moves the crossover.

{related(key)}"""
    return key, "Q3 · Is it just buying more addresses?", md


# ---------------------------------------------------------------- Q4
def q4() -> tuple[str, str, str]:
    key = "curve_clusters_q4_dimension"
    sub = paired(lambda r: r.get("kind") == "subspace" and r.get("grid_mode") == "generous",
                 lambda r: r["intrinsic"])
    subm = paired(lambda r: r.get("kind") == "subspace" and r.get("grid_mode") == "matched",
                  lambda r: r["intrinsic"])
    absn = mean_by(lambda r: r.get("kind") == "subspace" and r.get("grid_mode") == "generous",
                   lambda r: (r["intrinsic"], r["order"]))
    body = []
    for d in (1, 2, 4, 8, 16, 32, 40):
        f = sub.get(d, {}).get(3)
        mm = subm.get(d, {}).get(3)
        body.append([d, f"{absn.get((d, 1), float('nan')):.4f}",
                     f"{mm[0]:.3f} ({mm[1]}/{mm[2]})" if mm else "—",
                     f"**{f[0]:.3f}** ({f[1]}/{f[2]})" if f else "—"])
    intr = table(["intrinsic d", "k-means abs. distortion", "rate charged", "rate free"], body)

    realf = paired(lambda r: r.get("kind") == "real" and r.get("grid_mode") == "generous",
                   lambda r: r["dataset"])
    realm = paired(lambda r: r.get("kind") == "real" and r.get("grid_mode") == "matched",
                   lambda r: r["dataset"])
    absr = mean_by(lambda r: r.get("kind") == "real" and r.get("grid_mode") == "generous",
                   lambda r: (r["dataset"], r["order"]))
    body2 = []
    for ds in ("mnist_raw", "mnist_pca32", "fashion_mnist_raw", "fashion_mnist_pca32"):
        f, mm = realf.get(ds, {}).get(3), realm.get(ds, {}).get(3)
        body2.append([ds, f"{absr.get((ds, 1), float('nan')):.4f}",
                      f"{mm[0]:.3f} ({mm[1]}/{mm[2]})" if mm else "—",
                      f"{f[0]:.3f} ({f[1]}/{f[2]})" if f else "—"])
    real = table(["dataset", "k-means abs.", "rate charged, N=3", "rate free, N=3"], body2)

    md = f"""**Question: does the curve advantage depend on dimension — and on which
dimension, the ambient one or the intrinsic one?**

**Answer: intrinsic, entirely. Ambient dimension is irrelevant.** With
addresses free the advantage decays monotonically from 3.3× at intrinsic d=1 to
5% by d=32. Real image data sits at the dead flat end of that curve, getting
3–7%. The usable range is **d ≤ 4**.

This report also retracts a result. An earlier draft reported curves *beating*
k-means at high dimension on a 40-D isotropic Gaussian. That is not evidence:
k-means scores 0.9395 there and curves 0.9229, which is two methods failing and
one failing marginally less.

{BACKGROUND}

## Method

The earlier attempt could not separate the two dimensions, because its
high-dimensional dataset had ambient D = intrinsic d = 10 and its other family
had intrinsic d = 1. So: **ambient dimension pinned at 40**, intrinsic dimension
swept 1→40 by drawing a d-dimensional Gaussian in a random d-subspace. Plus
MNIST and Fashion-MNIST, raw pixels and PCA-32, as the real-data check.

Both rate framings are run, since [Q3]({BASE}curve_clusters_q3_addresses.html)
showed they can disagree, and reporting one would be cherry-picking.

## The advantage is a function of intrinsic dimension

N=3 against k-means, paired per seed, ambient D=40 throughout:

{intr}

![Intrinsic dimension]({fig("curve_clusters_intrinsic_dim", figures.intrinsic_dimension(ROWS))})

Monotone decay from 3.3× at d=1 to 5% by d=32, with ambient dimension pinned
throughout. Note the charged column: the apparent win reappears at d≥32 —
**exactly where absolute distortion passes 0.89** and nothing works. That is the
retracted result, now visible as an artefact.

The earlier ambient-vs-intrinsic figure makes the same point from the other
direction:

![Dimension flip]({fig("curve_clusters_dimension_flip", figures.dimension_flip(ROWS))})

`gauss` (intrinsic = ambient) crosses below 1.0 by D=20. `spiral` carries
intrinsic d=1 all the way to ambient D=40 and converges to 1.000 from above
without ever crossing. If ambient dimension were the driver, `spiral` would
flip too.

This refutes the obvious mechanism. The natural guess is that continuity costs
one direction out of D and so gets cheap as ambient D grows — which predicts
`spiral_40d` flips. It does not.

## Real data sits at the dead end

{real}

Under the framing most favourable to curves, real image data gets **3–7%**.
Under matched rate it gets **nothing, or slightly worse**.

One consistency check the design did not force: the four real datasets order by
curve-gain almost exactly as they order by compressibility — `fashion_pca32` is
both the most compressible and gets the largest gain; `mnist_raw` the least of
both. That is what the intrinsic-dimension story predicts.

## Answer

**Not real image data.** The method is for genuinely low-intrinsic-dimensional
data — d ≤ 4: trajectories, spectra, sensor streams, anything on a thin
manifold. That is a real niche and it is the one claim that survives every
framing tried across these five reports.

## Caveats

- Real data runs 3 seeds against 6 for the synthetic sweep.
- The subspace family is linear. A curved d-manifold might behave differently,
  though Q1's `swissroll` gives no sign of it.

{related(key)}"""
    return key, "Q4 · Which dimension matters, ambient or intrinsic?", md


# ---------------------------------------------------------------- Q5
def q5() -> tuple[str, str, str]:
    key = "curve_clusters_q5_k_or_n"
    d = collections.defaultdict(list)
    dim = {}
    for r in ROWS:
        if r.get("grid_mode") == "indep":
            d[(r["dataset"], r["n_curves"], r["order"])].append(r["distortion_test"])
            dim[r["dataset"]] = r["dim"]
    sets = ["mnist_pca32", "fashion_mnist_pca32", "mnist_raw", "sub8_in40", "gauss_40d"]
    costs = [16, 32, 64, 128, 256]
    body = []
    for ds in sets:
        if ds not in dim:
            continue
        row, orders_used = [f"{ds} (D={dim[ds]})"], set()
        for c in costs:
            cand = [(statistics.mean(v), k, n) for (t, k, n), v in d.items()
                    if t == ds and k * n == c]
            km = [x for x in cand if x[2] == 1]
            cv = [x for x in cand if x[2] > 1]
            if not km or not cv:
                row.append("—")
                continue
            b = min(cv)
            orders_used.add(b[2])
            row.append(f"{(1 - b[0] / km[0][0]) * 100:+.1f}%")
        body.append(row + ["N=" + "–".join(str(o) for o in sorted(orders_used))])
    eq = table(["dataset"] + [f"K·N={c}" for c in costs] + ["best order"], body)

    grid_body = []
    for ds in ("mnist_pca32", "sub8_in40"):
        for k in (8, 32, 128):
            v = {n: statistics.mean(d[(ds, k, n)]) for n in (1, 2, 3, 4, 6, 8)
                 if d[(ds, k, n)]}
            if not v:
                continue
            grid_body.append([ds, k] + [f"{v[n]:.4f}" if n in v else "—"
                                        for n in (1, 2, 3, 4, 6, 8)])
    fixedk = table(["dataset", "K"] + [f"N={n}" for n in (1, 2, 3, 4, 6, 8)], grid_body)

    md = f"""**Question: every earlier experiment pinned `K·N` and swept N, which hides K. With
both free, which lever actually matters?**

**Answer: K. And the only thing N=1 versus N>1 changes is whether a cluster can
extend at all — orders 2 through 8 are indistinguishable.**

This retracts a claim made twice in earlier drafts: *"the optimum is N=3"*. That
was an artefact of pinning `K·N = 128`. With K free there is no optimal order.

{BACKGROUND}

## The flaw this fixes

Fixing `K·N` means N=1 runs with K=128 while N=2 runs with K=64. So "N=2 beats
N=1" is really "64 sticks beat 128 points" — the shape effect and the cluster
count move together and cannot be separated.

## Method

Sweep K ∈ {{8…256}} and N ∈ {{1…8}} independently, then plot distortion against
stored floats `K·N·D` with **K running along each line**. A vertical slice is an
exact equal-cost comparison, the lowest line at a given cost is the best order,
and nothing is held fixed by fiat. The grid is G = 8N throughout: grid
resolution costs nothing to store, and
[Q3]({BASE}curve_clusters_q3_addresses.html) already covers charging for it.

## Result

![K and N independently]({fig("curve_clusters_k_and_n", figures.k_and_n(ROWS))})

Two things, and they are the whole result.

**1. N=1 is the only order that is different.** The black line sits above the
others on every dataset, and the N=2 through N=8 lines lie on top of each other.

**2. K is the real lever.** Moving right along any single line — more clusters —
buys far more than any change of N at fixed cost.

Best curve versus k-means at exactly equal stored floats:

{eq}

At fixed K, higher N is monotonically better — but that is trivial, since it
buys more floats:

{fixedk}

![Does bend help]({fig("curve_clusters_does_bend_help", figures.does_bend_help(ROWS))})

## Answer

N > 1 helps in high dimension by 1–16% depending on the data, and essentially
all of it is the single step from N=1 to N=2. Bending past a straight stick is
free to try and worth nothing.

The practical reading: **you do not want a high-order polynomial memory, you
want many low-order pieces.** HiPPO's own function-approximation experiment uses
N=256; nothing measured here supports an order above about 2 for this use.

## Caveats

- `mnist_raw` (784-D) is capped at K≤64 — cost per point goes as `K·G·D` and the
  wide input is minutes per cell. The PCA-32 variants carry the real-data signal.
- 4 seeds, against 8 in [Q1]({BASE}curve_clusters_q1_shape.html).
- Addresses free throughout, as in Q1/Q2. See
  [Q3]({BASE}curve_clusters_q3_addresses.html).

{related(key)}"""
    return key, "Q5 · K or N — which lever actually matters?", md


# ---------------------------------------------------------------- Q6
def q6() -> tuple[str, str, str]:
    key = "curve_clusters_q6_trimmed"
    src = from_exp("exp8")
    per = collections.defaultdict(dict)
    accept = collections.defaultdict(list)
    gated = collections.defaultdict(list)
    for r in src:
        per[(r["dataset"], r["order"], r["eps"], r["seed"])][r["trim_q"]] = (
            r["distortion_test"], r["trimmed50_test"], r["trimmed90_test"])
        accept[r["trim_q"]].append(r["accept_frac"])
        if r.get("gated_accept_rate") is not None:
            gated[r["trim_q"]].append(r["gated_accept_rate"])
    cal = table(["nominal q", "gated accept rate (compare to q)", "overall accept rate", "runs"],
                [[q, f"**{statistics.mean(gated[q]):.3f}**" if gated.get(q) else "n/a (accepts all)",
                  f"{statistics.mean(v):.3f}", len(v)] for q, v in sorted(accept.items())])

    sets = ["spiral2d", "square2d", "sub8_in40", "mnist_pca32"]
    qs = [0.25, 0.5, 0.75]

    def avg(mi, eps, q, order=1):
        rs = [per[k][q][mi] / per[k][1.0][mi] for k in per
              if k[1] == order and k[2] == eps and q in per[k] and 1.0 in per[k]]
        return statistics.mean(rs) if rs else float("nan")

    body = []
    for mi, lab in enumerate(["plain distortion", "trimmed-50", "trimmed-90"]):
        for eps in (0.0, 0.05, 0.20):
            cells = [f"{avg(mi, eps, q):.3f}" for q in qs]
            best = min(range(3), key=lambda i: float(cells[i]))
            cells = [f"**{c}**" if i == best and float(c) < 1.0 else c
                     for i, c in enumerate(cells)]
            body.append([lab, f"{eps:.0%}"] + cells)
    summary = table(["metric", "junk in training"] + [f"keep q={q}" for q in qs], body)

    body2 = []
    for ds in sets:
        for eps in (0.0, 0.20):
            row = [ds, f"{eps:.0%}"]
            for mi in (0, 1):
                cand = []
                for q in qs:
                    rs = [per[k][q][mi] / per[k][1.0][mi] for k in per
                          if k[0] == ds and k[1] == 1 and k[2] == eps
                          and q in per[k] and 1.0 in per[k]]
                    if len(rs) > 1:
                        cand.append((statistics.mean(rs), sum(x < 1 for x in rs), len(rs), q))
                b = min(cand) if cand else None
                row.append(f"{b[0]:.3f} ({b[1]}/{b[2]}) at q={b[3]}" if b else "—")
            body2.append(row)
    detail = table(["dataset", "junk", "best plain", "best trimmed-50"], body2)

    body3 = []
    for ds in sets:
        for eps in (0.0, 0.20):
            row = [ds, f"{eps:.0%}"]
            for order in (1, 2):
                cand = []
                for q in qs:
                    rs = [per[k][q][1] / per[k][1.0][1] for k in per
                          if k[0] == ds and k[1] == order and k[2] == eps
                          and q in per[k] and 1.0 in per[k]]
                    if len(rs) > 1:
                        cand.append((statistics.mean(rs), sum(x < 1 for x in rs), len(rs)))
                b = min(cand) if cand else None
                row.append(f"{b[0]:.3f} ({b[1]}/{b[2]})" if b else "—")
            body3.append(row)
    carry = table(["dataset", "junk", "N=1 (points)", "N=2 (sticks)"], body3)

    md = f"""**Question: instead of moving each centre using all the data assigned to it, move
it using only the closest half. Does that help?**

**Answer: on the original datasets, no — it is worse in all 16 cells tested.**
The mechanism is visible and measurable: trimming discards the far assigned
points, and those are exactly the points that stretch a curve to its full
extent, so the curves contract toward the dense core and stop covering their
clusters. Arc length drops 24–26% at order 6.

But trimming is a *small* effect next to the choice of order. At equal stored
floats a **trimmed** order-2 curve still beats **untrimmed** k-means on three of
four datasets. If you are going to trim, trim curves rather than points.

It only becomes useful if the training data is contaminated, which is a
different question and is at the bottom of this report.

{BACKGROUND}

## On the clean datasets, with curves

K is held fixed at 16 across orders rather than fixing the `K·N` budget, on
purpose: the question is what trimming does to a curve of a given order, so
everything except the order and the trim stays put. Distortion therefore falls
down each column simply because higher N buys more parameters — see
[Q5]({BASE}curve_clusters_q5_k_or_n.html) for the budget-matched comparison.

![Trim vs order]({fig("curve_clusters_trim_shape_orders", figures.trim_shape_grid())})

Rows are curve order, columns are how much of each cluster's data the update may
use. Grey is the data; black dots mark curve endpoints.

**Distortion and mean arc length on clean `spiral2d`:**

| order | untrimmed | q=0.75 | q=0.5 | q=0.25 | damage at q=0.25 |
|---|---|---|---|---|---|
| N=1 (points) | 0.0492 | 0.0549 | 0.0582 | 0.0579 | +18% |
| N=2 | 0.0073 | 0.0105 | 0.0122 | 0.0125 | +71% |
| N=3 | 0.0041 | 0.0066 | 0.0081 | 0.0084 | **+105%** |
| N=6 | 0.0046 | 0.0097 | 0.0105 | 0.0112 | **+143%** |
| *arc length, N=6* | *1.341* | *1.122* | *1.047* | *1.004* | *−25%* |

### How "damage" is measured, and a claim withdrawn

Damage is the relative increase *within a fixed order*:
`(distortion at q − distortion at q=1.0) / distortion at q=1.0`.

Those percentages must not be compared across rows, because the baseline shrinks
as N grows. An earlier version of this report did exactly that and concluded
"the damage grows with the order". In absolute terms it does not:

| order | untrimmed | q=0.25 | absolute increase | relative |
|---|---|---|---|---|
| N=1 | 0.0492 | 0.0579 | **+0.0087** | +18% |
| N=2 | 0.0073 | 0.0125 | +0.0052 | +71% |
| N=3 | 0.0041 | 0.0084 | +0.0043 | +105% |
| N=6 | 0.0046 | 0.0112 | +0.0066 | +143% |

At fixed K the absolute damage is **largest for points** and smaller for
curves — the opposite of the withdrawn claim. The +105% at N=3 is a small
absolute change measured against a baseline twelve times better than N=1's.

The claim does hold once the budget is fixed instead of K, and for a different
reason (below).

**Trimming is worse at every order** — that part stands, 16 of 16 cells,
monotone in q.

**And the mechanism is measurable, not just plausible.** Arc length falls
monotonically with trimming — 25% shorter at order 6 on `spiral2d`, 26% on
`square2d`, 24% on `checkerboard`. A curve's extent is determined by the spread
of its assigned points, and trimming throws away precisely the points at the
ends. The curves shrink into their cluster cores and stop covering them.

Across datasets at order 3:

![Trim across datasets]({fig("curve_clusters_trim_shape_datasets", figures.trim_shape_datasets())})

| dataset | untrimmed | q=0.5 | arc: untrimmed → q=0.5 |
|---|---|---|---|
| spiral2d | 0.0041 | 0.0081 | 0.894 → 0.730 |
| moons2d | 0.0020 | 0.0036 | 0.653 → 0.564 |
| square2d | 0.0103 | 0.0142 | 1.344 → 1.168 |
| checkerboard | 0.0073 | 0.0132 | 0.970 → 0.810 |

Same result on all four. There is no dataset in this set where trimming helps on
clean data.

### Order matters more than trimming

Holding K fixed, as above, gives curves more parameters than points, so it
cannot say whether a trimmed curve is a better use of memory than untrimmed
points. At equal stored floats (`K·N·D` fixed, 4 seeds):

| dataset | untrimmed k-means K=32,N=1 | trimmed K=16,N=2 q=0.5 | trimmed K=8,N=4 q=0.5 |
|---|---|---|---|
| spiral2d | 0.0248 | **0.0143** | 0.0332 |
| moons2d | **0.0095** | 0.0147 | 0.0213 |
| square2d | 0.0351 | **0.0230** | 0.0391 |
| checkerboard | 0.0231 | **0.0168** | 0.0243 |

**A trimmed order-2 curve beats untrimmed k-means at equal cost on three of the
four datasets.** On `spiral2d`, moving from N=1 to N=2 gains 0.0171 while
trimming costs 0.0066 — the order is worth 2.6× more than the trim costs. Only
`moons2d` flips, where the trim loss (0.0095) exceeds the order gain (0.0043).

This is also where the withdrawn "damage grows with order" claim becomes true,
for a mechanism that is about cluster size rather than order:

| | N=1 | N=2 | N=4 |
|---|---|---|---|
| absolute damage at fixed K=16 | +0.0087 | +0.0052 | +0.0043 *(N=3)* |
| absolute damage at fixed budget | +0.0034 | +0.0084 | **+0.0235** |

At fixed budget a higher order buys **fewer clusters**, so each curve spans more
data and has more extent for trimming to take away. Order 4 at K=8 nearly
triples its distortion under trimming. So the thing that makes trimming
dangerous is long curves, not high order as such — and the two only coincide
when the budget is held fixed.


## What was changed

Assignment is untouched — every point still goes to its nearest centre, so
distortion is still measured over all data. Only the **write** is restricted:
the centre moves toward a point only if that point is within the closest `q`
fraction of what has been assigned to the cluster so far. `q = 1.0` is the
standard rule, and `experiments8.py --selftest` confirms it is bit-identical to
the untrimmed model (max difference **0.0**), so the baseline is exact.

The threshold is a per-cluster stochastic quantile estimate, so extra state is
O(K). A buffer of recent distances would estimate it better but costs K×W
floats — at K=64, W=32 that roughly doubles the model's own memory on a
32-dimensional input, which would make the comparison unfair.

## Stated before running: on plain distortion this must lose

For a fixed partition the mean is the minimiser of the sum of squared distances.
That is what a mean is. So a trimmed centre is strictly worse on mean-squared
distortion **by construction**, and a result reporting only "distortion got
worse" would carry no information.

Trimming is a robustness device: trimmed centres sit in the dense core instead
of being dragged outward by outliers. So the test has to contaminate the data
and measure a robust objective, or it is rigged. Contamination replaces a
fraction of the **training** points with uniform junk over a 3× box; the test
split stays clean.

## How the trim works online, and how well it hits its target

"The closest half" needs the distribution of distances within a cluster, which a
single-pass algorithm cannot store. The threshold is instead a **stochastic
root-finder**: we want `t` with P(d ≤ t) = q, and the update

```
t  +=  eta * (q - 1[d <= t])
```

has expected drift `eta * (q - F(t))` — zero exactly at the q-quantile, positive
below it, negative above. So it walks to the quantile without ever storing a
distance. `eta` is scaled by the cluster's running mean distance to make it
dimensionless and decays as n^(-1/2). Total extra state is three scalars per
cluster.

Two details carry weight. A cluster that rejects everything never moves again —
dead for the rest of the run — so the first `warmup` points per cluster are
accepted unconditionally while the threshold is still meaningless. And
`trim_q >= 1.0` short-circuits before any threshold arithmetic, which is why the
untrimmed baseline is bit-identical rather than differing in the last bits.

**Measured against target:**

{cal}

The **gated** rate is the one to compare against q: the acceptance rate among
points the threshold actually judged. The overall rate is not comparable,
because it includes the unconditional warmup acceptances — at K=64 over 2000
points those are about a third of all assignments, which inflates every row and
makes a working tracker look badly mis-calibrated. An earlier version of this
report made exactly that mistake and wrongly concluded the estimator needed
replacing.

The tracker is accurate at the median and biased high toward the tail: q=0.5
lands within about a point, q=0.25 runs ~6 points high. That direction makes
sense — fewer samples inform a low quantile, and the step is scaled by the mean
distance, which is large relative to the 25th-percentile distance. The honest
residual weakness is the step schedule: `eta ∝ n^(-1/2)` satisfies
`sum(eta) = inf` but not `sum(eta^2) < inf`, so the estimate jitters rather than
formally converging.

## Secondary: it does help if the data is contaminated

Everything above is on the original datasets. Trimming is a robustness device,
so for completeness here is the case it is actually for — training data with
junk injected. This is a different question from the one asked, and it is
reported second for that reason.

K=16 centres so they are legible (exp8's tables use K=64, so these numbers are
not comparable to them).

![Trimmed updates on 2-D data]({fig("curve_clusters_trimmed_picture", figures.trimmed_picture())})

Grey is clean training data, red is injected junk, black dots are centres.
Arrows run from each untrimmed centre to where trimming moved it. The dashed
circle is the widest learned acceptance radius — the threshold itself, made
visible.

The view is clipped to 1.8× the extent of the clean data. Framing on the
contaminated data instead is useless: junk is spread over a box three times the
data's scale, so it sets the axis limits and squashes the actual dataset into a
few pixels. So junk runs past the frame, and each panel counts how many centres
were dragged out of view — which is the effect, counted rather than eyeballed:

| dataset | junk | q | clean distortion | centres dragged off-frame | mean centre move | median accept radius |
|---|---|---|---|---|---|---|
| spiral2d | 0% | 1.0 *(k-means)* | 0.0492 | 0/16 | — | ∞ |
| spiral2d | 0% | 0.75 | 0.0549 | 0/16 | 0.067 | 0.278 |
| spiral2d | 0% | 0.5 | 0.0582 | 0/16 | 0.083 | 0.169 |
| spiral2d | 20% | 1.0 *(k-means)* | 0.1771 | **7/16** | — | ∞ |
| spiral2d | 20% | 0.75 | **0.1431** | **4/16** | 0.277 | 0.691 |
| spiral2d | 20% | 0.5 | 0.1661 | 4/16 | 0.469 | 0.667 |
| square2d | 20% | 1.0 *(k-means)* | 0.1277 | 6/16 | — | ∞ |
| square2d | 20% | 0.75 | **0.1116** | 6/16 | 0.174 | 0.471 |

**On the clean spiral, trimming just makes things slightly worse and barely
moves the centres** — 0.0492 → 0.0549, arrows almost invisible. Nothing to fix,
so restricting the update only loses information.

**With 20% junk, untrimmed k-means loses 7 of its 16 centres entirely.** They
get dragged out of the data and into the surrounding junk, where they represent
nothing. Mild trimming pulls 3 of them back and cuts clean distortion by 19%.

On `square2d` the same 6 centres go off-frame either way, so the 13% gain there
comes from the *surviving* centres being better placed rather than from
rescuing any — two different mechanisms behind similar-looking numbers.

Note also what happens to the acceptance radius under contamination: it inflates
from 0.278 to 0.691, because the threshold is scaled by the cluster's mean
distance and the junk inflates that mean. The trim is fighting a target the junk
is moving.

## Result

Averaged over four datasets, N=1, ratio to untrimmed. Below 1.0 means trimming
wins:

{summary}

![Trimmed updates]({fig("curve_clusters_trimmed", figures.trimmed_updates(ROWS, order=1))})

Three things:

**1. On clean data, trimming loses on plain distortion** — 1.047× at the
mildest setting, monotonically worse as trimming gets more aggressive. Exactly
as the algebra requires.

**2. Under contamination it wins**, and the more junk the better it looks: at
20% junk, 0.969 on plain distortion and 0.929 on trimmed-50.

**3. The mildest trim is always best.** Nominal q=0.75 — a measured gated rate
of about 0.72 — beats q=0.5 and q=0.25 in every row, including under heavy
contamination. **So "use the top half" is over-trimming.** The half is q=0.5,
which is consistently worse than keeping roughly the closest three quarters.

Per dataset, best setting in each cell:

{detail}

The gain is concentrated in the higher-dimensional sets. On `sub8_in40` and
`mnist_pca32` trimming wins even on *clean* data under the trimmed-50 metric;
on the two 2-D sets it does not.

## Does it carry over to curves?

Best trimmed-50 ratio, N=1 versus N=2:

{carry}

Mostly no, and there is a mechanism: a curve needs the *spread* of its assigned
points to determine its direction and extent, and trimming to the dense core
removes exactly the points that define the extension. On `spiral2d` at N=2,
trimming is 1.09× worse on clean data against 1.02× at N=1. Where it still helps
— the higher-dimensional sets — it helps about as much as for points.

## Answer

**On the original datasets: no.** Worse in all 16 order × dataset cells, with a
structural reason — a curve's extent comes from the spread of its assigned
points, and trimming discards the points at the ends, shrinking arc length by
about a quarter.

Two things that are easy to get wrong here, and that this report got wrong once:

- **Damage is not larger for curves.** At fixed K the absolute cost of trimming
  is largest for *points*. It only grows with order once the budget is fixed,
  and then because fewer clusters means longer curves.
- **Trimming is a small effect next to order.** At equal cost a trimmed order-2
  curve beats untrimmed k-means on three of four datasets. So the earlier advice
  — "use it on points, not curves" — was backwards. If you must trim, trim
  short curves.

It remains a robustness knob and nothing else: worth 3–7% on contaminated
training data, and costing you on clean data. Keep about three quarters rather
than half.

## Caveats

- The quantile tracker runs ~6 points high at q=0.25 (accurate at q=0.5), and
  its step schedule jitters rather than formally converging. Conclusions rest on
  gated acceptance rates, not on the overall rate.
- Contamination is uniform junk over a box. Structured contamination — a second
  cluster, label noise, adversarial points — could behave differently.
- 6 seeds, K=64 fixed. No sweep over K, and
  [Q5]({BASE}curve_clusters_q5_k_or_n.html) showed K is the dominant lever.

{related(key)}"""
    return key, "Q6 · Move the centre with only the closest half?", md


# ---------------------------------------------------------------- overview
def overview() -> tuple[str, str, str]:
    key = "curve_clusters_probe"
    rows = [
        ["[Q1 · Is a curve better than a ball?]({0}curve_clusters_q1_shape.html)".format(BASE),
         "Yes, at equal stored floats — 35–55%. But it is the step to a *straight stick*, not the bending."],
        ["[Q2 · Does it survive more budget and data?]({0}curve_clusters_q2_scaling.html)".format(BASE),
         "Yes. The trend suggesting otherwise was budget entangled with points-per-cluster."],
        ["[Q3 · Is it just buying more addresses?]({0}curve_clusters_q3_addresses.html)".format(BASE),
         "Largely yes. At matched rate k-means wins on all five low-dim sets."],
        ["[Q4 · Which dimension matters?]({0}curve_clusters_q4_dimension.html)".format(BASE),
         "Intrinsic, entirely. 3.3× at d=1, nothing by d=32. Real image data: 3–7%."],
        ["[Q5 · K or N?]({0}curve_clusters_q5_k_or_n.html)".format(BASE),
         "K. Orders 2–8 are indistinguishable; only N=1 vs N>1 matters."],
        ["[Q6 · Trim the centre update to the closest half?]({0}curve_clusters_q6_trimmed.html)".format(BASE),
         "Only helps on contaminated data, and the half is too aggressive — keep ~80%."],
    ]
    md = f"""**Can a dataset with no natural order be compressed by storing it as smooth
curves rather than as points?**

Six questions, six reports, 7,148 runs. Each report is readable on its own and
carries the full background.

{table(["report", "answer"], rows)}

## The short version

**Storing clusters as curves rather than points is worth a few percent on real
data, and only if you do not charge for addresses.** The one claim that survives
every framing: the method is for genuinely low-intrinsic-dimensional data,
d ≤ 4 — trajectories, spectra, sensor streams. Not images.

Two claims from earlier drafts are retracted, and the reports say where:

- *"The optimum order is N=3"* — an artefact of pinning `K·N`. With K free,
  orders 2–8 are indistinguishable ([Q5]({BASE}curve_clusters_q5_k_or_n.html)).
- *"Curves win in high dimension"* — measured on a 40-D isotropic Gaussian where
  k-means scores 0.9395 and curves 0.9229. Two methods failing
  ([Q4]({BASE}curve_clusters_q4_dimension.html)).

{BACKGROUND}

## What would make this worth pursuing

1. **Not a high-order polynomial memory.** Many low-order pieces. K is where the
   performance is.
2. **The single-pass algorithm is the part that might be novel** — exact 1-D
   search for the assignment, closed-form minimum-norm write, clusters born as
   points that grow their own order. Prior art to clear first: principal-curve
   clustering, local principal curves, and the 1-D self-organising map, which
   this algorithm closely resembles.
3. **The defensible rate-distortion claim** is that k-means's model grows as 2^R
   in the rate while a curve's does not, so curves win at tight distortion
   targets. That needs a dense `(K, N, G)` sweep to pin down; the grid behind
   [Q3]({BASE}curve_clusters_q3_addresses.html) is coarse.

## Reproducing

```
projects/curve-clusters/
  lib/curves.py        CurveClusters, Legendre basis
  lib/toy.py           seven toy datasets spanning intrinsic dimension
  lib/figures.py       every figure in these reports
  experiments1.py      order sweep at fixed budget  (--selftest: the N=1 proof)
  experiments2.py      budget sweep
  experiments3.py      the starvation control
  experiments4.py      grid sweep — the address-bit accounting
  experiments5.py      ambient vs intrinsic dimension
  experiments6.py      intrinsic-d sweep at pinned ambient D, plus real data
  experiments7.py      K and N independent
  experiments8.py      trimmed centre updates  (--selftest: q=1 is exact)
  scripts/gen_reports.py   builds all seven of these documents
  results.jsonl        7,148 rows
```

Every table in these reports is computed from `results.jsonl` at build time
rather than typed in. Runs in minutes on CPU.
"""
    return key, "Curve Clusters — Overview", md


if __name__ == "__main__":
    print(f"{len(ROWS)} rows\n")
    for k, title, md in [overview(), q1(), q2(), q3(), q4(), q5(), q6()]:
        print(f"{title}\n  {save_report(k, md, title=title)}\n", flush=True)
