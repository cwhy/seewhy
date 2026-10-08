"""Evaluation of the final-recipe checkpoints for the method report.

Evaluation only — no training. Runs: the 12 context-aware runs of exp15
(tag "adam") and the 12 blind runs of exp18/19 (tag "adam"), seeds 50-61.

Appends to results.jsonl (experiment "eval_report"), one row per run plus one
row for the no-learning baselines, so every number the paper quotes is in
that file:

  classifier   one context per digit (16 random test images of it); each other
               test image gets the digit whose context scores it highest.
               5 random context draws. 10-way accuracy + per-digit AUROC.
  density      20 contexts of 16 random test images; every other test image
               scored. Context vs unseen AUROC, median scores, mean score for
               unseen images whose digit is / is not in the context, and a
               20-bin histogram of scores for each group.
  baselines    the classifier protocol with pixel nearest-neighbour and pixel
               RBF kernel density (median-heuristic bandwidth and the best of a
               grid chosen on these same results — the strongest it can be).

Also writes the pixel figures of the paper to paper/assets/*.png, from the
representative context-aware run (the one with the median classifier accuracy
here) unless stated.

    ~/.local/bin/uv run --no-sync python projects/recall-gan/scripts/eval_report.py
"""
import json
import pickle
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.append(str(PROJECT.parents[1]))          # repo root LAST (stray datasets.py on the box)

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from lib.model import Cfg, disc_logits, generate  # noqa: E402
from lib.train import append_result  # noqa: E402
from shared_lib.datasets import load_supervised_image  # noqa: E402

ASSETS = PROJECT / "paper" / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)
SEEDS = range(50, 62)
RUNS = ([("context-aware", s, f"exp15_ctx_m16_s{s}_adam") for s in SEEDS]
        + [("blind", s, f"exp{18 if s < 56 else 19}_blind_m16_s{s}_adam") for s in SEEDS])
DRAWS, N_DENS = 5, 20
BINS = np.linspace(0, 1, 21)

d = load_supervised_image("mnist")
te = np.asarray(d.X_test, np.float32).reshape(len(d.X_test), -1)
if te.max() > 1.5:
    te = te / 255.0
y = np.asarray(d.y_test)
TE = jnp.asarray(te)
SQ = (TE ** 2).sum(-1)
rng = np.random.default_rng(0)
CLS_DRAWS = [[rng.choice(np.where(y == k)[0], 16, replace=False) for k in range(10)] for _ in range(DRAWS)]
DENS_CTX = [rng.choice(len(te), 16, replace=False) for _ in range(N_DENS)]


def auroc(pos, neg):
    neg = neg[:: max(1, len(neg) // 3000)]
    return float(((pos[:, None] > neg[None]) + 0.5 * (pos[:, None] == neg[None])).mean())


def load(run):
    ck = pickle.load(open(PROJECT / f"params_{run}.pkl", "rb"))
    t = lambda p: jax.tree_util.tree_map(jnp.asarray, p)
    return t(ck["disc"]), t(ck["gen"]), Cfg(**ck["cfg"])


def d_scorer(pd, cfg):
    def score(ci):          # logit for every test image, context = test images ci
        ctx = TE[ci][None]
        return np.concatenate([np.asarray(disc_logits(pd, ctx, TE[c:c + 2500][None], cfg))[0]
                               for c in range(0, len(te), 2500)])
    return score


def classifier(score_fn):
    aucs, accs = np.zeros((DRAWS, 10)), []
    for di, ctxs in enumerate(CLS_DRAWS):
        keep = np.ones(len(te), bool); keep[np.concatenate(ctxs)] = False
        S = np.stack([score_fn(ci) for ci in ctxs], 1)
        for k in range(10):
            s = S[keep, k]; yk = y[keep] == k
            aucs[di, k] = auroc(s[yk], s[~yk])
        accs.append(float((S[keep].argmax(1) == y[keep]).mean()))
    return dict(cls_acc=float(np.mean(accs)), cls_acc_min=float(np.min(accs)), cls_acc_max=float(np.max(accs)),
                cls_auroc=[float(a) for a in aucs.mean(0)])


def density(score_fn):
    s_in, s_out, present = [], [], []
    for ci in DENS_CTX:
        p = 1 / (1 + np.exp(-score_fn(ci)))
        keep = np.ones(len(te), bool); keep[ci] = False
        s_in.append(p[ci]); s_out.append(p[keep]); present.append(np.isin(y[keep], y[ci]))
    s_in, s_out, present = map(np.concatenate, (s_in, s_out, present))
    hist = lambda v: (np.histogram(v, BINS)[0] / len(v)).tolist()
    return dict(dens_auroc=auroc(s_in, s_out), dens_med_ctx=float(np.median(s_in)),
                dens_med_unseen=float(np.median(s_out)), dens_share_unseen_above_half=float(np.mean(s_out > 0.5)),
                dens_mean_present=float(s_out[present].mean()), dens_mean_absent=float(s_out[~present].mean()),
                dens_hist_ctx=hist(s_in), dens_hist_present=hist(s_out[present]),
                dens_hist_absent=hist(s_out[~present]))


def pixel_nn(ci):
    c = TE[ci]
    return -np.asarray((SQ[:, None] + (c ** 2).sum(-1)[None] - 2 * TE @ c.T).min(1))


def rbf(sigma):
    def f(ci):
        c = TE[ci]
        d2 = jnp.maximum(SQ[:, None] + (c ** 2).sum(-1)[None] - 2 * TE @ c.T, 0)
        return np.asarray(jax.scipy.special.logsumexp(-d2 / (2 * sigma ** 2), axis=1))
    return f


# ── per-run evaluation ────────────────────────────────────────────────────────
rows = {}
for gen, seed, run in RUNS:
    pd, pg, cfg = load(run)
    sc = d_scorer(pd, cfg)
    r = dict(experiment="eval_report", run=f"eval_report_{run}", source_run=run, generator=gen, seed=seed)
    r |= classifier(sc) | density(sc)
    rows[run] = r
    append_result(r)
    print(f"{gen:<14} s{seed}  classifier {r['cls_acc']:.3f}  context-vs-unseen {r['dens_auroc']:.3f}")

# ── baselines ─────────────────────────────────────────────────────────────────
sub = TE[rng.choice(len(te), 2000, replace=False)]
pd2 = jnp.maximum((sub ** 2).sum(-1)[:, None] + (sub ** 2).sum(-1)[None] - 2 * sub @ sub.T, 0)
sig_med = float(jnp.sqrt(jnp.median(pd2[jnp.triu_indices(2000, 1)])))
grid = (0.1, 0.15, 0.2, 0.3, 0.5, 1.0)
rbf_grid = {f: classifier(rbf(sig_med * f)) for f in grid}
best_f = max(rbf_grid, key=lambda f: rbf_grid[f]["cls_acc"])
base = dict(experiment="eval_report", run="eval_report_baselines", rbf_sigma_median=sig_med,
            rbf_best_factor=best_f, rbf_grid={str(f): v["cls_acc"] for f, v in rbf_grid.items()})
for key, res in (("pixnn", classifier(pixel_nn)), ("rbf_med", rbf_grid[1.0]), ("rbf_best", rbf_grid[best_f])):
    base |= {f"{key}_{k}": v for k, v in res.items()}
append_result(base)
print(f"baselines: pixel NN {base['pixnn_cls_acc']:.3f}  RBF median {base['rbf_med_cls_acc']:.3f}  "
      f"RBF best ({best_f} x median) {base['rbf_best_cls_acc']:.3f}")

# ── pixel figures ─────────────────────────────────────────────────────────────
ctx_runs = sorted((r for r in rows.values() if r["generator"] == "context-aware"), key=lambda r: r["cls_acc"])
blind_runs = sorted((r for r in rows.values() if r["generator"] == "blind"), key=lambda r: r["cls_acc"])
REP, REP_B = ctx_runs[len(ctx_runs) // 2]["source_run"], blind_runs[len(blind_runs) // 2]["source_run"]
print("representative runs:", REP, REP_B)
pd, pg, cfg = load(REP)
_, pg_b, cfg_b = load(REP_B)


def show(ax, img, edge=None, cmap="gray"):
    ax.imshow(np.asarray(img).reshape(28, 28), cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(edge is not None)
        if edge:
            sp.set_edgecolor(edge); sp.set_linewidth(2.5)


# (1) one episode scored by the discriminator
er = np.random.default_rng(7)
ci = er.choice(len(te), 16, replace=False)
ctx = TE[ci][None]
fake = generate(pg, ctx, jax.random.normal(jax.random.PRNGKey(7), (1, 3, 64)), cfg)[0]
unseen = TE[er.choice(np.setdiff1d(np.arange(len(te)), ci), 3, replace=False)]
qs = [(TE[ci[i]], "context image", True) for i in (0, 5, 11)] + \
     [(fake[i], "generated", False) for i in range(3)] + [(unseen[i], "unseen real", None) for i in range(3)]
p = np.asarray(jax.nn.sigmoid(disc_logits(pd, ctx, jnp.stack([q for q, _, _ in qs])[None], cfg)))[0]
fig = plt.figure(figsize=(10, 3.3))
for i in range(16):
    show(fig.add_axes([0.005 + i * 0.062, 0.68, 0.056, 0.26]), te[ci[i]])
fig.text(0.005, 0.96, "context: 16 images, written into the discriminator's KDA state", fontsize=9)
for i, (img, lab, mem) in enumerate(qs):
    ax = fig.add_axes([0.02 + i * 0.108, 0.1, 0.085, 0.36])
    show(ax, img, edge={True: "#1a9850", False: "#d73027", None: "#777777"}[mem],
         cmap="magma" if lab == "generated" else "gray")
    ax.set_title(f"{p[i]:.2f}", fontsize=10, fontweight="bold", pad=2)
    ax.set_xlabel(lab, fontsize=7.5)
fig.text(0.005, 0.55, "queries, with the discriminator's P(in context)", fontsize=9)
fig.savefig(ASSETS / "episode.png", dpi=200); plt.close(fig)

# (2) generator samples: context-aware vs blind, same contexts and noise
sr = np.random.default_rng(1)
ctxs = TE[sr.choice(len(te), (4, 16))]
z = jax.random.normal(jax.random.PRNGKey(0), (4, 6, 64))
g_ctx, g_bl = generate(pg, ctxs, z, cfg), generate(pg_b, ctxs, z, cfg_b)
fig = plt.figure(figsize=(10, 4.6))
for e in range(4):
    y0 = 0.76 - e * 0.245
    for i in range(16):
        show(fig.add_axes([0.005 + i * 0.0335, y0, 0.031, 0.17]), ctxs[e, i])
    for j in range(6):
        show(fig.add_axes([0.555 + j * 0.036, y0, 0.033, 0.17]), g_ctx[e, j], cmap="magma")
        show(fig.add_axes([0.785 + j * 0.036, y0, 0.033, 0.17]), g_bl[e, j], cmap="magma")
fig.text(0.005, 0.96, "context (16 images)", fontsize=9)
fig.text(0.555, 0.96, "context-aware generator", fontsize=9)
fig.text(0.785, 0.96, "blind generator", fontsize=9)
fig.savefig(ASSETS / "samples.png", dpi=200); plt.close(fig)

# (3) same noise, contexts that differ in one digit class
cr = np.random.default_rng(3)
fig = plt.figure(figsize=(10, 3.7))
z = jax.random.normal(jax.random.PRNGKey(5), (1, 8, 64))
for row, k in enumerate((0, 4, 7)):
    ci = cr.choice(np.where(y == k)[0], 16, replace=False)
    g = generate(pg, TE[ci][None], z, cfg)[0]
    y0 = 0.68 - row * 0.32
    for i in range(16):
        show(fig.add_axes([0.005 + i * 0.0335, y0, 0.031, 0.25]), te[ci[i]])
    for j in range(8):
        show(fig.add_axes([0.56 + j * 0.055, y0, 0.05, 0.25]), g[j], cmap="magma")
fig.text(0.005, 0.96, "context: 16 images of one digit", fontsize=9)
fig.text(0.56, 0.96, "8 samples — the same noise in every row", fontsize=9)
fig.savefig(ASSETS / "same_noise.png", dpi=200); plt.close(fig)

# (4) density: highest / median / lowest scored unseen images for one context
ci = DENS_CTX[0]
s = d_scorer(pd, cfg)(ci)
others = np.setdiff1d(np.arange(len(te)), ci)
order = others[np.argsort(-s[others])]
mid = order[len(order) // 2 - 8: len(order) // 2 + 8]
P = lambda i: 1 / (1 + np.exp(-s[i]))
fig = plt.figure(figsize=(10, 5.0))
for row, (idx, lab) in enumerate([(ci, "context"), (order[:16], "highest-scored unseen images"),
                                  (mid, "median-scored unseen images"), (order[-16:], "lowest-scored unseen images")]):
    y0 = 0.76 - row * 0.245
    fig.text(0.005, y0 + 0.2, lab, fontsize=9)
    for i, j in enumerate(idx):
        ax = fig.add_axes([0.005 + i * 0.062, y0, 0.056, 0.15])
        show(ax, te[j])
        if row:
            ax.set_title(f"{P(j):.2f}", fontsize=6.5, pad=1)
fig.savefig(ASSETS / "density_strips.png", dpi=200); plt.close(fig)

json.dump(dict(representative=REP, representative_blind=REP_B), open(ASSETS / "representative.json", "w"))
print("wrote", ASSETS)
