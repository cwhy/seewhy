"""Does the generator copy the last context image, and is that why the
discriminator's digit classifier drops under prefix training?

Evaluation only, on the final checkpoints of three arms (seeds 50-55):
plain training (exp15, tag adam), prefix only (exp24), prefix + nearest-image
context (exp23).

1. Generator: a small MNIST classifier (trained here on the train split; test
   accuracy reported) labels generated images. After all 16 context images
   (random held-out contexts), how often does a sample's digit match the LAST
   context image's digit, an EARLIER one's (a random one of positions 1-15),
   and how often would a random digit drawn from the context match (base rate)?
2. Discriminator: each context image used as a query against the full-context
   state; P(in context) by its position 1..16. Every one of them is in the
   context, so any position effect is the discriminator forgetting.
3. Per run: copy rate, recency gap (mean P of the last 4 positions minus the
   first 4), and the final digit-classifier accuracy (`class_acc`).

Appends one row per run (experiment "copy_analysis") to results.jsonl and
writes figures + a JSON of URLs/statistics for report 2.

    ~/.local/bin/uv run --no-sync python projects/recall-gan/scripts/analyse_copying.py
"""
import json
import pickle
import statistics as st
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.append(str(PROJECT.parents[1]))

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import optax

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from lib.model import Cfg, disc_logits, generate  # noqa: E402
from lib.train import append_result  # noqa: E402
from shared_lib.datasets import load_supervised_image  # noqa: E402
from shared_lib.media import save_matplotlib_figure  # noqa: E402

V = "v1"
ROWS = [json.loads(l) for l in (PROJECT / "results.jsonl").read_text().splitlines()]
SEEDS = range(50, 56)
ARMS = {"method": [r for r in ROWS if r["experiment"] == "exp15" and r.get("tag") == "adam" and r["seed"] in SEEDS],
        "prefix": [r for r in ROWS if r["experiment"] == "exp24"],
        "prefix_nn": [r for r in ROWS if r["experiment"] == "exp23"]}
LABEL = {"method": "plain training", "prefix": "prefix only", "prefix_nn": "prefix + nearest image"}
COL = {"method": "#555555", "prefix": "#2a78d6", "prefix_nn": "#e8590c"}

d = load_supervised_image("mnist")
tr = np.asarray(d.X, np.float32).reshape(len(d.X), -1)
te = np.asarray(d.X_test, np.float32).reshape(len(d.X_test), -1)
if tr.max() > 1.5:
    tr, te = tr / 255.0, te / 255.0
ytr, yte = np.asarray(d.y), np.asarray(d.y_test)

# ── a labelling tool: a small MLP digit classifier ──────────────────────────────
k1, k2 = jax.random.split(jax.random.PRNGKey(0))
P = dict(W1=jax.random.normal(k1, (784, 512)) / 28.0, b1=jnp.zeros(512),
         W2=jax.random.normal(k2, (512, 10)) / 22.6, b2=jnp.zeros(10))
clf = lambda p, x: jax.nn.relu(x @ p["W1"] + p["b1"]) @ p["W2"] + p["b2"]
opt = optax.adam(1e-3)
os_ = opt.init(P)


@jax.jit
def clf_step(p, s, x, y):
    l, g = jax.value_and_grad(lambda p: optax.softmax_cross_entropy_with_integer_labels(clf(p, x), y).mean())(p)
    u, s = opt.update(g, s, p)
    return optax.apply_updates(p, u), s


rng = np.random.default_rng(0)
for ep in range(8):
    perm = rng.permutation(len(tr))
    for i in range(0, len(tr) - 255, 256):
        P, os_ = clf_step(P, os_, jnp.asarray(tr[perm[i:i + 256]]), jnp.asarray(ytr[perm[i:i + 256]]))
CLF_ACC = float((np.asarray(clf(P, jnp.asarray(te))).argmax(1) == yte).mean())
print(f"labelling classifier: test accuracy {CLF_ACC:.4f}")
label = lambda x: np.asarray(clf(P, x.reshape(-1, 784)).argmax(-1)).reshape(x.shape[:-1])

TE = jnp.asarray(te)
E = 512
ctx_idx = rng.choice(len(te), (E, 16))
CTX = TE[ctx_idx]
CY = yte[ctx_idx]                                                              # (E,16)


def load(run):
    ck = pickle.load(open(PROJECT / f"params_{run}.pkl", "rb"))
    t = lambda p: jax.tree_util.tree_map(jnp.asarray, p)
    return t(ck["disc"]), t(ck["gen"]), Cfg(**ck["cfg"])


res = {a: [] for a in ARMS}
for arm, rs in ARMS.items():
    for r in sorted(rs, key=lambda r: r["seed"]):
        pd, pg, cfg = load(r["run"])
        g = generate(pg, CTX, jax.random.normal(jax.random.PRNGKey(r["seed"]), (E, 8, cfg.z_dim)), cfg)
        gy = label(g)                                                          # (E,8)
        earlier = CY[np.arange(E), rng.integers(0, 15, E)]
        base = np.mean([(CY == k).mean(1) for k in range(10)], 0)             # not used directly
        match_last = float((gy == CY[:, -1:]).mean())
        match_earlier = float((gy == earlier[:, None]).mean())
        # base rate: chance a sample's digit equals a random context image's digit,
        # given the sample's digit — i.e. the share of the context with that digit
        share = np.stack([(CY == gy[:, j:j + 1]).mean(1) for j in range(8)], 1)
        base_rate = float(share.mean())
        in_ctx = float(np.mean([np.isin(gy[e], CY[e]).mean() for e in range(E)]))
        # within-context variety: share of distinct digits among the 8 samples
        distinct = float(np.mean([len(set(gy[e].tolist())) for e in range(E)]))
        # discriminator: P(in context) of each context image, by position
        pm = np.asarray(jax.nn.sigmoid(disc_logits(pd, CTX, CTX, cfg)))      # (E,16)
        by_pos = pm.mean(0)
        row = dict(experiment="copy_analysis", run=f"copy_analysis_{r['run']}", source_run=r["run"], arm=arm,
                   seed=r["seed"], clf_acc=CLF_ACC, match_last=match_last, match_earlier=match_earlier,
                   base_rate=base_rate, digit_in_ctx=in_ctx, distinct_digits_of_8=distinct,
                   d_p_by_position=[float(v) for v in by_pos],
                   recency_gap=float(by_pos[-4:].mean() - by_pos[:4].mean()),
                   class_acc=r["final_class_acc"])
        append_result(row)
        res[arm].append(row)
        print(f"{arm:<10} s{r['seed']}  last {match_last:.3f}  earlier {match_earlier:.3f}  base {base_rate:.3f}  "
              f"distinct {distinct:.2f}/8  recency {row['recency_gap']:+.3f}  class {row['class_acc']:.3f}")


def save(name, fig):
    url = save_matplotlib_figure(f"recall-gan_r02_{name}_{V}", fig, format="png", dpi=140)
    plt.close(fig)
    return url


# figure A: digit match rates
fig, ax = plt.subplots(figsize=(8, 3.4))
x = np.arange(3)
for i, (k, lab, c) in enumerate([("match_last", "same digit as the LAST context image", "#d73027"),
                                 ("match_earlier", "same digit as an EARLIER context image", "#4575b4"),
                                 ("base_rate", "base rate (share of the context with that digit)", "#bbbbbb")]):
    m = [st.median(r[k] for r in res[a]) for a in ARMS]
    lo = [m[j] - min(r[k] for r in res[a]) for j, a in enumerate(ARMS)]
    hi = [max(r[k] for r in res[a]) - m[j] for j, a in enumerate(ARMS)]
    ax.bar(x + (i - 1) * 0.27, m, 0.25, yerr=[lo, hi], capsize=3, color=c, label=lab)
ax.set_xticks(x, [LABEL[a] for a in ARMS]); ax.set_ylabel("share of generated images")
ax.set_ylim(0, 1); ax.legend(fontsize=7.5, frameon=False, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)
ax.set_title("Generator after all 16 context images (512 held-out contexts, 8 samples each; median of 6 runs, "
             "whiskers min-max)", fontsize=8.5)
fig.tight_layout()
u_match = save("copy_match", fig)

# figure B: discriminator P(in context) by position
fig, ax = plt.subplots(figsize=(8, 3.2))
for a in ARMS:
    curves = np.array([r["d_p_by_position"] for r in res[a]])
    ax.plot(range(1, 17), np.median(curves, 0), "o-", color=COL[a], label=LABEL[a], ms=3.5)
ax.set_xlabel("position of the context image in the sequence (16 = written last)")
ax.set_ylabel("D's P(in context)")
ax.set_ylim(0, 1); ax.legend(fontsize=8, frameon=False)
ax.spines[["top", "right"]].set_visible(False)
ax.set_title("Each context image as a query after all 16 writes (median of 6 runs). All are in the context.",
             fontsize=8.5)
fig.tight_layout()
u_pos = save("recency", fig)

# figure C: per run, copying / recency against the classifier
fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
for a in ARMS:
    axes[0].scatter([r["match_last"] - r["match_earlier"] for r in res[a]], [r["class_acc"] for r in res[a]],
                    color=COL[a], label=LABEL[a], s=25)
    axes[1].scatter([r["recency_gap"] for r in res[a]], [r["class_acc"] for r in res[a]], color=COL[a], s=25)
axes[0].set_xlabel("copy preference: match LAST − match EARLIER"); axes[1].set_xlabel("recency gap: P(last 4) − P(first 4)")
for ax in axes:
    ax.set_ylabel("digit classifier"); ax.spines[["top", "right"]].set_visible(False)
axes[0].legend(fontsize=7.5, frameon=False)
fig.suptitle("One point per run (18 runs)", fontsize=9)
fig.tight_layout()
u_scatter = save("copy_vs_classifier", fig)

stats = {a: {k: st.median(r[k] for r in res[a]) for k in
             ("match_last", "match_earlier", "base_rate", "digit_in_ctx", "distinct_digits_of_8", "recency_gap",
              "class_acc")} for a in ARMS}
stats |= {f"{a}_by_position": [float(v) for v in np.median([r["d_p_by_position"] for r in res[a]], 0)] for a in ARMS}
allr = [r for a in ARMS for r in res[a]]
cp = np.array([r["match_last"] - r["match_earlier"] for r in allr]); rg = np.array([r["recency_gap"] for r in allr])
ca = np.array([r["class_acc"] for r in allr])
stats["corr_copy_class_all"] = float(np.corrcoef(cp, ca)[0, 1])
stats["corr_recency_class_all"] = float(np.corrcoef(rg, ca)[0, 1])
pr = [r for a in ("prefix", "prefix_nn") for r in res[a]]
stats["corr_copy_class_prefix"] = float(np.corrcoef([r["match_last"] - r["match_earlier"] for r in pr],
                                                    [r["class_acc"] for r in pr])[0, 1])
stats["corr_recency_class_prefix"] = float(np.corrcoef([r["recency_gap"] for r in pr], [r["class_acc"] for r in pr])[0, 1])
stats["clf_acc"] = CLF_ACC
out = dict(urls=dict(copy_match=u_match, recency=u_pos, copy_scatter=u_scatter), stats=stats)
json.dump(out, open(PROJECT / "scripts" / "tmp" / "copy_analysis.json", "w"), indent=1)
print(json.dumps(out, indent=1))
