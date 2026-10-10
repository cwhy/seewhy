"""Why does the prefix-trained generator copy only the most recent context image?

Evaluation only, on the final checkpoints of plain training (exp15, adam),
prefix with newest real (exp24) and prefix with random reals (exp26), seeds
50-55. 256 held-out contexts of 16 test images, 8 noise samples each.

1. Which position does a generated image copy? The context image with the
   highest pixel correlation to it (a matching tool here, not a similarity
   yardstick). Share matched to each position 1..16.
2. Does the generator's read query vary with its noise? Mean cosine between
   the queries q = W_q z of different noise samples (1 = the same query).
3. Do the keys of different context images share a direction? Mean cosine
   between the keys of different images in the generator's own KDA state.
4. What does the read return? Cosine between the read-out o = S q and each
   stored value v_j; share of reads closest to each position.
(2-4 per head, averaged over the 4 heads.)

Appends one row per run (experiment "copy_target") and writes figures + JSON
(scripts/tmp/copy_target.json).

    ~/.local/bin/uv run --no-sync python projects/recall-gan/scripts/analyse_copy_target.py
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

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from lib.model import Cfg, _l2norm, encode, generate, write  # noqa: E402
from lib.train import append_result  # noqa: E402
from shared_lib.datasets import load_supervised_image  # noqa: E402
from shared_lib.media import save_matplotlib_figure  # noqa: E402

ROWS = [json.loads(l) for l in (PROJECT / "results.jsonl").read_text().splitlines()]
SEEDS = range(50, 56)
ARMS = {"method": [r for r in ROWS if r["experiment"] == "exp15" and r.get("tag") == "adam" and r["seed"] in SEEDS],
        "prefix": [r for r in ROWS if r["experiment"] == "exp24"],
        "prefix_rand": [r for r in ROWS if r["experiment"] == "exp26"]}
LABEL = {"method": "plain training", "prefix": "prefix, newest real", "prefix_rand": "prefix, random reals"}
COL = {"method": "#555555", "prefix": "#2a78d6", "prefix_rand": "#7fb3e8"}

d = load_supervised_image("mnist")
te = np.asarray(d.X_test, np.float32).reshape(len(d.X_test), -1)
if te.max() > 1.5:
    te = te / 255.0
rng = np.random.default_rng(0)
E, K = 256, 8
CTX = jnp.asarray(te[rng.choice(len(te), (E, 16))])


def zscore(x):
    x = x - x.mean(-1, keepdims=True)
    return x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def offdiag_mean_cos(x):
    """x (..., N, D) -> mean cosine between different rows."""
    x = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)
    c = jnp.einsum("...nd,...md->...nm", x, x)
    N = x.shape[-2]
    return float(((c.sum((-1, -2)) - jnp.trace(c, axis1=-2, axis2=-1)) / (N * (N - 1))).mean())


res = {a: [] for a in ARMS}
for arm, rs in ARMS.items():
    for r in sorted(rs, key=lambda r: r["seed"]):
        ck = pickle.load(open(PROJECT / f"params_{r['run']}.pkl", "rb"))
        pg = jax.tree_util.tree_map(jnp.asarray, ck["gen"]); cfg = Cfg(**ck["cfg"])
        H, DK = cfg.n_heads, cfg.dk
        z = jax.random.normal(jax.random.PRNGKey(r["seed"]), (E, K, cfg.z_dim))
        g = generate(pg, CTX, z, cfg)                                              # (E,K,P)
        # 1. copy target by pixel correlation
        corr = jnp.einsum("ekp,ejp->ekj", zscore(g), zscore(CTX))                 # (E,K,16)
        pos_share = np.bincount(np.asarray(corr.argmax(-1)).ravel(), minlength=16) / (E * K)
        # 2-4. inside the generator's memory
        e = encode(pg["enc"], CTX)                                                 # (E,16,D)
        keys = _l2norm((e @ pg["mem"]["Wk"]).reshape(E, 16, H, DK)).transpose(0, 2, 1, 3)   # (E,H,16,DK)
        vals = (e @ pg["mem"]["Wv"]).reshape(E, 16, H, DK).transpose(0, 2, 1, 3)
        q = _l2norm((z @ pg["Wq"]).reshape(E, K, H, DK)).transpose(0, 2, 1, 3)   # (E,H,K,DK)
        S = write(pg["mem"], e, cfg)                                               # (E,H,DK,DK)
        o = jnp.einsum("ehvk,ehnk->ehnv", S, q)                                    # (E,H,K,DK)
        ov = jnp.einsum("ehnv,ehjv->ehnj", o / (jnp.linalg.norm(o, axis=-1, keepdims=True) + 1e-8),
                        vals / (jnp.linalg.norm(vals, axis=-1, keepdims=True) + 1e-8))   # (E,H,K,16)
        read_share = np.bincount(np.asarray(ov.argmax(-1)).ravel(), minlength=16) / ov[..., 0].size
        row = dict(experiment="copy_target", run=f"copy_target_{r['run']}", source_run=r["run"], arm=arm,
                   seed=r["seed"], copy_pos_share=[float(v) for v in pos_share],
                   query_cos_across_noise=offdiag_mean_cos(q), key_cos_across_images=offdiag_mean_cos(keys),
                   read_pos_share=[float(v) for v in read_share],
                   read_cos_last=float(ov[..., -1].mean()), read_cos_others=float(ov[..., :-1].mean()))
        append_result(row); res[arm].append(row)
        print(f"{arm:<12} s{r['seed']}  copies pos16 {pos_share[-1]:.2f}  q-cos {row['query_cos_across_noise']:.2f}  "
              f"key-cos {row['key_cos_across_images']:.2f}  read->pos16 {read_share[-1]:.2f}  "
              f"cos(read,v16) {row['read_cos_last']:.2f} vs others {row['read_cos_others']:.2f}")


def save(name, fig):
    url = save_matplotlib_figure(f"recall-gan_r05_{name}_v1", fig, format="png", dpi=140)
    plt.close(fig)
    return url


fig, axes = plt.subplots(1, 2, figsize=(11, 3.3), sharey=True)
for ax, key, title in ((axes[0], "copy_pos_share", "generated image: which context image it matches best"),
                       (axes[1], "read_pos_share", "generator's memory read: which stored image it returns")):
    for i, a in enumerate(ARMS):
        m = np.median([r[key] for r in res[a]], 0)
        ax.bar(np.arange(1, 17) + (i - 1) * 0.27, m, 0.27, color=COL[a], label=LABEL[a])
    ax.axhline(1 / 16, color="k", ls=":", lw=1); ax.text(0.6, 1 / 16 + 0.02, "uniform", fontsize=7)
    ax.set_xlabel("context position (16 = written last)"); ax.set_title(title, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("share (median of 6 runs)"); axes[0].legend(fontsize=7.5, frameon=False)
fig.tight_layout()
u_pos = save("copy_position", fig)

stats = {a: {k: st.median(r[k] for r in res[a]) for k in
             ("query_cos_across_noise", "key_cos_across_images", "read_cos_last", "read_cos_others")}
         | {"copy_pos16": st.median(r["copy_pos_share"][-1] for r in res[a]),
            "read_pos16": st.median(r["read_pos_share"][-1] for r in res[a])} for a in ARMS}
json.dump(dict(url=u_pos, stats=stats), open(PROJECT / "scripts" / "tmp" / "copy_target.json", "w"), indent=1)
print(u_pos); print(json.dumps(stats, indent=1))
