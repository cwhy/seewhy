"""Report 2: training on every prefix ("teacher forcing"), with and without the
nearest-image context (exp23 / exp24), against the method (exp15, same seeds).

Builds the figures, assembles the markdown, publishes it and writes the same
string to reports/02-prefix-training.md. Run on the GPU box:

    ~/.local/bin/uv run --no-sync python projects/recall-gan/scripts/gen_report_2.py [--figs-only]
"""
import json
import pickle
import statistics as st
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

from lib.model import Cfg, disc_logits_prefix, encode, generate, generate_from_state, generate_prefix, write_step  # noqa: E402
from shared_lib.datasets import load_supervised_image  # noqa: E402
from shared_lib.media import save_matplotlib_figure  # noqa: E402
from shared_lib.report import save_report  # noqa: E402

V = "v2"
OUT_MD = PROJECT / "reports" / "02-prefix-training.md"
ROWS = [json.loads(l) for l in (PROJECT / "results.jsonl").read_text().splitlines()]
SEEDS = range(50, 56)
ARMS = {"method": [r for r in ROWS if r["experiment"] == "exp15" and r.get("tag") == "adam" and r["seed"] in SEEDS],
        "prefix": [r for r in ROWS if r["experiment"] == "exp24"],
        "prefix_nn": [r for r in ROWS if r["experiment"] == "exp23"]}
LABEL = {"method": "the method (plain training)", "prefix": "prefix training only",
         "prefix_nn": "prefix training + nearest-image context"}
COL = {"method": "#555555", "prefix": "#2a78d6", "prefix_nn": "#e8590c"}

d = load_supervised_image("mnist")
te = np.asarray(d.X_test, np.float32).reshape(len(d.X_test), -1)
if te.max() > 1.5:
    te = te / 255.0
TE = jnp.asarray(te)
TE_SQ = (TE ** 2).sum(-1)


def med(arm, k):
    v = [r[f"final_{k}"] for r in ARMS[arm] if f"final_{k}" in r]
    return (st.median(v), min(v), max(v)) if v else None


def median_run(arm, key):
    rs = sorted(ARMS[arm], key=lambda r: r[f"final_{key}"])
    return rs[len(rs) // 2]


def load(run):
    ck = pickle.load(open(PROJECT / f"params_{run}.pkl", "rb"))
    t = lambda p: jax.tree_util.tree_map(jnp.asarray, p)
    return t(ck["disc"]), t(ck["gen"]), Cfg(**ck["cfg"])


def save(name, fig):
    url = save_matplotlib_figure(f"recall-gan_r02_{name}_{V}", fig, format="png", dpi=140)
    plt.close(fig)
    return url


def show(ax, img, cmap="gray", edge=None):
    ax.imshow(np.asarray(img).reshape(28, 28), cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(edge is not None)
        if edge:
            sp.set_edgecolor(edge); sp.set_linewidth(2)


def build_nn_context(pg, cfg, warm, z, W):
    """Exactly the training construction, on the held-out pool: returns the
    context (n,M,P) and each prefix's proposal (n,M,P)."""
    n, M = z.shape[0], z.shape[1]
    S = jnp.zeros((n, cfg.n_heads, cfg.dk, cfg.dk))
    xs, props, nxt = [], [], None
    for t in range(M):
        x = warm[:, t] if t < W else nxt
        S = write_step(pg["mem"], S, encode(pg["enc"], x), cfg)
        prop = generate_from_state(pg, S, z[:, t][:, None], cfg)[:, 0]
        nxt = TE[jnp.argmin(TE_SQ[None] - 2 * prop @ TE.T, axis=1)]
        xs.append(x); props.append(prop)
    return jnp.stack(xs, 1), jnp.stack(props, 1)


def twin_figure(arm, run, nn):
    """One held-out sequence: context image t above the twin fake queried with
    it at prefix t, with the discriminator's P(in context) for both."""
    pd, pg, cfg = load(run["run"])
    rng = np.random.default_rng(5)
    warm = TE[rng.choice(len(te), (1, 16), replace=False)]
    z = jax.random.normal(jax.random.PRNGKey(5), (1, 16, cfg.z_dim))
    if nn:
        ctx, props = build_nn_context(pg, cfg, warm, z, run["nn_warm"])
    else:
        ctx = warm
        props = generate_prefix(pg, ctx, z[:, :, None], cfg)[:, :, 0]
    twin = jnp.concatenate([props[:, :1], props[:, :-1]], 1)               # twin at t = proposal from t-1
    q = jnp.stack([ctx, twin], 2)                                          # (1,16,2,P)
    p = np.asarray(jax.nn.sigmoid(disc_logits_prefix(pd, ctx, q, cfg)))[0]
    fig = plt.figure(figsize=(12, 2.9))
    for t in range(16):
        a1 = fig.add_axes([0.06 + t * 0.058, 0.52, 0.052, 0.3])
        show(a1, ctx[0, t], edge="#f2a33c" if nn and t < run["nn_warm"] else None)
        a1.set_title(f"{p[t, 0]:.2f}", fontsize=7.5, pad=2)
        if t >= 1:
            a2 = fig.add_axes([0.06 + t * 0.058, 0.08, 0.052, 0.3])
            show(a2, twin[0, t], cmap="magma")
            a2.set_title(f"{p[t, 1]:.2f}", fontsize=7.5, pad=2)
    fig.text(0.005, 0.66, "written", fontsize=8, rotation=90, va="center")
    fig.text(0.005, 0.23, "twin fake", fontsize=8, rotation=90, va="center")
    fig.text(0.06, 0.93, f"{LABEL[arm]} — seed {run['seed']}. Number above each image: the discriminator's "
             f"P(in context) at that prefix" + (" (orange frame: the 4 random warm-start images)" if nn else ""),
             fontsize=8.5)
    return save(f"twins_{arm}", fig)


def samples_figure():
    rng = np.random.default_rng(1)
    ctx = TE[rng.choice(len(te), (3, 16))]
    z = jax.random.normal(jax.random.PRNGKey(0), (3, 8, 64))
    fig = plt.figure(figsize=(12, 7.2))
    H = 1 / 13.5
    row = 0
    for arm in ("method", "prefix", "prefix_nn"):
        run = median_run(arm, "class_acc")
        _, pg, cfg = load(run["run"])
        g = generate(pg, ctx, z, cfg)
        fig.text(0.005, 1 - (row + 0.55) * H, f"{LABEL[arm]} (seed {run['seed']})", fontsize=10, weight="bold")
        row += 0.75
        for e in range(3):
            y0 = 1 - (row + 1) * H
            for i in range(16):
                show(fig.add_axes([0.005 + i * 0.03, y0, 0.028, H * 0.85]), ctx[e, i])
            for j in range(8):
                show(fig.add_axes([0.54 + j * 0.056, y0, 0.052, H * 0.85]), g[e, j], cmap="magma")
            row += 1
        row += 0.25
    return save("samples", fig)


def curves_figure():
    keys = [("twin_auroc", "twin AUROC (newest image vs its twin)"),
            ("auroc_real", "AUROC: context vs unseen real images"),
            ("class_acc", "digit classifier (chance 0.1)"),
            ("acc_gen", "share of fakes rejected")]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.0))
    for ax, (k, lab) in zip(axes, keys):
        for arm in ("method", "prefix", "prefix_nn"):
            rs = ARMS[arm]
            if k not in rs[0]["history"]:
                continue
            steps = rs[0]["history"]["step"]
            m = [st.median(r["history"][k][i] for r in rs) for i in range(len(steps))]
            ax.plot(steps, m, color=COL[arm], lw=2, label=LABEL[arm])
        ax.set_title(lab, fontsize=9); ax.set_xlabel("training step", fontsize=8)
        ax.set_ylim(0, 1.02); ax.spines[["top", "right"]].set_visible(False); ax.tick_params(labelsize=7)
    axes[1].legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    return save("curves", fig)


if __name__ == "__main__":
    urls = dict(curves=curves_figure(), samples=samples_figure(),
                twins_nn=twin_figure("prefix_nn", median_run("prefix_nn", "twin_auroc"), nn=True),
                twins_prefix=twin_figure("prefix", median_run("prefix", "twin_auroc"), nn=False))
    stats = {arm: {k: med(arm, k) for k in ("class_acc", "auroc_real", "auroc_gen", "auroc_gen_swap", "acc_gen",
                                            "twin_auroc", "g_ctx_dep", "gen_std")} for arm in ARMS}
    json.dump(dict(urls=urls, stats=stats), open(PROJECT / "scripts" / "tmp" / "report2_figs.json", "w"), indent=1)
    print(json.dumps(urls, indent=1))
    if "--figs-only" not in sys.argv:
        ca = json.load(open(PROJECT / "scripts" / "tmp" / "copy_analysis.json"))
        cu = json.load(open(PROJECT / "scripts" / "tmp" / "context_use.json"))
        S = ca["stats"]
        f3 = lambda v: f"{v:.3f}"
        cu_row = lambda a: " | ".join(f3(cu["stats"][a][str(n)]) for n in (1, 2, 4, 8, 16))
        fields = dict(urls, **ca["urls"], context_use=cu["url"], clf_acc=S["clf_acc"],
                      corr_prefix=S["corr_copy_class_prefix"],
                      cu_m=cu_row("method"), cu_p=cu_row("prefix"), cu_n=cu_row("prefix_nn"))
        for a, k in (("method", "m"), ("prefix", "p"), ("prefix_nn", "n")):
            fields |= {f"{k}_last": f3(S[a]["match_last"]), f"{k}_early": f3(S[a]["match_earlier"]),
                       f"{k}_base": f3(S[a]["base_rate"]), f"{k}_dist": f"{S[a]['distinct_digits_of_8']:.1f}"}
        md = (PROJECT / "scripts" / "report2_text.md").read_text().format(**fields)
        OUT_MD.write_text(md, encoding="utf-8")
        print("report:", save_report("recall-gan_report_02_prefix_training", md))
