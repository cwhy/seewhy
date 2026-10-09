"""Report 3: prefix training with RANDOM real queries (exp25/26) against the
"newest image" version (exp23/24) and plain training (exp15), seeds 50-55.

Needs the evaluation rows first:
    ~/.local/bin/uv run --no-sync python projects/recall-gan/scripts/analyse_copying.py --arms=prefix_rand,prefix_nn_rand --tag=r03
    ~/.local/bin/uv run --no-sync python projects/recall-gan/scripts/analyse_context_use.py --arms=prefix_rand,prefix_nn_rand --tag=r03
then:
    ~/.local/bin/uv run --no-sync python projects/recall-gan/scripts/gen_report_3.py [--figs-only]

Builds figures from results.jsonl (rows "copy_analysis", "context_use" and the
training rows), publishes the report, and writes the same markdown to
reports/03-random-real-queries.md. Text: scripts/report3_text.md.
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

from lib.model import Cfg, disc_logits_prefix, encode, generate, generate_from_state, generate_prefix, write_step  # noqa: E402
from shared_lib.datasets import load_supervised_image  # noqa: E402
from shared_lib.media import save_matplotlib_figure  # noqa: E402
from shared_lib.report import save_report  # noqa: E402

V = "v1"
OUT_MD = PROJECT / "reports" / "03-random-real-queries.md"
ROWS = [json.loads(l) for l in (PROJECT / "results.jsonl").read_text().splitlines()]
SEEDS = range(50, 56)
ARMS = {"method": ("exp15", "adam"), "prefix": ("exp24", None), "prefix_rand": ("exp26", None),
        "prefix_nn": ("exp23", None), "prefix_nn_rand": ("exp25", None)}
LABEL = {"method": "plain training", "prefix": "prefix, newest real", "prefix_rand": "prefix, random reals",
         "prefix_nn": "prefix + nearest, newest real", "prefix_nn_rand": "prefix + nearest, random reals"}
COL = {"method": "#555555", "prefix": "#2a78d6", "prefix_rand": "#7fb3e8", "prefix_nn": "#e8590c",
       "prefix_nn_rand": "#f5a97a"}
STYLE = {"method": "-", "prefix": "-", "prefix_rand": "--", "prefix_nn": "-", "prefix_nn_rand": "--"}


def train_rows(a):
    e, tag = ARMS[a]
    return sorted([r for r in ROWS if r["experiment"] == e and r["seed"] in SEEDS
                   and (tag is None or r.get("tag") == tag)], key=lambda r: r["seed"])


def eval_rows(kind, a):
    runs = {r["run"] for r in train_rows(a)}
    seen, out = set(), []
    for r in ROWS:
        if r["experiment"] == kind and r.get("source_run") in runs and r["source_run"] not in seen:
            seen.add(r["source_run"]); out.append(r)
    return out


def med(vals):
    vals = list(vals)
    return st.median(vals), min(vals), max(vals)


d = load_supervised_image("mnist")
te = np.asarray(d.X_test, np.float32).reshape(len(d.X_test), -1)
if te.max() > 1.5:
    te = te / 255.0
TE = jnp.asarray(te)
TE_SQ = (TE ** 2).sum(-1)


def save(name, fig):
    url = save_matplotlib_figure(f"recall-gan_r03_{name}_{V}", fig, format="png", dpi=140)
    plt.close(fig)
    return url


def show(ax, img, cmap="gray", edge=None):
    ax.imshow(np.asarray(img).reshape(28, 28), cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(edge is not None)
        if edge:
            sp.set_edgecolor(edge); sp.set_linewidth(2)


def load(run):
    ck = pickle.load(open(PROJECT / f"params_{run}.pkl", "rb"))
    t = lambda p: jax.tree_util.tree_map(jnp.asarray, p)
    return t(ck["disc"]), t(ck["gen"]), Cfg(**ck["cfg"])


# ── figures from rows ─────────────────────────────────────────────────────────

def fig_context_use():
    NS = (1, 2, 4, 8, 16)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for a in ARMS:
        rs = eval_rows("context_use", a)
        m = [st.median(r[f"class_acc_n{n}"] for r in rs) for n in NS]
        ax.plot(NS, m, "o" + STYLE[a], color=COL[a], label=LABEL[a], ms=4, lw=2)
    ax.set_xscale("log", base=2); ax.set_xticks(NS, [str(n) for n in NS])
    ax.axhline(0.1, color="k", ls=":", lw=1); ax.text(1.05, 0.12, "chance", fontsize=8)
    ax.set_xlabel("images per digit in the context"); ax.set_ylabel("digit classifier accuracy")
    ax.set_ylim(0, 1); ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Median of 6 runs", fontsize=8.5)
    fig.tight_layout()
    return save("context_use", fig)


def fig_copy():
    fig, ax = plt.subplots(figsize=(9, 3.4))
    x = np.arange(len(ARMS))
    for i, (k, lab, c) in enumerate([("match_last", "same digit as the LAST context image", "#d73027"),
                                     ("match_earlier", "same digit as an EARLIER context image", "#4575b4"),
                                     ("base_rate", "base rate", "#bbbbbb")]):
        ms = [med(r[k] for r in eval_rows("copy_analysis", a)) for a in ARMS]
        ax.bar(x + (i - 1) * 0.27, [m[0] for m in ms], 0.25, color=c, label=lab,
               yerr=[[m[0] - m[1] for m in ms], [m[2] - m[0] for m in ms]], capsize=3)
    ax.set_xticks(x, [LABEL[a].replace(", ", ",\n") for a in ARMS], fontsize=8)
    ax.set_ylabel("share of generated images"); ax.set_ylim(0, 1)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left"); ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Generator after all 16 context images (median of 6 runs, whiskers min-max)", fontsize=8.5)
    fig.tight_layout()
    return save("copy_match", fig)


def fig_position():
    fig, ax = plt.subplots(figsize=(8, 3.3))
    for a in ARMS:
        curves = np.array([r["d_p_by_position"] for r in eval_rows("copy_analysis", a)])
        ax.plot(range(1, 17), np.median(curves, 0), "o" + STYLE[a], color=COL[a], label=LABEL[a], ms=3.5)
    ax.set_xlabel("position of the context image (16 = written last)"); ax.set_ylabel("D's P(in context)")
    ax.set_ylim(0, 1); ax.legend(fontsize=7.5, frameon=False, loc="lower left", ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Each context image as a query after all 16 writes (median of 6 runs). All are in the context.",
                 fontsize=8.5)
    fig.tight_layout()
    return save("position", fig)


def fig_curves():
    keys = [("class_acc", "digit classifier (16 per digit)"), ("auroc_real", "context vs unseen real (AUROC)"),
            ("twin_auroc", "twin AUROC"), ("acc_gen", "share of fakes rejected")]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.0))
    for ax, (k, lab) in zip(axes, keys):
        for a in ARMS:
            rs = train_rows(a)
            if k not in rs[0]["history"]:
                continue
            steps = rs[0]["history"]["step"]
            ax.plot(steps, [st.median(r["history"][k][i] for r in rs) for i in range(len(steps))],
                    STYLE[a], color=COL[a], lw=2, label=LABEL[a])
        ax.set_title(lab, fontsize=9); ax.set_xlabel("training step", fontsize=8)
        ax.set_ylim(0, 1.02); ax.spines[["top", "right"]].set_visible(False); ax.tick_params(labelsize=7)
    axes[1].legend(fontsize=6.5, frameon=False, loc="lower right")
    fig.tight_layout()
    return save("curves", fig)


# ── pixel figures ─────────────────────────────────────────────────────────────

def median_run(a, key="class_acc"):
    rs = sorted(train_rows(a), key=lambda r: r[f"final_{key}"])
    return rs[len(rs) // 2]


def fig_twins(a, nn):
    run = median_run(a, "twin_auroc")
    pd, pg, cfg = load(run["run"])
    rng = np.random.default_rng(5)
    warm = TE[rng.choice(len(te), (1, 16), replace=False)]
    z = jax.random.normal(jax.random.PRNGKey(5), (1, 16, cfg.z_dim))
    if nn:
        S = jnp.zeros((1, cfg.n_heads, cfg.dk, cfg.dk)); xs, props, nxt = [], [], None
        for t in range(16):
            x = warm[:, t] if t < run["nn_warm"] else nxt
            S = write_step(pg["mem"], S, encode(pg["enc"], x), cfg)
            prop = generate_from_state(pg, S, z[:, t][:, None], cfg)[:, 0]
            nxt = TE[jnp.argmin(TE_SQ[None] - 2 * prop @ TE.T, axis=1)]
            xs.append(x); props.append(prop)
        ctx, props = jnp.stack(xs, 1), jnp.stack(props, 1)
    else:
        ctx = warm
        props = generate_prefix(pg, ctx, z[:, :, None], cfg)[:, :, 0]
    twin = jnp.concatenate([props[:, :1], props[:, :-1]], 1)
    p = np.asarray(jax.nn.sigmoid(disc_logits_prefix(pd, ctx, jnp.stack([ctx, twin], 2), cfg)))[0]
    fig = plt.figure(figsize=(12, 2.9))
    for t in range(16):
        a1 = fig.add_axes([0.06 + t * 0.058, 0.52, 0.052, 0.3])
        show(a1, ctx[0, t], edge="#f2a33c" if nn and t < run["nn_warm"] else None)
        a1.set_title(f"{p[t, 0]:.2f}", fontsize=7.5, pad=2)
        if t >= 1:
            a2 = fig.add_axes([0.06 + t * 0.058, 0.08, 0.052, 0.3])
            show(a2, twin[0, t], cmap="magma"); a2.set_title(f"{p[t, 1]:.2f}", fontsize=7.5, pad=2)
    fig.text(0.005, 0.66, "written", fontsize=8, rotation=90, va="center")
    fig.text(0.005, 0.23, "twin fake", fontsize=8, rotation=90, va="center")
    fig.text(0.06, 0.93, f"{LABEL[a]} — seed {run['seed']}. Number: the discriminator's P(in context) at that step"
             + ("; orange frame: the 4 random warm-start images" if nn else ""), fontsize=8.5)
    return save(f"twins_{a}", fig)


def fig_samples():
    rng = np.random.default_rng(1)
    ctx = TE[rng.choice(len(te), (3, 16))]
    z = jax.random.normal(jax.random.PRNGKey(0), (3, 8, 64))
    arms = ("method", "prefix_rand", "prefix_nn_rand")
    fig = plt.figure(figsize=(12, 7.2))
    H = 1 / 13.5
    row = 0
    for a in arms:
        run = median_run(a)
        _, pg, cfg = load(run["run"])
        g = generate(pg, ctx, z, cfg)
        fig.text(0.005, 1 - (row + 0.55) * H, f"{LABEL[a]} (seed {run['seed']})", fontsize=10, weight="bold")
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


def table():
    f = lambda t: f"{t[0]:.3f}"
    lines = ["| median of 6 runs | " + " | ".join(LABEL[a] for a in ARMS) + " |",
             "|---|" + "---|" * len(ARMS)]

    def add(name, vals):
        lines.append(f"| {name} | " + " | ".join(vals) + " |")

    add("digit classifier, 1 per digit (chance 0.1)",
        [f(med(r["class_acc_n1"] for r in eval_rows("context_use", a))) for a in ARMS])
    add("digit classifier, 16 per digit",
        [f(med(r["class_acc_n16"] for r in eval_rows("context_use", a))) for a in ARMS])
    add("generated digit = last context image's", [f(med(r["match_last"] for r in eval_rows("copy_analysis", a)))
                                                   for a in ARMS])
    add("generated digit = an earlier one's", [f(med(r["match_earlier"] for r in eval_rows("copy_analysis", a)))
                                               for a in ARMS])
    add("distinct digits among 8 samples", [f"{med(r['distinct_digits_of_8'] for r in eval_rows('copy_analysis', a))[0]:.1f}"
                                            for a in ARMS])
    for k, name in (("auroc_real", "context vs unseen real (AUROC)"), ("auroc_gen", "context vs generated (AUROC)"),
                    ("acc_gen", "share of fakes rejected"), ("twin_auroc", "twin AUROC")):
        add(name, [f(med(r[f"final_{k}"] for r in train_rows(a))) if f"final_{k}" in train_rows(a)[0] else "—"
                   for a in ARMS])
    return "\n".join(lines)


if __name__ == "__main__":
    urls = dict(context_use=fig_context_use(), copy_match=fig_copy(), position=fig_position(), curves=fig_curves(),
                twins_prefix_rand=fig_twins("prefix_rand", nn=False),
                twins_prefix_nn_rand=fig_twins("prefix_nn_rand", nn=True), samples=fig_samples())
    tab = table()
    json.dump(dict(urls=urls, table=tab), open(PROJECT / "scripts" / "tmp" / "report3_figs.json", "w"), indent=1)
    print(json.dumps(urls, indent=1)); print(tab)
    if "--figs-only" not in sys.argv:
        md = (PROJECT / "scripts" / "report3_text.md").read_text().format(table=tab, **urls)
        OUT_MD.write_text(md, encoding="utf-8")
        print("report:", save_report("recall-gan_report_03_random_real_queries", md))
