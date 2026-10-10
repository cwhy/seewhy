"""Report 5 (standalone): pretraining the discriminator on real images that are
not in the context, then the GAN (exp29 context-aware G, exp30 blind G), against
the same GAN from scratch (exp15; exp18/19 for blind), seeds 50-55.

Needs, first:
    ~/.local/bin/uv run --no-sync python projects/recall-gan/scripts/analyse_context_use.py --arms=pre_ctx,pre_blind --tag=r05
    ~/.local/bin/uv run --no-sync python projects/recall-gan/scripts/analyse_copying.py --arms=pre_ctx,pre_blind --tag=r05
then:
    ~/.local/bin/uv run --no-sync python projects/recall-gan/scripts/gen_report_5.py [--figs-only]

Text: scripts/report5_text.md. Output: reports/05-pretrained-discriminator.md.
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

from lib.model import Cfg, disc_logits, generate  # noqa: E402
from shared_lib.datasets import load_supervised_image  # noqa: E402
from shared_lib.media import save_matplotlib_figure  # noqa: E402
from shared_lib.report import save_report  # noqa: E402

V = "v1"
OUT_MD = PROJECT / "reports" / "05-pretrained-discriminator.md"
ROWS = [json.loads(l) for l in (PROJECT / "results.jsonl").read_text().splitlines()]
SEEDS = range(50, 56)
ARMS = {"scratch_ctx": (("exp15",), "adam"), "pre_ctx": (("exp29",), None),
        "scratch_blind": (("exp18", "exp19"), "adam"), "pre_blind": (("exp30",), None)}
LABEL = {"scratch_ctx": "from scratch, context-aware G", "pre_ctx": "pretrained D, context-aware G",
         "scratch_blind": "from scratch, blind G", "pre_blind": "pretrained D, blind G"}
COL = {"scratch_ctx": "#555555", "pre_ctx": "#d62728", "scratch_blind": "#9b59b6", "pre_blind": "#ff9896"}
STYLE = {"scratch_ctx": "-", "pre_ctx": "-", "scratch_blind": "--", "pre_blind": "--"}


def train_rows(a):
    e, tag = ARMS[a]
    return sorted([r for r in ROWS if r["experiment"] in e and r["seed"] in SEEDS
                   and (tag is None or r.get("tag") == tag)], key=lambda r: r["seed"])


def eval_rows(kind, a):
    runs = {r["run"] for r in train_rows(a)}
    seen, out = set(), []
    for r in ROWS:
        if r["experiment"] == kind and r.get("source_run") in runs and r["source_run"] not in seen:
            seen.add(r["source_run"]); out.append(r)
    return out


def med(vals):
    vals = [v for v in vals]
    return st.median(vals), min(vals), max(vals)


d = load_supervised_image("mnist")
te = np.asarray(d.X_test, np.float32).reshape(len(d.X_test), -1)
if te.max() > 1.5:
    te = te / 255.0
TE = jnp.asarray(te)


def save(name, fig):
    url = save_matplotlib_figure(f"recall-gan_r05_{name}_{V}", fig, format="png", dpi=140)
    plt.close(fig)
    return url


def show(ax, img, cmap="gray"):
    ax.imshow(np.asarray(img).reshape(28, 28), cmap=cmap, vmin=0, vmax=1)
    ax.axis("off")


def load(run):
    ck = pickle.load(open(PROJECT / f"params_{run}.pkl", "rb"))
    t = lambda p: jax.tree_util.tree_map(jnp.asarray, p)
    return t(ck["disc"]), t(ck["gen"]), Cfg(**ck["cfg"])


def median_run(a, key="class_acc"):
    rs = sorted(train_rows(a), key=lambda r: r[f"final_{key}"])
    return rs[len(rs) // 2]


def fig_pretrain():
    rs = train_rows("pre_ctx") + train_rows("pre_blind")
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    steps = rs[0]["pretrain_history"]["step"]
    for k, lab, c in (("auroc_real", "context vs unseen real (AUROC)", "#d62728"),
                      ("class_acc", "digit classifier, 16 per digit", "#1f77b4")):
        ax.plot(steps, [st.median(r["pretrain_history"][k][i] for r in rs) for i in range(len(steps))],
                "o-", color=c, label=lab, ms=3)
    ax.axhline(0.5, color="#d62728", ls=":", lw=0.8); ax.axhline(0.1, color="#1f77b4", ls=":", lw=0.8)
    ax.set_xlabel("pretraining step"); ax.set_ylim(0, 1.02); ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Pretraining only, before any generator (median of 12 runs; dotted: chance)", fontsize=8.5)
    fig.tight_layout()
    return save("pretrain", fig)


def fig_curves():
    keys = [("class_acc", "digit classifier (16 per digit)"), ("auroc_real", "context vs unseen real (AUROC)"),
            ("auroc_gen_swap", "swap test (lower = relies on state)"), ("gen_std", "generator variety")]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.0))
    for ax, (k, lab) in zip(axes, keys):
        for a in ARMS:
            rs = train_rows(a)
            steps = rs[0]["history"]["step"]
            ax.plot(steps, [st.median(r["history"][k][i] for r in rs) for i in range(len(steps))],
                    STYLE[a], color=COL[a], lw=2, label=LABEL[a])
        ax.set_title(lab, fontsize=9); ax.set_xlabel("GAN training step", fontsize=8)
        ax.set_ylim(0, 1.02 if k != "gen_std" else 0.25); ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=7)
    axes[1].legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    return save("curves", fig)


def fig_context_use():
    NS = (1, 2, 4, 8, 16)
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    for a in ARMS:
        rs = eval_rows("context_use", a)
        if not rs:
            continue
        ax.plot(NS, [st.median(r[f"class_acc_n{n}"] for r in rs) for n in NS], "o" + STYLE[a], color=COL[a],
                label=LABEL[a], ms=4, lw=2)
    ax.set_xscale("log", base=2); ax.set_xticks(NS, [str(n) for n in NS])
    ax.axhline(0.1, color="k", ls=":", lw=1); ax.text(1.05, 0.12, "chance", fontsize=8)
    ax.set_xlabel("images per digit in the context"); ax.set_ylabel("digit classifier accuracy")
    ax.set_ylim(0, 1); ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return save("context_use", fig)


def fig_samples():
    rng = np.random.default_rng(1)
    ctx = TE[rng.choice(len(te), (3, 16))]
    z = jax.random.normal(jax.random.PRNGKey(0), (3, 8, 64))
    arms = ("scratch_ctx", "pre_ctx", "pre_blind")
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


def fig_density():
    rng = np.random.default_rng(11)
    ci = rng.choice(len(te), 16, replace=False)
    arms = ("scratch_ctx", "pre_ctx")
    fig = plt.figure(figsize=(12, 1.6 + 3.3 * len(arms)))
    H = 1 / (len(arms) * 3.8 + 2.0)
    fig.text(0.005, 1 - 0.4 * H, "context", fontsize=9)
    for i in range(16):
        show(fig.add_axes([0.005 + i * 0.062, 1 - 1.3 * H, 0.055, H * 0.8]), te[ci[i]])
    row = 1.5
    for a in arms:
        run = median_run(a)
        pd, _, cfg = load(run["run"])
        s = np.concatenate([np.asarray(disc_logits(pd, TE[ci][None], TE[c:c + 2500][None], cfg))[0]
                            for c in range(0, len(te), 2500)])
        others = np.setdiff1d(np.arange(len(te)), ci)
        order = others[np.argsort(-s[others])]
        mid = order[len(order) // 2 - 8: len(order) // 2 + 8]
        fig.text(0.005, 1 - (row + 0.45) * H, f"{LABEL[a]} (seed {run['seed']})", fontsize=10, weight="bold")
        row += 0.6
        for idx, lab in ((order[:16], "highest"), (mid, "median"), (order[-16:], "lowest")):
            y0 = 1 - (row + 1) * H
            fig.text(0.005, y0 + H * 0.9, lab, fontsize=7.5, color="#555")
            for i, j in enumerate(idx):
                ax = fig.add_axes([0.005 + i * 0.062, y0, 0.055, H * 0.72])
                show(ax, te[j]); ax.set_title(f"{1 / (1 + np.exp(-s[j])):.2f}", fontsize=6, pad=1)
            row += 1
        row += 0.1
    return save("density", fig)


def table():
    f = lambda t: f"{t[0]:.3f} [{t[1]:.2f}, {t[2]:.2f}]"
    arms = list(ARMS)
    lines = ["| median [min, max] of 6 runs | " + " | ".join(LABEL[a] for a in arms) + " |",
             "|---|" + "---|" * len(arms)]

    def add(name, fn):
        lines.append(f"| {name} | " + " | ".join(fn(a) for a in arms) + " |")

    add("after pretraining only: context vs unseen real",
        lambda a: f(med(r["pretrain_history"]["auroc_real"][-1] for r in train_rows(a))) if "pretrain_history" in train_rows(a)[0] else "—")
    add("after pretraining only: digit classifier",
        lambda a: f(med(r["pretrain_history"]["class_acc"][-1] for r in train_rows(a))) if "pretrain_history" in train_rows(a)[0] else "—")
    for k, name in (("auroc_real", "end: context vs unseen real (AUROC)"), ("class_acc", "end: digit classifier (16 per digit)"),
                    ("auroc_gen", "end: context vs generated (AUROC)"), ("auroc_gen_swap", "end: swap test"),
                    ("acc_gen", "end: share of fakes rejected"), ("gen_std", "end: generator variety")):
        add(name, lambda a, k=k: f(med(r[f"final_{k}"] for r in train_rows(a))))
    add("end: generator collapsed (variety < 0.06)",
        lambda a: f"{sum(r['final_gen_std'] < 0.06 for r in train_rows(a))} of {len(train_rows(a))}")
    return "\n".join(lines)


if __name__ == "__main__":
    urls = dict(pretrain=fig_pretrain(), curves=fig_curves(), context_use=fig_context_use(), samples=fig_samples(),
                density=fig_density())
    tab = table()
    json.dump(dict(urls=urls, table=tab), open(PROJECT / "scripts" / "tmp" / "report5_figs.json", "w"), indent=1)
    print(json.dumps(urls, indent=1)); print(tab)
    if "--figs-only" not in sys.argv:
        md = (PROJECT / "scripts" / "report5_text.md").read_text().format(table=tab, **urls)
        OUT_MD.write_text(md, encoding="utf-8")
        print("report:", save_report("recall-gan_report_05_pretrained_discriminator", md))
