"""Does the discriminator use the whole context, or mostly the last image?

Evaluation only. The digit classifier (one context per digit, fixed draws,
1,000 other test images, as `class_acc` in training) with 1, 2, 4, 8 and 16
images per digit, for the final checkpoints of plain training (exp15, adam),
prefix only (exp24) and prefix + nearest image (exp23), seeds 50-55. If a
discriminator effectively reads only the last image written, its accuracy will
not grow with the number of images.

Appends one row per run (experiment "context_use") to results.jsonl and writes
a figure URL to scripts/tmp/context_use.json.

    ~/.local/bin/uv run --no-sync python projects/recall-gan/scripts/analyse_context_use.py
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

from lib.model import Cfg  # noqa: E402
from lib.train import append_result, make_class_probe  # noqa: E402
from shared_lib.datasets import load_supervised_image  # noqa: E402
from shared_lib.media import save_matplotlib_figure  # noqa: E402

ROWS = [json.loads(l) for l in (PROJECT / "results.jsonl").read_text().splitlines()]
SEEDS = range(50, 56)
ARMS = {"method": [r for r in ROWS if r["experiment"] == "exp15" and r.get("tag") == "adam" and r["seed"] in SEEDS],
        "prefix": [r for r in ROWS if r["experiment"] == "exp24"],
        "prefix_nn": [r for r in ROWS if r["experiment"] == "exp23"]}
LABEL = {"method": "plain training", "prefix": "prefix only", "prefix_nn": "prefix + nearest image"}
COL = {"method": "#555555", "prefix": "#2a78d6", "prefix_nn": "#e8590c"}
NS = (1, 2, 4, 8, 16)

d = load_supervised_image("mnist")
te = np.asarray(d.X_test, np.float32).reshape(len(d.X_test), -1)
if te.max() > 1.5:
    te = te / 255.0
TE, Y = jnp.asarray(te), np.asarray(d.y_test)

res = {a: [] for a in ARMS}
for arm, rs in ARMS.items():
    for r in sorted(rs, key=lambda r: r["seed"]):
        ck = pickle.load(open(PROJECT / f"params_{r['run']}.pkl", "rb"))
        pd = jax.tree_util.tree_map(jnp.asarray, ck["disc"]); cfg = Cfg(**ck["cfg"])
        acc = {n: float(make_class_probe(TE, Y, cfg, n_ctx=n)(pd)) for n in NS}
        row = dict(experiment="context_use", run=f"context_use_{r['run']}", source_run=r["run"], arm=arm,
                   seed=r["seed"], **{f"class_acc_n{n}": v for n, v in acc.items()})
        append_result(row); res[arm].append(row)
        print(f"{arm:<10} s{r['seed']}  " + "  ".join(f"n={n}: {v:.3f}" for n, v in acc.items()))

fig, ax = plt.subplots(figsize=(6.5, 3.3))
for a in ARMS:
    m = [st.median(r[f"class_acc_n{n}"] for r in res[a]) for n in NS]
    lo = [m[i] - min(r[f"class_acc_n{n}"] for r in res[a]) for i, n in enumerate(NS)]
    hi = [max(r[f"class_acc_n{n}"] for r in res[a]) - m[i] for i, n in enumerate(NS)]
    ax.errorbar(NS, m, yerr=[lo, hi], fmt="o-", color=COL[a], label=LABEL[a], capsize=3, ms=4)
ax.set_xscale("log", base=2); ax.set_xticks(NS, [str(n) for n in NS])
ax.axhline(0.1, color="k", ls=":", lw=1); ax.text(1.05, 0.12, "chance", fontsize=8)
ax.set_xlabel("images per digit in the context"); ax.set_ylabel("digit classifier accuracy")
ax.set_ylim(0, 1); ax.legend(fontsize=8, frameon=False, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)
ax.set_title("Median of 6 runs, whiskers min-max", fontsize=8.5)
fig.tight_layout()
url = save_matplotlib_figure("recall-gan_r02_context_use_v1", fig, format="png", dpi=140)
stats = {a: {n: st.median(r[f"class_acc_n{n}"] for r in res[a]) for n in NS} for a in ARMS}
json.dump(dict(url=url, stats=stats), open(PROJECT / "scripts" / "tmp" / "context_use.json", "w"), indent=1)
print(url); print(json.dumps(stats, indent=1))
