"""Generates Report 7: what does training the weights actually buy?

Run on the GPU box: `uv run --no-sync python projects/recall-gen/scripts/gen_report_07.py`
"""
import json
import pickle
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.parent))   # repo root
sys.path.insert(0, str(Path(__file__).parent.parent))              # projects/recall-gen

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shared_lib.media import save_matplotlib_figure
from shared_lib.report import save_report

from lib.core import row_mask, Cfg, predict
from lib import evalsets
from lib.train import Run, build_pools
from baselines import _soft_lookup

REPORT_MD_PATH = Path(__file__).parent.parent / "reports" / "07-frozen-mixer.md"
RESULTS = Path(__file__).parent.parent / "results.jsonl"
PROJECT = Path(__file__).parent.parent

rows = {}
for line in open(RESULTS):
    r = json.loads(line)
    rows[r["experiment"]] = r

PROJ = "recall-gen"
CFG = Cfg(d_model=256, n_layers=4, dk=64, n_heads=4, n_tokens=17)   # exp20/exp24 (Q=1)


# ── Figure 1: D-curves — exp20 vs exp24 (knn), exp1 vs exp26 (iid) ────────────
def fig_curves():
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), sharey=True)

    refs = [(0.552, "soft look-up ceiling, knn", "C2"),
            (0.631, "ridge, knn eval set", "C3")]
    ax = axes[0]
    for name, color, label in [("exp20", "C0", "exp20 (trained, knn)"),
                                ("exp24", "C1", "exp24 (frozen mixer, knn)")]:
        h = rows[name]["history"]
        steps, d = h["step"], h["nmse"]["D_novel_absent"]
        ax.plot(steps, d, color=color, lw=1.6, label=label)
        best_i = int(np.argmin(d))
        ax.scatter([steps[best_i]], [d[best_i]], color=color, zorder=5, s=36)
        ax.annotate(f"{d[best_i]:.3f}", (steps[best_i], d[best_i]),
                    textcoords="offset points", xytext=(5, 6), fontsize=8, color=color)
    for y, label, c in refs:
        ax.axhline(y, ls=":", lw=1.1, color=c)
        ax.annotate(label, (max(rows["exp20"]["history"]["step"]), y),
                    textcoords="offset points", xytext=(-4, 3), fontsize=7.5,
                    color=c, ha="right")
    ax.set_title("knn context (M=16 nearest neighbours)")
    ax.set_xlabel("training step")
    ax.set_ylabel("normalised MSE, D (novel, target absent)")
    ax.legend(fontsize=8, loc="upper right")

    refs2 = [(1.002, "soft look-up ceiling, iid", "C2"),
             (0.645, "ridge, iid eval set", "C3")]
    ax = axes[1]
    for name, color, label in [("exp1", "C0", "exp1 (trained, iid)"),
                                ("exp26", "C1", "exp26 (frozen mixer, iid)")]:
        h = rows[name]["history"]
        steps, d = h["step"], h["nmse"]["D_novel_absent"]
        ax.plot(steps, d, color=color, lw=1.6, label=label)
        best_i = int(np.argmin(d))
        ax.scatter([steps[best_i]], [d[best_i]], color=color, zorder=5, s=36)
        ax.annotate(f"{d[best_i]:.3f}", (steps[best_i], d[best_i]),
                    textcoords="offset points", xytext=(5, 6), fontsize=8, color=color)
    for y, label, c in refs2:
        ax.axhline(y, ls=":", lw=1.1, color=c)
        ax.annotate(label, (max(rows["exp1"]["history"]["step"]), y),
                    textcoords="offset points", xytext=(-4, 3), fontsize=7.5,
                    color=c, ha="right")
    ax.set_title("i.i.d. context (original task)")
    ax.set_xlabel("training step")
    ax.legend(fontsize=8, loc="lower right")

    fig.suptitle("Freezing the KDA mixer inverts the D-curve on knn, but not on i.i.d.",
                 fontsize=10)
    fig.tight_layout()
    url = save_matplotlib_figure(f"{PROJ}_frozen_curves", fig, format="svg")
    plt.close(fig)
    return url


# ── Figure 2: degradation vs achieved retrieval, all five runs ────────────────
def fig_degradation_vs_retrieval():
    pts = [
        ("exp1\n(trained, iid)",   1.000, 0.852 - 0.635),
        ("exp20\n(trained, knn)",  0.988, 0.666 - 0.505),
        ("exp24\n(frozen, knn)",   0.295, 0.474 - 0.471),
        ("exp26\n(frozen, iid)",   0.988, 0.778 - 0.570),
        ("exp25\n(frozen incl. head, knn)", 0.051, 2.974 - 2.974),
    ]
    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    for label, idb, rise in pts:
        ax.scatter([idb], [rise], s=60, zorder=5)
        ax.annotate(label, (idb, rise), textcoords="offset points",
                    xytext=(7, 4), fontsize=8)
    ax.axhline(0.0, ls="--", lw=0.8, color="grey")
    ax.set_xlabel("id(B): identification accuracy, novel target-present")
    ax.set_ylabel("degradation: final D − best D  (novel, target absent)")
    ax.set_title("Degradation tracks achieved retrieval, not trainable weights")
    ax.set_xlim(0.0, 1.05)
    fig.tight_layout()
    url = save_matplotlib_figure(f"{PROJ}_degradation_vs_retrieval", fig, format="svg")
    plt.close(fig)
    return url


# ── Figure 3: the ablation, exp24 beside exp20, best and final ────────────────
def fig_ablation():
    ab = rows["ctx_ablation2"]["ablation"]
    groups = [("exp20\nbest (step 1000)", ab["exp20_best"]),
              ("exp20\nfinal (step 12000)", ab["exp20_final"]),
              ("exp24\nbest (step 11500)", ab["exp24_best"]),
              ("exp24\nfinal (step 12000)", ab["exp24_final"])]
    conds = ["proper", "swapped", "iid"]
    colors = {"proper": "C0", "swapped": "C3", "iid": "C1"}

    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    x = np.arange(len(groups))
    w = 0.25
    for i, cond in enumerate(conds):
        vals = [g[cond] for _, g in groups]
        bars = ax.bar(x + (i - 1) * w, vals, width=w, label=cond, color=colors[cond])
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.3f}", (b.get_x() + b.get_width() / 2, v),
                        ha="center", va="bottom", fontsize=7.5)
    ax.axhline(1.0, ls="--", lw=0.8, color="grey")
    ax.set_xticks(x, [g[0] for g in groups], fontsize=8)
    ax.set_ylabel("normalised MSE, D (novel, target absent)")
    ax.set_title("exp24 reads its context more than exp20 ever did")
    ax.legend(fontsize=8, title="context given at eval time", title_fontsize=8)
    fig.tight_layout()
    url = save_matplotlib_figure(f"{PROJ}_frozen_ablation", fig, format="svg")
    plt.close(fig)
    return url


# ── Shared setup for the reconstruction figure ─────────────────────────────────
def _load(name: str):
    with open(PROJECT / f"params_{name}.pkl", "rb") as f:
        return jax.tree_util.tree_map(jnp.asarray, pickle.load(f))


def _per_sample_hidden_mse(pred, tgt, mask):
    return np.asarray((((pred - tgt) ** 2) * mask).sum(-1)[:, 0] / mask.sum())


def _completion_setup():
    rn = Run(exp_name="report07_completion", name="report07_completion",
             M=16, Q=1, mask_rows=14, cfg=CFG)
    pools, labels = build_pools(rn)
    mask = row_mask(14)
    mask_j = jnp.array(mask)
    mean_img = pools["train"].mean(0)

    knn_ev = evalsets.build(pools, mask, 16, 1, 512, mean_img,
                            ctx_mode="knn", labels=labels)["D_novel_absent"]

    soft_pred = np.asarray(_soft_lookup(knn_ev.ctx, knn_ev.qry, mask_j, 0.01))
    soft_err = _per_sample_hidden_mse(jnp.array(soft_pred), knn_ev.qry, mask_j)
    order = np.argsort(soft_err)
    pcts = [5, 23, 41, 59, 77, 95]
    cols = order[[int(round(p / 100 * (len(order) - 1))) for p in pcts]]

    return dict(mask=mask, mask_j=mask_j, mean_img=mean_img, knn_ev=knn_ev,
               soft_pred=soft_pred, cols=cols, pcts=pcts)


def _composite(truth, hidden_pred, mask):
    return truth * (1 - mask) + hidden_pred * mask


def _panel_grid(fig_name, row_specs, mask, pcts):
    ncols = len(pcts)
    nrows = len(row_specs)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.35, nrows * 1.6))
    if nrows == 1:
        axes = axes[None, :]
    for r, (label, truth, hidden_pred) in enumerate(row_specs):
        for c in range(ncols):
            ax = axes[r, c]
            t = np.asarray(truth[c]).reshape(28, 28)
            if hidden_pred is None:
                img = t
                err_txt = ""
            else:
                comp = _composite(np.asarray(truth[c]), np.asarray(hidden_pred[c]), mask)
                img = comp.reshape(28, 28)
                e = float((((np.asarray(hidden_pred[c]) - np.asarray(truth[c])) ** 2)
                          * mask).sum() / mask.sum())
                err_txt = f"{e:.2f}"
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if err_txt:
                ax.text(0.98, 0.03, err_txt, transform=ax.transAxes, ha="right", va="bottom",
                        fontsize=6.5, color="lime",
                        bbox=dict(boxstyle="square,pad=0.05", fc="black", alpha=0.6, lw=0))
            if c == 0:
                ax.set_ylabel(label, fontsize=7.5)
            if r == 0:
                ax.set_title(f"p{pcts[c]}", fontsize=7)
    fig.tight_layout(pad=0.25)
    url = save_matplotlib_figure(fig_name, fig, format="png", dpi=150)
    plt.close(fig)
    return url


def fig_completion(setup):
    knn_ev, cols = setup["knn_ev"], setup["cols"]
    mask, mask_j = setup["mask"], setup["mask_j"]
    truth = np.asarray(knn_ev.qry[cols, 0])
    ctx_c, qry_c = knn_ev.ctx[cols], knn_ev.qry[cols]
    mean_row = np.broadcast_to(setup["mean_img"], truth.shape)
    soft_row = setup["soft_pred"][cols, 0]

    p_exp20 = _load("exp20")
    p_exp24_best = _load("exp24_best")
    p_exp24 = _load("exp24")
    pred_exp20 = np.asarray(predict(p_exp20, ctx_c, qry_c, mask_j, CFG)[:, 0])
    pred_exp24_best = np.asarray(predict(p_exp24_best, ctx_c, qry_c, mask_j, CFG)[:, 0])
    pred_exp24 = np.asarray(predict(p_exp24, ctx_c, qry_c, mask_j, CFG)[:, 0])

    row_specs = [
        ("true target", truth, None),
        ("mean image\n(no info)", truth, mean_row),
        ("soft look-up\n(model-free)", truth, soft_row),
        ("exp20 final\nstep 12000, D=0.666", truth, pred_exp20),
        ("exp24 best\nstep 11500, D=0.471", truth, pred_exp24_best),
        ("exp24 final\nstep 12000, D=0.474", truth, pred_exp24),
    ]
    return _panel_grid(f"{PROJ}_frozen_completion", row_specs, mask, setup["pcts"])


def main():
    url_curves = fig_curves()
    url_degrad = fig_degradation_vs_retrieval()
    url_abl = fig_ablation()
    print("fig_curves:", url_curves)
    print("fig_degradation_vs_retrieval:", url_degrad)
    print("fig_ablation:", url_abl)

    setup = _completion_setup()
    url_completion = fig_completion(setup)
    print("fig_completion:", url_completion)

    # Filled in by hand after visually inspecting the published figure.
    completion_verdict = (
        "exp24's completions (best and final are visually near-identical, matching "
        "their close D scores of 0.471/0.474) are cleaner and closer to the true "
        "target than exp20 final's at every one of the six columns (per-panel "
        "errors 0.01/0.02/0.02/0.03/0.03/0.06 vs. exp20 final's "
        "0.01/0.03/0.05/0.04/0.05/0.14). The gap is clearest at the hardest column, "
        "p95: exp20 final shows visible doubling — a second, ghosted stroke behind "
        "the main one — where exp24 does not. That is not the \"confidently wrong "
        "digit\" failure mode; both models complete the right digit shape. It is a "
        "direct, visual confirmation of the aggregate gap (0.666 vs. 0.471/0.474): "
        "exp24 produces a cleaner completion at every sampled difficulty.")

    md = f"""# Report 7 — the project's best generaliser has a random mixer

exp24 freezes all four KDA layers at their random initialisation and trains only
the embedding and output head — 0.60M of 4.03M parameters. On the
nearest-neighbour context it reaches **0.471** absent-target error (`D`, novel,
target absent) — the lowest in the project, below the soft-look-up ceiling
(**0.552**) and below context-blind ridge (**0.631**) — and it gets there
**monotonically**: its best value is essentially its last checkpoint (0.471 at
step 11500, 0.474 at step 12000), where the fully-trained exp20 peaks at 0.505 by
step 1000 and then decays to 0.666.

![exp20 vs exp24 (knn) and exp1 vs exp26 (iid)]({url_curves})

Training the sequence-processing stack is not what produces this project's best
result. But the obvious conclusion — "no trainable mixer, so no overfitting" — is
wrong, and the second half of this report is why.

Rows: `exp24`, `exp25`, `exp26`, `ctx_ablation2`, cited against `exp1`, `exp20`
and the M=16 knn/iid gates from report 6. This report answers one question: what
does training the KDA mixer actually buy? It does not cover A2
(memorisation-blocking, exp18/19) — different question, own report.

---

## The frozen mixer is not immune to degradation

exp26 is the control: same freeze (KDA layers at random init, only embedding +
head trained, 0.60M of 4.03M parameters), same architecture, same schedule, but
on the *original i.i.d. context* instead of knn. It degrades exactly like the
fully-trained i.i.d. run does — best D 0.570 (step 1000) to final D 0.778 (step
12000), against exp1's fully-trained best D 0.635 to final D 0.852 (i.i.d. eval
set: soft look-up ceiling 1.002, ridge 0.645).

So an untrained mixer is perfectly capable of the drift that hurts exp1. The
embedding and head alone are enough to do it. "No trainable weights, no
overfitting" does not fit the data — the frozen embedding+head drifts just as
badly as the fully-trained model when the context is i.i.d.

## What actually predicts degradation: achieved retrieval

What fits all five runs (exp1, exp20, exp24, exp25, exp26) is that degradation
tracks how well the model *can* retrieve, not which parameters are trainable:

![degradation vs identification accuracy, all runs]({url_degrad})

* exp1 (trained, iid): id(B) 1.000 — near-perfect retrieval — degrades 0.852 −
  0.635 = 0.217.
* exp20 (trained, knn): id(B) 0.988 — degrades 0.666 − 0.505 = 0.161.
* exp26 (frozen, iid): id(B) 0.988 — degrades 0.778 − 0.570 = 0.208.
* exp24 (frozen, knn): id(B) **0.295** — cannot retrieve — degrades 0.474 − 0.471
  ≈ **0** (improves, if anything).
* exp25 (frozen incl. head, knn): id(B) **0.051** — retrieval is essentially
  gone along with everything else (D stuck at 2.97, far past the mean-image
  reference of 1.0) — degrades 2.974 − 2.974 ≈ 0. It sits at the same
  no-retrieval, no-degradation corner as exp24, for a different reason: it never
  learns to do anything at all (see below).

exp24's distractors are the query's own 16 nearest neighbours — retrieving one
specific neighbour is not a well-posed target, since several are nearly the
query itself. Its id(B) of 0.295 reflects that, not model weakness: it never has
anything to retrieve into. exp26's distractors are unrelated i.i.d. images, so
even with random mixer weights the embedding+head pair finds the one that
matches — id(B) 0.988 — and once retrieval is reachable, continued training
drifts toward it and generalisation decays, exactly as it does for the
fully-trained i.i.d. model.

**What learned mixer weights buy is discriminating near-duplicates, not
retrieval as such.** exp24 does not "have no retrieval circuit" in general — it
simply has nothing to discriminate, because every context item is already close
to the query. The random KDA layers pass along enough of the query's identity
that embedding+head can retrieve from *unrelated* distractors (exp26) but not
from *near-duplicate* ones (exp24). Degradation appears precisely where that
retrieval is achievable.

## exp24 is reading its context, not falling back on a prior

Beating ridge is not on its own evidence of context use (report 6). The same
`ctx_ablation` control run on exp24 (`ctx_ablation2`) settles it:

![context ablation: exp20 vs exp24, best and final]({url_abl})

| | proper | swapped | i.i.d. | context is worth |
|---|---|---|---|---|
| exp20 best (step 1000) | 0.505 | 0.785 | 0.781 | 0.281 |
| exp20 final (step 12000) | 0.666 | 1.358 | 1.194 | 0.693 |
| exp24 best (step 11500) | **0.471** | 0.763 | 0.694 | **0.292** |
| exp24 final (step 12000) | **0.474** | 0.764 | 0.694 | **0.290** |

exp24 gives up more from a context swap (0.292) than exp20 ever managed at its
best checkpoint (0.281) — with a random mixer, 0.60M trained parameters, and no
retrieval capability at all. 0.471 is not a context-free inpainting prior; it is
read from context more than the fully-trained model's best result ever was, and
that dependence does not decay as training continues (0.292 → 0.290), unlike
exp20's collapse from 0.281 to 0.693.

In pixels, at fixed model-free percentiles of the soft look-up's per-sample
error (same construction as report 6 — the ranking uses no trained model, and
every panel composites the true visible half back over the predicted hidden
half, labelled with its own hidden-pixel error):

![exp20 vs exp24 completions, same episodes]({url_completion})

{completion_verdict}

## exp25 pins the boundary

exp25 goes one step further and also freezes the head, training only the input
embedding (`W_pix`, `W_msk`, `role`). It fails outright: 2.869 (C) / 2.974 (D),
against exp24's 0.427 / 0.474, and id(B) collapses to 0.051. A fixed random
readout cannot be compensated for by the embedding alone — the trainable head is
load-bearing. "The embedding alone is enough" is too strong; embedding *and*
head, with the mixer frozen, is what reaches 0.471.

## What is still open

**Parameter count is not separated from mixer-freezing.** exp24 trains 0.60M
parameters against exp20's 4.03M — 16× fewer. exp26 makes the parameter-count
story unlikely as the sole explanation (same 0.60M budget, still degrades on
iid), but it does not close it: a fully-trained model at matched trainable count
(a small mixer, trained, on the knn context) is the control that would. That run
has not been done.

---

## What this changes

**The two-attractor account from report 6 survives, and sharpens.** It is not
"trained weights find generalisation, random weights find nothing" — exp24 beats
every trained model in the project. It is: **the attractor a model lands in
depends on whether retrieval is reachable from its representation**, not on
whether the sequence-processing weights are trained. Training finds retrieval
when the context makes retrieval possible (exp1, exp20 present-target; exp26)
and specialises toward it, degrading absent-target performance as it does.
When retrieval is not reachable — near-duplicate distractors and a frozen mixer
— there is nothing to specialise toward, and the embedding+head keep improving
generalisation for the full 12,000 steps.

For the published paper: the retrieval-crowds-out-generalisation claim should be
stated in terms of *reachable retrieval*, not *trained weights*, and exp24/exp26
are the pair of runs that forces the distinction. The matched-trainable-count
control above is the next run that would close the remaining gap.

## Sources

`results.jsonl` rows `exp24`, `exp25`, `exp26`, `ctx_ablation2`, cited against
`exp1`, `exp20`, `ctx_ablation` from report 6. Freezing is
`lib/train.py:freeze_labels` / `Run.train_only`; the ablation is
`scripts/ctx_ablation.py`.
"""

    report_url = save_report(f"{PROJ}_report_07", md)
    print("REPORT:", report_url)

    REPORT_MD_PATH.write_text(md)
    print("LOCAL FILE:", REPORT_MD_PATH)


if __name__ == "__main__":
    main()
