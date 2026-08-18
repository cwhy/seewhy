"""Generates Report 8: retrieval and identification are one operation at two temperatures.

Reframes report 7. Model-free evidence: `baselines_M16_r14_knn_Q1`,
`baselines_M16_r14` (`mse_knn_by_tau`). Model evidence: `effective_tau2`
(present-condition fit, valid), `effective_tau` (absent-condition fit, kept as
the methods-note negative example). Dial: `exp24`, `exp27`, `exp28` (knn,
frozen, distractors 0/64/512 ranks out) against `exp26` (frozen, iid).

Run on the GPU box:
    uv run --no-sync python projects/recall-gen/scripts/gen_report_08.py
"""
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.parent))   # repo root
sys.path.insert(0, str(Path(__file__).parent.parent))              # projects/recall-gen

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shared_lib.media import save_matplotlib_figure
from shared_lib.report import save_report

REPORT_MD_PATH = Path(__file__).parent.parent / "reports" / "08-retrieval-temperature.md"
RESULTS = Path(__file__).parent.parent / "results.jsonl"

rows = {}
for line in open(RESULTS):
    r = json.loads(line)
    rows[r["experiment"]] = r

PROJ = "recall-gen"
TAUS_PLOT = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]


# ── Figure 1: the model-free temperature sweep, knn and iid ────────────────
def fig_sweep():
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), sharey=False)

    specs = [
        ("baselines_M16_r14_knn_Q1", "A_seen_present", "D_novel_absent",
         "knn context (M=16 nearest neighbours)"),
        ("baselines_M16_r14", "A_seen_present", "D_novel_absent",
         "i.i.d. context (original task)"),
    ]
    for ax, (exp, present_key, absent_key, title) in zip(axes, specs):
        b = rows[exp]["baselines"]
        present = b[present_key]
        absent = b[absent_key]
        p_mean, a_mean = present["mse_mean"], absent["mse_mean"]
        p_vals = [present["mse_knn_by_tau"][str(t)] / p_mean for t in TAUS_PLOT]
        a_vals = [absent["mse_knn_by_tau"][str(t)] / a_mean for t in TAUS_PLOT]

        ax.plot(TAUS_PLOT, p_vals, "o-", color="C0", lw=1.6,
                label="target present (training objective)")
        ax.plot(TAUS_PLOT, a_vals, "o-", color="C1", lw=1.6,
                label="target absent (generalisation)")
        ax.set_xscale("log")

        pi = int(np.argmin(p_vals))
        ai = int(np.argmin(a_vals))
        ax.scatter([TAUS_PLOT[pi]], [p_vals[pi]], color="C0", zorder=5, s=70,
                   edgecolor="k", linewidth=0.8)
        ax.scatter([TAUS_PLOT[ai]], [a_vals[ai]], color="C1", zorder=5, s=70,
                   edgecolor="k", linewidth=0.8)
        ax.annotate(f"{p_vals[pi]:.3f} @ tau={TAUS_PLOT[pi]}", (TAUS_PLOT[pi], p_vals[pi]),
                    textcoords="offset points", xytext=(6, -10), fontsize=7.5, color="C0")
        ax.annotate(f"{a_vals[ai]:.3f} @ tau={TAUS_PLOT[ai]}", (TAUS_PLOT[ai], a_vals[ai]),
                    textcoords="offset points", xytext=(6, 6), fontsize=7.5, color="C1")
        ax.set_xlabel("tau (soft look-up temperature)")
        ax.set_ylabel("normalised MSE")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7.5, loc="center right")

    fig.suptitle("One kernel, two optima: sharpening trades the training objective\n"
                 "against generalisation, with no model in the picture",
                 fontsize=10)
    fig.tight_layout()
    url = save_matplotlib_figure(f"{PROJ}_tau_sweep", fig, format="svg")
    plt.close(fig)
    return url


# ── Figure 2: tau*(present) against D, trained vs frozen, exp20 arrow ──────
def fig_tau_vs_d():
    et = rows["effective_tau2"]["effective_tau"]

    def taustar(mk):
        return et[mk]["B_novel_present"]["tau_star"]

    trained = [
        ("exp20 best", taustar("exp20_best"), 0.505),
        ("exp20 final", taustar("exp20_final"), 0.666),
        ("exp1 final", taustar("exp1_final"), 0.843),
    ]
    frozen = [
        ("exp24 best", taustar("exp24_best"), 0.471),
        ("exp24 final", taustar("exp24_final"), 0.474),
        ("exp26 final", taustar("exp26_final"), 0.778),
        ("exp27 final", taustar("exp27_final"), 0.509),
        ("exp28 final", taustar("exp28_final"), 0.616),
    ]

    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    ax.scatter([taustar("exp1_final")], [0.843], color="C0", s=70, zorder=5)
    ax.annotate("exp1 final", (taustar("exp1_final"), 0.843), textcoords="offset points",
                xytext=(7, 4), fontsize=8, color="C0")
    ax.scatter([taustar("exp20_final")], [0.666], color="C0", s=70, zorder=5)
    ax.annotate("exp20 final", (taustar("exp20_final"), 0.666), textcoords="offset points",
                xytext=(-58, 8), fontsize=8, color="C0")

    for label, tau, d in frozen + [("exp20 best", taustar("exp20_best"), 0.505)]:
        color = "C0" if label == "exp20 best" else "C1"
        marker = "o" if label == "exp20 best" else "s"
        ax.scatter([tau], [d], color=color, s=70, zorder=5, marker=marker)

    # The trained "exp20 best" and all five frozen checkpoints sit within tau
    # 0.03 of each other and D 0.47-0.78 — too tight to label at the point
    # without collisions. Leader lines to a text column at fixed x, evenly
    # spaced in y, sorted by D so the column reads top-to-bottom as the table.
    cluster = [("exp26 final (frozen)", taustar("exp26_final"), 0.778, "C1"),
               ("exp28 final (frozen)", taustar("exp28_final"), 0.616, "C1"),
               ("exp27 final (frozen)", taustar("exp27_final"), 0.509, "C1"),
               ("exp20 best (trained)", taustar("exp20_best"), 0.505, "C0"),
               ("exp24 final (frozen)", taustar("exp24_final"), 0.474, "C1"),
               ("exp24 best (frozen)", taustar("exp24_best"), 0.471, "C1")]
    x_text = 0.09
    y_texts = np.linspace(0.80, 0.44, len(cluster))
    for (label, tau, d, color), y_text in zip(cluster, y_texts):
        ax.annotate(label, xy=(tau, d), xytext=(x_text, y_text),
                    fontsize=7.8, color=color, va="center",
                    arrowprops=dict(arrowstyle="-", color=color, lw=0.7, alpha=0.7))

    x0, y0 = taustar("exp20_best"), 0.505
    x1, y1 = taustar("exp20_final"), 0.666
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", color="k", lw=1.4))
    ax.annotate("57-fold sharper,\nD rises 0.505 to 0.666", xy=(1.4e-3, 0.585),
                fontsize=7.5)

    ax.set_xscale("log")
    ax.set_xlim(2e-4, 0.35)
    ax.set_ylim(0.40, 0.90)
    ax.set_xlabel("tau*(present): effective temperature fitted on target-present output")
    ax.set_ylabel("D (novel, target absent)")
    ax.set_title("Trained checkpoints sharpen and degrade together; frozen ones cannot move",
                 fontsize=10)
    ax.scatter([], [], color="C0", label="trained mixer")
    ax.scatter([], [], color="C1", marker="s", label="frozen mixer")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    url = save_matplotlib_figure(f"{PROJ}_tau_vs_d", fig, format="svg")
    plt.close(fig)
    return url


# ── Figure 3: the retrievability dial ───────────────────────────────────────
def fig_dial():
    cats = ["0 ranks\n(exp24)", "64 ranks\n(exp27)", "512 ranks\n(exp28)", "i.i.d.\n(exp26)"]
    idb = [0.294921875, 0.52734375, 0.888671875, 0.98779296875]
    degrad = [0.47360331965307195 - 0.4712455593161424,
              0.5093294482230349 - 0.502035564998649,
              0.6157017586047248 - 0.5216484950645995,
              0.7776823437356903 - 0.5695741578723181]

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    x = np.arange(len(cats))
    colors = ["C0", "C0", "C0", "C3"]

    ax = axes[0]
    bars = ax.bar(x, idb, color=colors)
    for b, v in zip(bars, idb):
        ax.annotate(f"{v:.3f}", (b.get_x() + b.get_width() / 2, v),
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, cats, fontsize=8)
    ax.set_ylabel("id(B): identification accuracy, novel target-present")
    ax.set_title("Distractor separation alone raises id(B)", fontsize=9.5)
    ax.set_ylim(0, 1.08)

    ax = axes[1]
    bars = ax.bar(x, degrad, color=colors)
    for b, v in zip(bars, degrad):
        ax.annotate(f"+{v:.3f}", (b.get_x() + b.get_width() / 2, v),
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, cats, fontsize=8)
    ax.set_ylabel("degradation: final D - best D (novel, target absent)")
    ax.set_title("...but exp26 degrades far more than its kernel sharpening explains",
                 fontsize=9.5)
    ax.set_ylim(0, 0.23)

    fig.suptitle("The dial: all four frozen runs, kernel never sharpens (tau*=0.03 "
                 "throughout) except exp26's degradation is unexplained by that alone",
                 fontsize=9.5)
    fig.tight_layout()
    url = save_matplotlib_figure(f"{PROJ}_dial", fig, format="svg")
    plt.close(fig)
    return url


def main():
    url_sweep = fig_sweep()
    url_tau_d = fig_tau_vs_d()
    url_dial = fig_dial()
    print("fig_sweep:", url_sweep)
    print("fig_tau_vs_d:", url_tau_d)
    print("fig_dial:", url_dial)

    md = f"""# Report 8 — retrieval and identification are one operation at two temperatures

Report 7 used identification accuracy as the definition of "retrieval" and
concluded that degradation appears when retrieval is achievable — exp24
(id(B) 0.295) was read as having "no retrieval capability at all". That framing
is wrong, and this report replaces it. Retrieval is not a separate ability that
is present or absent; it is one softmax-over-context operation with a sharpness
knob, and identification accuracy only asks whether the kernel is sharp enough
to land on the exact item. The model-free soft look-up on the knn baseline shows
the conflict with no model in it at all: sharpening from tau=0.03 to tau=0.003
buys **0.359** on the objective training actually optimises (target present,
0.372 → 0.013, `baselines_M16_r14_knn_Q1`) and **costs 0.119** on the
generalisation objective (target absent, 0.553 → 0.672, same run). Retrieving
similar things and identifying the exact one are the same operation at two
temperatures; the training objective only ever pays for the sharp end.

![the temperature sweep: present vs absent error against tau, knn and i.i.d.]({url_sweep})

This does not overturn report 7's headline number — exp24's frozen mixer at
0.471 (`D`, novel, target absent) is still the project's best generaliser, below
the soft look-up ceiling (0.552) and ridge (0.631). Only the explanation for
*why* changes.

Rows: `baselines_M16_r14_knn_Q1`, `baselines_M16_r14`, `effective_tau2`,
`effective_tau`, cited against `exp1`, `exp20`, `exp24`, `exp26`, `exp27`,
`exp28` from reports 6 and 7. One question: what does a trained kernel do that
a frozen one cannot, and is achieved identification the right way to describe it?

---

## The mechanism, measured in trained checkpoints

Fitting each checkpoint's *effective temperature* — the tau whose model-free
soft look-up best reproduces the model's actual output, on the target-present
condition where sharpness is exercised (method below) — shows the same
sharpen-and-degrade trade directly in trained models:

| checkpoint | context | tau*(present) | B (target present) | D (target absent) |
|---|---|---|---|---|
| exp20 best (step 1000) | knn | 0.03 | 0.439 | 0.505 |
| exp20 final (step 12000) | knn | **0.00053** | 0.036 | 0.666 |
| exp1 final (trained) | iid | 0.0017 | 0.017 | 0.843 |
| exp24 best (frozen) | knn | 0.03 | 0.437 | 0.471 |
| exp24 final (frozen) | knn | 0.03 | 0.439 | 0.474 |
| exp26 final (frozen) | iid | 0.03 | 0.150 | 0.778 |
| exp27 final (frozen, 64) | knn+64 | 0.03 | 0.428 | 0.509 |
| exp28 final (frozen, 512) | knn+512 | 0.03 | 0.355 | 0.616 |

(exp1's D is its original `exp1` row's `history` final value, 0.843 — not the
`final` field's 0.852, and not the A1 re-scored `exp1_sharedq` row, also 0.843
but a different protocol; report 6 mixed these once already, so here only one is
used and named.)

![tau*(present) against D, trained vs frozen, exp20's arrow]({url_tau_d})

exp20 sharpens 57-fold between its best checkpoint and its last (tau 0.03 →
0.00053) and its D rises with it, 0.505 → 0.666, over the same interval. exp1
sharpens further still (tau 0.0017) and degrades further still (D 0.843). Every
frozen checkpoint sits at tau*=0.03 — none of them move, because there is
nothing in a frozen mixer for training to sharpen. That is the mechanism claim,
measured rather than asserted: training buys sharpness, and sharpness is what
degradation tracks in every trained run measured here.

## The retrievability dial: identification without training

exp24, exp27 and exp28 are the same frozen architecture (0.60M of 4.03M
parameters trained, mixer at random init) on the same knn context, differing
only in how far the nearest-neighbour distractors are from the query — 0, 64,
512 ranks out. None of them sharpen; tau*(present) stays at 0.03 for all three.
Identification accuracy still climbs:

| ranks skipped | 0 (exp24) | 64 (exp27) | 512 (exp28) | i.i.d. (exp26) |
|---|---|---|---|---|
| id(B) | 0.295 | 0.527 | 0.889 | 0.988 |
| degradation (final − best D) | +0.002 | +0.007 | +0.094 | +0.208 |
| best D | 0.471 | 0.502 | 0.522 | 0.570 |

![id(B) and degradation against ranks skipped, exp26 marked]({url_dial})

So effective sharpness has two independent sources: training lowers tau (the
previous section), and context geometry spreads the distances a fixed-tau
softmax has to separate — when one item is much closer than the rest, even a
soft kernel concentrates on it by itself. exp24 → exp27 → exp28 shows geometry
raising identification with the kernel held fixed. Degradation rises across the
same three points too (+0.002 → +0.007 → +0.094), which is consistent with the
same "reachable retrieval invites drift" account from report 7 — except none of
these three ever trained a sharper kernel, so whatever is drifting toward it
here is the embedding/head, not the mixer's temperature.

## The complication: exp26 breaks the simple version of the claim

exp26 is frozen, its kernel never sharpens (tau*=0.03 throughout, identical to
exp24/27/28), and it still degrades 0.570 → 0.778 — the largest degradation of
any frozen run, larger than exp28's despite exp28 reaching a comparable id(B)
(0.889 vs. exp26's 0.988). Geometry explains exp26's high identification
(i.i.d. distances are well separated, so a soft kernel at tau=0.03 still picks
the right item), but geometry alone does not obviously predict *this much*
degradation from a kernel that never moves.

**Sharpening is a route to degradation, demonstrated cleanly in exp20 and exp1;
it is not the only one.** A plausible candidate for exp26's route, untested
here: the readout (embedding + head, the only trainable part) specialising to
emit whatever the fixed kernel already retrieves, which is a change in the
trained head rather than in the kernel's temperature. This is an open mechanism
question, not a resolved one — flagged here rather than folded into the tau
account it does not fit.

## Methods note: fitting temperature on the wrong condition gives nonsense

The first attempt (`effective_tau`, and the `D_novel_absent` column kept
alongside the valid fit in `effective_tau2`) fitted tau on the absent-target
condition and produced numbers that do not track sharpness at all: exp1 fitted
tau*=0.03 despite id(B) 1.000 (should be very sharp), exp27 fitted tau*=5.33,
exp28 fitted tau*=0.00095. When no context item is close to the query — the
absent-target condition, by construction — the softmax is diffuse at any
temperature, so that fit recovers the distance distribution of the distractors,
not the model's sharpness. The present-condition fit (used throughout this
report) is the valid one because that is the only condition where sharpness is
actually exercised. Both are stored in `effective_tau2`; this note is why the
present-condition numbers above are trustworthy and the absent-condition ones
are not used as evidence.

---

## What this changes

**Report 7's headline number is unchanged: exp24's frozen mixer at 0.471 is
still the project's best generaliser.** What changes is the explanation.
"Degradation appears when retrieval is achievable" implied retrieval is a
capability a model either has or lacks. The corrected account: retrieval is one
operation, degradation tracks how sharp it gets, and sharpness has two
independent sources — training (exp20, exp1) and context geometry (the dial).
Report 7's framing conflated "identification accuracy is low" with "no
retrieval", which mislabelled exp24 as retrieval-free when it is better
described as retrieval running at a temperature too high to identify one item —
the same operation the soft look-up baseline uses at tau=0.03.

**For the paper:** the two-attractor account becomes one mechanism — a
retrieval kernel with a temperature the training objective pays to lower — which
is a simpler and stronger claim than two separate modes. The caveat to carry
into the writeup is exp26: it shows a second degradation route, present even
with the kernel frozen, that is not yet identified. It should be named as an
open question, not folded into the temperature account.

**For the next run:** isolate exp26's route directly — freeze the head as well
(exp25's construction) but on the i.i.d. context, to see whether degradation
survives with *no* trainable readout at all. If it does not, the untested
candidate above (head specialisation) is confirmed; if it does, the second route
is something else again.

## Sources

`results.jsonl` rows `baselines_M16_r14_knn_Q1`, `baselines_M16_r14`,
`effective_tau2`, `effective_tau`, cited against `exp1`, `exp20`, `exp24`,
`exp26`, `exp27`, `exp28` from reports 6 and 7. Temperature fitting is
`scripts/effective_tau.py`; the dial runs are `exp24`/`exp27`/`exp28`, produced
by the knn-offset parameter in `lib/train.py` / `lib/evalsets.py`.
"""

    report_url = save_report(f"{PROJ}_report_08", md)
    print("REPORT:", report_url)

    REPORT_MD_PATH.write_text(md)
    print("LOCAL FILE:", REPORT_MD_PATH)


if __name__ == "__main__":
    main()
