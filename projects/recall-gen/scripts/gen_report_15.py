"""Report 15: train on Fashion-MNIST, test on MNIST.

Evaluation and figures only for exp36/37/38, with exp39 as the reverse-direction
control and exp30 (report 13) as the same-dataset class split to compare against.

Run on the GPU box:
    uv run --no-sync python projects/recall-gen/scripts/gen_report_15.py
"""
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR.parents[1]))
sys.path.insert(0, str(PROJECT_DIR))

import numpy as np

from lib.core import Cfg
from lib import splitfig
from lib.domains import PAIRED_SPLIT
from shared_lib.report import save_report

PROJ = "recall-gen"
REPORT_MD_PATH = PROJECT_DIR / "reports" / "15-fashion-to-mnist.md"
CFG = Cfg(d_model=256, n_layers=4, dk=64, n_heads=4, n_tokens=20, d_in=784)
TRAIN_CLS, HELD_CLS = PAIRED_SPLIT
MASK_ROWS = 14

rows = {}
for line in open(PROJECT_DIR / "results.jsonl"):
    r = json.loads(line)
    rows[r["experiment"]] = r

RECALL, COMPLETE, FROZEN, REVERSE = "exp36", "exp37", "exp38", "exp39"
KNN_TRAINED = "exp40"                       # exp36 with knn contexts during training
CLASS_SPLIT = "exp30"                       # report 13: within Fashion, 0-4 / 5-9
BL = "baselines_fashion_to_mnist_M16_r14_split"
BL_KNN = "baselines_fashion_to_mnist_M16_r14_split_knn_Q1"
BL_REV = "baselines_mnist_to_fashion_M16_r14_split"
BL_KNN4 = "baselines_fashion_to_mnist_M16_r14_split_knn"   # knn contexts at Q=4


def n(exp, cond):
    return rows[exp]["final"][cond]["nmse"]


def idacc(exp, cond):
    return rows[exp]["final"][cond]["id_acc"]


def bl(cond, key, row=BL):
    return rows[row]["baselines"][cond][key]


BANDS = ["Fashion-MNIST\nimages seen in training",
         "Fashion-MNIST\nimages never seen",
         "MNIST\na different dataset"]

LEGEND = [
    ("recall-trained: during training its answer was ALWAYS one of the sixteen context images, so copying always worked.",
     splitfig.BLUE),
    ("completion-trained: during training its answer was NEVER in the context, so it could only ever predict.",
     splitfig.ORANGE),
    ("frozen layers: trained like the recall network, but its four mixing layers keep their random starting values.",
     splitfig.GREEN),
]

SUB = ("Normalised error: 1.00 is no better than drawing the average FASHION image, which is the best constant these networks could have learned.\n"
       "RED is that network's score over all 512 episodes; numbers under tiles are single episodes. Columns are fixed difficulty percentiles, ranked without any network.")

GREY = "#7a7a7a"
MARGINS = {}          # filled by fig_margin, quoted in the prose
CTX = {}              # filled by figs_context: (train_ctx, test_ctx) -> scores
CTX_MARGIN = {}       # filled by figs_context: test_ctx -> median nearest rival

CTX_LEGEND = [
    ("trained on iid contexts (exp36): during training its sixteen context images were unrelated to the query.",
     splitfig.BLUE),
    ("trained on knn contexts (exp40): during training its sixteen context images were the query's own nearest neighbours.",
     splitfig.GREEN),
]

CTX_SUB = ("Both networks are recall-trained on Fashion-MNIST and identical apart from what their training contexts were made of. Q=4 throughout, so context TYPE is the only thing that differs.\n"
           "Normalised error against the average Fashion image. RED is the score over all 512 episodes; numbers under tiles are single episodes.")


def figs_context():
    """The 2x2: what the context was during TRAINING against what it is at TEST.

    The knn figures earlier in this report hold the network fixed and improve its
    context. This pair adds the other half — a network that trained on informative
    contexts — so the two factors can be separated instead of confounded.

    Q is 4 in both, matching how every network here was trained. A knn context
    hands each query M/Q neighbours, so Q changes what the context IS; the Q=1
    knn figure above is a different context and its numbers are not comparable
    to these.
    """
    nets = [("trained on iid", splitfig.load_params(RECALL)),
            ("trained on knn", splitfig.load_params(KNN_TRAINED))]
    urls = {}
    for test_ctx, headline in (
            ("iid", "Tested on sixteen UNRELATED images. Both networks, one trained each way."),
            ("knn", "Tested on the query's sixteen NEAREST NEIGHBOURS. Same two networks.")):
        urls[test_ctx] = splitfig.grid(
            f"{PROJ}_r15_ctx_{test_ctx}_v1", "fashion_to_mnist", MASK_ROWS,
            TRAIN_CLS, HELD_CLS, CFG, nets, BANDS,
            headline=headline, sub=CTX_SUB, legend=CTX_LEGEND,
            ctx_mode=test_ctx, Q=4, tile=0.80)
        # The margin the earlier figure measured, for this context type. A knn
        # context is BUILT from near neighbours, so it is by construction a
        # low-margin context; this checks that rather than assuming it.
        _, mk, evk = splitfig.build("fashion_to_mnist", MASK_ROWS, TRAIN_CLS,
                                    HELD_CLS, CFG, ctx_mode=test_ctx, Q=4)
        CTX_MARGIN[test_ctx] = float(np.median(
            splitfig.nearest_distractor(evk, mk, "B_novel_present")))
    for train_ctx, exp in (("iid", RECALL), ("knn", KNN_TRAINED)):
        params = splitfig.load_params(exp)
        for test_ctx in ("iid", "knn"):
            CTX[(train_ctx, test_ctx)] = splitfig.score_all(
                params, "fashion_to_mnist", MASK_ROWS, TRAIN_CLS, HELD_CLS, CFG,
                ctx_mode=test_ctx, Q=4)
            r = CTX[(train_ctx, test_ctx)]
            print(f"  ctx train={train_ctx} test={test_ctx}  "
                  f"D {r['D_novel_absent']['nmse']:.3f}  "
                  f"B {r['B_novel_present']['nmse']:.3f} id "
                  f"{r['B_novel_present']['id_acc']:.3f}")
    print("  ctx margins:", {k: round(v, 3) for k, v in CTX_MARGIN.items()})
    return urls


def fig_margin():
    """Why MNIST is the easier pool to retrieve FROM, with no network involved.

    Distance from each target to its nearest rival in the same context, on hidden
    pixels, in units of that pool's mean-image error. Identification flips when
    the network's output drifts far enough to cross that gap, so a pool whose
    targets sit close to their rivals is harder to retrieve from at any given
    reconstruction quality.
    """
    _, mask, ev = splitfig.build("fashion_to_mnist", MASK_ROWS, TRAIN_CLS,
                                 HELD_CLS, CFG, seed=12345)
    _, mask_f, ev_f = splitfig.build("fashion_mnist", MASK_ROWS, (0, 1, 2, 3, 4),
                                     (5, 6, 7, 8, 9), CFG, seed=12345)
    _, mask_r, ev_r = splitfig.build("mnist_to_fashion", MASK_ROWS, TRAIN_CLS,
                                     HELD_CLS, CFG, seed=12345)
    series = [
        ("MNIST test split", splitfig.BLUE,
         splitfig.nearest_distractor(ev, mask, "B_novel_present")),
        ("Fashion-MNIST test split", GREY,
         splitfig.nearest_distractor(ev, mask, "E_same_present")),
        ("Fashion-MNIST classes 5-9", splitfig.GREEN,
         splitfig.nearest_distractor(ev_f, mask_f, "B_novel_present")),
        ("Fashion-MNIST, all classes", splitfig.ORANGE,
         splitfig.nearest_distractor(ev_r, mask_r, "B_novel_present")),
    ]
    for lab, _, v in series:
        MARGINS[lab] = float(np.median(v))
        print(f"  margin {lab:28s} median {np.median(v):.3f}  "
              f"p25 {np.percentile(v, 25):.3f}")
    marks = [
        (f"exp36 error on MNIST ({n(RECALL, 'B_novel_present'):.2f})",
         splitfig.BLUE, n(RECALL, "B_novel_present")),
        (f"exp30 error on Fashion 5-9 ({n(CLASS_SPLIT, 'B_novel_present'):.2f})",
         splitfig.GREEN, n(CLASS_SPLIT, "B_novel_present")),
    ]
    return splitfig.margin_cdf(
        f"{PROJ}_r15_margin_v2", series, marks=marks,
        title="How far each target sits from its nearest rival in the context")


def main():
    nets = [("recall-trained", splitfig.load_params(RECALL)),
            ("completion-trained", splitfig.load_params(COMPLETE)),
            ("frozen layers", splitfig.load_params(FROZEN))]

    url_grid = splitfig.grid(
        f"{PROJ}_r15_f2m_grid_v2", "fashion_to_mnist", MASK_ROWS, TRAIN_CLS,
        HELD_CLS, CFG, nets, BANDS,
        headline="Context: sixteen unrelated images. Trained on Fashion-MNIST only.",
        sub=SUB, legend=LEGEND, ctx_mode="iid", Q=4, tile=0.80)
    url_knn = splitfig.grid(
        f"{PROJ}_r15_f2m_knn_v2", "fashion_to_mnist", MASK_ROWS, TRAIN_CLS,
        HELD_CLS, CFG, nets, BANDS,
        headline="Context: the query's sixteen nearest neighbours. Same three networks.",
        sub=SUB + " No network was trained on this kind of context.",
        legend=LEGEND, ctx_mode="knn", Q=1, tile=0.80)
    url_bars = splitfig.bars(
        f"{PROJ}_r15_f2m_bars_v2", rows,
        [("recall-trained", RECALL), ("completion-trained", COMPLETE),
         ("frozen layers", FROZEN)], BL,
        labels=["Fashion\nseen images", "Fashion\nnew images", "MNIST\nnew dataset"],
        nothing_label="no better than\nthe average Fashion image")
    url_margin = fig_margin()
    url_ctx = figs_context()

    _, mask, ev = splitfig.build("fashion_to_mnist", MASK_ROWS, TRAIN_CLS,
                                 HELD_CLS, CFG, seed=12345)
    ceil = {c: splitfig.oracle_id_acc(ev, mask, c)
            for c in ("A_seen_present", "E_same_present", "B_novel_present")}
    print("id ceiling:", ceil)
    for u in (url_grid, url_knn, url_bars, url_margin, *url_ctx.values()):
        print("fig:", u)

    md = f"""# A new dataset costs retrieval less than a new class

A network was trained on Fashion-MNIST and nothing else. Then it was shown
sixteen MNIST digits and asked to reproduce one of them from a half-erased copy.

It picked the right digit **{idacc(RECALL, 'B_novel_present'):.1%}** of the time.
Chance is 6.3%. It has never seen a digit.

Report 13 ran the easier-sounding test on the same architecture: train on
Fashion-MNIST classes 0 to 4, retrieve from classes 5 to 9. Same dataset, same
preprocessing, five held-out classes instead of a whole new world. That scored
**{idacc(CLASS_SPLIT, 'B_novel_present'):.1%}**.

So a network handles an entirely different dataset better than it handles five
held-out classes of its own. Whatever governs retrieval here, it is not novelty.

## The control

The obvious objection is that Fashion-MNIST is simply the richer thing to have
trained on, and a network that learned it can handle anything.

Run the crossing the other way and that dies. Train on MNIST, retrieve from
Fashion-MNIST: **{idacc(REVERSE, 'B_novel_present'):.1%}**.

| trained on | retrieving | seen it? | identification |
|---|---|---|---|
| Fashion-MNIST | MNIST | no | **{idacc(RECALL, 'B_novel_present'):.3f}** |
| Fashion 0-4 | Fashion 5-9 | no | {idacc(CLASS_SPLIT, 'B_novel_present'):.3f} |
| MNIST | Fashion-MNIST | no | **{idacc(REVERSE, 'B_novel_present'):.3f}** |
| Fashion-MNIST | Fashion-MNIST | yes | {idacc(RECALL, 'E_same_present'):.3f} |
| MNIST | MNIST | yes | {idacc(REVERSE, 'E_same_present'):.3f} |

Read the middle column, not the left one. Retrieving an MNIST digit scores high
whether or not the network trained on MNIST. Retrieving a Fashion image scores
low whether or not the network trained on Fashion.

The difficulty belongs to the pool being searched, not to the network's history
with it.

## Why

Identification asks which of the sixteen context images the output most
resembles, measured on hidden pixels only. Two separate things decide whether it
succeeds: how well the network reconstructs, and how far the target sits from
its nearest rival in that context. Only the first is about the network.

The second can be measured with no network at all.

![How far each target sits from its nearest rival]({url_margin})

The pools differ, and in the order the identification numbers need. A target in
the MNIST pool sits a median **{MARGINS['MNIST test split']:.2f}** from its
closest rival. In Fashion classes 5 to 9 that is
**{MARGINS['Fashion-MNIST classes 5-9']:.2f}**, and across all Fashion classes
**{MARGINS['Fashion-MNIST, all classes']:.2f}** — in the same units, on the same
axis as the model errors marked on the plot. A t-shirt, a pullover and a coat
have nearly the same lower half. Nothing else looks like a 4.

What makes this more than a correlation is that the two runs being compared
reconstruct about equally well. The network is slightly **worse** at rebuilding
an MNIST digit than report 13's network was at rebuilding a held-out garment —
{n(RECALL, 'B_novel_present'):.3f} against
{n(CLASS_SPLIT, 'B_novel_present'):.3f} — and identifies it far better anyway.
Reconstruction quality is held roughly fixed; the candidate geometry is what
moves, and identification moves with it.

Two honest limits on that. The reverse control has both a smaller margin **and**
a much worse reconstruction ({n(REVERSE, 'B_novel_present'):.3f}), so it is
consistent with the story but does not independently test it. And the margin
does not predict the error rate quantitatively — flipping needs the error to
point at the rival, not merely to be large enough — so this ranks pools rather
than forecasting accuracies.

An earlier version of this figure swept isotropic Gaussian noise on a perfect
reconstruction and asked where identification broke. It found nothing: all four
pools sat at 1.000 out to a noise scale of 1.1 per pixel. Independent noise over
392 pixels adds nearly the same offset to every candidate's distance, so it
barely disturbs the ranking. The measurement above replaced it.

## The task and the three networks

Each episode is sixteen complete 28x28 images, one per token, then a seventeenth
with its bottom fourteen rows erased. The network produces the missing 392
pixels.

The novel pool is a dataset rather than a class. The six conditions absorb that
without changing. The novel dataset's ten labels are shifted up by ten and the
two test splits concatenated, so `held_same` comes out as Fashion-MNIST's own
test split and `held` as MNIST's.

    A/C   Fashion-MNIST train split    images seen in training
    E/F   Fashion-MNIST test split     new images, same world
    B/D   MNIST test split             a different world

E/F is the control that separates image novelty from distribution shift. It
costs nothing: {n(RECALL, 'A_seen_present'):.3f} against
{n(RECALL, 'E_same_present'):.3f}, identification
{idacc(RECALL, 'A_seen_present'):.3f} against
{idacc(RECALL, 'E_same_present'):.3f}.

Three networks, identical in size and shape, differ only in training episodes.
**Recall-trained** always had its answer in the context. **Completion-trained**
never did. **Frozen** is the recall network with its four mixing layers held at
their random initialisation.

![Completions across three levels of novelty]({url_grid})

## Prediction crosses badly, as usual

With the answer absent, the recall-trained network scores
**{n(RECALL, 'D_novel_absent'):.3f}** on MNIST, against
{n(RECALL, 'F_same_absent'):.3f} on new Fashion images.

The reference that matters is the linear map: fit visible pixels to hidden
pixels on Fashion, ignore the context, apply to MNIST. It scores
{bl('D_novel_absent', 'n_ridge'):.3f}. The network scores
{n(RECALL, 'D_novel_absent'):.3f}. Sixteen images to look at are worth
essentially nothing.

Two cautions about that number, because the normaliser is doing work here.

1.0 means "no better than drawing the average Fashion image". On MNIST queries
that is a weak constant, so 1.0 is easier to beat than it looks. Predicting pure
black, which is a real strategy on MNIST, scores
{bl('D_novel_absent', 'n_zeros'):.3f} — worse than the Fashion mean, because the
bottom half of a digit does carry ink.

And the context did contain usable information the network ignored. The soft
look-up ceiling — the best a similarity-weighted blend of the sixteen can do,
which is the shape of computation linear attention performs — is
{bl('D_novel_absent', 'n_knn'):.3f} on MNIST, well below the network's
{n(RECALL, 'D_novel_absent'):.3f}.

![The same six blocks as numbers]({url_bars})

## The other two arms

The completion-trained network scores {n(COMPLETE, 'A_seen_present'):.3f} with
the answer present and {n(COMPLETE, 'C_seen_absent'):.3f} with it absent. Same
number: it does not read its context. That signature has now appeared in all
four domains this project has run.

On MNIST it scores **{n(COMPLETE, 'D_novel_absent'):.3f}**, worse than the
do-nothing line, and its identification drops to
{idacc(COMPLETE, 'B_novel_present'):.3f}.

The frozen network gives up retrieval — {idacc(FROZEN, 'B_novel_present'):.3f} on
MNIST against the recall network's {idacc(RECALL, 'B_novel_present'):.3f} — and
buys prediction with it: **{n(FROZEN, 'D_novel_absent'):.3f}** against
{n(RECALL, 'D_novel_absent'):.3f}, which does beat the linear map's
{bl('D_novel_absent', 'n_ridge'):.3f}.

That is the clearest present/absent trade this project has measured. The frozen
network is worse at finding and better at guessing, on the same images, in the
same run.

## A nearest-neighbour context

Replacing the sixteen unrelated images with the query's own sixteen nearest
neighbours makes the context informative. No network was trained that way.

![The same three networks on nearest-neighbour contexts]({url_knn})

The soft look-up ceiling on MNIST moves from
{bl('D_novel_absent', 'n_knn'):.3f} to
{bl('D_novel_absent', 'n_knn', BL_KNN):.3f}. Compare the red numbers here
against the first figure to see how much each network collects.

## Which context matters: the one it trained on, or the one it is given

The figure above holds the network fixed and improves its context. That leaves
the other half unasked: what if the network had been *trained* on informative
contexts to begin with?

exp40 answers it. It is exp36 with one change — during training its sixteen
context images were the query's own nearest neighbours rather than sixteen
unrelated pictures. Same data, same objective, same Q.

![Both networks on unrelated contexts]({url_ctx['iid']})

![Both networks on nearest-neighbour contexts]({url_ctx['knn']})

Completion on MNIST, with the answer absent:

| | tested on unrelated | tested on neighbours |
|---|---|---|
| **trained on unrelated** | {CTX[('iid', 'iid')]['D_novel_absent']['nmse']:.3f} | {CTX[('iid', 'knn')]['D_novel_absent']['nmse']:.3f} |
| **trained on neighbours** | {CTX[('knn', 'iid')]['D_novel_absent']['nmse']:.3f} | {CTX[('knn', 'knn')]['D_novel_absent']['nmse']:.3f} |

Read across a row and the score moves by about
{CTX[('iid', 'iid')]['D_novel_absent']['nmse'] - CTX[('iid', 'knn')]['D_novel_absent']['nmse']:.2f}.
Read down a column and it moves by about
{CTX[('iid', 'iid')]['D_novel_absent']['nmse'] - CTX[('knn', 'iid')]['D_novel_absent']['nmse']:.2f}.

Completion is governed by the context the network is handed, not by the context
it grew up on. Training on informative contexts for twelve thousand steps buys
almost nothing that being handed one at test time does not already give.

Identification behaves in the opposite way.

| | tested on unrelated | tested on neighbours |
|---|---|---|
| **trained on unrelated** | {CTX[('iid', 'iid')]['B_novel_present']['id_acc']:.3f} | {CTX[('iid', 'knn')]['B_novel_present']['id_acc']:.3f} |
| **trained on neighbours** | {CTX[('knn', 'iid')]['B_novel_present']['id_acc']:.3f} | {CTX[('knn', 'knn')]['B_novel_present']['id_acc']:.3f} |

Here training does the work. The knn-trained network is the better retriever in
both columns, including the unrelated contexts it never trained on, where it
reaches {CTX[('knn', 'iid')]['B_novel_present']['id_acc']:.3f} against
{CTX[('iid', 'iid')]['B_novel_present']['id_acc']:.3f}. Being made to tell
near-identical images apart during training produces a sharper matcher, and that
sharpness transfers to the easy case.

It also reconstructs better with the answer present:
{CTX[('knn', 'iid')]['B_novel_present']['nmse']:.3f} against
{CTX[('iid', 'iid')]['B_novel_present']['nmse']:.3f}.

## The same confound, found a second way

Look down the columns of the identification table rather than across. Both
networks identify worse on neighbour contexts than on unrelated ones —
{CTX[('iid', 'iid')]['B_novel_present']['id_acc']:.3f} to
{CTX[('iid', 'knn')]['B_novel_present']['id_acc']:.3f} for one,
{CTX[('knn', 'iid')]['B_novel_present']['id_acc']:.3f} to
{CTX[('knn', 'knn')]['B_novel_present']['id_acc']:.3f} for the other — even
though a neighbour context makes every other number better.

That is the nearest-rival effect again. A knn context is assembled *from* the
query's closest matches, so it is by construction a low-margin context. Measured
the same way as the figure above, the median distance from a target to its
nearest rival falls from **{CTX_MARGIN['iid']:.2f}** on unrelated contexts to
**{CTX_MARGIN['knn']:.2f}** on neighbour contexts.

This matters because it is an independent test. The earlier figure compared
different *pools* and found identification tracking the margin. This compares
different *contexts drawn from the same pool*, with the network held fixed, and
finds the same thing. Two unrelated manipulations, one mechanism.

It also means a neighbour context is not simply better. It makes the answer
easier to construct and harder to name.

## What this changes

Report 13 concluded that retrieval is not class-agnostic, because it degraded
from 1.000 on MNIST digits to
{idacc(CLASS_SPLIT, 'B_novel_present'):.3f} on held-out Fashion classes. That
conclusion was right about the number and wrong about the cause.

The degradation was not the network failing to generalise. It was Fashion-MNIST
being a harder pool to tell apart.

Two facts make that the better reading. Retrieving Fashion images stays hard for
a network that trained on MNIST — {idacc(REVERSE, 'B_novel_present'):.3f}, worse
than report 13's {idacc(CLASS_SPLIT, 'B_novel_present'):.3f}, not better. And
retrieving MNIST digits is easy for a network that has never seen one:
{idacc(RECALL, 'B_novel_present'):.3f}. In both crossings the score follows the
pool being searched, not what the network was trained on.

Report 12's original claim, that retrieval is content-addressed and largely
indifferent to what it has seen, survives this better than report 13's revision
of it. What has to be added is that identification accuracy is not a pure
measure of the network. It is confounded by how confusable the candidates are.
That confound is worth more than the entire class-novelty effect report 13
attributed to the network.

Two things follow. Any identification number in this project should be quoted
with its pool's nearest-rival distribution, or at least compared only within a
pool. `lib/splitfig.nearest_distractor` computes it and costs no training.

And the class-split design is a weaker instrument than it appears. Holding out
classes changes both what the network has seen and what it is searching among,
and the second effect is the larger one. Crossing datasets in both directions,
as here, separates them.

## What this does not establish

Two datasets, one direction each way. Both are 28x28 grey and centred, which is
what makes a single network scorable on both; a wider shift would need a
different instrument.

The tolerance curve uses isotropic noise, which is not what a network's error
looks like. It ranks the pools correctly but should not be read as predicting
any particular accuracy.

The frozen network's present/absent trade is one run at one size. It is the most
interesting thing here that has not been replicated.

## Sources

`results.jsonl` rows `{RECALL}`, `{COMPLETE}` and `{FROZEN}` — recall, completion
and frozen-layer training on Fashion-MNIST with MNIST as the novel pool.
`{REVERSE}` is the reverse-direction control and `{CLASS_SPLIT}` the
within-Fashion class split from report 13. `{BL}`, `{BL_KNN}` and `{BL_REV}` hold
the linear, look-up, predict-black and average-image references. Nearest-rival
distances and identification ceilings are computed in this script by
`lib/splitfig` and involve no trained network. Figures generated by `scripts/gen_report_15.py` from `lib/splitfig.py`.
No training was run for this report.
"""
    REPORT_MD_PATH.write_text(md)
    print("report:", save_report(f"{PROJ}_report_15", md))


if __name__ == "__main__":
    main()
