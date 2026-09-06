"""Report 14: the class-split experiment on chess positions.

Evaluation and figures only for exp33/34/35, which are already trained. Shares
`lib/splitfig.py` with report 13, so the Fashion-MNIST and chess versions of the
six-block figure are literally the same figure with a different renderer.

Run on the GPU box:
    uv run --no-sync python projects/recall-gen/scripts/gen_report_14.py
"""
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR.parents[1]))
sys.path.insert(0, str(PROJECT_DIR))

from lib.core import Cfg
from lib import splitfig
from shared_lib.report import save_report

PROJ = "recall-gen"
REPORT_MD_PATH = PROJECT_DIR / "reports" / "14-chess-phase-split.md"
CFG = Cfg(d_model=256, n_layers=4, dk=64, n_heads=4, n_tokens=20, d_in=832)
TRAIN_CLS, HELD_CLS = (0,), (2,)
MASK_FILES = 4

rows = {}
for line in open(PROJECT_DIR / "results.jsonl"):
    r = json.loads(line)
    rows[r["experiment"]] = r

RECALL, COMPLETE, FROZEN = "exp33", "exp34", "exp35"
BL = "baselines_chess_M16_r4_split"
BL_KNN = "baselines_chess_M16_r4_split_knn_Q1"


def n(exp, cond):
    return rows[exp]["final"][cond]["nmse"]


def idacc(exp, cond):
    return rows[exp]["final"][cond]["id_acc"]


def pc(exp, cond):
    """Piece accuracy as a percentage: hidden squares whose top piece is right."""
    return 100.0 * rows[exp]["final"][cond]["sq_acc"]


def empty(cond):
    return 100.0 * rows[BL]["baselines"][cond]["sq_acc_empty"]


def bl(cond, key):
    return rows[BL]["baselines"][cond][key]


BANDS = ["24 or more pieces\npositions seen in training",
         "24 or more pieces\npositions never seen",
         "endgames, 10 pieces or fewer\nnever seen at all"]

LEGEND = [
    ("recall-trained: during training its answer was ALWAYS one of the sixteen context positions, so copying always worked.",
     splitfig.BLUE),
    ("completion-trained: during training its answer was NEVER in the context, so it could only ever predict.",
     splitfig.ORANGE),
    ("frozen layers: trained like the recall network, but its four mixing layers keep their random starting values.",
     splitfig.GREEN),
]

SUB = ("Files a-d are hidden and must be reconstructed; files e-h are given. Glyph opacity is the network's confidence.\n"
       "RED is that network's score over all 512 episodes: normalised error, where 1.00 is the average board, then the share of hidden squares whose piece is right.")


def main():
    nets = [("recall-trained", splitfig.load_params(RECALL)),
            ("completion-trained", splitfig.load_params(COMPLETE)),
            ("frozen layers", splitfig.load_params(FROZEN))]

    url_grid = splitfig.grid(
        f"{PROJ}_r14_chess_grid_v5", "chess", MASK_FILES, TRAIN_CLS, HELD_CLS,
        CFG, nets, BANDS,
        headline="Context: sixteen unrelated positions. Trained on positions with 24+ pieces.",
        sub=SUB, legend=LEGEND, ctx_mode="iid", Q=4, tile=1.15, lab_w=2.10)
    url_knn = splitfig.grid(
        f"{PROJ}_r14_chess_knn_v5", "chess", MASK_FILES, TRAIN_CLS, HELD_CLS,
        CFG, nets, BANDS,
        headline="Context: the query's sixteen nearest positions. Same three networks.",
        sub=SUB + " No network was trained on this kind of context.",
        legend=LEGEND, ctx_mode="knn", Q=1, tile=1.15, lab_w=2.10)
    url_bars = splitfig.bars(
        f"{PROJ}_r14_chess_bars_v5", rows,
        [("recall-trained", RECALL), ("completion-trained", COMPLETE),
         ("frozen layers", FROZEN)], BL,
        labels=["24+ pieces\nseen positions", "24+ pieces\nnew positions",
                "endgames\nnew phase"],
        ylim=2.0, nothing_label="no better than\nthe average board")

    _, mask, ev = splitfig.build("chess", MASK_FILES, TRAIN_CLS, HELD_CLS,
                                 CFG, seed=12345)
    ceil = {c: splitfig.oracle_id_acc(ev, mask, c)
            for c in ("A_seen_present", "E_same_present", "B_novel_present")}
    print("id ceiling:", ceil)
    print("grid:", url_grid); print("knn:", url_knn); print("bars:", url_bars)

    md = f"""# Chess recall is exact. Prediction is worse than nothing.

A network was shown sixteen chess positions and then a seventeenth with its
queenside erased. When the seventeenth was one of the sixteen, it rebuilt the
missing half of the board with **{pc(RECALL, 'A_seen_present'):.1f}%** of hidden
squares carrying the right piece. Guessing that every hidden square is empty
gets {empty('A_seen_present'):.1f}%.

When the seventeenth was not among the sixteen, the same network scored
**{pc(RECALL, 'C_seen_absent'):.1f}%** — against that same
{empty('C_seen_absent'):.1f}% for guessing empty. In normalised squared error it
scored **{n(RECALL, 'C_seen_absent'):.3f}**, where 1.0 is the score for drawing
the average training board and ignoring the input. A linear map fitted to
predict the queenside from the kingside, which never sees the context at all,
scores {bl('C_seen_absent', 'n_ridge'):.3f}.

So the network is worse than doing nothing, on positions drawn from the very
distribution it trained on.

These are the same weights, the same evaluation run, the same positions. Only
whether the answer was in the context differs.

## What a token is

One token is one position, encoded as piece planes: 8 ranks x 8 files x 13
channels, one channel per piece type plus one for an empty square. That is 832
numbers, against MNIST's 784 pixels. Side to move, castling rights and the
en-passant square are not encoded — this is the piece representation, and a
completion task over hidden squares has no use for them.

The query hides files a to d, the queenside, and shows files e to h. Half the
board, like MNIST's bottom half.

Positions come from the Lichess Stockfish evaluation database, 37.16 million
positions, sampled uniformly at random. Uniform sampling matters: consecutive
rows are consecutive plies of one game, so a contiguous block would be a few
hundred games in move order rather than a pool.

The novelty axis is game phase. The training pool is positions with 24 or more
pieces on the board. The novel pool is endgames, 10 pieces or fewer. The middle
bucket is in neither, so the two ends are far apart rather than adjacent. The
train/test cut is a row-id threshold, so no game contributes positions to both.

Three networks, identical in size and shape, differing only in their training
episodes — recall-trained, completion-trained, and a frozen version whose four
mixing layers keep their random initialisation. All three saw only 24+ piece
positions.

![Completions across three levels of novelty]({url_grid})

## Retrieval survives the phase change, damaged

Read the left column of the figure, where the answer is present.

A position the network trained on scores **{n(RECALL, 'A_seen_present'):.3f}**.
A position it has never seen, from the same phase, scores
**{n(RECALL, 'E_same_present'):.3f}**. Identification accuracy — does the output
most resemble the correct one of the sixteen — is
{idacc(RECALL, 'A_seen_present'):.3f} and {idacc(RECALL, 'E_same_present'):.3f},
against a ceiling of {ceil['E_same_present']:.3f} and chance of 0.063. A new
position costs nothing at all.

An endgame scores **{n(RECALL, 'B_novel_present'):.3f}**, ten times worse, and
identification falls to **{idacc(RECALL, 'B_novel_present'):.3f}**.

That last number needs its ceiling stated, because on sparse boards it can be
misleading. Several endgame positions in one context can have identical, nearly
empty queensides. A network that rebuilds the target exactly still loses that
tie-break. Feeding the true answer in as the prediction scores
{ceil['B_novel_present']:.3f}, not 1.000. So about
{100 * (1 - ceil['B_novel_present']):.0f}% of these episodes are genuinely
ambiguous. The network's {idacc(RECALL, 'B_novel_present'):.3f} is still well
below the {ceil['B_novel_present']:.3f} that is available, so the deficit is
real.

In piece terms endgame retrieval still looks strong:
**{pc(RECALL, 'B_novel_present'):.1f}%** of hidden squares right. But guessing
empty on an endgame queenside already gets {empty('B_novel_present'):.1f}%. That
is the trap this domain sets, and the next section is about it.

![The same six blocks as numbers]({url_bars})

## Two metrics, opposite verdicts

On endgames with the answer absent the recall-trained network scores
**{n(RECALL, 'D_novel_absent'):.3f}** in normalised error. That is below 1.0. It
reads as better than drawing the average board.

On the same episodes it gets **{pc(RECALL, 'D_novel_absent'):.1f}%** of hidden
squares right. Guessing that the whole queenside is empty gets
**{empty('D_novel_absent'):.1f}%**.

Both numbers are correct. They disagree because squared error over one-hot piece
planes rewards spreading probability across plausible pieces. An endgame
queenside is mostly empty, so hedging scores well. Counting pieces does not
reward hedging.

The piece count is the metric that means something here. Read that way, the
network is worse than assuming nothing is there.

The disagreement does not arise on the trained phase, where the board is
crowded: **{n(RECALL, 'C_seen_absent'):.3f}** normalised error and
**{pc(RECALL, 'C_seen_absent'):.1f}%** of pieces right against
{empty('C_seen_absent'):.1f}% for empty. Both say the network is barely above the
trivial guess, and the normalised error says it is below the do-nothing line.

## The look-up references

Two model-free references bound what the context alone can give, and they move
in opposite directions across the phase boundary.

Copying the single context position whose kingside is closest scores
{bl('C_seen_absent', 'n_nn1'):.3f} on the trained phase and
{bl('D_novel_absent', 'n_nn1'):.3f} on endgames. A similarity-weighted blend of
all sixteen — which is the shape of computation linear attention can actually
perform — scores {bl('C_seen_absent', 'n_knn'):.3f} and
{bl('D_novel_absent', 'n_knn'):.3f}.

On endgames the blend is genuinely strong at {bl('D_novel_absent', 'n_knn'):.3f},
far better than the network's {n(RECALL, 'D_novel_absent'):.3f}. Sixteen sparse
boards averaged together is a good guess at a seventeenth sparse board. The
network does not find that strategy.

On the trained phase the blend is {bl('C_seen_absent', 'n_knn'):.3f} — no better
than the average board. Sixteen unrelated crowded positions say nothing about a
seventeenth. There is nothing in the context to use, and the network's
{n(RECALL, 'C_seen_absent'):.3f} is worse than admitting that.

## The completion arm collapses

The completion-trained network was never allowed to copy. It should be the one
that predicts.

It scores **{n(COMPLETE, 'A_seen_present'):.3f}** with the answer present and
**{n(COMPLETE, 'C_seen_absent'):.3f}** with the answer absent. Those are the same
number, to three decimals. It does not read its context, exactly as its MNIST and
Fashion-MNIST counterparts do not.

What it does instead is memorise. On positions from the same phase that it has
never seen it scores **{n(COMPLETE, 'E_same_present'):.3f}** — worse than the
average board. Its best score on unseen endgames over the whole of training came
at step {rows[COMPLETE]['best_step']} of {rows[COMPLETE]['steps']}, and its final
endgame score is **{n(COMPLETE, 'D_novel_absent'):.3f}**.

Its retrieval is gone too: identification {idacc(COMPLETE, 'B_novel_present'):.3f}
on endgames, against chance of 0.063 and a ceiling of
{ceil['B_novel_present']:.3f}.

## Freezing helps prediction slightly, costs retrieval heavily

The frozen network scores **{n(FROZEN, 'C_seen_absent'):.3f}** on the trained
phase with the answer absent, against the recall network's
{n(RECALL, 'C_seen_absent'):.3f}. Better, still above the do-nothing line of 1.0,
and still worse than the linear map's {bl('C_seen_absent', 'n_ridge'):.3f}.

Its retrieval on the trained phase is nearly intact: identification
{idacc(FROZEN, 'E_same_present'):.3f} against {idacc(RECALL, 'E_same_present'):.3f}.
On endgames it is not: **{idacc(FROZEN, 'B_novel_present'):.3f}** against
{idacc(RECALL, 'B_novel_present'):.3f}, with the ceiling at
{ceil['B_novel_present']:.3f}.

## A nearest-neighbour context

Everything above uses sixteen unrelated positions. Replace them with the query's
own sixteen nearest positions, ranked by kingside similarity, and the context
becomes informative. No network was trained that way; this is a transfer test.

![The same three networks on nearest-position contexts]({url_knn})

The blend ceiling on the trained phase moves from
{bl('C_seen_absent', 'n_knn'):.3f} to
{rows[BL_KNN]['baselines']['C_seen_absent']['n_knn']:.3f}. On endgames it does
not move at all: {bl('D_novel_absent', 'n_knn'):.3f} to
{rows[BL_KNN]['baselines']['D_novel_absent']['n_knn']:.3f}, because unrelated
endgames were already nearly as useful as similar ones.

Compare the red numbers here against the first figure to see how much each
network collects. The improvement available is much smaller than it was on
MNIST, where a neighbour context took the recall network from 1.009 to 0.686.

## What this changes

Report 12 said recall training does not buy completion, it spends it. On chess
that is an understatement. The recall-trained network's completion lands **below**
the do-nothing baseline on the distribution it trained on:
{n(RECALL, 'C_seen_absent'):.3f} against 1.0, and against
{bl('C_seen_absent', 'n_ridge'):.3f} for a linear map with no context.

Three concrete consequences.

Any further chess work in this project must report piece accuracy alongside
normalised error. The two disagree in sign on endgames, and normalised error is
the one that flatters. `lib/train.py` now logs `sq_acc` for board domains for
this reason.

The identification metric needs its ceiling reported whenever items can be
indistinguishable on hidden coordinates. On endgames that ceiling is
{ceil['B_novel_present']:.3f}, not 1.000, and treating 1.000 as the reference
would have overstated the deficit by about a fifth of it.

Report 13's Fashion-MNIST result and this one disagree about which capability
survives a class change, and MNIST disagrees with both. Three domains, three
answers. The pattern report 12 described is not a property of the mechanism.

For the next run: hide two files instead of four. A four-file hole removes both
queenside rooks, both knights, both bishops and the queen, which is most of what
a position carries. Two files would say whether the failure is about prediction
or about how much was taken away.

## What this does not establish

Piece placement only. Side to move and castling rights are not in the encoding,
so the network cannot know whose move it is, and some queenside reconstructions
are genuinely underdetermined without that.

One mask and one phase split. Files a-d against e-h is a single choice, and 24+
pieces against 10 or fewer is another.

About 0.2% of sampled positions are exact duplicates of another position in the
same pool, so a small fraction of answer-absent episodes contain a duplicate of
the answer. That rate is below the noise on every number quoted here.

## Sources

`results.jsonl` rows `{RECALL}`, `{COMPLETE}` and `{FROZEN}` (recall, completion
and frozen-layer training on 24+ piece positions), and `{BL}` / `{BL_KNN}` for the
linear, look-up, average-board and all-empty references. Positions come from
`shared_lib.datasets.load_chess_positions`, encoded by `fen_to_planes`.
Identification ceilings are computed in this script by scoring the true target
against its own context. Figures generated by `scripts/gen_report_14.py` from
`lib/splitfig.py`. No training was run for this report.
"""
    REPORT_MD_PATH.write_text(md)
    print("report:", save_report(f"{PROJ}_report_14", md))


if __name__ == "__main__":
    main()
