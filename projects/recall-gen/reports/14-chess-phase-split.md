# Chess recall is exact. Prediction is worse than nothing.

A network was shown sixteen chess positions and then a seventeenth with its
queenside erased. When the seventeenth was one of the sixteen, it rebuilt the
missing half of the board with **99.4%** of hidden
squares carrying the right piece. Guessing that every hidden square is empty
gets 57.2%.

When the seventeenth was not among the sixteen, the same network scored
**64.3%** — against that same
57.2% for guessing empty. In normalised squared error it
scored **1.335**, where 1.0 is the score for drawing
the average training board and ignoring the input. A linear map fitted to
predict the queenside from the kingside, which never sees the context at all,
scores 0.864.

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

![Completions across three levels of novelty](https://media.tanh.xyz/seewhy/26-08-25/recall-gen_r14_chess_grid_v5.png)

## Retrieval survives the phase change, damaged

Read the left column of the figure, where the answer is present.

A position the network trained on scores **0.017**.
A position it has never seen, from the same phase, scores
**0.019**. Identification accuracy — does the output
most resemble the correct one of the sixteen — is
1.000 and 1.000,
against a ceiling of 1.000 and chance of 0.063. A new
position costs nothing at all.

An endgame scores **0.180**, ten times worse, and
identification falls to **0.705**.

That last number needs its ceiling stated, because on sparse boards it can be
misleading. Several endgame positions in one context can have identical, nearly
empty queensides. A network that rebuilds the target exactly still loses that
tie-break. Feeding the true answer in as the prediction scores
0.955, not 1.000. So about
5% of these episodes are genuinely
ambiguous. The network's 0.705 is still well
below the 0.955 that is available, so the deficit is
real.

In piece terms endgame retrieval still looks strong:
**94.6%** of hidden squares right. But guessing
empty on an endgame queenside already gets 90.6%. That
is the trap this domain sets, and the next section is about it.

![The same six blocks as numbers](https://media.tanh.xyz/seewhy/26-08-26/recall-gen_r14_chess_bars_v5.svg)

## Two metrics, opposite verdicts

On endgames with the answer absent the recall-trained network scores
**0.652** in normalised error. That is below 1.0. It
reads as better than drawing the average board.

On the same episodes it gets **82.9%** of hidden
squares right. Guessing that the whole queenside is empty gets
**90.6%**.

Both numbers are correct. They disagree because squared error over one-hot piece
planes rewards spreading probability across plausible pieces. An endgame
queenside is mostly empty, so hedging scores well. Counting pieces does not
reward hedging.

The piece count is the metric that means something here. Read that way, the
network is worse than assuming nothing is there.

The disagreement does not arise on the trained phase, where the board is
crowded: **1.335** normalised error and
**64.3%** of pieces right against
57.2% for empty. Both say the network is barely above the
trivial guess, and the normalised error says it is below the do-nothing line.

## The look-up references

Two model-free references bound what the context alone can give, and they move
in opposite directions across the phase boundary.

Copying the single context position whose kingside is closest scores
1.895 on the trained phase and
0.845 on endgames. A similarity-weighted blend of
all sixteen — which is the shape of computation linear attention can actually
perform — scores 1.059 and
0.384.

On endgames the blend is genuinely strong at 0.384,
far better than the network's 0.652. Sixteen sparse
boards averaged together is a good guess at a seventeenth sparse board. The
network does not find that strategy.

On the trained phase the blend is 1.059 — no better
than the average board. Sixteen unrelated crowded positions say nothing about a
seventeenth. There is nothing in the context to use, and the network's
1.335 is worse than admitting that.

## The completion arm collapses

The completion-trained network was never allowed to copy. It should be the one
that predicts.

It scores **0.232** with the answer present and
**0.232** with the answer absent. Those are the same
number, to three decimals. It does not read its context, exactly as its MNIST and
Fashion-MNIST counterparts do not.

What it does instead is memorise. On positions from the same phase that it has
never seen it scores **1.332** — worse than the
average board. Its best score on unseen endgames over the whole of training came
at step 500 of 12000, and its final
endgame score is **1.888**.

Its retrieval is gone too: identification 0.076
on endgames, against chance of 0.063 and a ceiling of
0.955.

## Freezing helps prediction slightly, costs retrieval heavily

The frozen network scores **1.176** on the trained
phase with the answer absent, against the recall network's
1.335. Better, still above the do-nothing line of 1.0,
and still worse than the linear map's 0.864.

Its retrieval on the trained phase is nearly intact: identification
0.994 against 1.000.
On endgames it is not: **0.271** against
0.705, with the ceiling at
0.955.

## A nearest-neighbour context

Everything above uses sixteen unrelated positions. Replace them with the query's
own sixteen nearest positions, ranked by kingside similarity, and the context
becomes informative. No network was trained that way; this is a transfer test.

![The same three networks on nearest-position contexts](https://media.tanh.xyz/seewhy/26-08-26/recall-gen_r14_chess_knn_v5.png)

The blend ceiling on the trained phase moves from
1.059 to
0.921. On endgames it does
not move at all: 0.384 to
0.384, because unrelated
endgames were already nearly as useful as similar ones.

Compare the red numbers here against the first figure to see how much each
network collects. The improvement available is much smaller than it was on
MNIST, where a neighbour context took the recall network from 1.009 to 0.686.

## What this changes

Report 12 said recall training does not buy completion, it spends it. On chess
that is an understatement. The recall-trained network's completion lands **below**
the do-nothing baseline on the distribution it trained on:
1.335 against 1.0, and against
0.864 for a linear map with no context.

Three concrete consequences.

Any further chess work in this project must report piece accuracy alongside
normalised error. The two disagree in sign on endgames, and normalised error is
the one that flatters. `lib/train.py` now logs `sq_acc` for board domains for
this reason.

The identification metric needs its ceiling reported whenever items can be
indistinguishable on hidden coordinates. On endgames that ceiling is
0.955, not 1.000, and treating 1.000 as the reference
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

`results.jsonl` rows `exp33`, `exp34` and `exp35` (recall, completion
and frozen-layer training on 24+ piece positions), and `baselines_chess_M16_r4_split` / `baselines_chess_M16_r4_split_knn_Q1` for the
linear, look-up, average-board and all-empty references. Positions come from
`shared_lib.datasets.load_chess_positions`, encoded by `fen_to_planes`.
Identification ceilings are computed in this script by scoring the true target
against its own context. Figures generated by `scripts/gen_report_14.py` from
`lib/splitfig.py`. No training was run for this report.
