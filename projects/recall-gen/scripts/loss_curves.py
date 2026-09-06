"""Training curves for the four synthetic-prior runs: where the budget goes.

Each run logs its training loss and its evaluation error every 2000 steps. Put
the three side by side and the question "which part of the architecture is the
problem" becomes answerable without another run:

    loss                 how well the objective is being minimised
    A_seen_present       error on worlds the network trained on, answer present
    B_novel_present      the same on worlds from the prior it has never seen
    B - A                the generalisation gap

If the loss keeps falling while B flattens or turns upward, further training is
being spent on the training worlds and the limit is not the budget. If B tracks
the loss down, the run was still improving when it stopped.

Reads `results.jsonl` only — no checkpoints, no GPU work beyond plotting.

    .venv/bin/python projects/recall-gen/scripts/loss_curves.py
"""
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR.parents[1]))          # repo root LAST — see workflow.md
sys.path.insert(0, str(PROJECT_DIR))

from shared_lib.typst_plot import cm, line_chart, long_form
from shared_lib.typst_report import save_figure
from rescore import rows as read_rows

PROJ = "recall-gen"
V = "v1"
RUNS = [("delta rule, 4.06M", "exp45"),
        ("delta rule, 14.95M", "exp49"),
        ("delta rule, 14.94M wide-dk", "exp54"),
        ("attention, 13.88M", "exp56")]


def series(rows, key, cond=None):
    out = {}
    for lab, e in RUNS:
        h = rows[e]["history"]
        out[lab] = h[key] if cond is None else h[key][cond]
    return out


def main():
    rows = {r["experiment"]: r for r in read_rows()}
    steps = rows[RUNS[0][1]]["history"]["step"]
    for _, e in RUNS:
        assert rows[e]["history"]["step"] == steps, f"{e} logged on a different grid"

    loss = series(rows, "loss")
    A = series(rows, "nmse", "A_seen_present")
    B = series(rows, "nmse", "B_novel_present")
    gap = {k: [b - a for a, b in zip(A[k], B[k])] for k in A}

    print(f"{'run':<28} {'best B':>8} {'at step':>8} {'final B':>8} "
          f"{'B lost after':>13} {'loss at best':>13} {'final loss':>11}")
    summary = {}
    for lab, e in RUNS:
        b = np.array(B[lab])
        i = int(b.argmin())
        summary[lab] = dict(best=float(b[i]), at=steps[i], final=float(b[-1]),
                            lost=float(b[-1] - b[i]),
                            loss_at=loss[lab][i], loss_final=loss[lab][-1])
        s = summary[lab]
        print(f"{lab:<28} {s['best']:>8.4f} {s['at']:>8d} {s['final']:>8.4f} "
              f"{s['lost']:>+13.4f} {s['loss_at']:>13.5f} {s['loss_final']:>11.5f}")

    u1 = save_figure(line_chart(
        f"{PROJ}_r24_loss", long_form(steps, loss, x_name="step", y_name="loss",
                                      series_name="run"),
        x="step", y="loss", colour="run",
        title="Training loss: every run is still minimising the objective at the end",
        subtitle="Squared error on hidden coordinates, the quantity actually optimised.",
        x_label="training step", y_label="training loss",
        caption=("Attention reaches the delta rule's final loss in roughly a "
                 "seventh of the steps and keeps going."),
        width=cm(16), height=cm(9)), name=f"{PROJ}_r24_loss_{V}")

    u2 = save_figure(line_chart(
        f"{PROJ}_r24_novel", long_form(steps, B, x_name="step", y_name="nmse",
                                       series_name="run"),
        x="step", y="nmse", colour="run",
        title="Error on worlds never seen: flat from early on, and attention turns upward",
        subtitle=("Same prior, worlds held out. This is the curve the training loss "
                  "is supposed to be buying."),
        x_label="training step", y_label="normalised error, novel worlds",
        caption=("Attention bottoms out at "
                 f"{summary['attention, 13.88M']['best']:.3f} by step "
                 f"{summary['attention, 13.88M']['at']:,} and ends "
                 f"{summary['attention, 13.88M']['final']:.3f}. The last "
                 f"{steps[-1] - summary['attention, 13.88M']['at']:,} steps make it "
                 "worse."),
        width=cm(16), height=cm(9)), name=f"{PROJ}_r24_novel_{V}")

    u3 = save_figure(line_chart(
        f"{PROJ}_r24_gap", long_form(steps, gap, x_name="step", y_name="gap",
                                     series_name="run"),
        x="step", y="gap", colour="run",
        title="The generalisation gap widens for every architecture, throughout",
        subtitle="Error on novel worlds minus error on trained worlds.",
        x_label="training step", y_label="novel minus seen",
        caption=("Nothing here bends. Whatever the extra training buys, it is "
                 "specific to the worlds it was bought on."),
        width=cm(16), height=cm(9)), name=f"{PROJ}_r24_gap_{V}")

    for u in (u1, u2, u3):
        print("fig:", u)
    return summary


if __name__ == "__main__":
    main()
