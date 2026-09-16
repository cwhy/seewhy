"""Regenerates the five-epoch report: how much of the ranking is the budget?

Writes reports/02-five-epochs.md and publishes it to R2.

Run on the GPU box: `uv run --no-sync python projects/local-learning/scripts/gen_md_report_5ep.py`
"""
import statistics as st
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.append(str(PROJECT.parents[1]))          # repo root LAST — see workflow.md

from lib.figures import (FIVE_EP_ARMS, arm_cells, cells, five_epoch_curves,  # noqa: E402
                         five_epoch_rates, one_vs_five)
from shared_lib.results import load_results                                  # noqa: E402
from shared_lib.report import save_report                                    # noqa: E402
from shared_lib.typst_report import save_figure                              # noqa: E402

REPORT = PROJECT / "reports" / "02-five-epochs.md"
ROWS = load_results(PROJECT / "results.jsonl")

# one-epoch and five-epoch best rate per arm, from exp3/exp7/exp18 and exp19-21
RATES = {
    "backprop 32x32 res": (3e-2, 1e-2),
    "PC 32x32 res": (3e-3, 1e-3),
    "PC-ALM 32x32 res": (1e-2, 1e-2),
    "EGGROLL 32x32 res": (3e-2, 3e-2),
    "Forward-Forward 2x66": (1e-3, 3e-4),
    "backprop 2x66": (3e-3, 1e-3),
    "EGGROLL 2x66": (1e-2, 1e-3),
    "Forward-Forward 31x32": (3e-3, 1e-3),
}


def stat(rows, field="test_acc"):
    v = [r[field] for r in rows if r.get(field) is not None]
    if not v:
        return float("nan"), 0.0
    return st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0)


def pm(rows, field="test_acc"):
    return "{:.4f} ± {:.4f}".format(*stat(rows, field))


def main():
    urls = {name: save_figure(fig) for name, fig in (
        ("compare", one_vs_five(ROWS)),
        ("curves", five_epoch_curves(ROWS)),
        ("rates", five_epoch_rates(ROWS)),
    )}

    md = """# Five epochs, not one — does the budget hold the ranking up?

Follow-up to the [replication report](https://media.tanh.xyz/seewhy/26-09-16/local-learning_report_replication.html)
and its [paper](https://media.tanh.xyz/seewhy/paper/local-learning_paper.html).
Everything there runs for **one epoch**, because that is the budget of the table
being replicated and of the PC-ALM paper's grid. One epoch is 937 weight updates
at batch 64, which is little enough that it could be holding the whole ranking
up. This checks at five.

Chance is 0.10. Three seeds per arm, one RTX 4090 (shared with another job —
accuracies are unaffected, see the paper's §6.8).

## The rate has to be re-tuned first

A rate chosen for 937 updates is not the rate for 4,685, and treating it as one
would favour whichever arm is least sensitive to the difference. So the whole
grid was re-run at five epochs (exp19), and every peak that landed at an edge
was bracketed (exp20, exp21).

**Six of the eight arms want a different rate at five epochs than at one**:

| arm | best rate, 1 epoch | best rate, 5 epochs |
|---|---:|---:|
"""
    for label, (one_lr, five_lr) in RATES.items():
        moved = "" if one_lr == five_lr else "  ←"
        md += f"| {label} | {one_lr:g} | {five_lr:g}{moved} |\n"

    md += f"""
Only PC-ALM and EGGROLL on the residual net keep theirs. Everything else wants a
smaller step, which is the ordinary thing to expect and exactly why this had to
be checked rather than assumed.

![five-epoch rate sweep]({urls['rates']})

## The comparison at five epochs

Each arm at the rate best for *that* budget, three seeds:

| arm | 1 epoch | 5 epochs | change |
|---|---|---|---:|
"""
    for label, one_loc, five_loc in FIVE_EP_ARMS:
        one, five = arm_cells(ROWS, one_loc), arm_cells(ROWS, five_loc)
        if not one or not five:
            continue
        md += (f"| {label} | {pm(one)} | {pm(five)} | "
               f"{stat(five)[0] - stat(one)[0]:+.4f} |\n")

    md += f"""
![one epoch against five]({urls['compare']})

![per-epoch curves]({urls['curves']})

## What changes, and what does not

"""
    def acc(loc):
        return stat(arm_cells(ROWS, loc))[0]

    order = sorted(((acc(five), label) for label, _, five in FIVE_EP_ARMS
                    if arm_cells(ROWS, five)), reverse=True)
    md += "Ranking at five epochs:\n\n"
    for i, (a, label) in enumerate(order, 1):
        md += f"{i}. {label} — {a:.4f}\n"

    ff = arm_cells(ROWS, ("exp22", "ff", "5ep"))
    ff1 = arm_cells(ROWS, ("exp9", "ff", "tuned"))
    bp = arm_cells(ROWS, ("exp22", "bp", "5ep"))
    eg = arm_cells(ROWS, ("exp22", "eggroll", "5ep"))
    al = arm_cells(ROWS, ("exp22", "pcalm", "5ep"))
    pc = arm_cells(ROWS, ("exp22", "pc", "5ep"))

    md += f"""
**Backprop pulls further ahead.** {stat(bp)[0]:.4f} at five epochs against
{acc(('exp9','bp','tuned')):.4f} at one, and the gap to the best local rule widens.

**PC-ALM stays the best local rule** at {stat(al)[0]:.4f}, still tracking backprop
but no longer closing. PC is at {stat(pc)[0]:.4f}, so the dual variables are worth
{100*(stat(al)[0]-stat(pc)[0]):.0f} points at this budget against 25 at one epoch — the gap narrows
because PC gains more from the extra epochs than PC-ALM does ({100*(stat(pc)[0]-acc(("exp9","pc","tuned"))):+.1f} against
{100*(stat(al)[0]-acc(("exp9","pcalm","tuned"))):+.1f}), which is what you would expect of a method that is still
far from its own ceiling.

**Forward-Forward does not decay after all — at the right rate.** The one-epoch
report found it peaking at 0.901 after two epochs and falling to 0.860 by ten,
and read that as the local objective diverging from the classification task.
That was mostly a rate effect: at 1e-3 it decays, and at its five-epoch rate of
3e-4 the curve climbs and then sits flat — 0.597, 0.806, 0.902, 0.905, 0.901.
It ends at {stat(ff)[0]:.4f} against {stat(ff1)[0]:.4f} at one epoch, the
smallest gain of any arm that works, and it drops below both EGGROLL arms.

**Forward-Forward on the deep stack is the one arm that gets worse**, from
{acc(("exp9","ff","tuned_deep")):.4f} to {stat(arm_cells(ROWS, ("exp22","ff","5ep_deep")))[0]:.4f}. Its goodness readout does not survive the
extra training even at the rate chosen for five epochs.

**EGGROLL keeps up.** {stat(eg)[0]:.4f} on the residual net — a gradient-free
method within {100*(stat(bp)[0]-stat(eg)[0]):.1f} points of backprop — and its plain-network arm gains
the most of anything here ({100*(stat(arm_cells(ROWS, ("exp22","eggroll_plain","5ep")))[0]-acc(("exp16","eggroll_plain","tuned"))):+.1f} points), because one epoch was the budget
that flattered it least. The cost is the catch: five EGGROLL epochs take
{stat(eg,"train_s")[0]:.1f} s against backprop's {stat(bp,"train_s")[0]:.2f} s on the same GPU.

## What this says about the original table

Three orderings of the same three arms:

| | backprop | PC-ALM | Forward-Forward |
|---|---:|---:|---:|
| 1 epoch, shared rate (the posted table) | 0.8704 | 0.8533 | **0.8869** |
| 1 epoch, per-arm rate | **0.9374** | 0.9277 | 0.8869 |
| 5 epochs, per-arm rate | **0.9620** | 0.9405 | 0.9045 |

The budget is not what made Forward-Forward look good — the shared learning rate
was, and that was already clear at one epoch. What the budget did was keep the
arms close enough for the question to look interesting: the backprop-to-FF gap
is 1.3 points the wrong way in the posted table, 5.1 points the right way once
each arm gets its own rate, and 5.8 points at five epochs. Nothing here reverses
between one epoch and five; the spread just widens, and the arm that gains least
from more training is the one the original table put first.

---

Rows: `results.jsonl`, exp19 (24 cells), exp20 (5), exp21 (2), exp22 (24).
"""
    REPORT.write_text(md)
    url = save_report("local-learning_report_five_epochs", md)
    print(f"wrote {REPORT}")
    print(f"published {url}")


if __name__ == "__main__":
    main()
