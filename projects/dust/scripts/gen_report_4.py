"""Report 4 -- the 1M-token ladder: DUST against EGGROLL, population by population.

    CUDA_VISIBLE_DEVICES= ~/.local/bin/uv run --no-sync python projects/dust/scripts/gen_report_4.py
"""

import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.append(str(PROJECT.parents[1]))

from lib import paper_ref as ref, results as R
from lib.figures import cost, curves, ladder
from shared_lib.report import save_report
from shared_lib.typst_report import save_figure

SUFFIX = "_d"
rows = R.load()
cell = lambda m, p: R.cell(rows, method=m, budget="1M", pop=p)
bp = R.cell(rows, method="bp", budget="1M")
UNI = ref.UNIGRAM_BY_BUDGET["1M"]
POPS = [p for p in (64, 256, 1024, 4096) if cell("dust", p) or cell("eggroll", p)]


def ms(c):
    if c is None:
        return "–"
    return f"{c['mean']:.3f} ± {c['sd']:.3f}" if c["n"] > 1 else f"{c['mean']:.3f} (n=1)"


def pub(m, p):
    v = ref.TABLE1[m].get(p, {}).get("1M")
    return "–" if v is None else f"{v:.3f}"


d64 = cell("dust", 64)
e_top = max(p for p in (256, 1024, 4096) if cell("eggroll", p))
e_top_c = cell("eggroll", e_top)

fig_ladder = save_figure(ladder(rows, "1M", name=f"ladder_1M{SUFFIX}"), fmt="svg")
fig_cost = save_figure(cost(rows, "1M", name=f"cost_1M{SUFFIX}"), fmt="svg")
cv = curves(rows, "1M", name=f"curves_1M{SUFFIX}")
fig_curves = save_figure(cv, fmt="svg") if cv else None

tab = ("| population | DUST (ours) | DUST (paper) | EGGROLL (ours) | EGGROLL (paper) | gap, ours | gap, paper |\n"
       "|---:|---|---:|---|---:|---:|---:|\n")
for p in POPS:
    d, e = cell("dust", p), cell("eggroll", p)
    gap = "–" if not (d and e) else f"{e['mean'] - d['mean']:.2f}"
    pgap = "–" if p not in ref.TABLE1["dust"] else f"{ref.TABLE1['eggroll'][p]['1M'] - ref.TABLE1['dust'][p]['1M']:.2f}"
    tab += f"| {p:,} | {ms(d)} | {pub('dust', p)} | {ms(e)} | {pub('eggroll', p)} | {gap} | {pgap} |\n"

both = [p for p in POPS if cell("dust", p) and cell("eggroll", p) and p in ref.TABLE1["dust"]]
wider = all(cell("eggroll", p)["mean"] - cell("dust", p)["mean"] > ref.TABLE1["eggroll"][p]["1M"] - ref.TABLE1["dust"][p]["1M"] for p in both)
wider_txt = "every gap in the last two columns is wider here than in the paper." if wider else "compare the last two columns."
egg_gap = {p: cell("eggroll", p)["mean"] - ref.TABLE1["eggroll"][p]["1M"] for p in POPS if cell("eggroll", p) and p in ref.TABLE1["eggroll"]}

tab2 = "| population | DUST s/update | EGGROLL s/update | EGGROLL ÷ DUST |\n|---:|---:|---:|---:|\n"
for p in POPS:
    d, e = cell("dust", p), cell("eggroll", p)
    if d and e:
        tab2 += f"| {p:,} | {d['step_s']:.1f} | {e['step_s']:.1f} | {e['step_s'] / d['step_s']:.2f} |\n"
tab2 += f"| backprop | {bp['step_s']:.2f} | | |\n"

tab3 = "| population | DUST − backprop | backprop sd | DUST sd |\n|---:|---:|---:|---:|\n"
for p in POPS:
    d = cell("dust", p)
    if d:
        tab3 += f"| {p:,} | {d['mean'] - bp['mean']:+.3f} | {bp['sd']:.3f} | {('%.3f' % d['sd']) if d['n'] > 1 else 'one seed'} |\n"
d4k = cell("dust", 4096)
bp_txt = ("DUST closes on backprop as population grows and is level with it at 1,024 draws"
          + (f"; at 4,096 draws, one seed, it is {bp['mean'] - d4k['mean']:.3f} below it, against a backprop spread of {bp['sd']:.3f}. "
             f"The paper has DUST below backprop from 1,024 draws up ({ref.TABLE1['bp']['1M'] - ref.TABLE1['dust'][4096]['1M']:.3f} below at 4,096)."
             if d4k else ". The paper has it slightly *below*."))


def line_multiple(pops, losses, target):
    x, y = np.log2(np.asarray(pops, float)), np.asarray(losses, float)
    slope, icept = np.polyfit(x, y, 1)
    return (float(2 ** ((target - icept) / slope) / 64) if slope < 0 else None), float(slope)


ex = next((r for r in rows if r.get("kind") == "extrapolation"), None)
ex_txt = ""
if ex:
    f = ex["fits"]
    pops4, loss4 = ex["eggroll_pops"], ex["eggroll_loss"]
    early_m3, _ = line_multiple(pops4[:3], loss4[:3], ex["target"])
    s_early = (loss4[0] - loss4[2]) / 4
    paper_early = (ref.TABLE1["eggroll"][64]["1M"] - ref.TABLE1["eggroll"][1024]["1M"]) / 4
    have4 = len(pops4) == 4
    s_tail = (loss4[2] - loss4[3]) / 2 if have4 else None
    paper_tail = (ref.TABLE1["eggroll"][1024]["1M"] - ref.TABLE1["eggroll"][4096]["1M"]) / 2
    pl = f.get("power_law")

    def mult(k):
        return "–" if f[k]["multiple_of_64"] is None else f"{f[k]['multiple_of_64']:,.0f}×"

    ex_txt = (
        f"\n## How much population would EGGROLL need?\n\nThe paper extrapolates EGGROLL's ladder to the population at which it would reach "
        f"DUST's loss at 64 draws ({ex['target']:.3f} here) and reports it as a multiple of 64 (its Table 2). Same fits, on our ladder "
        f"({', '.join(f'{p:,}' for p in pops4)} passes).\n\n"
        "| fit | ours | paper |\n|---|---:|---:|\n"
        f"| straight line in log population, top three | {mult('line_top3')} | 8,200× |\n"
        f"| straight line, top two | {mult('line_top2')} | 6,000× |\n"
        + (f"| power law `A·P^-α + E`, all four points | {'never (floor ' + format(pl['E'], '.2f') + ' is above the target)' if pl['multiple_of_64'] is None else format(pl['multiple_of_64'], ',.0f') + '×'} | 23,000× |\n"
           if pl else "")
        + f"\nThe fits spread, as they do in the paper, and the order of magnitude is the paper's: thousands to tens of thousands of times DUST's "
          f"64 draws, or never if the ladder flattens at {pl['E']:.2f}" + (".\n" if pl else ".\n")
        + (f"\nBefore the 4,096-pass rung we had three points, and the line through them gave {early_m3:,.0f}×. The ladder curves: our EGGROLL gained "
           f"{s_early:.3f} nats per doubling of population from 64 to 1,024 passes against the paper's {paper_early:.3f}, and "
           f"{s_tail:.3f} from 1,024 to 4,096 against {paper_tail:.3f}. It is weaker than the paper's where the population is small and "
           "the same where it is large. A line through the first three points was too optimistic about EGGROLL by a factor of "
           f"{(f['line_top3']['multiple_of_64'] / early_m3):.0f}. The 4,096 points are single seeds.\n" if have4 else ""))

md = f"""# 64 DUST draws beat {e_top:,} EGGROLL passes: {e_top_c['mean'] - d64['mean']:.1f} nats, at {e_top_c['step_s'] / d64['step_s']:.0f}× less compute

At 1M training tokens, DUST with **64 draws** reaches a test loss of **{d64['mean']:.3f}** at {d64['step_s']:.1f} s per update.
EGGROLL with **{e_top:,} forward passes** reaches **{e_top_c['mean']:.3f}** at {e_top_c['step_s']:.1f} s per update. For reference, backprop
scores {bp['mean']:.3f}, and a model that knows only token frequencies scores {UNI}. EGGROLL is {'above' if cell('eggroll', 1024)['mean'] > UNI else 'below'} that
floor at 1,024 passes and {'above' if e_top_c['mean'] > UNI else 'below'} it at {e_top:,}. {', '.join(f"DUST at {p:,} draws: {cell('dust', p)['mean']:.3f}" for p in (256, 1024, 4096) if cell('dust', p))}.
All of this is in our own implementation, with EGGROLL set up as in `local-learning`. Three seeds per cell, one at 4,096.

## The ladder

![Test loss against population, 1M tokens]({fig_ladder})

Test loss at the best-validation checkpoint. Population is draws for DUST and forward passes of the batch for EGGROLL. Dashed
lines: backprop (ours and the paper's) and the unigram floor. The gap is EGGROLL's loss minus DUST's.

{tab}
Every DUST cell is within {max(abs(cell('dust', p)['mean'] - ref.TABLE1['dust'][p]['1M']) for p in POPS if cell('dust', p) and p in ref.TABLE1['dust']):.3f} of the paper's. Our EGGROLL is weaker than the paper's by
{min(egg_gap.values()):.2f} to {max(egg_gap.values()):.2f} nats (report 2), and the shortfall shrinks with population, so {wider_txt}

## DUST against backprop

{tab3}
{bp_txt}

## What it costs

![Test loss against measured seconds per update]({fig_cost})

{tab2}
A DUST draw and an EGGROLL forward pass cost about the same here, so at equal population the comparison is also equal-compute.
The cost is wall-clock on one RTX 4090, from code written for clarity (see the paper's limitations).

{('![Validation loss over training](' + fig_curves + ')' + chr(10) + chr(10) + 'Validation loss over the 61 updates, mean over seeds.') if fig_curves else ''}
{ex_txt}
## What this changes

- The ordering reproduces on a baseline we implemented ourselves: DUST at 64 draws beats EGGROLL at {e_top:,} passes by {e_top_c['mean'] - d64['mean']:.2f} nats at {e_top_c['step_s'] / d64['step_s']:.0f}× less measured cost.
- The paper's order of magnitude for the efficiency advantage reproduces, with a wide interval. It rests on the slope of EGGROLL's ladder
  beyond what any of us measured, and the ladder bends: three points would have said hundreds, four say thousands.
- DUST at 4,096 draws is below our backprop on one seed, as in the paper. One seed against a backprop spread of {bp['sd']:.3f} is suggestive, not established.
- Not tested: populations above 4,096, where the paper's ladder goes to 16,384, and the budgets above 1M tokens beyond one seed (report 5).

Code: `experiments3.py`, `experiments9.py`, `experiments10.py`, `experiments17.py`, `scripts/extrapolate.py`.
"""
out = PROJECT / "reports" / "04-ladder.md"
out.write_text(md)
print(save_report("dust_report_4_ladder", md))
