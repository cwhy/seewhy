"""Report 5 -- the other budgets: 10M tokens (one seed) and 100K tokens (six updates).

    CUDA_VISIBLE_DEVICES= ~/.local/bin/uv run --no-sync python projects/dust/scripts/gen_report_5.py
"""

import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.append(str(PROJECT.parents[1]))

from lib import paper_ref as ref, results as R
from lib.figures import ladder
from shared_lib.report import save_report
from shared_lib.typst_report import save_figure

SUFFIX = "_a"
rows = R.load()


def cell(b, m, p):
    return R.cell(rows, method=m, budget=b, pop=p)


def ms(c):
    if c is None:
        return "–"
    return f"{c['mean']:.3f} ± {c['sd']:.3f}" if c["n"] > 1 else f"{c['mean']:.3f} (n=1)"


def pub(m, p, b):
    v = ref.TABLE1["bp"][b] if m == "bp" else ref.TABLE1[m][p][b]
    return f"{v:.3f}"


bp10, d10, e10 = cell("10M", "bp", 0), cell("10M", "dust", 256), cell("10M", "eggroll", 256)
bp100, floor10, floor100 = cell("100K", "bp", 0), ref.UNIGRAM_BY_BUDGET["10M"], ref.UNIGRAM_BY_BUDGET["100K"]

t10 = ("| run | ours | paper | ours − paper | s/update |\n|---|---:|---:|---:|---:|\n"
       f"| backprop | {ms(bp10)} | {pub('bp', 0, '10M')} | {bp10['mean'] - ref.TABLE1['bp']['10M']:+.3f} | {bp10['step_s']:.2f} |\n"
       f"| DUST, 256 draws | {ms(d10)} | {pub('dust', 256, '10M')} | {d10['mean'] - ref.TABLE1['dust'][256]['10M']:+.3f} | {d10['step_s']:.1f} |\n"
       f"| EGGROLL, 256 passes | {ms(e10)} | {pub('eggroll', 256, '10M')} | {e10['mean'] - ref.TABLE1['eggroll'][256]['10M']:+.3f} | {e10['step_s']:.1f} |\n")

t100 = ("| population | DUST (ours) | DUST (paper) | EGGROLL (ours) | EGGROLL (paper) | gap, ours | gap, paper |\n|---:|---|---:|---|---:|---:|---:|\n")
for p in (64, 256, 1024):
    d, e = cell("100K", "dust", p), cell("100K", "eggroll", p)
    t100 += (f"| {p:,} | {ms(d)} | {pub('dust', p, '100K')} | {ms(e)} | {pub('eggroll', p, '100K')} | "
             f"{e['mean'] - d['mean']:.2f} | {ref.TABLE1['eggroll'][p]['100K'] - ref.TABLE1['dust'][p]['100K']:.2f} |\n")

fig10 = save_figure(ladder(rows, "10M", name=f"ladder_10M{SUFFIX}"), fmt="svg")
fig100 = save_figure(ladder(rows, "100K", name=f"ladder_100K{SUFFIX}"), fmt="svg")
below100 = [(m, p) for m in ("dust", "eggroll") for p in (64, 256, 1024) if cell("100K", m, p)["mean"] < floor100]
paper_min100 = min([ref.TABLE1["bp"]["100K"]] + [ref.TABLE1[m][p]["100K"] for m in ("dust", "eggroll") for p in ref.TABLE1[m]])

md = f"""# At 10M tokens DUST trails backprop by {d10['mean'] - bp10['mean']:.2f} nats; at 100K nothing clears the floor

At 10M training tokens DUST with 256 draws reaches **{d10['mean']:.3f}** (paper {pub('dust', 256, '10M')}), {d10['mean'] - bp10['mean']:.2f} nats behind backprop at
**{bp10['mean']:.3f}** (paper {pub('bp', 0, '10M')}). EGGROLL with 256 passes reaches **{e10['mean']:.3f}** (paper {pub('eggroll', 256, '10M')}), above a
token-frequency floor of {floor10}. At 100K tokens, which is six updates, {'no method gets below its floor of ' + str(floor100) if not below100 and paper_min100 > floor100 else 'some methods get below the floor of ' + str(floor100)}{', the paper\'s included' if paper_min100 > floor100 else ''}.
These are the budgets either side of the 1M-token ladder of report 4.

## 10M tokens

![Test loss at 10M tokens]({fig10})

{t10}
Backprop has three seeds; DUST and EGGROLL one each, because a 610-update run at 256 draws takes about 80 minutes. EGGROLL's
settings are its 1M-token winner, not tuned at this budget, where the paper tunes every cell. DUST uses the authors' released
10M settings. DUST's distance to backprop here, {d10['mean'] - bp10['mean']:.3f}, is the paper's ({ref.TABLE1['dust'][256]['10M'] - ref.TABLE1['bp']['10M']:.3f}) to within
{abs((d10['mean'] - bp10['mean']) - (ref.TABLE1['dust'][256]['10M'] - ref.TABLE1['bp']['10M'])):.3f}. The EGGROLL–DUST gap is {e10['mean'] - d10['mean']:.2f} nats here against
{ref.TABLE1['eggroll'][256]['10M'] - ref.TABLE1['dust'][256]['10M']:.2f}, wider by {(e10['mean'] - d10['mean']) - (ref.TABLE1['eggroll'][256]['10M'] - ref.TABLE1['dust'][256]['10M']):.2f}.
DUST at 256 draws is {d10['mean'] - bp10['mean']:.2f} nats from backprop here, as in the paper, where DUST stays above backprop at 10M
tokens at every population (its best, {min(ref.TABLE1['dust'][p]['10M'] for p in ref.TABLE1['dust']):.3f} at 16,384 draws, is {min(ref.TABLE1['dust'][p]['10M'] for p in ref.TABLE1['dust']) - ref.TABLE1['bp']['10M']:.3f} above). We ran 256 draws only at this budget.

## 100K tokens

![Test loss at 100K tokens]({fig100})

Six updates. Every method is tuned on the paper's grids (seed 42, winner repeated, grid extended one step where a winner sat on an edge).

{t100}
Backprop here is {ms(bp100)} (paper {pub('bp', 0, '100K')}), and the unigram floor from the 100K prefix is {floor100}. {'Nothing is below that floor' if not below100 else 'Some cells are below the floor'}, ours or the paper's:
six updates teach a model how often each token occurs and little else. DUST is ahead of EGGROLL at every population here, by more than in the paper.
Backprop's spread over three seeds, {bp100['sd']:.3f}, is {bp100['sd'] / R.cell(rows, method='bp', budget='1M')['sd']:.0f} times what it is at 1M tokens, so we do not rank DUST against backprop at this budget.

## What this changes

- The 10M-token DUST number reproduces the paper's, and so does its distance to backprop: DUST at small population is not backprop-competitive at 10M.
- The paper's own caveat applies to us a fortiori: the 100K-token column is a measurement of who learns token frequencies fastest.
- Not tested: tuning EGGROLL at 10M tokens, a second and third seed at 10M, and any population above 256 at this budget.

Code: `experiments11.py`, `experiments12.py`, `experiments13.py`.
"""
out = PROJECT / "reports" / "05-other-budgets.md"
out.write_text(md)
print(save_report("dust_report_5_other_budgets", md))
