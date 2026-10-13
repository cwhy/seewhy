"""Report 6 -- Adam and Muon: what a better optimiser does for backprop, DUST and EGGROLL.

    CUDA_VISIBLE_DEVICES= ~/.local/bin/uv run --no-sync python projects/dust/scripts/gen_report_6.py
"""

import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.append(str(PROJECT.parents[1]))

from lib import paper_ref as ref, results as R
from lib.figures import opt_cell, optimiser_curves, optimiser_gap, optimiser_ladder, optimiser_zoom
from shared_lib.report import save_report
from shared_lib.typst_report import save_figure

SUFFIX = "_h"
rows = R.load()
OPTS = ("sgd", "adam", "muon")


def ms(c):
    if c is None:
        return "–"
    return f"{c['mean']:.3f} ± {c['sd']:.3f}" if c["n"] > 1 else f"{c['mean']:.3f} (n=1)"


bp = {o: opt_cell(rows, "bp", 0, o) for o in OPTS}
POPS = (64, 256, 1024, 4096)
dust = {(p, o): opt_cell(rows, "dust", p, o) for p in POPS for o in OPTS}
egg = {(p, o): opt_cell(rows, "eggroll", p, o) for p in (64, 256, 1024) for o in ("adam", "muon")}

# the grids (seed 42): what each learning rate scored, so the winner is visible and the sensitivity with it
grid = {}
for r in R.train_rows(rows, exps=("exp19", "exp20", "exp21", "exp22", "exp23"), budget="1M", seed=42):
    grid.setdefault((r["method"], r["pop"], r["opt"]), []).append((r["lr"], r["test_at_best_val"]))
tab_grid = "| method | population | optimiser | matrix learning rate → test loss (seed 42) |\n|---|---:|---|---|\n"
for (m, p, o), v in sorted(grid.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
    tab_grid += f"| {m} | {p or '–'} | {o} | " + ", ".join(f"{lr:g} → {t:.3f}" for lr, t in sorted(v)) + " |\n"

tab = "| | SGD (report 4) | Adam | Muon |\n|---|---:|---:|---:|\n"
tab += f"| backprop | {ms(bp['sgd'])} | {ms(bp['adam'])} | {ms(bp['muon'])} |\n"
for p in POPS:
    if any(dust[(p, o)] for o in OPTS):
        tab += f"| DUST, {p:,} draws | {ms(dust[(p, 'sgd')])} | {ms(dust[(p, 'adam')])} | {ms(dust[(p, 'muon')])} |\n"
for p in (64, 256, 1024):
    if egg[(p, "adam")] or egg[(p, "muon")]:
        tab += f"| EGGROLL, {p:,} passes | – | {ms(egg[(p, 'adam')])} | {ms(egg[(p, 'muon')])} |\n"

gap = "| population | SGD | Adam | Muon |\n|---:|---:|---:|---:|\n"
for p in POPS:
    cells = [(dust[(p, o)]["mean"] - bp[o]["mean"]) if dust[(p, o)] else None for o in OPTS]
    if any(c is not None for c in cells):
        gap += f"| {p:,} | " + " | ".join("–" if c is None else f"{c:+.3f}" for c in cells) + " |\n"

bp_gain = bp["sgd"]["mean"] - min(bp["adam"]["mean"], bp["muon"]["mean"])
d256_gain = dust[(256, "sgd")]["mean"] - min(dust[(256, "adam")]["mean"], dust[(256, "muon")]["mean"])
d64_gain = dust[(64, "sgd")]["mean"] - min(dust[(64, "adam")]["mean"], dust[(64, "muon")]["mean"])
best256 = min(("adam", "muon"), key=lambda o: dust[(256, o)]["mean"])
muon_vs_adam_egg = {p: egg[(p, "muon")]["mean"] - egg[(p, "adam")]["mean"] for p in (64, 256) if egg[(p, "muon")] and egg[(p, "adam")]}
fig = save_figure(optimiser_ladder(rows, name=f"optimiser_ladder{SUFFIX}"), fmt="svg")
fig_zoom = save_figure(optimiser_zoom(rows, name=f"optimiser_zoom{SUFFIX}"), fmt="svg")
fig_gap = save_figure(optimiser_gap(rows, name=f"optimiser_gap{SUFFIX}"), fmt="svg")
fig_curves = save_figure(optimiser_curves(rows, name=f"optimiser_curves{SUFFIX}"), fmt="svg")

# run-to-run noise at a fixed seed: the same cell run twice (exp21 grid vs exp24 repeat)
rep = [(a, b) for a in R.train_rows(rows, exps=("exp21",), seed=42, lr=0.0015)
       for b in R.train_rows(rows, exps=("exp24",), method="dust", pop=256, opt="adam", seed=42)]
noise_txt = (f"The same cell run twice (seed 42, 256 draws, Adam at 0.0015) scored {rep[0][0]['test_at_best_val']:.3f} and {rep[0][1]['test_at_best_val']:.3f}: "
             f"identical for the first four updates, then apart. That is GPU floating-point nondeterminism amplified over training, "
             f"so a single-seed grid under Adam has about {abs(rep[0][0]['test_at_best_val'] - rep[0][1]['test_at_best_val']):.2f} nats of noise before seeds differ at all." if rep else "")

# rate check at 1,024 draws under Muon: exp25 ran 0.003 (the 256-draw winner), exp28 the two points around it
chk = sorted([(r["lr"], r["test_at_best_val"], r["best_val"]) for r in R.train_rows(rows, exps=("exp25", "exp28"), method="dust", pop=1024, opt="muon", seed=42)])
chk_txt = (", ".join(f"{lr:g} → {t:.3f}" for lr, t, _ in chk) +
           (f": the carried-over rate was the best of the three by validation loss." if chk and min(chk, key=lambda c: c[2])[0] == 0.003 else
            ": a different rate won on validation loss."))

bp_best = min(bp["adam"]["mean"], bp["muon"]["mean"])
d1k_gain = dust[(1024, "sgd")]["mean"] - dust[(1024, "muon")]["mean"]
d1k_gap = dust[(1024, "muon")]["mean"] - bp["muon"]["mean"]

am_gap = [dust[(p, "adam")]["mean"] - dust[(p, "muon")]["mean"] for p in (64, 256, 1024)]
muon_gains = [dust[(p, "sgd")]["mean"] - dust[(p, "muon")]["mean"] for p in (64, 256, 1024)]
if dust[(4096, "muon")]:
    d4 = dust[(4096, "muon")]["mean"]
    gap4 = d4 - bp["muon"]["mean"]
    lead4096 = (f"At 4,096 draws (one seed) DUST under Muon scores {d4:.3f}, {abs(gap4):.3f} nats {'ahead of' if gap4 < 0 else 'behind'} backprop under Muon, "
                f"where under SGD it was {abs(dust[(4096, 'sgd')]['mean'] - bp['sgd']['mean']):.3f} {'ahead' if dust[(4096, 'sgd')]['mean'] < bp['sgd']['mean'] else 'behind'}. ")
    mg = [dust[(p, "muon")]["mean"] - bp["muon"]["mean"] for p in (64, 256, 1024, 4096)]
    shrinks = all(b < 0.55 * a for a, b in zip(mg, mg[1:]))
    tail4096 = (f"- At 4,096 draws, DUST under Muon scores {d4:.3f} (one seed), {abs(gap4):.3f} nats {'ahead of' if gap4 < 0 else 'behind'} backprop under Muon. "
                f"Under SGD the same population was {abs(dust[(4096, 'sgd')]['mean'] - bp['sgd']['mean']):.3f} {'ahead' if dust[(4096, 'sgd')]['mean'] < bp['sgd']['mean'] else 'behind'}. "
                f"The distance under Muon at 64, 256, 1,024 and 4,096 draws is {', '.join(f'{g:+.3f}' for g in mg)}"
                f"{': it shrinks by more than half with each quadrupling of the population and has not reached zero. Four points, the last on one seed, do not say whether it will.' if shrinks else '.'}\n")
else:
    tail4096 = lead4096 = ""

md = f"""# Better optimisers cut DUST's loss, but backprop's more

At 1M tokens, Muon takes DUST with 1,024 draws from **{dust[(1024, 'sgd')]['mean']:.3f}** under SGD to **{dust[(1024, 'muon')]['mean']:.3f}** (one seed), a gain of
{d1k_gain:.2f} nats. Backprop gains more, from **{bp['sgd']['mean']:.3f}** to **{bp_best:.3f}** ({bp_gain:.2f}), so DUST goes from level with backprop
(+{dust[(1024, 'sgd')]['mean'] - bp['sgd']['mean']:.3f}) to {d1k_gap:.2f} nats behind it (Muon against Muon). At 256 draws, three seeds, DUST goes {dust[(256, 'sgd')]['mean']:.3f} →
{dust[(256, 'adam')]['mean']:.3f} under Adam and {dust[(256, 'muon')]['mean']:.3f} under Muon, backprop {bp['sgd']['mean']:.3f} → {bp['adam']['mean']:.3f} and {bp['muon']['mean']:.3f}.
{lead4096}Muon is clearly the better optimiser for DUST; for backprop Adam and Muon are {'level' if abs(bp['muon']['mean'] - bp['adam']['mean']) < 0.05 else 'different'}.
EGGROLL, which was already run with Adam, is not helped by Muon. It is slightly worse, by a few seed standard deviations (0.01 to 0.015) but far less than the others gain: {', '.join(f"{p} passes {v:+.3f}" for p, v in muon_vs_adam_egg.items())} nats against its Adam arm.

## The two optimisers, in short

*SGD with momentum* steps along a running average of the gradient, with one learning rate per group of parameters.
*Adam* divides each coordinate's step by the typical size of that coordinate's gradient, so every weight moves by about the learning
rate whatever its gradient's scale. *Muon* works on a weight *matrix* as a whole: it takes the momentum matrix, replaces it by the
nearest matrix whose singular values are all equal (five Newton–Schulz steps), and steps along that. The update has no dominant direction by
construction. Embeddings, the head and the scalars keep Adam. Both optimisers are scale-free, so the unknown constant in a
zeroth-order estimate does not matter to them.

Why try Muon on these estimates: an EGGROLL estimate of a matrix is a sum of rank-one terms, so its spectrum is very uneven, and Muon flattens spectra.

## What was run

Paper's Table 7 learning-rate groups for Adam: DUST matrices and head at the rate in the grid, token and value embeddings at 300× that, residual scalars at 100×,
β₁ = 0.7; backprop embeddings 0.3 and scalars 0.2, β₁ = 0.8; EGGROLL 0.003 throughout, β₁ = 0.5; β₂ = 0.95 for all. Under Adam the paper also re-weights DUST's
draws toward the token embedding, at noise 0.4. Its per-block split is not given, so the SGD recipe's tilt is kept and rescaled; the draws
per update are 1.375 times the nominal population. Muon: Moonlight scaling, so its learning rate is comparable to Adam's. Grids on seed 42, then the winner on three seeds (one seed at 1,024 and 4,096).

{tab_grid}
{noise_txt}

Rate check at 1,024 draws under Muon (seed 42): {chk_txt}

## Results

![DUST and backprop under SGD, Adam and Muon]({fig_zoom})

DUST at each population under the three optimisers (violet, light to dark) and backprop under the same three as flat lines (red, light to dark; Adam and Muon, {bp['adam']['mean']:.3f} and {bp['muon']['mean']:.3f}, nearly coincide).

![All three methods under each optimiser]({fig})

The same, with EGGROLL (grey): its loss is 1.30 to 1.95 nats above DUST under the same optimiser's, which is why the figure above leaves it out.

{tab}
Three seeds except 1,024 and 4,096 draws and 1,024 passes (one seed, at the 256-population winner's rate). SGD rows are report 4's, EGGROLL's Adam row is the ladder of report 4.

DUST minus backprop, under each optimiser (negative means DUST is ahead):

{gap}
![DUST minus backprop under the same optimiser]({fig_gap})

![Validation loss over training, backprop and DUST at 256 draws]({fig_curves})

Validation loss over the 61 updates, mean over seeds: backprop (red) and DUST at 256 draws (violet) under SGD, Adam and Muon, light to dark.

## What this changes

- Both optimisers lower DUST at every population, and Muon lowers it more: {dust[(64, 'sgd')]['mean']:.2f} → {dust[(64, 'muon')]['mean']:.2f} at 64 draws ({d64_gain:.2f} nats), {dust[(1024, 'sgd')]['mean']:.2f} → {dust[(1024, 'muon')]['mean']:.2f} at 1,024.
  The paper's claim that DUST works under Adam holds in that sense. Adam, in this port, is the weaker of the two for DUST, by {min(am_gap):.2f} to {max(am_gap):.2f} nats.
- Backprop gains more ({bp_gain:.2f} nats against {min(muon_gains):.2f} to {max(muon_gains):.2f} for DUST under Muon at 64 to 1,024 draws), so report 4's headline, DUST level with backprop at 1,024 draws, is a statement
  about SGD. Under Muon the same population leaves DUST {d1k_gap:.2f} nats behind (one seed); under Adam it leaves it {dust[(1024, 'adam')]['mean'] - bp['adam']['mean']:.2f} behind, but that cell is
  one seed at a rate taken from a 256-draw grid with {abs(rep[0][0]['test_at_best_val'] - rep[0][1]['test_at_best_val']):.2f} nats of noise, so it is weak evidence.
{tail4096}- The paper's own Adam section (its Section 3) fits a power law to DUST's Adam ladder and reports a limit of {ref.ADAM_PAPER['dust_limit']:.3f}, below its backprop's {ref.ADAM_PAPER['bp_1M']:.3f}.
  Our backprop under Adam, {bp['adam']['mean']:.3f}, agrees with theirs. Our DUST under Adam is {dust[(1024, 'adam')]['mean'] - bp['adam']['mean']:.2f} nats behind it at 1,024 draws, and three populations do not support a fitted limit.
- EGGROLL is not helped by Muon (slightly worse at all three populations, 1,024 on one seed). A reading that fits the alignment result of report 3, offered as a hypothesis and not tested here: an estimate whose cosine to
  the gradient is about 0.01 has no spectrum worth flattening, and normalising noise does not produce signal.
- Not tested: EGGROLL's own tuning at the new settings (its Muon rate was reused across populations), a rate search at 4,096 draws (1,024 got a check, above), DUST under Adam at 4,096, and the paper's exact draw
  split under Adam (its per-block numbers are not given; the draws here are 1.375 times the nominal population).

Code: `lib/optim.py`, `experiments19.py` to `experiments28.py`, `scripts/gpu_queue.py`.
"""
out = PROJECT / "reports" / "06-optimisers.md"
out.write_text(md)
print(save_report("dust_report_6_optimisers", md))
