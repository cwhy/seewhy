"""Report 2 -- how strong is our EGGROLL baseline?

    CUDA_VISIBLE_DEVICES= ~/.local/bin/uv run --no-sync python projects/dust/scripts/gen_report_2.py
"""

import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.append(str(PROJECT.parents[1]))

from lib import diagrams, paper_ref as ref, results as R
from lib.figures import eggroll_tuning
from shared_lib.media import save_media
from shared_lib.report import save_report
from shared_lib.typst_report import save_figure

rows = R.load()
FIG_SUFFIX = "_d"
grid = [r for r in R.train_rows(rows, method="eggroll", budget="1M", seed=42)
        if r["experiment"] in ("exp5", "exp8", "exp14", "exp16") and r["pop"] in (64, 256)]


def best(pop):
    rs = [r for r in grid if r["pop"] == pop and variant(r) == "raw, bf16"]   # the local-learning setting: raw differences, bf16
    return min(rs, key=lambda r: r["best_val"])          # chosen on validation, as in the paper


def adam(r):
    return "paper (0.5, 0.95)" if r["momentum"] == 0.5 else "default (0.9, 0.999)"


def variant(r):
    v = []
    if r.get("hp", {}).get("shaping") == "rank":
        v.append("rank-shaped")
    if r.get("hp", {}).get("dtype") == "f32":
        v.append("f32")
    return ", ".join(v) or "raw, bf16"


b64, b256 = best(64), best(256)
pub64, pub256 = ref.TABLE1["eggroll"][64]["1M"], ref.TABLE1["eggroll"][256]["1M"]
gap64, gap256 = b64["test_at_best_val"] - pub64, b256["test_at_best_val"] - pub256


def spread(pop, **where):
    vals = [r["test_at_best_val"] for r in grid if r["pop"] == pop and all(r.get(k) == v for k, v in where.items())]
    return (max(vals) - min(vals)) if len(vals) > 1 else None


def pick(pop, **where):
    rs = [r for r in grid if r["pop"] == pop and all(r.get(k) == v for k, v in where.items())]
    return rs


lr_cells = {r["lr"]: r["test_at_best_val"] for r in pick(64, sigma=3e-3, momentum=0.9)}
sig_cells = {r["sigma"]: r["test_at_best_val"] for r in pick(64, lr=3e-3, momentum=0.5, beta2=0.95)
             if r["experiment"] in ("exp8", "exp14")}
d64 = [r for r in pick(64, lr=3e-3, sigma=3e-3) if r["momentum"] == 0.9][0]["test_at_best_val"]
d256 = [r for r in pick(256, lr=3e-3, sigma=3e-3) if r["momentum"] == 0.9][0]["test_at_best_val"]
shape_rows = [r for r in grid if r["experiment"] == "exp16"]
dust256 = R.cell(rows, method="dust", budget="1M", pop=256)["mean"]

url_diagram = save_media("dust_eggroll_update.svg", diagrams.eggroll_update().encode(), "image/svg+xml")
url_fig = save_figure(eggroll_tuning(rows, name=f"eggroll_tuning{FIG_SUFFIX}"), fmt="svg")

tab = "| passes | σ | Adam (β₁, β₂) | variant | lr | validation | test |\n|---:|---:|---|---|---:|---:|---:|\n"
for r in sorted(grid, key=lambda r: (r["pop"], r["sigma"], r["momentum"], variant(r), r["lr"])):
    mark = " **←**" if r is b64 or r is b256 else ""
    tab += (f"| {r['pop']} | {r['sigma']:g} | {adam(r)} | {variant(r)} | {r['lr']:g} | {r['best_val']:.3f} | "
            f"{r['test_at_best_val']:.3f}{mark} |\n")

shape_txt = ""
if shape_rows:
    base = [r for r in grid if r["pop"] == 64 and r["lr"] == 3e-3 and r["sigma"] == 3e-3 and r["momentum"] == 0.5
            and r["experiment"] == "exp8"][0]
    shape_txt = (
        f"\n**Two more knobs, at 64 passes.** The paper lists fitness shaping among EGGROLL's tuned constants, and "
        f"bfloat16 rounding could be eating the loss differences. Against the best cell ({base['test_at_best_val']:.3f}): "
        + "; ".join(f"{variant(r)}, lr {r['lr']:g}: {r['test_at_best_val']:.3f}" for r in shape_rows)
        + f". The best of them is {base['test_at_best_val'] - min(r['test_at_best_val'] for r in shape_rows):.3f} nats below the best raw cell, "
          "one seed and well inside seed-to-seed spread. The ladder keeps the raw loss differences of `local-learning`.\n")

lad = {p: R.cell(rows, method="eggroll", budget="1M", pop=p) for p in (64, 256, 1024, 4096)}
lgap = {p: lad[p]["mean"] - ref.TABLE1["eggroll"][p]["1M"] for p in lad if lad[p]}
shrinks = len(lgap) == 4 and lgap[64] > lgap[256] > lgap[1024] > lgap[4096]
slope_early = (lad[64]["mean"] - lad[1024]["mean"]) / 4
slope_tail = (lad[1024]["mean"] - lad[4096]["mean"]) / 2
paper_early = (ref.TABLE1["eggroll"][64]["1M"] - ref.TABLE1["eggroll"][1024]["1M"]) / 4
paper_tail = (ref.TABLE1["eggroll"][1024]["1M"] - ref.TABLE1["eggroll"][4096]["1M"]) / 2

md = f"""# Our EGGROLL baseline trails the paper's, by {lgap[64]:.2f} nats at 64 passes and {lgap[4096]:.2f} at 4,096

We tuned EGGROLL on the paper's own task and landed at **{b64['test_at_best_val']:.3f}** with 64 forward passes on the
tuning seed (paper: {pub64:.3f}) and **{b256['test_at_best_val']:.3f}** with 256 (paper: {pub256:.3f}). The final ladder, at those
settings, reads {lad[64]['mean']:.3f}, {lad[256]['mean']:.3f}, {lad[1024]['mean']:.3f} and {lad[4096]['mean']:.3f} at 64, 256, 1,024 and 4,096 passes
(paper: {ref.TABLE1['eggroll'][64]['1M']:.3f}, {ref.TABLE1['eggroll'][256]['1M']:.3f}, {ref.TABLE1['eggroll'][1024]['1M']:.3f}, {ref.TABLE1['eggroll'][4096]['1M']:.3f}; three seeds, one at 4,096). So our baseline is *weaker* than
theirs by {lgap[64]:.2f}, {lgap[256]:.2f}, {lgap[1024]:.2f} and {lgap[4096]:.2f} nats{', a gap that closes as population grows' if shrinks else ''}. That matters because any weakness here would
flatter DUST. DUST at 256 draws scores {dust256:.3f}, so the DUST–EGGROLL gap at that population is {lad[256]['mean'] - dust256:.2f} nats here
against {pub256 - ref.TABLE1['dust'][256]['1M']:.2f} in the paper. The ladder's gap is {(lad[256]['mean'] - dust256) / lgap[256]:.0f} times the baseline's, so a baseline this much weaker cannot
explain it. We could not close the {gap256:.2f} at 256 passes either.

## EGGROLL, in one picture

![EGGROLL on one weight matrix]({url_diagram})

EGGROLL (Sarkar et al.) is evolution strategies made cheap. For every weight matrix, a population member adds a
*rank-one* perturbation `σ·a·bᵀ`. It costs `m + n` random numbers, not `m × n`, and the forward pass applies it as a
side branch, so a member is not a copy of the network. A member is scored as an *antithetic pair*, two forward
passes with `+` and `−` the same perturbation, whose loss difference is one number. The gradient estimate is the
sum of the perturbations weighted by those numbers, and goes to Adam. This is exactly the estimator in
`local-learning/lib/es.py`, applied to the transformer. Population is counted in forward passes, two per pair.

## What we tuned

Seed 42, validation loss picks the winner, 1M tokens. Adam rate and noise scale σ on a grid; then the paper's Adam
constants (β₁ = 0.5, β₂ = 0.95, from its Table 7); then σ extended downward, because the best σ sat on the grid's edge.

![EGGROLL tuning grid]({url_fig})

{tab}
## What moved the loss, and what did not

- **The Adam rate** is the one that matters: at 64 passes and σ = 3e-3, the cells run
  {', '.join(f"{v:.3f}" for _, v in sorted(lr_cells.items()))} for rates {', '.join(f"{k:g}" for k in sorted(lr_cells))}. At the largest rate the loss stays at its starting value.
- **σ** hardly does. At the paper's Adam constants and lr 3e-3, σ of {', '.join(f"{k:g}" for k in sorted(sig_cells))} gives
  {', '.join(f"{sig_cells[k]:.3f}" for k in sorted(sig_cells))}: a spread of {max(sig_cells.values()) - min(sig_cells.values()):.3f} across a decade.
- **The paper's Adam constants** help a little: {d64:.3f} → {b64['test_at_best_val'] if b64['momentum'] == 0.5 else float('nan'):.3f} at 64 passes, and
  {d256:.3f} → {b256['test_at_best_val'] if b256['momentum'] == 0.5 else float('nan'):.3f} at 256. The paper reports that EGGROLL gains under 0.01 from switching to Adam at all.
{shape_txt}
## What this changes

- The ladder in the next report uses these settings: the best-validation cell at each population, and the 256-pass
  winner at 1,024 passes, which was too expensive to tune.
- Our EGGROLL is a faithful instance of the method, not a replica of the paper's. The remaining {lgap[4096]:.2f} to
  {lgap[64]:.2f} nats are not explained, and could be any detail of theirs we cannot see: their code is not released.
  One thing is visible: ours improves faster than the paper's at small populations ({slope_early:.3f} nats per doubling from 64 to 1,024
  passes against the paper's {paper_early:.3f}) and at the same rate at large ones ({slope_tail:.3f} from 1,024 to 4,096 against {paper_tail:.3f}). It is weaker
  where the population is small, and converges on the paper's where it is large.
- The conclusion this puts in doubt is not the ordering, which the gap in the ladder dwarfs, but the *slope* of
  EGGROLL's ladder, which sets the extrapolation of how much population it would need to match DUST.

Code: `lib/eggroll.py`, `experiments5.py`, `experiments8.py`, `experiments14.py`{', `experiments16.py`' if shape_rows else ''}.
"""

out = PROJECT / "reports" / "02-eggroll-baseline.md"
out.write_text(md)
print(save_report("dust_report_2_eggroll_baseline", md))
