"""How much EGGROLL population would match DUST at 64 draws? The paper's Appendix D, on our ladder.

The paper extrapolates EGGROLL's ladder (Table 1) to the population at which it would reach the test loss of
DUST at 64 draws, and reports the multiple of 64 (Table 2). It uses a power law L(P) = A P^-a + E and straight
lines in log population through the top three and the top two cells. This does the lines on our own ladder, and the
power law when there are enough points for it to be more than an interpolation, and appends one row to results.jsonl
(kind="extrapolation") so the paper's numbers trace to the log.

    CUDA_VISIBLE_DEVICES= ~/.local/bin/uv run --no-sync python projects/dust/scripts/extrapolate.py
"""

import json
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from lib import paper_ref as ref, results as R

BUDGET = "1M"


def line_multiple(pops, losses, target):
    """Population (as a multiple of 64) at which a least-squares line in log2(P) hits `target`."""
    x, y = np.log2(np.asarray(pops, float)), np.asarray(losses, float)
    slope, icept = np.polyfit(x, y, 1)
    if slope >= 0:
        return None, float(slope)
    return float(2 ** ((target - icept) / slope) / 64), float(slope)


def power_law_multiple(pops, losses, target):
    from scipy.optimize import least_squares

    P, L = np.asarray(pops, float), np.asarray(losses, float)
    best = None
    for E0 in (L.min() - 0.5, 0.0, L.min() - 2.0):
        r = least_squares(lambda p: p[0] * P ** (-p[1]) + p[2] - L, [1.0, 0.3, E0],
                          bounds=([0, 0.01, -50], [1e6, 3.0, L.min()]))
        if best is None or r.cost < best.cost:
            best = r
    A, a, E = best.x
    if E >= target:
        return None, dict(A=A, alpha=a, E=E)
    return float(((target - E) / A) ** (-1 / a) / 64), dict(A=float(A), alpha=float(a), E=float(E))


def main() -> None:
    rows = R.load()
    egg = {p: R.cell(rows, method="eggroll", budget=BUDGET, pop=p) for p in (64, 256, 1024, 4096)}
    egg = {p: c for p, c in egg.items() if c}
    dust64 = R.cell(rows, method="dust", budget=BUDGET, pop=64)
    if len(egg) < 3 or not dust64:
        raise SystemExit("need the EGGROLL ladder (3+ populations) and DUST at 64 draws")
    pops = sorted(egg)
    loss = {p: egg[p]["mean"] for p in pops}
    out = dict(experiment="extrapolation", name=f"extrapolation|{BUDGET}", kind="extrapolation", budget=BUDGET,
               target=dust64["mean"], target_label="DUST at 64 draws", eggroll_pops=pops, eggroll_loss=[loss[p] for p in pops],
               published=dict(line_top3=8200, line_top2=6000, power_law=23000), fits={})
    for name, k in (("line_top3", 3), ("line_top2", 2)):
        sel = pops[-k:]
        mult, slope = line_multiple(sel, [loss[p] for p in sel], dust64["mean"])
        out["fits"][name] = dict(pops=sel, multiple_of_64=mult, slope_per_doubling=slope)
    if len(pops) >= 4:
        mult, par = power_law_multiple(pops, [loss[p] for p in pops], dust64["mean"])
        out["fits"]["power_law"] = dict(pops=pops, multiple_of_64=mult, **par)
    print(json.dumps(out, indent=1)[:1800])
    if "--write" in sys.argv:
        with open(PROJECT / "results.jsonl", "a") as f:
            f.write(json.dumps(out) + "\n")
        print("appended")


if __name__ == "__main__":
    main()
