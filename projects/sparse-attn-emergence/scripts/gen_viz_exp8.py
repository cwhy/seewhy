"""
exp8 crossover figure and table, from results.jsonl.

Usage:
    uv run --no-sync python projects/sparse-attn-emergence/scripts/gen_viz_exp8.py
"""

import json
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

from lib.viz import save_crossover_panel          # noqa: E402

rows, seen = [], set()
for line in (PROJECT / "results.jsonl").read_text().splitlines():
    try:
        r = json.loads(line)
    except Exception:
        continue
    e = str(r.get("experiment", ""))
    if e.startswith(("exp8_", "exp9_")) and e not in seen:      # exp9 adds the KDA arm
        seen.add(e)
        rows.append(r)

if not rows:
    raise SystemExit("no exp8 rows yet")

# best LR per (s, arm): highest solve rate, then highest exact rate, then faster
cells = {}
for r in rows:
    key = (r["s"], r["arch"])
    rank = (r["solve_rate"], r["exact_rate"], -(r.get("median_t_star") or 1e9))
    if key not in cells or rank > cells[key]["rank"]:
        iou = r.get("support_iou")
        cells[key] = {"rank": rank, "solve": r["solve_rate"], "exact": r["exact_rate"],
                      "t": r.get("median_t_star"), "lr": r["lr"],
                      "iou_solved": r.get("support_iou_solved"),
                      "iou_all": float(np.mean(iou)) if iou else None,
                      "chance": r["support_iou_chance"]}

print(f"{'s':>3} {'arm':<12} {'lr':>7} {'solved':>7} {'exact':>6} {'median t*':>10} "
      f"{'iou(solved)':>12} {'chance':>7}")
print("-" * 70)
for (s, arm) in sorted(cells):
    d = cells[(s, arm)]
    t = f"{d['t']:.0f}" if d["t"] else "—"
    io = f"{d['iou_solved']:.2f}" if d["iou_solved"] is not None else "—"
    print(f"{s:>3} {arm:<12} {d['lr']:>7.0e} {d['solve']:>7.2f} {d['exact']:>6.2f} "
          f"{t:>10} {io:>12} {d['chance']:>7.2f}")

url = save_crossover_panel("sparse_attn_emergence_exp8_crossover", cells, rows[0]["n_seeds"])
print(f"\ncrossover → {url}")
