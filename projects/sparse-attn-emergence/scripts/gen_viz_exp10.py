"""
exp10 trajectory-length figure, from results.jsonl.

Usage:
    uv run --no-sync python projects/sparse-attn-emergence/scripts/gen_viz_exp10.py
"""

import json
import sys
from pathlib import Path

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

from lib.viz import save_traj_panel          # noqa: E402

rows, seen = [], set()
for line in (PROJECT / "results.jsonl").read_text().splitlines():
    try:
        r = json.loads(line)
    except Exception:
        continue
    e = str(r.get("experiment", ""))
    if e.startswith("exp10_") and e not in seen:
        seen.add(e)
        rows.append(r)

if not rows:
    raise SystemExit("no exp10 rows yet")

cells = {}
for r in rows:
    key = (r["s"], r["T"])
    rank = (r["solve_rate"], r["exact_rate"], -(r.get("median_t_star") or 1e9))
    if key not in cells or rank > cells[key]["rank"]:
        cells[key] = {"rank": rank, "solve": r["solve_rate"], "exact": r["exact_rate"],
                      "t": r.get("median_t_star"), "lr": r["lr"], "batch": r["batch_size"],
                      "targets": r["batch_size"] * (r["T"] - 1) * r["S"]}

print(f"{'s':>3} {'T':>3} {'batch':>6} {'targets/step':>13} {'lr':>7} {'solved':>7} "
      f"{'exact':>6} {'median t*':>10}")
print("-" * 62)
for (s, T) in sorted(cells):
    d = cells[(s, T)]
    t = f"{d['t']:.0f}" if d["t"] else "—"
    print(f"{s:>3} {T:>3} {d['batch']:>6} {d['targets']:>13,} {d['lr']:>7.0e} "
          f"{d['solve']:>7.2f} {d['exact']:>6.2f} {t:>10}")

print(f"\ntrajectory panel → {save_traj_panel('sparse_attn_emergence_exp10_traj', cells, rows[0]['n_seeds'])}")
