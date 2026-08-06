"""
exp12 pool-size figure, from results.jsonl.

Usage:
    uv run --no-sync python projects/sparse-attn-emergence/scripts/gen_viz_exp12.py
"""

import json
import sys
from pathlib import Path

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

from lib.viz import save_pool_panel          # noqa: E402

rows, seen = [], set()
for line in (PROJECT / "results.jsonl").read_text().splitlines():
    try:
        r = json.loads(line)
    except Exception:
        continue
    e = str(r.get("experiment", ""))
    if e.startswith("exp12_") and e not in seen:
        seen.add(e)
        rows.append(r)

if not rows:
    raise SystemExit("no exp12 rows yet")

cells = {}
for r in rows:
    key = r["n_rules"]                       # None == fresh rule per sequence
    rank = (r["solve_rate"], -(r.get("median_t_star") or 1e9))
    if key not in cells or rank > cells[key]["rank"]:
        cells[key] = {"rank": rank, "solve": r["solve_rate"], "t": r.get("median_t_star"),
                      "gain": r["in_context_gain"], "lr": r["lr"],
                      "bits": r.get("memorisable_bits")}

print(f"{'N':>7} {'memorisable':>12} {'lr':>7} {'solved':>7} {'median t*':>10} {'gain':>7}")
print("-" * 56)
for k in sorted(cells, key=lambda x: (x is None, x or 0)):
    d = cells[k]
    bits = "impossible" if d["bits"] is None else f"{d['bits'] // 8:,} B"
    t = f"{d['t']:.0f}" if d["t"] else "—"
    print(f"{('fresh' if k is None else k):>7} {bits:>12} {d['lr']:>7.0e} "
          f"{d['solve']:>7.2f} {t:>10} {d['gain']:>7.3f}")

print(f"\npool panel → {save_pool_panel('sparse_attn_emergence_exp12_pool', cells, rows[0]['n_seeds'])}")
