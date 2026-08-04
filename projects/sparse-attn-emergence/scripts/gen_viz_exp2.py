"""
exp2 difficulty-surface figures, from results.jsonl (no re-training).

Usage:
    uv run --no-sync python projects/sparse-attn-emergence/scripts/gen_viz_exp2.py
"""

import json
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

from math import comb                                            # noqa: E402

from lib.viz import save_search_space_panel, save_sweep_panels    # noqa: E402

rows, seen = [], set()
for line in (PROJECT / "results.jsonl").read_text().splitlines():
    try:
        r = json.loads(line)
    except Exception:
        continue
    e = str(r.get("experiment", ""))
    if e.startswith("exp2_S") and e not in seen and r.get("steps", 0) >= 10_000:
        seen.add(e)
        rows.append(r)

if not rows:
    raise SystemExit("no full-length exp2 rows in results.jsonl yet")

S_values = sorted({r["S"] for r in rows})
s_values = sorted({r["s"] for r in rows})
solve = np.full((len(S_values), len(s_values)), np.nan)
median_t = np.full_like(solve, np.nan)
for r in rows:
    i, j = S_values.index(r["S"]), s_values.index(r["s"])
    solve[i, j] = r["solve_rate"]
    if r.get("median_t_star"):
        median_t[i, j] = r["median_t_star"]

n_seeds = rows[0]["n_seeds"]
print(f"{len(rows)} cells; S={S_values}; s={s_values}")
for i, S in enumerate(S_values):
    cells = "  ".join(
        "  · " if np.isnan(solve[i, j]) else f"{solve[i, j]:.2f}" for j in range(len(s_values)))
    print(f"  S={S:>2}  {cells}")

url = save_sweep_panels("sparse_attn_emergence_exp2_sweep", s_values, S_values,
                        solve, median_t, n_seeds)
print(f"\nsweep panels → {url}")

cells = [{"S": r["S"], "s": r["s"], "comb": comb(r["S"], r["s"]),
          "final_loss2": float(np.median(r["final_loss2"])),
          "median_t_star": r.get("median_t_star"), "solve_rate": r["solve_rate"]}
         for r in rows]
print("\n  S   s   C(S,s)      final loss2   median t*")
for c in sorted(cells, key=lambda c: (c["S"], c["s"])):
    t = f"{c['median_t_star']:.0f}" if c["median_t_star"] else "—"
    print(f" {c['S']:>2} {c['s']:>3} {c['comb']:>10}  {c['final_loss2']:>11.4f}  {t:>9}")

url2 = save_search_space_panel("sparse_attn_emergence_exp2_search_space", cells)
print(f"\nsearch-space panels → {url2}")
