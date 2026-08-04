"""
Compact view of results.jsonl — headline fields only, no curves.

Usage:
    uv run --no-sync python projects/sparse-attn-emergence/scripts/summarize.py [filter]
"""

import json
import sys
from pathlib import Path

JSONL = Path(__file__).parent.parent / "results.jsonl"
want = sys.argv[1] if len(sys.argv) > 1 else ""

rows, seen = [], set()
for line in JSONL.read_text().splitlines():
    try:
        r = json.loads(line)
    except Exception:
        continue
    e = r.get("experiment", "?")
    if e not in seen and want in e:
        seen.add(e)
        rows.append(r)

print(f"{'experiment':<22} {'seeds':>5} {'solve':>6} {'med t*':>7} {'final loss':>11} {'time':>6}")
print("-" * 64)
for r in rows:
    loss = r.get("final_loss2") or r.get("final_loss_last") or [float("nan")]
    med = sorted(loss)[len(loss) // 2]
    t = r.get("median_t_star")
    print(f"{r.get('experiment', '?'):<22} {r.get('n_seeds', 0):>5} "
          f"{r.get('solve_rate', float('nan')):>6.2f} "
          f"{('—' if not t else f'{t:.0f}'):>7} {med:>11.4f} {r.get('time_s', 0):>5.0f}s")

for r in rows:
    ts = r.get("t_star")
    if isinstance(ts, dict):          # exp1 stores t* per threshold
        ts = ts.get("0.95")
    if ts and r.get("experiment", "").startswith(("exp1", "exp4")):
        vals = sorted(t for t in ts if t)
        print(f"\n{r['experiment']} t* sorted ({len(vals)}/{len(ts)} emerged): {vals}")
