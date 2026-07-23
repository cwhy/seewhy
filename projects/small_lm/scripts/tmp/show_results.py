"""Print a summary table of all results in results.jsonl."""
import json
from pathlib import Path

JSONL = Path(__file__).parent.parent.parent / "results.jsonl"

rows = {}
for line in JSONL.read_text().splitlines():
    r = json.loads(line)
    rows[r["experiment"]] = r

print(f"{'exp':<15} {'params':>10} {'ppl':>8} {'time_s':>8}")
print("-" * 50)
for r in sorted(rows.values(), key=lambda r: r["experiment"]):
    print(f"{r['experiment']:<15} {r['n_params']:>10,} {r['final_eval_ppl']:>8.4f} {r['time_s']:>8.1f}s")
