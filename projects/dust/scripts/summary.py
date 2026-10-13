"""Print every ladder number from results.jsonl, beside the paper's, in one place.

The paper's prose is written from this output, so each figure quoted there has one origin.

    CUDA_VISIBLE_DEVICES= ~/.local/bin/uv run --no-sync python projects/dust/scripts/summary.py
"""

import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from lib import paper_ref as ref, results as R

rows = R.load()
print(f"{len(rows)} rows\n")
for budget in ("100K", "1M", "10M"):
    print(f"== {budget} tokens: test loss, mean ± sd (n)  | paper | s/update")
    bp = R.cell(rows, method="bp", budget=budget)
    if bp:
        print(f"  backprop            {bp['mean']:.3f} ± {bp['sd']:.3f} (n={bp['n']})  | {ref.TABLE1['bp'][budget]:.3f} | {bp['step_s']:.2f}")
    for method in ("dust", "eggroll"):
        for pop in (64, 256, 1024, 4096):
            c = R.cell(rows, method=method, budget=budget, pop=pop)
            if c:
                pub = ref.TABLE1[method].get(pop, {}).get(budget)
                print(f"  {method:8s} {pop:5d}    {c['mean']:.3f} ± {c['sd']:.3f} (n={c['n']})  | {pub:.3f} | {c['step_s']:.2f}   diff {c['mean'] - pub:+.3f}")
    print()

print("== cross-method at matched populations, 1M (ours; paper in brackets)")
for pop in (64, 256, 1024, 4096):
    d, e = R.cell(rows, method="dust", budget="1M", pop=pop), R.cell(rows, method="eggroll", budget="1M", pop=pop)
    if d and e:
        print(f"  {pop:5d}: EGGROLL − DUST = {e['mean'] - d['mean']:.2f}   (paper {ref.TABLE1['eggroll'][pop]['1M'] - ref.TABLE1['dust'][pop]['1M']:.2f});"
              f"  cost EGGROLL/DUST = {e['step_s'] / d['step_s']:.2f}")
d64 = R.cell(rows, method="dust", budget="1M", pop=64)
for pop in (1024, 4096):
    e = R.cell(rows, method="eggroll", budget="1M", pop=pop)
    if d64 and e:
        print(f"  DUST 64 draws ({d64['mean']:.3f}, {d64['step_s']:.1f} s/update) vs EGGROLL {pop} passes ({e['mean']:.3f}, {e['step_s']:.1f} s/update):"
              f" {e['mean'] - d64['mean']:+.2f} nats, {e['step_s'] / d64['step_s']:.0f}x the cost")
bp = R.cell(rows, method="bp", budget="1M")
if bp:
    for pop in (64, 256, 1024, 4096):
        d = R.cell(rows, method="dust", budget="1M", pop=pop)
        if d:
            print(f"  DUST {pop:5d} − backprop = {d['mean'] - bp['mean']:+.3f}  (bp sd {bp['sd']:.3f}, dust sd {d['sd']:.3f})")

print("\n== extrapolation rows")
for r in rows:
    if r.get("kind") == "extrapolation":
        print(" ", r["name"], {k: (v.get("multiple_of_64") if isinstance(v, dict) else v) for k, v in r["fits"].items()}, "published", r["published"])
print("\n== unigram floor", [r for r in rows if r.get("kind") == "constants"][0]["unigram_floor"] if any(r.get("kind") == "constants" for r in rows) else "n/a")
