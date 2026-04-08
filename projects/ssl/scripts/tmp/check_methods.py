import json
rows = [json.loads(l) for l in open("projects/ssl/results.jsonl")]
methods = {}
for r in rows:
    m = r.get("method", "none")
    methods.setdefault(m, []).append(r["experiment"])
for m, exps in sorted(methods.items()):
    print(f"{m}: {len(exps)} exps — e.g. {exps[0]}")

print()
# Check what pkl files exist
from pathlib import Path
pkls = sorted(Path("projects/ssl").glob("params_*.pkl"))
print(f"{len(pkls)} pkl files:")
for p in pkls:
    print(f"  {p.name}")
