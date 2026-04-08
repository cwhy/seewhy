"""Deduplicate results_knn_comparison.jsonl by (encoder, training, dataset) key."""
import json
from pathlib import Path

path = Path(__file__).parent.parent.parent / "projects" / "ssl" / "results_knn_comparison.jsonl"
rows = []
with open(path) as f:
    for line in f:
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass

seen = set()
unique = []
for r in rows:
    k = (r.get("encoder"), r.get("training"), r.get("dataset"))
    if k not in seen:
        seen.add(k)
        unique.append(r)

with open(path, "w") as f:
    for r in unique:
        f.write(json.dumps(r) + "\n")

print(f"Deduped {len(rows)} -> {len(unique)} rows")
