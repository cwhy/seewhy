"""Add 'arch' field to existing MLP-shallow entries in results_vae_kl.jsonl."""
import json
from pathlib import Path

path = Path(__file__).parent.parent.parent / "projects" / "ssl" / "results_vae_kl.jsonl"
rows = [json.loads(l) for l in open(path) if l.strip()]
for r in rows:
    if "arch" not in r:
        r["arch"] = "MLP-shallow"
with open(path, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(f"Updated {len(rows)} rows")
