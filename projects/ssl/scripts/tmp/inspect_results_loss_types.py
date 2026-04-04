"""Inspect results.jsonl for loss types and experiment names."""
import json
from pathlib import Path

SSL = Path(__file__).parent.parent.parent
rows = []
with open(SSL / "results.jsonl") as f:
    for l in f:
        l = l.strip()
        if l:
            try:
                rows.append(json.loads(l))
            except Exception:
                pass

for r in rows:
    exp  = r.get("experiment", "")
    acc  = r.get("final_probe_acc", r.get("probe_acc", ""))
    loss = r.get("loss_type", r.get("method", ""))
    keys = [k for k in r if "sig" in k.lower() or "ema" in k.lower() or "loss" in k.lower()]
    print(f"{exp:35s}  acc={acc}  loss={loss}  keys={keys[:3]}")
