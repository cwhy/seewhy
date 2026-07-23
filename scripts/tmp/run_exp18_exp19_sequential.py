"""
Wait for exp17 to complete, then run exp18 and exp19 sequentially.

Usage:
    uv run python scripts/tmp/run_exp18_exp19_sequential.py
"""

import json, time, subprocess, sys
from pathlib import Path

ROOT    = Path(__file__).parent.parent.parent
JSONL   = ROOT / "projects" / "variations" / "results.jsonl"
POLL_S  = 30

def is_done(exp_name):
    if not JSONL.exists():
        return False
    for line in JSONL.read_text().strip().splitlines():
        try:
            if json.loads(line).get("experiment") == exp_name:
                return True
        except Exception:
            pass
    return False

def run_chain(exp_name):
    print(f"\n{'='*60}")
    print(f"Starting {exp_name} chain...")
    print(f"{'='*60}\n", flush=True)

    exp_script     = ROOT / "projects" / "variations" / f"experiments{exp_name[3:]}.py"
    report_script  = ROOT / "projects" / "variations" / "scripts" / f"gen_report_{exp_name}.py"
    html_script    = ROOT / "scripts" / "tmp" / f"gen_html_report_{exp_name}.py"

    for script in [exp_script, report_script, html_script]:
        print(f"Running: {script.name}", flush=True)
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(ROOT),
        )
        if result.returncode != 0:
            print(f"ERROR: {script.name} failed with exit code {result.returncode}", flush=True)
            return False
        print(f"Done: {script.name}\n", flush=True)
    return True

# ── Wait for exp17 ────────────────────────────────────────────────────────────

print("Waiting for exp17 to complete...", flush=True)
while not is_done("exp17"):
    time.sleep(POLL_S)
    print(f"  still waiting... ({time.strftime('%H:%M:%S')})", flush=True)

print("exp17 complete! Starting exp18.", flush=True)

# ── Run exp18 ─────────────────────────────────────────────────────────────────

if not is_done("exp18"):
    ok = run_chain("exp18")
    if not ok:
        print("exp18 chain failed — aborting exp19", flush=True)
        sys.exit(1)
else:
    print("exp18 already done — skipping", flush=True)

# ── Run exp19 ─────────────────────────────────────────────────────────────────

if not is_done("exp19"):
    run_chain("exp19")
else:
    print("exp19 already done — skipping", flush=True)

print("\nAll done: exp18 + exp19 complete.", flush=True)
