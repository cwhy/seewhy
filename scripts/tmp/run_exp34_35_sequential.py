"""
Run exp35, exp34 sequentially (one GPU process at a time).

exp35: warm-start from exp33, 200 epochs (~8.2h) — diagnostic: training-limited?
exp34: N_VAR=256, B=32, LATENT=2048, 200 epochs (~15.5h) — match eval coverage

exp35 runs first (faster result, confirms whether more training helps).

Usage:
    uv run python scripts/tmp/run_exp34_35_sequential.py
"""

import subprocess, sys, json, os
from pathlib import Path

ROOT  = Path(__file__).parent.parent.parent
JSONL = ROOT / "projects" / "variations" / "results.jsonl"

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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

def run(script_path):
    print(f"\n>>> {script_path.name}", flush=True)
    r = subprocess.run([sys.executable, str(script_path)], cwd=str(ROOT))
    if r.returncode != 0:
        print(f"FAILED: {script_path.name} (exit {r.returncode})", flush=True)
        return False
    print(f"OK: {script_path.name}", flush=True)
    return True

EXPS = ["exp35", "exp34"]  # exp35 first (8h faster result, diagnostic value)

for exp in EXPS:
    num = exp[3:]
    exp_script    = ROOT / "projects" / "variations" / f"experiments{num}.py"
    report_script = ROOT / "projects" / "variations" / "scripts" / f"gen_report_{exp}.py"
    html_script   = ROOT / "scripts" / "tmp" / f"gen_html_report_{exp}.py"

    if is_done(exp):
        print(f"\n{exp} already done — skipping training", flush=True)
    else:
        if not run(exp_script):
            print(f"Aborting after {exp} failure", flush=True)
            sys.exit(1)

    for s in [report_script, html_script]:
        if s.exists():
            run(s)
        else:
            print(f"  (no report script at {s.name} — skipping)", flush=True)

print("\nAll done.", flush=True)
