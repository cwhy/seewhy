"""
launch.py — Standardized experiment launcher for babystep experiments.

Creates the experiment directory, snapshots the latest script, and runs it
with output redirected to a timestamped log file.

Usage:
    uv run python projects/small_lm/babystep/scripts/launch.py <experiment-name> [args...]

Example:
    uv run python projects/small_lm/babystep/scripts/launch.py \\
        experiment-17-kylo-50k --tokenizer gpt2 --random-init --batches 3000
"""

import subprocess, sys, shutil, time
from pathlib import Path

BABYSTEP = Path(__file__).resolve().parent.parent
SOURCE   = BABYSTEP / "exp_kylo_factsbased.py"

if len(sys.argv) < 2:
    print(__doc__)
    sys.exit(1)

exp_name   = sys.argv[1]
extra_args = sys.argv[2:]

exp_dir  = BABYSTEP / exp_name
exp_dir.mkdir(parents=True, exist_ok=True)

snapshot = exp_dir / "exp_snapshot.py"
if not snapshot.exists():
    shutil.copy2(SOURCE, snapshot)
    print(f"Snapshot: {snapshot}")
else:
    print(f"Snapshot already exists, reusing: {snapshot}")

log_path = exp_dir / f"run_{time.strftime('%Y%m%d_%H%M%S')}.log"

cmd = [sys.executable, str(snapshot), "--out-dir", str(exp_dir)] + extra_args

print(f"Launching: {' '.join(str(a) for a in cmd)}")
print(f"Log: {log_path}")

with open(log_path, "w") as log:
    subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
