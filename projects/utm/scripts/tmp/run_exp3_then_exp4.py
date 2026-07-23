"""Run exp3 to completion, then immediately start exp4."""
import subprocess
import sys
from pathlib import Path

PROJ = Path(__file__).parents[2]
RUNNER = PROJ / "scripts" / "run_experiments.py"

for exp in ["exp3", "exp4"]:
    print(f"\n=== Starting {exp} ===", flush=True)
    rc = subprocess.run([sys.executable, str(RUNNER), exp]).returncode
    if rc != 0:
        print(f"{exp} failed with exit code {rc}")
        sys.exit(rc)

print("\nAll done.")
