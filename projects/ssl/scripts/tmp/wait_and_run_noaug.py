"""
Wait for exp_ema to finish, then run no-aug ablations and generate viz.
"""
import time, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.parent
SSL  = ROOT / "projects/ssl"

def run(cmd):
    print(f"\n>>> {cmd}", flush=True)
    subprocess.run(cmd, shell=True, cwd=ROOT, check=True)

# Wait for exp_ema
log = SSL / "logs/exp_ema.log"
print("Waiting for exp_ema...", flush=True)
while True:
    txt = log.read_text() if log.exists() else ""
    if "Done." in txt:
        print("exp_ema done.", flush=True)
        break
    time.sleep(10)

# Run no-aug ablations
run("uv run python projects/ssl/scripts/run_experiments.py exp_noaug_512 exp_noaug_1024 exp_noaug_2048")

# Generate visualization
run("uv run python projects/ssl/scripts/gen_viz_latent_comparison.py")

print("All done.", flush=True)
