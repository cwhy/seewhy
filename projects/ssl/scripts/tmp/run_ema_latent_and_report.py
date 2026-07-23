"""
Wait for no-aug ablations to finish, then run EMA latent sweep + report.
"""
import time, subprocess
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.parent
SSL  = ROOT / "projects/ssl"

def run(cmd):
    print(f"\n>>> {cmd}", flush=True)
    subprocess.run(cmd, shell=True, cwd=ROOT, check=True)

def wait_for(log_path, label):
    print(f"Waiting for {label}...", flush=True)
    p = Path(log_path)
    while True:
        txt = p.read_text() if p.exists() else ""
        if "Done." in txt:
            print(f"{label} done.", flush=True)
            break
        time.sleep(10)

# Wait for no-aug ablations (last one: exp_noaug_2048)
wait_for(SSL / "logs/exp_noaug_2048.log", "exp_noaug_2048")

# Run EMA latent sweep (also re-run exp_ema with sigreg diagnostic)
run("uv run python projects/ssl/scripts/run_experiments.py exp_ema_512 exp_ema_1024 exp_ema_2048 exp_ema_diag")

# Generate report
run("uv run python projects/ssl/scripts/gen_report.py")

print("All done.", flush=True)
