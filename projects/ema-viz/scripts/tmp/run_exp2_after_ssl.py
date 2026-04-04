"""Wait for ssl small encoder params to exist, then run exp2."""
import subprocess, time, sys
from pathlib import Path

SSL_DIR = Path(__file__).parent.parent.parent.parent / "ssl"
DIMS    = [16, 32, 64, 128]

print("Waiting for encoder params...")
while True:
    missing = [d for d in DIMS if not (SSL_DIR / f"params_exp_mlp_ema_{d}.pkl").exists()]
    if not missing:
        break
    print(f"  still missing: {missing}")
    time.sleep(10)

print("All encoder params found — running exp2")
result = subprocess.run(
    [sys.executable, str(Path(__file__).parent.parent / "exp2.py")],
    check=True,
)
