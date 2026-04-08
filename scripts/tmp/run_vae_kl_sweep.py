"""Run all VAE KL sweep experiments sequentially."""
import subprocess, sys
from pathlib import Path

EXPS = [
    "exp_vae_kl_b0",
    "exp_vae_kl_b1e4",
    "exp_vae_kl_b1e3",
    "exp_vae_kl_b1e2",
    "exp_vae_kl_b1e1",
    "exp_vae_kl_b1",
]

SSL = Path(__file__).parent.parent.parent / "projects" / "ssl"

for exp in EXPS:
    print(f"\n{'='*60}\nRunning {exp}\n{'='*60}", flush=True)
    result = subprocess.run(
        [sys.executable, str(SSL / f"{exp}.py")],
        check=False,
    )
    if result.returncode != 0:
        print(f"ERROR: {exp} failed with code {result.returncode}", flush=True)

print("\nAll VAE KL sweep experiments done.")
