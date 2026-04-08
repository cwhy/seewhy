"""Run all VAE KL sweep experiments for the 8 non-MLP-shallow architectures."""
import subprocess, sys
from pathlib import Path

SSL = Path(__file__).parent.parent.parent / "projects" / "ssl"

ARCHS_ORDER = ["mlp2", "conv", "conv2", "vit", "vit_patch", "gram_dual", "gram_single", "gram_glu"]
BETAS       = ["b0", "b1e4", "b1e3", "b1e2", "b1e1", "b1"]

exps = [f"exp_vae_kl_{arch}_{beta}" for arch in ARCHS_ORDER for beta in BETAS]

for exp in exps:
    print(f"\n{'='*60}\nRunning {exp}\n{'='*60}", flush=True)
    result = subprocess.run([sys.executable, str(SSL / f"{exp}.py")], check=False)
    if result.returncode != 0:
        print(f"ERROR: {exp} failed (code {result.returncode})", flush=True)

print("\nAll done.")
