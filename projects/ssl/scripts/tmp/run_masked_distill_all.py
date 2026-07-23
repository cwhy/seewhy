"""Run all 8 masked_distill architecture experiments sequentially."""
import subprocess, sys
from pathlib import Path

SSL = Path(__file__).parent.parent.parent

ARCHS = ["mlp2", "conv", "conv2", "vit", "vit_patch", "gram_dual", "gram_single", "gram_glu"]

for arch in ARCHS:
    exp = f"exp_masked_distill_{arch}"
    print(f"\n{'='*60}\nRunning {exp}\n{'='*60}", flush=True)
    result = subprocess.run([sys.executable, str(SSL / f"{exp}.py")], check=False)
    if result.returncode != 0:
        print(f"ERROR: {exp} failed (code {result.returncode})", flush=True)

print("\nAll done.")
