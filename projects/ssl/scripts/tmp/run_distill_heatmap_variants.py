"""Run sigreg, dae, and random distillation heatmap experiments sequentially."""
import subprocess, sys
from pathlib import Path

SSL = Path(__file__).parent.parent.parent

for variant in ["sigreg", "dae", "random"]:
    exp = f"exp_distill_heatmap_{variant}"
    print(f"\n{'='*60}\nRunning {exp}\n{'='*60}", flush=True)
    result = subprocess.run([sys.executable, str(SSL / f"{exp}.py")], check=False)
    if result.returncode != 0:
        print(f"ERROR: {exp} failed (code {result.returncode})", flush=True)

print("\nAll done.")
