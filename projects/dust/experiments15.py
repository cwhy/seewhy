"""exp15 — EGGROLL's alignment at 65,536 and 131,072 forward passes.

The paper's claim (Section 4.2): at 128k passes EGGROLL's cosine to backprop "is still below 0.05
in cosine ... for every type but the head". exp6 measures it up to 16,384 passes; the cosine of a
random-direction estimator grows as sqrt(P/d), so the cheapest check of the claim is to go to the
population the paper quotes, once, on the 100M-token checkpoint. 131,072 passes is ~70 minutes.

Usage:
    uv run --no-sync python projects/dust/scripts/run_experiments.py --bg --gpu 0 exp15
"""

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

from lib.align import run_alignment

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

SMOKE = bool(os.environ.get("SMOKE"))
EXP = "smoke15" if SMOKE else "exp15"

if __name__ == "__main__":
    run_alignment(EXP, ["init"] if SMOKE else ["bp100M_step6100"], dust_pops=[],
                  egg_pops=[64] if SMOKE else [65536, 131072], reps={})
