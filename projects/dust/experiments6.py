"""exp6 — gradient alignment with backprop against population (paper Figure 5, Table 8).

How well does each zeroth-order estimate point along backprop's gradient on the same
batch and weights, and how does that grow with population? Measured at initialisation and
on backprop checkpoints at 10M and 100M tokens (exp2). The paper's claim to test: DUST's
cosine rises with population along cos(K) = c_max / sqrt(1 + c/K) and approaches backprop's
gradient, while EGGROLL's stays below 0.05 at 128k forward passes. The 100M checkpoint
has published fit parameters per layer type (Table 8) to compare against.

Needs: ckpt/bp100M_step610.pkl and ckpt/bp100M_step6100.pkl from exp2.

Usage:
    uv run --no-sync python projects/dust/scripts/run_experiments.py --bg --gpu 0 exp6
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
EXP = "smoke6" if SMOKE else "exp6"

if __name__ == "__main__":
    if SMOKE:
        run_alignment(EXP, ["init"], [64], [64], {})
    else:
        run_alignment(EXP, ["init", "bp100M_step610", "bp100M_step6100"],
                      dust_pops=[64, 256, 1024, 4096, 16384],
                      egg_pops=[64, 256, 1024, 4096, 16384],
                      reps={64: 2, 256: 2, 1024: 2})
