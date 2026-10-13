"""exp16 — what else could close EGGROLL's remaining gap to the paper's EGGROLL?

After exp5 / exp8 / exp14 our EGGROLL reaches 7.775 at 64 passes and 7.399 at 256, against the paper's
tuned EGGROLL-Transformer at 7.428 and 7.200. Rate, noise scale and Adam constants are exhausted: each
moved the result by 0.1 or less. Two untried knobs are cheap to test at 64 passes (about 3-6 minutes a cell):

  * rank-shaped fitness. The paper lists "fitness shaping" among EGGROLL's tuned constants. Here, the 2P
    losses are replaced by their centred ranks (the OpenAI-ES form) before the pairs are weighted.
  * float32 compute. If bfloat16 rounding of the loss differences were costing signal, EGGROLL would
    gain here, and the cosine probe (sigma barely mattered) could not have shown it.

Everything else is the best exp8 cell: paper's Adam constants, lr 3e-3, sigma 3e-3, seed 42.

Usage:
    uv run --no-sync python projects/dust/scripts/run_experiments.py --bg --gpu 0 exp16
"""

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

from lib.train import Cell, run_cells

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

SMOKE = bool(os.environ.get("SMOKE"))
EXP = "smoke16" if SMOKE else "exp16"

BASE = dict(exp=EXP, method="eggroll", budget="1M", pop=64, seed=42, lr=3e-3, sigma=3e-3, momentum=0.5, beta2=0.95,
            steps=3 if SMOKE else None)
CELLS = [Cell(**BASE, shaping="rank", tag="rank"),
         Cell(**BASE, dtype="f32", tag="f32"),
         Cell(**{**BASE, "lr": 1e-2}, shaping="rank", tag="rank_lr0.01")]

if __name__ == "__main__":
    run_cells(CELLS[:1] if SMOKE else CELLS)
