"""exp18 — is DUST's edge over backprop a learning-rate-group artefact?

The released DUST optimiser gives three parameter groups their own rates: value embeddings 0.3, the residual
scalars 0.03 and the token embedding 1000. The released backprop baseline uses its base rate for the value
embeddings and the scalars, and 1000 only for the token embedding (the paper's Table 6: "main comparison").
So when DUST ends level with, or below, backprop, part of that could be better-set rates for two small groups
rather than anything about its gradient estimate. This runs backprop at 1M tokens with DUST's group rates and
the baseline's own base rate and momentum (0.175, 0.98), three seeds, to see whether backprop moves.

Usage:
    uv run --no-sync python projects/dust/scripts/run_experiments.py --bg --gpu 0 exp18
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
EXP = "smoke18" if SMOKE else "exp18"

CELLS = [Cell(exp=EXP, method="bp", budget="1M", seed=s, lr=0.175, momentum=0.98, groups="dust", tag="dust_groups",
              steps=3 if SMOKE else None) for s in (42, 43, 44)]

if __name__ == "__main__":
    run_cells(CELLS[:1] if SMOKE else CELLS)
