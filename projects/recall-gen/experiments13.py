"""Recall-Gen — exp13: does the recall solution CONTAIN generalisable structure?

exp1 ends with a model that retrieves perfectly and completes badly. Two very
different things could be true of it:

  (a) the recall solution is a good representation of digits that simply is not
      being asked to complete, in which case a short nudge should turn it into a
      strong completer; or
  (b) recall was solved by machinery that is useless for completion — matching
      and copying — in which case the nudge buys nothing that training from
      scratch for the same budget would not.

So: load exp1's weights, train 2 000 steps of completion, and compare against
exp14, which is 2 000 steps of completion from random initialisation. Same
budget, same schedule, same data, same seed; the only difference is where the
weights started.

This is the cleanest test in the project of the question in its title, because
it does not ask whether a recall model generalises — it asks whether what it
learned was worth anything to a model that must.

Requires params_exp1.pkl, so exp1 must have been run with param saving.

Usage:
    uv run python projects/recall-gen/scripts/run_experiments.py --bg exp13 exp14
"""

import logging
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from lib.core import Cfg
from lib.train import Run, run

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

SMOKE = bool(os.environ.get("SMOKE"))

RN = Run(
    exp_name="smoke_exp13" if SMOKE else "exp13",
    name="fine-tune exp1 (recall) into completion for 2000 steps",
    M=16, Q=4, mask_rows=14,
    batch=256, steps=200 if SMOKE else 2000, lr=3e-4, seed=0,
    train_mode="gen", init_from="exp1",
    cfg=Cfg(d_model=256, n_layers=4, dk=64, n_heads=4, n_tokens=20),
    n_eval=64 if SMOKE else 512,
    eval_every=100 if SMOKE else 200,
)

if __name__ == "__main__":
    run(RN)
