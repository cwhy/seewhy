"""exp3 — the width sweep (H2).

Hypothesis: the capability of a random transformer is a function of its width.
The paper's Table 1 shows width 16 collapsing while width 1024 is perfect; this
fills in what happens between, and does the same sweep for fully trained models
so the two curves can be compared. If the effect were about the training
procedure rather than the random stack, the two curves would have the same
shape.

Grid: 4 tasks x {random, full} x widths {32, 64, 128, 256, 512} x 3 seeds. The
endpoints (16 and 1024) already exist at 5 seeds in exp1 and are not rerun;
`lib/figures.py` joins them.

Usage:
    RT_TASKS=needle,parens uv run python projects/random-transformers/experiments3.py
"""

import logging
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
sys.path.append(str(Path(__file__).resolve().parent))

import jax

from experiments1 import BATCH, EVAL_EVERY, LR, N_LAYER, STEPS, WD
from lib.grid import run_grid
from lib.tasks import get_task

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

EXP = "exp3"
WIDTHS = (32, 64, 128, 256, 512)
MODES = ("random", "full")
SEEDS = (0, 1, 2)
N_HEAD = 8
ALL_TASKS = ("mod_add", "needle", "decimal", "parens")
TASKS = tuple(os.environ.get("RT_TASKS", ",".join(ALL_TASKS)).split(","))

PROJECT = Path(__file__).parent
JSONL = PROJECT / "results.jsonl"


def cells():
    for task_name in TASKS:
        for width in WIDTHS:
            for mode in MODES:
                for seed in SEEDS:
                    yield {
                        "key": f"{task_name}/{mode}/{width}/{seed}",
                        "task_name": task_name,
                        "mode": mode, "d": width, "n_layer": N_LAYER, "n_head": N_HEAD,
                        "seed": seed, "steps": STEPS[task_name], "batch": BATCH[task_name],
                        "lr": LR, "wd": WD, "eval_every": EVAL_EVERY,
                    }


if __name__ == "__main__":
    logging.info(f"devices: {jax.devices()}  tasks: {TASKS}")
    run_grid(cells(), jsonl=JSONL, exp=EXP, task_for=get_task,
             extra_row=lambda c: {"name": f"width sweep — {c['key']}"})
