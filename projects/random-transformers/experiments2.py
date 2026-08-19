"""exp2 — which embedding matrices have to be trained (H3).

Hypothesis (paper Table 2): embedding-only training works because *all three*
of E_token, E_pos and U are optimised. Freeze any of them and at least one task
breaks — in particular, training only the unembedding should fail everywhere,
because it cannot change what the random stack is actually computing, only how
the answer is read off.

Grid: 4 tasks x {u_only, e_only, etoken_u} x width 1024 x 5 seeds. The
`random` and `full` rows for the same cells come from exp1 and are not rerun.

Usage:
    uv run python projects/random-transformers/scripts/run_experiments.py --bg exp2
"""

import logging
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
sys.path.append(str(Path(__file__).resolve().parent))

import jax

from experiments1 import BATCH, EVAL_EVERY, LR, N_HEAD, N_LAYER, SEEDS, STEPS, WARMUP, WD
from lib.grid import run_grid
from lib.tasks import get_task

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

EXP = "exp2"
MODES = ("u_only", "e_only", "etoken_u")
WIDTH = 1024
ALL_TASKS = ("mod_add", "needle", "decimal", "parens")
TASKS = tuple(os.environ.get("RT_TASKS", ",".join(ALL_TASKS)).split(","))

PROJECT = Path(__file__).parent
JSONL = PROJECT / "results.jsonl"


def cells():
    for task_name in TASKS:
        for mode in MODES:
            for seed in SEEDS:
                yield {
                    "key": f"{task_name}/{mode}/{WIDTH}/{seed}",
                    "task_name": task_name,
                    "mode": mode, "d": WIDTH, "n_layer": N_LAYER, "n_head": N_HEAD,
                    "seed": seed, "steps": STEPS[task_name], "batch": BATCH[task_name],
                    "lr": LR, "wd": WD, "warmup": WARMUP, "eval_every": EVAL_EVERY,
                }


if __name__ == "__main__":
    logging.info(f"devices: {jax.devices()}  tasks: {TASKS}")
    run_grid(cells(), jsonl=JSONL, exp=EXP, task_for=get_task,
             extra_row=lambda c: {"name": f"embedding ablation — {c['key']}"})
