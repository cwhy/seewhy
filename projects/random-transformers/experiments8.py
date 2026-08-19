"""exp8 — a learning-rate search for the small models and the recurrent baseline.

After fixing the warmup (reports/exp1-budget.md), every width-1024 cell matches
the paper. What does not match is the *baselines*: width-16 models and LSTMs
land well below the paper's numbers, with losses still visibly mid-descent —
decimal/lstm at token accuracy 0.601 and loss 1.10, mod_add/full/16 at loss
1.53. They are under-trained at lr 3e-4, which suits the width-1024 models.

This matters in one direction specifically. The paper's claim is that random
transformers match or beat fully trained models and LSTMs. An under-tuned
baseline inflates exactly that claim. So the baselines get a learning-rate
search here that the width-1024 random condition never gets, and the results
tables report the best learning rate per cell.

Stating the asymmetry plainly: this biases against our own headline, which is
the direction an honest replication should err.

Grid: 4 tasks x {random/16, full/16, lstm/1024} x lr {1e-3, 3e-3} x 5 seeds.
lr 3e-4 for these cells already exists in exp1 and is not rerun; the tables take
the best of the three.

Usage:
    RT_TASKS=needle,parens uv run python projects/random-transformers/experiments8.py
"""

import logging
import os
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
sys.path.append(str(Path(__file__).resolve().parent))

import jax

from experiments1 import BATCH, EVAL_EVERY, LSTM_SEEDS, N_LAYER, SEEDS, STEPS, WARMUP, WD
from lib.lstm import train_lstm
from lib.results_io import append_result, read_rows
from lib.tasks import get_task
from lib.train import strip_params, train_task

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

EXP = "exp8"
LRS = (1e-3, 3e-3)
CONDITIONS = (("random", 16), ("full", 16), ("lstm", 1024))
ALL_TASKS = ("mod_add", "needle", "decimal", "parens")
TASKS = tuple(os.environ.get("RT_TASKS", ",".join(ALL_TASKS)).split(","))

PROJECT = Path(__file__).parent
JSONL = PROJECT / "results.jsonl"


def main():
    done = {r["cell"] for r in read_rows(JSONL) if r.get("experiment") == EXP}
    logging.info(f"{len(done)} cells done; tasks = {TASKS}")

    for task_name in TASKS:
        task = get_task(task_name)
        logging.info(f"=== {task_name} (chance {task.chance:.4f}) ===")
        for mode, width in CONDITIONS:
            seeds = LSTM_SEEDS if mode == "lstm" else SEEDS
            for lr in LRS:
                for seed in seeds:
                    cell = f"{task_name}/{mode}/{width}/lr{lr:g}/{seed}"
                    if cell in done:
                        continue
                    logging.info(f"  {cell}")
                    if mode == "lstm":
                        t0 = time.time()
                        row = train_lstm(task, d=width, seed=seed, steps=STEPS[task_name],
                                         batch=BATCH[task_name], eval_every=EVAL_EVERY,
                                         lr=lr, warmup=WARMUP)
                        row["time_s"] = time.time() - t0
                    else:
                        row = train_task(task, mode=mode, d=width, n_layer=N_LAYER, n_head=8,
                                         seed=seed, steps=STEPS[task_name],
                                         batch=BATCH[task_name], lr=lr, wd=WD,
                                         warmup=WARMUP, eval_every=EVAL_EVERY, log=False)
                        row = strip_params(row)
                    row.update({"experiment": EXP, "cell": cell,
                                "name": f"lr search — {cell}"})
                    append_result(JSONL, row)
                    logging.info(f"    seq_acc={row['test_seq_acc']:.4f} "
                                 f"tok_acc={row['test_tok_acc']:.4f} ({row['time_s']:.0f}s)")

    logging.info("exp8 complete")


if __name__ == "__main__":
    logging.info(f"devices: {jax.devices()}")
    main()
