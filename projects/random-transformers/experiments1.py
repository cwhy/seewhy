"""exp1 — the main table (H1, H2).

Hypothesis: a 2-layer transformer whose attention and feed-forward weights are
frozen at their random initialisation still reaches ~100% on all four
algorithmic tasks, provided it is wide; and the width-16 version collapses,
while a width-16 *fully trained* model does not.

Grid: 4 tasks x {random, full} x {1024, 16} x 5 seeds, plus a fully trained
LSTM at width 1024 for the paper's recurrent baseline.

One row per cell, appended as it finishes, so an interrupted run loses nothing
and can be resumed. Set RT_TASKS to a comma-separated subset to split the grid
across GPUs, e.g. RT_TASKS=needle,decimal.

Usage:
    uv run python projects/random-transformers/scripts/run_experiments.py --bg exp1
"""

import json
import logging
import os
import pickle
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))   # repo root
sys.path.append(str(Path(__file__).resolve().parent))       # project dir

import jax

from lib.lstm import train_lstm
from lib.results_io import append_result as _append, read_rows
from lib.tasks import get_task
from lib.train import strip_params, train_task

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

# ── hyperparameters ───────────────────────────────────────────────────────────
EXP = "exp1"
WIDTHS = (1024, 16)
MODES = ("random", "full")
SEEDS = (0, 1, 2, 3, 4)
N_LAYER = 2
N_HEAD = 8

# Budget. The paper uses 10k steps x batch 1000 for the streamed tasks and 5000
# epochs x batch 4000 for modular addition. A convergence probe (see
# reports/exp1.md) showed needle-in-a-haystack saturating at ~750 steps, so the
# budget here is set from measured convergence with a wide margin rather than
# copied. Every row records the budget it actually ran.
STEPS = {"mod_add": 20000, "needle": 3000, "decimal": 6000, "parens": 4000}
BATCH = {"mod_add": 1000, "needle": 500, "decimal": 500, "parens": 500}
LSTM_SEEDS = (0, 1, 2)      # the recurrent baseline is ~4x the cost per step
EVAL_EVERY = 250

LR, WD = 1e-3, 1e-3

PROJECT = Path(__file__).parent
JSONL = PROJECT / "results.jsonl"
ALL_TASKS = ("mod_add", "needle", "decimal", "parens")
TASKS = tuple(os.environ.get("RT_TASKS", ",".join(ALL_TASKS)).split(","))

# Parameters are kept only for the cells the subspace analysis (exp5) reads:
# seeds 0-2 at full width. Saving all 40 would be ~4 GB of frozen noise.
SAVE_CELLS = {(t, m, 1024, s) for t in ALL_TASKS for m in MODES for s in (0, 1, 2)}


def done_cells() -> set:
    return {r["cell"] for r in read_rows(JSONL)
            if r.get("experiment", "").startswith(EXP) and "cell" in r}


def append_result(row: dict):
    _append(JSONL, row)


def main():
    done = done_cells()
    logging.info(f"{len(done)} cells already done; tasks = {TASKS}")

    for task_name in TASKS:
        task = get_task(task_name)
        logging.info(f"=== {task_name} — {task.describe} (chance {task.chance:.4f}) ===")

        for width in WIDTHS:
            for mode in MODES:
                for seed in SEEDS:
                    cell = f"{task_name}/{mode}/{width}/{seed}"
                    if cell in done:
                        continue
                    logging.info(f"  {cell}")
                    row = train_task(
                        task, mode=mode, d=width, n_layer=N_LAYER,
                        n_head=N_HEAD if width >= N_HEAD else 1,
                        seed=seed, steps=STEPS[task_name], batch=BATCH[task_name],
                        lr=LR, wd=WD, eval_every=EVAL_EVERY, log=True,
                    )
                    if (task_name, mode, width, seed) in SAVE_CELLS:
                        with open(PROJECT / f"params_{EXP}_{task_name}_{mode}_{width}_{seed}.pkl", "wb") as f:
                            pickle.dump({k: v for k, v in row["_params"].items()}, f)
                    row = strip_params(row)
                    row.update({"experiment": f"{EXP}_{task_name}", "cell": cell,
                                "name": f"main table — {cell}"})
                    append_result(row)
                    logging.info(f"  {cell}  seq_acc={row['test_seq_acc']:.4f} "
                                 f"tok_acc={row['test_tok_acc']:.4f} "
                                 f"({row['time_s']:.0f}s)")

        # LSTM baseline at width 1024, the paper's recurrent comparison
        for seed in LSTM_SEEDS:
            cell = f"{task_name}/lstm/1024/{seed}"
            if cell in done:
                continue
            logging.info(f"  {cell}")
            t0 = time.time()
            row = train_lstm(task, d=1024, seed=seed, steps=STEPS[task_name],
                             batch=BATCH[task_name], eval_every=EVAL_EVERY)
            row.update({"experiment": f"{EXP}_{task_name}", "cell": cell,
                        "name": f"main table — {cell}", "time_s": time.time() - t0})
            append_result(row)
            logging.info(f"  {cell}  seq_acc={row['test_seq_acc']:.4f} ({row['time_s']:.0f}s)")

    logging.info("exp1 complete")


if __name__ == "__main__":
    logging.info(f"devices: {jax.devices()}")
    main()
