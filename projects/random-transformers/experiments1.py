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
# The paper never states the head count. The authors' code uses 4 in every
# config file, so we match it. This is not a neutral choice for us: more heads
# means more independent random circuits for embedding-only training to select
# among, so 8 would have flattered the condition under test.
N_HEAD = 4

# Budget. The paper uses 10k steps x batch 1000 for the streamed tasks and 5000
# epochs x batch 4000 for modular addition. A convergence probe (see
# reports/exp1.md) showed needle-in-a-haystack saturating at ~750 steps, so the
# budget here is set from measured convergence with a wide margin rather than
# copied. Every row records the budget it actually ran.
# Budgets are set by the *slowest* condition, not the fastest. Calibrating on
# embedding-only training was the original error: it converges on needle by step
# 750, while a fully trained model at the same width is still climbing at 3000
# (0.965 and rising). These cover both.
STEPS = {"mod_add": 20000, "needle": 8000, "decimal": 8000, "parens": 6000}
BATCH = {"mod_add": 1000, "needle": 500, "decimal": 500, "parens": 500}
LSTM_SEEDS = (0, 1, 2)      # the recurrent baseline is ~4x the cost per step
EVAL_EVERY = 250

# The paper's Appendix D.3 gives only AdamW at lr 1e-3, weight decay 1e-3, and
# gradient clipping at 1.0. That is not sufficient to reproduce its results: a
# fully trained width-1024 model under those settings plateaus at 0.14 on
# needle-in-a-haystack and never recovers, even at the paper's own budget.
#
# The authors' released code (github.com/fjzzq2002/random_transformers,
# experiment_stream.py) additionally sets `warmup_steps = 500` and
# `lr_scheduler_type = "cosine"`, neither of which is stated in the paper. With
# both, lr 1e-3 works as published. We match the code.
#
# Embedding-only training reaches 1.00 with or without either, which is exactly
# why the omission is invisible in the paper's own tables — see
# reports/exp1-budget.md.
LR, WD = 1e-3, 1e-3
WARMUP = 500

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
                        lr=LR, wd=WD, warmup=WARMUP, eval_every=EVAL_EVERY, log=True,
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
                             batch=BATCH[task_name], eval_every=EVAL_EVERY,
                             warmup=WARMUP)
            row.update({"experiment": f"{EXP}_{task_name}", "cell": cell,
                        "name": f"main table — {cell}", "time_s": time.time() - t0})
            append_result(row)
            logging.info(f"  {cell}  seq_acc={row['test_seq_acc']:.4f} ({row['time_s']:.0f}s)")

    logging.info("exp1 complete")


if __name__ == "__main__":
    logging.info(f"devices: {jax.devices()}")
    main()
