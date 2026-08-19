"""exp4 — memorization: how many bits fit in the embeddings (H4).

Hypothesis (paper Table 3): on a task with no structure to exploit — 262,144
arbitrary (x, y) -> z associations — a random transformer stores far less than
a fully trained one, ~0.4 vs ~2.9 bits per trainable parameter. This is the
experiment that shows embedding-only training is not free: the frozen interior
cannot be used as storage, so everything has to fit in E and U.

The metric is deliberately about *training* data. There is no test split; the
question is capacity, not generalisation.

Usage:
    uv run python projects/random-transformers/scripts/run_experiments.py --bg exp4
"""

import dataclasses
import logging
import math
import pickle
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
sys.path.append(str(Path(__file__).resolve().parent))

import jax
import jax.numpy as jnp

from lib.model import init_params, split_params
from lib.results_io import append_result, read_rows
from lib.tasks import make_memorization
from lib.train import evaluate, strip_params, train_task

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

EXP = "exp4"
WIDTH = 128
N_LAYER, N_HEAD = 2, 4
MODES = ("random", "full")
SEEDS = (0, 1, 2)

# The paper runs 21,000 epochs at batch 2^15 over 262,144 pairs — ~168k steps.
# Memorisation is the one task where the budget genuinely matters, since the
# metric *is* how much got stored, so this keeps the same order of magnitude.
STEPS = 60_000
BATCH = 8192
EVAL_EVERY = 5_000
LR, WD, WARMUP = 1e-3, 1e-3, 500

BITS_PER_PAIR = math.log2(512)

PROJECT = Path(__file__).parent
JSONL = PROJECT / "results.jsonl"


def full_set_accuracy(params, task, chunk=8192):
    """Exact-match rate over all 262,144 associations."""
    toks, mask = task.test_set()
    seq, _ = evaluate(params, toks, mask, n_layer=N_LAYER, n_head=N_HEAD, batch=chunk)
    return seq


def main():
    done = {r["cell"] for r in read_rows(JSONL) if r.get("experiment") == EXP}
    task = make_memorization()
    logging.info(f"memorization — {task.describe} (chance {task.chance:.5f})")

    # Training-time monitoring uses a fixed subsample; the reported number is
    # the full-set pass below. Evaluating 262k examples every 250 steps would
    # cost more than the training.
    sub = 16_384
    full_test = task.test_set()
    monitor_task = dataclasses.replace(
        task, test_set=lambda: (full_test[0][:sub], full_test[1][:sub]))

    for mode in MODES:
        for seed in SEEDS:
            cell = f"memorization/{mode}/{WIDTH}/{seed}"
            if cell in done:
                continue
            logging.info(f"  {cell}")
            row = train_task(monitor_task, mode=mode, d=WIDTH, n_layer=N_LAYER, n_head=N_HEAD,
                             seed=seed, steps=STEPS, batch=BATCH, lr=LR, wd=WD, warmup=WARMUP,
                             eval_every=EVAL_EVERY, log=True)

            acc = full_set_accuracy(row["_params"], task)
            n_pairs = int(acc * 512 * 512)
            bits = n_pairs * BITS_PER_PAIR
            trainable = row["n_trainable_params"]

            with open(PROJECT / f"params_{EXP}_{mode}_{seed}.pkl", "wb") as f:
                pickle.dump(row["_params"], f)

            row = strip_params(row)
            row.update({
                "experiment": EXP, "cell": cell,
                "name": f"memorization — {mode} width {WIDTH} seed {seed}",
                "full_set_acc": acc,
                "memorized_pairs": n_pairs,
                "memorized_bits": bits,
                "bits_per_trainable_param": bits / trainable,
            })
            append_result(JSONL, row)
            logging.info(f"  {cell}  acc={acc:.4f}  bits={bits:,.0f}  "
                         f"bits/param={row['bits_per_trainable_param']:.2f} "
                         f"({row['time_s']:.0f}s)")

    logging.info("exp4 complete")


if __name__ == "__main__":
    logging.info(f"devices: {jax.devices()}")
    main()
