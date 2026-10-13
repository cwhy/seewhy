"""FineWeb token files, as numpy, and the batch order the reference uses.

`scripts/prepare_data.py` writes `data/{train_100K,train_1M,train_10M,val,test}.npy`,
each a [N, 2049] array of token ids. Training consumes `steps` batches of 8
sequences; the first `steps * 8` rows of the budget's file are shuffled once with
the run's seed (the reference does the same with torch.randperm(seed + 1), so the
permutation differs but the construction does not).
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np

DATA = Path(__file__).resolve().parents[1] / "data"
BATCH_SEQS = 8                      # sequences per update; 8 * 2048 = 16,384 tokens
STEPS = {"100K": 6, "1M": 61, "10M": 610, "100M": 6100}
EVAL_EVERY = {"100K": 1, "1M": 2, "10M": 10, "100M": 200}


def load_rows(name: str) -> np.ndarray:
    return np.load(DATA / f"{name}.npy").astype(np.int32)


def train_batches(budget: str, seed: int, steps: int | None = None) -> np.ndarray:
    """[steps, 8, 2049] -- the shuffled training stream for one run."""
    n = (steps or STEPS[budget]) * BATCH_SEQS
    rows = load_rows(f"train_{budget}")[: STEPS[budget] * BATCH_SEQS]
    order = np.random.default_rng(seed + 1).permutation(len(rows))
    rows = rows[order][:n]
    return rows.reshape(-1, BATCH_SEQS, rows.shape[1])


def split_xy(rows):
    """Inputs and next-token targets from [.., 2049] rows."""
    return rows[..., :-1], rows[..., 1:]


def eval_batches(split: str, batch: int = 32):
    rows = load_rows(split)
    assert len(rows) % batch == 0
    x, y = split_xy(rows)
    return [(jnp.asarray(x[i:i + batch]), jnp.asarray(y[i:i + batch])) for i in range(0, len(rows), batch)]
