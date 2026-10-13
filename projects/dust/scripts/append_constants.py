"""Append the dataset and model constants the paper quotes to results.jsonl, computed from the data.

The paper's numeric check reads results.jsonl, so a constant that is only ever typed into prose
can be silenced but never verified. This puts them in the log, derived rather than asserted: the
unigram floors are counted from the token files, the parameter count from the model.

    ~/.local/bin/uv run --no-sync python projects/dust/scripts/append_constants.py
"""

import json
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.append(str(PROJECT.parents[1]))

import jax

from lib import data
from lib.model import Cfg, init_params, n_params

cfg = Cfg()
V, T = cfg.vocab, cfg.seq
val, test = data.load_rows("val"), data.load_rows("test")
floors = {}
for budget, rows_n in (("100K", 48), ("1M", 488), ("10M", 4880)):
    tr = data.load_rows("train_10M")[:rows_n, 1:].ravel()
    p = np.bincount(tr, minlength=V) + 1.0
    p /= p.sum()
    floors[budget] = dict(val=float(-np.log(p[val[:, 1:].ravel()]).mean()), test=float(-np.log(p[test[:, 1:].ravel()]).mean()))

row = dict(
    experiment="constants", name="constants|data-and-model", kind="constants",
    vocab=V, seq=T, tokens_per_update=data.BATCH_SEQS * T, sequences_per_update=data.BATCH_SEQS,
    updates=dict(data.STEPS), heldout_sequences=len(val), heldout_tokens=len(val) * T,
    chance_loss=float(np.log(V)), unigram_floor=floors, n_params=n_params(init_params(cfg, jax.random.PRNGKey(0))),
    n_layer=cfg.n_layer, n_embd=cfg.n_embd, n_head=cfg.n_head, windows=list(cfg.windows),
)
print(json.dumps(row)[:600])
with open(PROJECT / "results.jsonl", "a") as f:
    f.write(json.dumps(row) + "\n")
