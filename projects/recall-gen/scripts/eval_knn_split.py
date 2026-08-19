"""Score the digit-split networks on nearest-neighbour contexts.

The split networks were all trained with contexts of sixteen unrelated images.
This is evaluation only: same weights, a context built instead from the query's
own sixteen nearest neighbours (visible-half distance, within the same pool).

It answers a question the per-panel figure cannot: on unseen digit classes, does
an informative context recover what an uninformative one loses? Writes row
`knn_split_eval`.

Usage:
    uv run --no-sync python projects/recall-gen/scripts/eval_knn_split.py
"""
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

PROJECT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT.parents[1]))
sys.path.insert(0, str(PROJECT))

from lib.core import Cfg, row_mask, masked_mse
from lib import evalsets
from lib.train import Run, build_pools, make_eval, append_result, already_done

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

EXP = "knn_split_eval"
CFG = Cfg(d_model=256, n_layers=4, dk=64, n_heads=4, n_tokens=20)
SPLIT = {
    "A_seen_present":  ("train",     True),
    "B_novel_present": ("held",      True),
    "C_seen_absent":   ("train",     False),
    "D_novel_absent":  ("held",      False),
    "E_same_present":  ("held_same", True),
    "F_same_absent":   ("held_same", False),
}
NETS = {"recall": "exp8", "completion": "exp9", "frozen": "exp29"}


def main():
    if already_done(EXP):
        logging.info(f"{EXP} already done — skipping")
        return
    rn = Run(exp_name="", name="", M=16, Q=1, mask_rows=14, cfg=CFG,
             train_digits=(0, 1, 2, 3, 4), held_digits=(5, 6, 7, 8, 9),
             conditions=SPLIT)
    pools, labels = build_pools(rn)
    mask = row_mask(14)
    mask_j = jnp.array(mask)
    mean_img = pools["train"].mean(0)
    eval_fn = make_eval(rn, mask_j)

    out = {}
    for ctx_mode in ("iid", "knn"):
        ev = evalsets.build(pools, mask, 16, 1, 512, mean_img,
                            conditions=SPLIT, labels=labels, ctx_mode=ctx_mode)
        out[ctx_mode] = {"lookup": {c: ev[c].mse_nn / ev[c].mse_mean for c in ev}}
        for tag, exp in NETS.items():
            with open(PROJECT / f"params_{exp}.pkl", "rb") as f:
                p = jax.tree_util.tree_map(jnp.asarray, pickle.load(f))
            row = {}
            for c, es in ev.items():
                pred, argmin = eval_fn(p, es.ctx, es.qry)
                row[c] = {"nmse": float(masked_mse(pred, es.qry, mask_j)) / es.mse_mean,
                          "id_acc": float((argmin == es.tgt_idx).mean()) if es.present else None}
            out[ctx_mode][tag] = row
            logging.info(f"  {ctx_mode:<4} {tag:<11} " + "  ".join(
                f"{c[0]}={row[c]['nmse']:.3f}" for c in
                ("A_seen_present", "E_same_present", "B_novel_present",
                 "C_seen_absent", "F_same_absent", "D_novel_absent")))

    append_result(dict(experiment=EXP, M=16, Q=1, mask_rows=14, n_eval=512, time_s=0.0,
                       name="digit-split networks scored on iid and knn contexts",
                       train_digits=[0, 1, 2, 3, 4], held_digits=[5, 6, 7, 8, 9],
                       nets=NETS, eval=out))
    logging.info(f"wrote {EXP}")


if __name__ == "__main__":
    main()
