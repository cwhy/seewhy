"""Is B1's number actually coming from the context? Swap the context and see.

exp20's best absent-target error, 0.505, is below both the soft-look-up ceiling
(0.552) and ridge (0.631), and it was tempting to read that as proof of context
use: neither a pure look-up nor a context-free linear map can produce it. But
exp18 reaches 0.573 on a task whose context is worthless, so beating ridge only
shows a nonlinear prior, not context use. The claim needs measuring.

So: hold the model and the queries fixed and replace the context.

  proper     the query's own 16 nearest neighbours — what it was trained on
  swapped    another episode's knn context: same statistics, same near-duplicate
             structure, but about a DIFFERENT query
  iid        16 unrelated images, the original task's context

If the error is flat across the three, the model is not reading the context and
its score is a prior. If `proper` is well below the other two, the gap is what
the context is worth to it — and `swapped` is the honest control, since it
differs from `proper` in nothing but which query the neighbours belong to.

Usage:
    uv run --no-sync python projects/recall-gen/scripts/ctx_ablation.py
"""

import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

PROJECT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT.parents[1]))          # repo root LAST — see workflow.md
sys.path.insert(0, str(PROJECT))

from lib.core import Cfg, row_mask, masked_mse
from lib import evalsets
from lib.train import Run, build_pools, make_eval, append_result, already_done

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

CFG = Cfg(d_model=256, n_layers=4, dk=64, n_heads=4, n_tokens=17)
MODELS = {
    "exp20_best":  "params_exp20_best.pkl",    # the 0.505 point
    "exp20_final": "params_exp20.pkl",         # after it degrades to 0.665
    "exp23_best":  "params_exp23_best.pkl",    # same-class arm, for contrast
}
COND = "D_novel_absent"        # the only condition where the context has work to do
EXP = "ctx_ablation"


def main():
    if already_done(EXP):
        logging.info(f"{EXP} already in results.jsonl — skipping")
        return

    rn = Run(exp_name="", name="", M=16, Q=1, mask_rows=14, cfg=CFG)
    pools, labels = build_pools(rn)
    mask = row_mask(14)
    mask_j = jnp.array(mask)
    mean_img = pools["train"].mean(0)

    knn = evalsets.build(pools, mask, 16, 1, 512, mean_img, ctx_mode="knn", labels=labels)
    iid = evalsets.build(pools, mask, 16, 1, 512, mean_img, ctx_mode="iid", labels=labels)
    es = knn[COND]
    contexts = {
        "proper": es.ctx,
        # Roll by one episode: every context is a real knn context, just not this
        # query's. Same distribution, same near-duplicate structure, wrong query.
        "swapped": jnp.roll(es.ctx, 1, axis=0),
        "iid": iid[COND].ctx,
    }

    eval_fn = make_eval(rn, mask_j)
    out = {}
    for mk, fname in MODELS.items():
        path = PROJECT / fname
        if not path.exists():
            logging.info(f"{mk}: no checkpoint at {fname}, skipping")
            continue
        with open(path, "rb") as f:
            p = jax.tree_util.tree_map(jnp.asarray, pickle.load(f))
        row = {}
        for tag, ctx in contexts.items():
            se = 0.0
            for i in range(0, ctx.shape[0], 128):
                c, q = ctx[i:i + 128], es.qry[i:i + 128]
                pred, _ = eval_fn(p, c, q)
                se += (c.shape[0] / ctx.shape[0]) * float(masked_mse(pred, q, mask_j))
            row[tag] = se / es.mse_mean
        row["context_worth"] = row["swapped"] - row["proper"]
        out[mk] = row
        logging.info(f"  {mk:<12} proper={row['proper']:.3f}  swapped={row['swapped']:.3f}"
                     f"  iid={row['iid']:.3f}   context worth {row['context_worth']:+.3f}")

    append_result(dict(experiment=EXP,
                       name="context ablation: same model and queries, three contexts",
                       M=16, Q=1, mask_rows=14, n_eval=512, condition=COND,
                       time_s=0.0, ablation=out))
    logging.info(f"wrote {EXP}")


if __name__ == "__main__":
    main()
