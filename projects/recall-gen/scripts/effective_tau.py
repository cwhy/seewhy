"""How sharp is a trained model's retrieval kernel? Fit its effective temperature.

Report 7 used identification accuracy as the definition of "retrieval". That was
wrong, and the model-free baseline shows why: the soft look-up is ONE mechanism
with one knob, and the two objectives want opposite ends of it. On the knn
context, sharpening from tau=0.03 to tau=0.003 is worth 0.36 on the
target-present objective and costs 0.12 on the absent-target one. Identification
accuracy only asks whether the kernel is sharp enough to pick the exact item, so
it measures the knob, not a separate ability.

This measures the knob directly. For each checkpoint, find the tau whose
soft-look-up output most closely matches the MODEL's output — its effective
temperature — by minimising the hidden-pixel distance between the two. Low tau
means the model is behaving like an argmax over the context; high tau means it is
blending. The prediction the rewrite rests on: recall training drives tau down,
and a frozen mixer cannot follow it.

Each model is measured on the context construction it trained on, since a
temperature is only comparable within a context distribution.

Usage:
    uv run --no-sync python projects/recall-gen/scripts/effective_tau.py
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

EXP = "effective_tau2"
# Fit on BOTH conditions. The first attempt fitted only the absent-target one and
# gave tau=0.03 for exp1 and exp26, which have id(B) ~ 1.0 and must be sharp:
# when no context item is close, the softmax is diffuse at ANY temperature, so
# that fit measures the distance distribution as much as the model. The
# target-present condition is where sharpness is actually exercised, and is the
# number to read; the absent fit is kept alongside to show the confound rather
# than hide it.
CONDS = ("B_novel_present", "D_novel_absent")
TAUS = tuple(float(f"{t:.5g}") for t in
             [3e-4 * (10 ** (i / 4)) for i in range(18)])   # 3e-4 .. ~3, 4/decade

# (checkpoint, context mode, Q, n_tokens, knn_offset) — each model is measured on
# the context distribution it trained on. The offset matters: exp27 and exp28
# trained with their neighbours 64 and 512 ranks out, and scoring them on
# rank-0 neighbours would measure a distribution they never saw.
MODELS = {
    "exp1_final":   ("params_exp1.pkl",       "iid", 4, 20, 0),
    "exp20_best":   ("params_exp20_best.pkl", "knn", 1, 17, 0),
    "exp20_final":  ("params_exp20.pkl",      "knn", 1, 17, 0),
    "exp24_best":   ("params_exp24_best.pkl", "knn", 1, 17, 0),
    "exp24_final":  ("params_exp24.pkl",      "knn", 1, 17, 0),
    "exp26_final":  ("params_exp26.pkl",      "iid", 4, 20, 0),
    # The retrievability dial: same freeze, distractors progressively further
    # apart. If sharpness can rise from context geometry alone, tau* should fall
    # across these even though none of them can change its kernel.
    "exp27_final":  ("params_exp27.pkl",      "knn", 1, 17, 64),
    "exp28_final":  ("params_exp28.pkl",      "knn", 1, 17, 512),
}


@jax.jit
def soft_lookup(ctx, qry, mask, tau):
    """softmax(-d_visible / tau) weighted average of the context hidden halves."""
    vis = 1.0 - mask
    d = (((qry[:, :, None, :] - ctx[:, None, :, :]) ** 2) * vis).sum(-1) / vis.sum()
    w = jax.nn.softmax(-d / tau, axis=-1)
    return jnp.einsum("eqm,emp->eqp", w, ctx)


def main():
    if already_done(EXP):
        logging.info(f"{EXP} already in results.jsonl — skipping")
        return
    mask = row_mask(14)
    mask_j = jnp.array(mask)
    out = {}
    cache = {}

    for mk, (fname, mode, Q, n_tok, off) in MODELS.items():
        path = PROJECT / fname
        if not path.exists():
            logging.info(f"{mk}: no checkpoint at {fname}, skipping")
            continue
        cfg = Cfg(d_model=256, n_layers=4, dk=64, n_heads=4, n_tokens=n_tok)
        rn = Run(exp_name="", name="", M=16, Q=Q, mask_rows=14, cfg=cfg)
        if (mode, Q, off) not in cache:
            pools, labels = build_pools(rn)
            cache[(mode, Q, off)] = evalsets.build(pools, mask, 16, Q, 512,
                                                   pools["train"].mean(0),
                                                   ctx_mode=mode, labels=labels,
                                                   knn_offset=off)
        ev = cache[(mode, Q, off)]

        with open(path, "rb") as f:
            p = jax.tree_util.tree_map(jnp.asarray, pickle.load(f))
        eval_fn = make_eval(rn, mask_j)

        row = {"ctx_mode": mode, "knn_offset": off}
        for cond in CONDS:
            es = ev[cond]
            pred, _ = eval_fn(p, es.ctx, es.qry)
            # Distance from the MODEL's output to each temperature's look-up output.
            d = {t: float(masked_mse(soft_lookup(es.ctx, es.qry, mask_j, t), pred, mask_j))
                 for t in TAUS}
            t_star = min(d, key=d.get)
            row[cond] = {
                "tau_star": t_star,
                "nmse": float(masked_mse(pred, es.qry, mask_j)) / es.mse_mean,
                "dist_by_tau": {str(t): v for t, v in d.items()},
            }
        out[mk] = row
        logging.info(f"  {mk:<12} ctx={mode}{off if mode=='knn' else '':<4} "
                     f"tau*(present)={row['B_novel_present']['tau_star']:<8} "
                     f"tau*(absent)={row['D_novel_absent']['tau_star']:<8} "
                     f"B={row['B_novel_present']['nmse']:.3f} "
                     f"D={row['D_novel_absent']['nmse']:.3f}")

    append_result(dict(experiment=EXP, M=16, mask_rows=14, n_eval=512, time_s=0.0,
                       name="effective temperature of each checkpoint's retrieval kernel",
                       conditions_fitted=list(CONDS), taus=list(TAUS),
                       effective_tau=out))
    logging.info(f"wrote {EXP}")


if __name__ == "__main__":
    main()
