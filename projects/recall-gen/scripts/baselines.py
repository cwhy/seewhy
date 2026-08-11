"""Model-independent reference points for every eval condition.

These do not depend on any trained model, so they live in their own
`results.jsonl` row (`experiment: "baselines_M{M}_r{rows}"`) and are computed
once. Four references, spanning the two things a completion could be built from:

  mean       predict the train-set mean image                 — the trivial prior
  ridge      a global linear inpainter, visible -> hidden,     — the best you can do
             fitted by ridge regression on the train pool        from the DISTRIBUTION,
                                                                 ignoring the context
  nn1        copy the hidden half of the context image whose  — the best you can do
             visible half is closest to the query                by pure LOOK-UP
  knn_soft   softmax(-d/tau)-weighted average of the context  — the best you can do
             images' hidden halves                              by SOFT look-up; this
                                                                 is exactly the shape of
                                                                 computation linear
                                                                 attention can perform

`knn_soft` matters most: it is the strongest thing reachable by attending to the
context alone. A recall-trained model that beats it is doing something its
training objective never asked for.

Usage:
    uv run python projects/recall-gen/scripts/baselines.py [--M 16] [--rows 14]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

sys.path.append(str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.core import row_mask, masked_mse, PIX
from lib import evalsets
from lib.train import Run, build_pools, append_result, already_done

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

TAUS = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
LAMBDAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)


def fit_ridge(X, mask, lam):
    """Least-squares map from the visible pixels to the hidden ones, plus a bias."""
    vis = mask < 0.5
    A = np.concatenate([X[:, vis], np.ones((X.shape[0], 1), np.float32)], 1)  # (N, dv+1)
    G = A.T @ A
    G[np.diag_indices_from(G)] += lam * X.shape[0] / 1000.0
    return np.linalg.solve(G, A.T @ X[:, ~vis]).astype(np.float32)            # (dv+1, dh)


def apply_ridge(W, q, mask):
    vis = mask < 0.5
    flat = q.reshape(-1, PIX)
    A = np.concatenate([flat[:, vis], np.ones((flat.shape[0], 1), np.float32)], 1)
    out = np.zeros_like(flat)
    out[:, ~vis] = np.clip(A @ W, 0.0, 1.0)
    return out.reshape(q.shape)


@jax.jit
def _soft_lookup(ctx, qry, mask, tau):
    """softmax(-d_visible / tau) weighted average of the context hidden halves."""
    vis = 1.0 - mask
    d = (((qry[:, :, None, :] - ctx[:, None, :, :]) ** 2) * vis).sum(-1) / vis.sum()
    w = jax.nn.softmax(-d / tau, axis=-1)                       # (E,Q,M)
    return jnp.einsum("eqm,emp->eqp", w, ctx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--rows", type=int, default=14)
    ap.add_argument("--Q", type=int, default=4)
    ap.add_argument("--n_eval", type=int, default=512)
    ap.add_argument("--split", action="store_true",
                    help="digit split 0-4 / 5-9 with exp8's six conditions")
    a = ap.parse_args()

    exp = f"baselines_M{a.M}_r{a.rows}" + ("_split" if a.split else "")
    if already_done(exp):
        logging.info(f"{exp} already done — skipping")
        return

    split = dict(train_digits=(0, 1, 2, 3, 4), held_digits=(5, 6, 7, 8, 9)) if a.split else {}
    conds = None
    if a.split:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from experiments8 import SPLIT_CONDITIONS as conds
    rn = Run(exp_name=exp, name="baselines", M=a.M, Q=a.Q, mask_rows=a.rows, **split)
    pools, _ = build_pools(rn)
    mask = row_mask(a.rows)
    mean_img = pools["train"].mean(0)
    ev = evalsets.build(pools, mask, a.M, a.Q, a.n_eval, mean_img, conditions=conds)
    mask_j = jnp.array(mask)

    # ridge: fit on 50k, pick lambda on the remaining 10k of the train pool
    n_fit = int(pools["train"].shape[0] * 5 / 6)
    Xtr, Xva = pools["train"][:n_fit], pools["train"][n_fit:]
    best = (None, np.inf, None)
    for lam in LAMBDAS:
        W = fit_ridge(Xtr, mask, lam)
        e = float(masked_mse(jnp.array(apply_ridge(W, Xva[:, None, :], mask)),
                             jnp.array(Xva[:, None, :]), mask_j))
        logging.info(f"  ridge lam={lam:<7} holdout mse={e:.5f}")
        if e < best[1]:
            best = (lam, e, W)
    lam_star, _, W = best
    logging.info(f"  ridge lambda* = {lam_star}")

    out = {}
    for cond, es in ev.items():
        qry = es.qry
        row = {"mse_mean": es.mse_mean, "mse_nn1": es.mse_nn}
        # Predicting black. MNIST is mostly background, so this is the degenerate
        # answer a model falls into when it has nothing; worth knowing where it sits.
        row["mse_zeros"] = float(masked_mse(jnp.zeros_like(qry), qry, mask_j))
        row["mse_ridge"] = float(masked_mse(
            jnp.array(apply_ridge(W, np.asarray(qry), mask)), qry, mask_j))
        soft = {}
        for tau in TAUS:
            soft[tau] = float(masked_mse(
                _soft_lookup(es.ctx, qry, mask_j, tau), qry, mask_j))
        row["mse_knn_by_tau"] = {str(t): v for t, v in soft.items()}
        t_star = min(soft, key=soft.get)
        row["mse_knn"], row["knn_tau"] = soft[t_star], t_star
        row.update({f"n{k[3:]}": v / es.mse_mean for k, v in row.items()
                    if k.startswith("mse_") and isinstance(v, float)})
        out[cond] = row
        logging.info(
            f"  {cond:<16} mean={row['mse_mean']:.4f}  zeros={row['n_zeros']:.3f}"
            f"  ridge={row['mse_ridge']:.4f}"
            f" ({row['n_ridge']:.3f})  nn1={row['mse_nn1']:.4f} ({row['n_nn1']:.3f})"
            f"  knn={row['mse_knn']:.4f} ({row['n_knn']:.3f}, tau={t_star})")

    append_result(dict(experiment=exp, name=f"baselines M={a.M} mask_rows={a.rows}",
                       M=a.M, Q=a.Q, mask_rows=a.rows, n_eval=a.n_eval,
                       ridge_lambda=lam_star, time_s=0.0, baselines=out))
    logging.info(f"wrote {exp}")


if __name__ == "__main__":
    main()
