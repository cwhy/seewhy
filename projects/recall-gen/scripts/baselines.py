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

from functools import partial

import numpy as np
import jax
import jax.numpy as jnp

sys.path.append(str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.core import Cfg, masked_mse
from lib import domains, evalsets
from lib.train import (Run, build_pools, build_mask, build_visible,
                       append_result, already_done)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

TAUS = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
LAMS_ICL = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
LAMBDAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)


def fit_ridge(X, mask, lam, vis=None):
    """Least-squares map from the visible coordinates to the HIDDEN ones, plus a bias.

    The target is `mask > 0.5`, not `~vis`. On a padded domain those differ by
    the padding, and regressing onto padding would both waste columns and make
    the returned matrix the wrong shape for `apply_ridge` to write back.
    """
    vis = (mask < 0.5) if vis is None else (np.asarray(vis) > 0.5)
    hid = mask > 0.5
    A = np.concatenate([X[:, vis], np.ones((X.shape[0], 1), np.float32)], 1)  # (N, dv+1)
    G = A.T @ A
    G[np.diag_indices_from(G)] += lam * X.shape[0] / 1000.0
    return np.linalg.solve(G, A.T @ X[:, hid]).astype(np.float32)             # (dv+1, dh)


def apply_ridge(W, q, mask, vis=None):
    vis = (mask < 0.5) if vis is None else (np.asarray(vis) > 0.5)
    hid = mask > 0.5
    flat = q.reshape(-1, mask.shape[0])
    A = np.concatenate([flat[:, vis], np.ones((flat.shape[0], 1), np.float32)], 1)
    out = np.zeros_like(flat)
    # Written to the HIDDEN coordinates, not to `~vis`: on a padded domain those
    # differ by the padding, and `~vis` would try to fill it.
    out[:, hid] = np.clip(A @ W, 0.0, 1.0)
    return out.reshape(q.shape)


@partial(jax.jit, static_argnums=())
def _in_context_ls(ctx, qry, mask, vis, lam):
    """Per-episode least squares from the visible coordinates to the hidden ones,
    fitted on the M context items and applied to the query.

    The soft look-up is a kernel smoother: it can only return a weighted average
    of context items it has seen. That is the right ceiling when items are drawn
    independently, and the wrong one when they share a low-dimensional generative
    structure — a latent-factor world's items lie near a subspace, so the visible
    half of a SEVENTEENTH item determines its hidden half through a linear map
    that sixteen examples can estimate, even though that item is not among them.

    Solved in the dual. With M = 16 examples and several hundred visible
    coordinates the primal is hopelessly underdetermined, but

        w = X^T (X X^T + lam I)^-1 Y

    needs only an M x M inverse and is the minimum-norm ridge solution. That form
    is also precisely what linear attention computes, which is what makes this
    the ceiling for THIS architecture rather than merely a strong baseline.
    """
    # Masked by multiplication rather than sliced: a boolean index is not
    # concrete under jit, and zeroed coordinates contribute nothing to any of
    # the inner products below, so the two forms agree exactly.
    X = ctx * vis                                        # (E, M, P) visible only
    Y = ctx * mask                                       # (E, M, P) hidden only
    Q = qry * vis                                        # (E, Q, P)
    G = jnp.einsum("emp,enp->emn", X, X)                 # (E, M, M) Gram
    G = G + lam * jnp.eye(G.shape[-1])[None]
    A = jnp.linalg.solve(G, Y)                           # (E, M, P)
    K = jnp.einsum("eqp,emp->eqm", Q, X)                 # (E, Q, M)
    return jnp.clip(jnp.einsum("eqm,emp->eqp", K, A), 0.0, 1.0)


@jax.jit
def _soft_lookup(ctx, qry, mask, tau, vis):
    """softmax(-d_visible / tau) weighted average of the context hidden halves."""
    d = (((qry[:, :, None, :] - ctx[:, None, :, :]) ** 2) * vis).sum(-1) / vis.sum()
    w = jax.nn.softmax(-d / tau, axis=-1)                       # (E,Q,M)
    return jnp.einsum("eqm,emp->eqp", w, ctx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--domain", default="mnist", choices=list(domains.DOMAINS))
    ap.add_argument("--rows", type=int, default=14,
                    help="hidden slices: image rows from the bottom, board files "
                         "from the queenside")
    ap.add_argument("--Q", type=int, default=4)
    ap.add_argument("--n_eval", type=int, default=512)
    ap.add_argument("--split", action="store_true",
                    help="class split with exp8's six conditions: digits 0-4 / 5-9 "
                         "for the two image domains, 24+ pieces / endgames for chess")
    ap.add_argument("--ctx_mode", default="iid", choices=list(evalsets.DRAWS),
                    help="what the context is made of — the B1 gate measurement")
    ap.add_argument("--knn_offset", type=int, default=0,
                    help="ranks to skip in knn mode; dials how informative the context is")
    a = ap.parse_args()

    # Q is part of the name because a knn context gives each query M/Q neighbours,
    # so Q changes what the context IS, not merely how many queries score on it.
    tag = "" if a.ctx_mode == "iid" else f"_{a.ctx_mode}" + (
        f"{a.knn_offset}" if a.ctx_mode == "knn" and a.knn_offset else "")
    dom_tag = "" if a.domain == "mnist" else f"{a.domain}_"
    exp = (f"baselines_{dom_tag}M{a.M}_r{a.rows}" + ("_split" if a.split else "") + tag
           + (f"_Q{a.Q}" if a.Q != 4 else ""))
    if already_done(exp):
        logging.info(f"{exp} already done — skipping")
        return

    # Each domain declares its own split: digit / garment classes 0-4 against
    # 5-9, chess 24+ pieces against endgames, and a cross-dataset pair's ten
    # training classes against the novel dataset's ten.
    classes = domains.get(a.domain).split
    split = dict(train_digits=classes[0], held_digits=classes[1]) if a.split else {}
    conds = None
    if a.split:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from experiments8 import SPLIT_CONDITIONS as conds
    rn = Run(exp_name=exp, name="baselines", domain=a.domain, M=a.M, Q=a.Q,
             mask_rows=a.rows, cfg=Cfg(d_in=domains.get(a.domain).d_in), **split)
    pools, labels = build_pools(rn)
    mask = build_mask(rn)
    vis = build_visible(rn)
    mean_img = pools["train"].mean(0)
    ev = evalsets.build(pools, mask, a.M, a.Q, a.n_eval, mean_img, conditions=conds,
                        ctx_mode=a.ctx_mode, labels=labels, knn_offset=a.knn_offset,
                        vis=vis)
    mask_j, vis_j = jnp.array(mask), jnp.array(vis)

    # ridge: fit on 50k, pick lambda on the remaining 10k of the train pool
    n_fit = int(pools["train"].shape[0] * 5 / 6)
    Xtr, Xva = pools["train"][:n_fit], pools["train"][n_fit:]
    best = (None, np.inf, None)
    for lam in LAMBDAS:
        W = fit_ridge(Xtr, mask, lam, vis)
        e = float(masked_mse(jnp.array(apply_ridge(W, Xva[:, None, :], mask, vis)),
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
        if domains.get(a.domain).kind == "board":
            # The all-empty board: ~70% of squares are empty, so this is the
            # reference point any piece accuracy has to be read against.
            empty = np.zeros((*qry.shape[:2], *domains.get(a.domain).shape), np.float32)
            empty[..., 0] = 1.0
            row["sq_acc_empty"] = domains.piece_accuracy(
                empty.reshape(qry.shape), np.asarray(qry), mask, a.domain)
        row["mse_ridge"] = float(masked_mse(
            jnp.array(apply_ridge(W, np.asarray(qry), mask, vis)), qry, mask_j))
        soft = {}
        for tau in TAUS:
            soft[tau] = float(masked_mse(
                _soft_lookup(es.ctx, qry, mask_j, tau, vis_j), qry, mask_j))
        row["mse_knn_by_tau"] = {str(t): v for t, v in soft.items()}
        t_star = min(soft, key=soft.get)
        row["mse_knn"], row["knn_tau"] = soft[t_star], t_star
        icl = {}
        for lam in LAMS_ICL:
            icl[lam] = float(masked_mse(
                _in_context_ls(es.ctx, qry, mask_j, vis_j, lam), qry, mask_j))
        row["mse_icl_by_lam"] = {str(l): v for l, v in icl.items()}
        l_star = min(icl, key=icl.get)
        row["mse_icl"], row["icl_lam"] = icl[l_star], l_star
        row.update({f"n{k[3:]}": v / es.mse_mean for k, v in row.items()
                    if k.startswith("mse_") and isinstance(v, float)})
        out[cond] = row
        logging.info(
            f"  {cond:<16} mean={row['mse_mean']:.4f}  zeros={row['n_zeros']:.3f}"
            f"  ridge={row['mse_ridge']:.4f}"
            f" ({row['n_ridge']:.3f})  nn1={row['mse_nn1']:.4f} ({row['n_nn1']:.3f})"
            f"  knn={row['mse_knn']:.4f} ({row['n_knn']:.3f}, tau={t_star})"
            f"  icl={row['mse_icl']:.4f} ({row['n_icl']:.3f}, lam={l_star})")

    append_result(dict(experiment=exp,
                       name=f"baselines {a.domain} M={a.M} mask_rows={a.rows} ctx={a.ctx_mode}"
                            + (f" offset={a.knn_offset}" if a.ctx_mode == "knn" else ""),
                       domain=a.domain, M=a.M, Q=a.Q, mask_rows=a.rows, n_eval=a.n_eval,
                       ctx_mode=a.ctx_mode, knn_offset=a.knn_offset,
                       ridge_lambda=lam_star, time_s=0.0, baselines=out))
    logging.info(f"wrote {exp}")


if __name__ == "__main__":
    main()
