"""Recall measured as "did something from the context come back", not "which one".

`identification` is argmin over the sixteen context items: it scores 1 for the
target and 0 for everything else, including a near-duplicate sitting 0.013 away.
On the synthetic prior a world's latent dimension is drawn as low as 1, so
sixteen items can lie almost on a line and near-duplicates are the norm rather
than the exception. Against that data the metric charges the network for a
property of the prior.

This script measures the thing that does not care which copy came back.

    cost        mse(picked item, true target) — what the naming mistake actually
                costs in the output space. Zero when the pick is a duplicate.
    cost_rand   the same for a uniformly random context item: the reference for
                "no retrieval happened at all".
    R           1 - cost / cost_rand. 1.0 = the pick is free, 0.0 = no better
                than guessing. Degrades smoothly instead of stepping.
    d_ctx       mse(output, nearest context item) — is the output committed to
                context content at all, or floating between items?
    present/absent
                nmse with the answer in the context against the same queries
                with it removed. The direct test of whether the context is used.

All on hidden coordinates, normalised by the mean-item error, so every number is
comparable to the nmse figures the project already quotes.

    .venv/bin/python projects/recall-gen/scripts/recall_quality.py exp45 exp49 exp54
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import jax.numpy as jnp

PROJECT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT.parents[1]))
sys.path.insert(0, str(PROJECT))

from lib import splitfig, domains
from lib.core import build_tokens, masked_mse, predict
from rescore import rows as read_rows, run_from_row
from diag_identification import episode_geometry, quartiles

import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")


def analyse(exp, row, n_eval=512, chunk=128, as_domain=None, M=None):
    """Score a checkpoint, optionally under a different domain than it trained on.

    A synth-trained network is scored against MNIST, Fashion-MNIST or chess by
    naming `synth_to_<target>`: the A band stays synthetic training worlds and the
    B band becomes the real dataset, so "seen" and "novel" keep their meaning and
    the novel side is the transfer question. Follows `standard_eval.evaluate_one`.
    """
    rn = run_from_row(row)
    cfg = rn.cfg
    domain = as_domain or rn.domain
    dom = domains.get(domain)
    if dom.d_in != cfg.d_in:
        raise ValueError(f"{exp} has d_in={cfg.d_in}, {domain!r} needs {dom.d_in}")
    mask_rows = dom.n_mask_default if as_domain else rn.mask_rows
    if as_domain:
        tr, hd = dom.split
    else:
        tr = tuple(rn.train_digits) if rn.train_digits else None
        hd = tuple(rn.held_digits) if rn.held_digits else None
    M = rn.M if M is None else M
    _, mask, ev = splitfig.build(domain, mask_rows, tr, hd, cfg,
                                 ctx_mode=row.get("ctx_mode", "iid"),
                                 Q=rn.Q, M=M, n_eval=n_eval)
    params = splitfig.load_params(exp)
    mask_j = jnp.array(mask)
    out = {"M": M}
    # The absent conditions need no pick, only the error. They are recomputed
    # here rather than read from the row's `final` block because that block was
    # written at the TRAINING context length, and this function is used to score
    # one checkpoint at several test lengths.
    for cond in ("C_seen_absent", "D_novel_absent"):
        es = ev[cond]
        se = 0.0
        for i in range(0, es.ctx.shape[0], chunk):
            c, q = es.ctx[i:i + chunk], es.qry[i:i + chunk]
            pr = predict(params, c, q, mask_j, cfg)[:, :1, :]
            se += (c.shape[0] / es.ctx.shape[0]) * float(
                masked_mse(pr, q[:, :1], mask_j))
        out[cond] = {"nmse": se / es.mse_mean}
    for cond in ("A_seen_present", "B_novel_present"):
        es = ev[cond]
        preds = []
        for i in range(0, es.ctx.shape[0], chunk):
            preds.append(np.asarray(predict(params, es.ctx[i:i + chunk],
                                            es.qry[i:i + chunk], mask_j, cfg)[:, 0, :]))
        pred = np.concatenate(preds)
        g = episode_geometry(pred, np.asarray(es.ctx), np.asarray(es.qry)[:, 0],
                             np.asarray(es.tgt_idx)[:, 0], mask)
        C, T, pick = g["C"], g["T"], g["pick"]
        E, M, _ = C.shape
        ar = np.arange(E)
        mm = es.mse_mean

        cost = ((C[ar, pick] - T) ** 2).mean(-1) / mm
        d_all = ((C - T[:, None, :]) ** 2).mean(-1) / mm          # (E,M)
        cost_rand = d_all.mean(-1)                                 # uniform pick
        d_ctx = (((g["P"][:, None, :] - C) ** 2).mean(-1)).min(-1) / mm
        spread = (((C - T[:, None, :]) ** 2).mean(-1)).mean(-1) / mm
        qi = quartiles(g["margin"])
        out[cond] = dict(
            id_acc=float(g["hit"].mean()),
            nmse=float(g["err"].mean() / mm),
            cost=float(cost.mean()), cost_rand=float(cost_rand.mean()),
            R=float(1.0 - cost.mean() / cost_rand.mean()),
            d_ctx=float(d_ctx.mean()),
            committed=float((d_ctx / spread).mean()),
            by_q=[dict(q=int(q), id_acc=float(g["hit"][qi == q].mean()),
                       cost=float(cost[qi == q].mean()),
                       cost_rand=float(cost_rand[qi == q].mean()),
                       R=float(1.0 - cost[qi == q].mean() / cost_rand[qi == q].mean()),
                       d_ctx=float(d_ctx[qi == q].mean()),
                       committed=float((d_ctx[qi == q] / spread[qi == q]).mean()))
                  for q in range(4)],
        )
    if as_domain or M != rn.M:
        return out
    f = row["final"]
    out["present_absent"] = {
        "A_present": f["A_seen_present"]["nmse"], "C_absent": f["C_seen_absent"]["nmse"],
        "B_present": f["B_novel_present"]["nmse"], "D_absent": f["D_novel_absent"]["nmse"]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exps", nargs="+")
    ap.add_argument("--as-domain", default=None)
    a = ap.parse_args()
    by_exp = {r["experiment"]: r for r in read_rows()}
    for exp in a.exps:
        # `snapshot_best` writes params_<exp>_best.pkl, which has no row of its
        # own; it is the same Run, scored at a different step.
        row = by_exp.get(exp) or by_exp[exp.rsplit("_best", 1)[0]]
        r = analyse(exp, row, as_domain=a.as_domain)
        logging.info(f"\n== {exp}" + (f"  scored as {a.as_domain}" if a.as_domain else ""))
        if "present_absent" in r:
            pa = r["present_absent"]
            logging.info(f"  uses the context:  present {pa['A_present']:.4f}  vs "
                         f"absent {pa['C_absent']:.4f}   ({pa['C_absent']/pa['A_present']:.1f}x "
                         f"better when the answer is there)")
        for cond in ("A_seen_present", "B_novel_present"):
            d = r[cond]
            logging.info(f"  {cond}")
            logging.info(f"    identification {d['id_acc']:.3f}   "
                         f"recall quality R {d['R']:.3f}   "
                         f"committed {d['committed']:.3f}   "
                         f"nmse {d['nmse']:.4f}")
            logging.info("     quartile   id_acc       R   committed    cost")
            for q in d["by_q"]:
                logging.info(f"       {q['q']}       {q['id_acc']:.3f}   {q['R']:.3f}   "
                             f"  {q['committed']:.3f}    {q['cost']:.4f}")


if __name__ == "__main__":
    main()
