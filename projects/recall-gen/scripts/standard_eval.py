"""The project's standard evaluation. One row per checkpoint, every cell filled.

Reports 12 to 15 each rebuilt their own scoring by hand, and each one found a
way the numbers could mislead that the previous had missed:

  * report 12   present and absent conditions need SHARED queries, or their
                normalisers differ by ~2% and put a floor under every comparison
  * report 13   a class split confounds "what the network has seen" with "what
                it is searching among"
  * report 14   normalised error and piece accuracy disagree in SIGN on sparse
                boards, and identification has a ceiling below 1.0 whenever two
                items can share a hidden half
  * report 15   identification tracks the nearest-rival distance of the pool,
                which no earlier report quoted; and a knn context changes that
                distance, so context type and pool are not separable by accident

This script is those four lessons as a fixed procedure. Every number a report
wants is computed here, together with the reference point it has to be read
against, so a future report quotes rather than re-derives.

What one row contains, for a checkpoint:

    scores[ctx][cond]     nmse, id_acc and (boards) sq_acc, for ctx in iid/knn
    pool[ctx][cond]       id_ceiling  — what a PERFECT reconstruction identifies
                          margin_med  — median distance, target to nearest rival
                          margin_p25  — its lower quartile
    refs                  the model-free references from the baselines row

Both context types are scored at the same Q, because a knn context hands each
query M/Q neighbours: changing Q changes what the context IS, and a comparison
across different Q is not a comparison of context type.

Usage:
    uv run --no-sync python projects/recall-gen/scripts/standard_eval.py exp36 exp40
    uv run --no-sync python projects/recall-gen/scripts/standard_eval.py --all
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT.parents[1]))          # repo root LAST — see workflow.md
sys.path.insert(0, str(PROJECT))

from lib import domains, splitfig
from lib.train import append_result, already_done
from rescore import rows as read_rows, run_from_row

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

SUFFIX = "_stdeval"
# Context types every checkpoint is scored under. A run's OWN training context
# type is added to these when it is neither — a network trained on single-world
# episodes (`ctx_mode="class"`) scored only on iid mixtures is being asked a
# question it was never trained on, and reads 1.35 where its own task reads 0.41.
CONTEXTS = ("iid", "knn")
Q_STD = 4                      # same Q on both sides; see the module docstring
# The eval-set draw is fixed and RECORDED, because the pool diagnostics are
# properties of a draw: at 512 episodes a median margin moves by a few
# hundredths between seeds, which is enough to look like a discrepancy when one
# report quotes 0.54 and another 0.50 for the same pool. Numbers from different
# seeds are not directly comparable; quote one row.
SEED_STD = 20260825
PRESENT = ("A_seen_present", "E_same_present", "B_novel_present")


def pool_diagnostics(domain, mask_rows, train_cls, held_cls, cfg, ctx_mode, Q, M):
    """Everything about the eval sets that does not involve a trained network.

    `M` is passed rather than defaulted: the nearest rival is the closest of the
    OTHER M-1 context items, so a diagnostic built at M=16 does not describe an
    episode scored at M=256.
    """
    _, mask, ev = splitfig.build(domain, mask_rows, train_cls, held_cls, cfg,
                                 ctx_mode=ctx_mode, Q=Q, M=M, seed=SEED_STD)
    out = {}
    for cond in PRESENT:
        if cond not in ev:
            continue
        m = splitfig.nearest_distractor(ev, mask, cond)
        out[cond] = {
            "id_ceiling": splitfig.oracle_id_acc(ev, mask, cond),
            "margin_med": float(np.median(m)),
            "margin_p25": float(np.percentile(m, 25)),
        }
    return out


def evaluate_one(exp: str, row: dict, as_domain: str | None = None) -> dict:
    """Score a checkpoint, optionally under a DIFFERENT domain than it trained on.

    A network trained on the synthetic prior has to be scored against MNIST,
    Fashion-MNIST and chess in turn, and those are three domains that differ only
    in which dataset fills the B/D band. `as_domain` supplies that domain, and
    with it that domain's split, mask and width — the checkpoint only has to
    agree on `d_in`, which padding guarantees.
    """
    rn = run_from_row(row)
    domain = as_domain or rn.domain
    dom = domains.get(domain)
    if dom.d_in != rn.cfg.d_in:
        raise ValueError(f"{exp} has d_in={rn.cfg.d_in}, {domain!r} needs {dom.d_in}")
    mask_rows = dom.n_mask_default if as_domain else rn.mask_rows
    params = splitfig.load_params(exp)
    if as_domain:
        train_cls, held_cls = dom.split
    else:
        train_cls = tuple(rn.train_digits) if rn.train_digits else None
        held_cls = tuple(rn.held_digits) if rn.held_digits else None

    train_ctx = row.get("ctx_mode", "iid")
    contexts = tuple(CONTEXTS) + ((train_ctx,) if train_ctx not in CONTEXTS else ())
    scores, pools = {}, {}
    for ctx in contexts:
        scores[ctx] = splitfig.score_all(
            params, domain, mask_rows, train_cls, held_cls, rn.cfg,
            ctx_mode=ctx, Q=Q_STD, M=rn.M, seed=SEED_STD)
        pools[ctx] = pool_diagnostics(domain, mask_rows, train_cls,
                                      held_cls, rn.cfg, ctx, Q_STD, rn.M)
        for cond in PRESENT:
            if cond not in scores[ctx]:
                continue
            s, p = scores[ctx][cond], pools[ctx][cond]
            logging.info(
                f"  {ctx:3s} {cond:16s} nmse={s['nmse']:.4f}  "
                f"id={s['id_acc']:.3f} (ceiling {p['id_ceiling']:.3f}, "
                f"margin {p['margin_med']:.2f})")
        for cond in ("C_seen_absent", "F_same_absent", "D_novel_absent"):
            if cond in scores[ctx]:
                logging.info(f"  {ctx:3s} {cond:16s} nmse={scores[ctx][cond]['nmse']:.4f}")

    return dict(
        experiment=exp + SUFFIX + (f"_{as_domain}" if as_domain else ""),
        evaluated=exp, name=row.get("name", "") + " [standard eval]",
        domain=domain, trained_domain=rn.domain, kind=dom.kind,
        M=rn.M, Q=Q_STD, mask_rows=mask_rows,
        train_ctx_mode=train_ctx,
        train_digits=list(train_cls) if train_cls else None,
        held_digits=list(held_cls) if held_cls else None,
        contexts=list(contexts), train_ctx_scored=train_ctx,
        eval_seed=SEED_STD, n_eval=512, time_s=0.0,
        scores=scores, pool=pools,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exps", nargs="*", help="experiments to evaluate")
    ap.add_argument("--all", action="store_true",
                    help="every trained run that has a checkpoint")
    ap.add_argument("--force", action="store_true", help="re-evaluate existing rows")
    ap.add_argument("--as-domain", default=None,
                    help="score under this domain instead of the one trained on; "
                         "needs a matching cfg.d_in, which padding provides")
    a = ap.parse_args()

    by_exp = {r["experiment"]: r for r in read_rows()}
    todo = []
    for exp, row in by_exp.items():
        if "cfg" not in row or exp.endswith(SUFFIX) or exp.endswith("_sharedq"):
            continue
        if a.exps and exp not in a.exps:
            continue
        if not a.exps and not a.all:
            continue
        if not (PROJECT / f"params_{exp}.pkl").exists():
            logging.info(f"{exp}: no checkpoint, skipping")
            continue
        tag = SUFFIX + (f"_{a.as_domain}" if a.as_domain else "")
        if already_done(exp + tag) and not a.force:
            logging.info(f"{exp}{tag} already present — skipping")
            continue
        todo.append((exp, row))

    if not todo:
        logging.info("nothing to do (pass experiment names or --all)")
        return
    logging.info(f"standard eval on {len(todo)}: {[e for e, _ in todo]}")
    for exp, row in todo:
        logging.info(f"=== {exp} as {a.as_domain or row.get('domain', 'mnist')} ===")
        append_result(evaluate_one(exp, row, a.as_domain))


if __name__ == "__main__":
    main()
