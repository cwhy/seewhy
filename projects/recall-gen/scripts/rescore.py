"""Re-score existing checkpoints on the shared-query eval sets (plan step A1).

No training. Every trained run in `results.jsonl` is reconstructed from its own
row — the row carries every `Run` field it was built with — its checkpoint is
loaded, and it is evaluated again on the eval sets `lib/evalsets.py` now builds,
where a pool's present and absent conditions share their queries.

The point is the normaliser. Every reported number is model MSE over
mean-image MSE, and the denominator depends only on the queries; drawing the two
conditions separately left the two denominators differing by up to ~2%, so a
present-vs-absent difference below ~0.02 could not be read. Shared queries make
them equal by construction. This script prints the old and new spread so the
floor it removes is a measured number rather than an assertion.

Results land as `<exp>_sharedq` rows, leaving the originals untouched: the paper
cites the old rows and can be moved over deliberately.

Usage:
    uv run --no-sync python projects/recall-gen/scripts/rescore.py            # all
    uv run --no-sync python projects/recall-gen/scripts/rescore.py exp1 exp5
"""

import argparse
import json
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

from lib.core import Cfg, row_mask
from lib import evalsets
from lib.train import Run, build_pools, make_eval, evaluate, append_result, already_done

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

SUFFIX = "_sharedq"
RUN_FIELDS = set(Run.__dataclass_fields__)


def rows() -> list[dict]:
    out = []
    for line in (PROJECT / "results.jsonl").read_text().strip().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def run_from_row(row: dict) -> Run:
    """Rebuild the Run a checkpoint was trained with, from its own result row."""
    kw = {k: v for k, v in row.items()
          if k in RUN_FIELDS and k not in ("exp_name", "name")}
    for k in ("train_digits", "held_digits"):
        if kw.get(k) is not None:
            kw[k] = tuple(kw[k])
    kw["cfg"] = Cfg(**row["cfg"])
    kw["conditions"] = {c: (pool, bool(present))
                        for c, (pool, present) in row["conditions"].items()}
    return Run(exp_name=row["experiment"], name=row["name"], **kw)


def rescore(row: dict, params: Path) -> dict:
    rn = run_from_row(row)
    pools_np, _ = build_pools(rn)
    mask = row_mask(rn.mask_rows)
    mean_img = pools_np["train"].mean(0)
    ev = evalsets.build(pools_np, mask, rn.M, rn.Q, rn.n_eval, mean_img,
                        conditions=rn.conditions)

    with open(params, "rb") as f:
        p = jax.tree_util.tree_map(jnp.asarray, pickle.load(f))
    eval_fn = make_eval(rn, jnp.array(mask))
    final = evaluate(eval_fn, p, ev, mask, mean_img,
                     chunk=32 if rn.M >= 256 else 128)
    final = {c: {k: v for k, v in final[c].items() if k != "preds"} for c in final}

    # The A1 deliverable: what the normalisers were, and what they now are.
    old = {c: row["final"][c]["mse_mean"] for c in final if c in row["final"]}
    new = {c: float(ev[c].mse_mean) for c in final}
    by_pool: dict[str, list[str]] = {}
    for c, (pool, _) in rn.conditions.items():
        by_pool.setdefault(pool, []).append(c)
    spread = {}
    for pool, cs in by_pool.items():
        if len(cs) < 2:
            continue
        f = lambda d: (max(d[c] for c in cs if c in d) / min(d[c] for c in cs if c in d)) - 1.0
        spread[pool] = {"old": f(old), "new": f(new)}
        logging.info(f"  normaliser spread on {pool:<10} "
                     f"{spread[pool]['old']*100:.2f}% -> {spread[pool]['new']*100:.4f}%")

    for c in final:
        logging.info(f"  {c:<16} nmse {row['final'][c]['nmse']:.4f} -> {final[c]['nmse']:.4f}"
                     f"   id_acc {row['final'][c]['id_acc']:.3f} -> {final[c]['id_acc']:.3f}")

    return dict(
        experiment=row["experiment"] + SUFFIX,
        name=row["name"] + " [shared-query eval]",
        rescored_from=row["experiment"], params=params.name,
        **{k: (list(v) if isinstance(v, tuple) else v)
           for k, v in vars(rn).items()
           if k not in ("exp_name", "name", "cfg", "conditions")},
        cfg=rn.cfg._asdict(), state_floats=rn.cfg.state_floats,
        conditions={c: [pool, present] for c, (pool, present) in rn.conditions.items()},
        time_s=0.0, final=final,
        normaliser_spread=spread,
        prev_final={c: {k: v for k, v in row["final"][c].items()} for c in row["final"]},
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exps", nargs="*", help="experiments to re-score (default: all trained)")
    a = ap.parse_args()

    todo = []
    for row in rows():
        exp = row.get("experiment", "")
        if "cfg" not in row or exp.endswith(SUFFIX):
            continue
        if a.exps and exp not in a.exps:
            continue
        params = PROJECT / f"params_{exp}.pkl"
        if not params.exists():
            logging.info(f"{exp}: no checkpoint, skipping")
            continue
        if already_done(exp + SUFFIX):
            logging.info(f"{exp}{SUFFIX} already in results.jsonl — skipping")
            continue
        todo.append((row, params))

    logging.info(f"re-scoring {len(todo)}: {[r['experiment'] for r, _ in todo]}")
    for row, params in todo:
        logging.info(f"=== {row['experiment']} ===")
        append_result(rescore(row, params))


if __name__ == "__main__":
    main()
