"""Gradient alignment against population: the paper's Figure 5 / Table 8 measurement.

For one batch and one set of weights, compute backprop's gradient, then each method's
estimate at a range of populations, and record the cosine per layer type. Rows go to
`results.jsonl` with `kind="cosine"`.

Populations are forward passes of the batch for EGGROLL (two per pair) and direct draws
for DUST, as everywhere else in the project. DUST uses the 1M-token tuned draw allocation
scaled by K/256 (and the paper's own 64-draw recipe), whatever checkpoint is measured.
"""

import logging
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np

from lib import data, diagnostics as dg, dust, eggroll
from lib.model import SIZES, init_params
from lib.train import CKPT, append_result, done_keys

BATCH = 8            # held-out sequences: the first eight of the validation split
EGG_SIGMA = 1e-2     # near the peak of the sigma probe at init (scripts/tmp/test_eggroll.py)
EGG_CHUNK = 4


def load_params(name: str, cfg):
    if name == "init":
        return init_params(cfg, jax.random.PRNGKey(42))
    with open(CKPT / f"{name}.pkl", "rb") as f:
        return jax.tree_util.tree_map(jnp.asarray, pickle.load(f))


def measure(exp: str, ckpt: str, method: str, pop: int, rep: int, params, ids, tg, ref, cfg, sigma: float = EGG_SIGMA) -> dict:
    name = f"{exp}|cosine|{ckpt}|{method}|{pop}|{rep}"
    t0 = time.perf_counter()
    key = jax.random.PRNGKey(1000 * rep + pop % 977)
    if method == "dust":
        recipe = dust.make_recipe("1M", pop)
        est, _ = dust.estimate_gradient(params, ids, tg, key, recipe, cfg)
        extra = dict(population=recipe.population, nrep=recipe.nrep)
    else:
        est, _ = eggroll.es_gradient(params, ids, tg, key, jnp.float32(sigma), cfg=cfg, n_pairs=pop // 2, chunk=EGG_CHUNK)
        extra = dict(sigma=sigma, n_pairs=pop // 2)
    jax.block_until_ready(est)
    dt = time.perf_counter() - t0
    c = dg.cosines(est, ref)
    return dict(experiment=exp, name=name, kind="cosine", ckpt=ckpt, method=method, pop=pop, rep=rep,
                cos={k: v for k, v in c.items() if "/" not in k}, cos_by_layer={k: v for k, v in c.items() if "/" in k},
                norm_ratio=dg.norm_ratio(est, ref), time_s=dt, **extra)


def run_alignment(exp: str, ckpts: list[str], dust_pops: list[int], egg_pops: list[int], reps: dict, cfg=SIZES["L8d512"]) -> None:
    done = done_keys()
    rows = data.load_rows("val")[:BATCH]
    ids, tg = [jnp.asarray(a) for a in data.split_xy(rows)]
    for ckpt in ckpts:
        params = load_params(ckpt, cfg)
        ref = dg.true_gradient(params, ids, tg, cfg)
        for method, pops in (("dust", dust_pops), ("eggroll", egg_pops)):
            for pop in pops:
                for rep in range(reps.get(pop, 1)):
                    name = f"{exp}|cosine|{ckpt}|{method}|{pop}|{rep}"
                    if name in done:
                        continue
                    row = measure(exp, ckpt, method, pop, rep, params, ids, tg, ref, cfg)
                    if not exp.startswith("smoke"):
                        append_result(row)
                    logging.info("%s  %.0fs  %s", name, row["time_s"], {k: round(v, 3) for k, v in row["cos"].items()})
