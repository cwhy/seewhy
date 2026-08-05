"""
Sparse-attn-emergence — exp9: KDA linear attention on the crossover sweep.

exp8 mapped where a static mixing matrix overtakes softmax attention (crossover at s=4).
This adds a third point of comparison from the same family the paper surveys — Kimi Delta
Attention, a matrix-valued memory written by the delta rule with per-channel decay
(the paper compares Gated DeltaNet, Mamba, RWKV, xLSTM and a linear RNN; KDA sits among
them). Reference implementation: projects/universal-ar/experiments30.py.

Why it is an interesting third case. The two arms so far sit at opposite extremes:

  transformer   position mixing computed per input, through a softmax competition — has to
                SEARCH for the pattern
  mixer         position mixing is a fixed learned matrix — no search, but the same pattern
                for every sequence
  KDA           position mixing is computed per input like attention, but WITHOUT the
                softmax: a linear associative memory, read by key match

So if the difficulty really comes from softmax competition rather than from
input-dependence, KDA should behave like the mixer. If it comes from having to select
positions at all, KDA should behave like the transformer.

Same grid as exp8 (s in 3..8, two learning rates, 16 seeds) so the numbers drop straight
into the same figure. Sharded: SHARD=0 -> s in {3,5,7}, SHARD=1 -> s in {4,6,8}.

Usage:
    SHARD=0 uv run --no-sync python .../scripts/run_experiments.py --bg --gpu 0 exp9
"""

import json
import logging
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from lib.metrics import time_to_emergence
from lib.models import Config, forward_kda, init_kda_params, n_params
from lib.tasks import linear_map_batch, linear_map_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
JSONL = Path(__file__).parent / "results.jsonl"

SMOKE = os.environ.get("SMOKE", "0") == "1"
SHARD = os.environ.get("SHARD")

S, T, C = 16, 2, 2
SPARSITIES = (3, 4, 5, 6, 7, 8)
LRS = (3e-4, 1e-3)
if SMOKE:
    SPARSITIES, LRS = (5,), (1e-3,)
elif SHARD is not None:
    SPARSITIES = tuple(s for i, s in enumerate(SPARSITIES) if i % 2 == int(SHARD))

N_LAYERS, D_MODEL, D_MLP = 1, 128, 512
# H=8 was copied from the transformer's config and is the WRONG default for KDA: at
# S=16 it learns only 11-13 of the 16 rows and so never clears the threshold, which is
# why the first pass reported 0/16 everywhere. scripts/check_kda_positions.py sweeps this
# — at H>=16 every seed solves all 16 rows. Same "more heads, each smaller" effect exp3
# found for attention, in a third architecture.
N_HEADS = int(os.environ.get("HEADS", 8))          # matches the transformer/mixer arms
D_HEAD = D_MODEL // N_HEADS
# Decay horizon. Initialising it to the sequence length (the reference's choice, where the
# horizon was the episode) attenuates the earliest x0 positions by ~1/e before the late
# queries read them, and scripts/check_kda_positions.py shows that is what limits KDA here:
# mean accuracy 0.836 at horizon 32 against 0.985 at 3200 (decay ~ 1). Long by default.
HORIZON = float(os.environ.get("HORIZON", 100 * S * T))
SEQ_LEN = S * T
BATCH = 8192 // SEQ_LEN
WARMUP, WD = 200, 0.01
STEPS, CHUNK = (200, 50) if SMOKE else (10_000, 100)
N_SEEDS = 4 if SMOKE else 16
SEED = 0
MAIN_THRESH, EXACT = 0.95, 0.01
CURVE_EVERY, CURVE_ROUND = 100, 5
CFG = Config(N_LAYERS, D_MODEL, D_MLP, N_HEADS, D_HEAD, C, SEQ_LEN)


def append_result(row: dict):
    with open(JSONL, "a") as f:
        f.write(json.dumps(row) + "\n")


def chance_iou(s: int) -> float:
    inter = s * s / S
    return inter / (2 * s - inter)


def run(s: int, lr: float) -> dict:
    seed_keys = jax.random.split(jax.random.key(SEED), N_SEEDS)
    a_keys = jax.random.split(jax.random.fold_in(jax.random.key(SEED), 99), N_SEEDS)
    params = jax.vmap(lambda k: init_kda_params(k, CFG, HORIZON))(seed_keys)
    A_all = jax.vmap(lambda k: linear_map_matrix(k, S, s))(a_keys)

    opt = optax.adamw(optax.join_schedules(
        [optax.linear_schedule(0.0, lr, WARMUP), optax.constant_schedule(lr)], [WARMUP]),
        weight_decay=WD)
    opt_state = jax.vmap(opt.init)(params)

    def loss_fn(p, b):
        logits = forward_kda(p, b, CFG)[:, S - 1 : 2 * S - 1, :]
        tgt = b[:, S:]
        ls = jax.nn.log_softmax(logits, -1)
        return (-jnp.take_along_axis(ls, tgt[..., None], -1).squeeze(-1).mean(),
                (logits.argmax(-1) == tgt).mean())

    def chunk_one(p, st, keys, A):
        def body(carry, key):
            p, st = carry
            b = linear_map_batch(key, A, BATCH)
            (loss, acc), g = jax.value_and_grad(loss_fn, has_aux=True)(p, b)
            upd, st = opt.update(g, st, p)
            return (optax.apply_updates(p, upd), st), (loss, acc)

        (p, st), (loss, acc) = jax.lax.scan(body, (p, st), keys)
        return p, st, loss, acc

    chunk_fn = jax.jit(jax.vmap(chunk_one))
    base = jax.random.fold_in(jax.random.key(SEED), 1)
    loss_c, acc_c = [], []
    t0 = time.perf_counter()
    for c in range(STEPS // CHUNK):
        keys = jax.random.split(jax.random.fold_in(base, c), N_SEEDS * CHUNK)
        params, opt_state, loss, acc = chunk_fn(
            params, opt_state, keys.reshape(N_SEEDS, CHUNK), A_all)
        loss_c.append(np.asarray(loss))
        acc_c.append(np.asarray(acc))

    loss2 = np.concatenate(loss_c, axis=1)
    acc2 = np.concatenate(acc_c, axis=1)
    tstar = [time_to_emergence(acc2[i], MAIN_THRESH) for i in range(N_SEEDS)]
    emerged = [t for t in tstar if t is not None]
    solved = loss2[:, -1] < EXACT
    elapsed = time.perf_counter() - t0
    sl = slice(CURVE_EVERY - 1, None, CURVE_EVERY)

    logging.info(
        f"  [kda] s={s} lr={lr:<7.0e} solve {len(emerged):>2}/{N_SEEDS}  "
        f"exact {int(solved.sum()):>2}/{N_SEEDS}  "
        f"median t* {int(np.median(emerged)) if emerged else -1:>5}  "
        f"loss2 med {np.median(loss2[:, -1]):.4f}  ({elapsed:.0f}s)")
    return {
        "experiment": f"{'smoke_' if SMOKE else ''}exp9_kda_H{N_HEADS}_hz{HORIZON:.0f}_s{s}_lr{lr:.0e}",
        "decay_horizon": HORIZON,
        "name": f"kda s={s} lr={lr:.0e} @ S={S}, {N_SEEDS} seeds",
        "arch": "kda", "causal": True, "task": "linear_map",
        "S": S, "s": s, "T": T, "C": C, "per_seed_matrix": True,
        "n_seeds": N_SEEDS, "seed": SEED,
        "n_layers": N_LAYERS, "d_model": D_MODEL, "d_mlp": D_MLP,
        "n_heads": N_HEADS, "d_head": D_HEAD,
        "lr": lr, "warmup": WARMUP, "weight_decay": WD,
        "steps": STEPS, "batch_size": BATCH,
        "n_params": n_params(init_kda_params(jax.random.key(0), CFG)),
        "time_s": round(elapsed, 1), "main_thresh": MAIN_THRESH,
        "solve_rate": len(emerged) / N_SEEDS,
        "exact_rate": float(solved.mean()),
        "median_t_star": float(np.median(emerged)) if emerged else None,
        "t_star": tstar,
        "final_loss2": np.round(loss2[:, -1], 6).tolist(),
        # No support_iou: KDA has no explicit attention matrix to compare against the
        # support. An effective one would need gradient attribution, which is a separate
        # experiment rather than a free diagnostic.
        "support_iou": None,
        "support_iou_chance": chance_iou(s),
        "curve_step": (np.arange(1, STEPS + 1)[sl]).tolist(),
        "curve_loss2": np.round(loss2[:, sl], CURVE_ROUND).tolist(),
    }


if __name__ == "__main__":
    done = set()
    if JSONL.exists():
        for line in JSONL.read_text().splitlines():
            try:
                done.add(json.loads(line).get("experiment"))
            except Exception:
                pass

    logging.info(f"exp9 shard={SHARD}: KDA, s in {SPARSITIES}, lrs {LRS}")
    for s in SPARSITIES:
        for lr in LRS:
            name = f"{'smoke_' if SMOKE else ''}exp9_kda_H{N_HEADS}_hz{HORIZON:.0f}_s{s}_lr{lr:.0e}"
            if name in done:
                logging.info(f"  {name} already done — skipping")
                continue
            try:
                append_result(run(s, lr))
            except Exception as e:
                logging.error(f"  {name} FAILED, continuing: {type(e).__name__}: "
                              f"{str(e).splitlines()[0][:200]}")
    logging.info("exp9 shard finished")
