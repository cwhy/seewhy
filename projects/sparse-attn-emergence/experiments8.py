"""
Sparse-attn-emergence — exp8: where does the mixer overtake attention?

exp7 compared architectures at two sparsities and produced an odd result: the causal mixer
solved 5/16 at s=7 but only 4/16 at s=3, even though s=3 has a 20x smaller search space.
Checking the stored per-seed data explained why, and turned up two analysis errors:

  * s=3 was only ever run at ONE learning rate, while s=7 was swept over three. The easy-cell
    comparison was therefore untuned — the same mistake that made exp6 wrong.
  * support_iou was averaged over all seeds, mixing solvers with seeds that never learned
    anything, and reported with no chance baseline. Conditioned on solved seeds the mixer's
    alignment is 0.479 at s=7 (chance 0.280), not the 0.349 first reported.

So: sweep sparsity properly, both arms, two learning rates each, and record alignment
conditioned on success alongside the chance level. If the paper's story is right there should
be a CROSSOVER — attention better while the search is easy, the mixer better once it is hard.

Sharded so the two GPUs can run it at once:
    SHARD=0 -> s in {3, 5, 7}      SHARD=1 -> s in {4, 6, 8}

Usage:
    SHARD=0 uv run --no-sync python .../scripts/run_experiments.py --bg --gpu 0 exp8
    SHARD=1 uv run --no-sync python .../scripts/run_experiments.py --bg --gpu 1 exp8
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

from lib.metrics import support_iou, time_to_emergence
from lib.models import (Config, forward, forward_mixer, init_mixer_params, init_params,
                        n_params)
from lib.tasks import linear_map_batch, linear_map_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
JSONL = Path(__file__).parent / "results.jsonl"

SMOKE = os.environ.get("SMOKE", "0") == "1"
SHARD = os.environ.get("SHARD")

S, T, C = 16, 2, 2
SPARSITIES = (3, 4, 5, 6, 7, 8)
LRS = (3e-4, 1e-3)
ARMS = ("transformer", "mixer")
if SMOKE:
    SPARSITIES, LRS = (5,), (1e-3,)
elif SHARD is not None:
    SPARSITIES = tuple(s for i, s in enumerate(SPARSITIES) if i % 2 == int(SHARD))

N_LAYERS, D_MODEL, D_MLP, N_HEADS = 1, 128, 512, 8
D_HEAD = D_MODEL // N_HEADS
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
    """Top-s of S positions chosen at random: hypergeometric mean overlap s^2/S."""
    inter = s * s / S
    return inter / (2 * s - inter)


def mixer_support_iou(p, A, s):
    W = jnp.abs(p["l0_Wtok"])[S - 1 + jnp.arange(S), :S]
    top = jnp.argsort(-W, -1)[:, :s]
    inter = jnp.take_along_axis(A.astype(jnp.float32), top, -1).sum(-1)
    return (inter / (2 * s - inter)).mean()


def run(arm: str, s: int, lr: float) -> dict:
    init_fn = init_params if arm == "transformer" else init_mixer_params
    fwd = forward if arm == "transformer" else forward_mixer

    seed_keys = jax.random.split(jax.random.key(SEED), N_SEEDS)
    a_keys = jax.random.split(jax.random.fold_in(jax.random.key(SEED), 99), N_SEEDS)
    params = jax.vmap(lambda k: init_fn(k, CFG))(seed_keys)
    A_all = jax.vmap(lambda k: linear_map_matrix(k, S, s))(a_keys)

    opt = optax.adamw(optax.join_schedules(
        [optax.linear_schedule(0.0, lr, WARMUP), optax.constant_schedule(lr)], [WARMUP]),
        weight_decay=WD)
    opt_state = jax.vmap(opt.init)(params)

    def loss_fn(p, b):
        logits = fwd(p, b, CFG)[:, S - 1 : 2 * S - 1, :]
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

    if arm == "mixer":
        iou = np.asarray(jax.jit(jax.vmap(lambda p, A: mixer_support_iou(p, A, s)))(
            params, A_all))
    else:
        ek = jax.random.split(jax.random.key(12345), N_SEEDS)
        eb = jax.vmap(lambda k, A: linear_map_batch(k, A, BATCH))(ek, A_all)
        iou = np.asarray(jax.jit(jax.vmap(lambda p, A, b: support_iou(
            forward(p, b, CFG, return_attn=True)[1][0].mean(0), A, s, S, N_HEADS)[1]))(
                params, A_all, eb))

    tstar = [time_to_emergence(acc2[i], MAIN_THRESH) for i in range(N_SEEDS)]
    emerged = [t for t in tstar if t is not None]
    solved = loss2[:, -1] < EXACT
    elapsed = time.perf_counter() - t0
    sl = slice(CURVE_EVERY - 1, None, CURVE_EVERY)

    logging.info(
        f"  [{arm:>11}] s={s} lr={lr:<7.0e} solve {len(emerged):>2}/{N_SEEDS}  "
        f"exact {int(solved.sum()):>2}/{N_SEEDS}  "
        f"median t* {int(np.median(emerged)) if emerged else -1:>5}  "
        f"loss2 med {np.median(loss2[:, -1]):.4f}  "
        f"iou {iou.mean():.2f} (solved {iou[solved].mean() if solved.any() else float('nan'):.2f}, "
        f"chance {chance_iou(s):.2f})  ({elapsed:.0f}s)")
    return {
        "experiment": f"{'smoke_' if SMOKE else ''}exp8_{arm}_s{s}_lr{lr:.0e}",
        "name": f"{arm} s={s} lr={lr:.0e} @ S={S}, {N_SEEDS} seeds",
        "arch": arm, "causal": True, "task": "linear_map",
        "S": S, "s": s, "T": T, "C": C, "per_seed_matrix": True,
        "n_seeds": N_SEEDS, "seed": SEED,
        "n_layers": N_LAYERS, "d_model": D_MODEL, "d_mlp": D_MLP,
        "n_heads": N_HEADS if arm == "transformer" else None,
        "lr": lr, "warmup": WARMUP, "weight_decay": WD,
        "steps": STEPS, "batch_size": BATCH,
        "n_params": n_params(init_fn(jax.random.key(0), CFG)),
        "time_s": round(elapsed, 1), "main_thresh": MAIN_THRESH,
        "solve_rate": len(emerged) / N_SEEDS,
        "exact_rate": float(solved.mean()),
        "median_t_star": float(np.median(emerged)) if emerged else None,
        "t_star": tstar,
        "final_loss2": np.round(loss2[:, -1], 6).tolist(),
        "support_iou": np.round(iou, 4).tolist(),
        "support_iou_solved": float(iou[solved].mean()) if solved.any() else None,
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

    logging.info(f"exp8 shard={SHARD}: s in {SPARSITIES}, lrs {LRS}, arms {ARMS}")
    for s in SPARSITIES:
        for arm in ARMS:
            for lr in LRS:
                name = f"{'smoke_' if SMOKE else ''}exp8_{arm}_s{s}_lr{lr:.0e}"
                if name in done:
                    logging.info(f"  {name} already done — skipping")
                    continue
                try:
                    append_result(run(arm, s, lr))
                except Exception as e:
                    logging.error(f"  {name} FAILED, continuing: {type(e).__name__}: "
                                  f"{str(e).splitlines()[0][:200]}")
    logging.info("exp8 shard finished")
