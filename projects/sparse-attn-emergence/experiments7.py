"""
Sparse-attn-emergence — exp7: why does our mixer lose when the paper's wins? (H5, debugged)

exp6 found the opposite of the paper's claim, but exp6 asked the wrong question. Re-reading
the paper against our setup turned up three differences:

  1. WRONG CONFIG. The paper's mixer claim is at S=16, s=7 — "medium sparsity", which in our
     exp2 is inside the unlearnable band (s=6 and s=8 were both 0/16). Their claim is that
     the mixer succeeds WHERE THE TRANSFORMER FAILS. exp6 compared at s=3 and s=4, where the
     transformer is fine, so it never tested the claim.

  2. MASKING UNSPECIFIED. The paper says only that the mixer "replaces dot-product attention
     with a static learned matrix that mixes information across sequence positions" — no
     mention of causal masking. On a next-token objective an UNMASKED mixing matrix leaks:
     position t draws on position t+1, the token being predicted. That alone would produce
     "learns much faster". The mixer_nomask arm measures the size of that effect.

  3. NO HYPERPARAMETERS PUBLISHED for the synthetic runs, so our lr=3e-4 (picked for the
     transformer) may simply not suit a mixer. Swept here for every arm.

Also measures, for the mixers, whether the mixing matrix itself encodes the support: IoU
between the top-s entries of |Wtok| for query row i and the true support of row i of A. The
paper's argument is that a mixer learns the pattern DIRECTLY rather than searching for it, so
it matters a great deal whether a failing mixer never found the pattern or found it and could
not use it.

Usage:
    uv run --no-sync python projects/sparse-attn-emergence/scripts/run_experiments.py --bg exp7
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

S, T, C = 16, 2, 2
# (s, learning rates). s=7 is the paper's own mixer config and is unlearnable for our
# transformer; s=3 is learnable for it, so the pair separates "wins where attention fails"
# from "learns the same thing faster".
CONFIGS = ((7, (3e-4, 1e-3, 3e-3)), (3, (3e-4,)))
ARMS = ("transformer", "mixer", "mixer_nomask")
if SMOKE:
    CONFIGS = ((7, (3e-4,)),)

N_LAYERS, D_MODEL, D_MLP, N_HEADS = 1, 128, 512, 8
D_HEAD = D_MODEL // N_HEADS
SEQ_LEN = S * T
BATCH = 8192 // SEQ_LEN
WARMUP, WD = 200, 0.01
STEPS, CHUNK = (200, 50) if SMOKE else (10_000, 100)
N_SEEDS = 4 if SMOKE else 16
SEED = 0
MAIN_THRESH = 0.95
CURVE_EVERY, CURVE_ROUND = 100, 5
CFG = Config(N_LAYERS, D_MODEL, D_MLP, N_HEADS, D_HEAD, C, SEQ_LEN)


def append_result(row: dict):
    with open(JSONL, "a") as f:
        f.write(json.dumps(row) + "\n")


def mixer_support_iou(p, A, sparsity):
    """Does the mixing matrix itself encode the support? Top-s of |Wtok| per query row
    against the true support of that row of A. Direct analogue of the attention IoU."""
    W = jnp.abs(p["l0_Wtok"])[S - 1 + jnp.arange(S), :S]          # (S, S)
    top = jnp.argsort(-W, -1)[:, :sparsity]
    inter = jnp.take_along_axis(A.astype(jnp.float32), top, -1).sum(-1)
    return (inter / (2 * sparsity - inter)).mean()


def run_arm(arm: str, sparsity: int, lr: float) -> dict:
    init_fn = init_params if arm == "transformer" else init_mixer_params
    if arm == "transformer":
        fwd = forward
    else:
        causal = arm != "mixer_nomask"
        fwd = lambda p, b, cfg: forward_mixer(p, b, cfg, causal=causal)

    seed_keys = jax.random.split(jax.random.key(SEED), N_SEEDS)
    a_keys = jax.random.split(jax.random.fold_in(jax.random.key(SEED), 99), N_SEEDS)
    params = jax.vmap(lambda k: init_fn(k, CFG))(seed_keys)
    A_all = jax.vmap(lambda k: linear_map_matrix(k, S, sparsity))(a_keys)

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

    if arm.startswith("mixer"):
        iou = np.asarray(jax.jit(jax.vmap(lambda p, A: mixer_support_iou(p, A, sparsity)))(
            params, A_all))
    else:
        eval_keys = jax.random.split(jax.random.key(12345), N_SEEDS)
        eval_b = jax.vmap(lambda k, A: linear_map_batch(k, A, BATCH))(eval_keys, A_all)
        iou = np.asarray(jax.jit(jax.vmap(lambda p, A, b: support_iou(
            forward(p, b, CFG, return_attn=True)[1][0].mean(0), A, sparsity, S, N_HEADS)[1]
        ))(params, A_all, eval_b))

    tstar = [time_to_emergence(acc2[i], MAIN_THRESH) for i in range(N_SEEDS)]
    emerged = [t for t in tstar if t is not None]
    exact = int((loss2[:, -1] < 0.01).sum())      # stricter than acc2 > 0.95
    elapsed = time.perf_counter() - t0
    sl = slice(CURVE_EVERY - 1, None, CURVE_EVERY)

    logging.info(
        f"  [{arm:>12}] s={sparsity} lr={lr:<7.0e} solve {len(emerged):>2}/{N_SEEDS}  "
        f"exact {exact:>2}/{N_SEEDS}  "
        f"median t* {int(np.median(emerged)) if emerged else -1:>5}  "
        f"loss2 med {np.median(loss2[:, -1]):.4f}  support_iou {iou.mean():.2f}  ({elapsed:.0f}s)")
    return {
        "experiment": f"{'smoke_' if SMOKE else ''}exp7_{arm}_s{sparsity}_lr{lr:.0e}",
        "name": f"{arm} s={sparsity} lr={lr:.0e} @ S={S}, {N_SEEDS} seeds",
        "arch": arm, "causal": arm != "mixer_nomask",
        "task": "linear_map", "S": S, "s": sparsity, "T": T, "C": C,
        "per_seed_matrix": True, "n_seeds": N_SEEDS, "seed": SEED,
        "n_layers": N_LAYERS, "d_model": D_MODEL, "d_mlp": D_MLP,
        "n_heads": N_HEADS if arm == "transformer" else None,
        "lr": lr, "warmup": WARMUP, "weight_decay": WD,
        "steps": STEPS, "batch_size": BATCH,
        "n_params": n_params(init_fn(jax.random.key(0), CFG)),
        "time_s": round(elapsed, 1), "main_thresh": MAIN_THRESH,
        "solve_rate": len(emerged) / N_SEEDS,
        "exact_rate": exact / N_SEEDS,
        "median_t_star": float(np.median(emerged)) if emerged else None,
        "t_star": tstar,
        "final_loss2": np.round(loss2[:, -1], 6).tolist(),
        "support_iou": np.round(iou, 4).tolist(),
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

    n = sum(len(lrs) for _, lrs in CONFIGS) * len(ARMS)
    logging.info(f"exp7: {n} runs — arms {ARMS} over {CONFIGS} @ S={S}")
    for sparsity, lrs in CONFIGS:
        for arm in ARMS:
            for lr in lrs:
                name = f"{'smoke_' if SMOKE else ''}exp7_{arm}_s{sparsity}_lr{lr:.0e}"
                if name in done:
                    logging.info(f"  {name} already done — skipping")
                    continue
                try:
                    append_result(run_arm(arm, sparsity, lr))
                except Exception as e:
                    logging.error(f"  {name} FAILED, continuing: {type(e).__name__}: "
                                  f"{str(e).splitlines()[0][:200]}")
