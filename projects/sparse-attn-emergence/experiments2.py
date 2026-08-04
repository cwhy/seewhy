"""
Sparse-attn-emergence — exp2: is there a hard window in sparsity, and does it grow
with context length? (H2)

exp1 showed S=16, s=3 always solves (16/16) with wildly varying timing. That is one
point on the paper's difficulty surface. The claim here is stronger and stranger:
difficulty is NON-MONOTONE in s. Both extremes should be easy — s=1 is a single
position to find, s=S is "attend to everything", which uniform attention already
approximates — while intermediate s requires a genuinely sparse, specific pattern.
The paper reports S=8 solvable at every s, with medium-sparsity becoming unlearnable
at S=16 and S=32.

Difference from exp1: each seed draws its OWN A. exp1 fixed A to isolate search noise
at constant difficulty; here solve_rate should average over problem instances, so that
a cell of the heatmap describes the (S, s) regime rather than one lucky matrix.

24 configs x 16 seeds, one results.jsonl row per config (skip-if-done is per config,
so an interrupted sweep resumes where it stopped).

Usage:
    uv run --no-sync python projects/sparse-attn-emergence/scripts/run_experiments.py --bg exp2
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
from lib.models import Config, forward, init_params, n_params
from lib.tasks import linear_map_batch, linear_map_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
JSONL = Path(__file__).parent / "results.jsonl"

SMOKE = os.environ.get("SMOKE", "0") == "1"

# ── sweep ──
SPARSITIES = {
    8: (1, 2, 3, 4, 6, 8),
    16: (1, 2, 3, 4, 6, 8, 12, 16),
    32: (1, 2, 3, 4, 6, 8, 12, 16, 24, 32),
}
if SMOKE:
    SPARSITIES = {8: (1, 4), 32: (4,)}

T, C = 2, 2
N_LAYERS, D_MODEL, D_MLP, N_HEADS = 1, 128, 512, 8
D_HEAD = D_MODEL // N_HEADS
BATCH_TOKENS = 8192
LR, WARMUP, WD = 3e-4, 200, 0.01
STEPS, CHUNK = (200, 50) if SMOKE else (10_000, 100)
N_SEEDS = 4 if SMOKE else 16
SEED = 0

PLATEAU = float(np.log(C))
MAIN_THRESH = 0.95
CURVE_EVERY = 100                # 24 configs share one jsonl — keep curves coarse
CURVE_ROUND = 5


def append_result(row: dict):
    with open(JSONL, "a") as f:
        f.write(json.dumps(row) + "\n")


def run_config(S: int, s: int) -> dict:
    seq_len = S * T
    batch = BATCH_TOKENS // seq_len
    cfg = Config(N_LAYERS, D_MODEL, D_MLP, N_HEADS, D_HEAD, C, seq_len)

    seed_keys = jax.random.split(jax.random.key(SEED), N_SEEDS)
    a_keys = jax.random.split(jax.random.fold_in(jax.random.key(SEED), 99), N_SEEDS)
    params = jax.vmap(lambda k: init_params(k, cfg))(seed_keys)
    A_all = jax.vmap(lambda k: linear_map_matrix(k, S, s))(a_keys)      # (N_SEEDS, S, S)

    sched = optax.join_schedules(
        [optax.linear_schedule(0.0, LR, WARMUP), optax.constant_schedule(LR)], [WARMUP]
    )
    opt = optax.adamw(sched, weight_decay=WD)
    opt_state = jax.vmap(opt.init)(params)

    def loss_fn(p, b):
        logits = forward(p, b, cfg)[:, S - 1 : 2 * S - 1, :]
        tgt = b[:, S:]
        ls = jax.nn.log_softmax(logits, -1)
        return (-jnp.take_along_axis(ls, tgt[..., None], -1).squeeze(-1).mean(),
                (logits.argmax(-1) == tgt).mean())

    # A is a per-seed argument, so the scan body reads it from the vmapped closure.
    def chunk_one(p, st, keys, A):
        def body(carry, key):
            p, st = carry
            b = linear_map_batch(key, A, batch)
            (loss, acc), g = jax.value_and_grad(loss_fn, has_aux=True)(p, b)
            upd, st = opt.update(g, st, p)
            return (optax.apply_updates(p, upd), st), (loss, acc)

        (p, st), (loss, acc) = jax.lax.scan(body, (p, st), keys)
        return p, st, loss, acc

    chunk_fn = jax.jit(jax.vmap(chunk_one))

    def final_diag(p, A, b):
        _, attn = forward(p, b, cfg, return_attn=True)
        return support_iou(attn[0].mean(0), A, s, S, N_HEADS)

    diag_fn = jax.jit(jax.vmap(final_diag))

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
    eval_keys = jax.random.split(jax.random.key(12345), N_SEEDS)
    eval_b = jax.vmap(lambda k, A: linear_map_batch(k, A, batch))(eval_keys, A_all)
    iou_head, iou_row = (np.asarray(x) for x in diag_fn(params, A_all, eval_b))

    tstar = [time_to_emergence(acc2[i], MAIN_THRESH) for i in range(N_SEEDS)]
    emerged = [t for t in tstar if t is not None]
    elapsed = time.perf_counter() - t0
    sl = slice(CURVE_EVERY - 1, None, CURVE_EVERY)

    logging.info(
        f"  S={S:>2} s={s:>2}  solve {len(emerged):>2}/{N_SEEDS}  "
        f"median t* {int(np.median(emerged)) if emerged else -1:>5}  "
        f"loss2 med {np.median(loss2[:, -1]):.4f}  iou_row {iou_row.mean():.2f}  ({elapsed:.0f}s)"
    )
    return {
        # SMOKE must not collide with real config names, or skip-if-done poisons the
        # sweep with 200-step results.
        "experiment": f"{'smoke_' if SMOKE else ''}exp2_S{S}_s{s}",
        "name": f"sweep S={S} s={s}, {N_SEEDS} seeds, per-seed A",
        "task": "linear_map", "S": S, "s": s, "T": T, "C": C,
        "per_seed_matrix": True, "n_seeds": N_SEEDS, "seed": SEED,
        "n_layers": N_LAYERS, "d_model": D_MODEL, "d_mlp": D_MLP,
        "n_heads": N_HEADS, "d_head": D_HEAD,
        "lr": LR, "warmup": WARMUP, "weight_decay": WD,
        "steps": STEPS, "batch_size": batch, "batch_tokens": BATCH_TOKENS,
        "n_params": n_params(init_params(jax.random.key(0), cfg)),
        "time_s": round(elapsed, 1), "plateau": PLATEAU, "main_thresh": MAIN_THRESH,
        "solve_rate": len(emerged) / N_SEEDS,
        "median_t_star": float(np.median(emerged)) if emerged else None,
        "t_star": tstar,
        "final_loss2": np.round(loss2[:, -1], 6).tolist(),
        "final_acc2": np.round(acc2[:, -1], 5).tolist(),
        "final_iou_rowbest": np.round(iou_row, 4).tolist(),
        "final_iou_headbest": np.round(iou_head, 4).tolist(),
        "curve_step": (np.arange(1, STEPS + 1)[sl]).tolist(),
        "curve_loss2": np.round(loss2[:, sl], CURVE_ROUND).tolist(),
    }


if __name__ == "__main__":
    done = set()
    if JSONL.exists():
        with open(JSONL) as f:
            for line in f:
                try:
                    done.add(json.loads(line).get("experiment"))
                except Exception:
                    pass

    configs = [(S, s) for S in sorted(SPARSITIES) for s in SPARSITIES[S]]
    logging.info(f"exp2 sweep: {len(configs)} configs x {N_SEEDS} seeds x {STEPS} steps")
    t_all = time.perf_counter()

    for S, s in configs:
        name = f"{'smoke_' if SMOKE else ''}exp2_S{S}_s{s}"
        if name in done:
            logging.info(f"  {name} already done — skipping")
            continue
        append_result(run_config(S, s))

    logging.info(f"exp2 sweep finished in {time.perf_counter() - t_all:.0f}s")
