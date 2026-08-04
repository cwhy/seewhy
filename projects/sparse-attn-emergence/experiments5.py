"""
Sparse-attn-emergence — exp5: does the same story hold on cellular automata, in context?

exp1–exp4 study a single fixed A learned into the weights. The CA task is structurally
different (see concepts.md): a pool of N=256 rules is drawn per run and ONE rule is
sampled per training example, so the model cannot memorise the map — it has to infer the
active rule from the sequence and then apply it. The attention pattern it needs is still
sparse and still known by construction: x_{t+1}[i] depends on a window of width 2k+1
around position i in the previous state.

So this is the in-context version of the same question, and k is the knob: composing the
rule k times per transition widens the required span from 3 to 2k+1.

Metrics differ from the linear map on purpose. The first state is uniform noise and the
early states are genuinely ambiguous — the rule is not yet identifiable — so predicting
them well is impossible. The headline is loss on the FINAL state, where all the in-context
evidence is available; the per-state loss profile is recorded as the in-context learning
curve. Plateau is ln 4 = 1.386.

Usage:
    uv run --no-sync python projects/sparse-attn-emergence/scripts/run_experiments.py --bg exp5
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
from lib.models import Config, forward, init_params, n_params
from lib.tasks import ca_batch, ca_rule_pool

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
JSONL = Path(__file__).parent / "results.jsonl"

SMOKE = os.environ.get("SMOKE", "0") == "1"

S, T, C, W = 16, 16, 4, 3          # S (cells) is ours — the paper does not state it
N_RULES = 256
DEPTHS = (1, 2, 3)                 # k: required span is 2k+1
if SMOKE:
    S, T, DEPTHS = 8, 4, (1,)

N_LAYERS, D_MODEL, D_MLP, N_HEADS = 4, 128, 512, 8      # paper: 4 layers for CA
D_HEAD = D_MODEL // N_HEADS
BATCH_TOKENS = 8192
LR, WARMUP, WD = 3e-4, 200, 0.01
STEPS, CHUNK = (200, 50) if SMOKE else (10_000, 100)
N_SEEDS = 2 if SMOKE else 8        # 4 layers x 256 tokens — 8 seeds, not 16
SEED = 0
PLATEAU = float(np.log(C))
MAIN_THRESH = 0.95
CURVE_EVERY, CURVE_ROUND = 100, 5


def append_result(row: dict):
    with open(JSONL, "a") as f:
        f.write(json.dumps(row) + "\n")


def run_depth(k: int) -> dict:
    seq_len = S * T
    batch = max(8, BATCH_TOKENS // seq_len)
    cfg = Config(N_LAYERS, D_MODEL, D_MLP, N_HEADS, D_HEAD, C, seq_len)

    seed_keys = jax.random.split(jax.random.key(SEED), N_SEEDS)
    pool_keys = jax.random.split(jax.random.fold_in(jax.random.key(SEED), 99), N_SEEDS)
    params = jax.vmap(lambda key: init_params(key, cfg))(seed_keys)
    pools = jax.vmap(lambda key: ca_rule_pool(key, N_RULES, C, W))(pool_keys)

    sched = optax.join_schedules(
        [optax.linear_schedule(0.0, LR, WARMUP), optax.constant_schedule(LR)], [WARMUP])
    opt = optax.adamw(sched, weight_decay=WD)
    opt_state = jax.vmap(opt.init)(params)

    def losses(p, b):
        """Returns (mean CE over states 2..T, per-state CE, final-state accuracy)."""
        logits = forward(p, b, cfg)[:, S - 1 : -1, :]          # predicts tokens S..S*T-1
        tgt = b[:, S:]
        ls = -jnp.take_along_axis(jax.nn.log_softmax(logits, -1), tgt[..., None], -1).squeeze(-1)
        per_state = ls.reshape(ls.shape[0], T - 1, S).mean(axis=(0, 2))   # (T-1,)
        acc_last = (logits[:, -S:].argmax(-1) == tgt[:, -S:]).mean()
        return ls.mean(), (per_state, acc_last)

    def chunk_one(p, st, keys, pool):
        def body(carry, key):
            p, st = carry
            b = ca_batch(key, pool, batch, S, T, k, C)
            (loss, (per_state, acc_last)), g = jax.value_and_grad(losses, has_aux=True)(p, b)
            upd, st = opt.update(g, st, p)
            return (optax.apply_updates(p, upd), st), (loss, per_state[-1], acc_last)

        (p, st), out = jax.lax.scan(body, (p, st), keys)
        return p, st, *out

    chunk_fn = jax.jit(jax.vmap(chunk_one))
    base = jax.random.fold_in(jax.random.key(SEED), 1)
    loss_c, last_c, acc_c = [], [], []
    t0 = time.perf_counter()

    for c in range(STEPS // CHUNK):
        keys = jax.random.split(jax.random.fold_in(base, c), N_SEEDS * CHUNK)
        params, opt_state, loss, loss_last, acc_last = chunk_fn(
            params, opt_state, keys.reshape(N_SEEDS, CHUNK), pools)
        loss_c.append(np.asarray(loss))
        last_c.append(np.asarray(loss_last))
        acc_c.append(np.asarray(acc_last))
        if (c + 1) % 20 == 0:
            logging.info(f"    k={k} step {(c+1)*CHUNK:>6}/{STEPS}  "
                         f"loss_all med {np.median(loss_c[-1][:, -1]):.4f}  "
                         f"loss_last med {np.median(last_c[-1][:, -1]):.4f}  "
                         f"acc_last med {np.median(acc_c[-1][:, -1]):.3f}  "
                         f"({time.perf_counter() - t0:.0f}s)")

    loss_all = np.concatenate(loss_c, axis=1)
    loss_last = np.concatenate(last_c, axis=1)
    acc_last = np.concatenate(acc_c, axis=1)

    # per-state profile at the end of training, on a fresh batch
    eval_keys = jax.random.split(jax.random.key(12345), N_SEEDS)
    eval_b = jax.vmap(lambda key, pool: ca_batch(key, pool, batch, S, T, k, C))(eval_keys, pools)
    per_state = np.asarray(jax.vmap(lambda p, b: losses(p, b)[1][0])(params, eval_b))

    tstar = [time_to_emergence(acc_last[i], MAIN_THRESH) for i in range(N_SEEDS)]
    emerged = [t for t in tstar if t is not None]
    elapsed = time.perf_counter() - t0
    sl = slice(CURVE_EVERY - 1, None, CURVE_EVERY)

    logging.info(
        f"  k={k} (span {2*k+1})  solve {len(emerged):>2}/{N_SEEDS}  "
        f"median t* {int(np.median(emerged)) if emerged else -1:>5}  "
        f"loss_last med {np.median(loss_last[:, -1]):.4f} / plateau {PLATEAU:.3f}  ({elapsed:.0f}s)")
    return {
        "experiment": f"{'smoke_' if SMOKE else ''}exp5_k{k}",
        "name": f"cellular automata k={k} (span {2*k+1}), {N_SEEDS} seeds, {N_RULES} rules",
        "task": "cellular_automata", "S": S, "T": T, "C": C, "W": W,
        "k": k, "span": 2 * k + 1, "n_rules": N_RULES, "in_context": True,
        "n_seeds": N_SEEDS, "seed": SEED,
        "n_layers": N_LAYERS, "d_model": D_MODEL, "d_mlp": D_MLP,
        "n_heads": N_HEADS, "d_head": D_HEAD,
        "lr": LR, "warmup": WARMUP, "weight_decay": WD,
        "steps": STEPS, "batch_size": batch, "batch_tokens": BATCH_TOKENS,
        "n_params": n_params(init_params(jax.random.key(0), cfg)),
        "time_s": round(elapsed, 1), "plateau": PLATEAU, "main_thresh": MAIN_THRESH,
        "solve_rate": len(emerged) / N_SEEDS,
        "median_t_star": float(np.median(emerged)) if emerged else None,
        "t_star": tstar,
        "final_loss_all": np.round(loss_all[:, -1], 6).tolist(),
        "final_loss_last": np.round(loss_last[:, -1], 6).tolist(),
        "final_acc_last": np.round(acc_last[:, -1], 5).tolist(),
        "per_state_loss": np.round(per_state, 5).tolist(),
        "curve_step": (np.arange(1, STEPS + 1)[sl]).tolist(),
        "curve_loss_last": np.round(loss_last[:, sl], CURVE_ROUND).tolist(),
    }


if __name__ == "__main__":
    done = set()
    if JSONL.exists():
        for line in JSONL.read_text().splitlines():
            try:
                done.add(json.loads(line).get("experiment"))
            except Exception:
                pass

    logging.info(f"exp5: CA S={S} T={T} C={C} N={N_RULES}, depths {DEPTHS}, "
                 f"{N_SEEDS} seeds x {STEPS} steps")
    for k in DEPTHS:
        name = f"{'smoke_' if SMOKE else ''}exp5_k{k}"
        if name in done:
            logging.info(f"  {name} already done — skipping")
            continue
        append_result(run_depth(k))
