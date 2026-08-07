"""
Sparse-attn-emergence — exp11: does any of this transfer to a CONTENT-matched pattern?

Both of the paper's synthetic tasks hide a positional pattern. The linear map's support is a
fixed set of slots; the cellular automaton's window is a fixed local offset. In each case the
correct attention pattern can be expressed from position information alone — and neither task
is in-context in the sense the paper's motivating examples are:

                        varies per sequence     head must key on
    linear map          nothing (A is fixed)    position
    cellular automata   the rule                position
    IOI / induction     the content             CONTENT

This runs the missing cell: associative recall, where a sequence is pairs `a, f(a)` with a
fresh permutation f per sequence. A repeated key can only be answered by matching the earlier
occurrence and copying the token after it — the position of the answer moves from sequence to
sequence, so no fixed pattern works.

Why it matters for the paper's conclusions:

  H5  a static mixer beat attention on the linear map — but it CAN only do that where the
      pattern is content-independent. Here a fixed mixing matrix cannot express the circuit
      at any width. If the mixer collapses while the transformer succeeds, the architecture
      claim is specific to positional tasks and does not transfer to the capabilities the
      paper set out to explain.
  H1  does the same abrupt, seed-random emergence appear when the pattern is content-matched?

KDA is included because a delta-rule memory keyed by content is close to the task's native
form — the interesting case is whether it beats attention here after losing everywhere else.

Two layers for every arm: an induction circuit needs a previous-token head feeding a matching
head, and a one-layer model cannot express it.

Usage:
    uv run --no-sync python .../scripts/run_experiments.py --bg exp11
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
from lib.models import (Config, forward, forward_kda, forward_mixer, init_kda_params,
                        init_mixer_params, init_params, n_params)
from lib.tasks import induction_batch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
JSONL = Path(__file__).parent / "results.jsonl"

SMOKE = os.environ.get("SMOKE", "0") == "1"
ARM = os.environ.get("ARM")

V, N_PAIRS = 32, 32                  # vocab, pairs per sequence -> seq_len = 64
LRS = (3e-4, 1e-3)
ARMS = ("transformer", "mixer", "kda")
if SMOKE:
    LRS, N_PAIRS = (1e-3,), 8
if ARM:
    ARMS = tuple(ARM.split(","))          # e.g. ARM=transformer,mixer

N_LAYERS = int(os.environ.get("LAYERS", 2))       # capacity control for the mixer arm
D_MODEL, D_MLP, N_HEADS = 128, 512, 8
D_HEAD = D_MODEL // N_HEADS
SEQ_LEN = 2 * N_PAIRS
BATCH_TOKENS = 8192
BATCH = BATCH_TOKENS // SEQ_LEN
WARMUP, WD = 200, 0.01
STEPS, CHUNK = (200, 50) if SMOKE else (int(os.environ.get("STEPS", 10_000)), 100)
SUF = "" if STEPS in (10_000, 200) else f"_st{STEPS // 1000}k"    # keep budgets distinct
if N_LAYERS != 2:
    SUF += f"_L{N_LAYERS}"
N_SEEDS = 4 if SMOKE else 16
SEED = 0
MAIN_THRESH = 0.95
CURVE_EVERY, CURVE_ROUND = 100, 5
PLATEAU = float(np.log(V))           # guessing a value token
CFG = Config(N_LAYERS, D_MODEL, D_MLP, N_HEADS, D_HEAD, V, SEQ_LEN)


def append_result(row: dict):
    with open(JSONL, "a") as f:
        f.write(json.dumps(row) + "\n")


def run(arm: str, lr: float) -> dict:
    init_fn = {"transformer": init_params, "mixer": init_mixer_params,
               "kda": init_kda_params}[arm]
    fwd = {"transformer": forward, "mixer": forward_mixer, "kda": forward_kda}[arm]

    params = jax.vmap(lambda k: init_fn(k, CFG))(jax.random.split(jax.random.key(SEED), N_SEEDS))
    opt = optax.adamw(optax.join_schedules(
        [optax.linear_schedule(0.0, lr, WARMUP), optax.constant_schedule(lr)], [WARMUP]),
        weight_decay=WD)
    opt_state = jax.vmap(opt.init)(params)

    def losses(p, seq, recallable):
        """Value tokens sit at odd positions, predicted from the key before them."""
        logits = fwd(p, seq, CFG)[:, ::2, :]              # positions 0,2,... predict values
        tgt = seq[:, 1::2]
        ls = -jnp.take_along_axis(jax.nn.log_softmax(logits, -1), tgt[..., None], -1).squeeze(-1)
        correct = (logits.argmax(-1) == tgt)
        r = recallable.astype(jnp.float32)
        # loss over all value tokens; accuracy reported on the RECALLABLE ones, where the
        # answer is determined rather than a 1/V guess
        return ls.mean(), ((correct * r).sum() / (r.sum() + 1e-6),
                           (ls * r).sum() / (r.sum() + 1e-6))

    def chunk_one(p, st, keys):
        def body(carry, key):
            p, st = carry
            seq, rec = induction_batch(key, BATCH, N_PAIRS, V)
            (loss, aux), g = jax.value_and_grad(losses, has_aux=True)(p, seq, rec)
            upd, st = opt.update(g, st, p)
            return (optax.apply_updates(p, upd), st), (loss, *aux)

        (p, st), out = jax.lax.scan(body, (p, st), keys)
        return p, st, *out

    chunk_fn = jax.jit(jax.vmap(chunk_one))
    base = jax.random.fold_in(jax.random.key(SEED), 1)
    loss_c, acc_c, rloss_c = [], [], []
    t0 = time.perf_counter()
    for c in range(STEPS // CHUNK):
        keys = jax.random.split(jax.random.fold_in(base, c), N_SEEDS * CHUNK)
        params, opt_state, loss, acc, rloss = chunk_fn(
            params, opt_state, keys.reshape(N_SEEDS, CHUNK))
        loss_c.append(np.asarray(loss))
        acc_c.append(np.asarray(acc))
        rloss_c.append(np.asarray(rloss))

    loss_all = np.concatenate(loss_c, axis=1)
    acc = np.concatenate(acc_c, axis=1)
    rloss = np.concatenate(rloss_c, axis=1)
    tstar = [time_to_emergence(acc[i], MAIN_THRESH) for i in range(N_SEEDS)]
    emerged = [t for t in tstar if t is not None]
    elapsed = time.perf_counter() - t0
    sl = slice(CURVE_EVERY - 1, None, CURVE_EVERY)

    logging.info(
        f"  [{arm:>11}] lr={lr:<7.0e} recall-acc med {np.median(acc[:, -1]):.3f}  "
        f"solve {len(emerged):>2}/{N_SEEDS}  "
        f"median t* {int(np.median(emerged)) if emerged else -1:>5}  "
        f"recall-loss med {np.median(rloss[:, -1]):.4f} / plateau {PLATEAU:.3f}  ({elapsed:.0f}s)")
    return {
        "experiment": f"{'smoke_' if SMOKE else ''}exp11_{arm}_lr{lr:.0e}{SUF}",
        "name": f"induction / associative recall — {arm} lr={lr:.0e}, {N_SEEDS} seeds",
        "arch": arm, "task": "induction", "content_matched": True,
        "V": V, "n_pairs": N_PAIRS, "seq_len": SEQ_LEN,
        "n_seeds": N_SEEDS, "seed": SEED,
        "n_layers": N_LAYERS, "d_model": D_MODEL, "d_mlp": D_MLP,
        "n_heads": N_HEADS if arm != "mixer" else None, "d_head": D_HEAD,
        "lr": lr, "warmup": WARMUP, "weight_decay": WD,
        "steps": STEPS, "batch_size": BATCH, "batch_tokens": BATCH_TOKENS,
        "n_params": n_params(init_fn(jax.random.key(0), CFG)),
        "time_s": round(elapsed, 1), "plateau": PLATEAU, "main_thresh": MAIN_THRESH,
        "solve_rate": len(emerged) / N_SEEDS,
        "median_t_star": float(np.median(emerged)) if emerged else None,
        "t_star": tstar,
        "final_recall_acc": np.round(acc[:, -1], 5).tolist(),
        "final_recall_loss": np.round(rloss[:, -1], 6).tolist(),
        "final_loss_all": np.round(loss_all[:, -1], 6).tolist(),
        "curve_step": (np.arange(1, STEPS + 1)[sl]).tolist(),
        "curve_recall_acc": np.round(acc[:, sl], CURVE_ROUND).tolist(),
    }


if __name__ == "__main__":
    done = set()
    if JSONL.exists():
        for line in JSONL.read_text().splitlines():
            try:
                done.add(json.loads(line).get("experiment"))
            except Exception:
                pass

    logging.info(f"exp11: induction, V={V} pairs={N_PAIRS} seq={SEQ_LEN}, arms {ARMS}")
    for arm in ARMS:
        for lr in LRS:
            name = f"{'smoke_' if SMOKE else ''}exp11_{arm}_lr{lr:.0e}{SUF}"
            if name in done:
                logging.info(f"  {name} already done — skipping")
                continue
            try:
                append_result(run(arm, lr))
            except Exception as e:
                logging.error(f"  {name} FAILED, continuing: {type(e).__name__}: "
                              f"{str(e).splitlines()[0][:200]}")
    logging.info("exp11 finished")
