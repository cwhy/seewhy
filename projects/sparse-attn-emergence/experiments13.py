"""
Sparse-attn-emergence — exp13: does the C(S,s) difficulty law survive content-keying?

exp2 found that on the linear map, learnability tracks C(S,s) — the number of candidate
position-subsets per row — with a threshold that holds across context lengths. That is a
statement about a POSITION-keyed pattern. exp11 showed content-keyed patterns behave
differently at least architecturally (the mixer goes from best to incapable). Whether the
difficulty LAW carries over is untested, by the paper and by us.

k-of-m recall puts both families on one axis. Each block is m attribute tokens plus a value;
a query block's value is the value of the earlier block agreeing with it on k RELEVANT
attributes, the rest re-randomised. The model must learn which k of m matter — a choice out
of C(m,k), fixed per run, exactly analogous to the row support — and then match on them,
which is content-keyed and per-sequence.

Predictions, so the result can be wrong:

  * if the law transfers, difficulty peaks at k = m/2 where C(m,k) is largest, and cells
    with equal C behave alike — k=2 and k=6 both have C(8,·) = 28
  * if instead difficulty grows with k itself (more attributes to compare, a harder match
    regardless of how many subsets exist), it rises monotonically and the equal-C pair
    splits

exp2's dense end had exactly that asymmetry (C(16,4) = C(16,12) = 1820, solve rates 0.50 vs
0.00), so a split here would not be a surprise — but the direction and size are the point.

Usage:
    SHARD=0 uv run --no-sync python .../scripts/run_experiments.py --bg --gpu 0 exp13
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
from lib.tasks import kofm_recall_batch, kofm_recall_unique, kofm_subset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
JSONL = Path(__file__).parent / "results.jsonl"

SMOKE = os.environ.get("SMOKE", "0") == "1"
SHARD = os.environ.get("SHARD")
# UNIQUE=1 gives every context block a distinct relevant-attribute tuple, so
# exactly one block matches at any k. Without it, small k is ambiguous by
# construction (A^-k spurious matches) and the k-curve measures that instead.
UNIQUE = os.environ.get("UNIQUE", "0") == "1"
PILOT = os.environ.get("PILOT")                     # run one k and stop

M, A, V, N_BLOCKS = 8, 4, 16, 8                     # attributes, alphabet, values, blocks
KS = (1, 2, 3, 4, 6)                                # C(8,k) = 8, 28, 56, 70, 28
LRS = (3e-4, 1e-3)
if SMOKE:
    KS, LRS = (2,), (1e-3,)
elif PILOT:
    KS, LRS = (int(PILOT),), (1e-3,)
elif SHARD is not None:
    KS = tuple(x for i, x in enumerate(KS) if i % 2 == int(SHARD))

N_LAYERS, D_MODEL, D_MLP, N_HEADS = 4, 128, 512, 8
D_HEAD = D_MODEL // N_HEADS
SEQ_LEN = N_BLOCKS * (M + 1)
VOCAB = A + V
BATCH_TOKENS = 8192
BATCH = max(8, BATCH_TOKENS // SEQ_LEN)
WARMUP, WD = 200, 0.01
STEPS, CHUNK = (200, 50) if SMOKE else (int(os.environ.get("STEPS", 30_000)), 100)
N_SEEDS = 4 if SMOKE else 16
SEED = 0
MAIN_THRESH = 0.95
CURVE_EVERY, CURVE_ROUND = 200, 5
PLATEAU = float(np.log(V))
CFG = Config(N_LAYERS, D_MODEL, D_MLP, N_HEADS, D_HEAD, VOCAB, SEQ_LEN)


def append_result(row: dict):
    with open(JSONL, "a") as f:
        f.write(json.dumps(row) + "\n")


def comb(n: int, r: int) -> int:
    from math import comb as _c
    return _c(n, r)


def run(k: int, lr: float) -> dict:
    seed_keys = jax.random.split(jax.random.key(SEED), N_SEEDS)
    sub_keys = jax.random.split(jax.random.fold_in(jax.random.key(SEED), 99), N_SEEDS)
    params = jax.vmap(lambda key: init_params(key, CFG))(seed_keys)
    subsets = jax.vmap(lambda key: kofm_subset(key, M, k))(sub_keys)       # (N_SEEDS, k)

    opt = optax.adamw(optax.join_schedules(
        [optax.linear_schedule(0.0, lr, WARMUP), optax.constant_schedule(lr)], [WARMUP]),
        weight_decay=WD)
    opt_state = jax.vmap(opt.init)(params)

    def losses(p, seq, value_pos, is_query):
        """Score value tokens of QUERY blocks only — context values are random."""
        logits = forward(p, seq, CFG)[:, value_pos - 1, :]                 # (B, n_blocks, V+A)
        tgt = seq[:, value_pos]
        ls = -jnp.take_along_axis(jax.nn.log_softmax(logits, -1), tgt[..., None], -1).squeeze(-1)
        w = is_query.astype(jnp.float32)[None]
        acc = ((logits.argmax(-1) == tgt) * w).sum() / (w.sum() * seq.shape[0])
        return (ls * w).sum() / (w.sum() * seq.shape[0]), acc

    def chunk_one(p, st, keys, subset):
        def body(carry, key):
            p, st = carry
            gen = kofm_recall_unique if UNIQUE else kofm_recall_batch
            seq, vp, isq = gen(key, BATCH, N_BLOCKS, M, k, A, V, subset)
            (loss, acc), g = jax.value_and_grad(losses, has_aux=True)(p, seq, vp, isq)
            upd, st = opt.update(g, st, p)
            return (optax.apply_updates(p, upd), st), (loss, acc)

        (p, st), out = jax.lax.scan(body, (p, st), keys)
        return p, st, *out

    chunk_fn = jax.jit(jax.vmap(chunk_one))
    base = jax.random.fold_in(jax.random.key(SEED), 1)
    loss_c, acc_c = [], []
    t0 = time.perf_counter()
    for c in range(STEPS // CHUNK):
        keys = jax.random.split(jax.random.fold_in(base, c), N_SEEDS * CHUNK)
        params, opt_state, loss, acc = chunk_fn(
            params, opt_state, keys.reshape(N_SEEDS, CHUNK), subsets)
        loss_c.append(np.asarray(loss))
        acc_c.append(np.asarray(acc))

    loss2 = np.concatenate(loss_c, axis=1)
    acc2 = np.concatenate(acc_c, axis=1)
    tstar = [time_to_emergence(acc2[i], MAIN_THRESH) for i in range(N_SEEDS)]
    emerged = [t for t in tstar if t is not None]
    elapsed = time.perf_counter() - t0
    sl = slice(CURVE_EVERY - 1, None, CURVE_EVERY)

    logging.info(
        f"  k={k} of {M} (C={comb(M, k):>3}) lr={lr:<7.0e} "
        f"recall-acc med {np.median(acc2[:, -1]):.3f}  solve {len(emerged):>2}/{N_SEEDS}  "
        f"median t* {int(np.median(emerged)) if emerged else -1:>6}  "
        f"loss med {np.median(loss2[:, -1]):.4f} / plateau {PLATEAU:.3f}  ({elapsed:.0f}s)")
    return {
        "experiment": f"{'smoke_' if SMOKE else ''}exp13{'u' if UNIQUE else ''}_k{k}_lr{lr:.0e}",
        "name": f"k-of-m recall, k={k} of m={M} (C={comb(M, k)}), lr={lr:.0e}, {N_SEEDS} seeds",
        "arch": "transformer", "task": "kofm_recall", "unique_match": UNIQUE, "content_matched": True,
        "m": M, "k": k, "candidates": comb(M, k), "alphabet": A, "n_values": V,
        "n_blocks": N_BLOCKS, "seq_len": SEQ_LEN,
        "n_seeds": N_SEEDS, "seed": SEED,
        "n_layers": N_LAYERS, "d_model": D_MODEL, "d_mlp": D_MLP,
        "n_heads": N_HEADS, "d_head": D_HEAD,
        "lr": lr, "warmup": WARMUP, "weight_decay": WD,
        "steps": STEPS, "batch_size": BATCH,
        "n_params": n_params(init_params(jax.random.key(0), CFG)),
        "time_s": round(elapsed, 1), "plateau": PLATEAU, "main_thresh": MAIN_THRESH,
        "solve_rate": len(emerged) / N_SEEDS,
        "median_t_star": float(np.median(emerged)) if emerged else None,
        "t_star": tstar,
        "final_recall_acc": np.round(acc2[:, -1], 5).tolist(),
        "final_loss": np.round(loss2[:, -1], 6).tolist(),
        "curve_step": (np.arange(1, STEPS + 1)[sl]).tolist(),
        "curve_recall_acc": np.round(acc2[:, sl], CURVE_ROUND).tolist(),
    }


if __name__ == "__main__":
    done = set()
    if JSONL.exists():
        for line in JSONL.read_text().splitlines():
            try:
                done.add(json.loads(line).get("experiment"))
            except Exception:
                pass

    logging.info(f"exp13: k-of-m recall, m={M} A={A} V={V} blocks={N_BLOCKS} seq={SEQ_LEN}, "
                 f"ks {KS}, lrs {LRS}, {STEPS} steps")
    for k in KS:
        for lr in LRS:
            name = f"{'smoke_' if SMOKE else ''}exp13{'u' if UNIQUE else ''}_k{k}_lr{lr:.0e}"
            if name in done:
                logging.info(f"  {name} already done — skipping")
                continue
            try:
                append_result(run(k, lr))
            except Exception as e:
                logging.error(f"  {name} FAILED, continuing: {type(e).__name__}: "
                              f"{str(e).splitlines()[0][:200]}")
    logging.info("exp13 shard finished")
