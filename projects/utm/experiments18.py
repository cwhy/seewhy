"""
Universal Transformer with Memory on Sudoku-Extreme (exp18).
Architectural change: 2 independent shared blocks per ponder step (instead of 1).
Mirrors TRM's 2-layer-per-recursion design. Fresh training, D=512 to keep ~6M params.
"""

import json
import logging
import pickle
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import random

# ─── Hyperparameters ──────────────────────────────────────────────────────────
# fmt: off
D_MODEL        = 512
N_HEADS        = 8
HEAD_DIM       = D_MODEL // N_HEADS          # 64
FFN_DIM        = int(D_MODEL * 8 / 3)        # 1365 — SwiGLU intermediate width
N_MEM          = 8                            # memory tokens
K_PONDER       = 18                           # max ACT ponder steps
N_BLOCKS       = 2                            # blocks per ponder step (new)
VOCAB_SIZE     = 10
N_CELLS        = 81
SEQ_LEN        = N_CELLS + N_MEM             # 89 total tokens
LR             = 3e-4
BATCH_SIZE     = 256
N_EPOCHS       = 8
EMA_DECAY      = 0.999
WEIGHT_DECAY   = 0.01
LAMBDA_PONDER  = 0.0
HALT_BIAS      = -3.0
N_TRAIN        = None
N_TEST         = 20_000
SEED           = 42
EXP_NAME       = "exp18"
CHECKPOINT_FROM = None                        # fresh training
# fmt: on

PROJ_DIR = Path(__file__).parent
JSONL    = PROJ_DIR / "results.jsonl"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

_d      = HEAD_DIM // 2
_theta  = 10000.0 ** (-jnp.arange(_d, dtype=jnp.float32) * 2.0 / HEAD_DIM)
_pos    = jnp.arange(SEQ_LEN, dtype=jnp.float32)
_COS    = jnp.cos(jnp.outer(_pos, _theta))
_SIN    = jnp.sin(jnp.outer(_pos, _theta))


def n_params(params) -> int:
    return sum(x.size for x in jax.tree.leaves(params))


def append_result(row: dict) -> None:
    with open(JSONL, "a") as f:
        f.write(json.dumps(row) + "\n")


# ─── Model ────────────────────────────────────────────────────────────────────

def rmsnorm(x, scale, eps=1e-6):
    rms = jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return (x / rms) * scale


def _rope(x):
    x1, x2 = x[..., :_d], x[..., _d:]
    c = _COS[:, None, :]
    s = _SIN[:, None, :]
    return jnp.concatenate([x1 * c - x2 * s, x1 * s + x2 * c], axis=-1)


def mha(params, x):
    Q = jnp.einsum("sd,dnh->snh", x, params["W_q"])
    K = jnp.einsum("sd,dnh->snh", x, params["W_k"])
    V = jnp.einsum("sd,dnh->snh", x, params["W_v"])
    Q = _rope(Q); K = _rope(K)
    Q = Q / (jnp.linalg.norm(Q, axis=-1, keepdims=True) + 1e-6)
    K = K / (jnp.linalg.norm(K, axis=-1, keepdims=True) + 1e-6)
    scores  = jnp.einsum("inh,jnh->nij", Q, K) / (HEAD_DIM ** 0.5)
    weights = jax.nn.softmax(scores, axis=-1)
    out     = jnp.einsum("nij,jnh->inh", weights, V).reshape(SEQ_LEN, N_HEADS * HEAD_DIM)
    return out @ params["W_o"]


def ffn(params, x):
    return (jax.nn.silu(x @ params["W_gate"]) * (x @ params["W_val"])) @ params["W_out"]


def utm_block(params, x):
    x = x + mha(params["attn"], rmsnorm(x, params["norm1"]))
    x = x + ffn(params["ffn"],  rmsnorm(x, params["norm2"]))
    return x


# Remat both blocks: 2 blocks × K=18 steps = 36 forward passes in the backward graph
_block_remat = jax.checkpoint(utm_block)


def halt_router(params, x):
    mem_mean = jnp.mean(x[N_CELLS:], axis=0)
    return jax.nn.sigmoid(mem_mean @ params["halt"]["W"] + params["halt"]["b"])


def forward(params, puzzle):
    """2-block-per-step ACT forward pass."""
    x = jnp.concatenate([params["embed"][puzzle], params["mem_tokens"]], axis=0)

    halt_probs = []
    grid_outs  = []
    for _ in range(K_PONDER):
        # Two independent blocks applied sequentially as a unit each ponder step
        x = _block_remat(params["block1"], x)
        x = _block_remat(params["block2"], x)
        halt_probs.append(halt_router(params, x))
        grid_outs.append(x[:N_CELLS])

    hp       = jnp.stack(halt_probs)
    survival = jnp.concatenate([jnp.ones(1), jnp.cumprod(1 - hp[:-1])])
    w        = survival * hp
    w        = w.at[-1].set(jnp.maximum(1.0 - jnp.sum(w[:-1]), 0.0))

    out    = jnp.einsum("k,knd->nd", w, jnp.stack(grid_outs))
    out    = rmsnorm(out, params["norm_out"])
    logits = out @ params["head"]

    ponder_cost = jnp.dot(w, jnp.arange(1, K_PONDER + 1, dtype=jnp.float32))
    return logits, ponder_cost


# ─── Loss ─────────────────────────────────────────────────────────────────────

def loss_fn(params, x, y):
    logits, ponder_cost = forward(params, x)
    xe = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, y - 1))
    return xe + LAMBDA_PONDER * ponder_cost, (xe, ponder_cost)


def batch_loss(params, Xb, yb):
    losses, (xes, pcs) = jax.vmap(loss_fn, in_axes=(None, 0, 0))(params, Xb, yb)
    return jnp.mean(losses), (jnp.mean(xes), jnp.mean(pcs))


# ─── Chunk scan ───────────────────────────────────────────────────────────────

CHUNK_SIZE = 25   # halved vs single-block: 2 blocks × K=18 with remat

def make_chunk_fn(optimizer):
    def step_body(carry, batch):
        params, ema_params, opt_state = carry
        Xb, yb = batch
        (loss, aux), grads = jax.value_and_grad(batch_loss, has_aux=True)(params, Xb, yb)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        new_ema = jax.tree.map(
            lambda e, p: EMA_DECAY * e + (1.0 - EMA_DECAY) * p,
            ema_params, new_params,
        )
        return (new_params, new_ema, new_opt_state), (loss, *aux)

    @jax.jit
    def chunk_fn(params, ema_params, opt_state, Xb_chunk, yb_chunk):
        (params, ema_params, opt_state), stats = jax.lax.scan(
            step_body, (params, ema_params, opt_state), (Xb_chunk, yb_chunk)
        )
        return params, ema_params, opt_state, stats

    return chunk_fn


# ─── Eval ─────────────────────────────────────────────────────────────────────

@jax.jit
def _predict_batch(params, Xb):
    logits, _ = jax.vmap(forward, in_axes=(None, 0))(params, Xb)
    return jnp.argmax(logits, axis=-1) + 1


def eval_metrics(params, X, y, bs=256):
    n = X.shape[0]
    n_correct = 0
    n_counted = 0
    for i in range(0, n - bs + 1, bs):
        preds = _predict_batch(params, X[i : i + bs])
        n_correct += int(jnp.sum(jnp.all(preds == y[i : i + bs], axis=-1)))
        n_counted += bs
    return n_correct / n_counted if n_counted > 0 else 0.0


# ─── Init ─────────────────────────────────────────────────────────────────────

def init_block(keys):
    def w(k, *shape):
        return random.normal(k, shape) * 0.02
    return {
        "attn": {
            "W_q": w(keys[0], D_MODEL, N_HEADS, HEAD_DIM),
            "W_k": w(keys[1], D_MODEL, N_HEADS, HEAD_DIM),
            "W_v": w(keys[2], D_MODEL, N_HEADS, HEAD_DIM),
            "W_o": w(keys[3], N_HEADS * HEAD_DIM, D_MODEL),
        },
        "ffn": {
            "W_gate": w(keys[4], D_MODEL, FFN_DIM),
            "W_val":  w(keys[5], D_MODEL, FFN_DIM),
            "W_out":  w(keys[6], FFN_DIM, D_MODEL),
        },
        "norm1": jnp.ones(D_MODEL),
        "norm2": jnp.ones(D_MODEL),
    }


def init_params(key):
    ks = random.split(key, 24)
    return {
        "embed":      random.normal(ks[0], (VOCAB_SIZE, D_MODEL)) * 0.02,
        "mem_tokens": random.normal(ks[1], (N_MEM, D_MODEL)) * 0.02,
        "block1":     init_block(ks[2:9]),
        "block2":     init_block(ks[9:16]),
        "norm_out":   jnp.ones(D_MODEL),
        "halt": {
            "W": random.normal(ks[16], (D_MODEL,)) * 0.02,
            "b": jnp.array(HALT_BIAS),
        },
        "head": random.normal(ks[17], (D_MODEL, 9)) * 0.02,
    }


# ─── Train ────────────────────────────────────────────────────────────────────

def train(params, X_train, y_train, X_test, y_test):
    n         = X_train.shape[0]
    n_batches = n // BATCH_SIZE
    total_steps = N_EPOCHS * n_batches

    sched = optax.cosine_decay_schedule(LR, total_steps)
    opt   = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(sched, weight_decay=WEIGHT_DECAY),
    )
    opt_state  = opt.init(params)
    ema_params = params
    chunk_fn   = make_chunk_fn(opt)
    key        = random.PRNGKey(SEED)

    n_chunks    = n_batches // CHUNK_SIZE
    chunk_steps = n_chunks * CHUNK_SIZE

    history = {"loss": [], "xe": [], "ponder": [], "exact_acc": []}
    t0 = time.time()

    for epoch in range(N_EPOCHS):
        key, sk = random.split(key)
        perm = random.permutation(sk, n)
        idx  = perm[: chunk_steps * BATCH_SIZE]

        Xb_all = X_train[idx].reshape(n_chunks, CHUNK_SIZE, BATCH_SIZE, N_CELLS)
        yb_all = y_train[idx].reshape(n_chunks, CHUNK_SIZE, BATCH_SIZE, N_CELLS)

        acc_loss = jnp.zeros(())
        acc_xe   = jnp.zeros(())
        acc_pc   = jnp.zeros(())
        for c in range(n_chunks):
            params, ema_params, opt_state, (losses, xes, pcs) = chunk_fn(
                params, ema_params, opt_state, Xb_all[c], yb_all[c]
            )
            acc_loss = acc_loss + jnp.sum(losses)
            acc_xe   = acc_xe   + jnp.sum(xes)
            acc_pc   = acc_pc   + jnp.sum(pcs)

        loss_m = float(acc_loss) / chunk_steps
        xe_m   = float(acc_xe)   / chunk_steps
        pc_m   = float(acc_pc)   / chunk_steps

        acc     = eval_metrics(ema_params, X_test, y_test)
        elapsed = time.time() - t0

        history["loss"].append(float(loss_m))
        history["xe"].append(float(xe_m))
        history["ponder"].append(float(pc_m))
        history["exact_acc"].append(acc)

        logging.info(
            f"Epoch {epoch+1}/{N_EPOCHS} | loss={loss_m:.4f} xe={xe_m:.4f} "
            f"ponder={pc_m:.2f} | exact_acc={acc:.4f} | {elapsed:.0f}s"
        )

    return params, ema_params, history, time.time() - t0


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sys.path.insert(0, str(PROJ_DIR.parent.parent))
    from shared_lib.datasets import load_sudoku_extreme

    done = set()
    if JSONL.exists():
        for line in JSONL.read_text().splitlines():
            try:
                done.add(json.loads(line).get("experiment"))
            except Exception:
                pass
    if EXP_NAME in done:
        logging.info(f"{EXP_NAME} already done — skipping")
        sys.exit(0)

    logging.info(f"Loading dataset (n_tr={N_TRAIN}, n_tst={N_TEST})")
    ds = load_sudoku_extreme(n_tr=N_TRAIN, n_tst=N_TEST)
    logging.info(f"Dataset ready: {ds.n_train} train, {ds.n_test} test")

    key = random.PRNGKey(SEED)
    params = init_params(key)
    logging.info(f"Model params: {n_params(params):,}")

    params, ema_params, history, elapsed = train(
        params, ds.X_train, ds.y_train, ds.X_test, ds.y_test
    )

    final_acc = history["exact_acc"][-1]
    logging.info(f"Done. Final exact_acc={final_acc:.4f} in {elapsed:.0f}s")

    with open(PROJ_DIR / f"params_{EXP_NAME}.pkl", "wb") as f:
        pickle.dump(jax.tree.map(np.array, ema_params), f)
    with open(PROJ_DIR / f"history_{EXP_NAME}.pkl", "wb") as f:
        pickle.dump(history, f)

    append_result({
        "experiment":     EXP_NAME,
        "name":           f"UTM 2-block-per-step D={D_MODEL} K={K_PONDER} T={N_MEM} fresh",
        "d_model":        D_MODEL,
        "n_heads":        N_HEADS,
        "head_dim":       HEAD_DIM,
        "ffn_dim":        FFN_DIM,
        "n_mem":          N_MEM,
        "k_ponder":       K_PONDER,
        "n_blocks":       N_BLOCKS,
        "lambda_ponder":  LAMBDA_PONDER,
        "halt_bias":      HALT_BIAS,
        "n_train":        N_TRAIN,
        "n_params":       n_params(params),
        "lr":             LR,
        "batch_size":     BATCH_SIZE,
        "n_epochs":       N_EPOCHS,
        "ema_decay":      EMA_DECAY,
        "weight_decay":   WEIGHT_DECAY,
        "exact_acc":      final_acc,
        "history":        history,
        "time_s":         elapsed,
    })
