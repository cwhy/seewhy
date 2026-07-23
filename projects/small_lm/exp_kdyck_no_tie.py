"""
exp_kdyck_no_tie — k-Dyck procedural pre-training + separate (untied) LM head.

Combines:
  - Phase 1: 1,000 steps on k-Dyck balanced-bracket sequences (k=16)
  - Phase 2: 10,000 steps on Guppy conversations
  - Separate lm_head weight (not tied to token_embed)

What changed vs exp_kdyck:
  - Added: p["lm_head"] — a separate (VOCAB_SIZE, D_MODEL) output projection
  - forward() returns x @ p["lm_head"].T  instead of  x @ p["token_embed"].T
  - Parameter count increases by ~1.57M → ~10.3M total

Usage:
    uv run python projects/small_lm/scripts/run_experiments.py --bg exp_kdyck_no_tie
"""

import json, time, sys, pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
import numpy as np
import logging

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared_lib.random_utils import infinite_safe_keys_from_key

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

EXP_NAME = "exp_kdyck_no_tie"

# ── Hyperparameters ────────────────────────────────────────────────────────────

VOCAB_SIZE  = 4096
MAX_SEQ_LEN = 128
D_MODEL     = 384
N_LAYERS    = 6
N_HEADS     = 6
FFN_HIDDEN  = 768
HEAD_DIM    = D_MODEL // N_HEADS

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2

BATCH_SIZE    = 32
ADAM_LR       = 3e-4
WARMUP_STEPS  = 200
MAIN_STEPS    = 10_000
GRAD_CLIP     = 1.0
EVAL_EVERY    = 500

# k-Dyck warm-up
KDYCK_K             = 16
PRETRAIN_STEPS      = 1_000
PRETRAIN_LR         = 3e-4
KDYCK_EPOCH_SAMPLES = 10_000

SEED = 42
JSONL   = Path(__file__).parent / "results.jsonl"
PKL_DIR = Path(__file__).parent / "pkl"

# ── Precomputed constants ──────────────────────────────────────────────────────

CAUSAL_MASK = jnp.tril(jnp.ones((MAX_SEQ_LEN, MAX_SEQ_LEN), dtype=jnp.bool_))

# ── Utilities ─────────────────────────────────────────────────────────────────

def n_params(p):
    return sum(v.size for v in jax.tree_util.tree_leaves(p))

def append_result(row):
    with open(JSONL, "a") as f:
        f.write(json.dumps(row, default=lambda x: float(x) if hasattr(x, "__float__") else str(x)) + "\n")

# ── Parameter initialisation ──────────────────────────────────────────────────

def init_params(key_gen):
    def W(shape, std=0.02):
        return jax.random.normal(next(key_gen).get(), shape) * std

    p = {}
    p["token_embed"] = W((VOCAB_SIZE, D_MODEL))
    p["pos_embed"]   = W((MAX_SEQ_LEN, D_MODEL))

    for i in range(N_LAYERS):
        p[f"block_{i}_ln1_s"] = jnp.ones(D_MODEL)
        p[f"block_{i}_ln1_b"] = jnp.zeros(D_MODEL)
        p[f"block_{i}_wq"] = W((D_MODEL, D_MODEL))
        p[f"block_{i}_bq"] = jnp.zeros(D_MODEL)
        p[f"block_{i}_wk"] = W((D_MODEL, D_MODEL))
        p[f"block_{i}_bk"] = jnp.zeros(D_MODEL)
        p[f"block_{i}_wv"] = W((D_MODEL, D_MODEL))
        p[f"block_{i}_bv"] = jnp.zeros(D_MODEL)
        p[f"block_{i}_wo"] = W((D_MODEL, D_MODEL))
        p[f"block_{i}_bo"] = jnp.zeros(D_MODEL)
        p[f"block_{i}_ln2_s"] = jnp.ones(D_MODEL)
        p[f"block_{i}_ln2_b"] = jnp.zeros(D_MODEL)
        p[f"block_{i}_ffn_w1"] = W((D_MODEL, FFN_HIDDEN))
        p[f"block_{i}_ffn_b1"] = jnp.zeros(FFN_HIDDEN)
        p[f"block_{i}_ffn_w2"] = W((FFN_HIDDEN, D_MODEL))
        p[f"block_{i}_ffn_b2"] = jnp.zeros(D_MODEL)

    p["final_ln_s"] = jnp.ones(D_MODEL)
    p["final_ln_b"] = jnp.zeros(D_MODEL)
    p["lm_head"]    = W((VOCAB_SIZE, D_MODEL))  # separate, not tied to token_embed
    return p

# ── Core model ────────────────────────────────────────────────────────────────

def layer_norm(x, scale, bias, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var  = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return scale * (x - mean) / jnp.sqrt(var + eps) + bias


def attention(params, x, layer_i):
    B, T, _ = x.shape
    q = x @ params[f"block_{layer_i}_wq"] + params[f"block_{layer_i}_bq"]
    k = x @ params[f"block_{layer_i}_wk"] + params[f"block_{layer_i}_bk"]
    v = x @ params[f"block_{layer_i}_wv"] + params[f"block_{layer_i}_bv"]
    q = q.reshape(B, T, N_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
    k = k.reshape(B, T, N_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
    v = v.reshape(B, T, N_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
    scores  = (q @ k.transpose(0, 1, 3, 2)) * (HEAD_DIM ** -0.5)
    scores  = jnp.where(CAUSAL_MASK[:T, :T][None, None], scores, jnp.finfo(scores.dtype).min)
    weights = jax.nn.softmax(scores, axis=-1)
    out = (weights @ v).transpose(0, 2, 1, 3).reshape(B, T, D_MODEL)
    return out @ params[f"block_{layer_i}_wo"] + params[f"block_{layer_i}_bo"]


def transformer_block(params, x, layer_i):
    x = x + attention(params, layer_norm(x, params[f"block_{layer_i}_ln1_s"], params[f"block_{layer_i}_ln1_b"]), layer_i)
    h = layer_norm(x, params[f"block_{layer_i}_ln2_s"], params[f"block_{layer_i}_ln2_b"])
    h = jax.nn.relu(h @ params[f"block_{layer_i}_ffn_w1"] + params[f"block_{layer_i}_ffn_b1"])
    h = h @ params[f"block_{layer_i}_ffn_w2"] + params[f"block_{layer_i}_ffn_b2"]
    return x + h


def forward(params, token_ids):
    B, T = token_ids.shape
    x = params["token_embed"][token_ids] + params["pos_embed"][:T][None]
    for i in range(N_LAYERS):
        x = transformer_block(params, x, i)
    x = layer_norm(x, params["final_ln_s"], params["final_ln_b"])
    return x @ params["lm_head"].T


def loss_fn(params, token_ids):
    inputs  = token_ids[:, :-1]
    targets = token_ids[:, 1:]
    logits  = forward(params, inputs)
    B, T1, V = logits.shape
    flat_logits  = logits.reshape(-1, V)
    flat_targets = targets.reshape(-1)
    ce   = optax.softmax_cross_entropy_with_integer_labels(flat_logits, flat_targets)
    mask = (flat_targets != PAD_ID).astype(jnp.float32)
    return (ce * mask).sum() / (mask.sum() + 1e-8)

# ── Training step ──────────────────────────────────────────────────────────────

def make_train_step_fn(optimizer):
    def train_step(carry, batch):
        params, opt_state = carry
        loss, grads = jax.value_and_grad(loss_fn)(params, batch)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        return (optax.apply_updates(params, updates), new_opt_state), loss
    return jax.jit(train_step)


@jax.jit
def eval_loss_batch(params, token_ids):
    return loss_fn(params, token_ids)


def eval_metrics(params, eval_tokens, batch_size=32):
    losses = []
    n = (len(eval_tokens) // batch_size) * batch_size
    for i in range(0, n, batch_size):
        losses.append(float(eval_loss_batch(params, jnp.array(eval_tokens[i:i+batch_size]))))
    mean_loss = float(np.mean(losses))
    return {"eval_loss": mean_loss, "eval_ppl": float(np.exp(mean_loss))}

# ── Phase 1: k-Dyck warm-up ───────────────────────────────────────────────────

def pretrain_kdyck(params, kdyck):
    logging.info(f"Phase 1: k-Dyck warm-up for {PRETRAIN_STEPS} steps  "
                 f"(k={KDYCK_K}, seq_len={MAX_SEQ_LEN})")

    optimizer  = optax.chain(optax.clip_by_global_norm(GRAD_CLIP), optax.adamw(PRETRAIN_LR, weight_decay=0.0))
    opt_state  = optimizer.init(params)
    train_step = make_train_step_fn(optimizer)

    history = {"kdyck_loss": []}
    t0, step, epoch = time.perf_counter(), 0, 0

    while step < PRETRAIN_STEPS:
        for batch_np in kdyck.generate_epoch(KDYCK_EPOCH_SAMPLES, BATCH_SIZE, epoch):
            if step >= PRETRAIN_STEPS:
                break
            (params, opt_state), loss = train_step((params, opt_state), jnp.array(batch_np))
            history["kdyck_loss"].append(float(loss))
            step += 1
            if step % 100 == 0 or step == PRETRAIN_STEPS:
                logging.info(f"  [phase1] step {step:4d}/{PRETRAIN_STEPS}  "
                             f"kdyck_loss={float(loss):.4f}  {time.perf_counter()-t0:.0f}s")
        epoch += 1

    logging.info(f"Phase 1 done in {time.perf_counter()-t0:.0f}s")
    return params, opt_state, history

# ── Phase 2: Guppy fine-tuning ────────────────────────────────────────────────

def finetune_guppy(params, opt_state_phase1, train_tokens, eval_tokens):
    logging.info(f"Phase 2: Guppy fine-tuning for {MAIN_STEPS} steps")

    optimizer  = optax.chain(
        optax.clip_by_global_norm(GRAD_CLIP),
        optax.adamw(
            optax.warmup_cosine_decay_schedule(
                init_value=0.0, peak_value=ADAM_LR,
                warmup_steps=WARMUP_STEPS, decay_steps=MAIN_STEPS,
                end_value=ADAM_LR * 0.1,
            ),
            weight_decay=0.0,
        ),
    )
    opt_state  = optimizer.init(params)
    train_step = make_train_step_fn(optimizer)

    history = {"loss": [], "eval_loss": []}
    t0, step = time.perf_counter(), 0
    n_full = (len(train_tokens) // BATCH_SIZE) * BATCH_SIZE

    while step < MAIN_STEPS:
        perm   = np.random.permutation(len(train_tokens))
        tokens = train_tokens[perm[:n_full]].reshape(-1, BATCH_SIZE, MAX_SEQ_LEN)
        for batch in tokens:
            if step >= MAIN_STEPS:
                break
            (params, opt_state), loss = train_step((params, opt_state), jnp.array(batch))
            history["loss"].append(float(loss))
            step += 1
            if step % EVAL_EVERY == 0 or step == MAIN_STEPS:
                m = eval_metrics(params, eval_tokens)
                history["eval_loss"].append((step, m["eval_loss"]))
                logging.info(f"  [phase2] step {step:5d}  loss={float(loss):.4f}  "
                             f"eval_loss={m['eval_loss']:.4f}  eval_ppl={m['eval_ppl']:.2f}  "
                             f"{time.perf_counter()-t0:.0f}s")

    return params, history, time.perf_counter() - t0

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    done = set()
    if JSONL.exists():
        with open(JSONL) as f:
            for line in f:
                try: done.add(json.loads(line).get("experiment"))
                except Exception: pass
    if EXP_NAME in done:
        logging.info(f"{EXP_NAME} already done — skipping")
        exit(0)

    logging.info(f"JAX devices: {jax.devices()}")

    data_dir = Path(__file__).parent / "data"
    if not (data_dir / "train.jsonl").exists():
        raise FileNotFoundError("No data found. Run setup_data.py first.")

    sys.path.insert(0, str(Path(__file__).parent))
    from lib.data_processing import load_tokenizer, load_tokens
    from lib.kdyck import KDyck

    tok          = load_tokenizer(data_dir / "tokenizer.json")
    train_tokens = load_tokens(data_dir / "train.jsonl", tok, MAX_SEQ_LEN)
    eval_tokens  = load_tokens(data_dir / "eval.jsonl",  tok, MAX_SEQ_LEN)
    logging.info(f"Train: {train_tokens.shape}  Eval: {eval_tokens.shape}")

    kdyck = KDyck(k=KDYCK_K, seq_len=MAX_SEQ_LEN)

    key_gen = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED))
    params  = init_params(key_gen)
    logging.info(f"Parameters: {n_params(params):,}  (separate lm_head adds ~1.57M vs exp_kdyck)")

    t_total = time.perf_counter()

    params, opt_state_p1, history_p1 = pretrain_kdyck(params, kdyck)

    PKL_DIR.mkdir(exist_ok=True)
    with open(PKL_DIR / f"params_{EXP_NAME}_phase1.pkl", "wb") as f:
        pickle.dump({k: np.array(v) for k, v in params.items()}, f)
    logging.info(f"Phase 1 checkpoint saved → pkl/params_{EXP_NAME}_phase1.pkl")

    params, history_p2, elapsed_p2 = finetune_guppy(params, opt_state_p1, train_tokens, eval_tokens)

    elapsed = time.perf_counter() - t_total

    with open(PKL_DIR / f"params_{EXP_NAME}.pkl", "wb") as f:
        pickle.dump({k: np.array(v) for k, v in params.items()}, f)
    with open(PKL_DIR / f"history_{EXP_NAME}.pkl", "wb") as f:
        pickle.dump({"phase1": history_p1, "phase2": history_p2}, f)

    final = eval_metrics(params, eval_tokens)
    append_result({
        "experiment"      : EXP_NAME,
        "n_params"        : n_params(params),
        "pretrain_steps"  : PRETRAIN_STEPS,
        "max_steps"       : MAIN_STEPS,
        "time_s"          : round(elapsed, 1),
        "final_eval_loss" : round(final["eval_loss"], 6),
        "final_eval_ppl"  : round(final["eval_ppl"], 4),
        "loss_curve"      : [round(v, 6) for v in history_p2["loss"]],
        "eval_loss_curve" : history_p2["eval_loss"],
        "kdyck_loss_curve": [round(v, 6) for v in history_p1["kdyck_loss"]],
        "vocab_size"      : VOCAB_SIZE,
        "max_seq_len"     : MAX_SEQ_LEN,
        "d_model"         : D_MODEL,
        "n_layers"        : N_LAYERS,
        "n_heads"         : N_HEADS,
        "ffn_hidden"      : FFN_HIDDEN,
        "batch_size"      : BATCH_SIZE,
        "adam_lr"         : ADAM_LR,
        "warmup_steps"    : WARMUP_STEPS,
        "kdyck_k"         : KDYCK_K,
        "pretrain_lr"     : PRETRAIN_LR,
        "weight_tied"     : False,
    })
    logging.info(f"Done. eval_ppl={final['eval_ppl']:.4f}  total={elapsed:.0f}s")
