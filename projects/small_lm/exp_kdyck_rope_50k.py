"""
exp_kdyck_rope_50k — Same as exp_kdyck_rope but with GPT-2 vocabulary (50,260 tokens).

VOCAB_SIZE  = 50260  (GPT-2 base 50,257 + 3 special tokens)
PAD_ID      = 50257  (<pad>)
BOS_ID      = 50258  (<|im_start|>)
EOS_ID      = 50259  (<|im_end|>)

Everything else (D_MODEL=384, N_LAYERS=6, N_HEADS=6, MAX_SEQ_LEN=128, RoPE) is identical.
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

EXP_NAME = "exp_kdyck_rope_50k"

# ── Hyperparameters ────────────────────────────────────────────────────────────

VOCAB_SIZE  = 50260
MAX_SEQ_LEN = 128
D_MODEL     = 384
N_LAYERS    = 6
N_HEADS     = 6
FFN_HIDDEN  = 768
HEAD_DIM    = D_MODEL // N_HEADS

PAD_ID = 50257
BOS_ID = 50258
EOS_ID = 50259

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

def _build_rope_tables(seq_len: int, head_dim: int):
    half  = head_dim // 2
    theta = 1.0 / (10000 ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    t     = jnp.arange(seq_len, dtype=jnp.float32)
    freqs = jnp.outer(t, theta)
    cos   = jnp.concatenate([jnp.cos(freqs), jnp.cos(freqs)], axis=-1)
    sin   = jnp.concatenate([jnp.sin(freqs), jnp.sin(freqs)], axis=-1)
    return cos, sin

ROPE_COS, ROPE_SIN = _build_rope_tables(MAX_SEQ_LEN, HEAD_DIM)

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
    # No pos_embed — RoPE handles position in attention

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
    return p

# ── RoPE ──────────────────────────────────────────────────────────────────────

def rope_rotate(x, T):
    cos = ROPE_COS[:T][None, None, :, :]
    sin = ROPE_SIN[:T][None, None, :, :]
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    x_rot = jnp.concatenate([-x2, x1], axis=-1)
    return x * cos + x_rot * sin

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
    q = rope_rotate(q, T)
    k = rope_rotate(k, T)
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
    x = params["token_embed"][token_ids]   # no pos_embed
    for i in range(N_LAYERS):
        x = transformer_block(params, x, i)
    x = layer_norm(x, params["final_ln_s"], params["final_ln_b"])
    return x @ params["token_embed"].T


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
