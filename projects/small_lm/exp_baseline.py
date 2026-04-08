"""
exp_baseline — Direct JAX port of GuppyLM: 6-layer vanilla transformer, d_model=384.

Architecture:
    Token + learned positional embeddings → 6× pre-norm transformer blocks
    (MHA + ReLU FFN) → LayerNorm → weight-tied LM head.
    No dropout, no RoPE, no GQA — educational simplicity.

Training: teacher-forcing cross-entropy, cosine LR with linear warmup.
Eval metric: test-set perplexity (exp of mean CE loss, ignoring pad tokens).

Status: FORWARD PASS ONLY.  Training: TODO.

Usage:
    uv run python projects/small_lm/scripts/tmp/check_forward.py   # sanity check
    uv run python projects/small_lm/exp_baseline.py                 # (training not yet impl)
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

EXP_NAME = "exp_baseline"

# ── Hyperparameters ────────────────────────────────────────────────────────────

# Architecture (matches GuppyLM exactly)
VOCAB_SIZE  = 4096
MAX_SEQ_LEN = 128
D_MODEL     = 384
N_LAYERS    = 6
N_HEADS     = 6
FFN_HIDDEN  = 768
HEAD_DIM    = D_MODEL // N_HEADS   # 64

# Special token IDs
PAD_ID = 0   # <pad>
BOS_ID = 1   # <|im_start|>
EOS_ID = 2   # <|im_end|>

# Training (TODO)
BATCH_SIZE    = 32
ADAM_LR       = 3e-4
WARMUP_STEPS  = 200
MAX_STEPS     = 10_000
GRAD_CLIP     = 1.0
EVAL_EVERY    = 500   # steps

SEED = 42
JSONL = Path(__file__).parent / "results.jsonl"
PKL_DIR = Path(__file__).parent / "pkl"

# ── Precomputed constants ──────────────────────────────────────────────────────

# Causal mask: True where attention is allowed (lower triangle)
CAUSAL_MASK = jnp.tril(jnp.ones((MAX_SEQ_LEN, MAX_SEQ_LEN), dtype=jnp.bool_))

# ── Utilities ─────────────────────────────────────────────────────────────────

def n_params(p: dict) -> int:
    return sum(v.size for v in jax.tree_util.tree_leaves(p))

def append_result(row: dict) -> None:
    with open(JSONL, "a") as f:
        f.write(json.dumps(row, default=lambda x: float(x) if hasattr(x, "__float__") else str(x)) + "\n")

# ── Parameter initialisation ──────────────────────────────────────────────────

def init_params(key_gen) -> dict:
    """Initialise all model parameters (flat dict, UPPER_SNAKE_CASE hyperparams above).

    Embeddings:   normal(0, 0.02)  — matches GuppyLM _init_weights
    Projections:  normal(0, 0.02)
    LayerNorm:    scale=1, bias=0
    FFN biases:   zeros
    """
    def W(shape, std=0.02):
        return jax.random.normal(next(key_gen).get(), shape) * std

    p = {}

    # ── Embeddings ─────────────────────────────────────────────────────────────
    p["token_embed"] = W((VOCAB_SIZE, D_MODEL))     # also the LM head (weight-tied)
    p["pos_embed"]   = W((MAX_SEQ_LEN, D_MODEL))

    # ── Transformer blocks ─────────────────────────────────────────────────────
    for i in range(N_LAYERS):
        # Pre-attention LayerNorm
        p[f"block_{i}_ln1_s"] = jnp.ones(D_MODEL)
        p[f"block_{i}_ln1_b"] = jnp.zeros(D_MODEL)

        # Multi-head attention projections (bias on all, matching nn.Linear defaults)
        p[f"block_{i}_wq"] = W((D_MODEL, D_MODEL))
        p[f"block_{i}_bq"] = jnp.zeros(D_MODEL)
        p[f"block_{i}_wk"] = W((D_MODEL, D_MODEL))
        p[f"block_{i}_bk"] = jnp.zeros(D_MODEL)
        p[f"block_{i}_wv"] = W((D_MODEL, D_MODEL))
        p[f"block_{i}_bv"] = jnp.zeros(D_MODEL)
        p[f"block_{i}_wo"] = W((D_MODEL, D_MODEL))
        p[f"block_{i}_bo"] = jnp.zeros(D_MODEL)

        # Pre-FFN LayerNorm
        p[f"block_{i}_ln2_s"] = jnp.ones(D_MODEL)
        p[f"block_{i}_ln2_b"] = jnp.zeros(D_MODEL)

        # FFN (ReLU, matching GuppyLM)
        p[f"block_{i}_ffn_w1"] = W((D_MODEL, FFN_HIDDEN))
        p[f"block_{i}_ffn_b1"] = jnp.zeros(FFN_HIDDEN)
        p[f"block_{i}_ffn_w2"] = W((FFN_HIDDEN, D_MODEL))
        p[f"block_{i}_ffn_b2"] = jnp.zeros(D_MODEL)

    # ── Final LayerNorm ────────────────────────────────────────────────────────
    p["final_ln_s"] = jnp.ones(D_MODEL)
    p["final_ln_b"] = jnp.zeros(D_MODEL)

    # Note: no separate lm_head — logits = h @ p["token_embed"].T  (weight-tied)
    return p


# ── Core primitives ───────────────────────────────────────────────────────────

def layer_norm(x: jnp.ndarray, scale: jnp.ndarray, bias: jnp.ndarray, eps: float = 1e-5) -> jnp.ndarray:
    """Pre-norm LayerNorm over last axis."""
    mean = x.mean(axis=-1, keepdims=True)
    var  = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return scale * (x - mean) / jnp.sqrt(var + eps) + bias


def attention(params: dict, x: jnp.ndarray, layer_i: int) -> jnp.ndarray:
    """Causal multi-head self-attention.

    Args:
        x: (B, T, D_MODEL)
    Returns:
        (B, T, D_MODEL)
    """
    B, T, _ = x.shape

    # Project Q, K, V
    q = x @ params[f"block_{layer_i}_wq"] + params[f"block_{layer_i}_bq"]  # (B, T, D)
    k = x @ params[f"block_{layer_i}_wk"] + params[f"block_{layer_i}_bk"]
    v = x @ params[f"block_{layer_i}_wv"] + params[f"block_{layer_i}_bv"]

    # Split heads: (B, T, D) → (B, H, T, d)
    q = q.reshape(B, T, N_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
    k = k.reshape(B, T, N_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
    v = v.reshape(B, T, N_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)

    # Scaled dot-product attention
    scale  = HEAD_DIM ** -0.5
    scores = (q @ k.transpose(0, 1, 3, 2)) * scale              # (B, H, T, T)
    scores = jnp.where(CAUSAL_MASK[:T, :T][None, None], scores, jnp.finfo(scores.dtype).min)
    weights = jax.nn.softmax(scores, axis=-1)                   # (B, H, T, T)

    # Aggregate and project output
    out = weights @ v                                            # (B, H, T, d)
    out = out.transpose(0, 2, 1, 3).reshape(B, T, D_MODEL)      # (B, T, D)
    return out @ params[f"block_{layer_i}_wo"] + params[f"block_{layer_i}_bo"]


def transformer_block(params: dict, x: jnp.ndarray, layer_i: int) -> jnp.ndarray:
    """Pre-norm transformer block: LayerNorm → Attention → residual + LayerNorm → FFN → residual."""
    # Attention sub-layer
    x = x + attention(params, layer_norm(x, params[f"block_{layer_i}_ln1_s"], params[f"block_{layer_i}_ln1_b"]), layer_i)

    # FFN sub-layer (ReLU, matching GuppyLM)
    h = layer_norm(x, params[f"block_{layer_i}_ln2_s"], params[f"block_{layer_i}_ln2_b"])
    h = jax.nn.relu(h @ params[f"block_{layer_i}_ffn_w1"] + params[f"block_{layer_i}_ffn_b1"])
    h = h @ params[f"block_{layer_i}_ffn_w2"] + params[f"block_{layer_i}_ffn_b2"]
    return x + h


def forward(params: dict, token_ids: jnp.ndarray) -> jnp.ndarray:
    """Full transformer forward pass.

    Args:
        token_ids: int32 array (B, T), T ≤ MAX_SEQ_LEN.

    Returns:
        logits: float32 array (B, T, VOCAB_SIZE).
    """
    B, T = token_ids.shape

    # Embeddings: token + position
    tok_emb = params["token_embed"][token_ids]       # (B, T, D_MODEL)
    pos_emb = params["pos_embed"][:T]                # (T, D_MODEL)
    x = tok_emb + pos_emb[None, :, :]               # (B, T, D_MODEL)

    # 6 transformer blocks (Python loop — unrolled by JAX JIT)
    for i in range(N_LAYERS):
        x = transformer_block(params, x, i)

    # Final LayerNorm
    x = layer_norm(x, params["final_ln_s"], params["final_ln_b"])

    # Weight-tied LM head: logits = x @ token_embed.T
    logits = x @ params["token_embed"].T             # (B, T, VOCAB_SIZE)
    return logits


# ── Loss function ─────────────────────────────────────────────────────────────

def loss_fn(params: dict, token_ids: jnp.ndarray) -> jnp.ndarray:
    """Cross-entropy loss for next-token prediction, ignoring pad tokens.

    Args:
        token_ids: int32 (B, T)  — full padded sequences (length MAX_SEQ_LEN).

    Returns:
        scalar mean CE loss over non-pad target tokens.
    """
    inputs  = token_ids[:, :-1]    # (B, T-1)
    targets = token_ids[:, 1:]     # (B, T-1)

    logits = forward(params, inputs)   # (B, T-1, VOCAB_SIZE)

    # Flatten and compute cross-entropy
    B, T_minus1, V = logits.shape
    flat_logits  = logits.reshape(-1, V)        # (B*(T-1), V)
    flat_targets = targets.reshape(-1)           # (B*(T-1),)

    ce_per_token = optax.softmax_cross_entropy_with_integer_labels(flat_logits, flat_targets)

    # Mask out padding positions in targets
    pad_mask = (flat_targets != PAD_ID).astype(jnp.float32)
    return (ce_per_token * pad_mask).sum() / (pad_mask.sum() + 1e-8)


# ── Epoch function (lax.scan over batches) ────────────────────────────────────

def make_train_step_fn(optimizer):
    """Return a JIT-compiled function that processes one batch."""
    def train_step(carry, token_ids_batch):
        params, opt_state = carry
        loss, grads = jax.value_and_grad(loss_fn)(params, token_ids_batch)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return (new_params, new_opt_state), loss

    return jax.jit(train_step)


def make_epoch_fn(optimizer):
    """Return a JIT-compiled epoch function (lax.scan over batches)."""
    def scan_body(carry, token_ids_batch):
        params, opt_state = carry
        loss, grads = jax.value_and_grad(loss_fn)(params, token_ids_batch)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        return (optax.apply_updates(params, updates), new_opt_state), loss

    def epoch_fn(params, opt_state, tokens_batched):
        # tokens_batched: (num_batches, BATCH_SIZE, MAX_SEQ_LEN)
        (params, opt_state), losses = jax.lax.scan(
            scan_body, (params, opt_state), tokens_batched
        )
        return params, opt_state, losses.mean()

    return jax.jit(epoch_fn)


# ── Evaluation ────────────────────────────────────────────────────────────────

@jax.jit
def eval_loss(params: dict, token_ids: jnp.ndarray) -> jnp.ndarray:
    """CE loss on a single batch (no grad)."""
    return loss_fn(params, token_ids)


def eval_metrics(params: dict, eval_tokens: np.ndarray, batch_size: int = 32) -> dict:
    """Compute mean CE loss and perplexity over eval set.

    Processes in chunks to avoid OOM.
    """
    losses = []
    n = (len(eval_tokens) // batch_size) * batch_size
    for i in range(0, n, batch_size):
        batch = jnp.array(eval_tokens[i:i + batch_size])
        losses.append(float(eval_loss(params, batch)))

    mean_loss = float(np.mean(losses))
    return {"eval_loss": mean_loss, "eval_ppl": float(np.exp(mean_loss))}


# ── Training loop ─────────────────────────────────────────────────────────────

def train(params: dict, train_tokens: np.ndarray, eval_tokens: np.ndarray) -> tuple:
    """Train for MAX_STEPS with cosine LR + linear warmup.

    Returns:
        (params, history, elapsed_seconds)
    """
    num_steps   = MAX_STEPS
    optimizer   = optax.chain(
        optax.clip_by_global_norm(GRAD_CLIP),
        optax.adamw(
            optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=ADAM_LR,
                warmup_steps=WARMUP_STEPS,
                decay_steps=num_steps,
                end_value=ADAM_LR * 0.1,
            ),
            weight_decay=0.0,
        ),
    )
    opt_state = optimizer.init(params)
    train_step = make_train_step_fn(optimizer)
    key_gen    = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED + 1))

    history = {"loss": [], "eval_loss": []}
    t0      = time.perf_counter()
    step    = 0

    n_full = (len(train_tokens) // BATCH_SIZE) * BATCH_SIZE

    while step < num_steps:
        # Shuffle and batch
        perm   = np.random.permutation(len(train_tokens))
        tokens = train_tokens[perm[:n_full]].reshape(-1, BATCH_SIZE, MAX_SEQ_LEN)

        for batch in tokens:
            if step >= num_steps:
                break
            (params, opt_state), loss = train_step((params, opt_state), jnp.array(batch))
            loss = float(loss)
            history["loss"].append(loss)
            step += 1

            if step % EVAL_EVERY == 0 or step == num_steps:
                metrics = eval_metrics(params, eval_tokens)
                history["eval_loss"].append((step, metrics["eval_loss"]))
                logging.info(
                    f"step {step:5d}  loss={loss:.4f}  "
                    f"eval_loss={metrics['eval_loss']:.4f}  "
                    f"eval_ppl={metrics['eval_ppl']:.2f}  "
                    f"{time.perf_counter()-t0:.0f}s"
                )

    elapsed = time.perf_counter() - t0
    return params, history, elapsed


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Skip-if-done
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

    # Load data
    data_dir = Path(__file__).parent / "data"
    tok_path = data_dir / "tokenizer.json"
    if not (data_dir / "train.jsonl").exists():
        raise FileNotFoundError(
            f"No data found at {data_dir}. "
            "Run: uv run python projects/small_lm/scripts/tmp/setup_data.py"
        )

    sys.path.insert(0, str(Path(__file__).parent))
    from lib.data_processing import load_tokenizer, load_tokens

    tok          = load_tokenizer(tok_path)
    train_tokens = load_tokens(data_dir / "train.jsonl", tok, MAX_SEQ_LEN)
    eval_tokens  = load_tokens(data_dir / "eval.jsonl",  tok, MAX_SEQ_LEN)
    logging.info(f"Train: {train_tokens.shape}  Eval: {eval_tokens.shape}")

    # Init
    key_gen = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED))
    params  = init_params(key_gen)
    logging.info(f"Parameters: {n_params(params):,}")

    # Train
    params, history, elapsed = train(params, train_tokens, eval_tokens)

    # Save params
    PKL_DIR.mkdir(exist_ok=True)
    with open(PKL_DIR / f"params_{EXP_NAME}.pkl", "wb") as f:
        pickle.dump({k: np.array(v) for k, v in params.items()}, f)
    with open(PKL_DIR / f"history_{EXP_NAME}.pkl", "wb") as f:
        pickle.dump(history, f)

    final = eval_metrics(params, eval_tokens)

    append_result({
        "experiment"  : EXP_NAME,
        "n_params"    : n_params(params),
        "max_steps"   : MAX_STEPS,
        "time_s"      : round(elapsed, 1),
        "final_eval_loss": round(final["eval_loss"], 6),
        "final_eval_ppl" : round(final["eval_ppl"], 4),
        "loss_curve"  : [round(v, 6) for v in history["loss"]],
        "eval_loss_curve": history["eval_loss"],
        # Hyperparams
        "vocab_size"  : VOCAB_SIZE,
        "max_seq_len" : MAX_SEQ_LEN,
        "d_model"     : D_MODEL,
        "n_layers"    : N_LAYERS,
        "n_heads"     : N_HEADS,
        "ffn_hidden"  : FFN_HIDDEN,
        "batch_size"  : BATCH_SIZE,
        "adam_lr"     : ADAM_LR,
        "warmup_steps": WARMUP_STEPS,
    })
    logging.info(f"Done. eval_ppl={final['eval_ppl']:.2f}  {elapsed:.0f}s")
