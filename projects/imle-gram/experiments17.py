"""
exp17 — Clustering with load-balanced assignment.

Same N_VARS=1 Hebbian architecture as exp16.

Fixes codebook collapse via a two-phase assignment per batch:

  Phase 1 (top half — well covered):
    Sort batch images by distance to nearest prototype (ascending).
    Top half keep standard nearest-neighbor assignment.

  Phase 2 (bottom half — poorly covered):
    Remaining images are redistributed to under-used clusters.
    Clusters are served in ascending usage-count order; each cluster
    picks its best available (unassigned) image from the bottom half.

Algorithm (per batch):

  dists : (N, N_CODES)        full distance matrix
  nn    : argmin(dists, 1)    nearest prototype per image
  order : argsort(nn_dist)    sort by match quality
  top, bot = order[:N//2], order[N//2:]

  counts = bincount(nn[top], N_CODES)   # usage from top half

  rank[:,c] = argsort(dists[bot, c])    # per-cluster ranking of bot images
  ptr[c] = 0                            # pointer per cluster into rank[:,c]
  assigned = bool[n_bot]

  for _ in range(n_bot):
      c = argmin(counts)
      advance ptr[c] past already-assigned images
      assign rank[ptr[c], c] → cluster c
      counts[c] += 1

Complexity: O(N × N_CODES) per batch (dominated by distance matrix).
The greedy loop is O(n_bot × N_CODES) total pointer steps.

Pool is replaced by direct prototype decoding (N_CODES images, no sampling).

Usage:
    uv run python projects/imle-gram/experiments17.py
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
from shared_lib.datasets import load_supervised_image

sys.path.insert(0, str(Path(__file__).parent))
from lib.viz import save_reconstruction_grid, save_sample_grid, save_learning_curves

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

EXP_NAME = "exp17"

# ── Hyperparameters ────────────────────────────────────────────────────────────

N_VAL_BINS = 32
BATCH_SIZE = 128
SEED       = 42
MAX_EPOCHS = 100

D_R, D_C   = 6, 6
D_RC       = D_R * D_C

N_VARS   = 1     # pure clustering
N_CODES  = 32    # cluster count
D_EMB    = 16
D_U      = 16

LATENT_DIM = D_U * D_U    # 256
MLP_HIDDEN = 64
TEMP       = 10.0

ADAM_LR    = 3e-4

EVAL_EVERY = 5
VIZ_EVERY  = 20
M_EVAL     = 5000

JSONL = Path(__file__).parent / "results.jsonl"
ROWS  = jnp.arange(784) // 28
COLS  = jnp.arange(784) % 28
BIN_CENTERS = (jnp.arange(N_VAL_BINS) + 0.5) / N_VAL_BINS * 255.0

# ── Utilities ─────────────────────────────────────────────────────────────────

def quantize(x):
    return jnp.clip((x / 255.0 * N_VAL_BINS).astype(jnp.int32), 0, N_VAL_BINS - 1)

def n_params(p):
    return sum(v.size for v in jax.tree_util.tree_leaves(p))

def append_result(row):
    with open(JSONL, "a") as f:
        f.write(json.dumps(row, default=lambda x: float(x) if hasattr(x, "__float__") else str(x)) + "\n")

# ── Parameters ────────────────────────────────────────────────────────────────

def init_params(key_gen):
    def n(shape, scale=0.02):
        return jax.random.normal(next(key_gen).get(), shape) * scale
    f_dim = D_R * D_C * MLP_HIDDEN
    return {
        "codebook": n((N_CODES, D_EMB), scale=D_EMB**-0.5),
        "W_u": n((D_EMB, D_U), scale=D_EMB**-0.5),
        "b_u": jnp.zeros(D_U),
        "W_v": n((D_EMB, D_U), scale=D_EMB**-0.5),
        "b_v": jnp.zeros(D_U),
        "W_mlp1": n((LATENT_DIM, MLP_HIDDEN), scale=LATENT_DIM**-0.5),
        "b_mlp1": jnp.zeros(MLP_HIDDEN),
        "row_emb": n((28, D_R), scale=D_R**-0.5),
        "col_emb": n((28, D_C), scale=D_C**-0.5),
        "W_val1": n((MLP_HIDDEN, f_dim), scale=f_dim**-0.5),
        "b_val1": jnp.zeros(MLP_HIDDEN),
        "W_val2": n((N_VAL_BINS, MLP_HIDDEN), scale=MLP_HIDDEN**-0.5),
        "b_val2": jnp.zeros(N_VAL_BINS),
    }

# ── Codebook lookup ────────────────────────────────────────────────────────────

def _to_gram(Z, params):
    U = Z @ params["W_u"] + params["b_u"]
    V = Z @ params["W_v"] + params["b_v"]
    M = jnp.einsum("bni,bnj->bij", U, V)
    return M.reshape(Z.shape[0], -1)

def hard_lookup(params, indices):
    Z = params["codebook"][indices, :]
    return _to_gram(Z, params)

def soft_lookup(params, indices):
    oh = jax.nn.one_hot(indices, N_CODES)
    w  = jax.nn.softmax(oh * TEMP, axis=-1)
    Z  = w @ params["codebook"]
    return _to_gram(Z, params)

# ── Decoder ───────────────────────────────────────────────────────────────────

def decode(params, z):
    h = jax.nn.gelu(z @ params["W_mlp1"] + params["b_mlp1"])

    def decode_image(h_i):
        def decode_pixel(r, c):
            re = params["row_emb"][r]
            ce = params["col_emb"][c]
            f  = (re[:, None, None] * ce[None, :, None] * h_i[None, None, :]).reshape(-1)
            h1 = jax.nn.gelu(params["W_val1"] @ f + params["b_val1"])
            return params["W_val2"] @ h1 + params["b_val2"]
        return jax.vmap(decode_pixel, in_axes=(0, 0))(ROWS, COLS)

    return jax.vmap(decode_image)(h)

# ── Prototype decoding ────────────────────────────────────────────────────────

def decode_prototypes(params):
    """Decode all N_CODES cluster prototypes to pixel space. (N_CODES, 784)"""
    idx = jnp.arange(N_CODES).reshape(N_CODES, 1)
    z   = hard_lookup(params, idx)
    logits  = decode(params, z)
    probs   = jax.nn.softmax(logits, axis=-1)
    return np.array((probs * BIN_CENTERS[None, None, :]).sum(-1))   # (N_CODES, 784)

# ── Load-balanced assignment ──────────────────────────────────────────────────

def load_balanced_assign(X_real, X_proto):
    """
    X_real  : (N, 784)       batch of training images
    X_proto : (N_CODES, 784) decoded cluster prototypes
    Returns : (N,) int       cluster index per image
    """
    N     = len(X_real)
    half  = N // 2

    # Full distance matrix (N, N_CODES)
    d_r   = (X_real**2).sum(1, keepdims=True)
    d_t   = (X_proto**2).sum(1)
    dists = d_r + d_t - 2.0 * (X_real @ X_proto.T)

    nn      = np.argmin(dists, axis=1)                     # (N,)
    order   = np.argsort(dists[np.arange(N), nn])          # ascending by nn dist
    top_idx = order[:half]
    bot_idx = order[half:]
    n_bot   = len(bot_idx)

    assignment = nn.copy()

    # Top half: keep nearest-neighbor
    counts = np.bincount(nn[top_idx], minlength=N_CODES).astype(np.float32)

    # Bottom half: greedy load-balanced
    bot_dists = dists[bot_idx]                              # (n_bot, N_CODES)
    rank      = np.argsort(bot_dists, axis=0)               # (n_bot, N_CODES)
    # rank[:, c] = local bot indices sorted by distance to cluster c

    ptr      = np.zeros(N_CODES, dtype=int)
    assigned = np.zeros(n_bot,   dtype=bool)

    for _ in range(n_bot):
        c = int(np.argmin(counts))
        # Advance past already-assigned images
        while ptr[c] < n_bot and assigned[rank[ptr[c], c]]:
            ptr[c] += 1
        if ptr[c] >= n_bot:
            counts[c] = np.inf   # cluster exhausted its options
            continue
        local = rank[ptr[c], c]
        assignment[bot_idx[local]] = c
        assigned[local] = True
        counts[c] += 1
        ptr[c] += 1

    return assignment

# ── Loss & training step ──────────────────────────────────────────────────────

def imle_loss(params, x, matched_indices):
    z_soft = soft_lookup(params, matched_indices)
    logits = decode(params, z_soft)
    bins   = quantize(x)
    log_p  = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.take_along_axis(log_p, bins[..., None], axis=-1)[..., 0].mean()

def make_train_step(optimizer):
    @jax.jit
    def step(params, opt_state, x, matched_indices):
        loss, grads = jax.value_and_grad(imle_loss)(params, x, matched_indices)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), new_opt_state, loss
    return step

# ── Evaluation ────────────────────────────────────────────────────────────────

def eval_metrics(params, X_eval):
    X_proto = decode_prototypes(params)
    match_losses, bin_correct, bin_total = [], 0, 0
    for i in range(0, len(X_eval), BATCH_SIZE):
        x_batch  = np.array(X_eval[i:i+BATCH_SIZE])
        d_r      = (x_batch**2).sum(1, keepdims=True)
        d_t      = (X_proto**2).sum(1)
        dists    = d_r + d_t - 2.0 * (x_batch @ X_proto.T)
        idx      = np.argmin(dists, axis=1)             # nearest prototype (no load balancing for eval)
        match_losses.append(((x_batch - X_proto[idx])**2).mean())
        logits   = decode(params, hard_lookup(params, jnp.array(idx).reshape(-1, 1)))
        preds    = np.array(jnp.argmax(logits, axis=-1))
        true     = np.clip((x_batch / 255.0 * N_VAL_BINS).astype(int), 0, N_VAL_BINS - 1)
        bin_correct += (preds == true).sum()
        bin_total   += preds.size
    return float(np.mean(match_losses)), float(bin_correct / bin_total)

# ── Training loop ─────────────────────────────────────────────────────────────

def train(params, X_tr, X_te):
    num_steps   = MAX_EPOCHS * (len(X_tr) // BATCH_SIZE)
    optimizer   = optax.adam(optax.cosine_decay_schedule(ADAM_LR, num_steps, alpha=0.05))
    opt_state   = optimizer.init(params)
    step_fn     = make_train_step(optimizer)
    key_gen     = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED + 1))
    num_batches = len(X_tr) // BATCH_SIZE
    history     = {"match_loss": [], "match_loss_eval": [], "bin_acc": []}
    t0 = time.perf_counter()

    for epoch in range(MAX_EPOCHS):
        key  = next(key_gen).get()
        perm = np.array(jax.random.permutation(key, len(X_tr)))
        Xs   = np.array(X_tr)[perm]

        # Decode prototypes once per epoch
        X_proto = decode_prototypes(params)

        # Log cluster usage
        all_assign  = load_balanced_assign(Xs[:2048], X_proto)
        usage       = np.bincount(all_assign, minlength=N_CODES)
        n_active    = int((usage > 0).sum())

        e_loss = e_match = 0.0
        for b in range(num_batches):
            x_batch = Xs[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
            idx     = load_balanced_assign(x_batch, X_proto)          # (BATCH_SIZE,)
            e_match += ((x_batch - X_proto[idx])**2).mean()
            matched = jnp.array(idx).reshape(-1, 1)                   # (BATCH_SIZE, 1)
            params, opt_state, loss = step_fn(params, opt_state, jnp.array(x_batch), matched)
            e_loss += float(loss)

        e_loss /= num_batches; e_match /= num_batches
        history["match_loss"].append(e_match)

        eval_str = ""
        if (epoch + 1) % EVAL_EVERY == 0:
            ml_eval, bin_acc = eval_metrics(params, X_te)
            history["match_loss_eval"].append((epoch, ml_eval))
            history["bin_acc"].append((epoch, bin_acc))
            eval_str = f"  eval_match={ml_eval:.1f}  bin_acc={bin_acc:.1%}"

        logging.info(f"epoch {epoch:3d}  nll={e_loss:.4f}  match_l2={e_match:.1f}"
                     f"  active={n_active}/{N_CODES}{eval_str}  {time.perf_counter()-t0:.1f}s")

        if (epoch + 1) % VIZ_EVERY == 0:
            save_reconstruction_grid(params, X_te, decode, LATENT_DIM, BIN_CENTERS, EXP_NAME, f"epoch{epoch+1:03d}")

    return params, history, time.perf_counter() - t0

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.info(f"JAX devices: {jax.devices()}")
    data    = load_supervised_image("mnist")
    X_train = data.X.reshape(data.n_samples,           784).astype(np.float32)
    X_test  = data.X_test.reshape(data.n_test_samples, 784).astype(np.float32)

    key_gen = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED))
    params  = init_params(key_gen)
    logging.info(f"Parameters: {n_params(params):,}  (load-balanced clustering N_CODES={N_CODES})")

    params, history, elapsed = train(params, X_train, X_test)

    with open(Path(__file__).parent / f"params_{EXP_NAME}.pkl", "wb") as f:
        pickle.dump({k: np.array(v) for k, v in params.items()}, f)

    save_learning_curves(history, EXP_NAME, subtitle=f"load-balanced clustering N_CODES={N_CODES} D_U={D_U}")
    save_reconstruction_grid(params, X_test, decode, LATENT_DIM, BIN_CENTERS, EXP_NAME, "final")
    save_sample_grid(params, decode, LATENT_DIM, BIN_CENTERS, EXP_NAME, rows=4, cols=8)

    final_ml, final_acc = eval_metrics(params, X_test)
    append_result({
        "experiment": EXP_NAME,
        "name": f"imle-cluster-lb N_CODES={N_CODES} D_EMB={D_EMB} D_U={D_U}",
        "n_params": n_params(params), "epochs": MAX_EPOCHS, "time_s": round(elapsed, 1),
        "final_bin_acc": round(float(final_acc), 6), "final_match_l2": round(float(final_ml), 4),
        "match_loss_curve": [round(float(v), 4) for v in history["match_loss"]],
        "bin_acc_curve": history["bin_acc"],
        "n_codes": N_CODES, "d_emb": D_EMB, "d_u": D_U,
        "n_val_bins": N_VAL_BINS, "adam_lr": ADAM_LR,
    })
    logging.info(f"Done. match_l2={final_ml:.1f}  bin_acc={final_acc:.1%}  {elapsed:.1f}s")
