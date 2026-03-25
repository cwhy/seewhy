"""
exp26 — 128-cluster flat delta_sum + codebook cosine repulsion.

Same as exp21 but adds a pairwise cosine repulsion loss on the codebook
to push embeddings apart and prevent prototype collapse:

    Z_n   = codebook / ||codebook||          # unit-normed rows
    G     = Z_n @ Z_n.T                     # (K,K) cosine similarities
    L_rep = (||G||_F² - K) / (K*(K-1))     # normalised to [0,1]
    loss  = nll + REP_LAMBDA * L_rep

REP_LAMBDA=0.1 caps the repulsion at ~7% of the NLL.

Same architecture as exp20 (N_VARS=1 Hebbian, N_CODES=128), but replaces
the mean-split load-balanced assignment with a delta_sum priority scheme.

Assignment algorithm (per batch):
  - Maintain delta_sum[c]: accumulated assignment cost for each cluster.
  - Build cost matrix: costs[i, c] = delta_sum[c] + dist(image_i, prototype_c).
  - Greedily pick the minimum-cost (image, cluster) pair:
      assign image i to cluster c,
      set costs[i, :] = inf  (image taken),
      add dist[i, c] to costs[:, c]  (delta_sum propagates in-place).
  - Repeat until all images are assigned.

No top/bottom split — every image goes through the same priority queue.
delta_sum resets each batch (fresh per call).

Compare with exp20 (mean-split load-balanced) to see if the natural
priority-queue approach yields better cluster utilisation.

Usage:
    uv run python projects/imle-gram/experiments26.py
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

EXP_NAME = "exp26"

# ── Hyperparameters ────────────────────────────────────────────────────────────

N_VAL_BINS = 32
BATCH_SIZE = 128
SEED       = 42
MAX_EPOCHS = 100

D_R, D_C   = 6, 6
D_RC       = D_R * D_C

N_VARS   = 1
N_CODES  = 128    # 7×7 — baseline for exp19
D_EMB    = 16
D_U      = 16

LATENT_DIM = D_U * D_U
MLP_HIDDEN = 64
TEMP       = 10.0

ADAM_LR    = 3e-4

REP_LAMBDA = 0.1   # weight for codebook cosine repulsion loss

EVAL_EVERY = 5
VIZ_EVERY  = 999
M_EVAL     = 5000

N_MNIST_CLASSES = 10

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
    """Decode all N_CODES cluster prototypes. Returns (N_CODES, 784)."""
    idx    = jnp.arange(N_CODES).reshape(N_CODES, 1)
    z      = hard_lookup(params, idx)
    logits = decode(params, z)
    probs  = jax.nn.softmax(logits, axis=-1)
    return np.array((probs * BIN_CENTERS[None, None, :]).sum(-1))   # (N_CODES, 784)

# ── Delta-sum assignment ───────────────────────────────────────────────────────

def load_balanced_assign(X_real, X_proto):
    """
    Assign each image to a cluster via delta_sum priority.

    cost(i, c) = delta_sum[c] + dist(i, c).
    Greedy: each step picks the minimum-cost (image, cluster) pair.
    delta_sum resets each call (per-batch scope).

    Optimised O(N·N_CODES) implementation:
      Precompute rank = argsort(dists, axis=0) once per batch.
      Per step, only evaluate per-cluster best images (N_CODES values)
      instead of scanning the full N×N_CODES matrix.
      Mathematically identical to the naive O(N²·N_CODES) version.
    """
    N    = len(X_real)
    cidx = np.arange(N_CODES)

    d_r   = (X_real**2).sum(1, keepdims=True)
    d_t   = (X_proto**2).sum(1)
    dists = d_r + d_t - 2.0 * (X_real @ X_proto.T)   # (N, N_CODES)

    rank       = np.argsort(dists, axis=0)             # (N, N_CODES) — sorted ascending per cluster
    delta_sum  = np.zeros(N_CODES, dtype=dists.dtype)
    ptr        = np.zeros(N_CODES, dtype=np.intp)
    assigned   = np.zeros(N, dtype=bool)
    assignment = np.empty(N, dtype=np.intp)

    for _ in range(N):
        cur_imgs  = rank[ptr, cidx]                          # best unassigned image per cluster
        cur_costs = delta_sum + dists[cur_imgs, cidx]        # (N_CODES,)

        c   = int(np.argmin(cur_costs))
        img = int(cur_imgs[c])

        assignment[img] = c
        assigned[img]   = True
        delta_sum[c]   += dists[img, c]

        # Advance pointers for all clusters that were pointing to the now-assigned image
        for cs in np.where(cur_imgs == img)[0]:
            ptr[cs] += 1
            while ptr[cs] < N and assigned[rank[ptr[cs], cs]]:
                ptr[cs] += 1

    return assignment

# ── Label accuracy on test set ────────────────────────────────────────────────

def label_accuracy(params, X_te, Y_te):
    X_proto = decode_prototypes(params)
    d_r     = (X_te**2).sum(1, keepdims=True)
    d_t     = (X_proto**2).sum(1)
    dists   = d_r + d_t - 2.0 * (X_te @ X_proto.T)
    assign  = np.argmin(dists, axis=1)

    label_probs = np.zeros((N_CODES, N_MNIST_CLASSES))
    for c in range(N_CODES):
        mask = assign == c
        if mask.sum() > 0:
            label_probs[c] = np.bincount(Y_te[mask], minlength=N_MNIST_CLASSES) / mask.sum()
        else:
            label_probs[c] = 1.0 / N_MNIST_CLASSES

    majority = label_probs.argmax(axis=1)
    hard_acc = float((majority[assign] == Y_te).mean())

    scale       = dists.mean()
    p_c_given_x = np.exp(-dists / (scale + 1e-8))
    p_c_given_x /= p_c_given_x.sum(axis=1, keepdims=True)
    p_y_given_x  = p_c_given_x @ label_probs
    soft_pred    = p_y_given_x.argmax(axis=1)
    soft_acc     = float((soft_pred == Y_te).mean())

    return hard_acc, soft_acc, label_probs

# ── Loss & training step ──────────────────────────────────────────────────────

def imle_loss(params, x, matched_indices):
    z_soft = soft_lookup(params, matched_indices)
    logits = decode(params, z_soft)
    bins   = quantize(x)
    log_p  = jax.nn.log_softmax(logits, axis=-1)
    nll    = -jnp.take_along_axis(log_p, bins[..., None], axis=-1)[..., 0].mean()

    # Codebook cosine repulsion: push all pairs toward orthogonality
    Z_n   = params["codebook"] / (jnp.linalg.norm(params["codebook"], axis=1, keepdims=True) + 1e-8)
    G     = Z_n @ Z_n.T                                         # (K, K) cosine similarities
    L_rep = ((G ** 2).sum() - N_CODES) / (N_CODES * (N_CODES - 1))   # normalised to [0,1]

    return nll + REP_LAMBDA * L_rep

def make_train_step(optimizer):
    @jax.jit
    def step(params, opt_state, x, matched_indices):
        loss, grads = jax.value_and_grad(imle_loss)(params, x, matched_indices)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), new_opt_state, loss
    return step

# ── Evaluation (pixel accuracy) ───────────────────────────────────────────────

def eval_metrics(params, X_eval):
    X_proto = decode_prototypes(params)
    match_losses, bin_correct, bin_total = [], 0, 0
    for i in range(0, len(X_eval), BATCH_SIZE):
        x_batch  = np.array(X_eval[i:i+BATCH_SIZE])
        d_r      = (x_batch**2).sum(1, keepdims=True)
        d_t      = (X_proto**2).sum(1)
        dists    = d_r + d_t - 2.0 * (x_batch @ X_proto.T)
        idx      = np.argmin(dists, axis=1)
        match_losses.append(((x_batch - X_proto[idx])**2).mean())
        logits   = decode(params, hard_lookup(params, jnp.array(idx).reshape(-1, 1)))
        preds    = np.array(jnp.argmax(logits, axis=-1))
        true     = np.clip((x_batch / 255.0 * N_VAL_BINS).astype(int), 0, N_VAL_BINS - 1)
        bin_correct += (preds == true).sum()
        bin_total   += preds.size
    return float(np.mean(match_losses)), float(bin_correct / bin_total)

# ── Training loop ─────────────────────────────────────────────────────────────

def train(params, X_tr, X_te, Y_te):
    num_steps   = MAX_EPOCHS * (len(X_tr) // BATCH_SIZE)
    optimizer   = optax.adam(optax.cosine_decay_schedule(ADAM_LR, num_steps, alpha=0.05))
    opt_state   = optimizer.init(params)
    step_fn     = make_train_step(optimizer)
    key_gen     = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED + 1))
    num_batches = len(X_tr) // BATCH_SIZE
    history     = {
        "match_loss": [], "match_loss_eval": [], "bin_acc": [],
        "cluster_usage": [],
        "label_acc_hard": [],
        "label_acc_soft": [],
    }
    t0 = time.perf_counter()

    for epoch in range(MAX_EPOCHS):
        key  = next(key_gen).get()
        perm = np.array(jax.random.permutation(key, len(X_tr)))
        Xs   = np.array(X_tr)[perm]

        X_proto = decode_prototypes(params)

        all_assign = load_balanced_assign(Xs[:2048], X_proto)
        usage      = np.bincount(all_assign, minlength=N_CODES)
        n_active   = int((usage > 0).sum())
        history["cluster_usage"].append(usage.tolist())

        e_loss = e_match = 0.0
        for b in range(num_batches):
            x_batch = Xs[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
            idx     = load_balanced_assign(x_batch, X_proto)
            e_match += ((x_batch - X_proto[idx])**2).mean()
            matched = jnp.array(idx).reshape(-1, 1)
            params, opt_state, loss = step_fn(params, opt_state, jnp.array(x_batch), matched)
            e_loss += float(loss)

        e_loss /= num_batches; e_match /= num_batches
        history["match_loss"].append(e_match)

        eval_str = ""
        if (epoch + 1) % EVAL_EVERY == 0:
            ml_eval, bin_acc = eval_metrics(params, X_te)
            hard_acc, soft_acc, _ = label_accuracy(params, np.array(X_te), Y_te)
            history["match_loss_eval"].append((epoch, ml_eval))
            history["bin_acc"].append((epoch, bin_acc))
            history["label_acc_hard"].append((epoch, hard_acc))
            history["label_acc_soft"].append((epoch, soft_acc))
            eval_str = (f"  eval_match={ml_eval:.1f}  bin_acc={bin_acc:.1%}"
                        f"  lbl_hard={hard_acc:.1%}  lbl_bayes={soft_acc:.1%}")

        logging.info(f"epoch {epoch:3d}  nll={e_loss:.4f}  match_l2={e_match:.1f}"
                     f"  active={n_active}/{N_CODES}{eval_str}  {time.perf_counter()-t0:.1f}s")

    return params, history, time.perf_counter() - t0

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.info(f"JAX devices: {jax.devices()}")
    data    = load_supervised_image("mnist")
    X_train = data.X.reshape(data.n_samples,           784).astype(np.float32)
    X_test  = data.X_test.reshape(data.n_test_samples, 784).astype(np.float32)
    Y_test  = np.array(data.y_test).astype(np.int32)

    key_gen = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED))
    params  = init_params(key_gen)
    logging.info(f"Parameters: {n_params(params):,}  (flat clustering N_CODES={N_CODES})")

    params, history, elapsed = train(params, X_train, X_test, Y_test)

    hist_path = Path(__file__).parent / f"history_{EXP_NAME}.pkl"
    with open(hist_path, "wb") as f:
        pickle.dump(history, f)
    logging.info(f"History saved → {hist_path}")

    params_path = Path(__file__).parent / f"params_{EXP_NAME}.pkl"
    with open(params_path, "wb") as f:
        pickle.dump({k: np.array(v) for k, v in params.items()}, f)
    logging.info(f"Params saved → {params_path}")

    save_learning_curves(history, EXP_NAME, subtitle=f"flat clustering N_CODES={N_CODES} D_U={D_U}")

    final_ml, final_acc = eval_metrics(params, X_test)
    hard_acc, soft_acc, label_probs = label_accuracy(params, X_test, Y_test)

    np.save(Path(__file__).parent / f"label_probs_{EXP_NAME}.npy", label_probs)

    append_result({
        "experiment": EXP_NAME,
        "name": f"imle-cluster-deltasum-rep N_CODES={N_CODES} D_EMB={D_EMB} D_U={D_U}",
        "n_params": n_params(params), "epochs": MAX_EPOCHS, "time_s": round(elapsed, 1),
        "final_bin_acc": round(float(final_acc), 6), "final_match_l2": round(float(final_ml), 4),
        "final_label_acc_hard": round(hard_acc, 6), "final_label_acc_soft": round(soft_acc, 6),
        "n_codes": N_CODES, "d_emb": D_EMB, "d_u": D_U,
        "n_val_bins": N_VAL_BINS, "adam_lr": ADAM_LR,
    })
    logging.info(f"Done. match_l2={final_ml:.1f}  bin_acc={final_acc:.1%}"
                 f"  lbl_hard={hard_acc:.1%}  lbl_bayes={soft_acc:.1%}  {elapsed:.1f}s")

