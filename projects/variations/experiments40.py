"""
exp40 — Global IMLE + exclusive Voronoi + recon loss + 2nd-nearest target.

Same as exp39 except each variation targets its 2nd-nearest training image within
its Voronoi cell (instead of 1st-nearest). The hypothesis is that targeting the
2nd-nearest forces variations to overshoot slightly, potentially improving diversity
and coverage of the local neighbourhood recall metric.

Loss: L_imle + ALPHA * L_recon
  L_recon = MSE(decode(z, cats_zero), x)        — anchors encoder
  L_imle  = MSE(Y_var, 2nd-NN_in_cell(Y_var))  — 2nd nearest within Voronoi cell

Key difference from exp39:
- In exclusive_assign, target = argsort(masked)[:, 1]  (2nd nearest in cell)
  instead of argmin(masked)                             (1st nearest in cell)

Architecture identical to exp39/exp35 (LATENT=2048, wide-deconvnet, CNN encoder).

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run python projects/variations/experiments40.py
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
from lib.logging_utils import setup_logging

EXP_NAME = "exp40"

# ── Hyperparameters ────────────────────────────────────────────────────────────

LATENT_DIM    = 2048
N_CAT         = 4
VOCAB_SIZE    = 8
E_DIM         = 4
COND_DIM      = N_CAT * E_DIM   # 16
IN_DIM        = LATENT_DIM + COND_DIM  # 2064
BATCH_SIZE    = 64
N_VAR_SAMPLES = 64
MAX_EPOCHS    = 200
ADAM_LR       = 3e-4
ALPHA         = 1.0   # weight of reconstruction loss (L_imle + ALPHA * L_recon)
SEED          = 42

# Early stopping: halt when recall_mse hasn't improved by MIN_DELTA over PATIENCE evals
EARLY_STOP_PATIENCE = 8   # evals without improvement (= PATIENCE * EVAL_EVERY epochs)
EARLY_STOP_MIN_DELTA = 5e-4  # minimum absolute improvement to count

# CNN encoder channels (same as exp35)
ENC_C1        = 64
ENC_C2        = 128
ENC_C3        = 256
ENC_FC1       = 2048

# Wide decoder channels (same as exp35)
DC_STEM       = 1024
DC_C1         = 256
DC_C2         = 128
DC_C3         = 64

IMG_FLAT      = 3072
EVAL_EVERY    = 5
K_NN          = 16    # global nearest neighbours used in recall eval (same as exp35)
EVAL_RECALL_B = 8     # images per recall batch

JSONL = Path(__file__).parent / "results.jsonl"

_rng_eval = np.random.RandomState(0)
EVAL_COND_CATS = jnp.array(
    _rng_eval.randint(0, VOCAB_SIZE, (256, N_CAT)), dtype=jnp.int32
)  # 256 fixed eval codes — same count as exp35

# ── Utilities ──────────────────────────────────────────────────────────────────

def n_params(p):
    return sum(v.size for v in jax.tree_util.tree_leaves(p))

def append_result(row):
    with open(JSONL, "a") as f:
        f.write(json.dumps(row, default=lambda x: float(x) if hasattr(x, "__float__") else str(x)) + "\n")

# ── Parameters ────────────────────────────────────────────────────────────────

def init_params(key_gen):
    def n(shape, scale=0.02):
        return jax.random.normal(next(key_gen).get(), shape) * scale

    return {
        # Categorical embeddings (no recon_emb)
        "cat_emb": n((N_CAT, VOCAB_SIZE, E_DIM), scale=0.1),
        # CNN Encoder: HWIO format
        "enc_c1": n((3, 3,     3, ENC_C1), scale=(3*3*3    )**-0.5),
        "enc_b1": jnp.zeros(ENC_C1),
        "enc_c2": n((3, 3, ENC_C1, ENC_C2), scale=(3*3*ENC_C1)**-0.5),
        "enc_b2": jnp.zeros(ENC_C2),
        "enc_c3": n((3, 3, ENC_C2, ENC_C3), scale=(3*3*ENC_C2)**-0.5),
        "enc_b3": jnp.zeros(ENC_C3),
        # FC head: 4×4×256 → ENC_FC1 → LATENT_DIM
        "enc_W1": n((ENC_C3 * 4 * 4, ENC_FC1), scale=(ENC_C3*16)**-0.5),
        "enc_b4": jnp.zeros(ENC_FC1),
        "enc_W2": n((ENC_FC1, LATENT_DIM),     scale=ENC_FC1**-0.5),
        "enc_b5": jnp.zeros(LATENT_DIM),
        # Wide decoder MLP stem: IN_DIM → DC_STEM → DC_C1*4*4
        "dc_W1": n((IN_DIM,  DC_STEM),       scale=IN_DIM**-0.5),
        "dc_b1": jnp.zeros(DC_STEM),
        "dc_W2": n((DC_STEM, DC_C1 * 4*4),   scale=DC_STEM**-0.5),
        "dc_b2": jnp.zeros(DC_C1 * 4*4),
        # ConvTranspose weights: OIHW (C_out, C_in, kH, kW)
        "dc_ct1": n((DC_C1, DC_C2, 4, 4), scale=(DC_C1 * 4 * 4)**-0.5),
        "dc_ct2": n((DC_C2, DC_C3, 4, 4), scale=(DC_C2 * 4 * 4)**-0.5),
        "dc_ct3": n((DC_C3, 3, 4, 4),     scale=(DC_C3 * 4 * 4)**-0.5),
    }

# ── Model ─────────────────────────────────────────────────────────────────────

def cats_to_cond(params, cats):
    embs = params["cat_emb"][jnp.arange(N_CAT)[None, :], cats]
    return embs.reshape(cats.shape[0], COND_DIM)

_ENC_DIMS = ("NHWC", "HWIO", "NHWC")

def _conv_stride2(x, W, b):
    return jax.lax.conv_general_dilated(
        x, W, (2, 2), "SAME", dimension_numbers=_ENC_DIMS
    ) + b

def encode(params, x_flat):
    B = x_flat.shape[0]
    h = x_flat.reshape(B, 32, 32, 3)
    h = jax.nn.gelu(_conv_stride2(h, params["enc_c1"], params["enc_b1"]))
    h = jax.nn.gelu(_conv_stride2(h, params["enc_c2"], params["enc_b2"]))
    h = jax.nn.gelu(_conv_stride2(h, params["enc_c3"], params["enc_b3"]))
    h = h.reshape(B, -1)
    h = jax.nn.gelu(h @ params["enc_W1"] + params["enc_b4"])
    return h @ params["enc_W2"] + params["enc_b5"]

def decode(params, zc):
    B = zc.shape[0]
    h = jax.nn.gelu(zc @ params["dc_W1"] + params["dc_b1"])
    h = jax.nn.gelu(h  @ params["dc_W2"] + params["dc_b2"])
    h = h.reshape(B, DC_C1, 4, 4)
    h = jax.lax.conv_transpose(h, params["dc_ct1"], strides=(2,2), padding="SAME",
                                dimension_numbers=("NCHW","IOHW","NCHW"))
    h = jax.nn.gelu(h)
    h = jax.lax.conv_transpose(h, params["dc_ct2"], strides=(2,2), padding="SAME",
                                dimension_numbers=("NCHW","IOHW","NCHW"))
    h = jax.nn.gelu(h)
    h = jax.lax.conv_transpose(h, params["dc_ct3"], strides=(2,2), padding="SAME",
                                dimension_numbers=("NCHW","IOHW","NCHW"))
    return jax.nn.sigmoid(h.reshape(B, IMG_FLAT))

# ── Global NN search (JAX GPU) ────────────────────────────────────────────────

@jax.jit
def forward_and_find_nn(params, x_batch, cats, X_train_jax):
    """
    Forward-pass + exclusive global nearest-neighbour assignment, entirely on GPU.

    For each input image independently, assigns its N_VAR variations to N_VAR
    *distinct* training images (Voronoi / exclusive assignment):
      1. Compute (N_VAR × 50k) distances per image.
      2. For each training image, find its nearest variation (inverse direction).
      3. For each variation j, its target = the closest training image assigned to j.

    This prevents mode collapse: N_VAR=64 variations partition 50k training images
    into 64 Voronoi cells (~781 images each), so different codes cover different images.
    Each variation targets its 2nd-nearest image within its cell (not 1st-nearest).

    Peak VRAM: ~820MB for (N_VAR × 50k) per image, vmapped over B.
    Returns: targets (B*N_VAR, IMG_FLAT) float32 in [0,1], stop-gradient safe.
    """
    B    = x_batch.shape[0]
    x_nm = x_batch / 255.0
    z    = encode(params, x_nm)                                               # (B, LATENT)
    cond = cats_to_cond(params, cats)                                         # (N_VAR, COND_DIM)
    z_rep = jnp.repeat(z[:, None, :], N_VAR_SAMPLES, axis=1).reshape(B * N_VAR_SAMPLES, LATENT_DIM)
    c_rep = jnp.broadcast_to(cond[None], (B, N_VAR_SAMPLES, COND_DIM)).reshape(B * N_VAR_SAMPLES, COND_DIM)
    Y_01  = decode(params, jnp.concatenate([z_rep, c_rep], axis=-1))         # (B*N_VAR, IMG_FLAT)

    Y_per = (Y_01 * 255.0).reshape(B, N_VAR_SAMPLES, IMG_FLAT)               # (B, N_VAR, IMG_FLAT)
    sq_X  = (X_train_jax ** 2).sum(-1)                                        # (50k,)

    def exclusive_assign(y_vars):
        """Exclusive assignment for one image's N_VAR variations → N_VAR distinct targets.
        Each variation targets its 2nd-nearest training image within its Voronoi cell.
        """
        # (N_VAR × 50k) distance matrix
        sq_y  = (y_vars ** 2).sum(-1, keepdims=True)                          # (N_VAR, 1)
        dot   = y_vars @ X_train_jax.T                                         # (N_VAR, 50k)
        dists = sq_y + sq_X - 2.0 * dot                                        # (N_VAR, 50k)

        # Inverse: for each training image, which variation is nearest?
        best_var = jnp.argmin(dists, axis=0)                                   # (50k,) ← inverse

        # For each variation j, mask distances to only its Voronoi cell
        var_ids = jnp.arange(N_VAR_SAMPLES)                                    # (N_VAR,)
        mask    = (var_ids[:, None] == best_var[None, :])                      # (N_VAR, 50k) bool
        masked  = jnp.where(mask, dists, jnp.inf)                              # (N_VAR, 50k)

        # Target = 2nd nearest within cell — top_k on negated dists (O(n) heap vs O(n log n) sort)
        _, top2_idx = jax.lax.top_k(-masked, k=2)                              # (N_VAR, 2)
        nn_idx      = top2_idx[:, 1]                                            # (N_VAR,) 2nd nearest
        return X_train_jax[nn_idx] / 255.0                                     # (N_VAR, IMG_FLAT)

    targets = jax.vmap(exclusive_assign)(Y_per)                               # (B, N_VAR, IMG_FLAT)
    return targets.reshape(B * N_VAR_SAMPLES, IMG_FLAT)

def loss_fn(params, x_batch, cats, target_flat):
    """
    cats:        (N_VAR, N_CAT) int32
    target_flat: (B*N_VAR, IMG_FLAT) [0,1] — GPU NN-matched targets (stop-gradient)
    Returns: (total_loss, (L_recon, L_imle))
    """
    B     = x_batch.shape[0]
    x_nm  = x_batch / 255.0
    z     = encode(params, x_nm)

    # IMLE loss
    cond  = cats_to_cond(params, cats)
    z_rep = jnp.repeat(z[:, None, :], N_VAR_SAMPLES, axis=1).reshape(B * N_VAR_SAMPLES, LATENT_DIM)
    c_rep = jnp.broadcast_to(cond[None], (B, N_VAR_SAMPLES, COND_DIM)).reshape(B * N_VAR_SAMPLES, COND_DIM)
    Y_all = decode(params, jnp.concatenate([z_rep, c_rep], axis=-1))
    L_imle = ((Y_all - jax.lax.stop_gradient(target_flat)) ** 2).mean()

    # Reconstruction loss: decode(z, cats_zero) ≈ x
    cats_zero  = jnp.zeros((1, N_CAT), dtype=jnp.int32)
    recon_cond = jnp.broadcast_to(cats_to_cond(params, cats_zero), (B, COND_DIM))
    x_recon    = decode(params, jnp.concatenate([z, recon_cond], axis=-1))
    L_recon    = ((x_recon - x_nm) ** 2).mean()

    return L_imle + ALPHA * L_recon, (L_recon, L_imle)

def make_train_step(optimizer):
    @jax.jit
    def _train_step(params, opt_state, x_batch, cats, target_flat):
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, x_batch, cats, target_flat)
        updates, new_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), new_state, loss, aux
    return _train_step

# ── Evaluation ────────────────────────────────────────────────────────────────

@jax.jit
def _eval_recon_batch(params, x_batch):
    """Decode with zero-category cond; tracks recon quality (not trained on)."""
    x_nm      = x_batch / 255.0
    B         = x_batch.shape[0]
    z         = encode(params, x_nm)
    cats_zero = jnp.zeros((1, N_CAT), dtype=jnp.int32)
    cond      = jnp.broadcast_to(cats_to_cond(params, cats_zero), (B, COND_DIM))
    x_recon   = decode(params, jnp.concatenate([z, cond], axis=-1))
    return ((x_recon - x_nm) ** 2).mean()

def eval_recon_mse(params, X_te):
    mses = [float(_eval_recon_batch(params, jnp.array(X_te[i:i+256])))
            for i in range(0, len(X_te), 256)]
    return float(np.mean(mses))

@jax.jit
def _recall_batch_jit(params, x_norm, X_train_norm, all_conds):
    """
    Exact same metric as exp35: for each image find K global NN, generate N_SLOTS variations,
    assign best variation to each neighbour, report per-rank MSE.
    Global NN computed on-the-fly via GEMM (no precomputed knn_idx needed).
    """
    B       = x_norm.shape[0]
    N_SLOTS = all_conds.shape[0]

    # --- K global nearest neighbours via GEMM ---
    sq_x    = (x_norm ** 2).sum(-1, keepdims=True)           # (B, 1)
    sq_tr   = (X_train_norm ** 2).sum(-1)                     # (N_train,)
    dot_xt  = x_norm @ X_train_norm.T                         # (B, N_train)
    dists_x = sq_x + sq_tr - 2.0 * dot_xt                    # (B, N_train)
    knn_idx  = jnp.argsort(dists_x, axis=1)[:, :K_NN]        # (B, K_NN)
    knn_norm = X_train_norm[knn_idx]                          # (B, K_NN, IMG_FLAT)

    # --- Generate N_SLOTS variations ---
    z     = encode(params, x_norm)
    z_rep = jnp.repeat(z[:, None, :], N_SLOTS, axis=1).reshape(B * N_SLOTS, LATENT_DIM)
    c_rep = jnp.broadcast_to(all_conds[None], (B, N_SLOTS, COND_DIM)).reshape(B * N_SLOTS, COND_DIM)
    Y_all = decode(params, jnp.concatenate([z_rep, c_rep], axis=-1)).reshape(B, N_SLOTS, IMG_FLAT)

    # --- Best variation for each neighbour ---
    sq_knn   = (knn_norm ** 2).sum(-1)                        # (B, K_NN)
    sq_Y     = (Y_all ** 2).sum(-1)                           # (B, N_SLOTS)
    dot_kv   = jnp.einsum("bkd,bvd->bkv", knn_norm, Y_all)   # (B, K_NN, N_SLOTS)
    dists_kv = sq_knn[:, :, None] + sq_Y[:, None, :] - 2.0 * dot_kv  # (B, K_NN, N_SLOTS)
    best     = jnp.argmin(dists_kv, axis=-1)                  # (B, K_NN)
    matched  = Y_all[jnp.arange(B)[:, None], best]            # (B, K_NN, IMG_FLAT)

    # --- Sort neighbours by distance to x (rank order) ---
    sq_xi      = (x_norm ** 2).sum(-1)                        # (B,)
    dot_xk     = jnp.einsum("bd,bkd->bk", x_norm, knn_norm)  # (B, K_NN)
    dist_x_knn = sq_xi[:, None] + sq_knn - 2.0 * dot_xk      # (B, K_NN)
    rank_order = jnp.argsort(dist_x_knn, axis=-1)             # (B, K_NN)

    sorted_matched = matched[jnp.arange(B)[:, None], rank_order]
    sorted_knn     = knn_norm[jnp.arange(B)[:, None], rank_order]

    per_rank = ((sorted_matched - sorted_knn) ** 2).mean(axis=-1).mean(axis=0)  # (K_NN,)
    return per_rank.mean(), per_rank

def eval_recall_mse(params, X_train_jax, n_images=512):
    all_conds    = cats_to_cond(params, EVAL_COND_CATS)    # (256, COND_DIM)
    X_train_norm = X_train_jax / 255.0
    n            = (n_images // EVAL_RECALL_B) * EVAL_RECALL_B
    means, per_ranks = [], []
    for start in range(0, n, EVAL_RECALL_B):
        x_norm = X_train_norm[start:start + EVAL_RECALL_B]
        mean_r, rank_r = _recall_batch_jit(params, x_norm, X_train_norm, all_conds)
        means.append(float(mean_r))
        per_ranks.append(np.array(rank_r))
    return float(np.mean(means)), np.mean(per_ranks, axis=0).tolist()

# ── Training ──────────────────────────────────────────────────────────────────

def train(params, X_train_jax, X_te, X_tr_np):
    num_batches = X_tr_np.shape[0] // BATCH_SIZE   # 781
    num_steps   = MAX_EPOCHS * num_batches
    optimizer   = optax.adam(optax.cosine_decay_schedule(ADAM_LR, num_steps, alpha=0.05))
    opt_state   = optimizer.init(params)
    key_gen     = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED + 1))
    _train_step = make_train_step(optimizer)

    history = {
        "loss": [],
        "L_recon": [],
        "L_imle": [],
        "eval_recon_mse": [],
        "eval_recall_mse": [],
        "eval_recall_by_rank": [],
    }
    t0 = time.perf_counter()
    best_recall = float("inf")
    no_improve  = 0

    for epoch in range(MAX_EPOCHS):
        key  = next(key_gen).get()
        perm = np.array(jax.random.permutation(key, X_tr_np.shape[0]))

        epoch_losses, epoch_lrecon, epoch_limle = [], [], []
        for i in range(0, num_batches * BATCH_SIZE, BATCH_SIZE):
            x_batch  = X_train_jax[perm[i:i+BATCH_SIZE]]   # slice on GPU — no CPU transfer
            step_idx = i // BATCH_SIZE

            # Same cats for NN search and gradient update — correct global IMLE.
            # Each (z, cats) pair consistently maps to whichever training image it's closest to.
            ck   = jax.random.fold_in(key, step_idx)
            cats = jax.random.randint(ck, (N_VAR_SAMPLES, N_CAT), 0, VOCAB_SIZE)

            # 1. Forward + global GPU NN search → targets (all on GPU, no CPU)
            targets = forward_and_find_nn(params, x_batch, cats, X_train_jax)

            # 2. Gradient step with same cats against GPU-found targets
            params, opt_state, loss, (l_recon, l_imle) = _train_step(params, opt_state, x_batch, cats, targets)
            epoch_losses.append(float(loss))
            epoch_lrecon.append(float(l_recon))
            epoch_limle.append(float(l_imle))

        avg_loss   = float(np.mean(epoch_losses))
        avg_lrecon = float(np.mean(epoch_lrecon))
        avg_limle  = float(np.mean(epoch_limle))
        history["loss"].append(avg_loss)
        history["L_recon"].append(avg_lrecon)
        history["L_imle"].append(avg_limle)

        eval_str = ""
        if (epoch + 1) % EVAL_EVERY == 0:
            recon_mse          = eval_recon_mse(params, X_te)
            recall_mse, rank_r = eval_recall_mse(params, X_train_jax)
            history["eval_recon_mse"].append((epoch, recon_mse))
            history["eval_recall_mse"].append((epoch, recall_mse))
            history["eval_recall_by_rank"].append((epoch, rank_r))
            eval_str = f"  eval_recon={recon_mse:.5f}  recall_mse={recall_mse:.5f}"

            if recall_mse < best_recall - EARLY_STOP_MIN_DELTA:
                best_recall = recall_mse
                no_improve  = 0
            else:
                no_improve += 1
                if no_improve >= EARLY_STOP_PATIENCE:
                    logging.info(
                        f"Early stop: recall_mse flat for {no_improve} evals "
                        f"(best={best_recall:.5f})  {time.perf_counter()-t0:.1f}s"
                    )
                    logging.info(
                        f"epoch {epoch:3d}  loss={avg_loss:.5f}  L_recon={avg_lrecon:.5f}  L_imle={avg_limle:.5f}"
                        f"{eval_str}  {time.perf_counter()-t0:.1f}s"
                    )
                    return params, history, time.perf_counter() - t0

        logging.info(
            f"epoch {epoch:3d}  loss={avg_loss:.5f}  L_recon={avg_lrecon:.5f}  L_imle={avg_limle:.5f}"
            f"{eval_str}  {time.perf_counter()-t0:.1f}s"
        )

    return params, history, time.perf_counter() - t0

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    setup_logging(EXP_NAME)
    done = set()
    if JSONL.exists():
        for line in JSONL.read_text().strip().splitlines():
            try: done.add(json.loads(line).get("experiment"))
            except Exception: pass
    if EXP_NAME in done:
        logging.info(f"{EXP_NAME} already done — skipping"); exit(0)

    logging.info(f"JAX devices: {jax.devices()}")
    data    = load_supervised_image("cifar10")
    X_train_np  = np.array(data.X).reshape(data.n_samples,           IMG_FLAT).astype(np.float32)
    X_test_np   = np.array(data.X_test).reshape(data.n_test_samples, IMG_FLAT).astype(np.float32)

    # Move full training set to GPU — 614MB, stays there as a constant
    logging.info(f"Loading X_train to GPU: {X_train_np.shape} ({X_train_np.nbytes/1e6:.0f}MB)…")
    X_train_jax = jnp.array(X_train_np)
    logging.info(f"X_train on GPU. Peak VRAM during NN GEMM: ~{BATCH_SIZE*N_VAR_SAMPLES*data.n_samples*4/1e6:.0f}MB distance matrix")

    key_gen = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED))
    params  = init_params(key_gen)
    logging.info(
        f"Parameters: {n_params(params):,}  "
        f"LATENT={LATENT_DIM}  N_CAT={N_CAT}  VOCAB={VOCAB_SIZE}  E_DIM={E_DIM}"
        f"  COND_DIM={COND_DIM}  IN_DIM={IN_DIM}"
        f"  N_VAR={N_VAR_SAMPLES}  B={BATCH_SIZE}"
        f"  nn_search=exact-GPU-GEMM({BATCH_SIZE*N_VAR_SAMPLES}×{data.n_samples}×{IMG_FLAT})"
        f"  encoder=CNN  decoder=wide-deconvnet"
        f"  imle=global-jax-gpu  dataset=cifar10  MAX_EPOCHS={MAX_EPOCHS}"
    )

    params, history, elapsed = train(params, X_train_jax, X_test_np, X_train_np)

    with open(Path(__file__).parent / f"params_{EXP_NAME}.pkl", "wb") as f:
        pickle.dump({k: np.array(v) for k, v in params.items()}, f)
    with open(Path(__file__).parent / f"history_{EXP_NAME}.pkl", "wb") as f:
        pickle.dump(history, f)

    final_recon_mse              = eval_recon_mse(params, X_test_np)
    final_recall_mse, final_ranks = eval_recall_mse(params, X_train_jax)
    append_result({
        "experiment":       EXP_NAME,
        "name":             f"Global IMLE CIFAR-10 LATENT=2048 N_VAR=64 B=64 200ep exclusive-Voronoi-2ndNN+recon",
        "n_params":         n_params(params),
        "epochs":           MAX_EPOCHS,
        "time_s":           round(elapsed, 1),
        "final_recon_mse":  round(float(final_recon_mse),  6),
        "final_recall_mse": round(float(final_recall_mse), 6),
        "final_recall_by_rank": [round(v, 6) for v in final_ranks],
        "loss_curve":              [round(v, 6) for v in history["loss"]],
        "L_recon_curve":           [round(v, 6) for v in history["L_recon"]],
        "L_imle_curve":            [round(v, 6) for v in history["L_imle"]],
        "eval_recon_mse_curve":    history["eval_recon_mse"],
        "eval_recall_mse_curve":      history["eval_recall_mse"],
        "eval_recall_by_rank_curve":  history["eval_recall_by_rank"],
        "latent_dim":    LATENT_DIM,
        "n_cat":         N_CAT,
        "vocab_size":    VOCAB_SIZE,
        "e_dim":         E_DIM,
        "cond_dim":      COND_DIM,
        "n_var_samples": N_VAR_SAMPLES,
        "enc_c1": ENC_C1, "enc_c2": ENC_C2, "enc_c3": ENC_C3, "enc_fc1": ENC_FC1,
        "dc_c1":  DC_C1,  "dc_c2":  DC_C2,  "dc_c3":  DC_C3,  "dc_stem": DC_STEM,
        "adam_lr":    ADAM_LR,
        "alpha":      ALPHA,
        "batch_size": BATCH_SIZE,
        "encoder":    "cnn",
        "decoder":    "wide-deconvnet",
        "imle":       "global-jax-gpu-exclusive-voronoi-2ndNN+recon",
        "dataset":    "cifar10",
        "img_flat":   IMG_FLAT,
    })
    logging.info(f"Done. recon_mse={final_recon_mse:.5f}  recall_mse={final_recall_mse:.5f}  {elapsed:.1f}s")
