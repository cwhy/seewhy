"""
exp24 — Gram VAE: lax.scan over batch loop (code optimization).

Changes from exp19 (MLP_HIDDEN=256, DEC_RANK=16, LATENT_DIM=64, D=144):
  1. Replace Python `for b in range(num_batches)` with `jax.lax.scan`.
     - Exp19: 100 epochs × 468 batches = 46,800 Python→JAX dispatch calls.
     - Exp24: 100 Python iterations; each dispatches one fused XLA computation
       covering all 468 batches via scan. XLA can pipeline/overlap ops.
  2. Everything else (architecture, LR, schedule) identical to exp19.

Goal: faster wall-clock time by removing Python loop overhead.

Usage:
    uv run python projects/vae-gram/experiments24.py
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

EXP_NAME = "exp24"

# ── Hyperparameters ────────────────────────────────────────────────────────────

N_VAL_BINS = 32
BATCH_SIZE = 128
SEED       = 42
MAX_EPOCHS = 100

D_R, D_C, D_V = 6, 6, 4
D    = D_R * D_C * D_V   # 144
D_RC = D_R * D_C         # 36

N_ENC_Q          = 8
LATENT_DIM        = 64
DEC_RANK          = 16
MLP_HIDDEN        = 256

ADAM_LR          = 3e-4
KL_WEIGHT        = 1.0
KL_WARMUP_EPOCHS = 20
FREE_BITS        = 0.5

EVAL_EVERY = 5
VIZ_EVERY  = 20

JSONL = Path(__file__).parent / "results.jsonl"
ROWS = jnp.arange(784) // 28
COLS = jnp.arange(784) % 28
BIN_CENTERS = (jnp.arange(N_VAL_BINS) + 0.5) / N_VAL_BINS * 255.0

# ── Utilities ─────────────────────────────────────────────────────────────────

def layer_norm(x, gamma, beta, eps=1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var  = jnp.var(x,  axis=-1, keepdims=True)
    return gamma * (x - mean) / jnp.sqrt(var + eps) + beta

def quantize(x):
    return jnp.clip((x / 255.0 * N_VAL_BINS).astype(jnp.int32), 0, N_VAL_BINS - 1)

def n_params(p):
    return sum(v.size for v in jax.tree_util.tree_leaves(p))

def append_result(row):
    with open(JSONL, "a") as f:
        f.write(json.dumps(row) + "\n")

# ── Parameters ────────────────────────────────────────────────────────────────

def init_params(key_gen):
    def n(shape, scale=0.02):
        return jax.random.normal(next(key_gen).get(), shape) * scale
    feat_dim = N_ENC_Q * D
    return {
        "row_emb": n((28, D_R)), "col_emb": n((28, D_C)), "val_emb": n((N_VAL_BINS, D_V)),
        "h_enc": n((N_ENC_Q, D)),
        "ln_enc_gamma": jnp.ones(feat_dim), "ln_enc_beta": jnp.zeros(feat_dim),
        "W_mu": n((feat_dim, LATENT_DIM), scale=feat_dim**-0.5), "b_mu": jnp.zeros(LATENT_DIM),
        "W_lv": n((feat_dim, LATENT_DIM), scale=feat_dim**-0.5), "b_lv": jnp.zeros(LATENT_DIM),
        "W_mlp1": n((LATENT_DIM, MLP_HIDDEN), scale=LATENT_DIM**-0.5), "b_mlp1": jnp.zeros(MLP_HIDDEN),
        "W_A": n((MLP_HIDDEN, DEC_RANK * D), scale=MLP_HIDDEN**-0.5),
        "W_B": n((MLP_HIDDEN, DEC_RANK * D), scale=MLP_HIDDEN**-0.5),
        "dec_row_emb": n((28, D_R), scale=D_R**-0.5), "dec_col_emb": n((28, D_C), scale=D_C**-0.5),
        "W_pos_dec": n((D, D_RC), scale=D_RC**-0.5),
        "W_val1": n((2*D, D), scale=D**-0.5), "b_val1": jnp.zeros(2*D),
        "W_val2": n((N_VAL_BINS, 2*D), scale=(2*D)**-0.5),
    }

# ── Encoder ───────────────────────────────────────────────────────────────────

def encode(params, x):
    B = x.shape[0]
    bins = quantize(x)
    def embed_pixel(r, c, v_bin):
        re = params["row_emb"][r]; ce = params["col_emb"][c]; ve = params["val_emb"][v_bin]
        pos = (re[:, None] * ce[None, :]).reshape(D_RC)
        return (pos[:, None] * ve[None, :]).reshape(D)
    def embed_image(bins_i):
        return jax.vmap(embed_pixel, in_axes=(0,0,0))(ROWS, COLS, bins_i)
    emb = jax.vmap(embed_image)(bins)
    M   = jnp.einsum("bti,btj->bij", emb, emb)
    queried = jnp.einsum("bij,qj->bqi", M, params["h_enc"])
    feat = queried.reshape(B, N_ENC_Q * D)
    feat = layer_norm(feat, params["ln_enc_gamma"], params["ln_enc_beta"])
    mu      = feat @ params["W_mu"] + params["b_mu"]
    log_var = feat @ params["W_lv"] + params["b_lv"]
    return mu, log_var

# ── Decoder ───────────────────────────────────────────────────────────────────

def decode(params, z):
    B = z.shape[0]
    h     = jax.nn.gelu(z @ params["W_mlp1"] + params["b_mlp1"])
    A     = (h @ params["W_A"]).reshape(B, DEC_RANK, D)
    B_fac = (h @ params["W_B"]).reshape(B, DEC_RANK, D)
    gram  = jnp.einsum("bri,brj->bij", A, B_fac)
    def decode_pixel(r, c, g):
        re = params["dec_row_emb"][r]; ce = params["dec_col_emb"][c]
        pos_rc = (re[:, None] * ce[None, :]).reshape(D_RC)
        q = params["W_pos_dec"] @ pos_rc; f = g @ q
        h1 = jax.nn.gelu(params["W_val1"] @ f + params["b_val1"])
        return params["W_val2"] @ h1
    def decode_image(g):
        return jax.vmap(decode_pixel, in_axes=(0,0,None))(ROWS, COLS, g)
    return jax.vmap(decode_image)(gram)

# ── Loss ──────────────────────────────────────────────────────────────────────

def vae_loss(params, x, z_key, beta):
    mu, log_var = encode(params, x)
    z = mu + jax.random.normal(z_key, mu.shape) * jnp.exp(0.5 * log_var)
    logits = decode(params, z)
    bins   = quantize(x)
    log_p      = jax.nn.log_softmax(logits, axis=-1)
    recon_loss = -jnp.take_along_axis(log_p, bins[..., None], axis=-1)[..., 0].mean()
    kl_per_dim = -0.5 * (1.0 + log_var - mu**2 - jnp.exp(log_var))
    kl_loss    = jnp.maximum(kl_per_dim, FREE_BITS).mean()
    return recon_loss + beta * kl_loss, (recon_loss, kl_loss)

# ── Epoch step via lax.scan ───────────────────────────────────────────────────

def make_epoch_fn(optimizer):
    def scan_body(carry, inputs):
        params, opt_state, beta = carry
        batch, z_key = inputs
        (loss, aux), grads = jax.value_and_grad(vae_loss, has_aux=True)(params, batch, z_key, beta)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return (new_params, new_opt_state, carry[2]), (loss, aux[0], aux[1])

    # beta changes per epoch; pass as static via closure trick: rebuild per epoch
    def epoch_fn(params, opt_state, Xs_batched, epoch_key, beta):
        """Run all batches for one epoch via lax.scan. Returns updated params/state and mean stats."""
        num_batches = Xs_batched.shape[0]
        z_keys = jax.vmap(lambda b: jax.random.fold_in(epoch_key, b))(jnp.arange(num_batches))
        # Pack beta into carry so scan_body can read it
        (params, opt_state, _), (losses, recons, kls) = jax.lax.scan(
            scan_body, (params, opt_state, beta), (Xs_batched, z_keys)
        )
        return params, opt_state, losses.mean(), recons.mean(), kls.mean()

    return jax.jit(epoch_fn)

# ── Evaluation ────────────────────────────────────────────────────────────────

@jax.jit
def _bin_accuracy_batch(params, x):
    mu, _ = encode(params, x)
    logits = decode(params, mu)
    return jnp.sum(jnp.argmax(logits, axis=-1) == quantize(x)), quantize(x).size

def bin_accuracy(params, X, batch_size=256):
    correct = total = 0
    for i in range(0, len(X), batch_size):
        c, t = _bin_accuracy_batch(params, X[i:i+batch_size])
        correct += int(c); total += int(t)
    return correct / total

# ── Training loop ─────────────────────────────────────────────────────────────

def train(params, X_tr, X_te, y_te):
    num_steps = MAX_EPOCHS * (len(X_tr) // BATCH_SIZE)
    optimizer = optax.adam(optax.cosine_decay_schedule(ADAM_LR, num_steps, alpha=0.05))
    opt_state = optimizer.init(params)
    epoch_fn  = make_epoch_fn(optimizer)
    key_gen   = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED + 1))
    num_batches = len(X_tr) // BATCH_SIZE
    history = {"recon": [], "kl": [], "total": [], "bin_acc": []}
    t0 = time.perf_counter()
    for epoch in range(MAX_EPOCHS):
        beta = min(1.0, epoch / max(KL_WARMUP_EPOCHS, 1)) * KL_WEIGHT
        key  = next(key_gen).get()
        # Permute and pre-batch the data into (num_batches, BATCH_SIZE, 784)
        Xs = X_tr[jax.random.permutation(key, len(X_tr))]
        Xs_batched = Xs[:num_batches * BATCH_SIZE].reshape(num_batches, BATCH_SIZE, -1)
        params, opt_state, e_total, e_recon, e_kl = epoch_fn(params, opt_state, Xs_batched, key, beta)
        e_total = float(e_total); e_recon = float(e_recon); e_kl = float(e_kl)
        history["recon"].append(e_recon); history["kl"].append(e_kl); history["total"].append(e_total)
        acc_str = ""
        if (epoch + 1) % EVAL_EVERY == 0:
            acc = bin_accuracy(params, X_te); history["bin_acc"].append((epoch, acc)); acc_str = f"  acc={acc:.1%}"
        logging.info(f"epoch {epoch:3d}  β={beta:.2f}  recon={e_recon:.4f}  kl={e_kl:.4f}  total={e_total:.4f}{acc_str}  {time.perf_counter()-t0:.1f}s")
        if (epoch + 1) % VIZ_EVERY == 0:
            save_reconstruction_grid(params, X_te, encode, decode, BIN_CENTERS, EXP_NAME, f"epoch{epoch+1:03d}")
    return params, history, time.perf_counter() - t0

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.info(f"JAX devices: {jax.devices()}")
    data    = load_supervised_image("mnist")
    X_train = data.X.reshape(data.n_samples,           784).astype(jnp.float32)
    X_test  = data.X_test.reshape(data.n_test_samples, 784).astype(jnp.float32)
    y_test  = data.y_test
    key_gen = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED))
    params  = init_params(key_gen)
    logging.info(f"Parameters: {n_params(params):,}")
    params, history, elapsed = train(params, X_train, X_test, y_test)
    with open(Path(__file__).parent / f"params_{EXP_NAME}.pkl", "wb") as f:
        pickle.dump({k: np.array(v) for k, v in params.items()}, f)
    save_learning_curves(history, FREE_BITS, EXP_NAME, subtitle=f"lax.scan batch loop MLP_HIDDEN={MLP_HIDDEN} DEC_RANK={DEC_RANK}")
    save_reconstruction_grid(params, X_test, encode, decode, BIN_CENTERS, EXP_NAME, "final")
    save_sample_grid(params, decode, LATENT_DIM, BIN_CENTERS, EXP_NAME, rows=5, cols=8)
    final_acc = bin_accuracy(params, X_test)
    append_result({
        "experiment": EXP_NAME, "name": f"gram-vae lax.scan MLP_HIDDEN={MLP_HIDDEN} DEC_RANK={DEC_RANK}",
        "n_params": n_params(params), "epochs": MAX_EPOCHS, "time_s": round(elapsed, 1),
        "final_bin_acc": round(final_acc, 6), "final_recon_nll": round(history["recon"][-1], 6),
        "final_kl": round(history["kl"][-1], 6), "final_total": round(history["total"][-1], 6),
        "recon_curve": [round(v, 6) for v in history["recon"]],
        "kl_curve":    [round(v, 6) for v in history["kl"]],
        "bin_acc_curve": history["bin_acc"],
        "d_r": D_R, "d_c": D_C, "d_v": D_V, "n_enc_q": N_ENC_Q,
        "latent_dim": LATENT_DIM, "dec_rank": DEC_RANK, "mlp_hidden": MLP_HIDDEN,
        "n_val_bins": N_VAL_BINS, "adam_lr": ADAM_LR, "kl_warmup_epochs": KL_WARMUP_EPOCHS,
        "free_bits": FREE_BITS, "batch_size": BATCH_SIZE,
    })
    logging.info(f"Done. bin_acc={final_acc:.1%}  recon={history['recon'][-1]:.4f}  kl={history['kl'][-1]:.4f}  {elapsed:.1f}s")
