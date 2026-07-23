"""
exp_vae_gram_dual — Gram-dual encoder + VAE, LATENT_DIM=1024.

Training: Variational autoencoder, beta-VAE (BETA=0.0001), encode returns mu at eval.

Usage:
    uv run python projects/ssl/exp_vae_gram_dual.py
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

EXP_NAME = "exp_vae_kl_gram_dual_b1e4"

# ── Hyperparameters ────────────────────────────────────────────────────────────

BATCH_SIZE   = 256
SEED         = 42
MAX_EPOCHS   = 200
ADAM_LR      = 3e-4
EVAL_EVERY   = 10
PROBE_EPOCHS = 10

LATENT_DIM   = 1024
DEC_HIDDEN   = 512

NOISE_STD    = 75.0
MASK_RATIOS  = [0.25, 0.50, 0.75]

JSONL = Path(__file__).parent / "results_vae_kl.jsonl"

# ── Utilities ─────────────────────────────────────────────────────────────────

def n_params(p):
    return sum(v.size for v in jax.tree_util.tree_leaves(p))

def append_result(row):
    with open(JSONL, "a") as f:
        f.write(json.dumps(row, default=lambda x: float(x) if hasattr(x, "__float__") else str(x)) + "\n")

# ── Architecture constants ─────────────────────────────────────────────────────

D_R, D_C, D_V = 6, 6, 4
D              = D_R * D_C * D_V   # 144
D_RC           = D_R * D_C         # 36
DEC_RANK       = 64                # DEC_RANK^2 = 4096 -> projected to LATENT_DIM=1024

ROWS = jnp.arange(784) // 28
COLS = jnp.arange(784) % 28

# ── Parameters ────────────────────────────────────────────────────────────────

def init_params(key_gen):
    def n(shape, scale=0.02):
        return jax.random.normal(next(key_gen).get(), shape) * scale
    return {
        "enc_row_emb" : n((28, D_R),                         scale=D_R**-0.5),
        "enc_col_emb" : n((28, D_C),                         scale=D_C**-0.5),
        "W_pos_enc"   : n((D_RC, D),                         scale=D_RC**-0.5),
        "W_enc"       : n((D, DEC_RANK),                     scale=D**-0.5),
        "W_enc_A"     : n((DEC_RANK**2, LATENT_DIM),         scale=(DEC_RANK**2)**-0.5),
        "W_enc_B"     : n((DEC_RANK**2, LATENT_DIM),         scale=(DEC_RANK**2)**-0.5),
        # ── Decoder ───────────────────────────────────────────────────────────
        "W_dec1"      : n((LATENT_DIM, DEC_HIDDEN),   scale=LATENT_DIM**-0.5),
        "b_dec1"      : jnp.zeros(DEC_HIDDEN),
        "W_dec2"      : n((DEC_HIDDEN, 784),           scale=DEC_HIDDEN**-0.5),
        "b_dec2"      : jnp.zeros(784),
        # ── VAE moments projection ─────────────────────────────────────────────
        "W_vae"       : jnp.zeros((LATENT_DIM, 2 * LATENT_DIM)),
        "b_vae"       : jnp.zeros(2 * LATENT_DIM),
    }

def encode(params, x):
    """Gram-dual (Hadamard readout).  x: (B, 784) -> (B, LATENT_DIM)."""
    x_norm = x / 255.0
    re       = params["enc_row_emb"][ROWS]
    ce       = params["enc_col_emb"][COLS]
    pos_rc   = jax.vmap(lambda r, c: (r[:, None] * c[None, :]).reshape(D_RC))(re, ce)
    pos_feat = pos_rc @ params["W_pos_enc"]
    f        = pos_feat @ params["W_enc"]
    F = x_norm[:, :, None] * f[None, :, :]
    H = jnp.einsum("bpi,bpj->bij", F, F).reshape(x.shape[0], DEC_RANK ** 2)
    a = H @ params["W_enc_A"]
    b = H @ params["W_enc_B"]
    return a * b

# ── Decoder ───────────────────────────────────────────────────────────────────

def decode(params, z):
    """z: (B, LATENT_DIM) -> (B, 784) in [0,1] space."""
    h = jax.nn.gelu(z @ params["W_dec1"] + params["b_dec1"])
    return h @ params["W_dec2"] + params["b_dec2"]

# ── VAE encode ────────────────────────────────────────────────────────────────

BETA = 0.0001

def encode_base(params, x):
    """Encoder without VAE projection.  x: (B, 784) -> (B, LATENT_DIM)."""
    return encode(params, x)

# ── Loss ──────────────────────────────────────────────────────────────────────

def total_loss(params, x, key):
    """x: (B, 784).  Returns (loss, (recon_loss, kl_loss))."""
    x_norm  = x / 255.0
    h       = encode_base(params, x)
    moments = h @ params["W_vae"] + params["b_vae"]
    mu, lv  = moments[:, :LATENT_DIM], moments[:, LATENT_DIM:]
    lv      = jnp.clip(lv, -10.0, 10.0)
    eps     = jax.random.normal(key, mu.shape)
    z       = mu + eps * jnp.exp(0.5 * lv)
    recon_loss = ((decode(params, z) - x_norm) ** 2).mean()
    kl_loss    = -0.5 * (1 + lv - mu ** 2 - jnp.exp(lv)).mean()
    loss       = recon_loss + BETA * kl_loss
    return loss, (recon_loss, kl_loss)

# ── Epoch function ────────────────────────────────────────────────────────────

def make_epoch_fn(optimizer):
    def scan_body(carry, inputs):
        params, opt_state = carry
        x_batch, key = inputs
        (loss, aux), grads = jax.value_and_grad(total_loss, has_aux=True)(params, x_batch, key)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        return (optax.apply_updates(params, updates), new_opt_state), (loss, *aux)

    def epoch_fn(params, opt_state, Xs_batched, epoch_key):
        num_batches = Xs_batched.shape[0]
        keys = jax.vmap(lambda b: jax.random.fold_in(epoch_key, b))(jnp.arange(num_batches))
        (params, opt_state), stats = jax.lax.scan(
            scan_body, (params, opt_state), (Xs_batched, keys)
        )
        return params, opt_state, *[s.mean() for s in stats]

    return jax.jit(epoch_fn)

# ── Linear probe ──────────────────────────────────────────────────────────────

def linear_probe_accuracy(params, X_tr, Y_tr, X_te, Y_te):
    def encode_all(X):
        parts = []
        for i in range(0, len(X), 512):
            parts.append(np.array(encode(params, jnp.array(np.array(X[i:i+512])))))
        return np.concatenate(parts, 0)

    E_tr = encode_all(X_tr)
    E_te = encode_all(X_te)
    n_tr = (len(E_tr) // BATCH_SIZE) * BATCH_SIZE
    E_b  = jnp.array(E_tr[:n_tr].reshape(-1, BATCH_SIZE, LATENT_DIM))
    Y_b  = jnp.array(np.array(Y_tr[:n_tr]).reshape(-1, BATCH_SIZE))

    W, b        = jnp.zeros((LATENT_DIM, 10)), jnp.zeros(10)
    probe_opt   = optax.adam(1e-3)
    probe_state = probe_opt.init((W, b))

    def probe_step(carry, inputs):
        (W, b), state = carry
        e, y = inputs
        def ce(wb): return optax.softmax_cross_entropy_with_integer_labels(e @ wb[0] + wb[1], y).mean()
        loss, grads = jax.value_and_grad(ce)((W, b))
        updates, new_state = probe_opt.update(grads, state, (W, b))
        return (optax.apply_updates((W, b), updates), new_state), loss

    @jax.jit
    def probe_epoch(W, b, state):
        (W, b), state = jax.lax.scan(probe_step, ((W, b), state), (E_b, Y_b))[0]
        return W, b, state

    for _ in range(PROBE_EPOCHS):
        W, b, probe_state = probe_epoch(W, b, probe_state)

    logits = jnp.array(E_te) @ W + b
    return float((jnp.argmax(logits, 1) == jnp.array(np.array(Y_te))).mean())

# ── Training loop ─────────────────────────────────────────────────────────────

def train(params, X_tr, Y_tr, X_te, Y_te):
    num_steps   = MAX_EPOCHS * (len(X_tr) // BATCH_SIZE)
    optimizer   = optax.adam(optax.cosine_decay_schedule(ADAM_LR, num_steps, alpha=0.05))
    opt_state   = optimizer.init(params)
    epoch_fn    = make_epoch_fn(optimizer)
    key_gen     = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED + 1))
    num_batches = len(X_tr) // BATCH_SIZE
    history     = {"loss": [], "recon_loss": [], "kl_loss": [], "probe_acc": []}
    t0          = time.perf_counter()

    for epoch in range(MAX_EPOCHS):
        key  = next(key_gen).get()
        perm = np.array(jax.random.permutation(key, len(X_tr)))
        Xs   = jnp.array(np.array(X_tr)[perm][:num_batches * BATCH_SIZE].reshape(num_batches, BATCH_SIZE, 784))
        params, opt_state, loss, recon_loss, kl_loss = epoch_fn(params, opt_state, Xs, jax.random.fold_in(key, epoch))
        loss, recon_loss, kl_loss = float(loss), float(recon_loss), float(kl_loss)
        history["loss"].append(loss)
        history["recon_loss"].append(recon_loss)
        history["kl_loss"].append(kl_loss)

        probe_str = ""
        if (epoch + 1) % EVAL_EVERY == 0:
            acc = linear_probe_accuracy(params, X_tr, Y_tr, X_te, Y_te)
            history["probe_acc"].append((epoch, acc))
            probe_str = f"  probe_acc={acc:.1%}"

        logging.info(f"epoch {epoch:3d}  loss={loss:.4f}  recon={recon_loss:.4f}  kl={kl_loss:.4f}{probe_str}  {time.perf_counter()-t0:.1f}s")

    return params, history, time.perf_counter() - t0

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.info(f"JAX devices: {jax.devices()}")
    logging.info(f"BETA = {BETA}  arch = 'Gram-dual'")

    # Skip if already done
    done_exps = set()
    if JSONL.exists():
        with open(JSONL) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        done_exps.add(json.loads(line).get("experiment"))
                    except Exception:
                        pass
    if EXP_NAME in done_exps:
        logging.info(f"{EXP_NAME} already in results_vae_kl.jsonl — skipping")
        exit(0)

    mnist   = load_supervised_image("mnist")
    fmnist  = load_supervised_image("fashion_mnist")
    X_tr    = mnist.X.reshape(mnist.n_samples,           784).astype(np.float32)
    X_te    = mnist.X_test.reshape(mnist.n_test_samples, 784).astype(np.float32)
    Y_tr    = np.array(mnist.y)
    Y_te    = np.array(mnist.y_test)
    X_tr_f  = fmnist.X.reshape(fmnist.n_samples,           784).astype(np.float32)
    X_te_f  = fmnist.X_test.reshape(fmnist.n_test_samples, 784).astype(np.float32)
    Y_tr_f  = np.array(fmnist.y)
    Y_te_f  = np.array(fmnist.y_test)

    key_gen = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED))
    params  = init_params(key_gen)
    logging.info(f"Parameters: {n_params(params):,}")

    params, history, elapsed = train(params, X_tr, Y_tr, X_te, Y_te)

    with open(Path(__file__).parent / f"params_{EXP_NAME}.pkl", "wb") as f:
        pickle.dump({k: np.array(v) for k, v in params.items()}, f)

    probe_mnist  = linear_probe_accuracy(params, X_tr,   Y_tr,   X_te,   Y_te)
    probe_fmnist = linear_probe_accuracy(params, X_tr_f, Y_tr_f, X_te_f, Y_te_f)

    append_result({
        "experiment"       : EXP_NAME,
        "arch"             : 'Gram-dual',
        "beta"             : BETA,
        "probe_acc_mnist"  : round(probe_mnist,  6),
        "probe_acc_fmnist" : round(probe_fmnist, 6),
        "time_s"           : round(elapsed, 1),
    })
    logging.info(f"Done. probe_mnist={probe_mnist:.1%}  probe_fmnist={probe_fmnist:.1%}  {elapsed:.1f}s")
