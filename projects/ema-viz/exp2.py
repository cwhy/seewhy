"""
exp2 — Linear autodecoder: how well can small frozen SSL embeddings reconstruct digits?

Architecture
------------
  z    = encode(x)          frozen MLP-shallow encoder, (B, LATENT_DIM)
  x̂    = sigmoid(z @ P.T)   learned, P ∈ R^(784, LATENT_DIM)

No bottleneck projection — P is directly in the encoder's native space.
Reconstruction quality as a function of LATENT_DIM ∈ {16, 32, 64, 128}
reveals how much usable image information each embedding size captures.

Encoders loaded from projects/ssl/params_exp_mlp_ema_{dim}.pkl (trained with
EMA-JEPA, 200 epochs, same augmentations as the full pipeline).

Usage:
    uv run python projects/ema-viz/exp2.py
"""

import json, time, sys, pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logging

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared_lib.datasets import load_supervised_image
from shared_lib.media import save_matplotlib_figure

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

EXP_NAME = "exp2"

# ── Hyperparameters ────────────────────────────────────────────────────────────

BATCH_SIZE  = 512
MAX_EPOCHS  = 200
ADAM_LR     = 3e-3
SEED        = 42

LATENT_DIMS = [16, 32, 64, 128]

SSL_DIR = Path(__file__).parent.parent / "ssl"
JSONL   = Path(__file__).parent / "results.jsonl"

# ── Utilities ─────────────────────────────────────────────────────────────────

def append_result(row):
    with open(JSONL, "a") as f:
        f.write(json.dumps(row, default=lambda x: float(x) if hasattr(x, "__float__") else str(x)) + "\n")

# ── MLP-shallow encoder (matches ssl/exp_mlp_ema_*.py) ───────────────────────

MLP_ENC_HIDDEN = 512

def load_encoder(latent_dim):
    pkl = SSL_DIR / f"params_exp_mlp_ema_{latent_dim}.pkl"
    with open(pkl, "rb") as f:
        raw = pickle.load(f)
    return {k: jnp.array(v) for k, v in raw.items()}

@jax.jit
def encode_batch(enc_params, x):
    """MLP-shallow encode.  x: (B, 784) → (B, LATENT_DIM). Frozen."""
    h = jax.nn.gelu(x / 255.0 @ enc_params["W_enc1"] + enc_params["b_enc1"])
    return h @ enc_params["W_enc2"] + enc_params["b_enc2"]

def encode_dataset(enc_params, X, batch_size=512):
    chunks = []
    for i in range(0, X.shape[0], batch_size):
        xb = jnp.array(X[i:i+batch_size].astype(np.float32))
        chunks.append(np.array(encode_batch(enc_params, xb)))
    return np.concatenate(chunks, axis=0).astype(np.float32)

# ── Decoder ───────────────────────────────────────────────────────────────────

def init_decoder(latent_dim, key):
    k1, k2 = jax.random.split(key)
    return {
        "P": jax.random.normal(k1, (784, latent_dim)) * latent_dim ** -0.5,
        "b": jnp.zeros(784),   # per-pixel bias — can go negative for dark backgrounds
    }

def decode(dec_params, z):
    """z: (B, LATENT_DIM) → x̂: (B, 784) in [0,1].

    exp(P) ensures every dimension contributes a strictly positive pixel pattern;
    z (unconstrained) scales that pattern up/down. Bias b sets the per-pixel
    baseline, allowing dark pixels even when all dim contributions are positive.

    Per-dim visualisation: set inactive dims to z=0 → contribution = 0 * exp(P[:,d]) = 0.
    """
    P_pos = jnp.exp(jnp.clip(dec_params["P"], -10.0, 10.0))   # (784, D) positive
    return jax.nn.sigmoid(z @ P_pos.T + dec_params["b"])

@jax.jit
def loss_fn(dec_params, z, x_target):
    x_hat = decode(dec_params, z)
    x_t   = x_target / 255.0
    bce   = -(x_t * jnp.log(x_hat + 1e-7) + (1 - x_t) * jnp.log(1 - x_hat + 1e-7))
    return bce.mean()

def make_train_step(optimizer):
    @jax.jit
    def step(dec_params, opt_state, z_batch, x_batch):
        loss, grads = jax.value_and_grad(loss_fn)(dec_params, z_batch, x_batch)
        updates, new_opt = optimizer.update(grads, opt_state, dec_params)
        return optax.apply_updates(dec_params, updates), new_opt, loss
    return step

# ── Training ──────────────────────────────────────────────────────────────────

def train_decoder(Z_tr, X_tr, Z_te, X_te, latent_dim):
    key        = jax.random.PRNGKey(SEED)
    dec_params = init_decoder(latent_dim, key)
    optimizer  = optax.adam(ADAM_LR)
    opt_state  = optimizer.init(dec_params)
    train_step = make_train_step(optimizer)

    n = (len(Z_tr) // BATCH_SIZE) * BATCH_SIZE
    Z_b = Z_tr[:n].reshape(-1, BATCH_SIZE, Z_tr.shape[-1])
    X_b = X_tr[:n].reshape(-1, BATCH_SIZE, 784)

    history = []
    t0 = time.perf_counter()
    for epoch in range(MAX_EPOCHS):
        losses = []
        for b in range(len(Z_b)):
            dec_params, opt_state, loss = train_step(
                dec_params, opt_state, jnp.array(Z_b[b]), jnp.array(X_b[b])
            )
            losses.append(float(loss))

        x_hat    = np.array(decode(dec_params, jnp.array(Z_te)))
        test_mse = float(np.mean((x_hat - X_te / 255.0) ** 2))
        history.append({"epoch": epoch, "train_bce": float(np.mean(losses)), "test_mse": test_mse})

        if epoch % 20 == 0 or epoch == MAX_EPOCHS - 1:
            logging.info(f"  latent={latent_dim:3d}  epoch {epoch:3d}  "
                         f"bce={history[-1]['train_bce']:.4f}  mse={test_mse:.4f}  "
                         f"{time.perf_counter()-t0:.1f}s")

    return dec_params, history

# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_reconstructions(dec_params_list, Z_te_list, X_te, latent_dims, n_samples=8):
    """Rows: original + one per LATENT_DIM. Columns: sample images."""
    n_rows = len(latent_dims) + 1
    fig, axes = plt.subplots(n_rows, n_samples, figsize=(n_samples * 1.4, n_rows * 1.4))
    for j in range(n_samples):
        axes[0, j].imshow(X_te[j].reshape(28, 28), cmap="gray", vmin=0, vmax=255)
        axes[0, j].axis("off")
    axes[0, 0].set_ylabel("original", fontsize=7, rotation=0, ha="right", va="center")
    for i, (latent_dim, dec_params, Z_te) in enumerate(zip(latent_dims, dec_params_list, Z_te_list)):
        x_hat = np.array(decode(dec_params, jnp.array(Z_te[:n_samples])))
        for j in range(n_samples):
            axes[i+1, j].imshow(x_hat[j].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
            axes[i+1, j].axis("off")
        axes[i+1, 0].set_ylabel(f"dim={latent_dim}", fontsize=7, rotation=0, ha="right", va="center")
    fig.suptitle("Linear autodecoder: x̂ = sigmoid(z @ P.T)  —  frozen MLP-shallow encoder", fontsize=9)
    fig.tight_layout()
    return fig

def plot_loss_curves(histories, latent_dims):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for latent_dim, hist in zip(latent_dims, histories):
        epochs = [r["epoch"] for r in hist]
        ax1.plot(epochs, [r["train_bce"] for r in hist], label=f"dim={latent_dim}")
        ax2.plot(epochs, [r["test_mse"]  for r in hist], label=f"dim={latent_dim}")
    ax1.set_title("Train BCE"); ax2.set_title("Test MSE (pixels ÷ 255)")
    for ax in (ax1, ax2):
        ax.set_xlabel("epoch"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle("Linear autodecoder — encoder LATENT_DIM sweep", fontsize=10)
    fig.tight_layout()
    return fig

def plot_position_embeddings(dec_params_list, latent_dims):
    """P columns as 28×28 maps — first min(8, latent_dim) columns per row."""
    n_show = min(8, min(latent_dims))
    fig, axes = plt.subplots(len(latent_dims), n_show,
                             figsize=(n_show * 1.4, len(latent_dims) * 1.4))
    for i, (latent_dim, dec_params) in enumerate(zip(latent_dims, dec_params_list)):
        P = np.array(dec_params["P"])    # (784, latent_dim)
        for j in range(n_show):
            col = P[:, j]
            col = (col - col.min()) / (col.max() - col.min() + 1e-8)
            axes[i, j].imshow(col.reshape(28, 28), cmap="RdBu_r", vmin=0, vmax=1)
            axes[i, j].axis("off")
        axes[i, 0].set_ylabel(f"dim={latent_dim}", fontsize=7, rotation=0, ha="right", va="center")
    fig.suptitle("P columns (pixel position embeddings)", fontsize=9)
    fig.tight_layout()
    return fig

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t_start = time.perf_counter()
    logging.info(f"JAX devices: {jax.devices()}")

    data = load_supervised_image("mnist")
    X_tr = data.X.reshape(data.n_samples, 784).astype(np.float32)
    X_te = data.X_test.reshape(data.n_test_samples, 784).astype(np.float32)

    dec_params_list = []
    Z_te_list       = []
    histories       = []
    final_mses      = []

    for latent_dim in LATENT_DIMS:
        logging.info(f"\n── LATENT_DIM={latent_dim} ────────────────────────────")
        enc_params = load_encoder(latent_dim)
        Z_tr = encode_dataset(enc_params, X_tr)
        Z_te = encode_dataset(enc_params, X_te)
        logging.info(f"  Embeddings: {Z_tr.shape} train  {Z_te.shape} test")

        dec_params, history = train_decoder(Z_tr, X_tr, Z_te, X_te, latent_dim)
        dec_params_list.append(dec_params)
        Z_te_list.append(Z_te)
        histories.append(history)
        final_mses.append(history[-1]["test_mse"])

    fig = plot_reconstructions(dec_params_list, Z_te_list, X_te, LATENT_DIMS)
    url_recon = save_matplotlib_figure(f"{EXP_NAME}_reconstructions", fig, format="png")
    logging.info(f"Reconstructions → {url_recon}"); plt.close(fig)

    fig = plot_loss_curves(histories, LATENT_DIMS)
    url_loss = save_matplotlib_figure(f"{EXP_NAME}_loss_curves", fig, format="svg")
    logging.info(f"Loss curves → {url_loss}"); plt.close(fig)

    fig = plot_position_embeddings(dec_params_list, LATENT_DIMS)
    url_pemb = save_matplotlib_figure(f"{EXP_NAME}_position_embs", fig, format="png")
    logging.info(f"Position embeddings → {url_pemb}"); plt.close(fig)

    elapsed = time.perf_counter() - t_start
    logging.info(f"\nDone in {elapsed:.1f}s")
    for d, mse in zip(LATENT_DIMS, final_mses):
        logging.info(f"  latent={d:3d}  final test MSE={mse:.4f}")

    append_result({
        "exp_name"    : EXP_NAME,
        "description" : "Linear autodecoder (x̂=sigmoid(z@P.T)) sweep over frozen MLP-shallow encoder dims",
        "latent_dims" : LATENT_DIMS,
        "final_mses"  : final_mses,
        "url_recon"   : url_recon,
        "url_loss"    : url_loss,
        "url_pemb"    : url_pemb,
        "time_s"      : elapsed,
    })
