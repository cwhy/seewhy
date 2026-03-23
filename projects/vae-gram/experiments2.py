"""
exp2 — Gram VAE: fix posterior collapse from exp1.

Root cause of exp1 failure:
  Position query q = pos_rc ⊗ h_dec where both start at scale 0.02.
  pos_rc = row_emb ⊗ col_emb has scale ~4e-4, so q has scale ~8e-6 — essentially zero.
  f = gram @ q ≈ 0 at init → decoder outputs uniform distributions → ignores z entirely →
  encoder collapses to prior → all reconstructions identical.

Fixes applied:
  1. Separate decoder position embeddings (dec_row_emb, dec_col_emb) initialised at
     proper scale (D_R**-0.5, D_C**-0.5) so pos_rc has magnitude ~1.
  2. Linear projection W_pos_dec: (D, D_RC) maps position → gram query space,
     replacing the broken pos ⊗ h_dec outer product.
  3. Separate dec_val_emb for decoder scoring (no conflicting gradients with encoder val_emb).
  4. Free bits: min KL per latent dim = FREE_BITS to prevent full posterior collapse.
  5. Longer KL warmup (20 epochs) to let reconstruction establish before KL pressure.

Usage:
    uv run python projects/vae-gram/experiments2.py
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
from shared_lib.random_utils import infinite_safe_keys_from_key
from shared_lib.datasets import load_supervised_image
from shared_lib.media import save_matplotlib_figure

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ── Hyperparameters ────────────────────────────────────────────────────────────

N_VAL_BINS = 32
BATCH_SIZE = 128
SEED       = 42
MAX_EPOCHS = 50

D_R, D_C, D_V = 4, 4, 4
D    = D_R * D_C * D_V   # 64
D_RC = D_R * D_C         # 16

N_ENC_Q          = 4
LATENT_DIM       = 32
DEC_RANK         = 8

ADAM_LR          = 1e-3
KL_WEIGHT        = 1.0
KL_WARMUP_EPOCHS = 20    # longer warmup so reconstruction stabilises first
FREE_BITS        = 0.5   # minimum KL per latent dim (nats) — prevents full collapse

VIZ_EVERY = 10

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

    return {
        # ── encoder embeddings ────────────────────────────────────────────
        "row_emb": n((28, D_R)),
        "col_emb": n((28, D_C)),
        "val_emb": n((N_VAL_BINS, D_V)),

        # ── encoder: gram → latent ────────────────────────────────────────
        "h_enc":        n((N_ENC_Q, D)),
        "ln_enc_gamma": jnp.ones(N_ENC_Q * D),
        "ln_enc_beta":  jnp.zeros(N_ENC_Q * D),
        "W_mu": n((N_ENC_Q * D, LATENT_DIM), scale=(N_ENC_Q * D) ** -0.5),
        "b_mu": jnp.zeros(LATENT_DIM),
        "W_lv": n((N_ENC_Q * D, LATENT_DIM), scale=(N_ENC_Q * D) ** -0.5),
        "b_lv": jnp.zeros(LATENT_DIM),

        # ── decoder: latent → pseudo-gram ─────────────────────────────────
        "W_A": n((LATENT_DIM, DEC_RANK * D), scale=LATENT_DIM ** -0.5),
        "W_B": n((LATENT_DIM, DEC_RANK * D), scale=LATENT_DIM ** -0.5),

        # ── decoder: separate position embeddings (proper scale) ───────────
        # Initialised at D_R**-0.5 / D_C**-0.5 so pos_rc has magnitude ~1,
        # making gram queries non-zero from step 1.
        "dec_row_emb": n((28, D_R),         scale=D_R ** -0.5),
        "dec_col_emb": n((28, D_C),         scale=D_C ** -0.5),
        "W_pos_dec":   n((D, D_RC),         scale=D_RC ** -0.5),

        # ── decoder: separate value embeddings (no conflicting gradients) ──
        "dec_val_emb": n((N_VAL_BINS, D_V), scale=D_V ** -0.5),
    }


# ── Encoder ───────────────────────────────────────────────────────────────────

def encode(params, x):
    B    = x.shape[0]
    bins = quantize(x)

    def embed_pixel(r, c, v_bin):
        re  = params["row_emb"][r]
        ce  = params["col_emb"][c]
        ve  = params["val_emb"][v_bin]
        pos = (re[:, None] * ce[None, :]).reshape(D_RC)
        return (pos[:, None] * ve[None, :]).reshape(D)

    def embed_image(bins_i):
        return jax.vmap(embed_pixel, in_axes=(0, 0, 0))(ROWS, COLS, bins_i)

    emb = jax.vmap(embed_image)(bins)
    M   = jnp.einsum("bti,btj->bij", emb, emb)

    queried = jnp.einsum("bij,qj->bqi", M, params["h_enc"])
    feat    = queried.reshape(B, N_ENC_Q * D)
    feat    = layer_norm(feat, params["ln_enc_gamma"], params["ln_enc_beta"])

    mu      = feat @ params["W_mu"] + params["b_mu"]
    log_var = feat @ params["W_lv"] + params["b_lv"]
    return mu, log_var


# ── Decoder ───────────────────────────────────────────────────────────────────

def decode(params, z):
    B = z.shape[0]

    A     = (z @ params["W_A"]).reshape(B, DEC_RANK, D)
    B_fac = (z @ params["W_B"]).reshape(B, DEC_RANK, D)
    gram  = jnp.einsum("bri,brj->bij", A, B_fac)

    def decode_pixel(r, c, g):
        re     = params["dec_row_emb"][r]    # (D_R,)  — separate, proper scale
        ce     = params["dec_col_emb"][c]    # (D_C,)
        pos_rc = (re[:, None] * ce[None, :]).reshape(D_RC)   # (D_RC,) magnitude ~1
        q      = params["W_pos_dec"] @ pos_rc                # (D,)  linear projection
        f      = g @ q                                        # (D,)
        f_v    = f.reshape(D_RC, D_V).mean(0)                # (D_V,)
        return params["dec_val_emb"] @ f_v                   # (N_VAL_BINS,)

    def decode_image(g):
        return jax.vmap(decode_pixel, in_axes=(0, 0, None))(ROWS, COLS, g)

    return jax.vmap(decode_image)(gram)


# ── Loss ──────────────────────────────────────────────────────────────────────

def vae_loss(params, x, z_key, beta):
    mu, log_var = encode(params, x)
    z = mu + jax.random.normal(z_key, mu.shape) * jnp.exp(0.5 * log_var)

    logits = decode(params, z)
    bins   = quantize(x)

    log_p      = jax.nn.log_softmax(logits, axis=-1)
    recon_loss = -jnp.take_along_axis(log_p, bins[..., None], axis=-1)[..., 0].mean()

    # Free bits: only penalise KL above FREE_BITS per dim
    kl_per_dim = -0.5 * (1.0 + log_var - mu ** 2 - jnp.exp(log_var))  # (B, LATENT_DIM)
    kl_loss    = jnp.maximum(kl_per_dim, FREE_BITS).mean()

    return recon_loss + beta * kl_loss, (recon_loss, kl_loss)


# ── Training step ─────────────────────────────────────────────────────────────

def make_train_step(optimizer):
    @jax.jit
    def step(params, opt_state, x, z_key, beta):
        (loss, aux), grads = jax.value_and_grad(vae_loss, has_aux=True)(params, x, z_key, beta)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss, aux
    return step


# ── Inference ────────────────────────────────────────────────────────────────

@jax.jit
def reconstruct_mean(params, x):
    """Expected pixel value — smooth but blurry."""
    mu, _  = encode(params, x)
    logits = decode(params, mu)
    probs  = jax.nn.softmax(logits, axis=-1)
    return (probs * BIN_CENTERS[None, None, :]).sum(-1)


@jax.jit
def reconstruct_mode(params, x):
    """Most-likely bin — sharper, no averaging blur."""
    mu, _     = encode(params, x)
    logits    = decode(params, mu)
    mode_bins = jnp.argmax(logits, axis=-1)   # (B, 784)
    return BIN_CENTERS[mode_bins]              # (B, 784)


# alias used by other helpers
reconstruct = reconstruct_mean


@jax.jit
def encode_mu(params, x):
    mu, _ = encode(params, x)
    return mu


@jax.jit
def pixel_entropy(params, x):
    mu, _  = encode(params, x)
    logits = decode(params, mu)
    log_p  = jax.nn.log_softmax(logits, axis=-1)
    p      = jnp.exp(log_p)
    return -(p * log_p).sum(-1)


# ── Visualisations ────────────────────────────────────────────────────────────

def _encode_batched(params, X, batch=256):
    mus = []
    for i in range(0, len(X), batch):
        mus.append(np.array(encode_mu(params, X[i:i + batch])))
    return np.concatenate(mus, axis=0)


def save_reconstruction_grid(params, X, tag, n=8):
    """3-row grid: original / expected-value / mode (argmax)."""
    x_orig = X[:n]
    x_mean = reconstruct_mean(params, x_orig)
    x_mode = reconstruct_mode(params, x_orig)

    fig, axes = plt.subplots(3, n, figsize=(n * 1.5, 5))
    labels = ["original", "mean (soft)", "mode (sharp)"]
    for row, imgs in enumerate([x_orig, x_mean, x_mode]):
        for i in range(n):
            axes[row, i].imshow(imgs[i].reshape(28, 28), cmap="gray", vmin=0, vmax=255)
            axes[row, i].axis("off")
        axes[row, 0].set_title(labels[row], fontsize=9, loc="left")
    fig.suptitle(f"exp2 reconstructions — {tag}", fontsize=10)
    plt.tight_layout()
    url = save_matplotlib_figure(f"exp2_reconstructions_{tag}", fig, dpi=150)
    plt.close(fig)
    logging.info(f"  reconstructions → {url}")


def save_pixel_entropy_grid(params, X, tag, n=4):
    x_sample = X[:n]
    ent      = np.array(pixel_entropy(params, x_sample))

    fig, axes = plt.subplots(2, n, figsize=(n * 2, 4.5))
    for i in range(n):
        axes[0, i].imshow(x_sample[i].reshape(28, 28), cmap="gray", vmin=0, vmax=255)
        im = axes[1, i].imshow(ent[i].reshape(28, 28), cmap="hot", vmin=0)
        axes[0, i].axis("off")
        axes[1, i].axis("off")
    axes[0, 0].set_title("original",             fontsize=9, loc="left")
    axes[1, 0].set_title("pixel entropy (nats)", fontsize=9, loc="left")
    plt.colorbar(im, ax=axes[1, -1], fraction=0.046, pad=0.04)
    fig.suptitle(f"exp2 pixel uncertainty — {tag}", fontsize=10)
    plt.tight_layout()
    url = save_matplotlib_figure(f"exp2_entropy_{tag}", fig, dpi=150)
    plt.close(fig)
    logging.info(f"  pixel entropy   → {url}")


def save_latent_pca(params, X, y, tag, n_samples=2000):
    X_sub = X[:n_samples]
    y_sub = np.array(y[:n_samples])
    mus   = _encode_batched(params, X_sub)

    mus_c = mus - mus.mean(0)
    _, _, Vt = np.linalg.svd(mus_c, full_matrices=False)
    z2d = mus_c @ Vt[:2].T

    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = plt.get_cmap("tab10")
    for cls in range(10):
        mask = y_sub == cls
        ax.scatter(z2d[mask, 0], z2d[mask, 1], s=6, alpha=0.5, color=cmap(cls), label=str(cls))
    ax.legend(markerscale=2, fontsize=8)
    ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
    ax.set_title(f"exp2 latent PCA — {tag}")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    url = save_matplotlib_figure(f"exp2_latent_pca_{tag}", fig, dpi=150)
    plt.close(fig)
    logging.info(f"  latent PCA      → {url}")


def save_learning_curves(history):
    epochs = range(len(history["recon"]))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(epochs, history["recon"],  label="recon NLL")
    ax1.plot(epochs, history["total"],  label="total", linestyle="--", alpha=0.7)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("nats / pixel")
    ax1.set_title("Reconstruction loss"); ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["kl"], color="orange")
    ax2.axhline(FREE_BITS, color="gray", linestyle=":", label=f"free bits={FREE_BITS}")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("KL (nats)")
    ax2.set_title("KL divergence"); ax2.legend(); ax2.grid(True, alpha=0.3)

    fig.suptitle("exp2 — gram VAE (collapse fixed)", fontsize=12)
    plt.tight_layout()
    url = save_matplotlib_figure("exp2_learning_curves", fig, dpi=150)
    plt.close(fig)
    logging.info(f"  learning curves → {url}")


# ── Training loop ─────────────────────────────────────────────────────────────

def train(params, X_tr, X_te, y_te):
    optimizer   = optax.adam(ADAM_LR)
    opt_state   = optimizer.init(params)
    step_fn     = make_train_step(optimizer)
    key_gen     = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED + 1))
    num_batches = len(X_tr) // BATCH_SIZE

    history = {"recon": [], "kl": [], "total": []}

    t0 = time.perf_counter()
    for epoch in range(MAX_EPOCHS):
        beta = min(1.0, epoch / max(KL_WARMUP_EPOCHS, 1)) * KL_WEIGHT

        key  = next(key_gen).get()
        perm = jax.random.permutation(key, len(X_tr))
        Xs   = X_tr[perm]

        e_recon = e_kl = e_total = 0.0
        for b in range(num_batches):
            bx = Xs[b * BATCH_SIZE: (b + 1) * BATCH_SIZE]
            zk = jax.random.fold_in(key, b)
            params, opt_state, loss, (recon, kl) = step_fn(params, opt_state, bx, zk, beta)
            e_recon += float(recon)
            e_kl    += float(kl)
            e_total += float(loss)

        e_recon /= num_batches
        e_kl    /= num_batches
        e_total /= num_batches
        history["recon"].append(e_recon)
        history["kl"].append(e_kl)
        history["total"].append(e_total)

        logging.info(
            f"epoch {epoch:2d}  β={beta:.2f}  recon={e_recon:.4f}"
            f"  kl={e_kl:.4f}  total={e_total:.4f}  {time.perf_counter() - t0:.1f}s"
        )

        if (epoch + 1) % VIZ_EVERY == 0:
            save_reconstruction_grid(params, X_te, f"epoch{epoch + 1:02d}")

    return params, history, time.perf_counter() - t0


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.info(f"JAX devices: {jax.devices()}")
    logging.info("Loading MNIST...")
    data    = load_supervised_image("mnist")
    X_train = data.X.reshape(data.n_samples,           -1).astype(jnp.float32)
    X_test  = data.X_test.reshape(data.n_test_samples, -1).astype(jnp.float32)
    y_test  = data.y_test

    key_gen = infinite_safe_keys_from_key(jax.random.PRNGKey(SEED))
    params  = init_params(key_gen)
    logging.info(f"Parameters: {n_params(params):,}")

    logging.info("Training...")
    params, history, elapsed = train(params, X_train, X_test, y_test)

    # Save params so visualisations can be regenerated without retraining
    params_path = Path(__file__).parent / "params_exp2.pkl"
    with open(params_path, "wb") as f:
        pickle.dump({k: np.array(v) for k, v in params.items()}, f)
    logging.info(f"  params saved → {params_path}")

    save_learning_curves(history)
    save_reconstruction_grid(params, X_test, "final")
    save_pixel_entropy_grid(params, X_test, "final")
    save_latent_pca(params, X_test, y_test, "final")

    row = {
        "experiment": "exp2",
        "name": f"gram-vae-fixed d={D} latent={LATENT_DIM} dec_rank={DEC_RANK} free_bits={FREE_BITS}",
        "n_params":        n_params(params),
        "epochs":          MAX_EPOCHS,
        "time_s":          round(elapsed, 1),
        "final_recon_nll": round(history["recon"][-1], 6),
        "final_kl":        round(history["kl"][-1],    6),
        "final_total":     round(history["total"][-1],  6),
        "recon_curve": [round(v, 6) for v in history["recon"]],
        "kl_curve":    [round(v, 6) for v in history["kl"]],
        "d_r": D_R, "d_c": D_C, "d_v": D_V,
        "n_enc_q":          N_ENC_Q,
        "latent_dim":       LATENT_DIM,
        "dec_rank":         DEC_RANK,
        "n_val_bins":       N_VAL_BINS,
        "adam_lr":          ADAM_LR,
        "kl_weight":        KL_WEIGHT,
        "kl_warmup_epochs": KL_WARMUP_EPOCHS,
        "free_bits":        FREE_BITS,
    }
    append_result(row)
    logging.info(
        f"Done. recon={history['recon'][-1]:.4f}  kl={history['kl'][-1]:.4f}"
        f"  total={history['total'][-1]:.4f}  {elapsed:.1f}s"
    )
