"""
exp1 — Latent space geometry of the EMA-JEPA Gram-dual encoder.

Loads frozen params from projects/ssl/params_exp_ema_1024.pkl and visualises:
  1. t-SNE (2D) of all 10k MNIST test embeddings, coloured by class
  2. Class cluster quality: silhouette score, intra/inter-class distance ratio
  3. Nearest-neighbour retrieval: for 10 query images (one per class), show top-5 NNs
  4. Gram component attribution: for each of DEC_RANK=32 components,
     which pixels contribute most (via W_enc columns)

No training — this is a pure visualisation experiment.

Usage:
    uv run python projects/ema-viz/exp1.py
"""

import json, time, sys, pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logging

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared_lib.datasets import load_supervised_image
from shared_lib.media import save_matplotlib_figure
from lib.viz  import cluster_quality, knn_retrieval, plot_tsne, plot_retrieval, plot_gram_components, tsne_animation_html
from lib.tsne import capture_snapshots

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

EXP_NAME = "exp1"

# ── Encoder hyperparameters (must match ssl/exp_ema_1024.py exactly) ──────────

D_R, D_C, D_V = 6, 6, 4
D              = D_R * D_C * D_V   # 144
D_RC           = D_R * D_C         # 36
LATENT_DIM     = 1024
DEC_RANK       = 32                # DEC_RANK² = 1024 = LATENT_DIM

ROWS = jnp.arange(784) // 28
COLS = jnp.arange(784) % 28

# ── Paths ─────────────────────────────────────────────────────────────────────

SSL_DIR  = Path(__file__).parent.parent / "ssl"
JSONL    = Path(__file__).parent / "results.jsonl"

# ── Utilities ─────────────────────────────────────────────────────────────────

def append_result(row: dict):
    with open(JSONL, "a") as f:
        f.write(json.dumps(row, default=lambda x: float(x) if hasattr(x, "__float__") else str(x)) + "\n")

# ── Load encoder ──────────────────────────────────────────────────────────────

def load_encoder():
    pkl_path = SSL_DIR / "params_exp_ema_1024.pkl"
    logging.info(f"Loading encoder params from {pkl_path}")
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)
    return {k: jnp.array(v) for k, v in raw.items()}

# ── Encoder (copied from ssl/exp_ema_1024.py — do not modify) ────────────────

@jax.jit
def encode_batch(params, x):
    """Gram-dual encode.  x: (B, 784) float32 → (B, LATENT_DIM)."""
    x_norm   = x / 255.0
    re       = params["enc_row_emb"][ROWS]
    ce       = params["enc_col_emb"][COLS]
    pos_rc   = jax.vmap(lambda r, c: (r[:, None] * c[None, :]).reshape(D_RC))(re, ce)
    pos_feat = pos_rc @ params["W_pos_enc"]
    f        = pos_feat @ params["W_enc"]
    F        = x_norm[:, :, None] * f[None, :, :]
    H        = jnp.einsum("bpi,bpj->bij", F, F).reshape(x.shape[0], DEC_RANK ** 2)
    a        = H @ params["W_enc_A"]
    b        = H @ params["W_enc_B"]
    return a * b

def encode_dataset(params, X, batch_size=512):
    """Encode a full dataset in batches, returns (N, LATENT_DIM) np array."""
    N = X.shape[0]
    chunks = []
    for i in range(0, N, batch_size):
        xb = jnp.array(X[i:i+batch_size].astype(np.float32))
        chunks.append(np.array(encode_batch(params, xb)))
    return np.concatenate(chunks, axis=0)

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t0 = time.perf_counter()
    logging.info(f"JAX devices: {jax.devices()}")

    # Load data
    data   = load_supervised_image("mnist")
    X_test = data.X_test.reshape(data.n_test_samples, 784).astype(np.float32)
    Y_test = np.array(data.y_test, dtype=np.int32)
    logging.info(f"Test set: {X_test.shape}")

    # Load encoder
    params = load_encoder()
    n_enc_params = sum(v.size for v in jax.tree_util.tree_leaves(
        {k: v for k, v in params.items() if k not in ("W_pred1","b_pred1","W_pred2","b_pred2","W_film_s","b_film_s","W_film_t","b_film_t","action_emb")}
    ))
    logging.info(f"Encoder params: {n_enc_params:,}")

    # Embed test set
    logging.info("Encoding test set...")
    E_test = encode_dataset(params, X_test)
    logging.info(f"Embeddings: {E_test.shape}  (range [{E_test.min():.2f}, {E_test.max():.2f}])")

    # Cluster quality
    intra, inter, ratio = cluster_quality(E_test, Y_test)
    logging.info(f"Cluster quality — intra={intra:.3f}  inter={inter:.3f}  ratio={ratio:.2f}")

    # t-SNE — capture intermediate states for animation
    logging.info("Running t-SNE (capturing intermediate frames)...")
    snapshots = capture_snapshots(E_test, perplexity=40, seed=42, log_fn=logging.info)
    E_2d = snapshots[-1]
    logging.info(f"t-SNE done. {len(snapshots)} frames captured.")

    # Static final-frame plot (SVG)
    fig_tsne = plot_tsne(E_2d, Y_test,
        title=f"t-SNE of Gram-dual embeddings (MNIST test, LATENT={LATENT_DIM})\n"
              f"cluster ratio (inter/intra)={ratio:.2f}")
    url_tsne = save_matplotlib_figure(f"{EXP_NAME}_tsne", fig_tsne, format="svg")
    logging.info(f"  t-SNE static → {url_tsne}")
    plt.close(fig_tsne)

    # Animated HTML
    logging.info("Building t-SNE animation HTML...")
    html = tsne_animation_html(
        snapshots, Y_test,
        title=f"t-SNE optimisation drift — Gram-dual encoder (MNIST, LATENT={LATENT_DIM})",
        fps=2.0,
    )
    from shared_lib.media import save_media
    url_anim = save_media(f"{EXP_NAME}_tsne_anim.html", html.encode(), content_type="text/html")
    logging.info(f"  t-SNE animation → {url_anim}")

    # NN retrieval
    logging.info("Computing NN retrieval...")
    retrieval = knn_retrieval(E_test, Y_test, X_test, n_queries=10, k=5)
    fig_nn = plot_retrieval(retrieval, k=5)
    url_nn = save_matplotlib_figure(f"{EXP_NAME}_nn_retrieval", fig_nn, format="png")
    logging.info(f"  NN retrieval → {url_nn}")
    plt.close(fig_nn)

    # Gram component attribution
    logging.info("Plotting gram component attribution maps...")
    fig_comp = plot_gram_components(params, dec_rank=DEC_RANK, d_rc=D_RC)
    url_comp = save_matplotlib_figure(f"{EXP_NAME}_gram_components", fig_comp, format="png")
    logging.info(f"  Gram components → {url_comp}")
    plt.close(fig_comp)

    elapsed = time.perf_counter() - t0
    logging.info(f"Done in {elapsed:.1f}s")

    append_result({
        "exp_name"       : EXP_NAME,
        "description"    : "Latent space geometry: t-SNE, cluster quality, NN retrieval, gram attribution",
        "latent_dim"     : LATENT_DIM,
        "dec_rank"       : DEC_RANK,
        "n_test"         : int(X_test.shape[0]),
        "intra_dist"     : intra,
        "inter_dist"     : inter,
        "cluster_ratio"  : ratio,
        "url_tsne"       : url_tsne,
        "url_tsne_anim"  : url_anim,
        "n_tsne_frames"  : len(snapshots),
        "url_nn"         : url_nn,
        "url_components" : url_comp,
        "time_s"         : elapsed,
    })
