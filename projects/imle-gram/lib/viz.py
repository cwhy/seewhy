"""
Shared visualization utilities for imle-gram experiments.

IMLE has no encoder, so reconstruction is done via nearest-neighbor matching:
sample many z, decode, find closest to each real image.

All functions save to R2 via save_matplotlib_figure and return the URL.
Pass decode_fn directly so this module stays architecture-agnostic.
"""

import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from shared_lib.media import save_matplotlib_figure


def _decode_soft_mean(params, decode_fn, z, bin_centers):
    """Decode z to soft-mean pixel images. Returns numpy (B, 784)."""
    logits = decode_fn(params, z)
    probs  = jax.nn.softmax(logits, axis=-1)
    return np.array((probs * bin_centers[None, None, :]).sum(-1))


def _nearest_neighbors(X_real, X_tilde):
    """
    For each row in X_real, find index of nearest row in X_tilde (L2).
    X_real:  (n, d) numpy
    X_tilde: (m, d) numpy
    Returns: (n,) int indices into X_tilde
    """
    d_r = (X_real  ** 2).sum(1, keepdims=True)   # (n, 1)
    d_t = (X_tilde ** 2).sum(1)                   # (m,)
    dists = d_r + d_t - 2 * (X_real @ X_tilde.T)  # (n, m)
    return np.argmin(dists, axis=1)                # (n,)


def save_reconstruction_grid(params, X, decode_fn, latent_dim, bin_centers,
                              exp_name, tag, n=8, m_match=2000, seed=0):
    """
    Nearest-neighbor 'reconstruction': sample m_match latent codes, find the
    closest decoded sample for each of the first n real images, display side by side.
    """
    key = jax.random.PRNGKey(seed)
    Z   = jax.random.normal(key, (m_match, latent_dim))

    # Decode in chunks to avoid OOM
    chunks = [np.array(jax.nn.softmax(decode_fn(params, Z[i:i+256]), axis=-1))
              for i in range(0, m_match, 256)]
    probs_all = np.concatenate(chunks, axis=0)                     # (m_match, 784, bins)
    X_tilde   = (probs_all * np.array(bin_centers)[None, None, :]).sum(-1)  # (m_match, 784)

    X_real = np.array(X[:n])
    idx    = _nearest_neighbors(X_real, X_tilde)

    fig, axes = plt.subplots(2, n, figsize=(n * 1.5, 4))
    for i in range(n):
        axes[0, i].imshow(X_real[i].reshape(28, 28), cmap="gray", vmin=0, vmax=255)
        axes[1, i].imshow(X_tilde[idx[i]].reshape(28, 28), cmap="gray", vmin=0, vmax=255)
        axes[0, i].axis("off"); axes[1, i].axis("off")
    axes[0, 0].set_title("original", fontsize=9, loc="left")
    axes[1, 0].set_title("nearest sample", fontsize=9, loc="left")
    fig.suptitle(f"{exp_name} nearest-neighbor recon — {tag}", fontsize=10)
    plt.tight_layout()
    url = save_matplotlib_figure(f"{exp_name}_recon_{tag}", fig, format="png", dpi=150)
    plt.close(fig)
    logging.info(f"  recon grid      → {url}")
    return url


def save_sample_grid(params, decode_fn, latent_dim, bin_centers,
                     exp_name, rows=5, cols=8, seed=0):
    """Sample z ~ N(0,I), decode with soft mean (no sharpening), save grid."""
    n = rows * cols
    z = jax.random.normal(jax.random.PRNGKey(seed), (n, latent_dim))
    logits = decode_fn(params, z)
    probs  = jax.nn.softmax(logits, axis=-1)
    imgs   = np.array((probs * bin_centers[None, None, :]).sum(-1))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.4, rows * 1.4))
    for i, ax in enumerate(axes.flat):
        ax.imshow(imgs[i].reshape(28, 28), cmap="gray", vmin=0, vmax=255)
        ax.axis("off")
    fig.suptitle(f"{exp_name} prior samples (seed={seed})", fontsize=10)
    plt.tight_layout()
    url = save_matplotlib_figure(f"{exp_name}_samples_{rows}x{cols}_seed{seed}", fig, format="png", dpi=150)
    plt.close(fig)
    logging.info(f"  prior samples   → {url}")
    return url


def save_learning_curves(history, exp_name, subtitle=""):
    """Plot match loss (train + eval) and bin accuracy over epochs."""
    epochs = range(len(history["match_loss"]))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(epochs, history["match_loss"], label="train match loss")
    if history["match_loss_eval"]:
        eval_e = [e for e, _ in history["match_loss_eval"]]
        eval_v = [v for _, v in history["match_loss_eval"]]
        axes[0].plot(eval_e, eval_v, label="eval match loss", linestyle="--")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("L2 / pixel")
    axes[0].set_title("Match loss (L2 to nearest sample)")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    if history["bin_acc"]:
        acc_e = [e for e, _ in history["bin_acc"]]
        acc_v = [v for _, v in history["bin_acc"]]
        axes[1].plot(acc_e, acc_v, color="green", marker="o", markersize=3)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Bin accuracy")
    axes[1].set_title("NN bin accuracy (eval)")
    axes[1].grid(True, alpha=0.3)
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))

    fig.suptitle(f"{exp_name} — {subtitle}", fontsize=12)
    plt.tight_layout()
    url = save_matplotlib_figure(f"{exp_name}_learning_curves", fig, format="svg")
    plt.close(fig)
    logging.info(f"  learning curves → {url}")
    return url
