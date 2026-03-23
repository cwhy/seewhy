"""
Shared visualization utilities for vae-gram experiments.

All functions save to R2 via save_matplotlib_figure and return the URL.
Pass encode_fn / decode_fn directly so this module stays architecture-agnostic.
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


def save_reconstruction_grid(params, X, encode_fn, decode_fn, bin_centers, exp_name, tag, n=8):
    x_orig = X[:n]
    mu, _  = encode_fn(params, x_orig)
    logits = decode_fn(params, mu)
    probs  = jax.nn.softmax(logits, axis=-1)
    x_mean = np.array((probs * bin_centers[None, None, :]).sum(-1))

    fig, axes = plt.subplots(2, n, figsize=(n * 1.5, 4))
    for i in range(n):
        axes[0, i].imshow(np.array(x_orig[i]).reshape(28, 28), cmap="gray", vmin=0, vmax=255)
        axes[1, i].imshow(x_mean[i].reshape(28, 28),           cmap="gray", vmin=0, vmax=255)
        axes[0, i].axis("off"); axes[1, i].axis("off")
    axes[0, 0].set_title("original", fontsize=9, loc="left")
    axes[1, 0].set_title("recon",    fontsize=9, loc="left")
    fig.suptitle(f"{exp_name} reconstructions — {tag}", fontsize=10)
    plt.tight_layout()
    url = save_matplotlib_figure(f"{exp_name}_reconstructions_{tag}", fig, format="png", dpi=150)
    plt.close(fig)
    logging.info(f"  reconstructions → {url}")
    return url


def save_sample_grid(params, decode_fn, latent_dim, bin_centers, exp_name, rows=5, cols=8, seed=0):
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


def save_learning_curves(history, free_bits, exp_name, subtitle=""):
    epochs = range(len(history["recon"]))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    axes[0].plot(epochs, history["recon"], label="recon NLL")
    axes[0].plot(epochs, history["total"], label="total", linestyle="--", alpha=0.7)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("nats/pixel")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["kl"], color="orange")
    axes[1].axhline(free_bits, color="gray", linestyle=":", label=f"free_bits={free_bits}")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("KL (nats)")
    axes[1].set_title("KL divergence"); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    acc_epochs = [e for e, a in history["bin_acc"]]
    acc_vals   = [a for e, a in history["bin_acc"]]
    axes[2].plot(acc_epochs, acc_vals, color="green", marker="o", markersize=3)
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Bin accuracy")
    axes[2].set_title("Pixel bin accuracy"); axes[2].grid(True, alpha=0.3)
    axes[2].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))

    fig.suptitle(f"{exp_name} — {subtitle}", fontsize=12)
    plt.tight_layout()
    url = save_matplotlib_figure(f"{exp_name}_learning_curves", fig, format="svg")
    plt.close(fig)
    logging.info(f"  learning curves → {url}")
    return url


def save_pixel_entropy_grid(params, X, encode_fn, decode_fn, exp_name, tag, n=4):
    x_sample = X[:n]
    mu, _    = encode_fn(params, x_sample)
    logits   = decode_fn(params, mu)
    log_p    = jax.nn.log_softmax(logits, axis=-1)
    ent      = np.array(-(jnp.exp(log_p) * log_p).sum(-1))

    fig, axes = plt.subplots(2, n, figsize=(n * 2, 4.5))
    for i in range(n):
        axes[0, i].imshow(np.array(x_sample[i]).reshape(28, 28), cmap="gray", vmin=0, vmax=255)
        im = axes[1, i].imshow(ent[i].reshape(28, 28), cmap="hot", vmin=0)
        axes[0, i].axis("off"); axes[1, i].axis("off")
    axes[0, 0].set_title("original",             fontsize=9, loc="left")
    axes[1, 0].set_title("pixel entropy (nats)", fontsize=9, loc="left")
    plt.colorbar(im, ax=axes[1, -1], fraction=0.046, pad=0.04)
    fig.suptitle(f"{exp_name} pixel uncertainty — {tag}", fontsize=10)
    plt.tight_layout()
    url = save_matplotlib_figure(f"{exp_name}_entropy_{tag}", fig, format="png", dpi=150)
    plt.close(fig)
    logging.info(f"  pixel entropy   → {url}")
    return url


def save_latent_pca(params, X, y, encode_fn, exp_name, tag, n_samples=2000):
    mus   = np.concatenate([np.array(encode_fn(params, X[i:i+256])[0])
                            for i in range(0, min(n_samples, len(X)), 256)], axis=0)
    y_sub = np.array(y[:len(mus)])
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
    ax.set_title(f"{exp_name} latent PCA — {tag}"); ax.grid(True, alpha=0.2)
    plt.tight_layout()
    url = save_matplotlib_figure(f"{exp_name}_latent_pca_{tag}", fig, format="svg")
    plt.close(fig)
    logging.info(f"  latent PCA      → {url}")
    return url
