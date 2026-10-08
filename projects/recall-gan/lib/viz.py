"""Pixel figures. Line plots live in lib/figures.py."""

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from lib.model import generate  # noqa: E402


def sample_grid(cell, cfg, pg, pool, n_ep: int = 4, n_gen: int = 6) -> str:
    """Per episode: the M context images, then n_gen generated images, each
    above the context image nearest to it. If the generator ignores the
    context, the second row will not resemble the third."""
    k1, k2 = jax.random.split(jax.random.PRNGKey(cell.seed + 1000))
    M = cell.m_ctx
    ctx = pool[jax.random.randint(k1, (n_ep, M), 0, pool.shape[0])]
    fake = generate(pg, ctx, jax.random.normal(k2, (n_ep, n_gen, cfg.z_dim)), cfg)
    nn = jnp.argmin(((fake[:, :, None] - ctx[:, None]) ** 2).mean(-1), -1)
    near = jnp.take_along_axis(ctx, nn[..., None], 1)

    cols = max(M, n_gen)
    fig, axes = plt.subplots(3 * n_ep, cols, figsize=(cols * 0.6, 3 * n_ep * 0.65))
    for ax in axes.ravel():
        ax.axis("off")
    for e in range(n_ep):
        for j in range(M):
            axes[3 * e, j].imshow(ctx[e, j].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
        for j in range(n_gen):
            axes[3 * e + 1, j].imshow(fake[e, j].reshape(28, 28), cmap="magma", vmin=0, vmax=1)
            axes[3 * e + 2, j].imshow(near[e, j].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
        axes[3 * e, 0].set_title(f"episode {e}: context (grey), generated (colour), nearest context",
                                 fontsize=6, loc="left")
    fig.tight_layout(pad=0.1)
    from shared_lib.media import save_matplotlib_figure
    url = save_matplotlib_figure(f"recall-gan_{cell.run}_samples", fig, dpi=150)
    plt.close(fig)
    return url
