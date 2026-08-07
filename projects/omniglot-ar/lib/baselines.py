"""
Non-learned baselines for in-context classification.

A number above chance is not by itself evidence of in-context learning: on
Omniglot, raw-pixel nearest neighbour already gets a fair way there. The claim
in `proposal.md` is that the token model beats *this*, so it is computed on the
same episodes, over the same `n_ctx` observed pixels, rather than quoted from
the literature (where it is measured on full images at a different resolution).
"""

import numpy as np


def nearest_neighbour(
    sup_x: np.ndarray,
    sup_slot: np.ndarray,
    qry_x: np.ndarray,
    qry_slot: np.ndarray,
    metric: str = "cosine",
) -> float:
    """1-NN accuracy over episodes.

    Args:
        sup_x:    (batch, n_support, n_ctx) support pixels in [0, 1].
        sup_slot: (batch, n_support) the support drawing's label slot.
        qry_x:    (batch, n_query, n_ctx)
        qry_slot: (batch, n_query)
        metric:   "cosine" or "euclidean".

    Returns:
        Mean accuracy over every query in every episode.
    """
    if metric == "cosine":
        norm = lambda a: a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
        score = np.einsum("bqd,bsd->bqs", norm(qry_x), norm(sup_x))
    elif metric == "euclidean":
        d2 = (
            (qry_x ** 2).sum(-1)[:, :, None]
            - 2 * np.einsum("bqd,bsd->bqs", qry_x, sup_x)
            + (sup_x ** 2).sum(-1)[:, None, :]
        )
        score = -d2
    else:
        raise ValueError(f"unknown metric {metric!r}; use 'cosine' or 'euclidean'")

    pred = np.take_along_axis(
        sup_slot[:, None, :], np.argmax(score, -1)[..., None], axis=-1
    ).squeeze(-1)
    return float((pred == qry_slot).mean())
