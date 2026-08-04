"""
Shared metrics. Mirrors the inline definitions used in experiments1.py — exp1 keeps
its own copies so its recorded row stays exactly reproducible from that file.
"""

import jax.numpy as jnp
import numpy as np


def time_to_emergence(acc, thresh, win=10):
    """First step whose trailing-mean acc exceeds thresh; None if it never does."""
    sm = np.convolve(acc, np.ones(win) / win, mode="valid")
    hit = np.nonzero(sm > thresh)[0]
    return int(hit[0] + win - 1) if hit.size else None


def support_iou(a, A, s, S, n_heads):
    """Agreement between attention and the ground-truth support of A.

    a: (H, L, L) batch-mean attention of the layer being probed.
    Query row i is at position S-1+i (it predicts token S+i); its true support is
    the s columns where row i of A is 1. Candidates are restricted to [0, S) —
    where the support lives.

    Returns (head_best, row_best):
      head_best — pick one head, average its IoU over rows, take the best head.
                  This is what exp1 logged, and it understates the model when
                  different heads specialise on different rows.
      row_best  — per row take the best head, then average over rows. Correct
                  aggregation when the circuit spreads rows across heads.
    """
    qpos = S - 1 + jnp.arange(S)
    aq = a[:, qpos, :]                                          # (H, S, L)
    top = jnp.argsort(-aq[:, :, :S], -1)[:, :, :s]               # (H, S, s)
    sel = jnp.take_along_axis(jnp.broadcast_to(A.astype(jnp.float32), (n_heads, S, S)), top, -1)
    inter = sel.sum(-1)                                          # |top-s ∩ support|
    iou = inter / (2 * s - inter)                                 # (H, S)
    return iou.mean(-1).max(), iou.max(0).mean()


def attn_entropy(a, S):
    """Per-head attention entropy averaged over second-half query rows -> (H,)."""
    aq = a[:, S - 1 + jnp.arange(S), :]
    return -(aq * jnp.log(aq + 1e-12)).sum(-1).mean(-1)
