"""Fixed evaluation episodes for the (context novelty) x (target presence) matrix.

Built once with numpy from a fixed seed so every experiment in the project is
scored on the same episodes. Indices within an episode are drawn WITHOUT
replacement, so "target not in context" is exact rather than approximately so.
"""

from typing import NamedTuple

import numpy as np
import jax.numpy as jnp

from .core import masked_mse, nn_baseline

# name -> (pool key, target is inside the context)
DEFAULT_CONDITIONS = {
    "A_seen_present":  ("train", True),    # the training condition
    "B_novel_present": ("held",  True),    # does retrieval transfer to unseen images?
    "C_seen_absent":   ("train", False),   # nothing to recall
    "D_novel_absent":  ("held",  False),   # nothing to recall, unseen images
}


class EvalSet(NamedTuple):
    ctx: jnp.ndarray        # (E, M, 784)
    qry: jnp.ndarray        # (E, Q, 784) — the TRUE target images
    tgt_idx: jnp.ndarray    # (E, Q) index into ctx, or -1 when the target is absent
    present: bool
    mse_mean: float         # baseline: predict the train-set mean image
    mse_nn: float           # baseline: best pure look-up from the context
    nn_idx: jnp.ndarray     # (E, Q) which context item the look-up baseline picks


def build(pools: dict, mask: np.ndarray, M: int, Q: int, n_ep: int,
          mean_img: np.ndarray, seed: int = 12345,
          conditions: dict | None = None) -> dict[str, EvalSet]:
    conditions = conditions or DEFAULT_CONDITIONS
    out = {}
    for ci, (name, (pool_name, present)) in enumerate(conditions.items()):
        rng = np.random.default_rng(seed + 1000 * ci)
        pool = pools[pool_name]
        n = pool.shape[0]
        draw = 0 if present else Q
        idx = np.stack([rng.choice(n, M + draw, replace=False) for _ in range(n_ep)])
        ctx = pool[idx[:, :M]]                                   # (E,M,784)
        if present:
            tgt_idx = rng.integers(0, M, size=(n_ep, Q)).astype(np.int32)
            qry = np.take_along_axis(ctx, tgt_idx[..., None], axis=1)
        else:
            tgt_idx = -np.ones((n_ep, Q), np.int32)
            qry = pool[idx[:, M:]]                               # (E,Q,784)

        ctx_j, qry_j, mask_j = jnp.array(ctx), jnp.array(qry), jnp.array(mask)
        mean_pred = jnp.broadcast_to(jnp.array(mean_img), qry_j.shape)
        m_nn, nn_idx = nn_baseline(ctx_j, qry_j, mask_j)
        out[name] = EvalSet(
            ctx=ctx_j, qry=qry_j, tgt_idx=jnp.array(tgt_idx), present=present,
            mse_mean=float(masked_mse(mean_pred, qry_j, mask_j)),
            mse_nn=float(m_nn), nn_idx=nn_idx,
        )
    return out
