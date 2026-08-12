"""Fixed evaluation episodes for the (context novelty) x (target presence) matrix.

Built once with numpy from a fixed seed so every experiment in the project is
scored on the same episodes. Indices within an episode are drawn WITHOUT
replacement, so "target not in context" is exact rather than approximately so.

The present and absent conditions drawn from the same pool SHARE their queries.
Every number here is normalised by `mse_mean`, which depends only on the query
images; drawing the two conditions separately made the two denominators differ by
up to ~2%, which put a ~0.02 noise floor under every present-vs-absent
comparison. Sharing the queries makes the denominators equal by construction, so
"these two numbers are the same" is exact rather than merely within the floor.

The two contexts then differ in exactly Q of their M slots: the absent context is
the shared filler, and the present context overwrites Q randomly chosen slots
with the query images. Positions are random rather than at the end because the
state decays along the sequence, so a fixed slot would confound target presence
with recency.
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


def _draw(pool: np.ndarray, M: int, Q: int, n_ep: int, rng):
    """One shared draw for a pool: filler context, the queries, and their slots."""
    n = pool.shape[0]
    idx = np.stack([rng.choice(n, M + Q, replace=False) for _ in range(n_ep)])
    filler = pool[idx[:, :M]]                                    # (E,M,784)
    qry = pool[idx[:, M:]]                                       # (E,Q,784)
    slots = np.stack([rng.choice(M, Q, replace=False) for _ in range(n_ep)]).astype(np.int32)
    return filler, qry, slots


def build(pools: dict, mask: np.ndarray, M: int, Q: int, n_ep: int,
          mean_img: np.ndarray, seed: int = 12345,
          conditions: dict | None = None) -> dict[str, EvalSet]:
    conditions = conditions or DEFAULT_CONDITIONS
    assert Q <= M, f"need Q ({Q}) <= M ({M}) to place the targets in distinct slots"

    # One draw per pool, shared by that pool's present and absent conditions.
    pool_names = list(dict.fromkeys(p for p, _ in conditions.values()))
    draws = {name: _draw(pools[name], M, Q, n_ep, np.random.default_rng(seed + 1000 * i))
             for i, name in enumerate(pool_names)}

    mask_j = jnp.array(mask)
    out = {}
    for name, (pool_name, present) in conditions.items():
        filler, qry, slots = draws[pool_name]
        if present:
            ctx = filler.copy()
            np.put_along_axis(ctx, slots[..., None], qry, axis=1)
            tgt_idx = slots
        else:
            ctx = filler
            tgt_idx = -np.ones((n_ep, Q), np.int32)

        ctx_j, qry_j = jnp.array(ctx), jnp.array(qry)
        mean_pred = jnp.broadcast_to(jnp.array(mean_img), qry_j.shape)
        m_nn, nn_idx = nn_baseline(ctx_j, qry_j, mask_j)
        out[name] = EvalSet(
            ctx=ctx_j, qry=qry_j, tgt_idx=jnp.array(tgt_idx), present=present,
            mse_mean=float(masked_mse(mean_pred, qry_j, mask_j)),
            mse_nn=float(m_nn), nn_idx=nn_idx,
        )
    return out
