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

`ctx_mode` chooses what the context is made of. `iid` is the original task — M
unrelated images, which the baselines showed carry no usable information about a
seventeenth. `class` and `knn` build a context that is *about* the query, which is
the point of plan step B1: only then can an objective reward using it.

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


def _slots(M: int, Q: int, n_ep: int, rng) -> np.ndarray:
    return np.stack([rng.choice(M, Q, replace=False) for _ in range(n_ep)]).astype(np.int32)


def _draw_iid(pool, M, Q, n_ep, rng, labels=None, mask=None, knn_offset=0):
    """The original construction: M unrelated images, then Q unrelated queries."""
    n = pool.shape[0]
    idx = np.stack([rng.choice(n, M + Q, replace=False) for _ in range(n_ep)])
    return pool[idx[:, :M]], pool[idx[:, M:]], _slots(M, Q, n_ep, rng)


def _draw_class(pool, M, Q, n_ep, rng, labels=None, mask=None, knn_offset=0):
    """Every image in the episode — context and queries — is of one class."""
    assert labels is not None, "ctx_mode='class' needs labels for this pool"
    members = {c: np.flatnonzero(labels == c) for c in np.unique(labels)}
    classes = rng.choice(list(members), size=n_ep)
    idx = np.stack([rng.choice(members[c], M + Q, replace=False) for c in classes])
    return pool[idx[:, :M]], pool[idx[:, M:]], _slots(M, Q, n_ep, rng)


def _draw_knn(pool, M, Q, n_ep, rng, labels=None, mask=None, knn_offset=0):
    """The context is built FROM the queries: each query contributes its own
    k = M/Q nearest neighbours in the pool, ranked by distance on the VISIBLE
    half — the same quantity the soft-look-up ceiling is built from, which is what
    lets the ceiling be set rather than discovered.

    `knn_offset` skips that many ranks, dialling how informative the context is:
    offset 0 gives the closest neighbours available, a large offset gives
    same-dataset images that are not especially similar. Rank 0 is the query
    itself and is always dropped, so the absent condition stays exactly absent.
    """
    assert mask is not None, "ctx_mode='knn' needs the mask to define 'visible'"
    assert M % Q == 0, f"knn contexts need M ({M}) divisible by Q ({Q})"
    k = M // Q
    n = pool.shape[0]
    qidx = np.stack([rng.choice(n, Q, replace=False) for _ in range(n_ep)])   # (E,Q)
    qry = pool[qidx]

    vis = (mask < 0.5)
    Pv = jnp.array(pool[:, vis])
    flat = jnp.array(qry.reshape(n_ep * Q, -1)[:, vis])
    take = knn_offset + k + 1                       # +1 for the query itself at rank 0
    # |a-b|^2 = |a|^2 + |b|^2 - 2a.b, so only the (chunk, n) distance matrix is
    # ever materialised — the broadcast form needs tens of GB at pool scale.
    Pn = (Pv ** 2).sum(-1)[None, :]
    order = []
    for i in range(0, flat.shape[0], 256):
        c = flat[i:i + 256]
        d = (c ** 2).sum(-1)[:, None] + Pn - 2.0 * (c @ Pv.T)
        order.append(np.asarray(jnp.argsort(d, axis=-1)[:, :take]))
    order = np.concatenate(order).reshape(n_ep, Q, take)

    # Drop the query itself wherever it appears, then take the requested window.
    keep = np.empty((n_ep, Q, take - 1), np.int64)
    for e in range(n_ep):
        for j in range(Q):
            row = order[e, j]
            keep[e, j] = row[row != qidx[e, j]][:take - 1]
    nb = keep[:, :, knn_offset:knn_offset + k]                                # (E,Q,k)
    filler = pool[nb.reshape(n_ep, M)]
    return filler, qry, _slots(M, Q, n_ep, rng)


DRAWS = {"iid": _draw_iid, "class": _draw_class, "knn": _draw_knn}


def build(pools: dict, mask: np.ndarray, M: int, Q: int, n_ep: int,
          mean_img: np.ndarray, seed: int = 12345,
          conditions: dict | None = None, ctx_mode: str = "iid",
          labels: dict | None = None, knn_offset: int = 0) -> dict[str, EvalSet]:
    conditions = conditions or DEFAULT_CONDITIONS
    assert Q <= M, f"need Q ({Q}) <= M ({M}) to place the targets in distinct slots"
    draw_fn = DRAWS[ctx_mode]

    # One draw per pool, shared by that pool's present and absent conditions.
    pool_names = list(dict.fromkeys(p for p, _ in conditions.values()))
    draws = {name: draw_fn(pools[name], M, Q, n_ep, np.random.default_rng(seed + 1000 * i),
                           labels=(labels or {}).get(name), mask=mask,
                           knn_offset=knn_offset)
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
