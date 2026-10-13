"""Gradient-alignment diagnostics: how close is a zeroth-order estimate to backprop's gradient?

This is the paper's own measurement (Section 4.2, Figure 5): the cosine between
an estimate and the backprop gradient *on the same batch*, per layer type,
against population. It is also the cheapest honest test of a port, because it
needs no training: if the estimator is right, the cosine climbs with population
along cos(K) = c_max / sqrt(1 + c/K), and the paper gives the numbers to compare.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from lib.model import BF16, F32, Cfg, mean_loss, rope_tables

# Layer types of Figure 5 / Table 8, and where each lives in the parameter tree.
LAYER_TYPES = ("mlp_in", "mlp_out", "attn_out", "q", "k", "v", "gate", "value_emb", "token_emb", "head")
_BLOCK_KEY = dict(mlp_in="fc", mlp_out="mproj", attn_out="proj", q="q", k="k", v="v", gate="gate",
                  value_emb="ve")


def true_gradient(params: dict, ids: Array, targets: Array, cfg: Cfg, dtype=BF16) -> dict:
    """Backprop's gradient of the mean token loss at `params`."""
    rot = rope_tables(cfg)
    return jax.jit(jax.grad(lambda p: mean_loss(p, cfg, ids, targets, rot, dtype)))(params)


def _cos(a: Array, b: Array) -> float:
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def cosines(est: dict, ref: dict) -> dict:
    """Cosine to the reference gradient per layer type (mean over the blocks that have one),
    plus the per-block values under '<type>/<layer>'."""
    out: dict = {}
    for t, key in _BLOCK_KEY.items():
        vals = {l: _cos(e[key], r[key]) for l, (e, r) in enumerate(zip(est["blocks"], ref["blocks"])) if key in e}
        if vals:
            out[t] = float(np.mean(list(vals.values())))
            for l, v in vals.items():
                out[f"{t}/{l}"] = v
    out["token_emb"] = _cos(est["wte"], ref["wte"])
    out["head"] = _cos(est["head"], ref["head"])
    return out


def norm_ratio(est: dict, ref: dict) -> float:
    """||estimate|| / ||reference|| over all matrices -- 1.0 means the scale is right too."""
    ne = sum(float(jnp.sum(jnp.square(x))) for x in jax.tree_util.tree_leaves(est))
    nr = sum(float(jnp.sum(jnp.square(x))) for x in jax.tree_util.tree_leaves(ref))
    return float(np.sqrt(ne / nr))
