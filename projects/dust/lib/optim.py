"""Adam and Muon for the DUST / EGGROLL / backprop comparison, with per-group learning rates.

SGD with momentum (`dust.sgd_update`) is the paper's main protocol. Section 3.3 repeats the 1M-token ladder
under Adam, and the question here is also what Muon does to a zeroth-order gradient estimate. Both are written
as plain functions over the parameter pytree, so they take the estimates DUST and EGGROLL produce exactly as they
take backprop's gradient.

*Adam* (Kingma & Ba): per-coordinate running mean and mean-square of the gradient, step = lr * mhat / (sqrt(vhat) + eps).
Scale-free: multiplying every gradient by a constant changes nothing, which is why EGGROLL's factor-of-two estimate is harmless.

*Muon* (Jordan et al. 2024, in the Moonlight scaling of Liu et al. 2025) acts on 2-D hidden weight matrices only. It keeps
a Nesterov momentum buffer, replaces the buffer by the nearest matrix with all singular values equal to one (five steps of a
quintic Newton-Schulz iteration), and steps by lr * 0.2 * sqrt(max(m, n)) times that matrix. The 0.2 sqrt(max) factor makes the
update's per-entry RMS about 0.2, which is AdamW's, so Muon and Adam learning rates are comparable. Everything that is not a
hidden matrix (token and value embeddings, the head, the gates, the residual scalars) keeps Adam, as in modded-nanoGPT.

Why Muon is interesting for these estimators: an EGGROLL estimate of a matrix is a sum of P rank-one terms, so for P below the
matrix size it has rank at most P, with a very uneven spectrum. Orthogonalisation flattens the spectrum.
"""

import jax
import jax.numpy as jnp
from jax import Array

F32 = jnp.float32
MUON_KEYS = ("q", "k", "v", "proj", "fc", "mproj")      # the hidden matrices of a block
NS_COEF = (3.4445, -4.7750, 2.0315)                     # the quintic iteration of modded-nanoGPT


def newton_schulz(G: Array, steps: int = 5) -> Array:
    """The matrix with the same singular vectors as G and singular values driven to ~1 (not exactly: 0.7 to 1.2)."""
    a, b, c = NS_COEF
    X = G.astype(F32)
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.T
    X = X / (jnp.linalg.norm(X) + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * (A @ A)) @ X
    return X.T if transposed else X


def group_lrs(params: dict, lr: float, kind: str) -> dict:
    """Per-tensor learning rate under Adam or Muon, from the paper's Table 7.

    kind="dust":  matrices and head `lr`; token and value embeddings 300 x lr; residual scalars 100 x lr.
    kind="bp":    matrices and head `lr`; embeddings 0.3 and scalars 0.2, absolute.
    kind="eggroll": every tensor 0.003 (the paper's EGGROLL setting) -- `lr` is then used for the Muon matrices only.
    """
    lrs = jax.tree_util.tree_map(lambda _: lr, params)
    if kind == "dust":
        emb, scal = 300.0 * lr, 100.0 * lr
    elif kind == "bp":
        emb, scal = 0.3, 0.2
    elif kind == "eggroll":
        emb = scal = 0.003
        lrs = jax.tree_util.tree_map(lambda _: 0.003, params)
    else:
        raise ValueError(kind)
    lrs["wte"] = emb
    lrs["resid"] = lrs["x0"] = scal
    for b in lrs["blocks"]:
        if "ve" in b:
            b["ve"] = emb
    return lrs


def muon_mask(params: dict) -> dict:
    return {"wte": False, "head": False, "resid": False, "x0": False,
            "blocks": [{k: k in MUON_KEYS for k in b} for b in params["blocks"]]}


def make(kind: str, params: dict, lrs: dict, *, beta1: float, beta2: float = 0.95, eps: float = 1e-8, mu: float = 0.95,
         muon_lr: float | None = None):
    """Return (state, update). `update(params, state, grads) -> (params, state)` is jitted.

    kind "adam": every tensor Adam at its rate in `lrs`.
    kind "muon": hidden matrices Muon (rate `muon_lr`, default their rate in `lrs`), everything else Adam at `lrs`.
    """
    mask = muon_mask(params) if kind == "muon" else jax.tree_util.tree_map(lambda _: False, params)
    m_leaves, treedef = jax.tree_util.tree_flatten(mask)
    lr_leaves = jax.tree_util.tree_flatten(lrs)[0]
    if kind == "muon" and muon_lr is not None:
        lr_leaves = [muon_lr if is_m else lr for lr, is_m in zip(lr_leaves, m_leaves)]
    state = dict(t=jnp.int32(0), m=jax.tree_util.tree_map(jnp.zeros_like, params), v=jax.tree_util.tree_map(jnp.zeros_like, params))

    @jax.jit
    def update(params, state, grads):
        t = state["t"] + 1
        tf = t.astype(F32)
        p_l = jax.tree_util.tree_flatten(params)[0]
        g_l = jax.tree_util.tree_flatten(grads)[0]
        m_l = jax.tree_util.tree_flatten(state["m"])[0]
        v_l = jax.tree_util.tree_flatten(state["v"])[0]
        np_, nm, nv = [], [], []
        for p, g, m, v, lr, is_muon in zip(p_l, g_l, m_l, v_l, lr_leaves, m_leaves):
            if is_muon:
                m2 = mu * m + g
                O = newton_schulz(g + mu * m2)                       # Nesterov
                step = lr * 0.2 * (max(p.shape) ** 0.5) * O
                v2 = v
            else:
                m2 = beta1 * m + (1 - beta1) * g
                v2 = beta2 * v + (1 - beta2) * g * g
                step = lr * (m2 / (1 - beta1 ** tf)) / (jnp.sqrt(v2 / (1 - beta2 ** tf)) + eps)
            np_.append(p - step); nm.append(m2); nv.append(v2)
        un = lambda xs: jax.tree_util.tree_unflatten(treedef, xs)
        return un(np_), dict(t=t, m=un(nm), v=un(nv))

    return state, update
