"""
Sequence models: the "how" axis of the task × architecture grid.

Every architecture here is the *same network* except for one operator. Each
layer is

    h = h + Mix(LN(h))
    h = h + MLP(LN(h))

with a shared linear read-in (d_in -> d_model) and read-out (d_model -> 1), and
only `Mix` changes:

    transformer   softmax attention          content-based routing, O(L^2) state
    linear_attn   kernelised linear attention content-based routing, fixed-size state
    gru           gated recurrence           no routing, fixed-size state
    mixer         masked static L x L mix    position-based routing, no content

That is the point of the comparison. A ranking between "a transformer" and "an
LSTM" taken off the shelf confounds the mixing operator with depth, width,
normalisation, initialisation and optimiser. Here those are held fixed, so a
difference in the ICL curve is attributable to the operator — and the operators
were chosen to bracket a specific hypothesis: **in-context learning needs
content-based routing.** `mixer` can only mix by position; `gru` can only carry
a summary; the two attention variants can look up "the earlier x most like this
one". If that hypothesis is right, the grid separates along that line and not
along parameter count.

Params are a flat dict of arrays so they pickle as `{k: np.array(v)}` per the
project workflow and so a leading seed axis vmaps without special-casing.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class Config(NamedTuple):
    arch: str
    d_in: int
    d_model: int = 64
    n_layers: int = 4
    n_heads: int = 4
    d_head: int = 16
    d_mlp: int = 256
    max_len: int = 128


ARCHS = ("transformer", "linear_attn", "gru", "mixer")


def n_params(p: dict) -> int:
    return sum(int(v.size) for v in p.values())


def layer_norm(x, g, b, eps: float = 1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / jnp.sqrt(var + eps) * g + b


# ── Initialisation ────────────────────────────────────────────────────────────

def init_params(key, cfg: Config) -> dict:
    """Flat param dict for any arch in `ARCHS`."""
    if cfg.arch not in ARCHS:
        raise ValueError(f"unknown arch {cfg.arch!r}; expected one of {ARCHS}")

    ks = iter(jax.random.split(key, 16 + 12 * cfg.n_layers))
    lin = lambda k, shape: jax.random.normal(k, shape) * (1.0 / shape[0] ** 0.5)
    dh = cfg.n_heads * cfg.d_head

    p = {
        "in_W": lin(next(ks), (cfg.d_in, cfg.d_model)),
        "in_b": jnp.zeros((cfg.d_model,)),
    }
    # Only the position-aware archs get a position embedding. Handing one to the
    # GRU would be free extra capacity for a model whose recurrence already
    # encodes order; handing one to nobody would cripple the mixer, whose whole
    # mechanism is positional.
    if cfg.arch in ("transformer", "mixer"):
        p["pos_emb"] = jax.random.normal(next(ks), (cfg.max_len, cfg.d_model)) * 0.02

    for i in range(cfg.n_layers):
        p[f"l{i}_ln1_g"] = jnp.ones((cfg.d_model,))
        p[f"l{i}_ln1_b"] = jnp.zeros((cfg.d_model,))

        if cfg.arch in ("transformer", "linear_attn"):
            for n in "qkv":
                p[f"l{i}_W{n}"] = lin(next(ks), (cfg.d_model, dh))
            p[f"l{i}_Wo"] = lin(next(ks), (dh, cfg.d_model))
        elif cfg.arch == "gru":
            for n in ("z", "r", "h"):
                p[f"l{i}_W{n}"] = lin(next(ks), (cfg.d_model, cfg.d_model))
                p[f"l{i}_U{n}"] = lin(next(ks), (cfg.d_model, cfg.d_model))
                p[f"l{i}_b{n}"] = jnp.zeros((cfg.d_model,))
        elif cfg.arch == "mixer":
            p[f"l{i}_Wtok"] = lin(next(ks), (cfg.max_len, cfg.max_len))

        p[f"l{i}_ln2_g"] = jnp.ones((cfg.d_model,))
        p[f"l{i}_ln2_b"] = jnp.zeros((cfg.d_model,))
        p[f"l{i}_W1"] = lin(next(ks), (cfg.d_model, cfg.d_mlp))
        p[f"l{i}_b1"] = jnp.zeros((cfg.d_mlp,))
        p[f"l{i}_W2"] = lin(next(ks), (cfg.d_mlp, cfg.d_model))
        p[f"l{i}_b2"] = jnp.zeros((cfg.d_model,))

    p["lnf_g"] = jnp.ones((cfg.d_model,))
    p["lnf_b"] = jnp.zeros((cfg.d_model,))
    p["out_W"] = lin(next(ks), (cfg.d_model, 1))
    p["out_b"] = jnp.zeros((1,))
    return p


# ── Mixing operators ──────────────────────────────────────────────────────────

def _attention(p, x, cfg: Config, i: int):
    B, L, _ = x.shape
    shape = (B, L, cfg.n_heads, cfg.d_head)
    q = (x @ p[f"l{i}_Wq"]).reshape(shape)
    k = (x @ p[f"l{i}_Wk"]).reshape(shape)
    v = (x @ p[f"l{i}_Wv"]).reshape(shape)

    scores = jnp.einsum("bqhd,bkhd->bhqk", q, k) / jnp.sqrt(cfg.d_head)
    causal = jnp.tril(jnp.ones((L, L), bool))
    scores = jnp.where(causal, scores, -jnp.inf)
    out = jnp.einsum("bhqk,bkhd->bqhd", jax.nn.softmax(scores, axis=-1), v)
    return out.reshape(B, L, -1) @ p[f"l{i}_Wo"]


def _linear_attention(p, x, cfg: Config, i: int):
    """Causal linear attention with the elu+1 feature map.

    Included because it is the architecture for which "attention implements one
    step of gradient descent on the in-context least-squares objective" is a
    *theorem* rather than a story. If a construction that provably can do linear
    ICL underperforms softmax attention here, the gap is about optimisation, not
    expressivity — and that distinction is worth a section of the report.
    """
    B, L, _ = x.shape
    shape = (B, L, cfg.n_heads, cfg.d_head)
    phi = lambda z: jax.nn.elu(z) + 1.0
    q = phi((x @ p[f"l{i}_Wq"]).reshape(shape))
    k = phi((x @ p[f"l{i}_Wk"]).reshape(shape))
    v = (x @ p[f"l{i}_Wv"]).reshape(shape)

    # Inclusive cumsum over positions == causal masking for a linear kernel.
    kv = jnp.cumsum(jnp.einsum("blhd,blhe->blhde", k, v), axis=1)   # (B,L,H,d,e)
    z = jnp.cumsum(k, axis=1)                                       # (B,L,H,d)
    num = jnp.einsum("blhd,blhde->blhe", q, kv)
    den = jnp.einsum("blhd,blhd->blh", q, z)[..., None] + 1e-6
    return (num / den).reshape(B, L, -1) @ p[f"l{i}_Wo"]


def _gru(p, x, cfg: Config, i: int):
    B = x.shape[0]

    def cell(h, xt):
        z = jax.nn.sigmoid(xt @ p[f"l{i}_Wz"] + h @ p[f"l{i}_Uz"] + p[f"l{i}_bz"])
        r = jax.nn.sigmoid(xt @ p[f"l{i}_Wr"] + h @ p[f"l{i}_Ur"] + p[f"l{i}_br"])
        n = jnp.tanh(xt @ p[f"l{i}_Wh"] + (r * h) @ p[f"l{i}_Uh"] + p[f"l{i}_bh"])
        h = (1.0 - z) * n + z * h
        return h, h

    _, hs = jax.lax.scan(cell, jnp.zeros((B, cfg.d_model)), jnp.swapaxes(x, 0, 1))
    return jnp.swapaxes(hs, 0, 1)


def _mixer(p, x, cfg: Config, i: int):
    """One masked linear map over positions — no content-based routing at all.

    A standard Mixer's two-layer token MLP cannot be made causal (its hidden
    units see every position), so one lower-triangular matrix per layer is the
    honest causal analogue. Borrowed from projects/sparse-attn-emergence.
    """
    L = x.shape[1]
    W = p[f"l{i}_Wtok"][:L, :L] * jnp.tril(jnp.ones((L, L)))
    return jnp.einsum("lk,bkd->bld", W, x)


_MIX = {
    "transformer": _attention,
    "linear_attn": _linear_attention,
    "gru": _gru,
    "mixer": _mixer,
}


# ── Forward ───────────────────────────────────────────────────────────────────

def forward(p: dict, seq: jnp.ndarray, cfg: Config) -> jnp.ndarray:
    """(B, L, d_in) -> (B, L, 1). Strictly causal for every arch."""
    L = seq.shape[1]
    if L > cfg.max_len:
        raise ValueError(f"sequence length {L} exceeds cfg.max_len={cfg.max_len}")

    h = seq @ p["in_W"] + p["in_b"]
    if "pos_emb" in p:
        h = h + p["pos_emb"][:L]

    mix = _MIX[cfg.arch]
    for i in range(cfg.n_layers):
        h = h + mix(p, layer_norm(h, p[f"l{i}_ln1_g"], p[f"l{i}_ln1_b"]), cfg, i)
        y = layer_norm(h, p[f"l{i}_ln2_g"], p[f"l{i}_ln2_b"])
        h = h + jax.nn.gelu(y @ p[f"l{i}_W1"] + p[f"l{i}_b1"]) @ p[f"l{i}_W2"] + p[f"l{i}_b2"]

    h = layer_norm(h, p["lnf_g"], p["lnf_b"])
    return h @ p["out_W"] + p["out_b"]
