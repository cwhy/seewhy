"""
Tiny causal transformer.

Params are a FLAT dict of arrays so that (a) a leading seed axis vmaps cleanly —
all seeds of a config train in one process — and (b) the whole thing pickles as
{k: np.array(v)} per the project workflow.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp


class Config(NamedTuple):
    n_layers: int
    d_model: int
    d_mlp: int
    n_heads: int
    d_head: int
    vocab: int
    seq_len: int


def n_params(p: dict) -> int:
    """Param count for a SINGLE model — call on un-vmapped params."""
    return sum(int(v.size) for v in p.values())


def init_params(key, cfg: Config) -> dict:
    ks = iter(jax.random.split(key, 8 + 8 * cfg.n_layers))
    lin = lambda k, shape: jax.random.normal(k, shape) * (1.0 / shape[0] ** 0.5)
    dh = cfg.n_heads * cfg.d_head

    p = {
        "tok_emb": jax.random.normal(next(ks), (cfg.vocab, cfg.d_model)) * 0.02,
        "pos_emb": jax.random.normal(next(ks), (cfg.seq_len, cfg.d_model)) * 0.02,
    }
    for i in range(cfg.n_layers):
        p[f"l{i}_ln1_g"] = jnp.ones((cfg.d_model,))
        p[f"l{i}_ln1_b"] = jnp.zeros((cfg.d_model,))
        for n in "qkv":
            p[f"l{i}_W{n}"] = lin(next(ks), (cfg.d_model, dh))
        p[f"l{i}_Wo"] = lin(next(ks), (dh, cfg.d_model))
        p[f"l{i}_ln2_g"] = jnp.ones((cfg.d_model,))
        p[f"l{i}_ln2_b"] = jnp.zeros((cfg.d_model,))
        p[f"l{i}_W1"] = lin(next(ks), (cfg.d_model, cfg.d_mlp))
        p[f"l{i}_b1"] = jnp.zeros((cfg.d_mlp,))
        p[f"l{i}_W2"] = lin(next(ks), (cfg.d_mlp, cfg.d_model))
        p[f"l{i}_b2"] = jnp.zeros((cfg.d_model,))
    p["lnf_g"] = jnp.ones((cfg.d_model,))
    p["lnf_b"] = jnp.zeros((cfg.d_model,))
    p["Wout"] = lin(next(ks), (cfg.d_model, cfg.vocab))
    p["bout"] = jnp.zeros((cfg.vocab,))
    return p


def init_mixer_params(key, cfg: Config) -> dict:
    """Causal MLP-Mixer, for the exp6 architecture comparison.

    Token mixing is a SINGLE linear map over positions, masked lower-triangular. A
    standard Mixer's two-layer token MLP cannot be made causal — its hidden units mix
    every position — so one masked matrix per layer is the honest causal analogue.

    That leaves the mixer with L*L = 1024 mixing parameters against the transformer's
    ~65k of QKVO, i.e. the comparison is generous to the transformer on capacity. If the
    mixer still wins, the point stands more strongly.
    """
    ks = iter(jax.random.split(key, 8 + 8 * cfg.n_layers))
    lin = lambda k, shape: jax.random.normal(k, shape) * (1.0 / shape[0] ** 0.5)
    L = cfg.seq_len

    p = {
        "tok_emb": jax.random.normal(next(ks), (cfg.vocab, cfg.d_model)) * 0.02,
        "pos_emb": jax.random.normal(next(ks), (L, cfg.d_model)) * 0.02,
    }
    for i in range(cfg.n_layers):
        p[f"l{i}_ln1_g"] = jnp.ones((cfg.d_model,))
        p[f"l{i}_ln1_b"] = jnp.zeros((cfg.d_model,))
        p[f"l{i}_Wtok"] = lin(next(ks), (L, L))
        p[f"l{i}_ln2_g"] = jnp.ones((cfg.d_model,))
        p[f"l{i}_ln2_b"] = jnp.zeros((cfg.d_model,))
        p[f"l{i}_W1"] = lin(next(ks), (cfg.d_model, cfg.d_mlp))
        p[f"l{i}_b1"] = jnp.zeros((cfg.d_mlp,))
        p[f"l{i}_W2"] = lin(next(ks), (cfg.d_mlp, cfg.d_model))
        p[f"l{i}_b2"] = jnp.zeros((cfg.d_model,))
    p["lnf_g"] = jnp.ones((cfg.d_model,))
    p["lnf_b"] = jnp.zeros((cfg.d_model,))
    p["Wout"] = lin(next(ks), (cfg.d_model, cfg.vocab))
    p["bout"] = jnp.zeros((cfg.vocab,))
    return p


def forward_mixer(p: dict, seq, cfg: Config):
    """seq (B, L) int32 -> logits (B, L, vocab). Causal: position t sees only <= t."""
    B, L = seq.shape
    h = p["tok_emb"][seq] + p["pos_emb"][:L]
    mask = jnp.tril(jnp.ones((L, L)))

    for i in range(cfg.n_layers):
        x = layer_norm(h, p[f"l{i}_ln1_g"], p[f"l{i}_ln1_b"])
        h = h + jnp.einsum("lk,bkd->bld", p[f"l{i}_Wtok"] * mask, x)
        y = layer_norm(h, p[f"l{i}_ln2_g"], p[f"l{i}_ln2_b"])
        h = h + jax.nn.gelu(y @ p[f"l{i}_W1"] + p[f"l{i}_b1"]) @ p[f"l{i}_W2"] + p[f"l{i}_b2"]

    return layer_norm(h, p["lnf_g"], p["lnf_b"]) @ p["Wout"] + p["bout"]


def layer_norm(x, g, b, eps: float = 1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / jnp.sqrt(var + eps) * g + b


def forward(p: dict, seq, cfg: Config, return_attn: bool = False):
    """seq (B, L) int32 -> logits (B, L, vocab); position t predicts token t+1.

    With return_attn, also returns post-softmax attention (n_layers, B, H, L, L).
    """
    B, L = seq.shape
    h = p["tok_emb"][seq] + p["pos_emb"][:L]
    mask = jnp.tril(jnp.ones((L, L), bool))
    attns = []

    for i in range(cfg.n_layers):
        x = layer_norm(h, p[f"l{i}_ln1_g"], p[f"l{i}_ln1_b"])
        q, k, v = (
            (x @ p[f"l{i}_W{n}"]).reshape(B, L, cfg.n_heads, cfg.d_head).swapaxes(1, 2)
            for n in "qkv"
        )
        s = jnp.einsum("bhqd,bhkd->bhqk", q, k) / cfg.d_head**0.5
        a = jax.nn.softmax(jnp.where(mask, s, -1e9), axis=-1)
        if return_attn:
            attns.append(a)
        o = jnp.einsum("bhqk,bhkd->bhqd", a, v).swapaxes(1, 2).reshape(B, L, -1)
        h = h + o @ p[f"l{i}_Wo"]
        y = layer_norm(h, p[f"l{i}_ln2_g"], p[f"l{i}_ln2_b"])
        h = h + jax.nn.gelu(y @ p[f"l{i}_W1"] + p[f"l{i}_b1"]) @ p[f"l{i}_W2"] + p[f"l{i}_b2"]

    logits = layer_norm(h, p["lnf_g"], p["lnf_b"]) @ p["Wout"] + p["bout"]
    return (logits, jnp.stack(attns)) if return_attn else logits
