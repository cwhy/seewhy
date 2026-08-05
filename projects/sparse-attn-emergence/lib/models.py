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


def forward_mixer(p: dict, seq, cfg: Config, causal: bool = True):
    """seq (B, L) int32 -> logits (B, L, vocab).

    causal=True masks the mixing matrix lower-triangular, so position t sees only <= t.

    causal=False is deliberately UNSOUND for next-token prediction and exists only as a
    diagnostic: an unmasked mixing matrix lets position t draw on position t+1, i.e. the
    very token being predicted, so the task becomes trivial by leakage. The paper does not
    state whether its mixer is masked, and this arm measures what that choice is worth.
    """
    B, L = seq.shape
    h = p["tok_emb"][seq] + p["pos_emb"][:L]
    mask = jnp.tril(jnp.ones((L, L))) if causal else jnp.ones((L, L))

    for i in range(cfg.n_layers):
        x = layer_norm(h, p[f"l{i}_ln1_g"], p[f"l{i}_ln1_b"])
        h = h + jnp.einsum("lk,bkd->bld", p[f"l{i}_Wtok"] * mask, x)
        y = layer_norm(h, p[f"l{i}_ln2_g"], p[f"l{i}_ln2_b"])
        h = h + jax.nn.gelu(y @ p[f"l{i}_W1"] + p[f"l{i}_b1"]) @ p[f"l{i}_W2"] + p[f"l{i}_b2"]

    return layer_norm(h, p["lnf_g"], p["lnf_b"]) @ p["Wout"] + p["bout"]


def _decay_bias(horizon: float) -> float:
    """Bias b such that sigmoid(b)^horizon = 1/e — memory spans ~horizon tokens.

    Ported from projects/universal-ar. Initialising decay from the sequence length keeps
    the memory horizon matched to the data instead of annihilating most of it.
    """
    a = float(jnp.exp(-1.0 / horizon))
    return float(jnp.log(a / (1.0 - a)))


def init_kda_params(key, cfg: Config, horizon: float | None = None) -> dict:
    """Kimi Delta Attention — a matrix-valued memory written by the delta rule.

    Reference implementation: projects/universal-ar/experiments30.py. Same family as the
    Gated DeltaNet / linear-RNN arms the paper compares against.
    """
    ks = iter(jax.random.split(key, 8 + 10 * cfg.n_layers))
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
        p[f"l{i}_Wa"] = lin(next(ks), (cfg.d_model, dh)) * 0.1        # per-channel decay
        p[f"l{i}_ba"] = jnp.full((dh,), _decay_bias(horizon or cfg.seq_len))
        p[f"l{i}_Wb"] = lin(next(ks), (cfg.d_model, cfg.n_heads))     # write strength
        p[f"l{i}_bb"] = jnp.zeros((cfg.n_heads,))
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


def forward_kda(p: dict, seq, cfg: Config):
    """seq (B, L) int32 -> logits (B, L, vocab). Causal by construction.

    Per position:  forget  S ← S · Diag(α_t)      per-channel decay
                   predict v̂ = S k_t              what the memory holds for this key
                   correct e = β_t (v_t − v̂)      the error
                   write   S ← S + e k_tᵀ
                   read    o_t = S q_t / √d_head

    ADAPTED from the reference, which is a set model: there, every token reads the state
    after the whole sequence has been written, which would leak the future here. Each
    position reads the state as of itself, so nothing after t can influence position t.
    """
    B, L = seq.shape
    H, DK = cfg.n_heads, cfg.d_head
    h = p["tok_emb"][seq] + p["pos_emb"][:L]
    unit = lambda t: t / (jnp.linalg.norm(t, axis=-1, keepdims=True) + 1e-6)
    sh = lambda t: t.reshape(B, L, H, DK).transpose(0, 2, 1, 3)          # (B,H,L,DK)

    for i in range(cfg.n_layers):
        x = layer_norm(h, p[f"l{i}_ln1_g"], p[f"l{i}_ln1_b"])
        q = unit(sh(x @ p[f"l{i}_Wq"]))                                   # DeltaNet convention
        k = unit(sh(x @ p[f"l{i}_Wk"]))
        v = sh(x @ p[f"l{i}_Wv"])
        alpha = jax.nn.sigmoid(sh(x @ p[f"l{i}_Wa"] + p[f"l{i}_ba"]))
        beta = jax.nn.sigmoid(x @ p[f"l{i}_Wb"] + p[f"l{i}_bb"]).transpose(0, 2, 1)  # (B,H,L)

        def step(S, t):
            a_t, k_t, v_t, b_t, q_t = t
            S = S * a_t[:, :, None, :]
            vhat = jnp.einsum("bhvk,bhk->bhv", S, k_t)
            S = S + jnp.einsum("bhv,bhk->bhvk", b_t[..., None] * (v_t - vhat), k_t)
            return S, jnp.einsum("bhvk,bhk->bhv", S, q_t) / DK**0.5

        _, o = jax.lax.scan(
            step, jnp.zeros((B, H, DK, DK)),
            (alpha.transpose(2, 0, 1, 3), k.transpose(2, 0, 1, 3), v.transpose(2, 0, 1, 3),
             beta.transpose(2, 0, 1), q.transpose(2, 0, 1, 3)))
        # scan stacks outputs as (L, B, H, DV). The axis order here is load-bearing: an
        # earlier version used transpose(1, 2, 0, 3) -> (B, H, L, DV) and reshaped THAT to
        # (B, L, H*DV), which silently interleaves the head and position axes so a position
        # receives values belonging to other positions — including later ones. It leaked the
        # future and fit pure noise to zero loss. scripts/check_kda_leak.py guards this.
        h = h + o.transpose(1, 0, 2, 3).reshape(B, L, -1) @ p[f"l{i}_Wo"]

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
