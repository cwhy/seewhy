"""Discriminator and generator for the in-context membership game.

Discriminator  D(ctx, x) -> logit "x was one of the context images".
               The context is written into one KDA state; the query reads it.
               That state is D's only channel to the context.

Generator      G(z) or G(ctx, z) -> image.
               The context-aware form writes the context into its OWN state
               and reads it with a query derived from the noise. The
               context-blind form has no state at all, so it cannot depend
               on the context even by accident.

KDA write, per head (the same rule as recall-gen/lib/core.py, one layer):

    S <- S . Diag(alpha)                 per-channel decay
    S <- S + beta (v - S k) k^T          delta-rule correction
    o  = S q / sqrt(dk)                  read
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

PIX = 784


class Cfg(NamedTuple):
    d_model: int = 256
    n_heads: int = 4
    dk: int = 64                  # per-head key and value dim; state is dk x dk per head
    enc_hidden: int = 512
    z_dim: int = 64
    gen_hidden: int = 512
    m_ctx: int = 16               # sets the decay-horizon init only
    horizon_mult: float = 8.0
    # GAN stabilisers, added after exp2 showed the plain GAN collapsing. The
    # defaults are exp1/exp2's settings, so those rows rebuild unchanged.
    d_act: str = "gelu"           # discriminator nonlinearity: gelu | lrelu
    g_ln: bool = False            # LayerNorm after each generator hidden layer

    @property
    def state_floats(self) -> int:
        return self.n_heads * self.dk * self.dk


def _lin(k, s, scale=1.0):
    return jax.random.normal(k, s) * (scale / s[0] ** 0.5)


def _decay_bias(H: float) -> float:
    """Bias b with sigmoid(b)^H = 1/e, i.e. memory spans H writes."""
    a = float(np.exp(-1.0 / H))
    return float(np.log(a / (1.0 - a)))


def _l2norm(t, eps=1e-6):
    return t * jax.lax.rsqrt(jnp.sum(t * t, axis=-1, keepdims=True) + eps)


def n_params(p) -> int:
    return int(sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(p)))


# ── shared pieces ─────────────────────────────────────────────────────────────

def init_encoder(key, cfg: Cfg):
    k1, k2 = jax.random.split(key)
    return dict(W1=_lin(k1, (PIX, cfg.enc_hidden)), b1=jnp.zeros(cfg.enc_hidden),
                W2=_lin(k2, (cfg.enc_hidden, cfg.d_model)), b2=jnp.zeros(cfg.d_model))


ACTS = {"gelu": jax.nn.gelu, "lrelu": lambda t: jax.nn.leaky_relu(t, 0.2)}


def encode(p, x, act: str = "gelu"):
    return ACTS[act](x @ p["W1"] + p["b1"]) @ p["W2"] + p["b2"]


def _ln(t, eps=1e-5):
    m = t.mean(-1, keepdims=True)
    return (t - m) * jax.lax.rsqrt(t.var(-1, keepdims=True) + eps)


def init_memory(key, cfg: Cfg):
    D, H = cfg.d_model, cfg.n_heads
    ks = jax.random.split(key, 4)
    return dict(Wk=_lin(ks[0], (D, D)), Wv=_lin(ks[1], (D, D)),
                Wa=_lin(ks[2], (D, D), 0.1), ba=jnp.full((D,), _decay_bias(cfg.horizon_mult * cfg.m_ctx)),
                Wb=_lin(ks[3], (D, H)), bb=jnp.zeros(H))


def write(p, e_ctx, cfg: Cfg):
    """e_ctx (B,M,D) -> final state S (B,H,dk,dk)."""
    B, M, D = e_ctx.shape
    H, DK = cfg.n_heads, cfg.dk
    sh = lambda t: t.reshape(B, M, H, DK).transpose(1, 0, 2, 3)        # (M,B,H,DK)
    k = _l2norm(sh(e_ctx @ p["Wk"]))
    v = sh(e_ctx @ p["Wv"])
    alpha = jax.nn.sigmoid(sh(e_ctx @ p["Wa"] + p["ba"]))
    beta = jax.nn.sigmoid(e_ctx @ p["Wb"] + p["bb"]).transpose(1, 0, 2)  # (M,B,H)

    def step(S, t):
        a_t, k_t, v_t, b_t = t
        S = S * a_t[:, :, None, :]
        e = b_t[..., None] * (v_t - jnp.einsum("bhvk,bhk->bhv", S, k_t))
        return S + jnp.einsum("bhv,bhk->bhvk", e, k_t), None

    S, _ = jax.lax.scan(step, jnp.zeros((B, H, DK, DK)), (alpha, k, v, beta))
    return S


def read(S, q, cfg: Cfg):
    """S (B,H,dk,dk), q (B,Q,D) unnormalised -> (B,Q,D)."""
    B, Q, D = q.shape
    H, DK = cfg.n_heads, cfg.dk
    qh = _l2norm(q.reshape(B, Q, H, DK))
    return jnp.einsum("bhvk,bqhk->bqhv", S, qh).reshape(B, Q, D) / DK ** 0.5


# ── discriminator ─────────────────────────────────────────────────────────────

def init_disc(key, cfg: Cfg):
    D = cfg.d_model
    ks = jax.random.split(key, 5)
    return dict(enc=init_encoder(ks[0], cfg), mem=init_memory(ks[1], cfg),
                Wq=_lin(ks[2], (D, D)),
                W_h=_lin(ks[3], (2 * D, D)), b_h=jnp.zeros(D),
                w_out=_lin(ks[4], (D, 1)), b_out=jnp.zeros(1))


def disc_logits(p, ctx, qry, cfg: Cfg):
    """ctx (B,M,784), qry (B,Q,784) -> logits (B,Q).

    The head sees the read-out and the query's own embedding, so it can compare
    "what the state returned at this key" with "what this image is". A query
    embedding alone carries no context information; everything about the
    context arrives through `o`.
    """
    S = write(p["mem"], encode(p["enc"], ctx, cfg.d_act), cfg)
    e = encode(p["enc"], qry, cfg.d_act)
    o = read(S, e @ p["Wq"], cfg)
    h = ACTS[cfg.d_act](jnp.concatenate([o, e], -1) @ p["W_h"] + p["b_h"])
    return (h @ p["w_out"] + p["b_out"])[..., 0]


# ── generator ─────────────────────────────────────────────────────────────────

def init_gen(key, cfg: Cfg, use_ctx: bool):
    D, Z, Hd = cfg.d_model, cfg.z_dim, cfg.gen_hidden
    ks = jax.random.split(key, 6)
    p = dict(W1=_lin(ks[0], (Z + (D if use_ctx else 0), Hd)), b1=jnp.zeros(Hd),
             W2=_lin(ks[1], (Hd, Hd)), b2=jnp.zeros(Hd),
             W3=_lin(ks[2], (Hd, PIX), 0.1), b3=jnp.zeros(PIX))
    if use_ctx:
        p |= dict(enc=init_encoder(ks[3], cfg), mem=init_memory(ks[4], cfg),
                  Wq=_lin(ks[5], (Z, D)))
    return p


def generate(p, ctx, z, cfg: Cfg):
    """z (B,Q,z_dim) -> images (B,Q,784) in [0,1]. `ctx` is ignored unless the
    generator was built with a memory."""
    return jax.nn.sigmoid(generate_logits(p, ctx, z, cfg))


def generate_logits(p, ctx, z, cfg: Cfg):
    """Pre-sigmoid generator output, for the saturation diagnostic."""
    h = z
    if "mem" in p:
        S = write(p["mem"], encode(p["enc"], ctx), cfg)
        h = jnp.concatenate([z, read(S, z @ p["Wq"], cfg)], -1)
    norm = _ln if cfg.g_ln else (lambda t: t)
    h = jax.nn.gelu(norm(h @ p["W1"] + p["b1"]))
    h = jax.nn.gelu(norm(h @ p["W2"] + p["b2"]))
    return h @ p["W3"] + p["b3"]


# ── plain discriminator (generator positive control) ──────────────────────────

def init_plain_disc(key, cfg: Cfg):
    """Real-vs-fake on the image alone: no context, no state. Same encoder
    shape as the membership discriminator, so the two differ only in the task."""
    k1, k2 = jax.random.split(key)
    return dict(enc=init_encoder(k1, cfg), w_out=_lin(k2, (cfg.d_model, 1)), b_out=jnp.zeros(1))


def plain_logits(p, x, cfg: Cfg):
    """x (B,Q,784) -> logits (B,Q), 1 = real."""
    act = ACTS[cfg.d_act]
    return (act(encode(p["enc"], x, cfg.d_act)) @ p["w_out"] + p["b_out"])[..., 0]
