"""Recall-Gen core machinery: episode construction, KDA model, metrics.

Kept out of the experiment files because every experiment in this project is the
same model on the same task with one thing changed; the machinery is the
control. Hyperparameters stay in the experiment files.
"""

from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp
import optax

PIX = 784
SIDE = 28


# ── Config ────────────────────────────────────────────────────────────────────

class Cfg(NamedTuple):
    """Static (hashable) model shape — safe to pass through `static_argnums`."""
    d_model: int = 256
    n_layers: int = 4
    dk: int = 64            # per-head key AND value dim; state is dk x dk per head
    n_heads: int = 4
    ff_mult: int = 4
    n_tokens: int = 20      # M + Q, only used to set the decay-horizon init
    horizon_mult: float = 8.0

    @property
    def state_floats(self) -> int:
        return self.n_heads * self.dk * self.dk


def _decay_bias(H: float) -> float:
    """Bias b such that sigmoid(b)^H = 1/e — i.e. memory spans H tokens."""
    a = float(np.exp(-1.0 / H))
    return float(np.log(a / (1.0 - a)))


# ── Model ─────────────────────────────────────────────────────────────────────

def init_params(key, cfg: Cfg):
    D, H = cfg.d_model, cfg.dk * cfg.n_heads
    assert H == D, f"n_heads*dk ({H}) must equal d_model ({D})"
    g = jax.random.split(key, 6 + cfg.n_layers * 10)
    i = iter(g)
    lin = lambda k, s: jax.random.normal(k, s) * (1.0 / s[0] ** 0.5)

    p = {
        "W_pix": lin(next(i), (PIX, D)),      # pixel values -> embedding
        "W_msk": lin(next(i), (PIX, D)) * 0.1,  # binary "this pixel is hidden" channel
        "role": jax.random.normal(next(i), (2, D)) * 0.02,   # 0=context, 1=query
        "layers": [],
    }
    hb = _decay_bias(cfg.horizon_mult * cfg.n_tokens)
    for _ in range(cfg.n_layers):
        p["layers"].append(dict(
            ln1_g=jnp.ones(D), ln1_b=jnp.zeros(D),
            Wq=lin(next(i), (D, D)), Wk=lin(next(i), (D, D)), Wv=lin(next(i), (D, D)),
            Wa=lin(next(i), (D, D)) * 0.1, ba=jnp.full((D,), hb),        # forget gate
            Wb=lin(next(i), (D, cfg.n_heads)), bb=jnp.zeros(cfg.n_heads),  # write strength
            Wo=lin(next(i), (D, D)),
            ln2_g=jnp.ones(D), ln2_b=jnp.zeros(D),
            W1=lin(next(i), (D, cfg.ff_mult * D)), b1=jnp.zeros(cfg.ff_mult * D),
            W2=lin(next(i), (cfg.ff_mult * D, D)), b2=jnp.zeros(D),
        ))
    p["lnf_g"] = jnp.ones(D)
    p["lnf_b"] = jnp.zeros(D)
    p["head_W"] = lin(next(i), (D, PIX)) * 0.1
    p["head_b"] = jnp.zeros(PIX)
    return p


def n_params(p) -> int:
    return int(sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(p)))


def ln(x, g, b, eps=1e-5):
    m = x.mean(-1, keepdims=True)
    v = x.var(-1, keepdims=True)
    return g * (x - m) / jnp.sqrt(v + eps) + b


def kda(x, Lp, is_ctx, cfg: Cfg):
    """Kimi Delta Attention — a matrix-valued memory written by the delta rule.

        forget   S~ = S . Diag(alpha_t)     per-channel decay
        predict  vhat = S~ k_t              what is currently stored at this key
        correct  e = beta_t (v_t - vhat)
        write    S = S~ + e k_t^T
        read     o_t = S q_t / sqrt(dk)

    Context tokens WRITE; query tokens never write (beta gated to 0) and never
    decay (alpha gated to 1). Every token reads the COMPLETED state, so `S` is
    the only channel from context to query.
    """
    B, N, D = x.shape
    H, DK = cfg.n_heads, cfg.dk
    sh = lambda t: t.reshape(B, N, H, DK).transpose(0, 2, 1, 3)     # (B,H,N,DK)

    q = sh(x @ Lp["Wq"])
    k = sh(x @ Lp["Wk"])
    v = sh(x @ Lp["Wv"])
    q = q / (jnp.linalg.norm(q, axis=-1, keepdims=True) + 1e-6)     # DeltaNet convention
    k = k / (jnp.linalg.norm(k, axis=-1, keepdims=True) + 1e-6)
    alpha = jax.nn.sigmoid(sh(x @ Lp["Wa"] + Lp["ba"]))             # (B,H,N,DK)
    beta = jax.nn.sigmoid(x @ Lp["Wb"] + Lp["bb"]).transpose(0, 2, 1)  # (B,H,N)

    gate = is_ctx[:, None, :]                                       # (B,1,N)
    alpha = alpha * gate[..., None] + (1.0 - gate[..., None])
    beta = beta * gate

    def step(S, t):
        a_t, k_t, v_t, b_t = t
        S = S * a_t[:, :, None, :]
        vhat = jnp.einsum("bhvk,bhk->bhv", S, k_t)
        e = b_t[..., None] * (v_t - vhat)
        return S + jnp.einsum("bhv,bhk->bhvk", e, k_t), None

    seq = (alpha.transpose(2, 0, 1, 3), k.transpose(2, 0, 1, 3),
           v.transpose(2, 0, 1, 3), beta.transpose(2, 0, 1))
    S, _ = jax.lax.scan(step, jnp.zeros((B, H, DK, DK)), seq)
    o = jnp.einsum("bhvk,bhnk->bhnv", S, q) / DK ** 0.5
    return o.transpose(0, 2, 1, 3).reshape(B, N, D) @ Lp["Wo"]


def forward(p, pix, msk, is_ctx, cfg: Cfg):
    """pix,msk: (B,N,784) float. is_ctx: (B,N) 1.0 for context tokens.

    Returns per-token pixel predictions (B,N,784) in [0,1].
    """
    x = pix @ p["W_pix"] + msk @ p["W_msk"]
    x = x + jnp.where(is_ctx[..., None] > 0.5, p["role"][0], p["role"][1])
    for Lp in p["layers"]:
        x = x + kda(ln(x, Lp["ln1_g"], Lp["ln1_b"]), Lp, is_ctx, cfg)
        x = x + (jax.nn.gelu(ln(x, Lp["ln2_g"], Lp["ln2_b"]) @ Lp["W1"] + Lp["b1"])
                 @ Lp["W2"] + Lp["b2"])
    return jax.nn.sigmoid(ln(x, p["lnf_g"], p["lnf_b"]) @ p["head_W"] + p["head_b"])


# ── Episode assembly ──────────────────────────────────────────────────────────

def row_mask(mask_rows: int) -> np.ndarray:
    """1.0 on HIDDEN pixels — the bottom `mask_rows` rows of the image."""
    m = np.zeros((SIDE, SIDE), np.float32)
    m[SIDE - mask_rows:, :] = 1.0
    return m.reshape(PIX)


def build_tokens(ctx, qry_full, mask):
    """ctx (B,M,784) full images; qry_full (B,Q,784) the true target images.

    Returns (pix, msk, is_ctx) with N = M+Q tokens. Query tokens carry only the
    VISIBLE part of their image.
    """
    B, M, _ = ctx.shape
    Q = qry_full.shape[1]
    qry_vis = qry_full * (1.0 - mask)
    pix = jnp.concatenate([ctx, qry_vis], axis=1)
    msk = jnp.concatenate([jnp.zeros_like(ctx),
                           jnp.broadcast_to(mask, (B, Q, PIX))], axis=1)
    is_ctx = jnp.concatenate([jnp.ones((B, M)), jnp.zeros((B, Q))], axis=1)
    return pix, msk, is_ctx


def predict(p, ctx, qry_full, mask, cfg: Cfg):
    """Model prediction for the Q query tokens only: (B,Q,784)."""
    M = ctx.shape[1]
    pix, msk, is_ctx = build_tokens(ctx, qry_full, mask)
    return forward(p, pix, msk, is_ctx, cfg)[:, M:, :]


def masked_mse(pred, tgt, mask):
    """MSE over hidden pixels only, averaged over batch and queries."""
    return ((pred - tgt) ** 2 * mask).sum(-1).mean() / mask.sum()


def loss_fn(p, ctx, qry_full, mask, cfg: Cfg):
    return masked_mse(predict(p, ctx, qry_full, mask, cfg), qry_full, mask)


# ── Metrics ───────────────────────────────────────────────────────────────────

def identification(pred, ctx, tgt_idx, mask):
    """Which context image does the model's output most resemble?

    Distance is taken on the HIDDEN pixels only, so a model that merely copies
    the visible part of the query cannot score. Returns (acc, argmin indices).
    Chance = 1/M.
    """
    d = (((pred[:, :, None, :] - ctx[:, None, :, :]) ** 2) * mask).sum(-1)  # (B,Q,M)
    nn = jnp.argmin(d, axis=-1)
    return (nn == tgt_idx).mean(), nn


def nn_baseline(ctx, qry_full, mask):
    """Best pure-look-up answer: the context image whose VISIBLE part is closest
    to the query's visible part; its hidden part is the prediction.

    This is the ceiling for a model that only retrieves. Returns (mse, idx).
    """
    vis = 1.0 - mask
    d = (((qry_full[:, :, None, :] - ctx[:, None, :, :]) ** 2) * vis).sum(-1)  # (B,Q,M)
    idx = jnp.argmin(d, axis=-1)
    pick = jnp.take_along_axis(ctx, idx[..., None], axis=1)  # (B,Q,784)
    return masked_mse(pick, qry_full, mask), idx


def psnr_from_mse(mse):
    return float(-10.0 * np.log10(max(float(mse), 1e-12)))
