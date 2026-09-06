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

# MNIST's shape, kept as module constants because the original task is defined
# by them and every existing row in results.jsonl was produced under them. A
# domain that is not 28x28 grey pixels overrides `Cfg.d_in` and supplies its own
# mask; see `lib/domains.py`. Nothing below reads PIX except the MNIST-only
# `augment` and the MNIST-only `row_mask`.
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
    # Flat length of one token. 784 for a 28x28 image, 832 for an 8x8x13 board.
    d_in: int = PIX
    # Which mixer carries context to query. "kda" is the delta-rule state every
    # row in results.jsonl was produced under and stays the default; "softmax"
    # is ordinary quadratic attention over the same tokens. Last field, and
    # defaulted, so every Cfg already written still rebuilds through Cfg(**row).
    mixer: str = "kda"

    @property
    def state_floats(self) -> int:
        return self.n_heads * self.dk * self.dk


def _decay_bias(H: float) -> float:
    """Bias b such that sigmoid(b)^H = 1/e — i.e. memory spans H tokens."""
    a = float(np.exp(-1.0 / H))
    return float(np.log(a / (1.0 - a)))


# ── Model ─────────────────────────────────────────────────────────────────────

def init_params(key, cfg: Cfg):
    D, H, P = cfg.d_model, cfg.dk * cfg.n_heads, cfg.d_in
    assert H == D, f"n_heads*dk ({H}) must equal d_model ({D})"
    g = jax.random.split(key, 6 + cfg.n_layers * 10)
    i = iter(g)
    lin = lambda k, s: jax.random.normal(k, s) * (1.0 / s[0] ** 0.5)

    p = {
        "W_pix": lin(next(i), (P, D)),        # token values -> embedding
        "W_msk": lin(next(i), (P, D)) * 0.1,  # binary "this coordinate is hidden" channel
        "role": jax.random.normal(next(i), (2, D)) * 0.02,   # 0=context, 1=query
        "layers": [],
    }
    hb = _decay_bias(cfg.horizon_mult * cfg.n_tokens)
    for _ in range(cfg.n_layers):
        L = dict(
            ln1_g=jnp.ones(D), ln1_b=jnp.zeros(D),
            Wq=lin(next(i), (D, D)), Wk=lin(next(i), (D, D)), Wv=lin(next(i), (D, D)),
            Wa=lin(next(i), (D, D)) * 0.1, ba=jnp.full((D,), hb),        # forget gate
            Wb=lin(next(i), (D, cfg.n_heads)), bb=jnp.zeros(cfg.n_heads),  # write strength
            Wo=lin(next(i), (D, D)),
            ln2_g=jnp.ones(D), ln2_b=jnp.zeros(D),
            W1=lin(next(i), (D, cfg.ff_mult * D)), b1=jnp.zeros(cfg.ff_mult * D),
            W2=lin(next(i), (cfg.ff_mult * D, D)), b2=jnp.zeros(D),
        )
        if cfg.mixer == "softmax":
            # The decay and write-strength gates belong to the delta rule and
            # have nothing to drive under attention. Dropping them rather than
            # leaving them idle keeps `n_params` an honest count.
            for k in ("Wa", "ba", "Wb", "bb"):
                L.pop(k)
        p["layers"].append(L)
    p["lnf_g"] = jnp.ones(D)
    p["lnf_b"] = jnp.zeros(D)
    p["head_W"] = lin(next(i), (D, P)) * 0.1
    p["head_b"] = jnp.zeros(P)
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


def attn(x, Lp, is_ctx, cfg: Cfg):
    """Ordinary softmax attention over the same tokens, as the reference mixer.

    `kda` compresses sixteen items into a fixed dk x dk state, so a read is a
    linear combination of what was written and two similar keys blur together.
    Attention keeps every item and addresses it by a softmax, which can be
    arbitrarily sharp. That is the comparison: does recall transfer to a
    distribution the weights were not tuned on when addressing is exact?

    The information channel is held identical to `kda`. There, context tokens
    write and query tokens do not, so nothing downstream can read a query token;
    here, query tokens are masked out as KEYS. Causal masking is kept for the
    same reason: a scan cannot see the future.
    """
    B, N, D = x.shape
    H, DK = cfg.n_heads, cfg.dk
    sh = lambda t: t.reshape(B, N, H, DK).transpose(0, 2, 1, 3)     # (B,H,N,DK)

    q, k, v = sh(x @ Lp["Wq"]), sh(x @ Lp["Wk"]), sh(x @ Lp["Wv"])
    s = jnp.einsum("bhnd,bhmd->bhnm", q, k) / DK ** 0.5             # (B,H,N,N)

    causal = jnp.tril(jnp.ones((N, N), bool))[None, None, :, :]
    readable = is_ctx[:, None, None, :] > 0.5                       # context keys only
    keep = causal & readable
    # A row with no readable key would give softmax a row of -inf. Token 0 is a
    # context token in every episode this project builds, so no row is empty;
    # the self-key is added anyway to keep that true for any future layout.
    keep = keep | jnp.eye(N, dtype=bool)[None, None, :, :]
    s = jnp.where(keep, s, -jnp.inf)
    o = jnp.einsum("bhnm,bhmd->bhnd", jax.nn.softmax(s, axis=-1), v)
    return o.transpose(0, 2, 1, 3).reshape(B, N, D) @ Lp["Wo"]


MIXERS = {"kda": kda, "softmax": attn}


def forward(p, pix, msk, is_ctx, cfg: Cfg):
    """pix,msk: (B,N,784) float. is_ctx: (B,N) 1.0 for context tokens.

    Returns per-token pixel predictions (B,N,784) in [0,1].
    """
    x = pix @ p["W_pix"] + msk @ p["W_msk"]
    x = x + jnp.where(is_ctx[..., None] > 0.5, p["role"][0], p["role"][1])
    mix = MIXERS[cfg.mixer]
    for Lp in p["layers"]:
        x = x + mix(ln(x, Lp["ln1_g"], Lp["ln1_b"]), Lp, is_ctx, cfg)
        x = x + (jax.nn.gelu(ln(x, Lp["ln2_g"], Lp["ln2_b"]) @ Lp["W1"] + Lp["b1"])
                 @ Lp["W2"] + Lp["b2"])
    return jax.nn.sigmoid(ln(x, p["lnf_g"], p["lnf_b"]) @ p["head_W"] + p["head_b"])


# ── Episode assembly ──────────────────────────────────────────────────────────

def row_mask(mask_rows: int) -> np.ndarray:
    """1.0 on HIDDEN pixels — the bottom `mask_rows` rows of a 28x28 image.

    The MNIST / Fashion-MNIST mask. `lib/domains.mask_vector` is the general
    form and returns exactly this vector for those two domains.
    """
    m = np.zeros((SIDE, SIDE), np.float32)
    m[SIDE - mask_rows:, :] = 1.0
    return m.reshape(PIX)


def build_tokens(ctx, qry_full, mask):
    """ctx (B,M,P) full items; qry_full (B,Q,P) the true target items.

    `mask` is a fixed (P,) vector or a per-episode (B,1,P) one; both broadcast.

    Returns (pix, msk, is_ctx) with N = M+Q tokens. Query tokens carry only the
    VISIBLE part of their item, and the mask channel says which coordinates were
    removed — so a model trained on varied masks can be told, at test time, that
    a different set is missing.
    """
    B, M, P = ctx.shape
    Q = qry_full.shape[1]
    qry_vis = qry_full * (1.0 - mask)
    pix = jnp.concatenate([ctx, qry_vis], axis=1)
    msk = jnp.concatenate([jnp.zeros_like(ctx),
                           jnp.broadcast_to(mask, (B, Q, P))], axis=1)
    is_ctx = jnp.concatenate([jnp.ones((B, M)), jnp.zeros((B, Q))], axis=1)
    return pix, msk, is_ctx


def augment(key, x, max_shift=2.0, max_rot=15.0, scale=(0.9, 1.1),
            elastic=1.5, grid=4):
    """Independent random warp of each image in (N, 784) — the A2 instrument.

    MNIST-shaped only: it resamples on a 28x28 grid. `Run.augment_train` is
    asserted off for every other domain.

    A random affine (rotation, isotropic scale, translation) plus a
    low-frequency elastic displacement, sampled per image and resampled
    bilinearly. Applied to the training pool it makes the pool effectively
    infinite: no image is ever presented twice, so a model cannot reach a low
    recall loss by memorising the pool and must key on the episode's context.

    Coordinates are clipped rather than zero-padded at the border; MNIST margins
    are black, so the two agree, and clipping keeps the gather in bounds.
    """
    N = x.shape[0]
    im = x.reshape(N, SIDE, SIDE)
    k_rot, k_scale, k_shift, k_el = jax.random.split(key, 4)
    c = (SIDE - 1) / 2.0

    th = jax.random.uniform(k_rot, (N, 1, 1), minval=-max_rot, maxval=max_rot) * jnp.pi / 180.0
    s = jax.random.uniform(k_scale, (N, 1, 1), minval=scale[0], maxval=scale[1])
    sh = jax.random.uniform(k_shift, (N, 2, 1, 1), minval=-max_shift, maxval=max_shift)

    yy, xx = jnp.meshgrid(jnp.arange(SIDE) - c, jnp.arange(SIDE) - c, indexing="ij")
    cos, sin = jnp.cos(th), jnp.sin(th)
    xs = (cos * xx + sin * yy) / s + c + sh[:, 0]
    ys = (-sin * xx + cos * yy) / s + c + sh[:, 1]

    d = jax.random.normal(k_el, (N, 2, grid, grid))
    d = jax.image.resize(d, (N, 2, SIDE, SIDE), "bilinear") * elastic
    xs, ys = xs + d[:, 0], ys + d[:, 1]

    x0 = jnp.floor(xs)
    y0 = jnp.floor(ys)
    wx, wy = xs - x0, ys - y0
    ix = lambda t: jnp.clip(t.astype(jnp.int32), 0, SIDE - 1)
    x0i, x1i, y0i, y1i = ix(x0), ix(x0 + 1), ix(y0), ix(y0 + 1)
    n = jnp.arange(N)[:, None, None]
    g = lambda yi, xi: im[n, yi, xi]
    out = ((1 - wy) * ((1 - wx) * g(y0i, x0i) + wx * g(y0i, x1i))
           + wy * ((1 - wx) * g(y1i, x0i) + wx * g(y1i, x1i)))
    return out.reshape(N, PIX)


def predict(p, ctx, qry_full, mask, cfg: Cfg):
    """Model prediction for the Q query tokens only: (B,Q,784)."""
    M = ctx.shape[1]
    pix, msk, is_ctx = build_tokens(ctx, qry_full, mask)
    return forward(p, pix, msk, is_ctx, cfg)[:, M:, :]


def masked_mse(pred, tgt, mask):
    """MSE over hidden coordinates only, averaged over batch and queries.

    `mask` is either a fixed (P,) vector or a per-episode (B,1,P) one. The
    denominator is taken per episode rather than as a scalar, because a random
    mask hides a different NUMBER of coordinates each time and a shared
    denominator would then weight episodes by how much they happened to hide.
    For a fixed mask the two forms are algebraically identical.
    """
    num = ((pred - tgt) ** 2 * mask).sum(-1)                 # (B,Q)
    return (num / jnp.sum(mask, axis=-1)).mean()


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


def nn_baseline(ctx, qry_full, mask, vis=None):
    """Best pure-look-up answer: the context image whose VISIBLE part is closest
    to the query's visible part; its hidden part is the prediction.

    This is the ceiling for a model that only retrieves. Returns (mse, idx).

    `vis` marks coordinates that are real AND shown. It defaults to `1 - mask`,
    which is right for an unpadded domain and would count padding as visible for
    a padded one; `lib/domains.visible_vector` supplies the correct vector.
    """
    vis = (1.0 - mask) if vis is None else vis
    d = (((qry_full[:, :, None, :] - ctx[:, None, :, :]) ** 2) * vis).sum(-1)  # (B,Q,M)
    idx = jnp.argmin(d, axis=-1)
    pick = jnp.take_along_axis(ctx, idx[..., None], axis=1)  # (B,Q,784)
    return masked_mse(pick, qry_full, mask), idx


def psnr_from_mse(mse):
    return float(-10.0 * np.log10(max(float(mse), 1e-12)))
