"""The AR model: token embeddings, a mixer stack, a head over value symbols.

Next-token prediction over the stream `concepts.md` defines. Cross-entropy is
taken at value slots only — label and position tokens are drawn by the episode
generator, so predicting them is unlearnable noise.

`kda_full` selects between two parameterisations of the delta rule, so the
difference is an experiment rather than an assumption:

    kda_full=False   what recall-gen used: linear projections, full-rank decay,
                     no output gate
    kda_full=True    the paper's (arXiv:2510.26692 section 4): short convolution
                     and Swish on q/k/v, low-rank decay projection, head-wise
                     RMSNorm and a data-dependent output gate

**No position encoding.** For the delta rule that follows the source, which
delegates position entirely to KDA (`concepts.md` 5.2). An attention arm would
need one and does not have it yet.
"""

from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp

from .mixers import MIXERS, kda_chunkwise, _rms_headwise


class Cfg(NamedTuple):
    d_model: int = 512
    n_layers: int = 4
    dk: int = 64
    n_heads: int = 8
    ff_mult: int = 4
    vocab: int = 832
    n_values: int = 16
    mixer: str = "kda_wy"     # kda_wy | kda | attn
    kda_full: bool = True     # the paper's neural parameterisation
    conv_k: int = 4           # short-convolution kernel
    chunk: int = 128
    horizon: float = 4096.0   # decay-gate init: memory spans this many tokens

    @property
    def alpha_rank(self) -> int:
        return self.dk        # the paper's rank equals the head dimension


def _decay_bias(H: float) -> float:
    """Bias b with sigmoid(b)^H = 1/e, i.e. the gate's memory spans H tokens.

    Streams here are ~11,000 tokens, far longer than recall-gen's ~20, so the
    default horizon is correspondingly larger. It also has to stay high enough
    that the chunkwise kernel's division by the cumulative decay is safe over
    `chunk` steps — see `workflow.md`.
    """
    a = float(np.exp(-1.0 / H))
    return float(np.log(a / (1.0 - a)))


def init_params(key, cfg: Cfg):
    D, H, DK, R = cfg.d_model, cfg.n_heads, cfg.dk, cfg.alpha_rank
    assert H * DK == D, f"n_heads*dk ({H * DK}) must equal d_model ({D})"
    ks = iter(jax.random.split(key, 8 + cfg.n_layers * 16))
    lin = lambda k, s, g=1.0: jax.random.normal(k, s) * (g / s[0] ** 0.5)

    p = {"tok": lin(next(ks), (cfg.vocab, D)) * 0.5,
         "role": jax.random.normal(next(ks), (3, D)) * 0.02,
         "layers": []}
    hb = _decay_bias(cfg.horizon)
    for _ in range(cfg.n_layers):
        L = dict(
            ln1_g=jnp.ones(D), ln1_b=jnp.zeros(D),
            Wq=lin(next(ks), (D, D)), Wk=lin(next(ks), (D, D)),
            Wv=lin(next(ks), (D, D)), Wo=lin(next(ks), (D, D)),
            Wb=lin(next(ks), (D, H)), bb=jnp.zeros(H),
            ba=jnp.full((D,), hb),
            ln2_g=jnp.ones(D), ln2_b=jnp.zeros(D),
            W1=lin(next(ks), (D, cfg.ff_mult * D)), b1=jnp.zeros(cfg.ff_mult * D),
            W2=lin(next(ks), (cfg.ff_mult * D, D)), b2=jnp.zeros(D))
        if cfg.kda_full:
            # Identity tap on the CURRENT token, so the layer starts equivalent
            # to no convolution and learns the window. Index 0 is the current
            # token (`y[t] = sum_i W[i] x[t-i]`); putting the tap at -1 instead
            # makes the layer a shift, which leaves the first tokens with an
            # exactly-zero q and k.
            c0 = jnp.zeros((cfg.conv_k, D)).at[0].set(1.0)
            L |= dict(conv_q=c0, conv_k=c0, conv_v=c0,
                      Wa_down=lin(next(ks), (D, R)), Wa_up=lin(next(ks), (R, D)),
                      Wg_down=lin(next(ks), (D, R)), Wg_up=lin(next(ks), (R, D)),
                      rms_g=jnp.ones(D))
        else:
            L["Wa"] = lin(next(ks), (D, D), 0.1)
        p["layers"].append(L)
    p["lnf_g"] = jnp.ones(D)
    p["lnf_b"] = jnp.zeros(D)
    p["head_W"] = lin(next(ks), (D, cfg.n_values), 0.1)
    p["head_b"] = jnp.zeros(cfg.n_values)
    return p


def n_params(p) -> int:
    return int(sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(p)))


def ln(x, g, b, eps=1e-5):
    m = x.mean(-1, keepdims=True)
    v = x.var(-1, keepdims=True)
    return g * (x - m) / jnp.sqrt(v + eps) + b


def forward(p, tokens, cfg: Cfg):
    """(B, T) token ids -> (B, T, n_values) logits over value symbols.

    Logits are produced at every position; only value slots are ever scored.
    """
    T = tokens.shape[1]
    x = p["tok"][tokens] + p["role"][jnp.arange(T) % 3]
    mix = MIXERS[cfg.mixer] if cfg.mixer in MIXERS else kda_chunkwise
    for Lp in p["layers"]:
        x = x + mix(ln(x, Lp["ln1_g"], Lp["ln1_b"]), Lp, cfg, chunk=cfg.chunk)
        x = x + (jax.nn.gelu(ln(x, Lp["ln2_g"], Lp["ln2_b"]) @ Lp["W1"] + Lp["b1"])
                 @ Lp["W2"] + Lp["b2"])
    return ln(x, p["lnf_g"], p["lnf_b"]) @ p["head_W"] + p["head_b"]


def loss_and_acc(p, tokens, value_class, is_value, cfg: Cfg):
    """Cross-entropy over value slots, and accuracy on the same.

    A value slot's target is the value token at that slot; the prediction is the
    logits at the PREVIOUS position, which is where next-token prediction puts
    it. Slot i is a value slot when i mod 3 == 2, so its predictor is slot i-1,
    the position token — the model has seen the label and the position and must
    supply the value.
    """
    logits = forward(p, tokens, cfg)[:, :-1]        # predict token i+1 from i
    # `value_class` is `token - val_base`, which is NEGATIVE at label and
    # position slots. Those slots are masked out of the loss, but the gather
    # still runs at every position, and a masked-out NaN survives `nll * m`
    # because NaN * 0 is NaN. Clamp the index and mask with `where`.
    tgt = jnp.clip(value_class[:, 1:], 0, cfg.n_values - 1)
    m = is_value[:, 1:]
    lp = jax.nn.log_softmax(logits, -1)
    nll = -jnp.take_along_axis(lp, tgt[..., None], -1)[..., 0]
    n = jnp.maximum(m.sum(), 1)
    return (jnp.where(m, nll, 0.0).sum() / n,
            jnp.where(m, logits.argmax(-1) == tgt, False).sum() / n)


def cell_accuracy(p, tokens, value_class, masks, cfg: Cfg):
    """Accuracy restricted to each boolean mask in `masks`, shifted to match
    `loss_and_acc`'s next-token alignment."""
    logits = forward(p, tokens, cfg)[:, :-1]
    ok = logits.argmax(-1) == jnp.clip(value_class[:, 1:], 0, cfg.n_values - 1)
    return {k: (jnp.where(m[:, 1:], ok, False).sum()
                / jnp.maximum(m[:, 1:].sum(), 1)) for k, m in masks.items()}
