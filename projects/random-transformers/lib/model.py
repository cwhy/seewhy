"""GPT-2 style decoder-only transformer in JAX, with a frozen / trainable split.

The whole point of this project is that *most of this network never moves*. So
the parameters live in one flat dict keyed by strings like ``block0/attn/W_qkv``,
and :func:`split_params` cuts that dict into a trainable half and a frozen half
according to a training mode. The frozen half is passed to the loss as a
non-differentiated argument, so XLA never even builds the weight-gradient
matmuls for it.

Initialisation follows Radford et al. (GPT-2) as restated by the paper:
feed-forward weights ~ N(0, (0.02/sqrt(2 * n_layer))^2), every other weight
matrix ~ N(0, 0.02^2), biases zero, layer-norm affines identity.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

# ── parameter layout ──────────────────────────────────────────────────────────
#
# E_token   (v, d)     token embedding
# E_pos     (n, d)     positional embedding
# U         (v, d)     unembedding (weight tying is disabled, as in the paper)
# blockI/ln1/{g,b}     (d,)
# blockI/attn/W_qkv    (d, 3d)      fused query/key/value
# blockI/attn/b_qkv    (3d,)
# blockI/attn/W_o      (d, d)
# blockI/attn/b_o      (d,)
# blockI/ln2/{g,b}     (d,)
# blockI/mlp/W_fc      (d, 4d)      <- "feed-forward", gets the scaled init
# blockI/mlp/b_fc      (4d,)
# blockI/mlp/W_proj    (4d, d)      <- ditto
# blockI/mlp/b_proj    (d,)
# ln_f/{g,b}           (d,)

EMB_KEYS = ("E_token", "E_pos", "U")

#: Which parameters each training mode optimises. Everything else is frozen at
#: its random initialisation. ``full`` is the paper's "normal" transformer,
#: ``random`` its "random" transformer; the remaining three are the Table 2
#: component ablation.
TRAINABLE = {
    "full":     None,                      # None == everything
    "random":   ("E_token", "E_pos", "U"),
    "u_only":   ("U",),
    "e_only":   ("E_token", "E_pos"),
    "etoken_u": ("E_token", "U"),
}


def init_params(key, *, vocab: int, n_ctx: int, d: int, n_layer: int, n_head: int) -> dict:
    """Random initialisation. Returns the flat parameter dict described above."""
    assert d % n_head == 0, f"width {d} not divisible by {n_head} heads"
    std_ff = 0.02 / math.sqrt(2 * n_layer)
    p: dict = {}

    def normal(k, shape, std):
        return jax.random.normal(k, shape, dtype=jnp.float32) * std

    keys = jax.random.split(key, 3 + 5 * n_layer)
    ki = iter(keys)

    p["E_token"] = normal(next(ki), (vocab, d), 0.02)
    p["E_pos"]   = normal(next(ki), (n_ctx, d), 0.02)
    p["U"]       = normal(next(ki), (vocab, d), 0.02)

    for i in range(n_layer):
        b = f"block{i}"
        p[f"{b}/ln1/g"] = jnp.ones((d,));  p[f"{b}/ln1/b"] = jnp.zeros((d,))
        p[f"{b}/ln2/g"] = jnp.ones((d,));  p[f"{b}/ln2/b"] = jnp.zeros((d,))
        # query / key / value and the attention output projection are "other
        # weight matrices" in the paper's wording, so they take the flat 0.02.
        p[f"{b}/attn/W_qkv"]  = normal(next(ki), (d, 3 * d), 0.02)
        p[f"{b}/attn/b_qkv"]  = jnp.zeros((3 * d,))
        p[f"{b}/attn/W_o"]    = normal(next(ki), (d, d), 0.02)
        p[f"{b}/attn/b_o"]    = jnp.zeros((d,))
        p[f"{b}/mlp/W_fc"]    = normal(next(ki), (d, 4 * d), std_ff)
        p[f"{b}/mlp/b_fc"]    = jnp.zeros((4 * d,))
        p[f"{b}/mlp/W_proj"]  = normal(next(ki), (4 * d, d), std_ff)
        p[f"{b}/mlp/b_proj"]  = jnp.zeros((d,))

    p["ln_f/g"] = jnp.ones((d,));  p["ln_f/b"] = jnp.zeros((d,))
    return p


def split_params(params: dict, mode: str) -> tuple[dict, dict]:
    """Cut ``params`` into ``(trainable, frozen)`` for a training mode."""
    keep = TRAINABLE[mode]
    if keep is None:
        return dict(params), {}
    trainable = {k: v for k, v in params.items() if k in keep}
    frozen    = {k: v for k, v in params.items() if k not in keep}
    missing = set(keep) - set(trainable)
    assert not missing, f"mode {mode!r} wants {missing}, not in params"
    return trainable, frozen


def n_params(params: dict) -> int:
    return int(sum(v.size for v in params.values()))


# ── forward ───────────────────────────────────────────────────────────────────

def _layer_norm(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) * jax.lax.rsqrt(var + eps) * g + b


def _gelu(x):
    # GPT-2's tanh approximation, which is what HuggingFace's "gelu_new" uses.
    return 0.5 * x * (1 + jnp.tanh(0.7978845608028654 * (x + 0.044715 * x**3)))


def _attention(h, p, prefix, n_head, causal_mask):
    L, d = h.shape[-2], h.shape[-1]
    qkv = h @ p[f"{prefix}/attn/W_qkv"] + p[f"{prefix}/attn/b_qkv"]
    q, k, v = jnp.split(qkv, 3, axis=-1)
    dh = d // n_head
    # (..., L, H, dh) -> (..., H, L, dh)
    shape = h.shape[:-1] + (n_head, dh)
    q = jnp.swapaxes(q.reshape(shape), -3, -2)
    k = jnp.swapaxes(k.reshape(shape), -3, -2)
    v = jnp.swapaxes(v.reshape(shape), -3, -2)

    logits = q @ jnp.swapaxes(k, -1, -2) / math.sqrt(dh)
    logits = jnp.where(causal_mask[:L, :L], logits, -1e30)
    attn = jax.nn.softmax(logits, axis=-1)
    out = attn @ v
    out = jnp.swapaxes(out, -3, -2).reshape(h.shape[:-1] + (d,))
    return out @ p[f"{prefix}/attn/W_o"] + p[f"{prefix}/attn/b_o"], attn


def forward(params: dict, toks, *, n_layer: int, n_head: int, return_hidden: bool = False):
    """Logits for every position. ``toks`` is ``(B, L)`` of int32.

    With ``return_hidden``, also returns a dict of activations (post-embedding
    and post-block) and the attention maps — this is what the subspace analysis
    in §6 of the paper reads.
    """
    L = toks.shape[-1]
    causal_mask = jnp.tril(jnp.ones((L, L), dtype=bool))

    h = params["E_token"][toks] + params["E_pos"][:L]
    hidden = {"emb": h}
    attns = []

    for i in range(n_layer):
        b = f"block{i}"
        a, attn = _attention(_layer_norm(h, params[f"{b}/ln1/g"], params[f"{b}/ln1/b"]),
                             params, b, n_head, causal_mask)
        h = h + a
        m = _layer_norm(h, params[f"{b}/ln2/g"], params[f"{b}/ln2/b"])
        m = _gelu(m @ params[f"{b}/mlp/W_fc"] + params[f"{b}/mlp/b_fc"])
        h = h + m @ params[f"{b}/mlp/W_proj"] + params[f"{b}/mlp/b_proj"]
        hidden[f"L{i + 1}"] = h
        attns.append(attn)

    h = _layer_norm(h, params["ln_f/g"], params["ln_f/b"])
    logits = h @ params["U"].T

    if return_hidden:
        return logits, hidden, attns
    return logits


# ── loss ──────────────────────────────────────────────────────────────────────

def loss_fn(trainable: dict, frozen: dict, toks, mask, *, n_layer: int, n_head: int):
    """Mean cross-entropy over the positions ``mask`` marks as scored.

    ``mask[b, t] == 1`` means "the model's prediction at position ``t`` should be
    ``toks[b, t + 1]``". Everything else — prompt tokens, padding — is ignored,
    which is what makes accuracy comparable to the paper's per-task metric.
    """
    params = {**trainable, **frozen}
    logits = forward(params, toks, n_layer=n_layer, n_head=n_head)
    logp = jax.nn.log_softmax(logits[:, :-1].astype(jnp.float32), axis=-1)
    tgt = toks[:, 1:]
    m = mask[:, :-1]
    picked = jnp.take_along_axis(logp, tgt[..., None], axis=-1)[..., 0]
    return -(picked * m).sum() / jnp.maximum(m.sum(), 1)


def accuracy(params: dict, toks, mask, *, n_layer: int, n_head: int):
    """``(sequence_exact_match, token_accuracy)`` over the scored positions.

    Sequence accuracy is the strict metric — every scored token in a sequence
    must be right — and is what the paper's tables report.
    """
    logits = forward(params, toks, n_layer=n_layer, n_head=n_head)
    pred = logits[:, :-1].argmax(-1)
    tgt = toks[:, 1:]
    m = mask[:, :-1]
    hit = (pred == tgt) * m
    tok_acc = hit.sum() / jnp.maximum(m.sum(), 1)
    seq_acc = (hit.sum(-1) == m.sum(-1)).mean()
    return seq_acc, tok_acc
