"""
Token-level attention over a flat bag of `(pos, value, ref)` tokens.

Ported from `projects/universal-ar/experiments39.py` with the shape constants
lifted out of module scope so a `Spec` drives them. The architecture is
deliberately unchanged from the one that hit chance on MNIST: the point of this
project is to vary the *data*, not the model, so the two are comparable.

Efficiency choices carried over from universal-ar, each for a measured reason:

  * **One-hot matmul instead of a gather** for the embedding tables. Omniglot is
    mostly background, so a gather's backward scatter contends on a handful of
    rows and serialises — the same pathology that cost 345× on MNIST.
  * **`jax.checkpoint` per layer**, because episodes run to a few thousand
    tokens and the attention matrices dominate memory.
  * **No causal mask.** The bag is a set; `pos` is a field, not sequence order.
"""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .tasks import Spec


def init_params(key, spec: Spec, d_model: int, n_layers: int) -> dict:
    keys = iter(jax.random.split(key, 4 + n_layers * 4))
    emb = lambda k, shape: jax.random.normal(k, shape) * 0.02
    lin = lambda k, shape: jax.random.normal(k, shape) * (1.0 / shape[0] ** 0.5)

    p: dict[str, Any] = {
        "pos_emb": emb(next(keys), (spec.n_pos, d_model)),
        "val_emb": emb(next(keys), (spec.n_val, d_model)),
        "ref_emb": emb(next(keys), (spec.v_refs, d_model)),
        "layers": [],
    }
    for _ in range(n_layers):
        p["layers"].append({
            "ln1_g": jnp.ones(d_model), "ln1_b": jnp.zeros(d_model),
            "Wqkv": lin(next(keys), (d_model, 3 * d_model)),
            "Wo": lin(next(keys), (d_model, d_model)),
            "ln2_g": jnp.ones(d_model), "ln2_b": jnp.zeros(d_model),
            "W1": lin(next(keys), (d_model, 4 * d_model)), "b1": jnp.zeros(4 * d_model),
            "W2": lin(next(keys), (4 * d_model, d_model)), "b2": jnp.zeros(d_model),
        })
    p["lnf_g"] = jnp.ones(d_model)
    p["lnf_b"] = jnp.zeros(d_model)
    p["head_W"] = lin(next(keys), (d_model, spec.n_content))
    p["head_b"] = jnp.zeros(spec.n_content)
    if spec.label_field:
        # Keyed off a fold_in rather than another `split` draw, so that adding
        # this table does not shift every other key and silently change what
        # `label_field=False` runs (exp1/exp2) initialise to.
        p["lab_emb"] = jax.random.normal(
            jax.random.fold_in(key, 991), (spec.n_lab, d_model)
        ) * 0.02
    return p


def n_params(p) -> int:
    return int(sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(p)))


def layer_norm(x, g, b, eps: float = 1e-5):
    m = x.mean(-1, keepdims=True)
    v = x.var(-1, keepdims=True)
    return g * (x - m) / jnp.sqrt(v + eps) + b


def attention(x, lp, head_dim: int):
    B, N, D = x.shape
    H = D // head_dim
    q, k, v = jnp.split(x @ lp["Wqkv"], 3, -1)
    shape = lambda t: t.reshape(B, N, H, head_dim).transpose(0, 2, 1, 3)
    q, k, v = shape(q), shape(k), shape(v)
    a = jax.nn.softmax(jnp.einsum("bhid,bhjd->bhij", q, k) / head_dim ** 0.5, -1)
    out = jnp.einsum("bhij,bhjd->bhid", a, v)
    return out.transpose(0, 2, 1, 3).reshape(B, N, D) @ lp["Wo"]


def onehot_mm(ids, table, n: int):
    """Embedding lookup as a one-hot matmul — see the module docstring."""
    return jnp.einsum("bnk,kd->bnd", jax.nn.one_hot(ids, n, dtype=jnp.float32), table)


def forward(p, pos, val, ref, lab, spec: Spec, head_dim: int):
    """Logits over the unified value vocabulary, for every token.

    `lab` is the label field (see lib/tasks.py). It is embedded only when
    `spec.label_field` is set, so a run with it off computes exactly what
    exp1/exp2 computed before the field existed.
    """
    x = (
        onehot_mm(pos, p["pos_emb"], spec.n_pos)
        + onehot_mm(val, p["val_emb"], spec.n_val)
        + onehot_mm(ref, p["ref_emb"], spec.v_refs)
    )
    if spec.label_field:
        x = x + onehot_mm(lab, p["lab_emb"], spec.n_lab)

    @jax.checkpoint
    def block(lp, h):
        h = h + attention(layer_norm(h, lp["ln1_g"], lp["ln1_b"]), lp, head_dim)
        mlp = jax.nn.gelu(layer_norm(h, lp["ln2_g"], lp["ln2_b"]) @ lp["W1"] + lp["b1"])
        return h + (mlp @ lp["W2"] + lp["b2"])

    for lp in p["layers"]:
        x = block(lp, x)
    return layer_norm(x, p["lnf_g"], p["lnf_b"]) @ p["head_W"] + p["head_b"]


def slot_accuracy(logits, tgt, is_query, spec: Spec):
    """N-way accuracy: argmax restricted to the episode's label slots.

    The open-vocabulary argmax spans pixel bins as well, so an untrained head
    can answer a label query with a pixel bin — which scores zero and conflates
    "wrong class" with "did not emit a label at all". Restricting to the
    `n_way` slots measures the actual N-way decision, with chance `1/n_way`.
    """
    slot_logits = logits[..., spec.n_bins: spec.n_bins + spec.n_way]
    pred = jnp.argmax(slot_logits, -1) + spec.n_bins
    correct = (pred == tgt).astype(jnp.float32)
    return (correct * is_query).sum() / (is_query.sum() + 1e-6)


def open_accuracy(logits, tgt, is_query):
    """Same decision, but over the whole vocabulary — the stricter reading."""
    correct = (jnp.argmax(logits, -1) == tgt).astype(jnp.float32)
    return (correct * is_query).sum() / (is_query.sum() + 1e-6)
