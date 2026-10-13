"""The transformer DUST trains, ported to JAX from github.com/qlabs-eng/dust (model.py).

A decoder-only GPT of the modded-nanoGPT kind. Everything that differs from a
textbook GPT is here because the paper trains this exact network, and a
different network would make the comparison with its published numbers moot:

    * pre-norm blocks, RMS norm without a learned gain, no biases anywhere;
    * rotary embeddings, then RMS-normalised queries and keys (QK-norm);
    * a squared-ReLU MLP of width 4d;
    * causal attention with a left window: 1024 tokens in blocks 0-2 and 4-6,
      the full 2048 in blocks 3 and 7 (pattern SSSL, last block forced L);
    * a *value embedding* on every other block, counting back from the last: a
      second token embedding added to that block's values, scaled per head by
      2*sigmoid(gate) where the gate reads the first 32 channels of the block input;
    * the stream entering block i is resid[i]*x + x0[i]*x0, a learned mix of the
      previous block's output and the normalised token embedding (2L scalars);
    * an untied output head with logits capped as 15*tanh(logit/15).

Weights are stored float32 and cast to the compute dtype (bfloat16) once per
step. The reference stores the two embedding tables in bfloat16; keeping an f32
master copy is the one deliberate departure, and it only matters for the
rounding of very small embedding updates.

Parameters are a plain pytree so the same dict shape carries weights, gradients
and optimiser state:

    {"wte": [V,d], "head": [V,d], "resid": [L], "x0": [L],
     "blocks": [{"q","k","v","proj": [d,d], "fc": [4d,d], "mproj": [d,4d],
                 "gate": [H,32] and "ve": [V,d] on value-embedding blocks}, ...]}

Linear weights are [out, in], as in torch, so `y = x @ W.T`.
"""

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

BF16 = jnp.bfloat16
F32 = jnp.float32
GATE_CHANNELS = 32
ROPE_BASE = 10000.0


@dataclass(frozen=True)
class Cfg:
    n_layer: int = 8
    n_embd: int = 512
    vocab: int = 4096
    seq: int = 2048
    head_dim: int = 64
    window_pattern: str = "SSSL"
    softcap: float = 15.0

    @property
    def n_head(self) -> int:
        return self.n_embd // self.head_dim

    @property
    def windows(self) -> tuple[int, ...]:
        """Left window per block: S is half the sequence, L the whole of it."""
        w = [self.seq // 2 if self.window_pattern[i % len(self.window_pattern)] == "S" else self.seq
             for i in range(self.n_layer)]
        w[-1] = self.seq
        return tuple(w)

    def has_ve(self, layer: int) -> bool:
        """Value embeddings sit on every other block, counting back from the last."""
        return layer % 2 == (self.n_layer - 1) % 2


# The four sizes of the paper's overparameterisation study (Section 4.1).
SIZES = {
    "L2d128": Cfg(n_layer=2, n_embd=128),
    "L4d256": Cfg(n_layer=4, n_embd=256),
    "L8d512": Cfg(n_layer=8, n_embd=512),
    "L16d1024": Cfg(n_layer=16, n_embd=1024),
}


# ── parameters ───────────────────────────────────────────────────────────────

def init_params(cfg: Cfg, key: Array) -> dict:
    """The authors' initialisation, including their override of the c_proj zeros.

    q, k, v, c_fc: uniform(+-sqrt(3/d)). Both c_proj layers: uniform with std
    1.5/sqrt(fan_in) -- the reference's `make_model` replaces init_weights'
    zeros with this. wte: N(0,1). head: N(0, 1e-3). Value embeddings:
    uniform(+-sqrt(3/d)). Gates start at zero, so 2*sigmoid(0) = 1.
    """
    d, V, L = cfg.n_embd, cfg.vocab, cfg.n_layer
    s = 3 ** 0.5 * d ** -0.5
    ks = iter(jax.random.split(key, 4 + 8 * L))

    def uni(shape, scale):
        return jax.random.uniform(next(ks), shape, F32, -scale, scale)

    def proj_scale(fan_in):
        return 1.5 * (3 / fan_in) ** 0.5

    blocks = []
    for l in range(L):
        b = {
            "q": uni((d, d), s), "k": uni((d, d), s), "v": uni((d, d), s),
            "proj": uni((d, d), proj_scale(d)),
            "fc": uni((4 * d, d), s),
            "mproj": uni((d, 4 * d), proj_scale(4 * d)),
        }
        if cfg.has_ve(l):
            b["gate"] = jnp.zeros((cfg.n_head, GATE_CHANNELS), F32)
            b["ve"] = uni((V, d), s)
        blocks.append(b)
    return {
        "wte": jax.random.normal(next(ks), (V, d), F32),
        "head": 1e-3 * jax.random.normal(next(ks), (V, d), F32),
        "resid": jnp.ones((L,), F32),
        "x0": jnp.full((L,), 0.1, F32),
        "blocks": blocks,
    }


def n_params(params) -> int:
    return int(sum(x.size for x in jax.tree_util.tree_leaves(params)))


def cast(params, dtype=BF16):
    """Compute-dtype copy of the weights. Do this once per step, not per forward."""
    return jax.tree_util.tree_map(lambda x: x.astype(dtype), params)


# ── pieces ───────────────────────────────────────────────────────────────────

def rms(x: Array) -> Array:
    """RMS norm over the last axis with no gain. Statistics in f32, result in x.dtype."""
    x32 = x.astype(F32)
    return (x32 * jax.lax.rsqrt(jnp.mean(x32 * x32, axis=-1, keepdims=True) + 1e-6)).astype(x.dtype)


def rope_tables(cfg: Cfg) -> tuple[Array, Array]:
    inv = 1.0 / ROPE_BASE ** (jnp.arange(0, cfg.head_dim, 2, dtype=F32) / cfg.head_dim)
    freqs = jnp.outer(jnp.arange(cfg.seq, dtype=F32), inv)
    # The reference stores the tables in bfloat16 whatever the compute dtype; match its rounding.
    return (jnp.cos(freqs).astype(BF16)[None, :, None, :], jnp.sin(freqs).astype(BF16)[None, :, None, :])


def rotary(x: Array, cos: Array, sin: Array) -> Array:
    """Rotate channel pairs (i, i + hd/2). x: [B,T,H,hd]."""
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    c, s = cos.astype(x.dtype), sin.astype(x.dtype)
    return jnp.concatenate([x1 * c + x2 * s, -x1 * s + x2 * c], axis=-1)


def processed_qk(x: Array, cos: Array, sin: Array) -> Array:
    """Queries / keys as attention sees them: rotary, then per-head RMS norm."""
    return rms(rotary(x, cos, sin))


def attend(q: Array, k: Array, v: Array, window: int, block: int = 512) -> Array:
    """Causal attention with a left window. [B,T,H,hd] in, [B,T,H,hd] out.

    A query at s sees keys s-window..s. Queries are processed in blocks of
    `block`, and each block only looks at the keys it can possibly see, so the
    masked-out corners of the score matrix are never computed. cuDNN flash
    attention does not build on this box ("no valid execution plans"), and the
    dense XLA path computes and discards half of every score matrix.
    """
    T, D = q.shape[1], q.shape[-1]
    scale = D ** -0.5
    out = []
    for s0 in range(0, T, block):
        s1 = min(s0 + block, T)
        k0 = max(0, s0 - window)
        qb, kb, vb = q[:, s0:s1], k[:, k0:s1], v[:, k0:s1]
        sc = jnp.einsum("bqhd,bkhd->bhqk", qb, kb, preferred_element_type=F32) * scale
        qi = jnp.arange(s0, s1)[:, None]
        ki = jnp.arange(k0, s1)[None, :]
        keep = (qi >= ki) & (qi - ki <= window)
        p = jax.nn.softmax(jnp.where(keep, sc, -jnp.inf), axis=-1).astype(q.dtype)
        out.append(jnp.einsum("bhqk,bkhd->bqhd", p, vb))
    return jnp.concatenate(out, axis=1)


class LR(NamedTuple):
    """A weight with a rank-1 perturbation riding on it: W + s * a b^T, never materialised.

    This is EGGROLL's trick. Perturbing a [out, in] matrix densely costs out*in
    numbers per population member; the factors cost out + in, and the forward pass
    uses them as a side branch,  x W^T + s (x . b) a^T,  so a member's extra work is
    two thin products per layer instead of a fresh copy of the weights.
    """
    w: Array
    a: Array      # [out]
    b: Array      # [in]
    s: Array      # scalar: +sigma or -sigma


def lin(x: Array, w) -> Array:
    """x @ w.T with w stored [out, in]; w may be an `LR`."""
    if isinstance(w, LR):
        y = jnp.einsum("...i,oi->...o", x, w.w)
        t = jnp.einsum("...i,i->...", x, w.b.astype(x.dtype))
        return y + (w.s * t).astype(x.dtype)[..., None] * w.a.astype(x.dtype)
    return jnp.einsum("...i,oi->...o", x, w)


def gather(table, ids: Array) -> Array:
    """table[ids]; for an `LR` table the rows are perturbed too: W[ids] + s a[ids] b^T."""
    if isinstance(table, LR):
        return table.w[ids] + (table.s * table.a[ids]).astype(table.w.dtype)[..., None] * table.b.astype(table.w.dtype)
    return table[ids]


# ── a block ──────────────────────────────────────────────────────────────────

def block_fwd(bp: dict, cfg: Cfg, layer: int, x: Array, ve: Array | None, rot: tuple,
              jit: dict | None = None, cap: dict | None = None) -> Array:
    """One pre-norm block.

    `jit` adds activation noise at the named sites -- 'o' (attention output, the
    input of `proj`), 'proj', 'fc' (pre-activation) and 'mproj' -- which is how
    DUST perturbs. `cap`, if given, is filled with every input and output the
    estimator later needs, so one clean pass serves all of its draws.
    """
    B, T, D = x.shape
    H, hd = cfg.n_head, cfg.head_dim
    jit = jit or {}
    h = rms(x)
    q, k, v = lin(h, bp["q"]), lin(h, bp["k"]), lin(h, bp["v"])
    z = gate = None
    if ve is not None:
        z = lin(h[..., :GATE_CHANNELS], bp["gate"])                       # [B,T,H]
        gate = 2 * jax.nn.sigmoid(z)
        values = v.reshape(B, T, H, hd) + gate[..., None] * ve.reshape(B, T, H, hd)
    else:
        values = v.reshape(B, T, H, hd)
    cos, sin = rot
    qp = processed_qk(q.reshape(B, T, H, hd), cos, sin)
    kp = processed_qk(k.reshape(B, T, H, hd), cos, sin)
    y = attend(qp, kp, values, cfg.windows[layer]).reshape(B, T, D)
    if "o" in jit:
        y = y + jit["o"]
    a = lin(y, bp["proj"])
    if "proj" in jit:
        a = a + jit["proj"]
    x = x + a
    m = rms(x)
    fc = lin(m, bp["fc"])
    if "fc" in jit:
        fc = fc + jit["fc"]
    act = jnp.square(jax.nn.relu(fc))
    mp = lin(act, bp["mproj"])
    if "mproj" in jit:
        mp = mp + jit["mproj"]
    if cap is not None:
        cap.update(h=h, q=q, k=k, v=v, y=y, m=m, act=act)
        if ve is not None:
            cap.update(z=z, ve=ve)
    return x + mp


def embed(pb: dict, ids: Array, jit: Array | None = None) -> Array:
    e = gather(pb["wte"], ids)
    if jit is not None:
        e = e + jit
    return rms(e)


def run_blocks(pb: dict, cfg: Cfg, start: int, xpre: Array, x0: Array, ids: Array, rot: tuple,
               jit: dict | None = None, cap: list | None = None, remat: bool = False) -> Array:
    """Blocks start..L-1, given the stream *before* block `start`'s residual mix.

    This is what lets a draw that jitters block l skip blocks 0..l-1: their
    output is unchanged and was cached by the clean pass. `jit` is for block
    `start` only. `cap`, if given, collects per-block capture dicts.
    """
    x = xpre
    for i in range(start, cfg.n_layer):
        if cap is not None:
            cap_i = {"xpre": x}
        else:
            cap_i = None
        x = pb["resid"][i] * x + pb["x0"][i] * x0
        bp = pb["blocks"][i]
        ve = gather(bp["ve"], ids) if "ve" in bp else None
        if remat:
            # Backprop only: recompute the block in the backward pass instead of keeping
            # every block's f32 attention probabilities (~10 GB for 8 sequences).
            x = jax.checkpoint(lambda bp_, x_, ve_, i=i: block_fwd(bp_, cfg, i, x_, ve_, rot))(bp, x, ve)
        else:
            x = block_fwd(bp, cfg, i, x, ve, rot, jit if i == start else None, cap_i)
        if cap is not None:
            cap.append(cap_i)
    return x


def cap_logits(raw: Array, cfg: Cfg) -> Array:
    return cfg.softcap * jnp.tanh(raw.astype(F32) / cfg.softcap)


def token_loss(x_final: Array, head_w: Array, targets: Array, cfg: Cfg, chunk: int = 16) -> Array:
    """Per-token cross-entropy [N,T] from the final normed stream, chunked over sequences
    so that a 128-sequence draw does not materialise a 4 GB logits tensor."""
    def one(args):
        xf, tg = args
        logits = cap_logits(lin(xf, head_w), cfg)
        lse = jax.nn.logsumexp(logits, axis=-1)
        tgt = jnp.take_along_axis(logits, tg[..., None], axis=-1)[..., 0]
        return lse - tgt

    N = x_final.shape[0]
    if N <= chunk:
        return one((x_final, targets))
    assert N % chunk == 0
    xs = x_final.reshape(N // chunk, chunk, *x_final.shape[1:])
    ts = targets.reshape(N // chunk, chunk, -1)
    return jax.lax.map(one, (xs, ts)).reshape(N, -1)


def forward_loss(pb: dict, cfg: Cfg, ids: Array, targets: Array, rot: tuple, remat: bool = False) -> Array:
    """Per-token loss [B,T] for a compute-dtype parameter tree."""
    x0 = embed(pb, ids)
    x = run_blocks(pb, cfg, 0, x0, x0, ids, rot, remat=remat)
    return token_loss(rms(x), pb["head"], targets, cfg)


def mean_loss(params: dict, cfg: Cfg, ids: Array, targets: Array, rot: tuple, dtype=BF16) -> Array:
    """Mean token loss, rematerialised per block so it can be differentiated at 8 sequences."""
    return forward_loss(cast(params, dtype), cfg, ids, targets, rot, remat=True).mean()
