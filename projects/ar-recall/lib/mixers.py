"""Sequence mixers for AR-Recall: causal, and cheap in memory at long lengths.

Two differences from `recall-gen/lib/core.py`, both forced by this project.

**Causal.** recall-gen runs its scan to completion and then reads the FINAL
state for every token:

    S, _ = lax.scan(step, S0, seq)
    o = einsum("bhvk,bhnk->bhnv", S, q) / sqrt(dk)

That is a write-everything-then-read design, which is correct there because the
query is a single appended token that writes nothing. Next-token prediction needs
`o_t = S_t q_t` — each token reading the state as of its own position. Note that
`o_t` is taken AFTER token t's own write, which is the same convention as causal
attention allowing a token to attend to itself.

**Chunked.** The naive scan stores its carry at every timestep for the backward
pass, and the delta rule's carry is a matrix per head, `(B, H, dk, dk)`:

    128 x 8 x 64 x 64 x 4 bytes = 16.8 MB per step
    x 768 steps                 = 12.9 GB per layer

That is `dk` times more per token than attention keeps, and it is what made
recall-gen's M=64 run ask for 21.53 GiB. It is an artefact of the
implementation, not a property of the delta rule — linear attention exists
precisely because it should be the cheap one at long sequences.

The fix here is the simple exact one: an outer scan over chunks whose body is
wrapped in `jax.checkpoint`, so only chunk-boundary states are kept and the rest
are recomputed in the backward pass. At T=768 with chunk=32 that is 24 stored
states per layer instead of 768. Numerically identical, one extra forward pass of
compute. The chunkwise WY formulation used by DeltaNet proper would also turn the
scan into block matmuls and be faster still; this keeps the kernel recognisable
and is the change worth making first.

`scripts/test_mixers.py` checks the chunked path against a naive reference and
against recall-gen's own kernel.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp


def _split(t, H, DK):
    """(B,N,D) -> (B,H,N,DK). Biases are added in d_model space before this, the
    same order recall-gen uses, so the two kernels stay bit-comparable."""
    B, N, _ = t.shape
    return t.reshape(B, N, H, DK).transpose(0, 2, 1, 3)


def _delta_step(S, t):
    """One delta-rule update, then the read. Shared by both paths so the chunked
    version cannot drift from the reference."""
    a_t, k_t, v_t, b_t, q_t, scale = t
    S = S * a_t[:, :, None, :]                                     # forget
    vhat = jnp.einsum("bhvk,bhk->bhv", S, k_t)                     # what is stored
    e = b_t[..., None] * (v_t - vhat)                              # correction
    S = S + jnp.einsum("bhv,bhk->bhvk", e, k_t)                    # write
    return S, jnp.einsum("bhvk,bhk->bhv", S, q_t) * scale          # read


def _prepare(x, Lp, cfg, write):
    """q, k, v, alpha, beta for the delta rule, with `write` gating who writes.

    A token with `write = 0` neither decays the state nor adds to it, so it can
    read without disturbing what is stored. Defaults to every token writing,
    which is what an autoregressive stream wants.
    """
    B, N, _ = x.shape
    H, DK = cfg.n_heads, cfg.dk
    q = _split(x @ Lp["Wq"], H, DK)
    k = _split(x @ Lp["Wk"], H, DK)
    v = _split(x @ Lp["Wv"], H, DK)
    q = q / (jnp.linalg.norm(q, axis=-1, keepdims=True) + 1e-6)
    k = k / (jnp.linalg.norm(k, axis=-1, keepdims=True) + 1e-6)
    alpha = jax.nn.sigmoid(_split(x @ Lp["Wa"] + Lp["ba"], H, DK))
    beta = jax.nn.sigmoid(x @ Lp["Wb"] + Lp["bb"]).transpose(0, 2, 1)   # (B,H,N)
    if write is not None:
        g = write[:, None, :]                                      # (B,1,N)
        alpha = alpha * g[..., None] + (1.0 - g[..., None])
        beta = beta * g
    return q, k, v, alpha, beta


def kda_causal(x, Lp, cfg, write=None, chunk=32):
    """Causal delta-rule mixing. `chunk=None` runs the plain scan (the reference).

    Returns (B, N, d_model). Memory in the backward pass is
    O(T/chunk + chunk) states rather than O(T).
    """
    B, N, D = x.shape
    H, DK = cfg.n_heads, cfg.dk
    q, k, v, alpha, beta = _prepare(x, Lp, cfg, write)
    scale = jnp.asarray(DK ** -0.5, x.dtype)

    tr = lambda t: t.transpose(2, 0, 1, 3)                         # (N,B,H,DK)
    seq = (tr(alpha), tr(k), tr(v), beta.transpose(2, 0, 1), tr(q))
    S0 = jnp.zeros((B, H, DK, DK), x.dtype)
    step = lambda S, t: _delta_step(S, (*t, scale))

    if chunk is None or chunk >= N:
        S, o = jax.lax.scan(step, S0, seq)
    else:
        # Pad to a whole number of chunks with steps that do nothing: alpha = 1
        # leaves the state undecayed and beta = 0 writes nothing, so the padded
        # steps are exactly the identity and the carry is unchanged.
        pad = (-N) % chunk
        if pad:
            def ext(t, fill):
                return jnp.concatenate(
                    [t, jnp.full((pad, *t.shape[1:]), fill, t.dtype)], axis=0)
            a, kk, vv, bb, qq = seq
            seq = (ext(a, 1.0), ext(kk, 0.0), ext(vv, 0.0), ext(bb, 0.0),
                   ext(qq, 0.0))
        n_ch = (N + pad) // chunk
        seq = tuple(t.reshape(n_ch, chunk, *t.shape[1:]) for t in seq)

        @jax.checkpoint
        def chunk_body(S, cs):
            return jax.lax.scan(step, S, cs)

        S, o = jax.lax.scan(chunk_body, S0, seq)
        o = o.reshape(n_ch * chunk, *o.shape[2:])[:N]

    # scan output is (N,B,H,DK); heads must be adjacent to DK before the
    # reshape, so the batch and time axes come first: (B,N,H,DK).
    return o.transpose(1, 0, 2, 3).reshape(B, N, D) @ Lp["Wo"]


def attn_causal(x, Lp, cfg, write=None, chunk=None):
    """Ordinary causal softmax attention over the same tokens, as the reference.

    Uses the fused kernel when it can. Writing the scores out explicitly costs
    `(B, H, N, N)` several times over — the mask, the `where`, the softmax — and
    at N = 768, batch 64 that is already tens of gigabytes across four layers.
    `jax.nn.dot_product_attention` never materialises them.

    `write = 0` marks a token unreadable as a KEY, the attention analogue of not
    writing to the state. cuDNN takes causal and padding masks but not an
    arbitrary key mask, so that path falls back to the explicit form; the AR task
    does not use it, and recall-gen-style episodes are short enough not to care.
    """
    B, N, D = x.shape
    H, DK = cfg.n_heads, cfg.dk
    # (B, N, H, DK) — the layout the fused kernel wants; no head transpose.
    q, k, v = (x @ Lp[w] for w in ("Wq", "Wk", "Wv"))
    q, k, v = (t.reshape(B, N, H, DK) for t in (q, k, v))
    if write is None:
        # `implementation=None` lets JAX choose. The fused cuDNN kernel would be
        # the one worth having — it never materialises the scores — but it needs
        # bf16/fp16 AND cuDNN fails to build a plan for this shape on this box,
        # so what actually runs is the XLA path, which is O(N^2) in memory. That
        # is the ceiling on sequence length for this mixer here, not a property
        # of attention.
        o = jax.nn.dot_product_attention(q, k, v, is_causal=True,
                                         scale=DK ** -0.5)
    else:
        qh, kh, vh = (t.transpose(0, 2, 1, 3) for t in (q, k, v))
        s = jnp.einsum("bhnd,bhmd->bhnm", qh, kh) * (DK ** -0.5)
        keep = jnp.tril(jnp.ones((N, N), bool))[None, None]
        keep = keep & (write[:, None, None, :] > 0.5)
        keep = keep | jnp.eye(N, dtype=bool)[None, None]           # never an empty row
        o = jnp.einsum("bhnm,bhmd->bhnd",
                       jax.nn.softmax(jnp.where(keep, s, -jnp.inf), axis=-1), vh)
        o = o.transpose(0, 2, 1, 3)
    return o.reshape(B, N, D) @ Lp["Wo"]


MIXERS = {"kda": kda_causal, "attn": attn_causal}


def kda_chunkwise(x, Lp, cfg, write=None, chunk=64):
    """The delta rule as block matmuls: T/chunk sequential steps instead of T.

    The scan version is kernel-launch bound — 768 timesteps x 4 layers x
    (forward + remat recompute + backward) is ~30k tiny kernels per step, which
    measured 721 ms against attention's 17 ms. The fix is not remat, it is to
    stop stepping.

    Write `a_t` for the per-channel decay and `c_t = a_1 * ... * a_t` for the
    cumulative decay WITHIN a chunk. The recurrence

        S_t = S_{t-1} diag(a_t) + e_t k_t^T

    unrolls to `S_t = [S_0 + sum_{j<=t} e_j ktil_j^T] diag(c_t)` with
    `ktil_j = k_j / c_j`. Substituting that in, the decay leaves the recurrence
    entirely and what is left is, with `khat_t = c_t * k_t`:

        vhat_t = [S_0 + sum_{j<t} e_j ktil_j^T] khat_t
        e_t    = b_t (v_t - vhat_t)

    which is a linear system in the e's. Collecting rows,

        (I + diag(b) tril(KHAT KTIL^T, -1)) E = diag(b) (V - KHAT S_0^T)

    The matrix is unit lower triangular, so one triangular solve gives every
    correction in the chunk at once. The reads and the end state are then

        O   = QHAT S_0^T + tril(QHAT KTIL^T, 0) E        (0 = read after own write)
        S_C = (S_0 + E^T KTIL) diag(c_C)

    All matmuls. Only the chunk boundaries stay sequential.

    **The constraint this buys with.** `ktil = k / c` divides by a cumulative
    product of sigmoids, so it overflows if the decay strays far from 1 within a
    chunk: at a = 0.5 and chunk = 64, c ~ 1e-19. The project's decay bias is
    initialised so a token's memory spans `horizon_mult * n_tokens` steps, which
    puts a at ~0.994 and c_64 at ~0.67, and the inner products that actually
    matter depend only on ratios c_t/c_j <= 1. But it is a real precondition, not
    a free win: `scripts/test_mixers.py` measures the error against the reference
    as a function of the decay, and a run whose gates learn to forget fast needs
    a smaller chunk.
    """
    B, N, D = x.shape
    H, DK = cfg.n_heads, cfg.dk
    q, k, v, alpha, beta = _prepare(x, Lp, cfg, write)
    scale = jnp.asarray(DK ** -0.5, x.dtype)

    pad = (-N) % chunk
    if pad:
        def ext(t, fill):
            z = jnp.full((*t.shape[:2], pad, *t.shape[3:]), fill, t.dtype)
            return jnp.concatenate([t, z], axis=2)
        # alpha = 1 and beta = 0 make a padded step exactly the identity.
        q, k, v = (ext(t, 0.0) for t in (q, k, v))
        alpha = ext(alpha, 1.0)
        beta = jnp.concatenate(
            [beta, jnp.zeros((*beta.shape[:2], pad), beta.dtype)], axis=2)
    NC = (N + pad) // chunk

    rs = lambda t: t.reshape(B, H, NC, chunk, DK).transpose(2, 0, 1, 3, 4)
    q, k, v, alpha = (rs(t) for t in (q, k, v, alpha))
    b = beta.reshape(B, H, NC, chunk).transpose(2, 0, 1, 3)

    c = jnp.cumprod(alpha, axis=3)                       # (NC,B,H,C,DK)
    k_hat, k_til, q_hat = k * c, k / c, q * c
    eye = jnp.eye(chunk, dtype=x.dtype)

    def body(S, d):
        kh, kt, qh, vv, bb, c_last = d
        U = bb[..., None] * (vv - jnp.einsum("bhck,bhvk->bhcv", kh, S))
        L = eye + bb[..., None] * jnp.tril(
            jnp.einsum("bhck,bhjk->bhcj", kh, kt), -1)
        E = jax.lax.linalg.triangular_solve(
            L, U, left_side=True, lower=True, unit_diagonal=True)
        O = (jnp.einsum("bhck,bhvk->bhcv", qh, S)
             + jnp.einsum("bhcj,bhjv->bhcv",
                          jnp.tril(jnp.einsum("bhck,bhjk->bhcj", qh, kt), 0), E))
        S = (S + jnp.einsum("bhcv,bhck->bhvk", E, kt)) * c_last[:, :, None, :]
        return S, O

    S0 = jnp.zeros((B, H, DK, DK), x.dtype)
    _, o = jax.lax.scan(body, S0, (k_hat, k_til, q_hat, v, b, c[:, :, :, -1]))
    o = o.transpose(1, 2, 0, 3, 4).reshape(B, H, NC * chunk, DK)[:, :, :N]
    return (o * scale).transpose(0, 2, 1, 3).reshape(B, N, D) @ Lp["Wo"]


MIXERS["kda_wy"] = kda_chunkwise
