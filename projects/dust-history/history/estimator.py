"""One nudge per update: DUST's estimator with a population of one, nudging every site in the same forward pass.

DUST normally spends K forward passes per update, each nudging one site (say block 3's MLP) at every token. Here an update
spends one *noisy* pass (plus the clean pass DUST always runs) and puts an independent nudge at every site at once: the
token embedding, and in every block the attention output `o`, the attention projection `proj`, the MLP input `fc` and the
MLP output `mproj`. The head's logits are nudged slab by slab, as DUST does, which is cheap. The attention internals
(q, k, v, gate, value embedding) are scored by DUST's own local rule against the error estimated at `o`, one draw each.

Reward per token, as DUST does for a single draw: r_t = clean_loss_t - noisy_loss_t. Estimated output error at a site:
g_t = -r_t * a_t / sigma. The weight gradient is the outer product of that error with the cached layer input.

The other sites' nudges are noise to each site's estimate (they add zero on average and some variance). This is the price of
one pass, and averaging over the history is what pays it back.

Left out for now, on purpose: the 2L residual-mixing scalars, which DUST estimates with two-sided weight-space ES (two extra
full passes per draw). They stay at their initial values here, and the parallel-DUST control zeroes their gradient to match.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax import Array

from lib import dust
from lib.model import BF16, F32, GATE_CHANNELS, Cfg, block_fwd, cast, embed, gather, rms, rope_tables, token_loss

SIGMA_DIRECT, SIGMA_DEEP, SIGMA_EMB = dust.SIGMA_DIRECT, dust.SIGMA_DEEP, 0.2


def _widths(cfg: Cfg) -> dict:
    d = cfg.n_embd
    return dict(proj=d, mproj=d, fc=4 * d, o=d)


@partial(jax.jit, static_argnames=("cfg",))
def noisy_pass(pb: dict, ids: Array, targets: Array, key: Array, *, cfg: Cfg) -> tuple[Array, dict]:
    """One forward pass with an independent nudge at every site. Returns per-token loss [B,T] and the nudges."""
    B, T = ids.shape
    rot = rope_tables(cfg)
    widths = _widths(cfg)
    k_emb, k_blocks = jax.random.split(key)
    a_wte = jax.random.normal(k_emb, (B, T, cfg.n_embd), BF16)
    x0 = embed(pb, ids, (SIGMA_EMB * a_wte).astype(BF16))
    x = x0
    noise = {"wte": a_wte, "blocks": []}
    for i in range(cfg.n_layer):
        sig_o = SIGMA_DIRECT if i < cfg.n_layer // 2 else SIGMA_DEEP
        a = {s: jax.random.normal(jax.random.fold_in(jax.random.fold_in(k_blocks, i), j), (B, T, w), BF16)
             for j, (s, w) in enumerate(widths.items())}
        jit = {s: ((sig_o if s == "o" else SIGMA_DIRECT) * a[s]).astype(BF16) for s in a}
        x = pb["resid"][i] * x + pb["x0"][i] * x0
        bp = pb["blocks"][i]
        ve = gather(bp["ve"], ids) if "ve" in bp else None
        x = block_fwd(bp, cfg, i, x, ve, rot, jit, None)
        noise["blocks"].append(a)
    return token_loss(rms(x), pb["head"], targets, cfg), noise


def estimate(params: dict, ids: Array, targets: Array, key: Array, cfg: Cfg) -> tuple[dict, Array]:
    """One update's gradient estimate for the mean token loss, and the clean batch loss."""
    L, V = cfg.n_layer, cfg.vocab
    pb = cast(params)
    cache = dust.capture(pb, ids, targets, cfg=cfg)
    k_noisy, k_local, k_head = jax.random.split(key, 3)
    loss, noise = noisy_pass(pb, ids, targets, k_noisy, cfg=cfg)
    reward = (cache["loss"] - loss).astype(BF16)                     # [B,T]; one draw, so the clean loss is the reference

    def err(a: Array, sigma: float) -> Array:                       # -r * a / sigma, per token
        return -(reward[..., None] * a).astype(F32) / sigma

    grads = jax.tree_util.tree_map(jnp.zeros_like, params)
    blocks = grads["blocks"]
    hubs = {}
    for i in range(L):
        a, c = noise["blocks"][i], cache["blocks"][i]
        blocks[i]["proj"] = dust.outer(err(a["proj"], SIGMA_DIRECT), c["y"])
        blocks[i]["mproj"] = dust.outer(err(a["mproj"], SIGMA_DIRECT), c["act"])
        blocks[i]["fc"] = dust.outer(err(a["fc"], SIGMA_DIRECT), c["m"])
        hubs[i] = err(a["o"], SIGMA_DIRECT if i < L // 2 else SIGMA_DEEP)
    grads["wte"] = dust.scatter_rows(err(noise["wte"], SIGMA_EMB), ids, V)

    err_head = dust.head_error(cache, targets, k_head, jnp.int32(V // dust.SLAB), cfg=cfg, n=1)
    grads["head"] = dust.outer(err_head, cache["xf"])

    for j, kind in enumerate(("q", "v", "gate", "k", "ve")):
        errs = dust.local_errors(pb, cache, hubs, jax.random.fold_in(k_local, j), jnp.int32(1), cfg=cfg, kind=kind, n=1, gamma=0.98)
        for i, e in errs.items():
            c = cache["blocks"][i]
            if kind in ("q", "k", "v"):
                blocks[i][kind] = dust.outer(e, c["h"])
            elif kind == "gate":
                blocks[i]["gate"] = dust.outer(e, c["h"][..., :GATE_CHANNELS])
            else:
                blocks[i]["ve"] = dust.scatter_rows(e, ids, V)

    n_tok = ids.shape[0] * ids.shape[1]
    return jax.tree_util.tree_map(lambda g: g / n_tok, grads), cache["loss"].mean()
