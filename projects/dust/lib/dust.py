"""DUST in JAX: forward-only transformer training. A port of `dust.py` from
github.com/qlabs-eng/dust (the authors' minimal PyTorch release), keeping its
estimator, its tuned recipes and its sites, without autograd anywhere.

One update does this (see `step`):

 1. A clean forward caches every layer's input and output and the per-token loss.
 2. *Direct draws.* Pick a set of sites (say the two residual writers of block l).
    Add sigma * a, a ~ N(0, I), to their outputs at every token, independently, and
    rerun the network from block l. Token t's loss change rewards token t's noise:

        reward_t = mean_over_draws(loss_t) - loss_t          (centred, no rescaling)
        g_t      = - 1/(K sigma) sum_draws reward_t * a_t    (estimated output error)

    n draws run side by side in one batched forward ("nrep").
 3. *Attention internals* (q, k, v, value-embedding gate, value embedding) cannot be
    read from token losses -- their effect is spread over later tokens -- so they are
    scored against the *estimated attention-output error* from step 2: recompute
    only the block's attention with the jitter and take  c_s = <g_s, delta o_s>
    per head, then credit token t with sum_{s>=t} gamma^(s-t) c_s.
 4. The head is jittered directly on the cached logits, one 256-wide slab of the
    vocabulary per draw, and only the cross-entropy is recomputed.
 5. A weight gradient is `sum_t g_t x_t^T`: the estimated output error times the
    cached input, exactly the outer product backprop forms. Embeddings scatter-add.
 6. The 2L residual-mixing scalars are not activations; they use two-sided
    weight-space ES.  Then one SGD-with-momentum step.

Population K counts the *direct* draws (writers, MLP hidden, attention outputs,
token embedding). Head draws and the attention-internal draws come on top, as in
the paper (Appendix E).
"""

from dataclasses import dataclass, replace
from functools import partial

import jax
import jax.numpy as jnp
from jax import Array, lax

from lib.model import (BF16, F32, GATE_CHANNELS, Cfg, attend, block_fwd, cap_logits, cast, embed, lin,
                       processed_qk, rms, rope_tables, run_blocks, token_loss)

SLAB = 256            # vocabulary columns jittered by one head draw
SIGMA_DIRECT = 0.2    # noise scale of direct draws; attention outputs of the deeper half use 0.4
SIGMA_DEEP = 0.4
SIGMA_LOCAL = 0.05    # attention internals and the head
SIGMA_SCALAR = 0.03   # weight-space ES on the residual scalars
N_SCALAR_DRAWS = 8


# ── the recipe ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Recipe:
    """Draws per update for each part of the model. Tuples have one entry per block."""
    writers: tuple          # attention c_proj + MLP c_proj, jittered together in one pass
    hidden: tuple           # MLP c_fc
    hubs: tuple             # attention output, before attn.c_proj
    embedding: int          # token embedding
    head: int               # lm_head slab draws
    local: tuple            # ((kind, draws per block), ...) for q, v, gate, k, ve
    nrep: int               # draws evaluated together in one batched forward
    lr: float
    momentum: float
    gamma: float = 0.98     # credit decay per token of lag, attention internals
    sigma_embed: float = 0.2    # noise scale of the token-embedding draws (0.4 in the paper's Adam runs)

    @property
    def population(self) -> int:
        return sum(self.writers) + sum(self.hidden) + sum(self.hubs) + self.embedding


# Optimiser settings the authors tuned per (budget, population); 64 and 100K are not in the release.
OPTIMISER = {
    "1M": {256: (0.25, 0.95), 1024: (0.225, 0.95), 4096: (0.225, 0.98), 16384: (0.225, 0.98)},
    "10M": {256: (0.175, 0.95), 1024: (0.25, 0.95), 4096: (0.25, 0.95), 16384: (0.25, 0.95)},
}
DRAW_CHUNK = {64: 1, 256: 2, 1024: 8, 4096: 8, 16384: 16}

# 16k at 10M tokens has an allocation of its own (Appendix E, Table 4).
ALLOCATION_16K_10M = dict(writers=(34, 28, 14, 8, 6, 4, 2, 2), hubs=(12, 10, 8, 4, 4, 4, 4, 4), gamma=0.99)

# Appendix E gives the 64-draw recipe explicitly.
RECIPE_64 = dict(writers=(8, 7, 4, 3, 1, 1, 1, 1), hidden=(8, 7, 4, 3, 2, 1, 1, 1), hubs=(1,) * 8,
                 embedding=3, local=(("q", 1), ("v", 5), ("gate", 2), ("k", 3), ("ve", 2)))

# 100K tokens uses its own stream ratio, 6 : 1.5 : 0.5 (Appendix E, Table 4), in K/256 units.
RECIPE_100K = dict(writers=(32, 28, 14, 10, 4, 4, 2, 2), hidden=(32, 26, 14, 10, 4, 4, 2, 2),
                   hubs=(14, 12, 8, 6, 2, 2, 2, 2), embedding=18)


# Under Adam the paper re-weights the streams (Appendix E, Table 3, Adam column, K = 16,384): writers 4,096, MLP hidden 4,096,
# attention outputs 6,144, token embedding 8,192 at noise 0.4, head 49,152 -- as fractions of K. The per-block split under Adam
# is not given, so the SGD recipe's per-block tilt is kept and rescaled to these totals.
ADAM_STREAMS = dict(writers=0.25, hidden=0.25, hubs=0.375, embedding=0.5, head=3.0)
ADAM_SIGMA_EMBED = 0.4


def _spread(shape: tuple, total: int, mult: int) -> tuple:
    """`total` draws over blocks in proportion to `shape`, in multiples of `mult`, at least `mult` per block."""
    n = len(shape)
    total = max(int(round(total / mult)) * mult, mult * n)
    w = [s / sum(shape) for s in shape]
    cnt = [max(mult, int(round(x * total / mult)) * mult) for x in w]
    cnt[cnt.index(max(cnt))] += total - sum(cnt)
    assert all(c >= mult and c % mult == 0 for c in cnt) and sum(cnt) == total, (shape, total, mult, cnt)
    return tuple(cnt)


def make_recipe(budget: str, population: int, lr: float | None = None, momentum: float | None = None,
                opt: str = "sgd") -> Recipe:
    """The authors' recipe for a (token budget, population) cell.

    Populations of 256 and up scale the 256-draw allocation by K/256. 64 draws is
    spelled out in the paper's appendix. `lr` / `momentum` override the tuned values,
    which only exist for 1M and 10M at 256 and above. With opt "adam" or "muon" the draw
    streams are re-weighted as the paper does under Adam (see ADAM_STREAMS).
    """
    if population == 64:
        r = RECIPE_64
        base = dict(writers=r["writers"], hidden=r["hidden"], hubs=r["hubs"], embedding=r["embedding"],
                    local=r["local"], gamma=0.98)
    else:
        scale = population // 256
        if budget == "100K":
            a = RECIPE_100K
            writers, hidden, hubs, emb = a["writers"], a["hidden"], a["hubs"], a["embedding"]
        else:
            writers = (38, 32, 16, 10, 6, 4, 2, 2)
            hidden, hubs, emb = writers, (6, 6, 4, 2, 2, 2, 2, 2), 10
        gamma = 0.98
        if budget == "10M" and population == 16384:
            writers, hubs, gamma = ALLOCATION_16K_10M["writers"], ALLOCATION_16K_10M["hubs"], ALLOCATION_16K_10M["gamma"]
            hidden = writers
        base = dict(writers=tuple(v * scale for v in writers), hidden=tuple(v * scale for v in hidden),
                    hubs=tuple(v * scale for v in hubs), embedding=emb * scale,
                    local=tuple((k, v * scale) for k, v in (("q", 2), ("v", 20), ("gate", 6), ("k", 12), ("ve", 8))),
                    gamma=gamma)
    tuned = OPTIMISER.get(budget, {}).get(population)
    lr_, mom_ = (lr, momentum) if lr is not None else (tuned or (0.2, 0.95))
    if momentum is not None:
        mom_ = momentum
    recipe = Recipe(head=4 * population, nrep=DRAW_CHUNK[population], lr=lr_, momentum=mom_, **base)
    if opt in ("adam", "muon"):
        n, K, s = recipe.nrep, population, ADAM_STREAMS
        recipe = replace(recipe, writers=_spread(recipe.writers, s["writers"] * K, n), hidden=_spread(recipe.hidden, s["hidden"] * K, n),
                         hubs=_spread(recipe.hubs, s["hubs"] * K, n), embedding=int(round(s["embedding"] * K / n)) * n,
                         head=int(s["head"] * K), sigma_embed=ADAM_SIGMA_EMBED)
    return recipe


# ── the clean pass ───────────────────────────────────────────────────────────

@partial(jax.jit, static_argnames=("cfg",))
def capture(pb: dict, ids: Array, targets: Array, *, cfg: Cfg) -> dict:
    """Clean forward that keeps everything the estimator needs."""
    rot = rope_tables(cfg)
    x0 = embed(pb, ids)
    caps: list = []
    x = run_blocks(pb, cfg, 0, x0, x0, ids, rot, cap=caps)
    xf = rms(x)
    raw = lin(xf, pb["head"])
    logits = cap_logits(raw, cfg)
    loss = jax.nn.logsumexp(logits, -1) - jnp.take_along_axis(logits, targets[..., None], -1)[..., 0]
    return dict(blocks=caps, x0=x0, xf=xf, raw=raw.astype(F32), loss=loss)


# ── direct draws ─────────────────────────────────────────────────────────────

def _widths(cfg: Cfg) -> dict:
    return dict(proj=cfg.n_embd, mproj=cfg.n_embd, fc=4 * cfg.n_embd, o=cfg.n_embd, wte=cfg.n_embd)


@partial(jax.jit, static_argnames=("cfg", "layer", "sites", "n"))
def direct_errors(pb: dict, cache: dict, ids: Array, targets: Array, key: Array, iters: Array,
                  sigma: Array, *, cfg: Cfg, layer: int, sites: tuple, n: int) -> dict:
    """Estimated output error at `sites` of block `layer` (layer = -1: the token embedding).

    Runs iters * n draws; each iteration jitters all `sites` in one batched forward
    over n copies of the batch. Returns {site: [B,T,width]}, already divided by the
    number of draws.
    """
    B, T = ids.shape
    rot = rope_tables(cfg)
    widths = _widths(cfg)
    ids_n, tg_n = jnp.tile(ids, (n, 1)), jnp.tile(targets, (n, 1))
    start = max(layer, 0)
    xpre_n = jnp.tile(cache["blocks"][start]["xpre"], (n, 1, 1)) if layer >= 0 else None
    x0_n = jnp.tile(cache["x0"], (n, 1, 1))
    clean = cache["loss"]

    def draw(i, errs):
        k = jax.random.fold_in(key, i)
        noise = {s: jax.random.normal(jax.random.fold_in(k, j), (n * B, T, widths[s]), BF16)
                 for j, s in enumerate(sites)}
        if layer < 0:
            x0 = embed(pb, ids_n, (sigma * noise["wte"]).astype(BF16))
            x = run_blocks(pb, cfg, 0, x0, x0, ids_n, rot)
        else:
            jit = {s: (sigma * a).astype(BF16) for s, a in noise.items()}
            x = run_blocks(pb, cfg, layer, xpre_n, x0_n, ids_n, rot, jit=jit)
        loss = token_loss(rms(x), pb["head"], tg_n, cfg).reshape(n, B, T)
        reward = clean[None] - loss
        if n > 1:
            reward = reward - reward.mean(0, keepdims=True)
        reward = reward.astype(BF16)
        return {s: errs[s] - jnp.einsum("nbt,nbtd->btd", reward, a.reshape(n, B, T, -1),
                                         preferred_element_type=F32) / sigma
                for s, a in noise.items()}

    errs0 = {s: jnp.zeros((B, T, widths[s]), F32) for s in sites}
    errs = lax.fori_loop(0, iters, draw, errs0)
    return {s: e / (iters * n) for s, e in errs.items()}


# ── the head ─────────────────────────────────────────────────────────────────

@partial(jax.jit, static_argnames=("cfg", "n"))
def head_error(cache: dict, targets: Array, key: Array, iters: Array, *, cfg: Cfg, n: int,
               sigma: float = SIGMA_LOCAL) -> Array:
    """Estimated error at the head's raw logits.

    Each iteration jitters one SLAB-wide slice of the cached logits for n copies
    and recomputes the cross-entropy from the other slabs' cached partition sums,
    which is far cheaper than a forward pass. The reward centres over the sequence
    axis, as in the reference.
    """
    V = cfg.vocab
    slabs = V // SLAB
    raw = cache["raw"]                                              # [B,T,V] f32, before the cap
    B, T, _ = raw.shape
    logits = cap_logits(raw, cfg)
    slab_sums = jnp.exp(logits).reshape(B, T, slabs, SLAB).sum(-1)  # [B,T,slabs]
    total = slab_sums.sum(-1)
    clean_target = jnp.take_along_axis(logits, targets[..., None], -1)[..., 0]
    clean = cache["loss"]
    tgt_n = jnp.tile(targets, (n, 1))                               # [n*B, T]

    def draw(i, err):
        block = i % slabs
        lo = block * SLAB
        a = jax.random.normal(jax.random.fold_in(key, i), (n * B, T, SLAB), BF16)
        base = lax.dynamic_slice_in_dim(raw, lo, SLAB, axis=2)
        changed = cfg.softcap * jnp.tanh((jnp.tile(base, (n, 1, 1)) + sigma * a.astype(F32)) / cfg.softcap)
        part = jnp.tile(total - lax.dynamic_index_in_dim(slab_sums, block, axis=2, keepdims=False), (n, 1)) \
            + jnp.exp(changed).sum(-1)
        in_slab = (tgt_n >= lo) & (tgt_n < lo + SLAB)
        tl = jnp.where(in_slab,
                       jnp.take_along_axis(changed, jnp.clip(tgt_n - lo, 0, SLAB - 1)[..., None], -1)[..., 0],
                       jnp.tile(clean_target, (n, 1)))
        loss = (jnp.log(part) - tl).reshape(n, B, T)
        reward = clean[None] - loss
        if B > 1:
            reward = reward - reward.mean(1, keepdims=True)
        upd = jnp.einsum("nbt,nbtd->btd", (reward / sigma).astype(BF16), a.reshape(n, B, T, SLAB),
                         preferred_element_type=F32)
        return err.at[block].add(-upd)

    err = lax.fori_loop(0, iters, draw, jnp.zeros((slabs, B, T, SLAB), F32))
    err = err / (iters * n // slabs)
    return err.transpose(1, 2, 0, 3).reshape(B, T, V)


# ── attention internals ──────────────────────────────────────────────────────

@partial(jax.jit, static_argnames=("cfg", "kind", "n", "gamma"))
def local_errors(pb: dict, cache: dict, targets_hub: tuple, key: Array, iters: Array, *, cfg: Cfg,
                 kind: str, n: int, gamma: float, sigma: float = SIGMA_LOCAL) -> dict:
    """Estimated output error at q / k / v / gate / ve for every block that has them.

    Each draw recomputes one block's attention from the cached clean activations with
    a jitter on `kind`, and scores the change in its output (per head) against the
    estimated error at that output (`targets_hub`, from the hub draws):

        c_s = <g_s, delta o_s>        reward_t = sum_{s>=t, s-t<=window} gamma^(s-t) c_s

    Queries only affect their own token's output, so they keep c_t alone. Returns
    {layer: [B,T,width]}.
    """
    L, H, hd = cfg.n_layer, cfg.n_head, cfg.head_dim
    layers = [i for i in range(L) if kind not in ("ve", "gate") or cfg.has_ve(i)]
    B, T = cache["loss"].shape
    cos, sin = rope_tables(cfg)
    width = H if kind == "gate" else cfg.n_embd
    pos = jnp.arange(T)
    lag = pos[None, :] - pos[:, None]                                # lag[t, s] = s - t
    clean = {}
    for i in layers:
        c = cache["blocks"][i]
        q, k, v = [c[x].reshape(B, T, H, hd) for x in ("q", "k", "v")]
        z = c.get("z")
        ve = c["ve"].reshape(B, T, H, hd) if z is not None else None
        values = v if z is None else v + (2 * jax.nn.sigmoid(z))[..., None] * ve
        qp, kp = processed_qk(q, cos, sin), processed_qk(k, cos, sin)
        window = cfg.windows[i]
        base = attend(qp, kp, values, window)
        credit = ((lag >= 0) & (lag <= window)).astype(F32) * gamma ** jnp.clip(lag, 0).astype(F32)
        clean[i] = (q, k, z, ve, qp, kp, values, base, credit)

    def rep(x):
        return jnp.broadcast_to(x[None], (n,) + x.shape).reshape((n * B,) + x.shape[1:])

    def draw(it, errs):
        out = {}
        for i in layers:
            q, k, z, ve, qp, kp, values, base, credit = clean[i]
            window = cfg.windows[i]
            a = jax.random.normal(jax.random.fold_in(jax.random.fold_in(key, it), i), (n, B, T, width), F32)
            divisor = sigma
            if kind in ("q", "k"):
                raw = q if kind == "q" else k
                pert = processed_qk((raw[None] + sigma * a.reshape(n, B, T, H, hd)).reshape(n * B, T, H, hd),
                                    cos, sin).astype(BF16)
                qa, ka = (pert, rep(kp)) if kind == "q" else (rep(qp), pert)
                delta = attend(qa, ka, rep(values), window).reshape(n, B, T, H, hd) - base[None]
            else:
                if kind == "gate":
                    dg = 2 * jax.nn.sigmoid(z[None] + sigma * a) - 2 * jax.nn.sigmoid(z[None] - sigma * a)
                    dv = dg[..., None] * ve[None]
                    divisor = 2 * sigma
                else:
                    dv = sigma * a.reshape(n, B, T, H, hd)
                    if kind == "ve":
                        dv = dv * (2 * jax.nn.sigmoid(z))[None, ..., None]
                delta = attend(rep(qp), rep(kp), dv.reshape(n * B, T, H, hd).astype(BF16),
                               window).reshape(n, B, T, H, hd)
            tgt = targets_hub[i].reshape(B, T, H, hd)
            score = jnp.einsum("bthd,nbthd->nbth", tgt.astype(BF16), delta.astype(BF16),
                               preferred_element_type=F32)
            if kind != "q":
                score = jnp.einsum("nbsh,ts->nbth", score, credit)
            est = jnp.einsum("nbth,nbthd->bthd", score, a.reshape(n, B, T, H, -1))
            out[i] = errs[i] + est.reshape(B, T, width) / divisor
        return out

    errs = lax.fori_loop(0, iters, draw, {i: jnp.zeros((B, T, width), F32) for i in layers})
    return {i: e / (iters * n) for i, e in errs.items()}


# ── the weight-space scalars ─────────────────────────────────────────────────

@partial(jax.jit, static_argnames=("cfg", "draws"))
def scalar_gradients(params: dict, ids: Array, targets: Array, key: Array, *, cfg: Cfg, draws: int,
                     sigma: float = SIGMA_SCALAR) -> tuple:
    """Two-sided weight-space ES for the 2L residual mixing scalars, summed token loss as fitness."""
    rot = rope_tables(cfg)
    L = cfg.n_layer

    def loss_with(resid, x0):
        pb = cast({**params, "resid": resid, "x0": x0})
        from lib.model import forward_loss
        return forward_loss(pb, cfg, ids, targets, rot).sum()

    def one(i, acc):
        a = jax.random.normal(jax.random.fold_in(key, i), (2, L), F32)
        lp = loss_with(params["resid"] + sigma * a[0], params["x0"] + sigma * a[1])
        lm = loss_with(params["resid"] - sigma * a[0], params["x0"] - sigma * a[1])
        return acc + (lp - lm) / (2 * sigma) * a / draws

    est = lax.fori_loop(0, draws, one, jnp.zeros((2, L), F32))
    return est[0], est[1]


# ── assembling a weight gradient ─────────────────────────────────────────────

def outer(err: Array, inp: Array) -> Array:
    """sum_t err_t inp_t^T -- the exact local linear map. [B,T,o] x [B,T,i] -> [o,i]."""
    return jnp.einsum("bto,bti->oi", err.astype(F32), inp.astype(F32))


def scatter_rows(err: Array, ids: Array, V: int) -> Array:
    """Embedding weight gradient: error added into each token's row."""
    return jnp.zeros((V, err.shape[-1]), F32).at[ids.reshape(-1)].add(err.reshape(-1, err.shape[-1]))


# ── one update ───────────────────────────────────────────────────────────────

def estimate_gradient(params: dict, ids: Array, targets: Array, key: Array, recipe: Recipe, cfg: Cfg,
                      timings: dict | None = None) -> tuple[dict, Array]:
    """DUST's gradient estimate for the mean token loss, and the clean batch loss."""
    L, V = cfg.n_layer, cfg.vocab
    pb = cast(params)
    n = recipe.nrep
    cache = capture(pb, ids, targets, cfg=cfg)
    k_ctr = iter(range(10_000))

    def sub():
        return jax.random.fold_in(key, next(k_ctr))

    def timed(name, fn):
        if timings is None:
            return fn()
        import time
        jax.block_until_ready(cache["loss"])
        t0 = time.perf_counter()
        out = fn()
        jax.block_until_ready(out)
        timings[name] = timings.get(name, 0.0) + time.perf_counter() - t0
        return out

    blocks = [dict() for _ in range(L)]
    grads: dict = {"blocks": blocks}

    for l in range(L):
        for sites, count in ((("proj", "mproj"), recipe.writers[l]), (("fc",), recipe.hidden[l])):
            if count == 0:
                continue
            errs = timed("direct", lambda: direct_errors(pb, cache, ids, targets, sub(), jnp.int32(count // n),
                                                         jnp.float32(SIGMA_DIRECT), cfg=cfg, layer=l,
                                                         sites=sites, n=n))
            c = cache["blocks"][l]
            for s in sites:
                inp = {"proj": c["y"], "mproj": c["act"], "fc": c["m"]}[s]
                blocks[l][s] = outer(errs[s], inp)

    errs = timed("direct", lambda: direct_errors(pb, cache, ids, targets, sub(), jnp.int32(recipe.embedding // n),
                                                 jnp.float32(recipe.sigma_embed), cfg=cfg, layer=-1, sites=("wte",), n=n))
    grads["wte"] = scatter_rows(errs["wte"], ids, V)

    hubs = {}
    for l in range(L):
        sigma = SIGMA_DIRECT if l < L // 2 else SIGMA_DEEP
        hubs[l] = timed("hubs", lambda: direct_errors(pb, cache, ids, targets, sub(), jnp.int32(recipe.hubs[l] // n),
                                                      jnp.float32(sigma), cfg=cfg, layer=l, sites=("o",), n=n)["o"])

    err_head = timed("head", lambda: head_error(cache, targets, sub(), jnp.int32(recipe.head // n), cfg=cfg, n=n))
    grads["head"] = outer(err_head, cache["xf"])

    for kind, count in recipe.local:
        errs = timed("local", lambda: local_errors(pb, cache, hubs, sub(), jnp.int32(count // n), cfg=cfg,
                                                   kind=kind, n=n, gamma=recipe.gamma))
        for l, e in errs.items():
            c = cache["blocks"][l]
            if kind in ("q", "k", "v"):
                blocks[l][kind] = outer(e, c["h"])
            elif kind == "gate":
                blocks[l]["gate"] = outer(e, c["h"][..., :GATE_CHANNELS])
            else:
                blocks[l]["ve"] = scatter_rows(e, ids, V)

    g_resid, g_x0 = timed("scalar", lambda: scalar_gradients(params, ids, targets, sub(), cfg=cfg,
                                                              draws=N_SCALAR_DRAWS))
    grads["resid"], grads["x0"] = g_resid, g_x0

    n_tok = ids.shape[0] * ids.shape[1]
    grads = jax.tree_util.tree_map(lambda g: g / n_tok, grads)
    return grads, cache["loss"].mean()


# ── optimiser: SGD with momentum and the paper's per-group learning rates ────

def group_lrs(params: dict, base: float) -> dict:
    """Learning rate per tensor: token embedding 1000, value embeddings 0.3, residual scalars 0.03."""
    lrs = jax.tree_util.tree_map(lambda _: base, params)
    lrs["wte"] = 1000.0
    lrs["resid"] = lrs["x0"] = 0.03
    for b in lrs["blocks"]:
        if "ve" in b:
            b["ve"] = 0.3
    return lrs


@jax.jit
def sgd_update(params: dict, mom: dict, grads: dict, lrs: dict, momentum: Array) -> tuple[dict, dict]:
    """torch.optim.SGD(momentum): buf = momentum*buf + g;  p -= lr*buf.  No dampening, no decay."""
    mom = jax.tree_util.tree_map(lambda m, g: momentum * m + g, mom, grads)
    params = jax.tree_util.tree_map(lambda p, m, lr: p - lr * m, params, mom, lrs)
    return params, mom
