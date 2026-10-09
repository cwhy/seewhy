"""One training driver for every arm, so the arms differ only in the generator.

A `Cell` is one (arm, M, seed) run. `run_cells` executes a list, skips any
already in `results.jsonl`, and appends one row each.

Episode, for each of B episodes:

    ctx      M real images from the pool, written into D's state
    queries  Q_MEM members      (copies of random context images)    label 1
             Q_NON real images  (drawn from the pool)                label 1 iff its
                                                                     index is in ctx
             Q_GEN generated    (from G)                             label 0

Labels for generated images are the plain rule: generated = "not in context".
That is the version under test (concepts.md, "the margin question"). The
`none` arm has no generator and fills the Q_GEN slots with more real
non-members, so it is the membership positive control.

Arms
    none   no generator
    blind  generator without the context
    ctx    generator that writes the context into its own state

`ctx_gen > 0` mixes generated images into the context (see `Cell.ctx_gen`):
per episode, q_mem real members + q_gen generated members, q_non real
non-members + q_gen fresh generated non-members.

`disc="plain"` swaps the membership game for an ordinary GAN: a context-free
real-vs-fake discriminator, the same generator and the same loop. It is the
generator's positive control: if that cannot make digits, the fault is in the
generator or the loop, not in the membership game.
"""

import json
import logging
import pickle
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from lib.model import (Cfg, disc_logits, disc_logits_prefix, encode, generate, generate_from_state,
                       generate_logits, generate_prefix, init_disc, init_gen, init_plain_disc, n_params,
                       plain_logits, write_step)

PROJECT = Path(__file__).resolve().parents[1]
JSONL = PROJECT / "results.jsonl"
ARMS = ("none", "blind", "ctx")


@dataclass(frozen=True)
class Cell:
    exp: str
    arm: str                      # none | blind | ctx
    disc: str = "membership"      # membership | plain
    m_ctx: int = 16
    seed: int = 42
    steps: int = 20_000
    batch: int = 64               # episodes per step
    q_mem: int = 4
    q_non: int = 2
    q_gen: int = 2
    lr_d: float = 2e-4
    lr_g: float = 2e-4
    beta1: float = 0.5
    eval_every: int = 500
    eval_episodes: int = 512
    tag: str = ""                 # distinguishes tuning variants of the same arm
    r1: float = 0.0               # R1 penalty weight: (r1/2) E_real |dD/dx|^2
    r2: float = 0.0               # same penalty on the queries labelled 0 (generated): R2
    adam_b2: float = 0.999        # Adam beta2, both networks (beta1 is `beta1`)
    # "standard": each query scored on its own (binary cross-entropy).
    # "relativistic": D is trained on pairs (context image i, generated image i)
    # to score the context image higher; G to make its image score higher than
    # its paired context image. softplus(-(D(a) - D(b))), as in RpGAN / R3GAN.
    loss: str = "standard"
    # Train on every prefix of the context (agreed with the user, "teacher
    # forcing"): after each of the M writes, D answers 2 real queries (the image
    # just written, one random earlier context image) and 2 fakes (the "twin":
    # the generator image from the previous prefix, and a fresh one from this
    # prefix). Requires q_non == 0; q_mem / q_gen are not used.
    prefix: bool = False
    # 0: the context is M random real images. >0: the first `nn_warm` are random
    # (warm start); after that, each next context image is the real training
    # image nearest in pixel distance to the generator's image at the current
    # prefix — so that image's "twin" fake is queried right after it is written.
    nn_warm: int = 0
    d_act: str = "gelu"           # -> Cfg.d_act
    g_ln: bool = False            # -> Cfg.g_ln
    # Mixed context: this many of the M context slots hold GENERATED images.
    # Members and non-members then each carry the same share of generated
    # images (q_gen of each), so "looks generated" says nothing about
    # membership. 0 = exp1's all-real context.
    ctx_gen: int = 0
    # Where the generated context items come from. "" = the generator being
    # trained (exp5 — which lets a collapsed G plant its own output in every
    # context). A run name = that run's saved generator, frozen: its images
    # are members or non-members in equal share, and the trained G's images
    # are only ever non-members, so planting is impossible.
    frozen_gen: str = ""
    # Warm start: copy this run's saved (context-blind) generator into the
    # trained generator before training. For the `ctx` arm the weights reading
    # the context start at zero, so it starts as exactly that generator.
    init_gen_from: str = ""

    @property
    def run(self) -> str:
        d = "" if self.disc == "membership" else f"_{self.disc}"
        t = f"_{self.tag}" if self.tag else ""
        return f"{self.exp}_{self.arm}{d}_m{self.m_ctx}_s{self.seed}{t}"


def append_result(row: dict) -> None:
    with open(JSONL, "a") as f:
        f.write(json.dumps(row) + "\n")


def done_runs() -> set[str]:
    done = set()
    if JSONL.exists():
        for line in JSONL.read_text().splitlines():
            try:
                done.add(json.loads(line).get("run"))
            except Exception:
                pass
    return done


def load_pools():
    """(train, held, held labels) MNIST pools, flattened, values in [0,1].
    Labels are used only by the evaluation-only class probe, never in training."""
    from shared_lib.datasets import load_supervised_image
    d = load_supervised_image("mnist")
    tr = np.asarray(d.X, np.float32).reshape(len(d.X), -1)
    te = np.asarray(d.X_test, np.float32).reshape(len(d.X_test), -1)
    if tr.max() > 1.5:
        tr, te = tr / 255.0, te / 255.0
    return jnp.asarray(tr), jnp.asarray(te), np.asarray(d.y_test)


def make_class_probe(held_pool, held_y, cfg: Cfg, n_query: int = 1000, n_ctx: int = 16):
    """Evaluation only: the discriminator as a 10-way classifier.

    One fixed context per digit (16 random held-out images of it). Each of
    `n_query` other held-out images is assigned the digit whose context gives
    it the highest score. Returns a jitted fn pd -> accuracy (chance 0.10).
    Fixed draws (numpy seed 0), so the number is comparable across steps and runs.
    """
    rng = np.random.default_rng(0)
    ctx_idx = np.stack([rng.choice(np.where(held_y == k)[0], n_ctx, replace=False) for k in range(10)])
    rest = np.setdiff1d(np.arange(len(held_y)), ctx_idx.ravel())
    q_idx = rng.choice(rest, n_query, replace=False)
    ctx10 = held_pool[ctx_idx]                                    # (10,16,784)
    q = jnp.broadcast_to(held_pool[q_idx][None], (10, n_query, held_pool.shape[1]))
    qy = jnp.asarray(held_y[q_idx])

    @jax.jit
    def class_acc(pd, ctx10, q, qy):
        return (jnp.argmax(disc_logits(pd, ctx10, q, cfg), axis=0) == qy).mean()

    return lambda pd: class_acc(pd, ctx10, q, qy)


# ── episodes ──────────────────────────────────────────────────────────────────

def sample_episode(key, pool, B, M, q_mem, q_non):
    """Real part of a batch of episodes. Indices are drawn with replacement; a
    non-member that happens to share an index with the context is labelled a
    member, so every label is exact."""
    N = pool.shape[0]
    k1, k2, k3, k4 = jax.random.split(key, 4)
    ci = jax.random.randint(k1, (B, M), 0, N)
    ctx = pool[ci]
    pos = jax.random.randint(k2, (B, q_mem), 0, M)
    mem = jnp.take_along_axis(ctx, pos[..., None], axis=1)
    ni = jax.random.randint(k3, (B, q_non), 0, N)
    non = pool[ni]
    non_lab = (ni[:, :, None] == ci[:, None, :]).any(-1).astype(jnp.float32)
    ref = pool[jax.random.randint(k4, (B, M), 0, N)]       # M other images, for the distance control
    return ctx, mem, non, non_lab, ref


def nearest_sq(x, S):
    """x (B,Q,P), S (B,M,P) -> min_m mean-squared distance (B,Q)."""
    return ((x[:, :, None, :] - S[:, None, :, :]) ** 2).mean(-1).min(-1)


def auroc(pos, neg, w_neg=None):
    """P(score_pos > score_neg), ties counted half. Chance 0.5.

    `w_neg` (0/1, shaped like `neg`) drops negatives that are not negatives."""
    pos, neg = pos.reshape(-1), neg.reshape(-1)
    w = jnp.ones_like(neg) if w_neg is None else w_neg.reshape(-1)
    win = (pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])
    return (win * w[None, :]).sum() / (pos.shape[0] * w.sum())


# ── training ──────────────────────────────────────────────────────────────────

def make_fns(cell: Cell, cfg: Cfg):
    B, M = cell.batch, cell.m_ctx
    has_gen = cell.arm != "none"
    plain = cell.disc == "plain"
    assert not plain or cell.arm == "blind", "the plain game has no context to give a generator"
    q_non = cell.q_non + (0 if has_gen else cell.q_gen)
    mixed = cell.ctx_gen > 0
    rel = cell.loss == "relativistic"
    assert not cell.prefix or (has_gen and not plain and not mixed and not rel and q_non == 0), \
        "prefix training is defined for the plain game only"
    assert cell.nn_warm == 0 or cell.prefix, "the nearest-image context is built inside prefix training"
    assert not rel or (has_gen and not plain and not mixed and q_non == 0 and cell.q_mem == cell.q_gen), \
        "relativistic pairs need the plain game: q_mem context images vs q_gen generated, nothing else"
    frozen = bool(cell.frozen_gen)
    if frozen:
        # ~5 MB of weights, closed over as constants: small enough to be safe.
        with open(PROJECT / f"params_{cell.frozen_gen}.pkl", "rb") as f:
            ck = pickle.load(f)
        pg0 = jax.tree_util.tree_map(jnp.asarray, ck["gen"])
        cfg0 = Cfg(**ck["cfg"])
        assert "mem" not in pg0, "the frozen source must be a context-blind generator"
    assert not mixed or (has_gen and not plain), "a mixed context needs a generator and the membership game"
    opt_d = optax.adam(cell.lr_d, b1=cell.beta1, b2=cell.adam_b2)
    opt_g = optax.adam(cell.lr_g, b1=cell.beta1, b2=cell.adam_b2)
    bce = optax.sigmoid_binary_cross_entropy

    def D(pd, ctx, x):
        return plain_logits(pd, x, cfg) if plain else disc_logits(pd, ctx, x, cfg)

    def d_loss(pd, ctx, qry, lab):
        if rel:
            l = D(pd, ctx, qry)
            n = cell.q_mem               # queries are [context images | generated], paired by position
            loss = jax.nn.softplus(-(l[:, :n] - l[:, n:])).mean()
        else:
            loss = bce(D(pd, ctx, qry), lab).mean()
        if cell.r1:
            # R1 (Mescheder et al. 2018): penalise D's input gradient on the
            # queries labelled 1, which keeps D from becoming a cliff that
            # gives G no usable direction.
            g = jax.grad(lambda x: (D(pd, ctx, x) * lab).sum())(qry)
            loss = loss + 0.5 * cell.r1 * ((g ** 2).sum(-1) * lab).sum() / jnp.maximum(lab.sum(), 1.0)
        if cell.r2:
            # R2: the same penalty on the queries labelled 0 (the generated
            # images), as in R3GAN (Huang et al. 2024)
            neg = 1.0 - lab
            g = jax.grad(lambda x: (D(pd, ctx, x) * neg).sum())(qry)
            loss = loss + 0.5 * cell.r2 * ((g ** 2).sum(-1) * neg).sum() / jnp.maximum(neg.sum(), 1.0)
        return loss

    def g_loss(pg, pd, ctx, z, real=None):
        fake = generate(pg, ctx, z, cfg)
        if rel:                          # make each generated image outscore its paired context image
            return jax.nn.softplus(-(D(pd, ctx, fake) - D(pd, ctx, real))).mean()
        # non-saturating: maximise log D(G(z))
        return bce(D(pd, ctx, fake), jnp.ones(fake.shape[:2])).mean()

    def mixed_episode(key, pool, pg, n_ep, n_mem, n_non, n_gen):
        """Context = (M - ctx_gen) real + ctx_gen generated, shuffled.

        Returns the context, real members, generated members, real
        non-members with their labels, the noise for the fresh generated
        non-members, the real part of the context, and a reference set the
        size of that real part (for the distance control)."""
        k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
        G = cell.ctx_gen
        ctx_r, mem_r, non_r, non_lab, ref = sample_episode(k1, pool, n_ep, M - G, n_mem, n_non)
        src, src_cfg = (pg0, cfg0) if frozen else (pg, cfg)
        g_ctx = jax.lax.stop_gradient(
            generate(src, ctx_r, jax.random.normal(k2, (n_ep, G, src_cfg.z_dim)), src_cfg))
        ctx = jnp.concatenate([ctx_r, g_ctx], 1)
        perm = jnp.argsort(jax.random.uniform(k3, (n_ep, M)), axis=1)
        ctx = jnp.take_along_axis(ctx, perm[..., None], axis=1)
        mem_g = jnp.take_along_axis(g_ctx, jax.random.randint(k4, (n_ep, n_gen), 0, G)[..., None], axis=1)
        z = jax.random.normal(k5, (n_ep, n_gen, cfg.z_dim))
        if frozen:
            # fresh images from the frozen source: its non-member share, so
            # the source is 1 in 4 on both sides of the label
            fresh0 = generate(pg0, ctx_r, jax.random.normal(k6, (n_ep, n_gen, cfg0.z_dim)), cfg0)
            non_r = jnp.concatenate([non_r, fresh0], 1)
            non_lab = jnp.concatenate([non_lab, jnp.zeros((n_ep, n_gen))], 1)
        return ctx, mem_r, mem_g, non_r, non_lab, z, ctx_r, ref

    def build_context(key, pool, pool_sq, pg, n_ep):
        """Returns (ctx (n_ep,M,P), proposal noise (n_ep,M,z)). With nn_warm > 0
        the context is built one image at a time from the generator's proposals."""
        k1, k2 = jax.random.split(key)
        ctx_rand = pool[jax.random.randint(k1, (n_ep, M), 0, pool.shape[0])]
        zs = jax.random.normal(k2, (n_ep, M, cfg.z_dim))
        if cell.nn_warm == 0:
            return ctx_rand, zs
        W = cell.nn_warm
        pgs = jax.lax.stop_gradient(pg)

        def body(carry, t):
            S, nxt = carry
            x_t = jnp.where(t < W, ctx_rand[:, t], nxt)
            if "mem" in pgs:
                S = write_step(pgs["mem"], S, encode(pgs["enc"], x_t), cfg)
            prop = generate_from_state(pgs, S, zs[:, t][:, None], cfg)[:, 0]          # (n_ep,P)
            nn = jnp.argmin(pool_sq[None, :] - 2.0 * prop @ pool.T, axis=1)
            return (S, pool[nn]), x_t

        S0 = jnp.zeros((n_ep, cfg.n_heads, cfg.dk, cfg.dk))
        _, xs = jax.lax.scan(body, (S0, ctx_rand[:, 0]), jnp.arange(M))
        return jax.lax.stop_gradient(xs.transpose(1, 0, 2)), zs

    def prefix_fakes(pg, ctx, zs, zf):
        """Fakes per prefix t: [twin = G(prefix t-1, zs[t-1]), fresh = G(prefix t, zf)].
        At t = 0 there is no previous prefix, so the twin slot gets a second fresh image."""
        props = generate_prefix(pg, ctx, zs[:, :, None], cfg)[:, :, 0]                 # (B,M,P)
        fresh = generate_prefix(pg, ctx, zf, cfg)                                       # (B,M,2,P)
        twin = jnp.concatenate([fresh[:, :1, 1], props[:, :-1]], axis=1)                # (B,M,P)
        return jnp.stack([twin, fresh[:, :, 0]], axis=2)                                # (B,M,2,P)

    def step_prefix(pool, pool_sq, carry, key):
        pd, pg, sd, sg = carry
        kc, kr, kf = jax.random.split(key, 3)
        ctx, zs = build_context(kc, pool, pool_sq, pg, B)
        r = jnp.floor(jax.random.uniform(kr, (B, M)) * (jnp.arange(M) + 1)).astype(jnp.int32)
        real = jnp.stack([ctx, jnp.take_along_axis(ctx, r[..., None], axis=1)], axis=2)  # (B,M,2,P)
        zf = jax.random.normal(kf, (B, M, 2, cfg.z_dim))
        fake = jax.lax.stop_gradient(prefix_fakes(pg, ctx, zs, zf))

        def d_loss_p(pd):
            l = disc_logits_prefix(pd, ctx, jnp.concatenate([real, fake], 2), cfg)    # (B,M,4)
            loss = (bce(l[..., :2], jnp.ones_like(l[..., :2])).mean()
                    + bce(l[..., 2:], jnp.zeros_like(l[..., 2:])).mean()) / 2
            if cell.r1:
                g = jax.grad(lambda x: disc_logits_prefix(pd, ctx, jnp.concatenate([x, fake], 2), cfg)
                             [..., :2].sum())(real)
                loss = loss + 0.5 * cell.r1 * (g ** 2).sum(-1).mean()
            return loss

        ld, gd = jax.value_and_grad(d_loss_p)(pd)
        ud, sd = opt_d.update(gd, sd, pd)
        pd = optax.apply_updates(pd, ud)

        def g_loss_p(pg):
            f = prefix_fakes(pg, ctx, zs, zf)
            l = disc_logits_prefix(pd, ctx, f, cfg)
            return bce(l, jnp.ones_like(l)).mean()

        lg, gg = jax.value_and_grad(g_loss_p)(pg)
        ug, sg = opt_g.update(gg, sg, pg)
        pg = optax.apply_updates(pg, ug)
        return (pd, pg, sd, sg), (ld, lg, optax.global_norm(gg))

    def step(pool, carry, key):
        pd, pg, sd, sg = carry
        ke, kz = jax.random.split(key)
        if mixed:
            ctx, mem_r, mem_g, non, non_lab, z, _, _ = mixed_episode(
                ke, pool, pg, B, cell.q_mem, cell.q_non, cell.q_gen)
            fresh = jax.lax.stop_gradient(generate(pg, ctx, z, cfg))
            qry = jnp.concatenate([mem_r, mem_g, non, fresh], 1)
            lab = jnp.concatenate([jnp.ones((B, cell.q_mem + cell.q_gen)), non_lab,
                                   jnp.zeros((B, cell.q_gen))], 1)
            return update(pd, pg, sd, sg, ctx, qry, lab, z)
        ctx, mem, non, non_lab, _ = sample_episode(ke, pool, B, M, cell.q_mem, q_non)
        qry = [mem, non]
        lab = [jnp.ones((B, cell.q_mem)), non_lab]
        if plain:                         # real images vs as many fakes, labels 1 / 0
            qry, lab = [non[:, :cell.q_gen]], [jnp.ones((B, cell.q_gen))]
        if has_gen:
            z = jax.random.normal(kz, (B, cell.q_gen, cfg.z_dim))
            qry.append(jax.lax.stop_gradient(generate(pg, ctx, z, cfg)))
            lab.append(jnp.zeros((B, cell.q_gen)))
        return update(pd, pg, sd, sg, ctx, jnp.concatenate(qry, 1), jnp.concatenate(lab, 1),
                      z if has_gen else None, real=mem)

    def update(pd, pg, sd, sg, ctx, qry, lab, z, real=None):
        """One D step on (qry, lab), then one G step on noise z (None = no G)."""
        ld, gd = jax.value_and_grad(d_loss)(pd, ctx, qry, lab)
        ud, sd = opt_d.update(gd, sd, pd)
        pd = optax.apply_updates(pd, ud)
        lg, gnorm = jnp.float32(0.0), jnp.float32(0.0)
        if z is not None:
            lg, gg = jax.value_and_grad(g_loss)(pg, pd, ctx, z, real)
            gnorm = optax.global_norm(gg)
            ug, sg = opt_g.update(gg, sg, pg)
            pg = optax.apply_updates(pg, ug)
        return (pd, pg, sd, sg), (ld, lg, gnorm)

    @jax.jit
    def block(pd, pg, sd, sg, pool, key):
        # the pool is an argument, never a closed-over constant (workflow.md)
        keys = jax.random.split(key, cell.eval_every)
        if cell.prefix:
            pool_sq = (pool ** 2).sum(-1)
            body = lambda c, k: step_prefix(pool, pool_sq, c, k)
        else:
            body = lambda c, k: step(pool, c, k)
        (pd, pg, sd, sg), (ld, lg, gn) = jax.lax.scan(body, (pd, pg, sd, sg), keys)
        return pd, pg, sd, sg, ld.mean(), lg.mean(), gn.mean()

    @jax.jit
    def evaluate(pd, pg, pool, key):
        ke, kz = jax.random.split(key)
        E = cell.eval_episodes
        if mixed:
            return evaluate_mixed(pd, pg, pool, ke, kz)
        twin = {}
        if cell.prefix:
            # Evaluation only: the newest context image vs its twin fake, on
            # held-out contexts (built from the held-out pool), from prefix
            # max(nn_warm, 1) on — in the random-context arm the "twin" is just
            # the previous prefix's fake.
            kt1, kt2 = jax.random.split(jax.random.fold_in(ke, 7))
            n_tw = 256
            ctx_t, zs_t = build_context(kt1, pool, (pool ** 2).sum(-1), pg, n_tw)
            f_t = prefix_fakes(pg, ctx_t, zs_t, jax.random.normal(kt2, (n_tw, M, 2, cfg.z_dim)))
            l_t = disc_logits_prefix(pd, ctx_t, jnp.concatenate([ctx_t[:, :, None], f_t[:, :, :1]], 2), cfg)
            w0 = max(cell.nn_warm, 1)
            twin = dict(twin_auroc=auroc(l_t[:, w0:, 0], l_t[:, w0:, 1]))
        ctx, mem, non, non_lab, ref = sample_episode(ke, pool, E, M, cell.q_mem, cell.q_mem)
        if plain:
            z = jax.random.normal(kz, (E, cell.q_mem, cfg.z_dim))
            a = generate_logits(pg, ctx, z, cfg)
            fake = jax.nn.sigmoid(a)
            s_gen, s_real = plain_logits(pd, fake, cfg), plain_logits(pd, non, cfg)
            return dict(acc_real=(s_real > 0).mean(), acc_gen=(s_gen < 0).mean(),
                        p_gen=jax.nn.sigmoid(s_gen).mean(), gen_std=fake.std(axis=(0, 1)).mean(),
                        sat=(jnp.abs(a) > 6.0).mean(),
                        d_train_gen=nearest_sq(fake, ref).mean(), d_train_real=nearest_sq(non, ref).mean())
        s_mem = disc_logits(pd, ctx, mem, cfg)
        s_non = disc_logits(pd, ctx, non, cfg)
        keep = (non_lab < 0.5).astype(jnp.float32)     # true non-members only
        out = dict(
            auroc_real=auroc(s_mem, s_non, keep),
            acc_mem=(s_mem > 0).mean(),
            acc_non=jnp.sum((s_non < 0) * keep) / jnp.sum(keep),
            # distance control on REAL non-members: they ignore the context by
            # construction, so this fraction is the statistic's chance level
            closer_real=(nearest_sq(non, ctx) < nearest_sq(non, ref)).mean(),
        )
        if has_gen:
            z = jax.random.normal(kz, (E, cell.q_mem, cfg.z_dim))
            a = generate_logits(pg, ctx, z, cfg)
            fake = jax.nn.sigmoid(a)
            s_gen = disc_logits(pd, ctx, fake, cfg)
            d_ctx, d_ref = nearest_sq(fake, ctx), nearest_sq(fake, ref)
            # Evaluation only: the same queries scored against ANOTHER
            # episode's context. If D uses its memory, telling context images
            # from generated ones should get worse with the wrong memory.
            ctx_sw = jnp.roll(ctx, 1, axis=0)
            s_mem_sw = disc_logits(pd, ctx_sw, mem, cfg)
            s_gen_sw = disc_logits(pd, ctx_sw, fake, cfg)
            if cell.arm == "ctx":
                # Evaluation only: how much does the generator's output depend
                # on its context? Output change when the context is swapped,
                # relative to the change when the noise is redrawn (same
                # context). 0 = ignores the context.
                z2 = jax.random.normal(jax.random.fold_in(kz, 1), z.shape)
                dz = jnp.abs(fake - generate(pg, ctx, z2, cfg)).mean()
                dc = jnp.abs(fake - generate(pg, ctx_sw, z, cfg)).mean()
                out["g_ctx_dep"] = dc / jnp.maximum(dz, 1e-8)
            out |= dict(
                auroc_gen_swap=auroc(s_mem_sw, s_gen_sw),
                acc_mem_swap=(s_mem_sw > 0).mean(),
                acc_gen_swap=(s_gen_sw < 0).mean(),
                auroc_gen=auroc(s_mem, s_gen),
                acc_gen=(s_gen < 0).mean(),
                p_gen=jax.nn.sigmoid(s_gen).mean(),
                closer_gen=(d_ctx < d_ref).mean(),
                d_ctx_gen=d_ctx.mean(),
                d_ref_gen=d_ref.mean(),
                gen_std=fake.std(axis=(0, 1)).mean(),       # pixel spread across samples; 0 = collapse
                sat=(jnp.abs(a) > 6.0).mean(),              # output pixels with a dead sigmoid gradient
            )
        return out | twin

    def evaluate_mixed(pd, pg, pool, ke, kz):
        E, n = cell.eval_episodes, cell.q_mem
        ctx, mem_r, mem_g, non, non_lab, z, ctx_r, ref = mixed_episode(ke, pool, pg, E, n, n, n)
        if frozen:                          # split the frozen fresh images back out
            non, fresh0, non_lab = non[:, :n], non[:, n:], non_lab[:, :n]
        a = generate_logits(pg, ctx, z, cfg)
        fresh = jax.nn.sigmoid(a)
        s_mr, s_mg = disc_logits(pd, ctx, mem_r, cfg), disc_logits(pd, ctx, mem_g, cfg)
        s_non, s_gen = disc_logits(pd, ctx, non, cfg), disc_logits(pd, ctx, fresh, cfg)
        keep = (non_lab < 0.5).astype(jnp.float32)
        d_ctx, d_ref = nearest_sq(fresh, ctx_r), nearest_sq(fresh, ref)
        return dict(
            auroc_real=auroc(s_mr, s_non, keep),       # stored real vs unstored real
            # stored generated vs fresh generated FROM THE SAME SOURCE: the
            # trained G in exp5, the frozen G when there is one
            auroc_genmem=auroc(s_mg, disc_logits(pd, ctx, fresh0, cfg) if frozen else s_gen),
            acc_mem=(s_mr > 0).mean(),
            acc_genmem=(s_mg > 0).mean(),
            acc_non=jnp.sum((s_non < 0) * keep) / jnp.sum(keep),
            acc_gen=(s_gen < 0).mean(),
            p_gen=jax.nn.sigmoid(s_gen).mean(),
            # distance to the REAL part of the context only: the generated part
            # is G's own output, which a collapsed G would trivially be near
            closer_gen=(d_ctx < d_ref).mean(),
            closer_real=(nearest_sq(non, ctx_r) < nearest_sq(non, ref)).mean(),
            d_ctx_gen=d_ctx.mean(), d_ref_gen=d_ref.mean(),
            gen_std=fresh.std(axis=(0, 1)).mean(),
            sat=(jnp.abs(a) > 6.0).mean(),
        )

    return block, evaluate, opt_d, opt_g


def run_cell(cell: Cell, pools) -> dict:
    train_pool, held_pool, held_y = pools
    cfg = Cfg(m_ctx=cell.m_ctx, d_act=cell.d_act, g_ln=cell.g_ln)
    kd, kg, kt, kev = jax.random.split(jax.random.PRNGKey(cell.seed), 4)
    pd = init_plain_disc(kd, cfg) if cell.disc == "plain" else init_disc(kd, cfg)
    pg = init_gen(kg, cfg, use_ctx=cell.arm == "ctx") if cell.arm != "none" else {}
    if cell.init_gen_from:
        with open(PROJECT / f"params_{cell.init_gen_from}.pkl", "rb") as f:
            src = jax.tree_util.tree_map(jnp.asarray, pickle.load(f)["gen"])
        Z = cfg.z_dim
        for k in ("b1", "W2", "b2", "W3", "b3"):
            assert pg[k].shape == src[k].shape, k
            pg[k] = src[k]
        pg["W1"] = jnp.zeros_like(pg["W1"]).at[:Z].set(src["W1"])
    block, evaluate, opt_d, opt_g = make_fns(cell, cfg)
    sd, sg = opt_d.init(pd), opt_g.init(pg)

    hist: dict[str, list] = {"step": [], "loss_d": [], "loss_g": [], "g_grad": []}
    class_probe = make_class_probe(held_pool, held_y, cfg) if cell.disc == "membership" else None
    # with a longer training context, also probe with contexts of that length
    class_probe_m = (make_class_probe(held_pool, held_y, cfg, n_ctx=cell.m_ctx)
                     if cell.disc == "membership" and cell.m_ctx != 16 else None)

    def log_eval(step, ld, lg, gn):
        ev = {k: float(v) for k, v in evaluate(pd, pg, held_pool, kev).items()}
        if class_probe is not None:
            ev["class_acc"] = float(class_probe(pd))
        if class_probe_m is not None:
            ev["class_acc_m"] = float(class_probe_m(pd))
        hist["step"].append(step)
        hist["loss_d"].append(ld)
        hist["loss_g"].append(lg)
        hist["g_grad"].append(gn)
        for k, v in ev.items():
            hist.setdefault(k, []).append(v)
        logging.info(f"[{cell.run}] step {step:6d}  Ld {ld:.4f}  Lg {lg:.4f}  |gG| {gn:.3g}  "
                     + "  ".join(f"{k} {v:.3f}" for k, v in ev.items()))
        return ev

    t0 = time.time()
    ev = log_eval(0, float("nan"), float("nan"), float("nan"))
    for b in range(cell.steps // cell.eval_every):
        pd, pg, sd, sg, ld, lg, gn = block(pd, pg, sd, sg, train_pool, jax.random.fold_in(kt, b))
        ev = log_eval((b + 1) * cell.eval_every, float(ld), float(lg), float(gn))
    elapsed = time.time() - t0

    ckpt = PROJECT / f"params_{cell.run}.pkl"
    with open(ckpt, "wb") as f:
        pickle.dump({"disc": jax.device_get(pd), "gen": jax.device_get(pg), "cfg": cfg._asdict(),
                     "cell": asdict(cell)}, f)

    row = dict(experiment=cell.exp, run=cell.run, **asdict(cell), cfg=cfg._asdict(),
               state_floats=cfg.state_floats, n_params_d=n_params(pd), n_params_g=n_params(pg),
               time_s=round(elapsed, 1), **{f"final_{k}": v for k, v in ev.items()}, history=hist)
    try:
        from lib.viz import sample_grid
        row["samples_url"] = sample_grid(cell, cfg, pg, held_pool) if cell.arm != "none" else None
    except Exception as e:                  # a figure must never cost the row
        logging.warning(f"[{cell.run}] sample grid failed: {e}")
    return row


def run_cells(cells: list[Cell]) -> None:
    done = done_runs()
    todo = [c for c in cells if c.run not in done]
    for c in cells:
        if c.run in done:
            logging.info(f"{c.run} already done — skipping")
    if not todo:
        return
    pools = load_pools()
    for c in todo:
        append_result(run_cell(c, pools))
        logging.info(f"{c.run} appended")
