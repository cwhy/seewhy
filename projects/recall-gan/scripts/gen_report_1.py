"""Report 1: where recall-gan stands after exp1-exp8.

Builds every figure from results.jsonl and the saved checkpoints, assembles the
markdown, publishes it, and writes the same string to reports/01-progress.md.

Run on the GPU box (needs the checkpoints and the R2 credentials):
    ~/.local/bin/uv run --no-sync python projects/recall-gan/scripts/gen_report_1.py
"""
import json
import pickle
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.append(str(PROJECT.parents[1]))          # repo root LAST (stray datasets.py on the box)

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from lib.model import Cfg, disc_logits, generate  # noqa: E402
from shared_lib.datasets import load_supervised_image  # noqa: E402
from shared_lib.media import save_matplotlib_figure  # noqa: E402
from shared_lib.report import save_report  # noqa: E402

V = "v2"
OUT_MD = PROJECT / "reports" / "01-progress.md"
FROZEN = "exp3_blind_plain_m16_s42_r1"
GREEN, RED, GREY = "#1a9850", "#d73027", "#777777"
C_BLIND, C_CTX = "#4575b4", "#f46d43"

# ── data ─────────────────────────────────────────────────────────────────────

ROWS = {}
for line in (PROJECT / "results.jsonl").read_text().splitlines():
    r = json.loads(line)
    ROWS.setdefault(r["run"], r)


def row(run):
    return ROWS[run]


def seeds(fmt, field):
    v = [row(fmt.format(s=s))[f"final_{field}"] for s in (42, 43, 44)]
    return float(np.mean(v)), float(min(v)), float(max(v))


d = load_supervised_image("mnist")
TE = np.asarray(d.X_test, np.float32).reshape(len(d.X_test), -1)
TR = np.asarray(d.X, np.float32).reshape(len(d.X), -1)
if TE.max() > 1.5:
    TE, TR = TE / 255.0, TR / 255.0
YTE, YTR = np.asarray(d.y_test), np.asarray(d.y)
REAL_STD = float(TE.std(0).mean())


def load(run):
    ck = pickle.load(open(PROJECT / f"params_{run}.pkl", "rb"))
    t = lambda p: jax.tree_util.tree_map(jnp.asarray, p)
    return t(ck["disc"]), t(ck["gen"]), Cfg(**ck["cfg"])


def save(name, fig):
    url = save_matplotlib_figure(f"recall-gan_r01_{name}_{V}", fig, format="png", dpi=130)
    plt.close(fig)
    return url


def show(ax, img, edge=None, cmap="gray"):
    ax.imshow(np.asarray(img).reshape(28, 28), cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(edge is not None)
        if edge:
            s.set_edgecolor(edge); s.set_linewidth(3)


# ── figure 1: one episode, scored by the final discriminator ──────────────────

def fig_episode():
    """One held-out episode, as the exp8 setup builds it, scored by exp8's
    context-aware run (seed 44). Episode = the first draw of a fixed RNG; not
    chosen."""
    pd, pg, cfg = load("exp8_ctx_m16_s44_warm_pen1")
    _, pg0, cfg0 = load(FROZEN)
    rng = np.random.default_rng(7)
    idx = rng.choice(len(TE), 12 + 2, replace=False)
    real_ctx, real_non = TE[idx[:12]], TE[idx[12:]]
    k = jax.random.split(jax.random.PRNGKey(7), 3)
    gen_ctx = np.asarray(generate(pg0, None, jax.random.normal(k[0], (1, 4, 64)), cfg0))[0]
    fresh0 = np.asarray(generate(pg0, None, jax.random.normal(k[1], (1, 1, 64)), cfg0))[0]
    order = rng.permutation(16)
    ctx = np.concatenate([real_ctx, gen_ctx])[order]
    src = np.array(["real"] * 12 + ["gen"] * 4)[order]
    trained = np.asarray(generate(pg, jnp.asarray(ctx[None]), jax.random.normal(k[2], (1, 1, 64)), cfg))[0]
    queries = [(real_ctx[0], "stored real", True), (real_ctx[5], "stored real", True),
               (gen_ctx[1], "stored\nfrozen-G image", True), (real_non[0], "unstored real", False),
               (real_non[1], "unstored real", False), (fresh0[0], "fresh\nfrozen-G image", False),
               (trained[0], "trained-G\nimage", False)]
    q = jnp.asarray(np.stack([x for x, _, _ in queries])[None])
    p = np.asarray(jax.nn.sigmoid(disc_logits(pd, jnp.asarray(ctx[None]), q, cfg)))[0]

    fig = plt.figure(figsize=(12, 4.6))
    fig.text(0.01, 0.95, "Context: 16 images written into the discriminator's memory "
             "(orange frame = made by the frozen generator)", fontsize=11)
    for i in range(16):
        ax = fig.add_axes([0.01 + i * 0.061, 0.62, 0.055, 0.28])
        show(ax, ctx[i], edge="#fdae61" if src[i] == "gen" else None)
    fig.text(0.01, 0.47, "Queries: was this image in the context?  "
             "Green frame = yes, red = no.  Number = discriminator's P(in context).", fontsize=11)
    for i, (img, name, member) in enumerate(queries):
        ax = fig.add_axes([0.03 + i * 0.138, 0.12, 0.09, 0.30])
        show(ax, img, edge=GREEN if member else RED,
             cmap="magma" if "trained" in name else "gray")
        ax.set_title(f"{p[i]:.2f}", fontsize=13, fontweight="bold",
                     color=GREEN if (p[i] > 0.5) == member else RED)
        ax.set_xlabel(name, fontsize=9)
    return save("episode", fig), p


# ── figure 2: generator diversity over training, one line per fix ─────────────

STAGES = [
    ("exp1: original game", "exp1_ctx_m16_s42", GREY, ":"),
    ("exp2: plain GAN, no penalty", "exp2_blind_plain_m16_s42", "#999999", "--"),
    ("exp3: plain GAN + gradient penalty", "exp3_blind_plain_m16_s42_r1", "#1a9850", "-"),
    ("exp5: own outputs planted in context", "exp5_ctx_m16_s42_g4_pen1", "#d73027", "--"),
    ("exp7: frozen-G context, cold start", "exp7_ctx_m16_s42_frozen_pen1", "#fdae61", "-."),
    ("exp8: frozen-G context, warm start", "exp8_ctx_m16_s42_warm_pen1", C_CTX, "-"),
]


def fig_diversity():
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for name, run, c, ls in STAGES:
        h = row(run)["history"]
        ax.plot(h["step"], h["gen_std"], color=c, ls=ls, lw=2.2, label=name)
    ax.axhline(REAL_STD, color="k", lw=1, ls=":")
    ax.text(200, REAL_STD + 0.006, f"real MNIST: {REAL_STD:.2f}", fontsize=9)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("training step"); ax.set_ylabel("variety across samples\n(pixel std; 0 = one image)")
    ax.set_ylim(-0.01, 0.36)
    ax.legend(fontsize=8.5, loc="upper right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return save("diversity", fig)


# ── figure 3: what the generators draw ────────────────────────────────────────

def sample_panel(ax_grid, run, ctx):
    _, pg, cfg = load(run)
    z = jax.random.normal(jax.random.PRNGKey(0), (ctx.shape[0], 8, 64))
    g = np.asarray(generate(pg, jnp.asarray(ctx), z, cfg))
    for i in range(ctx.shape[0]):
        for j in range(8):
            show(ax_grid[i][j], g[i, j], cmap="magma")


def fig_samples():
    rng = np.random.default_rng(1)
    ctx = TE[rng.choice(len(TE), (4, 16))]
    runs = [("exp1: original game — collapsed", "exp1_ctx_m16_s42"),
            ("exp3: plain GAN with gradient penalty", FROZEN),
            ("exp8: final game, context-aware", "exp8_ctx_m16_s44_warm_pen1")]
    fig = plt.figure(figsize=(13, 2.9))
    for p, (title, run) in enumerate(runs):
        grid = [[fig.add_axes([0.01 + p * 0.335 + j * 0.039, 0.66 - i * 0.2, 0.036, 0.18])
                 for j in range(8)] for i in range(4)]
        sample_panel(grid, run, ctx)
        fig.text(0.01 + p * 0.335, 0.9, title, fontsize=10.5)
    return save("samples", fig)


# ── figure 4: the gradient penalty's cost to membership ───────────────────────

def fig_penalty():
    pens = [0, 1, 10]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, field, lab in [(axes[0], "auroc_real", "membership AUROC\n(stored vs unstored real)"),
                           (axes[1], "acc_non", "unstored real images\ncorrectly rejected")]:
        m = [seeds(f"exp6_none_m16_s{{s}}_pen{p}", field) for p in pens]
        ax.bar(range(3), [x[0] for x in m], color=["#4d4d4d", "#878787", "#d6604d"], width=0.6,
               yerr=[[x[0] - x[1] for x in m], [x[2] - x[0] for x in m]], capsize=4)
        for i, x in enumerate(m):
            ax.text(i, x[0] + 0.01, f"{x[0]:.3f}", ha="center", fontsize=10)
        ax.axhline(0.5, color="k", ls=":", lw=1); ax.text(2.35, 0.51, "chance", fontsize=8)
        ax.set_xticks(range(3), [f"penalty {p}" for p in pens]); ax.set_ylim(0.4, 1.08)
        ax.set_ylabel(lab); ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("No generator. 3 seeds per bar (whiskers = min–max).", fontsize=10)
    fig.tight_layout()
    return save("penalty", fig)


# ── figure 5: context use over training ───────────────────────────────────────

def fig_closer():
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    for arm, c, lab in [("blind", C_BLIND, "context-blind generator"), ("ctx", C_CTX, "context-aware generator")]:
        for i, s in enumerate((42, 43, 44)):
            h = row(f"exp8_{arm}_m16_s{s}_warm_pen1")["history"]
            ax.plot(h["step"], h["closer_gen"], color=c, lw=2, alpha=0.85, label=lab if i == 0 else None)
    ch = row("exp8_ctx_m16_s42_warm_pen1")["final_closer_real"]
    ax.axhline(ch, color="k", ls=":", lw=1)
    ax.text(300, ch - 0.035, f"chance ({ch:.2f}, measured on real non-members)", fontsize=9)
    ax.set_ylim(0.35, 0.85); ax.set_xlabel("training step")
    ax.set_ylabel("share of samples nearer their own\ncontext than an unrelated set")
    ax.legend(frameon=False, loc="upper left"); ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("exp8, gradient penalty 1, 3 seeds per generator", fontsize=10)
    fig.tight_layout()
    return save("closer", fig)


# ── figure 6: same noise, add a "1" to the context ────────────────────────────

def fig_ones():
    _, pg, cfg = load("exp8_ctx_m16_s44_warm_pen1")
    rng = np.random.default_rng(3)
    no1, ones = np.where(YTE != 1)[0], np.where(YTE == 1)[0]
    n_ep = 3
    base = TE[np.stack([rng.choice(no1, 16, replace=False) for _ in range(n_ep)])]
    with1 = base.copy(); with1[:, 0] = TE[rng.choice(ones, n_ep)]
    z = jax.random.normal(jax.random.PRNGKey(5), (n_ep, 8, 64))
    g0 = np.asarray(generate(pg, jnp.asarray(base), z, cfg))
    g1 = np.asarray(generate(pg, jnp.asarray(with1), z, cfg))

    fig = plt.figure(figsize=(13, 1.25 * 2 * n_ep + 0.6))
    H = 1.0 / (2 * n_ep + 0.6)
    for e in range(n_ep):
        for k, (ctx, g, lab) in enumerate([(base[e], g0[e], "no 1 in context"),
                                            (with1[e], g1[e], "one image swapped for a 1")]):
            y = 1 - (2 * e + k + 1) * H - 0.02
            for i in range(16):
                show(fig.add_axes([0.005 + i * 0.033, y, 0.03, H * 0.85]), ctx[i],
                     edge=GREEN if (k == 1 and i == 0) else None)
            fig.text(0.54, y + H * 0.35, "→", fontsize=18)
            for j in range(8):
                show(fig.add_axes([0.57 + j * 0.053, y, 0.05, H * 0.85]), g[j], cmap="magma")
            fig.text(0.005, y + H * 0.88, f"episode {e + 1}, {lab}", fontsize=8.5)
            fig.text(0.57, y + H * 0.88, "8 samples (same noise in both rows of an episode)", fontsize=8.5)
    return save("ones", fig)


def ones_table():
    """P(sample's nearest training image is a 1), contexts with no 1 vs one 1."""
    bank, bank_y = jnp.asarray(TR[:20000]), YTR[:20000]
    sq = (bank ** 2).sum(-1)

    def nn_is_one(x):
        return np.concatenate([np.asarray(jnp.argmin(sq[None] - 2 * x[c:c + 512] @ bank.T, 1))
                               for c in range(0, x.shape[0], 512)])

    rng = np.random.default_rng(0)
    no1, ones = np.where(YTE != 1)[0], np.where(YTE == 1)[0]
    E = 512
    base = np.stack([rng.choice(no1, 16, replace=False) for _ in range(E)])
    w1 = base.copy(); w1[:, 0] = rng.choice(ones, E)
    out = {}
    for arm in ("blind", "ctx"):
        for s in (42, 43, 44):
            _, pg, cfg = load(f"exp8_{arm}_m16_s{s}_warm_pen1")
            z = jax.random.normal(jax.random.PRNGKey(s), (E, 8, 64))
            r = []
            for idx in (base, w1):
                g = generate(pg, jnp.asarray(TE[idx]), z, cfg).reshape(-1, 784)
                r.append(float((bank_y[nn_is_one(g)] == 1).mean()))
            out[(arm, s)] = r
    return out


def fig_ones_bars(tab):
    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    labels, x = [], 0
    for arm, c in [("blind", C_BLIND), ("ctx", C_CTX)]:
        for s in (42, 43, 44):
            a, b = tab[(arm, s)]
            ax.bar(x - 0.2, a, 0.38, color=c, alpha=0.35)
            ax.bar(x + 0.2, b, 0.38, color=c)
            labels.append(f"{'blind' if arm == 'blind' else 'context-aware'}\nseed {s}")
            x += 1
    ax.set_xticks(range(6), labels, fontsize=8.5)
    ax.set_ylabel("share of samples\nthat are a 1")
    ax.set_title("pale = context has no 1;  solid = one context image swapped for a 1   "
                 "(512 held-out contexts)", fontsize=9.5)
    ax.set_ylim(0, 0.8); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return save("ones_bars", fig)


# ── assemble ──────────────────────────────────────────────────────────────────

def main():
    u_ep, p_ep = fig_episode()
    u_div, u_samp, u_pen = fig_diversity(), fig_samples(), fig_penalty()
    u_close, u_ones = fig_closer(), fig_ones()
    tab = ones_table()
    u_bars = fig_ones_bars(tab)

    f = lambda run, k: row(run)[f"final_{k}"]
    S = lambda fmt, k: seeds(fmt, k)
    s8c = "exp8_ctx_m16_s{s}_warm_pen1"
    s8b = "exp8_blind_m16_s{s}_warm_pen1"
    rng3 = lambda t: f"{t[0]:.3f} (seeds {t[1]:.2f}–{t[2]:.2f})"
    gm = S(s8c, "auroc_genmem"); ar = S(s8c, "auroc_real")
    cc, cb = S(s8c, "closer_gen"), S(s8b, "closer_gen")
    pc, pb = S(s8c, "p_gen"), S(s8b, "p_gen")
    ch = f("exp8_ctx_m16_s42_warm_pen1", "closer_real")

    md = f"""# The discriminator wins; the generator learns one trick

**Status after eight experiments.** A discriminator that sees the context only through one KDA memory
learned to tell stored images from unstored ones almost perfectly. Among images from the *same* generator,
it separates stored from fresh at AUROC **{gm[0]:.3f}** (chance 0.5). On real images it scores
**{ar[0]:.3f}**. The generator took six fixes to get playing at all. Once playing, the context-aware one
uses its context: **{cc[0]:.2f}** of its samples sit nearer their own context than an unrelated set,
against **{cb[0]:.2f}** for the context-blind one and chance **{ch:.2f}**. But what it learned is narrow.
In two of three seeds it is one rule: *if the context holds a 1, draw a 1.* The discriminator still
rejects {100 * (1 - pc[0]):.0f}% of its images.

All numbers: MNIST, 16-image contexts, held-out test-split episodes, 20k training steps unless stated.
"Seeds" means three independent runs (42, 43, 44).

## Where each piece came from

Several pieces of the setup were added by me (Claude) while running on my own. Marked here so
nothing reads as agreed when it was not.

| piece | source |
|---|---|
| discriminator = one KDA memory; its final state decides "was this in the context?" | **yours** |
| generator without the context; also a version with it | **yours** |
| the discriminator should win; it should be a proper density estimator over all images | **yours** |
| exp1: generated images always "not in context", plus the two diagnostics | agreed in discussion |
| plain-GAN check (exp2) after you doubted the failure was the idea's | agreed in discussion |
| **gradient penalty**: extra loss that keeps the discriminator's score from changing too sharply near stored images | introduced by me; then discussed (you asked what it is and whether modern GANs use it) |
| putting generated images into the context | mine — proposed, never agreed |
| **frozen generator**: a second generator, trained once as a plain GAN (exp3) and then locked; it only supplies fake images for the context | mine, not discussed |
| **warm start**: the generator being trained begins as a copy of the frozen generator, not from random weights | mine, not discussed |
| penalty sweep, 3 seeds, the "add a 1" test; MNIST, 16 images, network sizes | mine, not discussed |

## 1. The game, in one picture

The discriminator reads 16 images into its memory, then answers "was this image one of them?" for each
query. The generator makes images and is rewarded when the discriminator wrongly says yes.

The final setup (exp8) adds two things of mine. First, 4 of the 16 context images are fakes from the
**frozen generator** — a separate generator trained earlier and then locked. Fakes are now sometimes in
the context and sometimes not, so "looks fake" no longer tells the discriminator the answer. Second, the
generator being trained starts as a copy of that frozen generator (**warm start**).

![One episode]({u_ep})

*One held-out episode (first draw of a fixed random seed, not chosen), scored by the exp8 context-aware run,
seed 44. Bold number = the discriminator's probability that the query was in the context; green when on the
right side of 0.5. The two "frozen-G" images come from the same frozen generator; only one of them was
stored. The last query is the trained generator's attempt.*

## 2. What each experiment changed

| exp | what changed | what happened |
|---|---|---|
| 1 | the game as first proposed: generated images always labelled "not in context" | membership perfect (AUROC 1.000 without a generator); both generators collapsed to one noisy blob |
| 2 | **control:** same generator in a plain real-vs-fake GAN | collapsed too → exp1's failure was my GAN setup, not the idea |
| 3 | plain GAN + one stabiliser at a time | only the **gradient penalty** (weight 10) worked: varied, readable digits |
| 4 | exp1 + gradient penalty | generator alive, but the discriminator won by spotting "looks generated" and called 69% of unstored real images "stored" |
| 5 | 4 of 16 context slots filled with the generator's *own* images | every generator collapsed: one image, planted in every context, can never be rejected |
| 6 | penalty weight 0 / 1 / 10, no generator | the penalty costs membership: 99.7% → 94.6% → 77.5% of unstored images rejected |
| 7 | generated context images from a *frozen* generator | discriminator: AUROC 1.000 on stored vs fresh generated images; trained generator still collapsed (11 of 12 runs) |
| 8 | exp7 + the trained generator starts as a copy of the frozen one | **penalty 1: generator alive, context-aware one uses its context.** Penalty 0: still collapses |

## 3. Getting the generator to play

The generator's variety over training shows the story. A line at zero is a generator drawing one image.

![Generator variety over training]({u_div})

*Seed 42 for every line. Real MNIST's own variety is the dotted line.*

Three different things killed it, in order:

1. **An unstable GAN.** Without the gradient penalty, the discriminator became a cliff: certain on real
   images, certain on everything else, with no slope for the generator to follow. Even a plain GAN collapsed
   (exp2). The penalty keeps the discriminator's score smooth near real images (exp3).
2. **A shortcut.** With generated images always "not in context", the discriminator could win by
   recognising generated images, without reading its memory (exp4).
3. **A loophole I built.** Putting the generator's own images in the context let a collapsed generator plant
   its single image everywhere, so it could never be rejected (exp5). A frozen, separate generator for the
   context images closed it (exp7). Starting the trained generator as a copy of that frozen one got it
   playing (exp8).

What the generators draw:

![Samples]({u_samp})

*Same four held-out contexts and the same noise for every panel; rows = contexts, which were drawn at
random and all happen to contain a 1. Left: exp1's collapse. Middle: the frozen generator. Right: exp8's
context-aware generator, drawing 1s. Section 5 shows what it draws when the context has no 1.*

## 4. The discriminator is a real membership test — with a price

On the final setup the discriminator is near-perfect on every kind of query:

| exp8, penalty 1, 3 seeds | context-aware generator | context-blind generator | chance |
|---|---|---|---|
| stored vs unstored **real** images (AUROC) | {rng3(ar)} | {rng3(S(s8b, "auroc_real"))} | 0.5 |
| stored vs fresh **frozen-generator** images (AUROC) | {rng3(gm)} | {rng3(S(s8b, "auroc_genmem"))} | 0.5 |
| trained generator's images rejected | {rng3(S(s8c, "acc_gen"))} | {rng3(S(s8b, "acc_gen"))} | 0.5 |

The generator needs the gradient penalty, and the penalty works against membership. Membership needs the
discriminator to react sharply to small differences; the penalty smooths exactly that away.

![Penalty cost]({u_pen})

## 5. The generator reads its context — for one thing

Context use grows through training for every context-aware seed, and stays at chance for every blind one:

![Context use over training]({u_close})

What it reads, shown directly. Each episode: a context with no 1, then the same context with one image
swapped for a 1, and **the same noise** fed to the generator both times. Only the context changed.

![Same noise, add a 1]({u_ones})

*exp8 context-aware generator, seed 44. Three held-out episodes, first draws of a fixed seed. Green frame =
the swapped-in 1.*

Without a 1 in the context, this generator draws noisy blobs, not digits. It has not learned to draw
digits in general; it has learned to draw a 1 when that is likely to pay. The discriminator rejects the
blobs, which costs the generator little, because they are only a share of its attempts.

Across all six generators, on 512 held-out contexts:

![Share of 1s]({u_bars})

| exp8, penalty 1 | share of samples that are a 1: context without a 1 → with one |
|---|---|
""" + "\n".join(
        f"| {'context-blind' if a == 'blind' else 'context-aware'}, seed {s} | {tab[(a, s)][0]:.3f} → {tab[(a, s)][1]:.3f} |"
        for a in ("blind", "ctx") for s in (42, 43, 44)) + f"""

Seeds 43 and 44 learned "a 1 in the context → draw a 1". Seed 42 does not react to a 1. It still sits
above chance on the training mix ({f("exp8_ctx_m16_s42_warm_pen1", "closer_gen"):.2f}), so it reads
something else, not yet identified. Using the context pays: the context-aware generator fools the
discriminator on {100 * pc[0]:.1f}% of its images, the blind one on {100 * pb[0]:.1f}%.

## 6. Pending conclusions

- **The discriminator side of the idea works.** A single KDA memory of 16k numbers supports a near-perfect
  "was this stored?" test for 16 images, including among images from one generator (AUROC {gm[0]:.3f}).
- **"Discriminator wins" is the outcome, and by a wide margin.** Even the best generator is rejected
  {100 * (1 - pc[0]):.0f}% of the time.
- **A generator can learn to read the context to attack it.** Not by copying images, but by matching a
  class — in two of three seeds.
- **Every earlier "the generator can't" was a setup problem.** Each failure in exp1–exp7 traced to the GAN
  recipe or to my episode design, not to the idea.

## 7. Open questions

1. **Why 1s?** Guess: 1s look alike, so an unstored 1 is the easiest image to mistake for a stored one —
   the discriminator's blind spot. Testable now: its error rate on unstored 1s vs other digits.
2. **What does seed 42 read?** It uses the context, but not via 1s.
3. **Can the generator do more than one trick?** It found one class and stopped. Longer training, or a
   reward for variety, may push it to other weak spots.
4. **Can we keep the generator alive without the penalty**, so membership stays at full strength?
5. **Larger contexts.** 16 images is far below the memory's size. Where does membership start to fail, and
   does the generator find it first?

## What this changes

- The next run should test question 1 directly. If 1s are the blind spot, the game is doing what it was
  built for: the generator finds where the discriminator is weakest.
- The standard setup from here on: frozen-generator context images, warm start, gradient penalty 1.

*Source: `projects/recall-gan/experiments1-8.py`, `results.jsonl`, `concepts.md`. Generator:
`scripts/gen_report_1.py`.*
"""
    OUT_MD.write_text(md, encoding="utf-8")
    print("report:", save_report("recall-gan_report_01_progress", md))
    print("episode probs:", np.round(p_ep, 3))


if __name__ == "__main__":
    main()
