"""What a token IS, per dataset — the one thing the recall-gen task is generic in.

Every experiment in this project is a KDA linear RNN reading M context tokens
and completing a masked query token. Nothing in that machinery is about MNIST:
it needs a flat vector per item, a mask saying which coordinates are hidden, a
class label to split on, and a way to draw the thing. This module supplies those
four for each dataset and nothing else, so `lib/train.py` never learns what a
chess board is.

Three domains:

    mnist          784 = 28x28 grey pixels, bottom 14 rows hidden, classes are
                   the ten digits. The original task; kept byte-identical so
                   every existing row in results.jsonl still reproduces.
    fashion_mnist  the same shape and mask over Fashion-MNIST. Classes 0-4 are
                   garments (t-shirt, trouser, pullover, dress, coat), 5-9 are
                   mostly footwear and bags — a wider class gap than 0-4 / 5-9
                   in MNIST.
    chess          832 = 8x8x13 one-hot piece planes, the four QUEENSIDE files
                   hidden, classes are game-phase buckets by piece count.

The chess mask is a slice on files rather than ranks because a rank mask hides
one player's whole army: with ranks 1-4 gone the visible half says almost
nothing about what is missing, and the completion arm of the task degenerates.
Files a-d against e-h keeps both kings' neighbourhoods split across the boundary
so material and structure on one wing genuinely constrain the other.
"""

import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.patheffects as pe

sys.path.append(str(Path(__file__).resolve().parents[3]))
from shared_lib.datasets import (load_supervised_image, load_chess_positions,
                                 fen_to_planes, chess_piece_count, CHESS_PIECES)


class Domain(NamedTuple):
    name: str
    d_in: int                  # flat vector length of one token
    shape: tuple               # unflattened shape, for masks and figures
    mask_axis: int             # which axis of `shape` the mask slices
    n_mask_default: int        # default number of slices hidden
    class_names: dict          # class id -> a name a figure can print
    kind: str                  # "image" | "board" — which renderer a figure uses
    # (training classes, novel classes) — this domain's standard split, so a
    # script that only knows the domain name can build the six conditions.
    split: tuple = ((0, 1, 2, 3, 4), (5, 6, 7, 8, 9))
    # Content width, when it is narrower than `d_in`. A padded domain lays its
    # real coordinates in [0, d_real) and leaves the rest at zero forever, so one
    # network can be scored on domains of different natural widths. None means
    # the domain fills its own width and nothing is padded.
    d_real: int | None = None


# ── class vocabularies ────────────────────────────────────────────────────────

MNIST_CLASSES = {i: str(i) for i in range(10)}

FASHION_CLASSES = {
    0: "t-shirt", 1: "trouser", 2: "pullover", 3: "dress", 4: "coat",
    5: "sandal", 6: "shirt", 7: "sneaker", 8: "bag", 9: "ankle boot",
}

# Phase buckets by piece count. The split this project runs is class 0 as the
# training pool and class 2 as the novel pool, so the middle bucket exists only
# to keep the two ends genuinely far apart rather than adjacent.
CHESS_CLASSES = {0: "24+ pieces", 1: "11-23 pieces", 2: "<=10 pieces"}
CHESS_PHASE_EDGES = (24, 11)          # >=24 -> 0, >=11 -> 1, else 2

# A CROSS-DATASET pair. The novel pool is not a held-out class of the training
# set, it is a different dataset entirely, and the split machinery does not need
# to know that: the novel dataset's ten labels are shifted up by ten and the two
# test splits are concatenated, so "novel class" and "novel dataset" are the same
# operation. Classes 0-9 are the training dataset, 10-19 the novel one.
#
# This works only because both datasets are 28x28 grey images, so one token means
# the same thing on each side and the identical network can be scored on both.
PAIR_SHIFT = 10
PAIRED_SPLIT = (tuple(range(10)), tuple(range(10, 20)))


def _paired_classes(train_names: dict, novel_names: dict) -> dict:
    return {**train_names,
            **{k + PAIR_SHIFT: v for k, v in novel_names.items()}}


F2M_CLASSES = _paired_classes(FASHION_CLASSES, {i: f"digit {i}" for i in range(10)})
M2F_CLASSES = _paired_classes(MNIST_CLASSES, FASHION_CLASSES)


# The common width every padded domain is scored at. 832 rather than a rounder
# number because it is exactly chess's 8x8x13 — chess is the widest real domain,
# so nothing is padded that does not have to be, and 832 = 64 x 13 keeps the
# prior's simplex mode aligned to whole groups of 13.
PAD_W = 832
SYNTH_WORLDS = 1024          # distinct generative worlds in the training pool
SYNTH_PER_WORLD = 48         # items drawn from each; >= M + Q with room to spare
SYNTH_TEST_WORLDS = 128      # worlds held out entirely — the E/F band
SYNTH_NONLIN = ("identity", "tanh", "relu", "sign")
# name -> (training worlds, items per world). "synth" is the TabPFN analogue:
# many worlds, so an episode's sixteen items share a generative process the
# network has to infer. "synth1" is the control with a single world, where that
# process can instead be memorised in the weights.
# name -> (training worlds, items per world, P(simplex world)).
#
# `synth_cont` is the ABLATION. `synth` draws 40% of its worlds in simplex mode —
# one active coordinate per group of 13, which is exactly a chess piece plane —
# so chess sits almost inside the prior's support while natural images do not.
# That is the leading explanation for why a synthetic-trained network retrieves
# chess far better than it retrieves fresh draws from its own prior. Removing
# the mode tests it: if chess transfer collapses, the prior was generating
# chess-shaped items; if it survives, the explanation is wrong.
# name -> (worlds, items per world, P(simplex), minimum latent dimension)
SYNTH_SHAPES = {"synth": (SYNTH_WORLDS, SYNTH_PER_WORLD, 0.4, 1.0),
                "synth1": (1, SYNTH_WORLDS * SYNTH_PER_WORLD, 0.4, 1.0),
                "synth_cont": (SYNTH_WORLDS, SYNTH_PER_WORLD, 0.0, 1.0),
                # Recall is ill-posed below k=8; see `_synth_world`.
                "synth_k8": (SYNTH_WORLDS, SYNTH_PER_WORLD, 0.4, 8.0),
                # Same prior, more items per world, so an episode can be built at
                # M=64. A world only holds `per_world` items and an episode draws
                # M+Q of them without replacement, so 48 caps M at 44.
                "synth_long": (SYNTH_WORLDS, 96, 0.4, 1.0)}
SYNTH_REAL_BASE = 10_000     # label offset for the real dataset in a synth pair


SYNTH_PAIRS = {}

DOMAINS = {
    "mnist": Domain("mnist", 784, (28, 28), 0, 14, MNIST_CLASSES, "image"),
    "fashion_mnist": Domain("fashion_mnist", 784, (28, 28), 0, 14,
                            FASHION_CLASSES, "image"),
    "chess": Domain("chess", 832, (8, 8, 13), 1, 4, CHESS_CLASSES, "board",
                    split=((0,), (2,))),
    # Train on Fashion-MNIST, test on MNIST. Same shape, different world.
    "fashion_to_mnist": Domain("fashion_to_mnist", 784, (28, 28), 0, 14,
                               F2M_CLASSES, "image", split=PAIRED_SPLIT),
    # The reverse direction, as the control that tells "distribution shift" apart
    # from "one of these two datasets is the easier thing to have learned".
    "mnist_to_fashion": Domain("mnist_to_fashion", 784, (28, 28), 0, 14,
                               M2F_CLASSES, "image", split=PAIRED_SPLIT),
}

# Padded copies, all 832 wide, so ONE network can be scored on every one of them.
# The real coordinates are unchanged; only the width is.
for _n, _cls in (("mnist", MNIST_CLASSES), ("fashion_mnist", FASHION_CLASSES)):
    DOMAINS[f"{_n}_pad"] = Domain(f"{_n}_pad", PAD_W, (28, 28), 0, 14, _cls,
                                  "image", d_real=784)

# The synthetic training domains, and one pair per real target. The network is
# TRAINED on the bare domain at full width with random masks; a pair only decides
# which dataset fills the B/D band when it is scored.
_SYNTH_TRAIN_CLS = tuple(range(SYNTH_WORLDS + SYNTH_TEST_WORLDS))
_SYNTH_SPLIT = (_SYNTH_TRAIN_CLS,
                tuple(range(SYNTH_REAL_BASE, SYNTH_REAL_BASE + 20)))
for _b in SYNTH_SHAPES:
    DOMAINS[_b] = Domain(_b, PAD_W, (8, 8, 13), 1, 4, {}, "board",
                         split=(_SYNTH_TRAIN_CLS, _SYNTH_TRAIN_CLS))
    for _t, _shape, _msk, _kind, _w in (
            ("mnist", (28, 28), 14, "image", 784),
            ("fashion_mnist", (28, 28), 14, "image", 784),
            ("chess", (8, 8, 13), 4, "board", PAD_W)):
        DOMAINS[f"{_b}_to_{_t}"] = Domain(
            f"{_b}_to_{_t}", PAD_W, _shape, 0 if _kind == "image" else 1, _msk,
            {}, _kind, split=_SYNTH_SPLIT, d_real=_w)
        SYNTH_PAIRS[f"{_b}_to_{_t}"] = (_b, _t)

PAIRS = {"fashion_to_mnist": ("fashion_mnist", "mnist"),
         "mnist_to_fashion": ("mnist", "fashion_mnist")}

# ── the synthetic prior ───────────────────────────────────────────────────────


def _synth_world(rng, width: int, n_items: int, p_simplex: float = 0.4,
                 k_min: float = 1.0) -> np.ndarray:
    """One generative world, and `n_items` drawn from it.

    A latent-factor model: items are a random map of a low-dimensional code, so
    the visible coordinates genuinely predict the hidden ones and the strength of
    that prediction is set by `k`.

    `k` is the dial the prior sweeps. Small k puts items on a low-dimensional
    manifold — completion easy, and items mutually similar so retrieval is hard.
    Large k makes items near-independent — retrieval easy, completion close to
    impossible. Sampling k per world spans the range the real datasets occupy
    instead of picking a point in it.
    """
    # `k_min` is the floor on the latent dimension, and it decides whether RECALL
    # is a well-posed question. Sixteen items drawn from a k-dimensional world
    # sit close together when k is small: at k=1, 42% of episodes have a nearest
    # rival within 0.1 of the target, so no reconstruction is precise enough to
    # name the right one. The median margin reaches MNIST's 0.63 around k=8.
    #
    # It is a real trade, not a free fix. Small-k worlds are exactly the ones
    # where COMPLETION is easy, because a low-dimensional manifold is what makes
    # a seventeenth item predictable from sixteen. Raising the floor buys a
    # well-posed recall task by removing the easy-completion regime.
    k = int(np.clip(np.exp(rng.uniform(np.log(k_min), np.log(64.0))), 1, 64))
    spec = (np.arange(1, k + 1) ** -rng.uniform(0.0, 2.0)).astype(np.float32)
    A = (rng.standard_normal((width, k)) * spec).astype(np.float32)
    b = (rng.standard_normal(width) * rng.uniform(0.0, 1.0)).astype(np.float32)
    h = rng.standard_normal((n_items, k)).astype(np.float32) @ A.T + b

    nl = SYNTH_NONLIN[rng.integers(len(SYNTH_NONLIN))]
    if nl == "tanh":
        h = np.tanh(h)
    elif nl == "relu":
        h = np.maximum(h, 0.0)
    elif nl == "sign":
        h = np.sign(h)
    h = h + rng.standard_normal(h.shape).astype(np.float32) * rng.uniform(0.0, 0.3)

    # Simplex mode puts chess inside the prior's support rather than outside it:
    # one active coordinate per group of 13, which is exactly a piece plane.
    if width % 13 == 0 and rng.random() < p_simplex:
        g = h.reshape(n_items, width // 13, 13)
        out = np.zeros_like(g)
        np.put_along_axis(out, g.argmax(-1)[..., None], 1.0, axis=-1)
        return out.reshape(n_items, width)
    lo, hi = h.min(), h.max()
    return ((h - lo) / max(float(hi - lo), 1e-6)).astype(np.float32)


def _synth_pool(width: int, n_worlds: int, per_world: int, seed: int,
                p_simplex: float = 0.4, k_min: float = 1.0):
    """(items, world_id). The world id is the label, so `ctx_mode="class"` builds
    episodes whose items all come from ONE world — which is what makes the task
    "infer this world from sixteen examples" rather than "recognise a fixed one".
    """
    rng = np.random.default_rng(seed)
    X = np.empty((n_worlds * per_world, width), np.float32)
    y = np.empty(n_worlds * per_world, np.int64)
    for w in range(n_worlds):
        lo = w * per_world
        X[lo:lo + per_world] = _synth_world(rng, width, per_world, p_simplex, k_min)
        y[lo:lo + per_world] = w
    return X, y


def _synthetic_pools(base: str = "synth", width: int = PAD_W):
    """Train worlds, and a disjoint set of test worlds for the E/F band.

    Test worlds are numbered after the train worlds, so no id is shared and
    "a world never seen" is exact rather than approximate.
    """
    n_w, per_w, p_sx, k_min = SYNTH_SHAPES[base]
    # Test worlds carry as many items as train worlds, so an episode built at a
    # given M works in both bands. `synth1` is the exception: its train pool is a
    # single world holding everything, and its test worlds keep the usual size.
    test_per_w = SYNTH_PER_WORLD if base == "synth1" else per_w
    Xtr, ytr = _synth_pool(width, n_w, per_w, seed=7, p_simplex=p_sx, k_min=k_min)
    Xte, yte = _synth_pool(width, SYNTH_TEST_WORLDS, test_per_w, seed=8,
                           p_simplex=p_sx, k_min=k_min)
    return Xtr, ytr, Xte, yte + n_w


def _synth_paired_pools(base: str, novel: str):
    """Synthetic worlds as the training pool, a real dataset as the novel one.

    Coordinates beyond the real domain's own width are zeroed in the SYNTHETIC
    items too. Without that, one domain would hold bands with different valid
    sets — the synthetic bands using all 832 coordinates and the real band only
    784 — and "visible" would mean two different things inside one comparison.
    Zeroing costs the synthetic bands a 48-coordinate slice and buys one
    consistent validity vector.
    """
    w = 784 if novel in ("mnist", "fashion_mnist") else PAD_W
    Xtr, ytr, Xs, ys = _synthetic_pools(base, PAD_W)
    Xtr, Xs = Xtr.copy(), Xs.copy()
    Xtr[:, w:] = 0.0
    Xs[:, w:] = 0.0
    if novel == "chess":
        _, _, Xr, yr = _chess_pools()
    else:
        _, _, Xr, yr = _image_pools(novel)
    Xr = np.concatenate([Xr, np.zeros((len(Xr), PAD_W - Xr.shape[1]), np.float32)], 1)
    return (Xtr, ytr,
            np.concatenate([Xs, Xr]),
            np.concatenate([ys, yr + SYNTH_REAL_BASE]))


def get(name: str) -> Domain:
    if name not in DOMAINS:
        raise ValueError(f"unknown domain {name!r}; have {sorted(DOMAINS)}")
    return DOMAINS[name]


# ── masks ─────────────────────────────────────────────────────────────────────

def content_width(domain: str) -> int:
    """How many of a domain's `d_in` coordinates carry data."""
    d = get(domain)
    return d.d_real if d.d_real is not None else d.d_in


def _pad(v: np.ndarray, domain: str) -> np.ndarray:
    """Zero-extend a content-width vector (or row-stack) to the padded width."""
    d = get(domain)
    w = content_width(domain)
    if w == d.d_in:
        return v
    out = np.zeros((*v.shape[:-1], d.d_in), v.dtype)
    out[..., :w] = v
    return out


def mask_vector(domain: str, n_mask: int) -> np.ndarray:
    """1.0 on HIDDEN coordinates, flattened to (d_in,).

    Images hide the LAST `n_mask` rows — the bottom of the picture. Boards hide
    the FIRST `n_mask` files — the queenside, files a..d — and hide all 13
    channels of those squares together, so a hidden square is entirely unknown
    rather than known-to-be-not-a-rook.

    Padding is never hidden: it carries no information to reconstruct, and
    scoring it would dilute every error by a constant.
    """
    d = get(domain)
    m = np.zeros(d.shape, np.float32)
    if d.kind == "image":
        m[d.shape[0] - n_mask:, :] = 1.0
    else:
        m[:, :n_mask, :] = 1.0
    return _pad(m.reshape(content_width(domain)), domain)


def valid_vector(domain: str) -> np.ndarray:
    """1.0 on real coordinates, 0.0 on padding."""
    d = get(domain)
    v = np.zeros(d.d_in, np.float32)
    v[:content_width(domain)] = 1.0
    return v


def visible_vector(domain: str, n_mask: int) -> np.ndarray:
    """1.0 on coordinates that are real AND shown to the query.

    The quantity every look-up baseline is built from. Defining it as
    `1 - mask` instead would count padding as visible: harmless for distances,
    since padding is constant zero, but it inflates the `vis.sum()` that the
    soft look-up divides its distances by, which silently rescales the
    temperature that baseline is swept over. That baseline is the bar this
    project measures models against, so it may not drift with padding.
    """
    return valid_vector(domain) * (1.0 - mask_vector(domain, n_mask))


# ── pools ─────────────────────────────────────────────────────────────────────

def _image_pools(name: str):
    ds = load_supervised_image(name)
    flat = lambda X: np.asarray(X).reshape(len(X), -1).astype(np.float32) / 255.0
    return (flat(ds.X), np.asarray(ds.y), flat(ds.X_test), np.asarray(ds.y_test))


def _chess_phase(fens) -> np.ndarray:
    pc = chess_piece_count(fens)
    hi, mid = CHESS_PHASE_EDGES
    return np.where(pc >= hi, 0, np.where(pc >= mid, 1, 2)).astype(np.int64)


def _chess_pools():
    """Positions as flat one-hot planes, labelled by phase bucket.

    The train/test split is inherited from `load_chess_positions`, which cuts the
    source table at a row id: a chess game is a contiguous run of ids, so no
    game contributes positions to both sides.
    """
    ds = load_chess_positions()
    flat = lambda f: fen_to_planes(f).reshape(len(f), -1)
    return (flat(ds.fen), _chess_phase(ds.fen),
            flat(ds.fen_test), _chess_phase(ds.fen_test))


def _paired_pools(train_name: str, novel_name: str):
    """Training dataset's train split, and BOTH test splits concatenated.

    The novel dataset contributes only its test split, and its labels are shifted
    by PAIR_SHIFT. `build_pools` then does the rest with no special case:
    `held_same` comes out as the training dataset's own test split — new images
    of a seen world — and `held` as the novel dataset.
    """
    Xtr, ytr, Xtr_te, ytr_te = _image_pools(train_name)
    _, _, Xnov_te, ynov_te = _image_pools(novel_name)
    return (Xtr, ytr,
            np.concatenate([Xtr_te, Xnov_te]),
            np.concatenate([ytr_te, ynov_te + PAIR_SHIFT]))


def resample(domain: str, seed: int) -> np.ndarray:
    """A fresh TRAINING pool for a synthetic domain, drawn from the same prior.

    Only the training pool: held-out worlds stay fixed so the control band means
    the same thing at every point in a run.
    """
    if domain not in SYNTH_SHAPES and domain not in SYNTH_PAIRS:
        raise ValueError(f"{domain!r} has no generative prior to resample")
    base = domain if domain in SYNTH_SHAPES else SYNTH_PAIRS[domain][0]
    n_w, per_w, p_sx, k_min = SYNTH_SHAPES[base]
    X, _ = _synth_pool(PAD_W, n_w, per_w, seed=seed, p_simplex=p_sx, k_min=k_min)
    if domain in SYNTH_PAIRS:
        novel = SYNTH_PAIRS[domain][1]
        X = X.copy()
        X[:, (784 if novel in ("mnist", "fashion_mnist") else PAD_W):] = 0.0
    return X


def raw_pools(domain: str):
    """(X_train, y_train, X_test, y_test) — flattened, unsplit, [0,1] valued.

    Padded domains are zero-extended here, so everything downstream sees vectors
    of `d_in` and never has to know a domain was padded.
    """
    if domain in SYNTH_PAIRS:
        return _synth_paired_pools(*SYNTH_PAIRS[domain])
    if domain in SYNTH_SHAPES:
        return _synthetic_pools(domain, get(domain).d_in)
    base = domain[:-4] if domain.endswith("_pad") else domain
    if base == "chess":
        Xtr, ytr, Xte, yte = _chess_pools()
    elif base in PAIRS:
        Xtr, ytr, Xte, yte = _paired_pools(*PAIRS[base])
    else:
        Xtr, ytr, Xte, yte = _image_pools(base)
    return _pad(Xtr, domain), ytr, _pad(Xte, domain), yte


# ── rendering ─────────────────────────────────────────────────────────────────

BOARD_LIGHT, BOARD_DARK = "#f0d9b5", "#b58863"
BOARD_HIDDEN_LIGHT, BOARD_HIDDEN_DARK = "#e6e6e6", "#bdbdbd"

# The SOLID glyphs for both colours, filled light for White and dark for Black
# and outlined against the square. The outline set (U+2654..U+2659) is the
# obvious choice for White and is what a first version used, but an outline
# glyph on a light square is nearly invisible and, worse, reads as the same
# piece as its solid twin — the two sides became indistinguishable in the
# figure. Fill colour, not glyph shape, carries the side.
GLYPHS = " ♟♞♝♜♛♚♟♞♝♜♛♚"      # index matches shared_lib.datasets.CHESS_PIECES
WHITE_FILL, WHITE_EDGE = "#ffffff", "#3a3226"
BLACK_FILL, BLACK_EDGE = "#1b1b1b", "#e8e0d2"


def draw(ax, vec, domain: str, mask=None, hidden_shading: bool = False):
    """Draw one token into `ax`. Images imshow; boards get squares and glyphs.

    For a board, each square shows the argmax over the 13 channels — the piece
    the vector most asserts is there — and its glyph is drawn at an opacity
    equal to that channel's value, so an unconfident prediction reads as a faint
    piece rather than a confident wrong one. `hidden_shading` greys the masked
    files, which is how a query's input is shown.
    """
    d = get(domain)
    # Trim padding before reshaping: the figure draws content, not the vector.
    v = np.asarray(vec).reshape(-1)[:content_width(domain)].reshape(d.shape)
    ax.set_xticks([]); ax.set_yticks([])
    if d.kind == "image":
        if hidden_shading and mask is not None:
            m = np.asarray(mask).reshape(-1)[:content_width(domain)].reshape(d.shape)
            v = v * (1 - m) + 0.5 * m           # grey the hole
        ax.imshow(v, cmap="gray", vmin=0, vmax=1)
        return ax

    m = (None if mask is None else
         np.asarray(mask).reshape(-1)[:content_width(domain)].reshape(d.shape))
    ax.set_xlim(0, 8); ax.set_ylim(8, 0); ax.set_aspect("equal")
    conf = v / np.clip(v.sum(-1, keepdims=True), 1e-6, None)
    # One glyph should fill about three quarters of its square, whatever size the
    # axes ended up. get_window_extent is in pixels and fontsize is in points, so
    # the dpi divide is not optional.
    sq_in = ax.get_window_extent().width / ax.figure.dpi / 8.0
    size = sq_in * 72.0 * 0.78
    for r in range(8):
        for f in range(8):
            hid = hidden_shading and m is not None and m[r, f, 0] > 0.5
            light = (r + f) % 2 == 0
            face = ((BOARD_HIDDEN_LIGHT if light else BOARD_HIDDEN_DARK) if hid
                    else (BOARD_LIGHT if light else BOARD_DARK))
            ax.add_patch(Rectangle((f, r), 1, 1, facecolor=face, edgecolor="none"))
            if hid:
                continue
            p = int(np.argmax(conf[r, f]))
            if p == 0:
                continue
            # Opacity is the channel's own value, so an unconfident prediction
            # reads as a faint piece rather than a confident wrong one.
            a = 0.25 + 0.75 * float(np.clip(conf[r, f, p], 0.0, 1.0))
            fill, edge = (WHITE_FILL, WHITE_EDGE) if p < 7 else (BLACK_FILL, BLACK_EDGE)
            ax.text(f + 0.5, r + 0.54, GLYPHS[p], ha="center", va="center",
                    fontsize=size, alpha=a, color=fill,
                    path_effects=[pe.withStroke(linewidth=size * 0.075,
                                                foreground=edge, alpha=a)])
    for sp in ax.spines.values():
        sp.set_visible(True)
    return ax


def composite(truth, pred, mask):
    """The true visible coordinates with the prediction pasted into the hole.

    The head emits every coordinate but the loss only ever scores hidden ones,
    so a model's visible half is unconstrained and showing it raw makes every
    completion look broken.
    """
    mask = np.asarray(mask)
    return np.asarray(truth) * (1 - mask) + np.asarray(pred) * mask


def piece_accuracy(pred, truth, mask, domain: str = "chess") -> float:
    """Fraction of HIDDEN squares whose argmax piece is right. Chess only.

    Normalised MSE is the project's common currency and is kept, but on a board
    it is hard to read: this says the same thing in the units the domain is
    actually about. Chance is not 1/13 — the empty square is ~70% of a board, so
    the reference point to quote beside it is the all-empty guess.
    """
    d = get(domain)
    assert d.kind == "board", "piece accuracy is only defined for boards"
    w = content_width(domain)
    sq = np.asarray(mask).reshape(-1)[:w].reshape(d.shape)[:, :, 0] > 0.5
    p = np.asarray(pred).reshape(-1, d.d_in)[:, :w].reshape(-1, *d.shape)[:, sq, :]
    t = np.asarray(truth).reshape(-1, d.d_in)[:, :w].reshape(-1, *d.shape)[:, sq, :]
    return float((p.argmax(-1) == t.argmax(-1)).mean())
