"""
Episode construction for token-level in-context classification on Omniglot.

An episode is a flat bag of `(pos, value, ref)` tokens — no sample structure,
per `projects/universal-ar/proposal.md`. It holds `n_way` characters, each with
`k_shot` support drawings and `n_query` query drawings:

    support drawing  →  n_ctx pixel tokens + a label token with the class's slot
    query drawing    →  n_ctx pixel tokens + a MASKED label token (the target)

Two properties make this a real in-context test, and both are load-bearing:

  * **Label slots are drawn fresh per episode.** A slot means nothing across
    episodes, so a memorised class→slot map is worthless.
  * **A query's label appears nowhere in its own tokens.** The only route to it
    is matching the query's pixels against a *different* drawing that shares the
    character. `ref` cannot shortcut this — a query's `ref` is never seen beside
    a label. This is precisely the shortcut that made universal-ar's exp35/36
    vacuous, and Omniglot closes it without needing an anonymisation trick.

All drawings in an episode observe the **same** random position pool. Without
that, support and query would describe disjoint pixels and cross-drawing
matching would be ill-posed rather than merely hard.

## The label field (`spec.label_field`, added after exp1/exp2)

exp1 and exp2 put the class *only* on the label token at `pos_label`, and both
sat at chance. That layout forces a three-hop circuit: gather my own drawing's
pixels by `ref`, match them position-wise against other drawings, recover
*which* `ref` matched, then read that ref's label token. Hop three is the one
that cannot work — softmax attention averages over the ~`n_draw` tokens sharing
a position, so the identity of the drawing that agreed is destroyed in exactly
the step that needs it.

With `label_field=True` the class rides on **every token of a support drawing**,
not just its label token. A query pixel can then attend to support pixels at its
own position, weight them by value agreement, and read a *label* embedding
directly — the vote survives the averaging because the label, not the drawing
identity, is what gets averaged. The query's label token then pools those votes
over its own `ref`. Three hops become two, and the hard one disappears.

This is the token-bag analogue of what makes induction heads learnable in
sequence models, where `y` sits *adjacent* to its `x`. Sharing a `ref` was meant
to be that adjacency; carrying the label on the token is what actually makes it
readable. It is a deviation from proposal.md's "the label is just a token at
`pos_label`", and a deliberate one.
"""

from typing import NamedTuple, Sequence

import numpy as np


class Spec(NamedTuple):
    """Episode shape and vocabulary layout."""

    n_way: int = 5
    k_shot: int = 1
    n_query: int = 1
    n_ctx: int = 196        # observed pixels per drawing, from a shared pool
    img_size: int = 28
    n_bins: int = 8         # pixel-intensity bins; Omniglot is near-binary
    v_refs: int = 64        # coreference tag pool
    label_field: bool = False   # carry the class on EVERY token, not just the
                                # label token — see `lab` in Batch
    ink_pool: bool = False      # draw observed positions from where the SUPPORT
                                # drawings have ink, instead of uniformly
    identity_query: bool = False  # POSITIVE CONTROL: each query drawing is the
                                # very same image as its support drawing, so
                                # matching is exact. A model that cannot solve
                                # this cannot solve anything, and the failure is
                                # a bug rather than a difficulty.

    @property
    def n_lab(self) -> int:
        """Label-field vocabulary: one id per slot, plus MASK for query tokens."""
        return self.n_way + 1

    @property
    def lab_mask(self) -> int:
        return self.n_way

    @property
    def n_pos(self) -> int:
        """Position vocabulary: every pixel, plus one slot for the label."""
        return self.img_size * self.img_size + 1

    @property
    def pos_label(self) -> int:
        return self.img_size * self.img_size

    @property
    def n_content(self) -> int:
        """Unified value vocabulary: pixel bins ∪ label slots."""
        return self.n_bins + self.n_way

    @property
    def mask_id(self) -> int:
        return self.n_content

    @property
    def n_val(self) -> int:
        return self.n_content + 1

    @property
    def n_draw(self) -> int:
        return self.n_way * (self.k_shot + self.n_query)

    @property
    def n_tokens(self) -> int:
        return self.n_draw * (self.n_ctx + 1)


class Batch(NamedTuple):
    """A batch of episodes, host-side. All arrays are `(batch, n_tokens)`."""

    pos: np.ndarray      # int32
    val: np.ndarray      # int32 — observed value, or MASK for a query label
    ref: np.ndarray      # int32 — coreference tag, shared across a drawing
    tgt: np.ndarray      # int32 — label slot for scored tokens, -1 elsewhere
    is_query: np.ndarray  # float32 — 1.0 on scored (masked label) tokens
    lab: np.ndarray      # int32 — label field: the slot on every token of a
                         # support drawing, MASK on every token of a query one.
                         # All MASK unless spec.label_field, and the model only
                         # embeds it in that case (see lib/models.py).


def class_index(y: np.ndarray, n_classes: int) -> list[np.ndarray]:
    """Row indices for each character id — precompute once, reuse every step."""
    return [np.where(y == c)[0] for c in range(n_classes)]


def bin_pixels(x: np.ndarray, n_bins: int) -> np.ndarray:
    """Map 0-255 intensities onto `n_bins` value ids. Bin 0 is background."""
    return (x.astype(np.int32) * n_bins // 256).clip(0, n_bins - 1)


def _position_pool(
    rng: np.random.Generator,
    X: np.ndarray,
    support_rows: Sequence[int],
    spec: Spec,
) -> np.ndarray:
    """The positions every drawing in this episode will be observed at.

    Uniform by default. With `ink_pool`, positions where a *support* drawing has
    ink are preferred, topped up at random if there are too few. At 18.7% ink a
    uniform pool spends ~160 of its 196 tokens on background, and a background
    token matches nearly every support drawing, so its contribution to a
    content-matching vote is close to uniform noise. Biasing the pool raises the
    fraction of tokens that can discriminate at all.

    Drawn from SUPPORT drawings only. Using the queries' ink would let the pool
    itself carry information about the drawings being classified.
    """
    n_pix = spec.img_size ** 2
    if not spec.ink_pool:
        return rng.permutation(n_pix)[: spec.n_ctx]

    inked = np.where((X[list(support_rows)] > 0).any(axis=0))[0]
    if len(inked) >= spec.n_ctx:
        return rng.permutation(inked)[: spec.n_ctx]
    rest = np.setdiff1d(np.arange(n_pix), inked, assume_unique=False)
    top_up = rng.permutation(rest)[: spec.n_ctx - len(inked)]
    return np.concatenate([inked, top_up])


def build_batch(
    rng: np.random.Generator,
    X: np.ndarray,
    cls_idx: Sequence[np.ndarray],
    spec: Spec,
    batch: int,
) -> Batch:
    """Sample `batch` independent episodes.

    Args:
        X:       (n_images, img_size**2) uint8, ink high (see `load_omniglot`).
        cls_idx: per-character row indices into X, from `class_index`.
        spec:    episode shape.
        batch:   number of episodes.
    """
    n_chars = len(cls_idx)
    if n_chars < spec.n_way:
        raise ValueError(f"need at least {spec.n_way} characters, have {n_chars}")
    per_class = spec.k_shot + spec.n_query

    T = spec.n_tokens
    pos = np.zeros((batch, T), np.int32)
    val = np.zeros((batch, T), np.int32)
    ref = np.zeros((batch, T), np.int32)
    tgt = -np.ones((batch, T), np.int32)
    is_query = np.zeros((batch, T), np.float32)
    lab = np.full((batch, T), spec.lab_mask, np.int32)

    for b in range(batch):
        classes = rng.choice(n_chars, spec.n_way, replace=False)
        slots = rng.permutation(spec.n_way)           # fresh label semantics
        # The uniform pool is drawn here, at the position it has always been
        # drawn, so runs without `ink_pool` consume the RNG stream exactly as
        # they did before that option existed and stay reproducible.
        pool = None if spec.ink_pool else rng.permutation(spec.img_size ** 2)[: spec.n_ctx]
        refs = rng.permutation(spec.v_refs)[: spec.n_draw]

        # (image row, label slot, is_support) for every drawing in the episode
        draws: list[tuple[int, int, bool]] = []
        for i, c in enumerate(classes):
            rows = cls_idx[int(c)]
            if len(rows) < per_class:
                raise ValueError(
                    f"character {int(c)} has {len(rows)} drawings, need {per_class}"
                )
            if spec.identity_query:
                # Support drawings only; each query repeats its support image.
                sup = rng.choice(rows, spec.k_shot, replace=False)
                picked = np.concatenate([sup, np.resize(sup, spec.n_query)])
            else:
                picked = rng.choice(rows, per_class, replace=False)
            slot = int(slots[i])
            for j, r in enumerate(picked):
                draws.append((int(r), slot, j < spec.k_shot))

        if pool is None:   # ink-biased: needs the support rows chosen above
            pool = _position_pool(
                rng, X, [row for row, _, is_sup in draws if is_sup], spec
            )

        t = 0
        for k, (row, slot, is_support) in enumerate(draws):
            r = int(refs[k])
            bins = bin_pixels(X[row][pool], spec.n_bins)
            nxt = t + spec.n_ctx
            pos[b, t:nxt] = pool
            val[b, t:nxt] = bins
            ref[b, t:nxt] = r
            if spec.label_field and is_support:
                lab[b, t:nxt] = slot     # every support PIXEL carries its class
            t = nxt

            pos[b, t], ref[b, t] = spec.pos_label, r
            if is_support:
                val[b, t] = spec.n_bins + slot          # label GIVEN
                if spec.label_field:
                    lab[b, t] = slot
            else:
                val[b, t] = spec.mask_id                # label MASKED — the task
                tgt[b, t] = spec.n_bins + slot
                is_query[b, t] = 1.0
            t += 1

    return Batch(pos, val, ref, tgt, is_query, lab)


def observed_pixels(
    rng: np.random.Generator,
    X: np.ndarray,
    cls_idx: Sequence[np.ndarray],
    spec: Spec,
    batch: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The same episodes a model sees, as raw feature vectors — for baselines.

    Returns `(support_x, support_slot, query_x, query_slot)` with support shaped
    `(batch, n_way*k_shot, n_ctx)` and query `(batch, n_way*n_query, n_ctx)`.
    Crucially the pixels are the episode's shared `n_ctx` pool, so a
    nearest-neighbour baseline sees *exactly* the information the model does —
    a baseline on the full image would be measuring a different task.
    """
    n_sup, n_qry = spec.n_way * spec.k_shot, spec.n_way * spec.n_query
    sx = np.zeros((batch, n_sup, spec.n_ctx), np.float32)
    ss = np.zeros((batch, n_sup), np.int32)
    qx = np.zeros((batch, n_qry, spec.n_ctx), np.float32)
    qs = np.zeros((batch, n_qry), np.int32)

    for b in range(batch):
        classes = rng.choice(len(cls_idx), spec.n_way, replace=False)
        slots = rng.permutation(spec.n_way)
        pool = None if spec.ink_pool else rng.permutation(spec.img_size ** 2)[: spec.n_ctx]
        def _pick(c):
            rows = cls_idx[int(c)]
            if spec.identity_query:
                sup = rng.choice(rows, spec.k_shot, replace=False)
                return np.concatenate([sup, np.resize(sup, spec.n_query)])
            return rng.choice(rows, spec.k_shot + spec.n_query, replace=False)

        chosen = [(_pick(c), int(slots[i])) for i, c in enumerate(classes)]
        if pool is None:
            # Same pool distribution the model gets, or the baseline stops being
            # a floor for it — it would be solving a different task.
            pool = _position_pool(
                rng, X,
                [int(r) for picked, _ in chosen for r in picked[: spec.k_shot]],
                spec,
            )
        si = qi = 0
        for picked, slot in chosen:
            for j, row in enumerate(picked):
                feat = X[int(row)][pool].astype(np.float32) / 255.0
                if j < spec.k_shot:
                    sx[b, si], ss[b, si] = feat, slot
                    si += 1
                else:
                    qx[b, qi], qs[b, qi] = feat, slot
                    qi += 1
    return sx, ss, qx, qs
